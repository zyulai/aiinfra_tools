#!/usr/bin/env python3
"""
Unified interactive runner — auto-detect valid parallel strategies from config,
present a beautiful menu, let user pick by number.

Usage:
  python run.py --config config.json
  python run.py --config /path/to/model_dir
  python run.py --model /mnt/hdd/models/Qwen3.6-27B-DFlash-MS
"""
import argparse
import os
import sys
import json
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import IntPrompt, Confirm, Prompt
from rich.text import Text
from rich.columns import Columns
from rich import box
from rich.align import Align
from rich.rule import Rule

from parallel_tool.config_parser import parse_config, ModelConfig
from parallel_tool.model_kernels import build_kernels
from parallel_tool.parallel_engine import ParallelConfig, analyze_all_kernels
from parallel_tool.table_renderer import render_table, RICH_AVAILABLE
from parallel_tool.framework_refs import FRAMEWORK_REFS
from parallel_tool.main import compare_strategies

console = Console(width=120)

# ── Colors ──
C_TITLE = "bold bright_cyan"
C_OK = "bold green"
C_NO = "bold red"
C_DIM = "dim"
C_KEY = "bold yellow"
C_VAL = "white"
C_HEAD = "bold bright_magenta"


def fmt_bytes(n: int) -> str:
    val = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(val) < 1024:
            return f"{val:.1f} {unit}"
        val /= 1024
    return f"{val:.1f} PB"


def detect_strategies(cfg: ModelConfig) -> dict:
    """Auto-detect valid parallel strategies from model config."""
    result = {"tp": [], "pp": [], "cp": [], "ep": [], "sp": False, "moe": cfg.is_moe}

    # ── TP: check divisibility of all critical dims ──
    dims = [d for d in [cfg.num_attention_heads, cfg.kv_heads,
                        cfg.intermediate_size, cfg.vocab_size] if d > 0]
    for tp in [1, 2, 4, 8, 16, 32, 64]:
        if all(d % tp == 0 for d in dims):
            result["tp"].append(tp)

    # ── PP: split layers ──
    for pp in [1, 2, 3, 4, 8, 16]:
        if pp <= cfg.num_hidden_layers:
            result["pp"].append(pp)

    # ── CP: typically powers of 2 ──
    result["cp"] = [1, 2, 4, 8]

    # ── EP: only for MoE ──
    if cfg.is_moe and cfg.num_experts > 0:
        for ep in [1, 2, 4, 8, 16, 32, 60]:
            if cfg.num_experts % ep == 0:
                result["ep"].append(ep)

    # ── SP: requires TP > 1 ──
    result["sp"] = len(result["tp"]) > 1  # at least TP=1 and TP=2

    return result


def estimate_weight_size(cfg: ModelConfig, tp: int = 1, pp: int = 1, ep: int = 1, num_layers: int = None) -> int:
    """Estimate total weight bytes for a model (bf16) per GPU."""
    h = cfg.hidden_size
    hd = cfg.effective_head_dim
    n_heads = cfg.num_attention_heads
    n_kv = cfg.kv_heads
    inter = cfg.intermediate_size
    vocab = cfg.vocab_size
    layers = (num_layers or cfg.num_hidden_layers) // max(pp, 1)
    bytes_per = 2  # bf16

    total = 0
    # Embedding (only on stage 0 for PP)
    embed_bytes = (vocab // tp) * h * bytes_per
    if pp > 1:
        embed_bytes = embed_bytes  # still on first stage, but we show average
    total += embed_bytes

    # Per layer
    q = (n_heads // tp) * hd * h
    k = (n_kv // tp) * hd * h
    v = (n_kv // tp) * hd * h
    o = h * (n_heads // tp) * hd
    gate = (inter // tp) * h
    up = (inter // tp) * h
    down = h * (inter // tp)
    norms = h * 2  # input + post_attn
    layer_total = q + k + v + o + gate + up + down + norms

    if cfg.is_moe and ep > 1:
        # Experts are split by EP, not by the layer weight
        moe_inter = cfg.moe_intermediate_size or inter
        experts_per_rank = cfg.num_experts // ep
        per_expert = (moe_inter // tp * h + moe_inter // tp * h + h * moe_inter // tp) * bytes_per
        layer_total = (q + k + v + o + norms) * bytes_per  # non-expert per layer
        layer_total += per_expert * experts_per_rank  # expert weights per rank
    else:
        layer_total = layer_total * bytes_per

    total += layer_total * layers
    # Final norm + LM head (only on last stage for PP)
    total += h * bytes_per
    if not cfg.tie_word_embeddings:
        total += (vocab // tp) * h * bytes_per

    return total


def show_model_summary(cfg: ModelConfig):
    """Show model info in a beautiful panel."""
    size_full = estimate_weight_size(cfg, tp=1, pp=1)

    info = Table(show_header=False, box=None, padding=(0, 1))
    info.add_column("key", style=C_KEY, width=20)
    info.add_column("val", style=C_VAL)

    info.add_row("Model Type", cfg.model_type)
    info.add_row("Architecture", ", ".join(cfg.architectures) or "—")
    info.add_row("Hidden Size", str(cfg.hidden_size))
    info.add_row("Layers", str(cfg.num_hidden_layers))
    info.add_row("Attention Heads", str(cfg.num_attention_heads))
    info.add_row("KV Heads", f"{cfg.kv_heads} (GQA ratio = {cfg.num_attention_heads // cfg.kv_heads}:1)")
    info.add_row("Head Dim", str(cfg.effective_head_dim))
    info.add_row("Intermediate", str(cfg.intermediate_size))
    info.add_row("Vocab Size", str(cfg.vocab_size))
    if cfg.is_moe:
        info.add_row("MoE", f"[{C_OK}]Yes[/]  {cfg.num_experts} experts, {cfg.num_experts_per_tok}/tok")
        info.add_row("MoE Inter", str(cfg.moe_intermediate_size or cfg.intermediate_size))
    else:
        info.add_row("MoE", f"[{C_NO}]No[/]  (dense model)")
    info.add_row("Tied Embeddings", str(cfg.tie_word_embeddings))
    info.add_row("Weight Size (bf16)", fmt_bytes(size_full))

    console.print(Panel(info, title="[bold bright_cyan]Model Summary[/]",
                        border_style="bright_blue", width=60))


def show_strategy_menu(cfg: ModelConfig, strategies: dict):
    """Show valid parallel strategies in a nice table."""
    table = Table(title="[bold]Auto-Detected Parallel Strategies[/]",
                  box=box.ROUNDED, show_lines=True, width=110)
    table.add_column("Strategy", style=C_HEAD, width=16)
    table.add_column("Valid Sizes", width=50)
    table.add_column("Constraint", style=C_DIM, width=40)

    # TP
    tp_strs = []
    for tp in strategies["tp"]:
        if tp == 1:
            tp_strs.append(f"[dim]TP=1[/]")
        else:
            sz = estimate_weight_size(cfg, tp=tp, pp=1)
            tp_strs.append(f"[green]TP={tp}[/]  ({fmt_bytes(sz)}/GPU)")
    table.add_row("Tensor Parallel", "\n".join(tp_strs),
                  f"must divide heads({cfg.num_attention_heads}), KV({cfg.kv_heads}), "
                  f"inter({cfg.intermediate_size}), vocab({cfg.vocab_size})")

    # PP
    pp_strs = []
    for pp in strategies["pp"]:
        if pp == 1:
            pp_strs.append("[dim]PP=1  (no pipeline)[/]")
        else:
            base = cfg.num_hidden_layers // pp
            extra = cfg.num_hidden_layers % pp
            if extra:
                dist = ", ".join(str(base+1 if i < extra else base) for i in range(pp))
                pp_strs.append(f"[green]PP={pp}[/]  layers=[{dist}]")
            else:
                pp_strs.append(f"[green]PP={pp}[/]  {base} layers/stage")
    table.add_row("Pipeline Parallel", "\n".join(pp_strs),
                  f"must be ≤ num_layers({cfg.num_hidden_layers})")

    # CP
    cp_strs = []
    for cp in strategies["cp"]:
        if cp == 1:
            cp_strs.append("[dim]CP=1  (no context parallel)[/]")
        else:
            cp_strs.append(f"[green]CP={cp}[/]")
    table.add_row("Context Parallel", "\n".join(cp_strs), "splits seq_len (needs TP>1)")

    # EP
    if cfg.is_moe:
        ep_strs = []
        for ep in strategies["ep"]:
            if ep == 1:
                ep_strs.append("[dim]EP=1  (no expert parallel)[/]")
            else:
                ep_strs.append(f"[green]EP={ep}[/]  {cfg.num_experts // ep} experts/rank")
        table.add_row("Expert Parallel", "\n".join(ep_strs),
                      f"must divide num_experts({cfg.num_experts})")
    else:
        table.add_row("Expert Parallel", "[dim]N/A — dense model[/]", "requires MoE architecture")

    # SP
    if strategies["sp"]:
        table.add_row("Sequence Parallel", "[green]SP = on/off[/]", "requires TP > 1")
    else:
        table.add_row("Sequence Parallel", "[dim]N/A — TP must be > 1[/]", "requires TP > 1")

    console.print(table)


def generate_presets(cfg: ModelConfig, strategies: dict) -> list[dict]:
    """Auto-generate valid preset combinations from detected strategies."""
    presets = []
    tps = [t for t in strategies["tp"] if t > 1]  # exclude TP=1
    pps = [p for p in strategies["pp"] if p > 1]
    eps = [e for e in strategies["ep"] if e > 1]

    # Single TP
    for tp in tps[:4]:  # limit to first 4
        pc = ParallelConfig(tp=tp, sp=True)
        presets.append({"config": pc, "desc": f"TP={tp} + SP", "gpus": tp})

    # TP + PP combinations (pick reasonable ones)
    for tp in tps[:2]:
        for pp in pps[:2]:
            pc = ParallelConfig(tp=tp, pp=pp, sp=True)
            presets.append({"config": pc, "desc": f"TP={tp} + PP={pp} + SP", "gpus": tp * pp})

    # TP + CP
    for tp in tps[:2]:
        pc = ParallelConfig(tp=tp, cp=2, sp=True)
        presets.append({"config": pc, "desc": f"TP={tp} + CP=2 + SP", "gpus": tp * 2})

    # EP combos for MoE
    if eps:
        for ep in eps[:3]:  # limit to first 3 reasonable values
            tp = 2 if 2 in strategies["tp"] else 1
            pc = ParallelConfig(tp=tp, ep=ep)
            presets.append({"config": pc, "desc": f"TP={tp} + EP={ep}", "gpus": tp * ep})

    # Full 3D
    if tps and pps:
        tp = tps[0]
        pp = pps[0]
        pc = ParallelConfig(tp=tp, pp=pp, dp=2, sp=True)
        presets.append({"config": pc, "desc": f"TP={tp} + PP={pp} + DP=2 + SP", "gpus": tp * pp * 2})

    # Dedupe by description
    seen = set()
    unique = []
    for p in presets:
        if p["desc"] not in seen:
            seen.add(p["desc"])
            unique.append(p)

    return unique


def show_preset_menu(cfg: ModelConfig, presets: list[dict]):
    """Show preset options as a numbered table."""
    table = Table(title="[bold]Preset Parallel Configurations[/]",
                  box=box.ROUNDED, show_lines=True, width=100)
    table.add_column("#", style=C_HEAD, justify="center", width=4)
    table.add_column("Configuration", style="bold", width=35)
    table.add_column("GPUs", justify="center", width=6)
    table.add_column("Weight/GPU", justify="right", width=14)

    for i, p in enumerate(presets, 1):
        pc = p["config"]
        sz = estimate_weight_size(cfg, tp=pc.tp, pp=pc.pp, ep=pc.ep)
        if pc.ep > 1 and cfg.is_moe:
            sz_str = f"{fmt_bytes(sz)} + {cfg.num_experts // pc.ep} exp"
        else:
            sz_str = fmt_bytes(sz)
        table.add_row(str(i), p["desc"], str(p["gpus"]), sz_str)

    console.print(table)


def show_action_menu():
    """Show action selection."""
    console.print()
    actions = Table(box=box.ROUNDED, width=70, show_lines=True)
    actions.add_column("#", style=C_HEAD, justify="center", width=4)
    actions.add_column("Action", style="bold", width=50)
    actions.add_row("1", "🔍  Analyze kernel sharding (parallel_tool)")
    actions.add_row("2", "⚡  Generate random weights (weight_gen)")
    actions.add_row("3", "📊  Compare multiple strategies side-by-side")
    actions.add_row("4", "🔧  Custom config (pick each dimension)")
    actions.add_row("0", "🚪  Exit")
    console.print(Panel(actions, title="[bold]Choose Action[/]", border_style="bright_yellow"))


def custom_config_flow(cfg: ModelConfig, strategies: dict) -> ParallelConfig:
    """Let user pick each dimension interactively."""
    pc = ParallelConfig()

    # TP
    console.print(f"\n[{C_KEY}]Tensor Parallel[/] — valid: {strategies['tp']}")
    tp = IntPrompt.ask(f"  TP size", default=1, choices=[str(t) for t in strategies["tp"]])
    pc.tp = tp

    # PP
    console.print(f"\n[{C_KEY}]Pipeline Parallel[/] — valid: {strategies['pp']}")
    pp = IntPrompt.ask(f"  PP size", default=1, choices=[str(p) for p in strategies["pp"]])
    pc.pp = pp

    # CP (only if TP > 1)
    if tp > 1:
        console.print(f"\n[{C_KEY}]Context Parallel[/] — valid: {strategies['cp']}")
        cp = IntPrompt.ask(f"  CP size", default=1, choices=[str(c) for c in strategies["cp"]])
        pc.cp = cp

    # EP (only for MoE)
    if cfg.is_moe and strategies["ep"]:
        console.print(f"\n[{C_KEY}]Expert Parallel[/] — valid: {strategies['ep']}")
        ep = IntPrompt.ask(f"  EP size", default=1, choices=[str(e) for e in strategies["ep"]])
        pc.ep = ep

    # SP (only if TP > 1)
    if tp > 1:
        pc.sp = Confirm.ask(f"\n  Enable Sequence Parallel (SP)?", default=True)

    # DP
    dp = IntPrompt.ask(f"\n  Data Parallel size", default=1)
    pc.dp = max(dp, 1)

    return pc


def run_analysis(cfg: ModelConfig, pc: ParallelConfig):
    """Run parallel analysis and show results."""
    kernels = build_kernels(cfg)
    errors = pc.validate(cfg)
    if errors:
        console.print(f"\n[bold red]⚠ Validation:[/]")
        for e in errors:
            console.print(f"  [red]- {e}[/]")

    results = analyze_all_kernels(cfg, pc, kernels)
    table = render_table(cfg, pc, results, use_rich=True)
    console.print(table)

    # Framework refs
    console.print()
    console.print(Rule("[bold]Framework Implementation References[/]", style="bright_blue"))
    show_framework_refs(pc)

    # Offer JSON export
    if Confirm.ask("\n  Export to JSON?", default=False):
        path = Prompt.ask("  Output path", default="parallel_result.json")
        data = {
            "model": {"model_type": cfg.model_type, "hidden_size": cfg.hidden_size,
                      "num_hidden_layers": cfg.num_hidden_layers},
            "parallel_config": {"tp": pc.tp, "pp": pc.pp, "cp": pc.cp, "ep": pc.ep,
                                "dp": pc.dp, "sp": pc.sp, "world_size": pc.world_size},
            "kernels": [{"name": r.kernel.name, "category": r.kernel.category,
                         "original_weight_shape": list(r.original_weight_shape),
                         "sharded_weight_shape": list(r.sharded_weight_shape),
                         "comm_fwd": r.comm_fwd, "comm_bwd": r.comm_bwd,
                         "memory_per_gpu_bytes": r.memory_per_gpu_bytes}
                        for r in results],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        console.print(f"  [{C_OK}]✓ Saved to {path}[/]")


def show_framework_refs(pc: ParallelConfig):
    """Show framework references for active strategies."""
    refs_to_show = []
    if pc.tp > 1:
        refs_to_show.extend(["TP_embedding", "TP_qkv", "TP_o_proj", "TP_mlp_gate_up",
                             "TP_mlp_down", "TP_lm_head"])
    if pc.pp > 1:
        refs_to_show.append("PP_layer_split")
    if pc.cp > 1:
        refs_to_show.append("CP_ring_attention")
    if pc.ep > 1:
        refs_to_show.append("EP_dispatch")
    if pc.sp:
        refs_to_show.append("SP_regions")
    if pc.dp > 1:
        refs_to_show.append("DP_grad_sync")

    for key in refs_to_show:
        ref = FRAMEWORK_REFS.get(key, {})
        console.print(f"\n  [bold cyan]┌─ {key}[/]")
        console.print(f"  [{C_DIM}]├─ Pattern:[/] {ref.get('pattern', 'N/A')}")
        console.print(f"  [{C_DIM}]├─ SGLang:  [/] {ref.get('sglang', 'N/A')}")
        console.print(f"  [{C_DIM}]├─ vLLM:    [/] {ref.get('vllm', 'N/A')}")
        console.print(f"  [{C_DIM}]└─ Megatron:[/] {ref.get('megatron', 'N/A')}")


def run_weight_gen(cfg: ModelConfig, pc: ParallelConfig):
    """Run weight generation."""
    from weight_gen.weight_shaper import build_weight_map, compute_total_bytes
    from weight_gen.weight_writer import generate_random_weights, save_single_file, write_index_json
    from weight_gen.model_cloner import clone_aux_files, write_config

    entries = build_weight_map(cfg, pc)
    total = compute_total_bytes(entries)

    # Show plan
    plan = Table(box=box.ROUNDED, width=60, show_lines=True)
    plan.add_column("Item", style=C_KEY, width=20)
    plan.add_column("Value", style=C_VAL, width=35)
    plan.add_row("Total weights", f"{len(entries)} tensors")
    plan.add_row("Total size", fmt_bytes(total))
    plan.add_row("Dtype", "bfloat16")
    console.print(Panel(plan, title="[bold]Weight Generation Plan[/]", border_style="bright_green"))

    # Ask for options
    output_dir = Prompt.ask("\n  Output directory", default="./rand_model")
    num_layers = None
    if cfg.num_hidden_layers > 8:
        if Confirm.ask(f"  Truncate layers? (model has {cfg.num_hidden_layers} layers)", default=True):
            num_layers = IntPrompt.ask("  Num layers", default=min(4, cfg.num_hidden_layers))
    
    dtype = Prompt.ask("  Dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    init = Prompt.ask("  Init strategy", choices=["normal", "uniform", "zeros", "small"], default="normal")
    gen_benchmark = Confirm.ask("  Generate benchmark script?", default=True)
    bench_backend = "vllm"
    if gen_benchmark:
        bench_backend = Prompt.ask("  Benchmark backend", choices=["vllm", "sglang", "hf"], default="vllm")

    source_dir = Prompt.ask("  Tokenizer source dir (empty = skip)", default="")

    # Generate
    console.print(f"\n  [{C_OK}]Generating...[/]")
    
    overrides = {}
    actual_cfg = cfg
    if num_layers and num_layers != cfg.num_hidden_layers:
        overrides["num_hidden_layers"] = num_layers
        cfg.num_hidden_layers = num_layers
        actual_cfg = cfg

    from weight_gen.main import generate_model
    generate_model(
        cfg=actual_cfg, pc=pc, output_dir=output_dir,
        source_dir=source_dir if source_dir else None,
        num_layers=num_layers,
        dtype=dtype, init_strategy=init,
        benchmark=bench_backend if gen_benchmark else None,
        verbose=True,
    )


def run_compare(cfg: ModelConfig, strategies: dict):
    """Compare multiple strategies."""
    presets = generate_presets(cfg, strategies)
    console.print("\n[bold]Select strategies to compare (comma-separated numbers):[/]")
    show_preset_menu(cfg, presets)
    sel = Prompt.ask("\n  Select (e.g. 1,3,5)")
    indices = []
    for s in sel.split(","):
        s = s.strip()
        if s.isdigit():
            idx = int(s) - 1
            if 0 <= idx < len(presets):
                indices.append(idx)
    
    if not indices:
        console.print("  [red]No valid selection[/]")
        return
    
    strategies_list = [presets[i]["config"] for i in indices]
    console.print(compare_strategies(cfg, strategies_list))


def main():
    parser = argparse.ArgumentParser(description="AI Infra Parallel Tool — Interactive")
    parser.add_argument("--config", type=str, help="Path to HF config.json or model dir")
    parser.add_argument("--model", type=str, help="Model dir path")
    args = parser.parse_args()

    # Banner
    banner = Text()
    banner.append("╔" + "═" * 58 + "╗\n", style="bright_blue")
    banner.append("║", style="bright_blue")
    banner.append(Align.center(Text("🚀 AI Infra Parallel Tool", style=C_TITLE)).render() if False else "")
    banner.append(" 🚀 AI Infra Parallel Tool                                    ", style=C_TITLE)
    banner.append("║\n", style="bright_blue")
    banner.append("║", style="bright_blue")
    banner.append(" Analyze sharding • Generate random weights • Benchmark      ", style=C_DIM)
    banner.append("║\n", style="bright_blue")
    banner.append("╚" + "═" * 58 + "╝", style="bright_blue")
    console.print(banner)

    # Get config path
    config_path = args.config or args.model
    if not config_path:
        config_path = Prompt.ask("[bold]Path to config.json or model directory[/]")
    
    if os.path.isdir(config_path):
        config_path = os.path.join(config_path, "config.json")
    if not os.path.exists(config_path):
        console.print(f"[red]Config not found: {config_path}[/]")
        return

    # Parse
    try:
        cfg = parse_config(config_path)
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        return

    # Show model summary
    show_model_summary(cfg)

    # Auto-detect strategies
    strategies = detect_strategies(cfg)
    show_strategy_menu(cfg, strategies)

    # Main loop
    while True:
        presets = generate_presets(cfg, strategies)
        console.print()
        console.print(Rule("[bold]Preset Combinations[/]", style="bright_cyan"))
        show_preset_menu(cfg, presets)

        show_action_menu()
        choice = IntPrompt.ask("\n  Choice", default=1)

        if choice == 0:
            console.print("\n[dim]Bye! 👋[/]")
            break
        elif choice == 1:
            # Analyze
            console.print("\n[dim]Pick a preset (or 0 for custom):[/]")
            sel = IntPrompt.ask("  Preset #", default=1)
            if sel == 0:
                pc = custom_config_flow(cfg, strategies)
            elif 1 <= sel <= len(presets):
                pc = presets[sel - 1]["config"]
            else:
                console.print("[red]Invalid[/]")
                continue
            run_analysis(cfg, pc)
        elif choice == 2:
            # Weight gen
            console.print("\n[dim]Pick a preset (or 0 for custom):[/]")
            sel = IntPrompt.ask("  Preset #", default=1)
            if sel == 0:
                pc = custom_config_flow(cfg, strategies)
            elif 1 <= sel <= len(presets):
                pc = presets[sel - 1]["config"]
            else:
                console.print("[red]Invalid[/]")
                continue
            run_weight_gen(cfg, pc)
        elif choice == 3:
            run_compare(cfg, strategies)
        elif choice == 4:
            pc = custom_config_flow(cfg, strategies)
            console.print(f"\n  [bold]Config: {pc} | World size: {pc.world_size}[/]")
            sub = IntPrompt.ask("  1=Analyze, 2=Generate weights", default=1)
            if sub == 1:
                run_analysis(cfg, pc)
            else:
                run_weight_gen(cfg, pc)
        
        if not Confirm.ask("\n  Run another action?", default=True):
            break
        console.clear()


if __name__ == "__main__":
    main()
