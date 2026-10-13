#!/usr/bin/env python3
"""Interactive model sharding analysis with an editable current configuration."""
import argparse
import json
import os
from dataclasses import replace

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from parallel_tool.config_parser import ModelConfig, parse_config
from parallel_tool.parallel_engine import ParallelConfig
from parallel_tool.reporting import (
    CATEGORY_NAMES, DTYPE_BYTES, analysis_as_dict, analyze_model, format_bytes, summarize_memory,
)
from parallel_tool.table_renderer import print_analysis, print_comparison
from parallel_tool.framework_refs import active_framework_refs

console = Console()
fmt_bytes = format_bytes


def detect_strategies(cfg: ModelConfig) -> dict:
    tps = [size for size in range(1, min(cfg.num_attention_heads, 64) + 1)
           if not ParallelConfig(tp=size).validate(cfg, include_warnings=False)]
    pps = sorted({size for size in (1, 2, 3, 4, 8, 16, cfg.num_hidden_layers)
                  if not ParallelConfig(pp=size).validate(cfg, include_warnings=False)})
    eps = ([size for size in range(1, min(cfg.num_experts, 64) + 1)
            if not ParallelConfig(ep=size).validate(cfg, include_warnings=False)]
           if cfg.is_moe else [])
    if cfg.is_moe and cfg.num_experts not in eps:
        eps.append(cfg.num_experts)
    return {"tp": tps, "pp": pps, "cp": [1, 2, 4, 8], "ep": eps,
            "sp": any(size > 1 for size in tps), "moe": cfg.is_moe}


def estimate_weight_size(cfg, tp=1, pp=1, ep=1, num_layers=None, dtype="bfloat16", *, cache=None) -> int:
    # The optional cache belongs to one model session and stores only byte counts.
    key = (tp, pp, ep, num_layers, dtype)
    if cache is not None and key in cache:
        return cache[key]
    actual_cfg = replace(cfg, num_hidden_layers=num_layers) if num_layers is not None else cfg
    pc = ParallelConfig(tp=tp, pp=pp, ep=ep)
    size = summarize_memory(actual_cfg, pc, analyze_model(actual_cfg, pc, dtype)).peak_weight_bytes
    if cache is not None:
        cache[key] = size
    return size


def show_model_summary(cfg: ModelConfig, dtype="bfloat16", *, cache=None):
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column(overflow="fold")
    rows = [
        ("Model", cfg.model_type), ("Architecture", ", ".join(cfg.architectures) or "—"),
        ("Layers / hidden", f"{cfg.num_hidden_layers} / {cfg.hidden_size}"),
        ("Q / KV heads", f"{cfg.num_attention_heads} / {cfg.kv_heads}"),
        ("Head dimension", str(cfg.effective_head_dim)),
        ("FFN / vocabulary", f"{cfg.intermediate_size} / {cfg.vocab_size}"),
        ("Experts", f"{cfg.num_experts} ({cfg.num_experts_per_tok} active/token)" if cfg.is_moe else "Dense model"),
        (f"Weights ({dtype})", format_bytes(estimate_weight_size(cfg, dtype=dtype, cache=cache))),
    ]
    for key, value in rows:
        table.add_row(key, Text(value))
    console.print(Panel(table, title="Model", border_style="blue"))


def show_strategy_menu(cfg, strategies):
    table = Table(title="Available parallel dimensions", box=box.ROUNDED, expand=True)
    table.add_column("Dimension", style="bold cyan")
    table.add_column("Sizes", overflow="fold")
    table.add_column("Effect", overflow="fold")
    for key, description in (("tp", "Split heads, FFN and vocabulary"),
                             ("pp", "Assign layers to pipeline stages"),
                             ("cp", "Split sequence; backend-dependent"),
                             ("ep", "Distribute MoE experts")):
        table.add_row(key.upper(), ", ".join(map(str, strategies[key])) or "Unavailable", description)
    table.add_row("DP", "Any positive integer", "Replicate the model")
    table.add_row("SP", "On / off" if strategies["sp"] else "Unavailable", "Sequence sharding with TP > 1")
    console.print(table)


def generate_presets(cfg, strategies, gpu_budget=None) -> list[ParallelConfig]:
    presets = []
    seen = set()

    def add(pc):
        key = (pc.tp, pc.pp, pc.cp, pc.ep, pc.dp, pc.sp)
        if key in seen or pc.validate(cfg, include_warnings=False):
            return
        if gpu_budget is not None and pc.world_size > gpu_budget:
            return
        seen.add(key)
        presets.append(pc)

    add(ParallelConfig())
    tps = [size for size in strategies["tp"] if size > 1]
    pps = [size for size in strategies["pp"] if size > 1]
    for tp in tps[:3]:
        add(ParallelConfig(tp=tp, sp=True))
    for pp in pps[:2]:
        add(ParallelConfig(pp=pp))
    for tp in tps[:2]:
        for pp in pps[:2]:
            add(ParallelConfig(tp=tp, pp=pp, sp=True))
    for tp in tps[:2]:
        add(ParallelConfig(tp=tp, cp=2, sp=True))
    for ep in [size for size in strategies["ep"] if size > 1][:3]:
        add(ParallelConfig(tp=tps[0] if tps else 1, ep=ep))
    if tps and pps:
        add(ParallelConfig(tp=tps[0], pp=pps[0], dp=2, sp=True))
    return presets


def show_preset_menu(cfg, presets, dtype="bfloat16", *, cache=None):
    table = Table(title="Preset configurations", box=box.ROUNDED, expand=True)
    table.add_column("#", justify="right", width=3)
    table.add_column("Configuration", ratio=3, overflow="fold")
    table.add_column("GPUs", justify="right", width=5)
    table.add_column("Peak weight/GPU", justify="right", ratio=2)
    for index, pc in enumerate(presets, 1):
        memory = estimate_weight_size(cfg, tp=pc.tp, pp=pc.pp, ep=pc.ep, dtype=dtype, cache=cache)
        table.add_row(str(index), Text(str(pc)), str(pc.world_size), format_bytes(memory))
    console.print(table)
    console.print("[dim]Weight estimates exclude activations, KV cache and workspace.[/dim]")


def show_action_menu():
    console.print(
        "[bold]1[/] Analyze   [bold]2[/] Generate weights   [bold]3[/] Compare\n"
        "[bold]4[/] Edit config   [bold]5[/] Presets / GPU budget   [bold]6[/] View options\n"
        "[bold]7[/] Available dimensions   [bold]0[/] Exit"
    )


def _positive_int(label, default=1, maximum=None):
    while True:
        value = IntPrompt.ask(label, default=default)
        if value >= 1 and (maximum is None or value <= maximum):
            return value
        console.print(f"[red]Enter a value between 1 and {maximum}.[/red]" if maximum else
                      "[red]Enter a positive integer.[/red]")


def _show_validation(cfg, pc):
    errors = pc.validate(cfg, include_warnings=False)
    for error in errors:
        console.print(Text(error, style="red"))
    for warning in pc.validation_warnings(cfg):
        console.print(Text(warning, style="yellow"))
    return not errors


def custom_config_flow(cfg, strategies, current=None, gpu_budget=None):
    pc = replace(current) if current is not None else ParallelConfig()
    for key in ("tp", "pp", "cp", "ep"):
        if key == "cp" and pc.tp == 1:
            pc.cp = 1
            continue
        if key == "ep" and not cfg.is_moe:
            pc.ep = 1
            continue
        choices = strategies[key]
        default = getattr(pc, key) if getattr(pc, key) in choices else 1
        console.print(Text(f"{key.upper()} sizes: {', '.join(map(str, choices))}", style="dim"))
        setattr(pc, key, IntPrompt.ask(key.upper(), default=default, choices=[str(size) for size in choices]))
    pc.sp = Confirm.ask("Enable sequence parallel?", default=pc.sp) if pc.tp > 1 else False
    pc.dp = _positive_int("DP replicas", default=pc.dp)
    if not _show_validation(cfg, pc):
        return None
    if gpu_budget is not None and pc.world_size > gpu_budget:
        console.print(f"[red]This configuration needs {pc.world_size} GPUs; budget is {gpu_budget}.[/red]")
        return None
    return pc


def run_analysis(cfg, pc, *, dtype="bfloat16", view=None):
    if not _show_validation(cfg, pc):
        return
    results = analyze_model(cfg, pc, dtype)
    console.print(Text(f"Weight precision: {dtype}", style="dim"))
    print_analysis(cfg, pc, results, console=console, **(view or {}))
    if Confirm.ask("Show framework references?", default=False):
        show_framework_refs(pc)
    if Confirm.ask("Export complete analysis to JSON?", default=False):
        path = os.path.expanduser(Prompt.ask("Output path", default="parallel_result.json"))
        with open(path, "w") as handle:
            json.dump(analysis_as_dict(cfg, pc, results, dtype), handle, indent=2)
        console.print(Text(f"Saved all {len(results)} kernels to {path}", style="green"))


def show_framework_refs(pc):
    references = active_framework_refs(pc)
    if not references:
        console.print("[dim]No distributed strategies are enabled.[/dim]")
    for key, reference in references:
        text = "\n".join(f"{name}: {reference.get(name, 'N/A')}"
                         for name in ("pattern", "sglang", "vllm", "megatron"))
        console.print(Panel(Text(text), title=key, border_style="blue"))


def run_weight_gen(cfg, pc, *, dtype="bfloat16", source_dir=None):
    from weight_gen.main import generate_model
    from weight_gen.weight_shaper import build_weight_map, compute_total_bytes

    output_dir = os.path.expanduser(Prompt.ask("Output directory", default="./rand_model"))
    num_layers = cfg.num_hidden_layers
    if Confirm.ask(f"Truncate the {cfg.num_hidden_layers}-layer model?", default=False):
        num_layers = _positive_int("Layers to generate", default=min(4, cfg.num_hidden_layers),
                                   maximum=cfg.num_hidden_layers)
    dtype = Prompt.ask("Weight precision", choices=list(DTYPE_BYTES), default=dtype)
    init = Prompt.ask("Initialization", choices=["normal", "uniform", "zeros", "small"], default="normal")
    all_ranks = Confirm.ask("Generate all TP ranks in separate directories?", default=False)
    benchmark = None
    if Confirm.ask("Generate a benchmark script?", default=False):
        benchmark = Prompt.ask("Backend", choices=["vllm", "sglang", "hf"], default="vllm")
    source_dir = os.path.expanduser(Prompt.ask("Tokenizer source directory (empty to skip)", default=source_dir or ""))
    actual_cfg = replace(cfg, num_hidden_layers=num_layers)
    if not _show_validation(actual_cfg, pc):
        return None
    entries = build_weight_map(actual_cfg, pc, dtype=dtype)
    total_bytes = compute_total_bytes(entries)
    plan = Table.grid(padding=(0, 2))
    plan.add_column(style="bold cyan")
    plan.add_column(overflow="fold")
    for key, value in (
        ("Output", output_dir), ("Layers", f"{num_layers} / {cfg.num_hidden_layers}"),
        ("Precision / init", f"{dtype} / {init}"), ("Tensors per TP rank", str(len(entries))),
        ("Weights per TP rank", format_bytes(total_bytes)),
        ("Total files' weight size", format_bytes(total_bytes * (pc.tp if all_ranks else 1))),
        ("Checkpoint scope", "All PP stages; selected TP/EP shard"),
    ):
        plan.add_row(key, Text(value))
    console.print(Panel(plan, title="Weight generation plan", border_style="green"))
    if os.path.isdir(output_dir):
        with os.scandir(output_dir) as files:
            nonempty = next(files, None) is not None
        if nonempty and not Confirm.ask("Output directory is not empty. Replace generated files?", default=False):
            return None
    if not Confirm.ask("Generate these weights?", default=False):
        return None
    generate_model(cfg=cfg, pc=pc, output_dir=output_dir, source_dir=source_dir or None,
                   num_layers=num_layers, dtype=dtype, init_strategy=init,
                   all_ranks=all_ranks, benchmark=benchmark, verbose=True)
    return dtype


def run_compare(cfg, strategies, *, gpu_budget=None, dtype="bfloat16", cache=None):
    presets = generate_presets(cfg, strategies, gpu_budget)
    if len(presets) < 2:
        console.print("[yellow]Only one preset fits this GPU budget.[/yellow]")
        return
    show_preset_menu(cfg, presets, dtype, cache=cache)
    while True:
        selection = Prompt.ask("Compare preset numbers (comma-separated, 0 to cancel)", default="1,2")
        if selection.strip() == "0":
            return
        try:
            indices = list(dict.fromkeys(int(value.strip()) - 1 for value in selection.split(",")))
        except ValueError:
            indices = []
        if len(indices) >= 2 and all(0 <= index < len(presets) for index in indices):
            break
        console.print("[red]Select at least two distinct valid preset numbers.[/red]")
    print_comparison(cfg, [presets[index] for index in indices], console=console, dtype=dtype)


def _view_options(cfg, pc, current, dtype):
    view = dict(current)
    dtype = Prompt.ask("Weight precision", choices=list(DTYPE_BYTES), default=dtype)
    category = Prompt.ask("Kernel category", choices=["all", *CATEGORY_NAMES], default=view.get("category") or "all")
    view["category"] = None if category == "all" else category
    for key, label, maximum in (("layer_idx", "Layer", cfg.num_hidden_layers), ("pp_stage", "Stage", pc.pp)):
        while True:
            value = Prompt.ask(f"{label} (0–{maximum - 1}, or all)",
                               default=str(view[key]) if view.get(key) is not None else "all")
            if value == "all":
                view[key] = None
                break
            if value.isdigit() and int(value) < maximum:
                view[key] = int(value)
                break
            console.print("[red]Enter an available index or all.[/red]")
    view["show_all"] = Confirm.ask("Show every layer instead of representative layers?", default=view.get("show_all", False))
    view["detailed"] = Confirm.ask("Show activation, communication and HF tensor details?", default=view.get("detailed", False))
    return view, dtype


def interactive_session(cfg, *, source_dir=None):
    strategies = detect_strategies(cfg)
    pc = ParallelConfig()
    dtype = cfg.raw.get("torch_dtype", "bfloat16")
    if dtype not in DTYPE_BYTES:
        dtype = "bfloat16"
    gpu_budget = None
    view = {}
    cache = {}
    show_model_summary(cfg, dtype, cache=cache)
    while True:
        memory = estimate_weight_size(cfg, tp=pc.tp, pp=pc.pp, ep=pc.ep, dtype=dtype, cache=cache)
        filters = ", ".join(f"{key}={value}" for key, value in view.items() if value is not None and value is not False) or "Compact, representative layers"
        console.print(Panel(Text(
            f"{pc} | {pc.world_size} GPUs | Peak weights: {format_bytes(memory)} ({dtype})\n"
            f"GPU budget: {gpu_budget or 'unlimited'} | View: {filters}"
        ), title="Current selection", border_style="cyan"))
        show_action_menu()
        try:
            choice = IntPrompt.ask("Action", choices=[str(value) for value in range(8)], default=1)
            if choice == 0:
                break
            if choice == 1:
                run_analysis(cfg, pc, dtype=dtype, view=view)
            elif choice == 2:
                selected_dtype = run_weight_gen(cfg, pc, dtype=dtype, source_dir=source_dir)
                if selected_dtype:
                    dtype = selected_dtype
            elif choice == 3:
                run_compare(cfg, strategies, gpu_budget=gpu_budget, dtype=dtype, cache=cache)
            elif choice == 4:
                selected = custom_config_flow(cfg, strategies, current=pc, gpu_budget=gpu_budget)
                if selected is not None:
                    pc = selected
            elif choice == 5:
                while True:
                    value = Prompt.ask("GPU budget (positive integer, or all)", default=str(gpu_budget) if gpu_budget else "all")
                    if value == "all" or (value.isdigit() and int(value) > 0):
                        new_budget = None if value == "all" else int(value)
                        break
                    console.print("[red]Enter a positive integer or all.[/red]")
                presets = generate_presets(cfg, strategies, new_budget)
                show_preset_menu(cfg, presets, dtype, cache=cache)
                selection = IntPrompt.ask("Preset (0 to cancel)", default=1,
                                          choices=[str(index) for index in range(len(presets) + 1)])
                if selection:
                    pc = replace(presets[selection - 1])
                    gpu_budget = new_budget
            elif choice == 6:
                view, dtype = _view_options(cfg, pc, view, dtype)
            elif choice == 7:
                show_strategy_menu(cfg, strategies)
            if view.get("pp_stage") is not None and view["pp_stage"] >= pc.pp:
                view["pp_stage"] = None
        except (OSError, ValueError, ImportError) as error:
            console.print(Text(str(error), style="red"))
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session ended.[/dim]")
            break


def main():
    parser = argparse.ArgumentParser(description="Interactive model sharding analysis")
    parser.add_argument("--config", help="HF config.json or model directory")
    parser.add_argument("--model", help="Local model directory")
    parser.add_argument("--width", type=int, help="Override terminal width")
    args = parser.parse_args()
    global console
    if args.width is not None:
        if args.width < 20:
            parser.error("--width must be at least 20")
        console = Console(width=args.width)
    try:
        path = os.path.expanduser(args.config or args.model or Prompt.ask("Config path or model directory"))
        config_path = os.path.join(path, "config.json") if os.path.isdir(path) else path
        cfg = parse_config(config_path)
        errors = ParallelConfig().validate(cfg, include_warnings=False)
        if errors:
            parser.error("; ".join(errors))
        interactive_session(cfg, source_dir=os.path.dirname(os.path.abspath(config_path)))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Session ended.[/dim]")


if __name__ == "__main__":
    main()
