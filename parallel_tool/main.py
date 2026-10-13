#!/usr/bin/env python3
"""
Parallel Tool — Main CLI

Usage:
  python -m parallel_tool.main --config <path> [options]
  python -m parallel_tool.main --config config.json --tp 4 --pp 2
  python -m parallel_tool.main --config config.json --interactive
  python -m parallel_tool.main --model meta-llama/Llama-3-8B --tp 8

Analyzes how different parallel strategies shard every kernel of a model.
"""
import argparse
import json
import os
import sys

# Allow running as script
if __package__ is None or __package__ == "":
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_script_dir))

from parallel_tool.config_parser import parse_config, ModelConfig
from parallel_tool.model_kernels import build_kernels
from parallel_tool.parallel_engine import ParallelConfig, analyze_all_kernels
from parallel_tool.table_renderer import render_table, RICH_AVAILABLE
from parallel_tool.framework_refs import FRAMEWORK_REFS


PRESETS = {
    "1gpu":      {"tp": 1, "pp": 1, "cp": 1, "ep": 1, "dp": 1, "sp": False},
    "tp2":       {"tp": 2, "pp": 1, "cp": 1, "ep": 1, "dp": 1, "sp": False},
    "tp4":       {"tp": 4, "pp": 1, "cp": 1, "ep": 1, "dp": 1, "sp": False},
    "tp8":       {"tp": 8, "pp": 1, "cp": 1, "ep": 1, "dp": 1, "sp": False},
    "tp4_sp":    {"tp": 4, "pp": 1, "cp": 1, "ep": 1, "dp": 1, "sp": True},
    "tp4_pp2":   {"tp": 4, "pp": 2, "cp": 1, "ep": 1, "dp": 1, "sp": False},
    "tp2_pp4":   {"tp": 2, "pp": 4, "cp": 1, "ep": 1, "dp": 1, "sp": False},
    "tp8_pp4":   {"tp": 8, "pp": 4, "cp": 1, "ep": 1, "dp": 1, "sp": True},
    "tp4_pp2_dp4":{"tp": 4, "pp": 2, "cp": 1, "ep": 1, "dp": 4, "sp": True},
    "cp2_tp4":   {"tp": 4, "pp": 1, "cp": 2, "ep": 1, "dp": 1, "sp": True},
    "ep8_tp2":   {"tp": 2, "pp": 1, "cp": 1, "ep": 8, "dp": 1, "sp": False},
    "tp4_pp4_cp2":{"tp": 4, "pp": 4, "cp": 2, "ep": 1, "dp": 1, "sp": True},
}


def print_framework_refs(pc: ParallelConfig):
    """Print framework implementation references for active strategies."""
    print("\n" + "=" * 100)
    print("Framework Implementation References (from SGLang / vLLM / Megatron-LM source)")
    print("=" * 100)

    shown = set()

    if pc.tp > 1:
        refs = ["TP_embedding", "TP_qkv", "TP_o_proj", "TP_mlp_gate_up", "TP_mlp_down"]
        if not pc._skip_lm_head():
            refs.append("TP_lm_head")
        for key in refs:
            ref = FRAMEWORK_REFS.get(key, {})
            if key not in shown:
                print(f"\n[{key}]")
                print(f"  Pattern: {ref.get('pattern', 'N/A')}")
                for fw in ("sglang", "vllm", "megatron"):
                    print(f"  {fw:8s}: {ref.get(fw, 'N/A')}")
                shown.add(key)

    if pc.pp > 1:
        key = "PP_layer_split"
        if key not in shown:
            ref = FRAMEWORK_REFS.get(key, {})
            print(f"\n[{key}]")
            print(f"  Pattern: {ref.get('pattern', 'N/A')}")
            for fw in ("sglang", "vllm", "megatron"):
                print(f"  {fw:8s}: {ref.get(fw, 'N/A')}")
            shown.add(key)

    if pc.cp > 1:
        key = "CP_ring_attention"
        if key not in shown:
            ref = FRAMEWORK_REFS.get(key, {})
            print(f"\n[{key}]")
            print(f"  Pattern: {ref.get('pattern', 'N/A')}")
            for fw in ("sglang", "vllm", "megatron"):
                print(f"  {fw:8s}: {ref.get(fw, 'N/A')}")
            shown.add(key)

    if pc.ep > 1:
        key = "EP_dispatch"
        if key not in shown:
            ref = FRAMEWORK_REFS.get(key, {})
            print(f"\n[{key}]")
            print(f"  Pattern: {ref.get('pattern', 'N/A')}")
            for fw in ("sglang", "vllm", "megatron"):
                print(f"  {fw:8s}: {ref.get(fw, 'N/A')}")
            shown.add(key)

    if pc.sp:
        key = "SP_regions"
        if key not in shown:
            ref = FRAMEWORK_REFS.get(key, {})
            print(f"\n[{key}]")
            print(f"  Pattern: {ref.get('pattern', 'N/A')}")
            for fw in ("sglang", "vllm", "megatron"):
                print(f"  {fw:8s}: {ref.get(fw, 'N/A')}")
            shown.add(key)

    if pc.dp > 1:
        key = "DP_grad_sync"
        if key not in shown:
            ref = FRAMEWORK_REFS.get(key, {})
            print(f"\n[{key}]")
            print(f"  Pattern: {ref.get('pattern', 'N/A')}")
            for fw in ("sglang", "vllm", "megatron"):
                print(f"  {fw:8s}: {ref.get(fw, 'N/A')}")
            shown.add(key)

    print()


def compare_strategies(cfg: ModelConfig, strategies: list[ParallelConfig]) -> str:
    """Compare multiple parallel strategies side by side."""
    lines = []
    lines.append("=" * 120)
    lines.append("Strategy Comparison — Per-Kernel Sharding Summary")
    lines.append("=" * 120)

    # Header
    strategy_names = [str(s) for s in strategies]
    header = f"{'Kernel':<35}"
    for name in strategy_names:
        header += f" | {name[:25]:<25}"
    lines.append(header)
    lines.append("-" * 120)

    # Build kernel list once
    kernels = build_kernels(cfg)
    
    # For each kernel, show sharded shape under each strategy
    for i, k in enumerate(kernels):
        # Skip non-representative layers
        if k.layer_idx is not None and cfg.num_hidden_layers > 8:
            show_idx = {0, cfg.num_hidden_layers // 2, cfg.num_hidden_layers - 1}
            if k.layer_idx not in show_idx:
                continue

        row = f"{k.name:<35}"
        for pc in strategies:
            results = analyze_all_kernels(cfg, pc, kernels)
            r = results[i]
            shape_str = f"{r.sharded_weight_shape}"
            row += f" | {shape_str:<25}"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def interactive_mode(cfg: ModelConfig, base_dir: str):
    """Interactive mode: let user choose strategies interactively."""
    kernels = build_kernels(cfg)
    print(f"\n✓ Model loaded: {cfg.model_type}")
    print(cfg.summary())
    print(f"\n✓ {len(kernels)} kernels identified across {cfg.num_hidden_layers} layers\n")

    while True:
        print("\n" + "=" * 60)
        print("Choose parallel strategy:")
        print("=" * 60)
        print("  1. Single strategy (choose TP/PP/CP/EP/DP/SP)")
        print("  2. Preset combinations")
        print("  3. Custom (specify all)")
        print("  4. Compare multiple strategies")
        print("  5. Show framework references")
        print("  0. Exit")
        
        choice = input("\nChoice: ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            print("\n  Strategy: TP / PP / CP / EP / DP / SP")
            strat = input("  Enter strategy: ").strip().upper()
            size = int(input(f"  Enter {strat} size (default 2): ") or "2")
            
            pc = ParallelConfig()
            if strat == "TP": pc.tp = size
            elif strat == "PP": pc.pp = size
            elif strat == "CP": pc.cp = size
            elif strat == "EP": pc.ep = size
            elif strat == "DP": pc.dp = size
            elif strat == "SP":
                pc.sp = True
                pc.tp = max(size, 2)
            
            errors = pc.validate(cfg)
            if errors:
                print(f"\n⚠ Validation errors: {errors}")
                continue
            
            results = analyze_all_kernels(cfg, pc, kernels)
            table = render_table(cfg, pc, results, use_rich=True)
            print(table)
            print_framework_refs(pc)
            
        elif choice == "2":
            print("\n  Presets:")
            for i, (name, p) in enumerate(PRESETS.items(), 1):
                print(f"    {i}. {name:20s} TP={p['tp']} PP={p['pp']} CP={p['cp']} EP={p['ep']} DP={p['dp']} SP={p['sp']}")
            sel = input("\n  Select (number): ").strip()
            try:
                name = list(PRESETS.keys())[int(sel) - 1]
                params = PRESETS[name]
                pc = ParallelConfig(**params)
            except (ValueError, IndexError):
                print("Invalid selection")
                continue
            
            errors = pc.validate(cfg)
            if errors:
                print(f"\n⚠ Validation errors: {errors}")
                # Continue anyway with warnings
            
            results = analyze_all_kernels(cfg, pc, kernels)
            table = render_table(cfg, pc, results, use_rich=True)
            print(table)
            print_framework_refs(pc)
            
        elif choice == "3":
            pc = ParallelConfig()
            pc.tp = int(input(f"  TP (default 1): ") or "1")
            pc.pp = int(input(f"  PP (default 1): ") or "1")
            pc.cp = int(input(f"  CP (default 1): ") or "1")
            pc.ep = int(input(f"  EP (default 1): ") or "1")
            pc.dp = int(input(f"  DP (default 1): ") or "1")
            sp_input = input(f"  SP (y/N): ").strip().lower()
            pc.sp = sp_input == "y"
            
            errors = pc.validate(cfg)
            if errors:
                print(f"\n⚠ Validation warnings: {errors}")
            
            results = analyze_all_kernels(cfg, pc, kernels)
            table = render_table(cfg, pc, results, use_rich=True)
            print(table)
            print_framework_refs(pc)
            
        elif choice == "4":
            print("\n  Compare strategies. Enter preset names (comma-separated):")
            print(f"  Available: {', '.join(PRESETS.keys())}")
            sel = input("  Selection: ").strip()
            names = [s.strip() for s in sel.split(",")]
            strategies = []
            for n in names:
                if n in PRESETS:
                    strategies.append(ParallelConfig(**PRESETS[n]))
            if not strategies:
                print("No valid strategies selected")
                continue
            print(compare_strategies(cfg, strategies))
            
        elif choice == "5":
            for key, ref in FRAMEWORK_REFS.items():
                print(f"\n[{key}]")
                print(f"  Pattern: {ref.get('pattern', 'N/A')}")
                for fw in ("sglang", "vllm", "megatron"):
                    print(f"  {fw:8s}: {ref.get(fw, 'N/A')}")

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Parallel Tool — Analyze parallel strategy sharding for any HF model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze Llama-3-8B with TP=8
  python -m parallel_tool.main --config path/to/config.json --tp 8

  # Interactive mode
  python -m parallel_tool.main --config config.json --interactive

  # Compare multiple presets
  python -m parallel_tool.main --config config.json --compare tp2,tp4,tp8

  # 3D parallelism: TP=4, PP=2, DP=4, with SP
  python -m parallel_tool.main --config config.json --tp 4 --pp 2 --dp 4 --sp

  # Context parallel + TP
  python -m parallel_tool.main --config config.json --tp 4 --cp 2 --sp

  # Export to JSON
  python -m parallel_tool.main --config config.json --tp 4 --json output.json

  # Show framework references only
  python -m parallel_tool.main --config config.json --tp 4 --refs-only
""")
    parser.add_argument("--config", type=str, help="Path to HF config.json or model directory")
    parser.add_argument("--model", type=str, help="HF model name (will search common local paths)")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--cp", type=int, default=1, help="Context parallel size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallel size")
    parser.add_argument("--dp", type=int, default=1, help="Data parallel size")
    parser.add_argument("--sp", action="store_true", help="Enable sequence parallel")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--compare", type=str, help="Compare presets (comma-separated)")
    parser.add_argument("--json", type=str, help="Export results to JSON file")
    parser.add_argument("--plain", action="store_true", help="Use plain text tables (no rich)")
    parser.add_argument("--refs-only", action="store_true", help="Only show framework references")
    parser.add_argument("--list-presets", action="store_true", help="List available presets and exit")

    args = parser.parse_args()

    # List presets
    if args.list_presets:
        print("Available presets:")
        for name, params in PRESETS.items():
            print(f"  {name:20s} TP={params['tp']} PP={params['pp']} CP={params['cp']} "
                  f"EP={params['ep']} DP={params['dp']} SP={params['sp']}")
        return

    # Resolve config
    config_path = args.config
    if not config_path and args.model:
        # Try common local paths
        candidates = [
            os.path.expanduser(f"~/.cache/huggingface/hub/models--{args.model.replace('/', '--')}/snapshots/*/config.json"),
            os.path.expanduser(f"~/models/{args.model.split('/')[-1]}/config.json"),
            f"/mnt/hdd/models/{args.model.split('/')[-1]}/config.json",
        ]
        for c in candidates:
            import glob
            matches = glob.glob(c)
            if matches:
                config_path = matches[0]
                break
        if not config_path:
            print(f"Cannot find config for model: {args.model}")
            print("Please provide --config path/to/config.json")
            return

    if not config_path:
        parser.error("--config or --model is required")

    # Parse config
    try:
        cfg = parse_config(config_path)
    except Exception as e:
        print(f"Error parsing config: {e}")
        return

    print(f"\n✓ Model loaded: {cfg.model_type}")
    print(cfg.summary())

    # Build kernels
    kernels = build_kernels(cfg)
    print(f"\n✓ {len(kernels)} kernels identified across {cfg.num_hidden_layers} layers")

    # Interactive mode
    if args.interactive:
        base_dir = os.path.dirname(os.path.abspath(config_path))
        interactive_mode(cfg, base_dir)
        return

    # Compare mode
    if args.compare:
        names = [s.strip() for s in args.compare.split(",")]
        strategies = []
        for n in names:
            if n in PRESETS:
                strategies.append(ParallelConfig(**PRESETS[n]))
            elif "=" in n:
                # Parse like "tp=4,pp=2"
                k, v = n.split("=")
                pc = ParallelConfig()
                if k.lower() == "tp": pc.tp = int(v)
                elif k.lower() == "pp": pc.pp = int(v)
                elif k.lower() == "cp": pc.cp = int(v)
                elif k.lower() == "ep": pc.ep = int(v)
                elif k.lower() == "dp": pc.dp = int(v)
                elif k.lower() == "sp": pc.sp = (v.lower() in ("y", "true", "1"))
                strategies.append(pc)
        if strategies:
            print(compare_strategies(cfg, strategies))
            return

    # Build parallel config
    pc = ParallelConfig(
        tp=args.tp, pp=args.pp, cp=args.cp, ep=args.ep, dp=args.dp, sp=args.sp,
    )

    # Validate
    errors = pc.validate(cfg)
    if errors:
        print(f"\n⚠ Validation warnings:")
        for e in errors:
            print(f"  - {e}")
        print("  (Results will show non-divisible sharding as-is)\n")

    print(f"\nParallel: {pc}")
    print(f"World Size: {pc.world_size} GPU(s)")

    # Refs only
    if args.refs_only:
        print_framework_refs(pc)
        return

    # Analyze
    results = analyze_all_kernels(cfg, pc, kernels)

    # Render table
    use_rich = not args.plain
    table = render_table(cfg, pc, results, use_rich=use_rich)
    print(table)

    # Framework refs
    print_framework_refs(pc)

    # JSON export
    if args.json:
        data = {
            "model": {
                "model_type": cfg.model_type,
                "architectures": cfg.architectures,
                "hidden_size": cfg.hidden_size,
                "num_hidden_layers": cfg.num_hidden_layers,
                "num_attention_heads": cfg.num_attention_heads,
                "num_key_value_heads": cfg.kv_heads,
                "head_dim": cfg.effective_head_dim,
                "intermediate_size": cfg.intermediate_size,
                "vocab_size": cfg.vocab_size,
                "is_moe": cfg.is_moe,
                "num_experts": cfg.num_experts,
            },
            "parallel_config": {
                "tp": pc.tp, "pp": pc.pp, "cp": pc.cp, "ep": pc.ep, "dp": pc.dp, "sp": pc.sp,
                "world_size": pc.world_size,
            },
            "kernels": [
                {
                    "name": r.kernel.name,
                    "category": r.kernel.category,
                    "weight_name": r.kernel.weight_name,
                    "original_weight_shape": list(r.original_weight_shape),
                    "sharded_weight_shape": list(r.sharded_weight_shape),
                    "original_act_shape": list(r.original_act_shape),
                    "sharded_act_shape": list(r.sharded_act_shape),
                    "shard_dims_applied": r.shard_dims_applied,
                    "comm_fwd": r.comm_fwd,
                    "comm_bwd": r.comm_bwd,
                    "memory_per_gpu_bytes": r.memory_per_gpu_bytes,
                    "total_memory_bytes": r.total_memory_bytes,
                    "pp_stage": r.pp_stage,
                    "notes": r.notes,
                }
                for r in results
            ],
        }
        with open(args.json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n✓ Results exported to {args.json}")


if __name__ == "__main__":
    main()
