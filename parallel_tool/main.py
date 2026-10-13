#!/usr/bin/env python3
"""CLI for model sharding analysis, comparison, and JSON export."""
import argparse
import glob
import json
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parallel_tool.config_parser import ModelConfig, parse_config
from parallel_tool.parallel_engine import ParallelConfig
from parallel_tool.reporting import (
    CATEGORY_NAMES, DTYPE_BYTES, analysis_as_dict, analyze_model,
)
from parallel_tool.table_renderer import print_analysis, print_comparison, render_comparison
from parallel_tool.framework_refs import active_framework_refs

PRESETS = {
    "1gpu": {"tp": 1, "pp": 1, "cp": 1, "ep": 1, "dp": 1, "sp": False},
    "tp2": {"tp": 2, "pp": 1, "cp": 1, "ep": 1, "dp": 1, "sp": False},
    "tp4": {"tp": 4, "pp": 1, "cp": 1, "ep": 1, "dp": 1, "sp": False},
    "tp8": {"tp": 8, "pp": 1, "cp": 1, "ep": 1, "dp": 1, "sp": False},
    "tp4_sp": {"tp": 4, "pp": 1, "cp": 1, "ep": 1, "dp": 1, "sp": True},
    "tp4_pp2": {"tp": 4, "pp": 2, "cp": 1, "ep": 1, "dp": 1, "sp": False},
    "tp2_pp4": {"tp": 2, "pp": 4, "cp": 1, "ep": 1, "dp": 1, "sp": False},
    "tp8_pp4": {"tp": 8, "pp": 4, "cp": 1, "ep": 1, "dp": 1, "sp": True},
    "tp4_pp2_dp4": {"tp": 4, "pp": 2, "cp": 1, "ep": 1, "dp": 4, "sp": True},
    "cp2_tp4": {"tp": 4, "pp": 1, "cp": 2, "ep": 1, "dp": 1, "sp": True},
    "ep8_tp2": {"tp": 2, "pp": 1, "cp": 1, "ep": 8, "dp": 1, "sp": False},
    "tp4_pp4_cp2": {"tp": 4, "pp": 4, "cp": 2, "ep": 1, "dp": 1, "sp": True},
}


def print_framework_refs(pc: ParallelConfig):
    print("\nFramework implementation references")
    for key, reference in active_framework_refs(pc):
        print(f"\n[{key}]")
        for framework in ("pattern", "sglang", "vllm", "megatron"):
            print(f"  {framework}: {reference.get(framework, 'N/A')}")


def compare_strategies(cfg: ModelConfig, strategies: list[ParallelConfig], **options) -> str:
    """Return comparison text without writing to stdout."""
    return render_comparison(cfg, strategies, **options)


def interactive_mode(cfg: ModelConfig, base_dir: str):
    """Both interactive entry points use the same session and controls."""
    from run import interactive_session
    interactive_session(cfg, source_dir=base_dir)


def _parse_comparison(value: str) -> list[ParallelConfig]:
    strategies = []
    custom = {}

    def flush():
        if custom:
            strategies.append(ParallelConfig(**custom))
            custom.clear()

    for token in value.replace(";", ",;, ").split(","):
        token = token.strip()
        if token == ";":
            flush()
        elif token in PRESETS:
            flush()
            strategies.append(ParallelConfig(**PRESETS[token]))
        elif "=" in token:
            key, raw_value = token.split("=", 1)
            key = key.strip().lower()
            raw_value = raw_value.strip()
            if key not in ("tp", "pp", "cp", "ep", "dp", "sp"):
                raise ValueError(f"Unknown parallel dimension: {key}")
            if key in custom:
                flush()
            if key == "sp":
                if raw_value.lower() not in ("true", "false", "y", "n", "1", "0"):
                    raise ValueError("SP must be true or false")
                custom[key] = raw_value.lower() in ("true", "y", "1")
            else:
                custom[key] = int(raw_value)
        else:
            raise ValueError(f"Unknown preset: {token}")
    flush()
    return strategies


def main():
    parser = argparse.ArgumentParser(
        description="Analyze model sharding and per-stage weight memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python -m parallel_tool --config config.json --tp 4 --pp 2
  python -m parallel_tool --config config.json --interactive
  python -m parallel_tool --config config.json --tp 4 --layer 0 --details
  python -m parallel_tool --config config.json --compare 1gpu,tp4,tp4_pp2
  python -m parallel_tool --config config.json --compare 'tp=4,pp=2;tp=8'
""",
    )
    parser.add_argument("--config", help="HF config.json or model directory")
    parser.add_argument("--model", help="HF model name to find in common local paths")
    for dimension in ("tp", "pp", "cp", "ep", "dp"):
        parser.add_argument(f"--{dimension}", type=int, default=1, help=f"{dimension.upper()} size")
    parser.add_argument("--sp", action="store_true", help="Enable sequence parallel")
    parser.add_argument("--dtype", choices=list(DTYPE_BYTES), default="bfloat16", help="Weight estimate precision")
    parser.add_argument("--interactive", "-i", action="store_true", help="Open the interactive session")
    parser.add_argument("--compare", help="Comma-separated presets, or semicolon-separated custom plans")
    parser.add_argument("--json", help="Export the complete analysis, including hidden layers")
    parser.add_argument("--plain", action="store_true", help="Use plain text output")
    parser.add_argument("--width", type=int, help="Override terminal width")
    parser.add_argument("--details", action="store_true", help="Include activation, communication and tensor details")
    parser.add_argument("--all-layers", action="store_true", help="Show every layer")
    parser.add_argument("--category", choices=list(CATEGORY_NAMES), help="Filter kernels by category")
    parser.add_argument("--layer", type=int, help="Filter by zero-based layer index")
    parser.add_argument("--stage", type=int, help="Filter by zero-based pipeline stage")
    parser.add_argument("--refs", action="store_true", help="Append framework references")
    parser.add_argument("--refs-only", action="store_true", help="Show framework references only")
    parser.add_argument("--list-presets", action="store_true", help="List preset configurations")
    args = parser.parse_args()
    if args.width is not None and args.width < 20:
        parser.error("--width must be at least 20")
    if args.list_presets:
        for name, parameters in PRESETS.items():
            print(f"{name:20} {ParallelConfig(**parameters)}")
        return
    config_path = os.path.expanduser(args.config) if args.config else None
    if config_path is None and args.model:
        model_name = args.model.split("/")[-1]
        candidates = [
            os.path.expanduser(f"~/.cache/huggingface/hub/models--{args.model.replace('/', '--')}/snapshots/*/config.json"),
            os.path.expanduser(f"~/models/{model_name}/config.json"),
            f"/mnt/hdd/models/{model_name}/config.json",
        ]
        for candidate in candidates:
            matches = glob.glob(candidate)
            if matches:
                config_path = matches[0]
                break
    if config_path is None:
        parser.error("Provide --config or a --model available locally")
    try:
        cfg = parse_config(config_path)
        if args.interactive:
            base_dir = config_path if os.path.isdir(config_path) else os.path.dirname(os.path.abspath(config_path))
            interactive_mode(cfg, base_dir)
            return
        if args.compare:
            strategies = _parse_comparison(args.compare)
            for pc in strategies:
                errors = pc.validate(cfg, include_warnings=False)
                if errors:
                    raise ValueError("; ".join(errors))
                for warning in pc.validation_warnings(cfg):
                    print(f"Warning: {warning}")
            print_comparison(cfg, strategies, dtype=args.dtype, width=args.width, use_rich=not args.plain)
            return
        pc = ParallelConfig(tp=args.tp, pp=args.pp, cp=args.cp, ep=args.ep, dp=args.dp, sp=args.sp)
        errors = pc.validate(cfg, include_warnings=False)
        if errors:
            raise ValueError("; ".join(errors))
        for warning in pc.validation_warnings(cfg):
            print(f"Warning: {warning}")
        if args.refs_only:
            print_framework_refs(pc)
            return
        results = analyze_model(cfg, pc, args.dtype)
        filters = {"category": args.category, "layer_idx": args.layer, "pp_stage": args.stage,
                   "show_all": args.all_layers}
        print(f"Weight precision: {args.dtype}")
        print_analysis(cfg, pc, results, use_rich=not args.plain, width=args.width,
                       detailed=args.details, **filters)
        if args.refs:
            print_framework_refs(pc)
        if args.json:
            path = os.path.expanduser(args.json)
            with open(path, "w") as handle:
                json.dump(analysis_as_dict(cfg, pc, results, args.dtype), handle, indent=2)
            print(f"\nSaved all {len(results)} kernels to {path}")
    except (OSError, ValueError, ImportError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
