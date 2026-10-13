#!/usr/bin/env python3
"""
Weight Generator CLI — Generate random-weight models for speed benchmarking.

Usage:
  # Full model, random weights (TP=4 pre-sharded for rank 0)
  python -m weight_gen.main --config config.json --tp 4 --output ./random_model_tp4

  # Truncated model (only 8 layers, for quick speed test)
  python -m weight_gen.main --config config.json --tp 8 --num-layers 8 --output ./rand_model

  # Copy tokenizer from local model dir
  python -m weight_gen.main --config config.json --source-dir /path/to/original --output ./rand

  # Download tokenizer from HuggingFace Hub
  python -m weight_gen.main --config config.json --model-id meta-llama/Llama-3-8B --tp 4 --output ./rand

  # Generate for all TP ranks (subdirectories per rank)
  python -m weight_gen.main --config config.json --tp 4 --all-ranks --output ./rand_tp4

  # Generate benchmark script
  python -m weight_gen.main --config config.json --tp 4 --output ./rand --benchmark vllm
"""
import argparse
import json
import os
import sys
import shutil

# Allow running as script
if __package__ is None or __package__ == "":
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_script_dir))

from parallel_tool.config_parser import parse_config, ModelConfig
from parallel_tool.parallel_engine import ParallelConfig
from weight_gen.weight_shaper import build_weight_map, compute_total_bytes, compute_per_layer_bytes
from weight_gen.weight_writer import generate_random_weights, save_single_file, shard_weights_to_files, write_index_json
from weight_gen.model_cloner import clone_aux_files, download_aux_from_hub, write_config
from weight_gen.benchmark import generate_benchmark_script


def format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def generate_model(
    cfg: ModelConfig,
    pc: ParallelConfig,
    output_dir: str,
    source_dir: str = None,
    model_id: str = None,
    num_layers: int = None,
    dtype: str = "bfloat16",
    init_strategy: str = "normal",
    max_file_size_gb: float = 5.0,
    all_ranks: bool = False,
    rank: int = 0,
    benchmark: str = None,
    verbose: bool = True,
):
    """Main generation pipeline."""
    
    # Override num_layers if truncating
    overrides = {}
    if num_layers is not None and num_layers != cfg.num_hidden_layers:
        overrides["num_hidden_layers"] = num_layers
        cfg.num_hidden_layers = num_layers
        if verbose:
            print(f"\n⚠ Truncated to {num_layers} layers (original: {cfg.num_hidden_layers} → {num_layers})")

    # Compute weight map for this rank
    entries = build_weight_map(cfg, pc)
    total_bytes = compute_total_bytes(entries)
    per_layer_bytes = compute_per_layer_bytes(entries)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Weight Generation Plan")
        print(f"{'='*60}")
        print(f"  Model:        {cfg.model_type} ({cfg.architectures})")
        print(f"  Layers:       {cfg.num_hidden_layers}")
        print(f"  Hidden:       {cfg.hidden_size}")
        print(f"  Heads:        {cfg.num_attention_heads} (KV: {cfg.kv_heads}, dim: {cfg.effective_head_dim})")
        print(f"  Intermediate: {cfg.intermediate_size}")
        print(f"  Vocab:        {cfg.vocab_size}")
        if cfg.is_moe:
            print(f"  MoE:          {cfg.num_experts} experts, {cfg.num_experts_per_tok}/tok")
        print(f"  Parallel:     {pc}")
        print(f"  Rank:         {rank if not all_ranks else 'all'}")
        print(f"  Dtype:        {dtype}")
        print(f"  Init:         {init_strategy}")
        print(f"  Output:       {output_dir}")
        print(f"  Total weights: {len(entries)} tensors")
        print(f"  Per-layer:     {format_bytes(per_layer_bytes)}")
        print(f"  Total:        {format_bytes(total_bytes)}")
        print(f"  Full model:    {format_bytes(total_bytes)} (random, no download needed)")

    # Validate
    errors = pc.validate(cfg)
    if errors:
        print(f"\n⚠ Validation warnings:")
        for e in errors:
            print(f"  - {e}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # ── Copy auxiliary files ──
    if verbose:
        print(f"\n{'='*60}")
        print("Step 1: Auxiliary Files")
        print(f"{'='*60}")
    
    if source_dir:
        if verbose:
            print(f"  Source (local): {source_dir}")
        copied = clone_aux_files(source_dir, output_dir, verbose=verbose)
    elif model_id:
        if verbose:
            print(f"  Source (HF Hub): {model_id}")
        copied = download_aux_from_hub(model_id, output_dir, verbose=verbose)
    else:
        if verbose:
            print("  No source specified, skipping tokenizer files")
            print("  (You can copy tokenizer files manually later)")
        copied = []

    # Write config.json (with overrides)
    raw_cfg = cfg.raw.copy()
    write_config(output_dir, raw_cfg, overrides, verbose=verbose)

    # ── Generate random weights ──
    if verbose:
        print(f"\n{'='*60}")
        print("Step 2: Generate Random Weights")
        print(f"{'='*60}")
    
    # Set dtype on all entries
    for e in entries:
        e.dtype = dtype

    # Determine which rank(s) to generate
    ranks_to_gen = range(pc.tp) if all_ranks else [rank]
    
    for r in ranks_to_gen:
        if all_ranks:
            rank_dir = os.path.join(output_dir, f"tp_rank_{r}")
            os.makedirs(rank_dir, exist_ok=True)
            # For multi-rank, we need to adjust the sharding
            # The weight_map already has shapes for tp=pc.tp sharded
            # Each rank has the same shape, different random values
            rank_seed = 42 + r
            if verbose:
                print(f"\n  Generating rank {r}/{pc.tp}...")
                print(f"    Seed: {rank_seed}, Dir: {rank_dir}")
            tensors = generate_random_weights(entries, seed=rank_seed, init_strategy=init_strategy)
        else:
            rank_dir = output_dir
            if verbose:
                print(f"  Generating weights (seed=42)...")
            tensors = generate_random_weights(entries, seed=42, init_strategy=init_strategy)

        # Save weights
        if total_bytes > max_file_size_gb * 1024**3:
            if verbose:
                print(f"  Splitting into multiple files (max {max_file_size_gb} GB each)")
            files, index = shard_weights_to_files(
                tensors, rank_dir,
                max_file_size_gb=max_file_size_gb,
                file_prefix="model",
            )
            # Write index.json
            if all_ranks:
                write_index_json(rank_dir, index)
            else:
                write_index_json(rank_dir, index)
            if verbose:
                print(f"  ✓ Created {len(files)} safetensors files")
        else:
            fname, index = save_single_file(tensors, rank_dir, "model.safetensors")
            write_index_json(rank_dir, index)
            if verbose:
                print(f"  ✓ Created {fname}")

        # Free memory
        del tensors

    # ── Generate benchmark script ──
    if benchmark:
        if verbose:
            print(f"\n{'='*60}")
            print("Step 3: Benchmark Script")
            print(f"{'='*60}")
        
        script_path = generate_benchmark_script(
            output_dir, os.path.abspath(output_dir),
            backend=benchmark, tp=pc.tp, pp=pc.pp, dtype=dtype,
        )
        if verbose:
            print(f"  ✓ Benchmark script: {script_path}")
            print(f"    Run with: python {script_path} --model-dir {output_dir}")

    # ── Summary ──
    if verbose:
        print(f"\n{'='*60}")
        print("✅ Model Generation Complete!")
        print(f"{'='*60}")
        print(f"  Output directory: {output_dir}")
        print(f"  Total weight size: {format_bytes(total_bytes)}")
        print(f"  Parallel config: {pc}")
        print(f"  Random weights (no real model download needed)")
        
        if benchmark:
            print(f"\n  To benchmark:")
            print(f"    python {script_path} --model-dir {output_dir} --backend {benchmark}")
        
        print(f"\n  To load with vLLM:")
        print(f"    from vllm import LLM")
        print(f"    llm = LLM(model='{output_dir}', tensor_parallel_size={pc.tp}, "
              f"pipeline_parallel_size={pc.pp}, dtype='{dtype}')")
        
        print(f"\n  To load with SGLang:")
        print(f"    python -m sglang.srt.server --model-path {output_dir} "
              f"--tp {pc.tp} --pp {pc.pp} --dtype {dtype}")
        
        print(f"\n  Note: Weights are RANDOM — output is garbage.")
        print(f"  This is for SPEED TESTING only, not for quality evaluation.")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Generate random-weight model for speed benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full model, TP=4
  python -m weight_gen.main --config config.json --tp 4 --output ./rand_model

  # Truncated to 8 layers (quick test)
  python -m weight_gen.main --config config.json --tp 8 --num-layers 8 --output ./rand

  # Copy tokenizer from local source
  python -m weight_gen.main --config config.json --source-dir /path/to/model --tp 4 --output ./rand

  # Download tokenizer from HF Hub
  python -m weight_gen.main --config config.json --model-id meta-llama/Llama-3-8B --tp 4 --output ./rand

  # All TP ranks
  python -m weight_gen.main --config config.json --tp 4 --all-ranks --output ./rand_tp4

  # Generate benchmark script too
  python -m weight_gen.main --config config.json --tp 4 --output ./rand --benchmark vllm

  # Use fp16, zeros init (smallest possible)
  python -m weight_gen.main --config config.json --tp 8 --dtype float16 --init zeros --output ./rand

  # Context parallel + TP
  python -m weight_gen.main --config config.json --tp 4 --cp 2 --output ./rand
""")
    parser.add_argument("--config", required=True, help="Path to HF config.json")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--cp", type=int, default=1, help="Context parallel size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallel size")
    parser.add_argument("--dp", type=int, default=1, help="Data parallel size")
    parser.add_argument("--sp", action="store_true", help="Sequence parallel")
    
    parser.add_argument("--source-dir", type=str, help="Local model dir to copy tokenizer from")
    parser.add_argument("--model-id", type=str, help="HF model ID to download tokenizer from")
    
    parser.add_argument("--num-layers", type=int, help="Truncate to N layers (speed test)")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--init", default="normal", choices=["normal", "uniform", "zeros", "small"],
                        help="Weight initialization strategy")
    parser.add_argument("--max-file-gb", type=float, default=5.0, help="Max safetensors file size (GB)")
    
    parser.add_argument("--all-ranks", action="store_true", help="Generate for all TP ranks")
    parser.add_argument("--rank", type=int, default=0, help="Which TP rank to generate (default 0)")
    
    parser.add_argument("--benchmark", choices=["vllm", "sglang", "hf"], help="Generate benchmark script")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    # Parse config
    cfg = parse_config(args.config)
    
    # Build parallel config
    pc = ParallelConfig(
        tp=args.tp, pp=args.pp, cp=args.cp, ep=args.ep, dp=args.dp, sp=args.sp,
    )

    # Generate
    generate_model(
        cfg=cfg,
        pc=pc,
        output_dir=args.output,
        source_dir=args.source_dir,
        model_id=args.model_id,
        num_layers=args.num_layers,
        dtype=args.dtype,
        init_strategy=args.init,
        max_file_size_gb=args.max_file_gb,
        all_ranks=args.all_ranks,
        rank=args.rank,
        benchmark=args.benchmark,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
