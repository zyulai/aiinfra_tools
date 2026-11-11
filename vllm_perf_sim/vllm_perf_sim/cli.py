"""Command-line interface for analytical serving simulations."""
import argparse
import json
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vllm_perf_sim.hardware import HARDWARE, get_hardware
from vllm_perf_sim.model import ModelGeometry
from vllm_perf_sim.timing import TimingModel
from vllm_perf_sim.simulator import Simulator, make_workload
from vllm_perf_sim.metrics import calculate_metrics, render_report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vllm-perf-sim",
        description="Estimate vLLM serving throughput/latency. Output is simulated, not measured.")
    p.add_argument("--config", required=True, help="HF config.json or model directory")
    p.add_argument("--gpu", default="h100-sxm", choices=sorted(HARDWARE))
    p.add_argument("--num-gpus", type=int, default=1, help="Total GPUs (must equal TP*PP)")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32", "fp8", "int8", "int4"])
    p.add_argument("--kv-dtype", default="bf16", choices=["bf16", "fp16", "fp8", "int8"])
    p.add_argument("--num-prompts", type=int, default=100)
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--input-std", type=float, default=0.0)
    p.add_argument("--output-std", type=float, default=0.0)
    p.add_argument("--request-rate", type=float, default=float("inf"),
                   help="Poisson arrivals per second; inf sends all requests immediately")
    p.add_argument("--max-concurrency", type=int, default=None,
                   help="Admission cap, like bench serve --max-concurrency")
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--kv-cache-gb", type=float, default=None, help="Explicit KV cache capacity")
    p.add_argument("--flops-efficiency", type=float, default=None)
    p.add_argument("--memory-efficiency", type=float, default=None)
    p.add_argument("--base-overhead-ms", type=float, default=None)
    p.add_argument("--per-sequence-overhead-ms", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-json", type=str, default=None)
    p.add_argument("--save-steps", type=str, default=None)
    p.add_argument("--list-gpus", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.list_gpus:
        for key, spec in HARDWARE.items():
            print(f"{key:<12} {spec.memory_gb:>5.0f} GB  {spec.bandwidth_gbs:>5.0f} GB/s  "
                  f"{spec.compute_tflops['bf16']:>7.1f} BF16 TFLOPS")
        return
    from parallel_tool.config_parser import parse_config
    config = parse_config(args.config)
    expected_gpus = args.tp * args.pp * (args.ep if config.is_moe else 1)
    if expected_gpus != args.num_gpus:
        raise SystemExit("--num-gpus must equal TP*PP*EP for MoE, or TP*PP for dense models")
    if args.num_prompts < 1 or args.input_len < 1 or args.output_len < 1:
        raise SystemExit("Prompt count and sequence lengths must be positive")

    geometry = ModelGeometry(config, weight_dtype=args.dtype, kv_dtype=args.kv_dtype,
                             tp=args.tp, pp=args.pp, ep=args.ep)
    hardware = get_hardware(args.gpu)
    timing = TimingModel(
        geometry, hardware, flops_efficiency=args.flops_efficiency,
        memory_efficiency=args.memory_efficiency,
        base_overhead_ms=args.base_overhead_ms,
        per_sequence_overhead_ms=args.per_sequence_overhead_ms,
    )
    max_seqs = min(args.max_num_seqs, args.max_concurrency) if args.max_concurrency else args.max_num_seqs
    simulator = Simulator(
        geometry, timing, max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=max_seqs, gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_bytes=int(args.kv_cache_gb * 1024**3) if args.kv_cache_gb else None,
        collect_steps=bool(args.save_steps),
    )
    workload = make_workload(args.num_prompts, args.input_len, args.output_len,
                             args.request_rate, args.seed, args.input_std, args.output_std)
    simulation = simulator.run(workload)
    metrics = calculate_metrics(simulation)
    metadata = {
        "model": args.config, "gpu": args.gpu, "gpus": args.num_gpus,
        "tp": args.tp, "pp": args.pp, "ep": args.ep, "dtype": args.dtype,
        "kv_dtype": args.kv_dtype, "seed": args.seed,
        "request_rate": args.request_rate, "max_concurrency": args.max_concurrency,
        "weights_per_gpu_gib": geometry.weight_bytes_per_gpu / 1024**3,
        "kv_bytes_per_token_per_gpu": geometry.kv_bytes_per_token_per_gpu,
        "kv_capacity_tokens": int(simulator.max_cached_tokens),
        "warning": "SIMULATED estimates, not measured vLLM performance. Calibrate before capacity decisions.",
    }
    print(render_report(metrics, metadata))
    print(f"Weights/GPU: {metadata['weights_per_gpu_gib']:.2f} GiB | "
          f"KV/token/GPU: {metadata['kv_bytes_per_token_per_gpu'] / 1024:.1f} KiB | "
          f"KV capacity: {metadata['kv_capacity_tokens']:,} tokens")
    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as handle:
            json.dump({**metadata, **metrics}, handle, indent=2)
        print(f"Saved metrics to {args.save_json}")
    if args.save_steps:
        with open(args.save_steps, "w", encoding="utf-8") as handle:
            json.dump(simulation.step_samples, handle, indent=2)
        print(f"Saved {len(simulation.step_samples)} step records to {args.save_steps}")


if __name__ == "__main__":
    main()
