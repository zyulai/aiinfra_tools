"""Compare several serving loads with one model/hardware configuration."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "vllm_perf_sim")]

from parallel_tool.config_parser import parse_config
from vllm_perf_sim import ModelGeometry, TimingModel, Simulator, get_hardware, make_workload, calculate_metrics

SCENARIOS = {
    "sharegpt-like": (202, 179),
    "prefill-heavy": (462, 16),
    "decode-heavy": (462, 256),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--gpu", default="h100-sxm")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--num-prompts", type=int, default=500)
    p.add_argument("--request-rate", type=float, default=float("inf"))
    args = p.parse_args()
    geometry = ModelGeometry(parse_config(args.config), tp=args.tp, pp=args.pp)
    timing = TimingModel(geometry, get_hardware(args.gpu))
    print(f"{'scenario':<16} {'req/s':>9} {'out tok/s':>12} {'TTFT ms':>12} {'TPOT ms':>12} {'P99 E2EL ms':>14}")
    for name, (input_len, output_len) in SCENARIOS.items():
        sim = Simulator(geometry, timing)
        result = sim.run(make_workload(args.num_prompts, input_len, output_len, args.request_rate))
        m = calculate_metrics(result)
        print(f"{name:<16} {m['request_throughput']:>9.2f} {m['output_throughput']:>12.2f} "
              f"{m['mean_ttft_ms']:>12.2f} {m['mean_tpot_ms']:>12.2f} {m['p99_e2el_ms']:>14.2f}")
    print("\nSIMULATED estimates; scenario shapes follow the vLLM v0.6.0 blog, not its measured values.")


if __name__ == "__main__":
    main()
