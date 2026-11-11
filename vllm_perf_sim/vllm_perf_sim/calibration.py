"""Calibrate exposed efficiency knobs from vllm bench serve --save-json output."""
import json


def load_benchmark(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if "request_throughput" not in data or "mean_tpot_ms" not in data:
        raise ValueError("Expected a vllm bench serve JSON result with throughput and TPOT")
    return data


def calibrate(benchmark: dict, predicted_duration: float,
              predicted_tpot_ms: float, current_flops_efficiency: float,
              current_memory_efficiency: float) -> dict:
    """Derive one workload-specific correction from a real benchmark.

    TPOT correction primarily captures per-token execution overhead; duration
    correction captures queueing/prefill behavior. Keeping both explicit avoids
    presenting a fitted point as universal hardware performance.
    """
    if predicted_duration <= 0 or predicted_tpot_ms <= 0:
        raise ValueError("Predicted timing must be positive")
    duration_factor = benchmark["duration"] / predicted_duration
    tpot_factor = benchmark["mean_tpot_ms"] / predicted_tpot_ms
    # A single observed TPOT can constrain the aggregate time scale, but cannot
    # uniquely identify compute vs bandwidth efficiency. Report a conservative
    # bandwidth-only suggestion and leave the compute efficiency unchanged.
    return {
        "duration_factor": duration_factor,
        "tpot_factor": tpot_factor,
        "suggested_memory_efficiency": min(1.0, current_memory_efficiency / tpot_factor),
        "suggested_flops_efficiency": current_flops_efficiency,
        "source": {
            "duration_s": benchmark["duration"],
            "request_throughput": benchmark["request_throughput"],
            "output_throughput": benchmark["output_throughput"],
            "mean_tpot_ms": benchmark["mean_tpot_ms"],
            "mean_ttft_ms": benchmark.get("mean_ttft_ms"),
        },
        "warning": "Calibration is workload-specific; validate on other rates and lengths.",
    }
