"""Metrics compatible with vllm bench serve's JSON field names and formulas."""
import math
import statistics

from .simulator import SimulationResult


def percentile(values: list[float], p: float) -> float:
    """NumPy-compatible linear percentile without a NumPy dependency."""
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * p / 100
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def distribution(values: list[float], prefix: str, percentiles=(50, 90, 95, 99)) -> dict:
    result = {
        f"mean_{prefix}_ms": statistics.fmean(values) * 1000 if values else 0.0,
        f"median_{prefix}_ms": statistics.median(values) * 1000 if values else 0.0,
        f"std_{prefix}_ms": statistics.pstdev(values) * 1000 if values else 0.0,
    }
    for p in percentiles:
        result[f"p{int(p) if int(p) == p else p}_{prefix}_ms"] = percentile(values, p) * 1000
    return result


def calculate_metrics(simulation: SimulationResult) -> dict:
    requests = simulation.requests
    ttfts = [r.first_token_s - r.arrival_s for r in requests]
    e2els = [r.end_s - r.arrival_s for r in requests]
    latencies = [r.end_s - r.start_s for r in requests]
    tpots, all_itls = [], []
    for req in requests:
        req_tpot = ((req.end_s - req.first_token_s) / (req.output_len - 1)
                    if req.output_len > 1 else 0.0)
        if req.output_len > 1:
            tpots.append(req_tpot)
        itls = [b - a for a, b in zip(req.token_times, req.token_times[1:])]
        all_itls.extend(itls)
    duration = simulation.duration_s
    total_input = sum(r.input_len for r in requests)
    total_output = sum(r.output_len for r in requests)
    client_queues = [r.start_s - r.arrival_s for r in requests]
    server_ttfts = [r.first_token_s - r.start_s for r in requests]
    result = {
        "date": None,
        "backend": "vllm-perf-sim",
        "duration": duration,
        "completed": len(requests),
        "failed": 0,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "request_throughput": len(requests) / duration,
        "request_goodput": None,
        "output_throughput": total_output / duration,
        "total_token_throughput": (total_input + total_output) / duration,
        "input_lens": [r.input_len for r in requests],
        "output_lens": [r.output_len for r in requests],
        "ttfts": ttfts,
        "itls": [[b - a for a, b in zip(r.token_times, r.token_times[1:])] for r in requests],
        "tpots": [((r.end_s - r.first_token_s) / (r.output_len - 1)
                   if r.output_len > 1 else 0.0) for r in requests],
        "latencies": e2els,
        "server_latencies": latencies,
        "client_queue_times": client_queues,
        "server_ttfts": server_ttfts,
        "max_concurrent_requests": simulation.max_concurrency,
        "max_output_tokens_per_s": peak_output_throughput(requests),
        "simulation": {
            "engine_steps": simulation.steps,
            "prefill_steps": simulation.prefill_steps,
            "max_num_batched_tokens": simulation.token_budget,
            "max_num_seqs": simulation.max_num_seqs,
        },
    }
    for values, name in ((ttfts, "ttft"), (tpots, "tpot"), (all_itls, "itl"),
                         (e2els, "e2el"), (client_queues, "client_queue_time"),
                         (server_ttfts, "server_ttft")):
        result.update(distribution(values, name))
    return result


def peak_output_throughput(requests) -> float:
    if not requests:
        return 0.0
    start = min(r.arrival_s for r in requests)
    end = max(r.end_s for r in requests)
    buckets = [0] * (math.ceil(end - start) + 1)
    for req in requests:
        for timestamp in req.token_times:
            index = int(timestamp - start)
            if 0 <= index < len(buckets):
                buckets[index] += 1
    return float(max(buckets, default=0))


def render_report(metrics: dict, metadata: dict | None = None) -> str:
    """Render the familiar vllm bench serve console report."""
    def row(label, value, fmt=".2f"):
        return f"{label:<40} {value:<10{fmt}}"

    lines = [" Serving Benchmark Result ".center(50, "=")]
    lines.append(row("Successful requests:", metrics["completed"], ""))
    lines.append(row("Benchmark duration (s):", metrics["duration"]))
    lines.append(row("Total input tokens:", metrics["total_input_tokens"], ""))
    lines.append(row("Total generated tokens:", metrics["total_output_tokens"], ""))
    lines.append(row("Request throughput (req/s):", metrics["request_throughput"]))
    lines.append(row("Output token throughput (tok/s):", metrics["output_throughput"]))
    lines.append(row("Peak output token throughput (tok/s):", metrics["max_output_tokens_per_s"]))
    lines.append(row("Peak concurrent requests:", metrics["max_concurrent_requests"], ""))
    lines.append(row("Total token throughput (tok/s):", metrics["total_token_throughput"]))
    for title, name in (
        ("Time to First Token", "ttft"),
        ("Time per Output Token (excl. 1st token)", "tpot"),
        ("Inter-token Latency", "itl"),
        ("End-to-end Latency", "e2el"),
    ):
        lines.append(title.center(50, "-"))
        lines.append(row(f"Mean {name.upper()} (ms):", metrics[f"mean_{name}_ms"]))
        lines.append(row(f"Median {name.upper()} (ms):", metrics[f"median_{name}_ms"]))
        for p in (90, 95, 99):
            lines.append(row(f"P{p} {name.upper()} (ms):", metrics[f"p{p}_{name}_ms"]))
    lines.append("=" * 50)
    if metadata:
        lines.append("SIMULATION — not measured GPU performance")
        lines.append(f"Model: {metadata.get('model', 'unknown')} | GPU: {metadata.get('gpu', 'unknown')} "
                     f"x{metadata.get('gpus', 1)} | TP={metadata.get('tp', 1)} PP={metadata.get('pp', 1)}")
        if metadata.get("calibration"):
            lines.append(f"Calibration: {metadata['calibration']}")
    return "\n".join(lines)
