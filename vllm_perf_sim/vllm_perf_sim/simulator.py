"""Discrete-event continuous-batching simulator with chunked prefill."""
from dataclasses import dataclass, field
from math import ceil, exp
import random

from .model import ModelGeometry
from .timing import TimingModel, StepBreakdown


@dataclass
class SimRequest:
    request_id: int
    arrival_s: float
    input_len: int
    output_len: int
    prefilled: int = 0
    decoded: int = 0
    start_s: float | None = None
    first_token_s: float | None = None
    end_s: float | None = None
    token_times: list[float] = field(default_factory=list)
    active_time_s: float = 0.0

    @property
    def remaining_prefill(self) -> int:
        return self.input_len - self.prefilled


@dataclass
class SimulationResult:
    requests: list[SimRequest]
    duration_s: float
    steps: int
    prefill_steps: int
    token_budget: int
    max_num_seqs: int
    max_concurrency: int
    time_samples: list[tuple[float, int]] = field(default_factory=list)
    step_samples: list[dict] = field(default_factory=list)


class Simulator:
    def __init__(self, geometry: ModelGeometry, timing: TimingModel,
                 max_num_batched_tokens: int = 8192, max_num_seqs: int = 256,
                 gpu_memory_utilization: float = 0.90, kv_cache_bytes: int | None = None,
                 collect_steps: bool = False):
        self.geometry = geometry
        self.timing = timing
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.gpu_memory_utilization = gpu_memory_utilization
        self.kv_cache_bytes = kv_cache_bytes
        self.collect_steps = collect_steps
        if max_num_batched_tokens < 1 or max_num_seqs < 1:
            raise ValueError("Scheduler limits must be positive")

    @property
    def max_cached_tokens(self) -> float:
        if self.kv_cache_bytes is not None:
            return self.kv_cache_bytes / self.geometry.kv_bytes_per_token_per_gpu
        gib = self.timing.hardware.memory_gb * 1024 ** 3
        available = gib * self.gpu_memory_utilization - self.geometry.weight_bytes_per_gpu
        return max(0, available / self.geometry.kv_bytes_per_token_per_gpu)

    def run(self, requests: list[SimRequest]) -> SimulationResult:
        if not requests:
            raise ValueError("At least one request is required")
        requests = sorted(requests, key=lambda r: (r.arrival_s, r.request_id))
        waiting, running, completed = [], [], []
        next_request = 0
        now = requests[0].arrival_s
        steps = prefill_steps = 0
        peak_concurrency = 0
        time_samples = []
        step_samples = []
        kv_capacity = self.max_cached_tokens

        while len(completed) < len(requests):
            while next_request < len(requests) and requests[next_request].arrival_s <= now + 1e-12:
                waiting.append(requests[next_request])
                next_request += 1
            if not running and not waiting:
                now = requests[next_request].arrival_s
                continue

            # Admit waiting requests in arrival/FCFS order. Scheduler slots and
            # KV capacity constrain admission; individual chunks are limited below.
            kv_used = sum(r.prefilled + r.decoded for r in running)
            while waiting and len(running) < self.max_num_seqs:
                candidate = waiting[0]
                needed = candidate.input_len
                if kv_used + needed > kv_capacity and running:
                    break
                if needed > kv_capacity and not running:
                    raise ValueError(
                        f"Request {candidate.request_id} needs {needed} KV tokens, "
                        f"but capacity is only {int(kv_capacity)}")
                running.append(waiting.pop(0))
                kv_used += needed

            if not running:
                # No request fits alongside the current working set; this should
                # only occur after an admission race, never silently deadlock.
                raise RuntimeError("No schedulable request; check KV cache capacity")

            scheduled: list[tuple[SimRequest, int, bool]] = []
            budget = self.max_num_batched_tokens
            # vLLM prioritizes decode for already-running requests, then schedules
            # waiting/running prefills into the remaining token budget.
            for req in running:
                if req.prefilled >= req.input_len and req.decoded < req.output_len and budget > 0:
                    scheduled.append((req, 1, False))
                    budget -= 1
            for req in running:
                if req.remaining_prefill and budget > 0:
                    chunk = min(req.remaining_prefill, budget)
                    scheduled.append((req, chunk, True))
                    budget -= chunk

            query_tokens = sum(item[1] for item in scheduled)
            context_tokens = sum(req.prefilled + req.decoded for req in running)
            prefill_tokens = sum(n for _, n, is_prefill in scheduled if is_prefill)
            breakdown = self.timing.step(query_tokens, context_tokens, len(running), prefill_tokens)
            now += breakdown.total_s
            steps += 1
            prefill_steps += bool(prefill_tokens)
            if self.collect_steps:
                step_samples.append({
                    "end_s": now, "running": len(running), "query_tokens": query_tokens,
                    "context_tokens": context_tokens, "prefill_tokens": prefill_tokens,
                    **breakdown.as_dict(),
                })

            finished = []
            for req, tokens, is_prefill in scheduled:
                if req.start_s is None:
                    req.start_s = now - breakdown.total_s
                if is_prefill:
                    req.prefilled += tokens
                    # A forward pass that consumes the final prompt token also
                    # samples the first output token.
                    if req.remaining_prefill == 0:
                        req.decoded = 1
                        req.token_times.append(now)
                        req.first_token_s = now
                else:
                    req.decoded += 1
                    req.token_times.append(now)
                    if req.first_token_s is None:
                        req.first_token_s = now
                if req.decoded >= req.output_len:
                    req.end_s = now
                    finished.append(req)
            for req in finished:
                running.remove(req)
                completed.append(req)
            active = len(running)
            peak_concurrency = max(peak_concurrency, active)
            if time_samples is not None:
                time_samples.append((now, active))

        return SimulationResult(
            requests=requests, duration_s=max(r.end_s for r in completed) - requests[0].arrival_s,
            steps=steps, prefill_steps=prefill_steps,
            token_budget=self.max_num_batched_tokens, max_num_seqs=self.max_num_seqs,
            max_concurrency=peak_concurrency, time_samples=time_samples,
            step_samples=step_samples,
        )


def make_workload(num_prompts: int, input_len: int, output_len: int,
                  request_rate: float = float("inf"), seed: int = 0,
                  input_std: float = 0.0, output_std: float = 0.0) -> list[SimRequest]:
    """Random dataset analogue: Poisson arrivals and optional Gaussian lengths."""
    rng = random.Random(seed)
    requests = []
    arrival = 0.0
    for i in range(num_prompts):
        if request_rate != float("inf"):
            if request_rate <= 0:
                raise ValueError("request_rate must be positive or inf")
            if i:
                arrival += rng.expovariate(request_rate)
        ilen = max(1, round(rng.gauss(input_len, input_std))) if input_std else input_len
        olen = max(1, round(rng.gauss(output_len, output_std))) if output_std else output_len
        requests.append(SimRequest(i, arrival, ilen, olen))
    return requests
