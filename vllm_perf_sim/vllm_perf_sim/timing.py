"""Roofline timing model for a single scheduled engine iteration.

This is an analytical estimate, not a cycle-accurate GPU or kernel simulator.
"""
from dataclasses import dataclass
from math import ceil

from .hardware import HardwareSpec
from .model import DTYPE_BYTES, ModelGeometry


@dataclass(frozen=True)
class StepBreakdown:
    compute_s: float
    weight_read_s: float
    attention_s: float
    communication_s: float
    cpu_overhead_s: float
    total_s: float

    def as_dict(self) -> dict:
        return {key: value * 1000 for key, value in self.__dict__.items()}


@dataclass
class TimingModel:
    geometry: ModelGeometry
    hardware: HardwareSpec
    flops_efficiency: float | None = None
    memory_efficiency: float | None = None
    base_overhead_ms: float | None = None
    per_sequence_overhead_ms: float | None = None
    overlap_communication: bool = True

    def __post_init__(self):
        if self.flops_efficiency is None:
            self.flops_efficiency = self.hardware.flops_efficiency
        if self.memory_efficiency is None:
            self.memory_efficiency = self.hardware.memory_efficiency
        if self.base_overhead_ms is None:
            self.base_overhead_ms = self.hardware.base_step_overhead_s * 1000
        if self.per_sequence_overhead_ms is None:
            self.per_sequence_overhead_ms = self.hardware.per_sequence_overhead_s * 1000

    def step(self, query_tokens: int, context_tokens: int, sequences: int,
             prefill_tokens: int = 0) -> StepBreakdown:
        if min(query_tokens, context_tokens, sequences) < 0:
            raise ValueError("Step dimensions cannot be negative")
        g, h = self.geometry, self.hardware
        compute_key = "fp8" if g.weight_dtype == "fp8" else (
            "fp16" if g.weight_dtype in ("fp16", "fp32") else "bf16")
        flops_capacity = h.compute_tflops.get(compute_key, h.compute_tflops["bf16"]) * 1e12
        flops_capacity *= self.flops_efficiency
        bandwidth = h.bandwidth_gbs * 1e9 * self.memory_efficiency

        # Dense GEMMs plus only the experts selected by the router. The two GEMMs
        # in attention score/value evaluation are represented separately.
        matrix_flops = g.decode_weight_flops(query_tokens)
        score_flops = g.attention_score_flops(query_tokens, max(context_tokens, 1))
        compute_s = (matrix_flops + score_flops) / flops_capacity

        # PagedAttention reads each sequence's KV context. Decode weight reads are
        # shared by a batch; at large batches GEMMs become compute-bound instead.
        weight_read_s = g.weight_bytes_per_gpu / bandwidth
        kv_read = context_tokens * g.kv_bytes_per_token_per_gpu
        attention_s = kv_read / bandwidth
        if prefill_tokens:
            # Chunked prefill shares weight reads across the whole chunk.
            weight_read_s = g.weight_bytes_per_gpu / bandwidth

        tp = g.tp
        communication_s = 0.0
        if tp > 1:
            # Two all-reduces per transformer layer. Ring bandwidth term + a
            # latency term per collective; NVLink values are per-direction.
            activation_bytes = query_tokens * g.config.hidden_size * 2
            collectives = 2 * g.layers_per_stage_max
            ring_volume = activation_bytes * 2 * (tp - 1) / tp
            link = h.nvlink_gbs if h.nvlink_gbs else h.interconnect_gbs
            communication_s = collectives * (0.000008 + ring_volume / (link * 1e9))
        cpu_s = self.base_overhead_ms / 1000 + sequences * self.per_sequence_overhead_ms / 1000
        # Compute and memory access overlap within kernels; use the dominant
        # resource rather than summing both as if they were fully serialized.
        memory_s = weight_read_s + attention_s
        gpu_s = max(compute_s, memory_s)
        if self.overlap_communication:
            gpu_s = max(gpu_s, communication_s)
        else:
            gpu_s += communication_s
        # CPU work can overlap GPU execution in vLLM V1; it still becomes the
        # bottleneck when scheduling/sampling takes longer than the GPU step.
        total = max(gpu_s, cpu_s)
        return StepBreakdown(compute_s, weight_read_s, attention_s,
                             communication_s, cpu_s, total)
