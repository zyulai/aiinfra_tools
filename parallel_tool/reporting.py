"""Shared analysis, memory summaries, filtering, and JSON export."""
from dataclasses import dataclass

from .config_parser import ModelConfig
from .model_kernels import build_kernels
from .parallel_engine import (
    ParallelConfig, ShardedKernel, analyze_all_kernels, pipeline_layer_ranges,
)

DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}
CATEGORY_NAMES = {
    "embedding": "Embedding",
    "attention": "Attention",
    "mlp": "MLP / FFN",
    "moe": "MoE / Experts",
    "norm": "LayerNorm / RMSNorm",
    "lm_head": "LM Head",
}


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} PiB"


@dataclass(frozen=True)
class MemorySummary:
    full_weight_bytes: int
    stage_weight_bytes: tuple[int, ...]

    @property
    def peak_weight_bytes(self) -> int:
        return max(self.stage_weight_bytes, default=0)

    @property
    def min_weight_bytes(self) -> int:
        return min(self.stage_weight_bytes, default=0)


def analyze_model(
    cfg: ModelConfig, pc: ParallelConfig, dtype: str = "bfloat16",
) -> list[ShardedKernel]:
    kernels = build_kernels(cfg)
    for kernel in kernels:
        kernel.dtype_bytes = DTYPE_BYTES[dtype]
    return analyze_all_kernels(cfg, pc, kernels)


def summarize_memory(
    cfg: ModelConfig, pc: ParallelConfig, results: list[ShardedKernel],
) -> MemorySummary:
    stages = [0] * pc.pp
    for result in results:
        stages[max(result.pp_stage, 0)] += result.memory_per_gpu_bytes
    return MemorySummary(
        full_weight_bytes=sum(result.total_memory_bytes for result in results),
        stage_weight_bytes=tuple(stages),
    )


def select_results(
    cfg: ModelConfig,
    pc: ParallelConfig,
    results: list[ShardedKernel],
    *,
    category: str = None,
    layer_idx: int = None,
    pp_stage: int = None,
    show_all: bool = False,
) -> list[ShardedKernel]:
    if category is not None and category not in CATEGORY_NAMES:
        raise ValueError(f"Unknown category: {category}")
    if layer_idx is not None and not 0 <= layer_idx < cfg.num_hidden_layers:
        raise ValueError(f"Layer must be between 0 and {cfg.num_hidden_layers - 1}")
    if pp_stage is not None and not 0 <= pp_stage < pc.pp:
        raise ValueError(f"Stage must be between 0 and {pc.pp - 1}")
    representative_layers = None
    if not show_all and layer_idx is None and cfg.num_hidden_layers > 12:
        representative_layers = set()
        for stage, (start, end) in enumerate(pipeline_layer_ranges(cfg.num_hidden_layers, pc.pp)):
            if pp_stage is None or pp_stage == stage:
                representative_layers.update((start, (start + end - 1) // 2, end - 1))
    selected = []
    for result in results:
        kernel = result.kernel
        if category is not None and kernel.category != category:
            continue
        if layer_idx is not None and kernel.layer_idx != layer_idx:
            continue
        if pp_stage is not None and result.pp_stage != pp_stage:
            continue
        if (representative_layers is not None and kernel.layer_idx is not None
                and kernel.layer_idx not in representative_layers):
            continue
        selected.append(result)
    return selected


def analysis_as_dict(
    cfg: ModelConfig, pc: ParallelConfig, results: list[ShardedKernel],
    dtype: str = "bfloat16",
) -> dict:
    summary = summarize_memory(cfg, pc, results)
    return {
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
            "tp": pc.tp, "pp": pc.pp, "cp": pc.cp, "ep": pc.ep,
            "dp": pc.dp, "sp": pc.sp, "world_size": pc.world_size,
        },
        "dtype": dtype,
        "weight_shape_convention": "linear=[in,out]; embedding=[vocab,hidden]; norm=[hidden]",
        "memory_summary": {
            "full_weight_bytes": summary.full_weight_bytes,
            "stage_weight_bytes": list(summary.stage_weight_bytes),
            "peak_weight_per_gpu_bytes": summary.peak_weight_bytes,
            "scope": "weights only; excludes activations, KV cache, gradients and workspace",
        },
        "kernels": [
            {
                "name": result.kernel.name,
                "category": result.kernel.category,
                "layer_idx": result.kernel.layer_idx,
                "weight_name": result.kernel.weight_name,
                "original_weight_shape": list(result.original_weight_shape),
                "sharded_weight_shape": list(result.sharded_weight_shape),
                "original_act_shape": list(result.original_act_shape),
                "sharded_act_shape": list(result.sharded_act_shape),
                "shard_dims_applied": result.shard_dims_applied,
                "comm_fwd": result.comm_fwd,
                "comm_bwd": result.comm_bwd,
                "memory_per_gpu_bytes": result.memory_per_gpu_bytes,
                "total_memory_bytes": result.total_memory_bytes,
                "pp_stage": result.pp_stage,
                "notes": result.notes,
            }
            for result in results
        ],
    }
