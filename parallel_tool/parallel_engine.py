"""
Parallel engine — compute how each parallel strategy shards every kernel.

Strategies:
  TP  Tensor Parallel   — split weights by column/row/head/vocab
  PP  Pipeline Parallel — split layers across stages
  CP  Context Parallel  — split sequence dimension
  EP  Expert Parallel   — split MoE experts
  DP  Data Parallel     — full replication
  SP  Sequence Parallel — split sequence in non-attention regions (works with TP)
"""
from dataclasses import dataclass
from math import prod
from .config_parser import ModelConfig
from .model_kernels import KernelSpec


@dataclass
class ParallelConfig:
    """User-selected parallel configuration."""
    tp: int = 1   # tensor parallel
    pp: int = 1   # pipeline parallel
    cp: int = 1   # context parallel
    ep: int = 1   # expert parallel
    dp: int = 1   # data parallel
    sp: bool = False  # sequence parallel (requires TP > 1)
    use_te: bool = False  # TransformerEngine optimizations
    overlap_comm: bool = True  # overlap communication with computation

    @property
    def world_size(self) -> int:
        return self.tp * self.pp * self.cp * self.dp * (self.ep if self.ep > 1 else 1)

    def validate(self, cfg: ModelConfig, *, include_warnings: bool = True) -> list[str]:
        """Validate dimensions; optionally include non-fatal layout warnings."""
        errors = []
        for name in ("tp", "pp", "cp", "ep", "dp"):
            size = getattr(self, name)
            if not isinstance(size, int) or isinstance(size, bool) or size < 1:
                errors.append(f"{name.upper()} must be a positive integer (got {size})")
        if errors:
            return errors
        if cfg.num_hidden_layers < 1:
            errors.append("Model must have at least one hidden layer")
        if self.pp > cfg.num_hidden_layers:
            errors.append(f"PP={self.pp} exceeds num_hidden_layers={cfg.num_hidden_layers}")
        if self.tp > 1:
            dimensions = {
                "num_attention_heads": cfg.num_attention_heads,
                "num_key_value_heads": cfg.kv_heads,
                "vocab_size": cfg.vocab_size,
                "moe_intermediate_size" if cfg.is_moe else "intermediate_size":
                    (cfg.moe_intermediate_size or cfg.intermediate_size)
                    if cfg.is_moe else cfg.intermediate_size,
            }
            for name, size in dimensions.items():
                if size % self.tp:
                    errors.append(f"TP={self.tp} must divide {name}={size}")
        if self.ep > 1:
            if not cfg.is_moe:
                errors.append(f"EP={self.ep} requires a MoE model, but this model is dense")
            elif cfg.num_experts % self.ep:
                errors.append(f"EP={self.ep} must divide num_experts={cfg.num_experts}")
        if self.sp and self.tp == 1:
            errors.append("Sequence Parallel (SP) requires TP > 1")
        if include_warnings:
            errors.extend(self.validation_warnings(cfg))
        return errors

    def validation_warnings(self, cfg: ModelConfig) -> list[str]:
        """Layouts that are usable but should be explained to the user."""
        warnings = []
        if isinstance(self.pp, int) and self.pp > 1 and cfg.num_hidden_layers % self.pp:
            base = cfg.num_hidden_layers // self.pp
            warnings.append(
                f"PP={self.pp} distributes layers unevenly: "
                f"{base} or {base + 1} layers per stage"
            )
        if self.cp > 1 and self.tp == 1:
            warnings.append("CP without TP requires a backend that supports this layout")
        return warnings

    def __str__(self) -> str:
        parts = []
        if self.tp > 1: parts.append(f"TP={self.tp}")
        if self.pp > 1: parts.append(f"PP={self.pp}")
        if self.cp > 1: parts.append(f"CP={self.cp}")
        if self.ep > 1: parts.append(f"EP={self.ep}")
        if self.dp > 1: parts.append(f"DP={self.dp}")
        if self.sp: parts.append("SP=on")
        if not parts: parts.append("DP=1 (single GPU)")
        return " | ".join(parts)


@dataclass
class ShardedKernel:
    """Result of sharding a kernel under a parallel config."""
    kernel: KernelSpec
    original_weight_shape: tuple
    sharded_weight_shape: tuple         # per-GPU weight shape
    shard_dims_applied: str             # human-readable description
    original_act_shape: tuple
    sharded_act_shape: tuple            # per-GPU activation shape
    comm_fwd: str
    comm_bwd: str
    memory_per_gpu_bytes: int
    total_memory_bytes: int
    on_this_gpu: bool                   # PP: is this kernel on current stage?
    pp_stage: int                       # which PP stage (0-indexed), -1 if global
    notes: str = ""

    @property
    def is_sharded(self) -> bool:
        return self.sharded_weight_shape != self.original_weight_shape


def compute_sharding(
    kernel: KernelSpec,
    cfg: ModelConfig,
    pc: ParallelConfig,
) -> ShardedKernel:
    """Compute how a single kernel is sharded under the given parallel config."""
    orig_w = kernel.weight_shape
    orig_a = kernel.activation_out_shape
    h = cfg.hidden_size
    hd = cfg.effective_head_dim

    # Start from original shapes
    w = list(orig_w)
    a = list(orig_a)
    comm_fwd = kernel.comm_fwd
    comm_bwd = kernel.comm_bwd
    notes_parts = []
    shard_applied = []

    # Assign global weights as well as decoder layers to pipeline stages.
    on_this = True
    stage = pipeline_stage(kernel, cfg, pc)
    if pc.pp > 1:
        notes_parts.append(f"PP stage {stage}/{pc.pp}")
        shard_applied.append(f"PP: stage {stage}")

    # ── Tensor Parallel ──
    if pc.tp > 1:
        if kernel.shard_type == "column":
            # Linear weights are [in, out]; their bias is [out].
            dim = kernel.shard_dim
            if w and w[dim] % pc.tp == 0:
                w[dim] //= pc.tp
                if a:
                    a[-1] //= pc.tp
                shard_applied.append(f"TP: column split dim={dim} → {w[dim]}")
                comm_fwd = "none (fused with all-reduce via SP)" if pc.sp else comm_fwd
        elif kernel.shard_type == "row":
            # Row parallel: split input dimension (dim=0)
            if len(w) >= 2 and w[0] % pc.tp == 0:
                w[0] = w[0] // pc.tp
                shard_applied.append(f"TP: row split dim=0 → {w[0]}")
                comm_fwd = "all_reduce (fused→reduce_scatter via SP)" if pc.sp else comm_fwd
        elif kernel.shard_type == "head":
            # Head parallel: split by attention heads
            if kernel.category == "attention" and "q_proj" in kernel.name:
                heads_per_gpu = cfg.num_attention_heads // pc.tp
                w[1] = heads_per_gpu * hd
                if a: a[-1] = heads_per_gpu * hd
                shard_applied.append(f"TP: Q heads split → {heads_per_gpu} heads ({w[1]})")
            elif kernel.category == "attention" and "k_proj" in kernel.name:
                kv_heads_per_gpu = cfg.kv_heads // pc.tp
                w[1] = kv_heads_per_gpu * hd
                if a: a[-1] = kv_heads_per_gpu * hd
                shard_applied.append(f"TP: K heads split → {kv_heads_per_gpu} KV heads ({w[1]})")
            elif kernel.category == "attention" and "v_proj" in kernel.name:
                kv_heads_per_gpu = cfg.kv_heads // pc.tp
                w[1] = kv_heads_per_gpu * hd
                if a: a[-1] = kv_heads_per_gpu * hd
                shard_applied.append(f"TP: V heads split → {kv_heads_per_gpu} KV heads ({w[1]})")
            elif kernel.category == "attention" and "core_attention" in kernel.name:
                heads_per_gpu = cfg.num_attention_heads // pc.tp
                if a: a[-1] = heads_per_gpu * hd
                shard_applied.append(f"TP: attention heads → {heads_per_gpu} heads")
        elif kernel.shard_type == "vocab":
            if "embedding" in kernel.name:
                # Vocab parallel embedding: split vocab dimension
                if w[0] % pc.tp == 0:
                    w[0] = w[0] // pc.tp
                    shard_applied.append(f"TP: vocab split dim=0 → {w[0]}")
                    comm_fwd = "all_reduce (or reduce_scatter via SP)"
            elif "lm_head" in kernel.name:
                if w[1] % pc.tp == 0:
                    w[1] = w[1] // pc.tp
                    shard_applied.append(f"TP: vocab split dim=1 → {w[1]}")
        elif kernel.shard_type == "replicated":
            if pc.sp and kernel.category == "norm":
                # With SP, norms operate on sequence-sharded activation
                shard_applied.append("SP: norm on seq-sharded act")
                comm_fwd = "none (act is [seq/tp, hidden])"
            else:
                shard_applied.append("TP: replicated (no weight split)")
        elif kernel.shard_type == "sequence":
            if pc.sp:
                shard_applied.append("SP: norm on seq-sharded act")
                comm_fwd = "none (act is [seq/tp, hidden])"

    # ── Sequence Parallel (affects activations, not weights) ──
    if pc.sp and pc.tp > 1:
        if kernel.category in ("norm",) or kernel.shard_type == "replicated":
            # Activation sequence dim is divided by TP
            # a shape represents per-token; we note seq reduction
            notes_parts.append("SP: seq_len/TP in norm regions")
            comm_fwd = "scatter→gather (seq parallel)"
        elif kernel.shard_type == "column":
            comm_fwd = "none (input gathered from SP, then column-parallel)"
            notes_parts.append("SP: all-gather input before column-parallel")
        elif kernel.shard_type == "row":
            comm_fwd = "reduce-scatter (row→seq-sharded output)"
            notes_parts.append("SP: reduce-scatter output after row-parallel")

    # Expert shapes describe one expert; memory includes all resident experts.
    if kernel.shard_type == "expert" and cfg.is_moe:
        experts_per_gpu = cfg.num_experts // pc.ep
        notes_parts.append(f"{experts_per_gpu} experts/rank; shape is per expert")
        if pc.ep > 1:
            shard_applied.append(f"EP: {experts_per_gpu}/{cfg.num_experts} experts")
            comm_fwd = "all-to-all (dispatch tokens→experts)"
            comm_bwd = "all-to-all (gather expert outputs)"
        if pc.tp > 1:
            dim = kernel.shard_dim
            w[dim] //= pc.tp
            shard_applied.append(f"TP: expert dim={dim} → {w[dim]}")

    # ── Context Parallel (affects sequence dimension of activations) ──
    if pc.cp > 1:
        if kernel.category == "attention" and "core" in kernel.name:
            notes_parts.append(f"CP: seq/{pc.cp} per GPU, ring attention for KV")
            comm_fwd = f"ring attention (send/recv KV, {pc.cp} steps)"
            comm_bwd = f"ring attention backward ({pc.cp} steps)"
        elif kernel.shard_type in ("replicated", "norm"):
            notes_parts.append(f"CP: seq/{pc.cp} for norm/replicated")
        # Weights are not affected by CP

    # ── Data Parallel (no weight sharding) ──
    if pc.dp > 1:
        if kernel.shard_type == "replicated" or (pc.tp == 1 and pc.pp == 1):
            notes_parts.append(f"DP: full replica on {pc.dp} groups")
            comm_bwd = "all_reduce (gradient sync)" if pc.dp > 1 else comm_bwd

    # A weightless kernel contributes zero bytes, including under TP.
    w_numel = prod(w)
    orig_numel = prod(orig_w)
    if kernel.shard_type == "expert" and cfg.is_moe:
        w_numel *= cfg.num_experts // pc.ep
        orig_numel *= cfg.num_experts

    return ShardedKernel(
        kernel=kernel,
        original_weight_shape=tuple(orig_w),
        sharded_weight_shape=tuple(w),
        shard_dims_applied="; ".join(shard_applied) if shard_applied else "none",
        original_act_shape=tuple(orig_a),
        sharded_act_shape=tuple(a),
        comm_fwd=comm_fwd,
        comm_bwd=comm_bwd,
        memory_per_gpu_bytes=w_numel * kernel.dtype_bytes,
        total_memory_bytes=orig_numel * kernel.dtype_bytes,
        on_this_gpu=on_this,
        pp_stage=stage,
        notes=" | ".join(notes_parts) if notes_parts else "",
    )


def pipeline_layer_ranges(num_layers: int, stages: int) -> list[tuple[int, int]]:
    """Return inclusive-start, exclusive-end layer ranges for each stage."""
    base, extra = divmod(num_layers, stages)
    start = 0
    ranges = []
    for stage in range(stages):
        end = start + base + (stage < extra)
        ranges.append((start, end))
        start = end
    return ranges


def pipeline_stage(kernel: KernelSpec, cfg: ModelConfig, pc: ParallelConfig) -> int:
    if pc.pp == 1 or kernel.category == "embedding":
        return 0
    if kernel.layer_idx is None:
        return pc.pp - 1
    layer = kernel.layer_idx
    if not 0 <= layer < cfg.num_hidden_layers:
        raise ValueError(f"Layer {layer} is outside the model")
    base, extra = divmod(cfg.num_hidden_layers, pc.pp)
    larger_stages_end = (base + 1) * extra
    if layer < larger_stages_end:
        return layer // (base + 1)
    return extra + (layer - larger_stages_end) // base


def analyze_all_kernels(
    cfg: ModelConfig,
    pc: ParallelConfig,
    kernels: list[KernelSpec],
) -> list[ShardedKernel]:
    """Compute sharding for all kernels after rejecting invalid dimensions."""
    errors = pc.validate(cfg, include_warnings=False)
    if errors:
        raise ValueError("Invalid parallel configuration: " + "; ".join(errors))
    return [compute_sharding(kernel, cfg, pc) for kernel in kernels]
