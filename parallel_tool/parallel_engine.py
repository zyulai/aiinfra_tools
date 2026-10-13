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
from dataclasses import dataclass, field
from typing import Optional
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

    def validate(self, cfg: ModelConfig) -> list[str]:
        """Validate against model config; return list of error messages (empty = OK)."""
        errors = []
        if self.tp > 1:
            if cfg.num_attention_heads % self.tp != 0:
                errors.append(
                    f"TP={self.tp} must divide num_attention_heads={cfg.num_attention_heads}"
                )
            if cfg.kv_heads % self.tp != 0:
                # GQA with KV heads not divisible by TP — need to handle
                errors.append(
                    f"TP={self.tp} must divide num_key_value_heads={cfg.kv_heads} "
                    f"(consider TP={self._nearest_divisor(cfg.kv_heads, self.tp)})"
                )
            if cfg.intermediate_size % self.tp != 0:
                errors.append(
                    f"TP={self.tp} must divide intermediate_size={cfg.intermediate_size}"
                )
            if cfg.vocab_size % self.tp != 0:
                errors.append(
                    f"TP={self.tp} must divide vocab_size={cfg.vocab_size}"
                )
        if self.pp > 1:
            if cfg.num_hidden_layers % self.pp != 0:
                errors.append(
                    f"PP={self.pp} does not evenly divide num_hidden_layers={cfg.num_hidden_layers} "
                    f"(will be uneven: some stages get {cfg.num_hidden_layers // self.pp + 1} layers, "
                    f"others get {cfg.num_hidden_layers // self.pp})"
                )
        if self.ep > 1:
            if not cfg.is_moe:
                errors.append(f"EP={self.ep} requires a MoE model, but this model is dense")
            elif cfg.num_experts % self.ep != 0:
                errors.append(
                    f"EP={self.ep} must divide num_experts={cfg.num_experts}"
                )
        if self.sp and self.tp == 1:
            errors.append("Sequence Parallel (SP) requires TP > 1")
        if self.cp > 1 and self.tp == 1:
            errors.append("Context Parallel (CP) typically requires TP > 1 for ring attention")
        return errors

    @staticmethod
    def _nearest_divisor(value: int, target: int) -> int:
        for d in range(min(target, value), 0, -1):
            if value % d == 0:
                return d
        return 1

    def _skip_lm_head(self) -> bool:
        """Whether LM head is tied and skipped."""
        return False  # resolved by config_parser

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
    pp_stage: int = -1,
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

    # ── Pipeline Parallel: check if kernel is on this stage ──
    on_this = True
    stage = -1
    if pc.pp > 1 and kernel.layer_idx is not None:
        layers_per_stage = cfg.num_hidden_layers // pc.pp
        extra = cfg.num_hidden_layers % pc.pp
        # Distribute extra layers to first stages
        cumulative = 0
        for s in range(pc.pp):
            stage_layers = layers_per_stage + (1 if s < extra else 0)
            if cumulative <= kernel.layer_idx < cumulative + stage_layers:
                stage = s
                break
            cumulative += stage_layers
        # For a specific stage, only that stage has the kernel
        # We show ALL kernels but mark which stage they belong to
        on_this = True  # we show all, mark stage
        notes_parts.append(f"PP stage {stage}/{pc.pp}")
        shard_applied.append(f"PP: layer→stage {stage}")

    # ── Tensor Parallel ──
    if pc.tp > 1:
        if kernel.shard_type == "column":
            # Column parallel: split output dimension (dim=1)
            if len(w) >= 2 and w[1] % pc.tp == 0:
                w[1] = w[1] // pc.tp
                if len(a) >= 1:
                    a[-1] = a[-1] // pc.tp
                shard_applied.append(f"TP: column split dim=1 → {w[1]}")
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

    # ── Expert Parallel ──
    if pc.ep > 1 and kernel.shard_type == "expert" and cfg.is_moe:
        experts_per_gpu = cfg.num_experts // pc.ep
        if "gate" in kernel.name or "up" in kernel.name:
            notes_parts.append(f"EP: {experts_per_gpu} experts/GPU (col parallel within EP)")
            if pc.tp > 1:
                # Within EP, can also apply TP
                if len(w) >= 2 and w[1] % pc.tp == 0:
                    w[1] = w[1] // pc.tp
                    shard_applied.append(f"TP within EP: col split → {w[1]}")
        elif "down" in kernel.name:
            notes_parts.append(f"EP: {experts_per_gpu} experts/GPU (row parallel within EP)")
            if pc.tp > 1 and len(w) >= 2 and w[0] % pc.tp == 0:
                w[0] = w[0] // pc.tp
                shard_applied.append(f"TP within EP: row split → {w[0]}")
        comm_fwd = "all-to-all (dispatch tokens→experts)" if pc.ep > 1 else comm_fwd
        comm_bwd = "all-to-all (gather expert outputs)" if pc.ep > 1 else comm_bwd

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

    # Compute memory
    w_numel = 1
    for d in w:
        w_numel *= max(d, 1)
    orig_numel = 1
    for d in orig_w:
        orig_numel *= max(d, 1)

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


def analyze_all_kernels(
    cfg: ModelConfig,
    pc: ParallelConfig,
    kernels: list[KernelSpec],
) -> list[ShardedKernel]:
    """Compute sharding for all kernels."""
    results = []
    for k in kernels:
        # Determine PP stage for this kernel
        pp_stage = -1
        if pc.pp > 1 and k.layer_idx is not None:
            layers_per_stage = cfg.num_hidden_layers // pc.pp
            extra = cfg.num_hidden_layers % pc.pp
            cumulative = 0
            for s in range(pc.pp):
                stage_layers = layers_per_stage + (1 if s < extra else 0)
                if cumulative <= k.layer_idx < cumulative + stage_layers:
                    pp_stage = s
                    break
                cumulative += stage_layers

        result = compute_sharding(k, cfg, pc, pp_stage=pp_stage)
        results.append(result)
    return results
