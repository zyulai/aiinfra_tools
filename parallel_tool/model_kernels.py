"""
Define the kernel/module list for different transformer architectures.
Each kernel has: name, weight shape, activation shape, shardable dimension.
"""
from dataclasses import dataclass, field
from typing import Optional
from .config_parser import ModelConfig


@dataclass
class KernelSpec:
    """Specification of a single model kernel/module."""
    name: str
    category: str            # embedding, attention, mlp, norm, moe, lm_head
    layer_idx: Optional[int] # None for global modules (embedding, lm_head)
    weight_shape: tuple      # (in_dim, out_dim) for linear; (vocab, hidden) for embedding
    weight_name: str         # HF weight name pattern
    shard_type: str          # none, column, row, vocab, head, expert, sequence, replicated
    shard_dim: int           # which dimension of weight_shape gets sharded (0 or 1)
    activation_in_shape: tuple   # per-token activation input shape
    activation_out_shape: tuple  # per-token activation output shape
    is_layer_local: bool = True  # True if exists per-layer, False if global
    comm_fwd: str = "none"       # communication needed in forward
    comm_bwd: str = "none"       # communication needed in backward
    dtype_bytes: int = 2         # bf16 default

    @property
    def weight_numel(self) -> int:
        n = 1
        for d in self.weight_shape:
            n *= d
        return n

    @property
    def weight_bytes(self) -> int:
        return self.weight_numel * self.dtype_bytes


def build_kernels(cfg: ModelConfig) -> list[KernelSpec]:
    """Build the complete list of kernels for a model architecture."""
    kernels: list[KernelSpec] = []
    h = cfg.hidden_size
    hd = cfg.effective_head_dim
    n_heads = cfg.num_attention_heads
    n_kv = cfg.kv_heads
    inter = cfg.intermediate_size
    vocab = cfg.vocab_size

    # ── Embedding ──
    kernels.append(KernelSpec(
        name="token_embedding",
        category="embedding",
        layer_idx=None,
        weight_shape=(vocab, h),
        weight_name="model.embed_tokens.weight",
        shard_type="vocab",
        shard_dim=0,
        activation_in_shape=(1,),          # token id
        activation_out_shape=(h,),          # hidden
        is_layer_local=False,
        comm_fwd="none",
        comm_bwd="all_reduce (if TP>1)",
    ))

    # ── Per-layer kernels ──
    for i in range(cfg.num_hidden_layers):
        prefix = f"model.layers.{i}"

        # Input norm — weight is always replicated (1D tensor [hidden_size])
        # SP affects activation (seq dim), not the weight itself
        kernels.append(KernelSpec(
            name=f"layer_{i}_input_norm",
            category="norm",
            layer_idx=i,
            weight_shape=(h,),
            weight_name=f"{prefix}.input_layernorm.weight",
            shard_type="replicated",
            shard_dim=-1,
            activation_in_shape=(h,),
            activation_out_shape=(h,),
            comm_fwd="none",
            comm_bwd="none",
        ))

        # Q projection
        q_out = n_heads * hd
        kernels.append(KernelSpec(
            name=f"layer_{i}_q_proj",
            category="attention",
            layer_idx=i,
            weight_shape=(h, q_out),
            weight_name=f"{prefix}.self_attn.q_proj.weight",
            shard_type="head",
            shard_dim=1,
            activation_in_shape=(h,),
            activation_out_shape=(q_out,),
            comm_fwd="none (column parallel)",
            comm_bwd="none",
        ))

        # K projection
        k_out = n_kv * hd
        kernels.append(KernelSpec(
            name=f"layer_{i}_k_proj",
            category="attention",
            layer_idx=i,
            weight_shape=(h, k_out),
            weight_name=f"{prefix}.self_attn.k_proj.weight",
            shard_type="head",
            shard_dim=1,
            activation_in_shape=(h,),
            activation_out_shape=(k_out,),
            comm_fwd="none (column parallel)",
            comm_bwd="none",
        ))

        # V projection
        v_out = n_kv * hd
        kernels.append(KernelSpec(
            name=f"layer_{i}_v_proj",
            category="attention",
            layer_idx=i,
            weight_shape=(h, v_out),
            weight_name=f"{prefix}.self_attn.v_proj.weight",
            shard_type="head",
            shard_dim=1,
            activation_in_shape=(h,),
            activation_out_shape=(v_out,),
            comm_fwd="none (column parallel)",
            comm_bwd="none",
        ))

        # Core attention (no weight, but has KV cache activation)
        kernels.append(KernelSpec(
            name=f"layer_{i}_core_attention",
            category="attention",
            layer_idx=i,
            weight_shape=(0,),               # no weight
            weight_name="(attention computation)",
            shard_type="head",
            shard_dim=-1,
            activation_in_shape=(n_heads * hd,),
            activation_out_shape=(n_heads * hd,),
            comm_fwd="none (local heads)" if n_heads > 0 else "none",
            comm_bwd="none",
        ))

        # O projection
        kernels.append(KernelSpec(
            name=f"layer_{i}_o_proj",
            category="attention",
            layer_idx=i,
            weight_shape=(q_out, h),
            weight_name=f"{prefix}.self_attn.o_proj.weight",
            shard_type="row",
            shard_dim=0,
            activation_in_shape=(q_out,),
            activation_out_shape=(h,),
            comm_fwd="all_reduce (row parallel)" if n_heads > 0 else "none",
            comm_bwd="none",
        ))

        if cfg.is_moe:
            # Router / gating — naming depends on architecture
            from weight_gen.weight_shaper import get_moe_naming
            router_pfx, expert_pfx, gate_name, up_name, down_name = get_moe_naming(cfg.model_type)
            moe_inter = cfg.moe_intermediate_size or inter
            kernels.append(KernelSpec(
                name=f"layer_{i}_moe_router",
                category="moe",
                layer_idx=i,
                weight_shape=(h, cfg.num_experts),
                weight_name=f"{prefix}.{router_pfx}.weight",
                shard_type="replicated",
                shard_dim=-1,
                activation_in_shape=(h,),
                activation_out_shape=(cfg.num_experts,),
                comm_fwd="none",
                comm_bwd="none",
            ))

            # Expert gate proj (per expert)
            kernels.append(KernelSpec(
                name=f"layer_{i}_moe_expert_gate",
                category="moe",
                layer_idx=i,
                weight_shape=(h, moe_inter),
                weight_name=f"{prefix}.{expert_pfx}.*.{gate_name}.weight",
                shard_type="expert",
                shard_dim=-1,
                activation_in_shape=(h,),
                activation_out_shape=(moe_inter,),
                comm_fwd="none",
                comm_bwd="all_reduce (if TP>1)",
            ))

            # Expert up proj (per expert)
            kernels.append(KernelSpec(
                name=f"layer_{i}_moe_expert_up",
                category="moe",
                layer_idx=i,
                weight_shape=(h, moe_inter),
                weight_name=f"{prefix}.{expert_pfx}.*.{up_name}.weight",
                shard_type="expert",
                shard_dim=-1,
                activation_in_shape=(h,),
                activation_out_shape=(moe_inter,),
                comm_fwd="none",
                comm_bwd="all_reduce (if TP>1)",
            ))

            # Expert down proj (per expert)
            kernels.append(KernelSpec(
                name=f"layer_{i}_moe_expert_down",
                category="moe",
                layer_idx=i,
                weight_shape=(moe_inter, h),
                weight_name=f"{prefix}.{expert_pfx}.*.{down_name}.weight",
                shard_type="expert",
                shard_dim=-1,
                activation_in_shape=(moe_inter,),
                activation_out_shape=(h,),
                comm_fwd="none (all-to-all if EP>1)",
                comm_bwd="all_reduce (if TP>1)",
            ))
        else:
            # Gate proj (SwiGLU / gated MLP)
            kernels.append(KernelSpec(
                name=f"layer_{i}_gate_proj",
                category="mlp",
                layer_idx=i,
                weight_shape=(h, inter),
                weight_name=f"{prefix}.mlp.gate_proj.weight",
                shard_type="column",
                shard_dim=1,
                activation_in_shape=(h,),
                activation_out_shape=(inter,),
                comm_fwd="none (column parallel)",
                comm_bwd="none",
            ))

            # Up proj
            kernels.append(KernelSpec(
                name=f"layer_{i}_up_proj",
                category="mlp",
                layer_idx=i,
                weight_shape=(h, inter),
                weight_name=f"{prefix}.mlp.up_proj.weight",
                shard_type="column",
                shard_dim=1,
                activation_in_shape=(h,),
                activation_out_shape=(inter,),
                comm_fwd="none (column parallel)",
                comm_bwd="none",
            ))

            # Down proj
            kernels.append(KernelSpec(
                name=f"layer_{i}_down_proj",
                category="mlp",
                layer_idx=i,
                weight_shape=(inter, h),
                weight_name=f"{prefix}.mlp.down_proj.weight",
                shard_type="row",
                shard_dim=0,
                activation_in_shape=(inter,),
                activation_out_shape=(h,),
                comm_fwd="all_reduce (row parallel)",
                comm_bwd="none",
            ))

        # Post-attention norm
        kernels.append(KernelSpec(
            name=f"layer_{i}_post_attn_norm",
            category="norm",
            layer_idx=i,
            weight_shape=(h,),
            weight_name=f"{prefix}.post_attention_layernorm.weight",
            shard_type="replicated",
            shard_dim=-1,
            activation_in_shape=(h,),
            activation_out_shape=(h,),
            comm_fwd="none",
            comm_bwd="none",
        ))

    # ── Final norm ──
    kernels.append(KernelSpec(
        name="final_norm",
        category="norm",
        layer_idx=None,
        weight_shape=(h,),
        weight_name="model.norm.weight",
        shard_type="replicated",
        shard_dim=-1,
        activation_in_shape=(h,),
        activation_out_shape=(h,),
        is_layer_local=False,
        comm_fwd="none",
        comm_bwd="none",
    ))

    # ── LM Head ──
    if not cfg.tie_word_embeddings:
        kernels.append(KernelSpec(
            name="lm_head",
            category="lm_head",
            layer_idx=None,
            weight_shape=(h, vocab),
            weight_name="lm_head.weight",
            shard_type="vocab",
            shard_dim=1,
            activation_in_shape=(h,),
            activation_out_shape=(vocab,),
            is_layer_local=False,
            comm_fwd="none (column parallel)",
            comm_bwd="all_reduce (if TP>1)",
        ))

    return kernels
