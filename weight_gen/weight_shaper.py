"""
Weight shape mapping — generate the complete weight name → shape mapping
for common HF transformer architectures, with optional parallel sharding.
"""
from dataclasses import dataclass, field
from typing import Optional
from parallel_tool.config_parser import ModelConfig
from parallel_tool.parallel_engine import ParallelConfig


@dataclass
class WeightEntry:
    """A single weight tensor in the model."""
    name: str           # HF state_dict key, e.g. "model.layers.0.self_attn.q_proj.weight"
    shape: tuple        # weight shape
    shard_type: str    # none, column, row, vocab_col, vocab_row, head_q, head_kv, expert, replicated
    dtype: str = "bfloat16"
    numel: int = 0     # computed from shape

    def __post_init__(self):
        self.numel = 1
        for d in self.shape:
            self.numel *= d


def build_weight_map(cfg: ModelConfig, pc: ParallelConfig) -> list[WeightEntry]:
    """
    Build the complete list of weight entries for a model.
    If TP > 1, shapes are pre-sharded for a single TP rank.
    """
    h = cfg.hidden_size
    hd = cfg.effective_head_dim
    n_heads = cfg.num_attention_heads
    n_kv = cfg.kv_heads
    inter = cfg.intermediate_size
    vocab = cfg.vocab_size
    tp = pc.tp
    sp = pc.sp

    entries: list[WeightEntry] = []

    # ── Embedding ──
    embed_shape = (vocab // tp, h) if tp > 1 else (vocab, h)
    entries.append(WeightEntry(
        name="model.embed_tokens.weight",
        shape=embed_shape,
        shard_type="vocab_col" if tp > 1 else "none",
    ))

    # ── Per-layer weights ──
    for i in range(cfg.num_hidden_layers):
        prefix = f"model.layers.{i}"

        # Input layernorm (replicated)
        entries.append(WeightEntry(
            name=f"{prefix}.input_layernorm.weight",
            shape=(h,),
            shard_type="replicated",
        ))

        # Q proj — column parallel by heads
        q_out = (n_heads // tp) * hd if tp > 1 else n_heads * hd
        entries.append(WeightEntry(
            name=f"{prefix}.self_attn.q_proj.weight",
            shape=(q_out, h),
            shard_type="head_q" if tp > 1 else "none",
        ))
        # Q proj bias if attention_bias
        if cfg.attention_bias:
            entries.append(WeightEntry(
                name=f"{prefix}.self_attn.q_proj.bias",
                shape=(q_out,),
                shard_type="head_q" if tp > 1 else "none",
            ))

        # K proj — column parallel by KV heads
        k_out = (n_kv // tp) * hd if tp > 1 else n_kv * hd
        entries.append(WeightEntry(
            name=f"{prefix}.self_attn.k_proj.weight",
            shape=(k_out, h),
            shard_type="head_kv" if tp > 1 else "none",
        ))
        if cfg.attention_bias:
            entries.append(WeightEntry(
                name=f"{prefix}.self_attn.k_proj.bias",
                shape=(k_out,),
                shard_type="head_kv" if tp > 1 else "none",
            ))

        # V proj — column parallel by KV heads
        v_out = (n_kv // tp) * hd if tp > 1 else n_kv * hd
        entries.append(WeightEntry(
            name=f"{prefix}.self_attn.v_proj.weight",
            shape=(v_out, h),
            shard_type="head_kv" if tp > 1 else "none",
        ))
        if cfg.attention_bias:
            entries.append(WeightEntry(
                name=f"{prefix}.self_attn.v_proj.bias",
                shape=(v_out,),
                shard_type="head_kv" if tp > 1 else "none",
            ))

        # O proj — row parallel (input dim sharded)
        o_in = (n_heads // tp) * hd if tp > 1 else n_heads * hd
        entries.append(WeightEntry(
            name=f"{prefix}.self_attn.o_proj.weight",
            shape=(h, o_in),
            shard_type="row" if tp > 1 else "none",
        ))
        if cfg.attention_bias:
            entries.append(WeightEntry(
                name=f"{prefix}.self_attn.o_proj.bias",
                shape=(h,),
                shard_type="replicated",
            ))

        # Post-attention layernorm (replicated)
        entries.append(WeightEntry(
            name=f"{prefix}.post_attention_layernorm.weight",
            shape=(h,),
            shard_type="replicated",
        ))

        if cfg.is_moe:
            moe_inter = cfg.moe_intermediate_size or inter
            # Router (replicated)
            entries.append(WeightEntry(
                name=f"{prefix}.block_sparse_moe.gate.weight",
                shape=(cfg.num_experts, h),
                shard_type="replicated",
            ))
            # Expert weights — if EP, each rank holds num_experts/ep experts
            # If TP also applied, within each expert the FFN is column/row parallel
            ep = pc.ep if pc.ep > 1 else 1
            experts_per_rank = cfg.num_experts // ep
            # If TP within EP, the intermediate is also split
            tp_inter = moe_inter // tp if tp > 1 else moe_inter

            for e in range(experts_per_rank):
                entries.append(WeightEntry(
                    name=f"{prefix}.mlp.experts.{e}.gate_proj.weight",
                    shape=(tp_inter, h),
                    shard_type="expert" if ep > 1 else ("column" if tp > 1 else "none"),
                ))
                entries.append(WeightEntry(
                    name=f"{prefix}.mlp.experts.{e}.up_proj.weight",
                    shape=(tp_inter, h),
                    shard_type="expert" if ep > 1 else ("column" if tp > 1 else "none"),
                ))
                entries.append(WeightEntry(
                    name=f"{prefix}.mlp.experts.{e}.down_proj.weight",
                    shape=(h, tp_inter),
                    shard_type="expert" if ep > 1 else ("row" if tp > 1 else "none"),
                ))
        else:
            # Gate proj — column parallel
            gate_out = inter // tp if tp > 1 else inter
            entries.append(WeightEntry(
                name=f"{prefix}.mlp.gate_proj.weight",
                shape=(gate_out, h),
                shard_type="column" if tp > 1 else "none",
            ))
            if cfg.mlp_bias:
                entries.append(WeightEntry(
                    name=f"{prefix}.mlp.gate_proj.bias",
                    shape=(gate_out,),
                    shard_type="column" if tp > 1 else "none",
                ))

            # Up proj — column parallel
            entries.append(WeightEntry(
                name=f"{prefix}.mlp.up_proj.weight",
                shape=(gate_out, h),
                shard_type="column" if tp > 1 else "none",
            ))
            if cfg.mlp_bias:
                entries.append(WeightEntry(
                    name=f"{prefix}.mlp.up_proj.bias",
                    shape=(gate_out,),
                    shard_type="column" if tp > 1 else "none",
                ))

            # Down proj — row parallel
            entries.append(WeightEntry(
                name=f"{prefix}.mlp.down_proj.weight",
                shape=(h, gate_out),
                shard_type="row" if tp > 1 else "none",
            ))
            if cfg.mlp_bias:
                entries.append(WeightEntry(
                    name=f"{prefix}.mlp.down_proj.bias",
                    shape=(h,),
                    shard_type="replicated",
                ))

    # ── Final norm (replicated) ──
    entries.append(WeightEntry(
        name="model.norm.weight",
        shape=(h,),
        shard_type="replicated",
    ))

    # ── LM Head ──
    if not cfg.tie_word_embeddings:
        lm_shape = (vocab // tp, h) if tp > 1 else (vocab, h)
        entries.append(WeightEntry(
            name="lm_head.weight",
            shape=lm_shape,
            shard_type="vocab_col" if tp > 1 else "none",
        ))

    return entries


def compute_total_bytes(entries: list[WeightEntry]) -> int:
    """Compute total bytes for all weights."""
    dtype_bytes = {"bfloat16": 2, "float16": 2, "float32": 4, "uint8": 1}
    total = 0
    for e in entries:
        total += e.numel * dtype_bytes.get(e.dtype, 2)
    return total


def compute_per_layer_bytes(entries: list[WeightEntry]) -> int:
    """Compute bytes for a single transformer layer."""
    dtype_bytes = {"bfloat16": 2, "float16": 2, "float32": 4, "uint8": 1}
    total = 0
    for e in entries:
        if "layers.0." in e.name:
            total += e.numel * dtype_bytes.get(e.dtype, 2)
    return total
