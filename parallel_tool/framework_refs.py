"""
Reference table — how each framework implements each parallel strategy.
Based on actual source code analysis of SGLang, vLLM, Megatron-LM.
"""

FRAMEWORK_REFS = {
    # ── Tensor Parallel ──
    "TP_embedding": {
        "sglang": "sglang/srt/layers/embedding.py — VocabParallelEmbedding (adapted from Megatron)",
        "vllm": "vllm/model_executor/layers/vocab_parallel_embedding.py — VocabParallelEmbedding",
        "megatron": "megatron/core/tensor_parallel/layers.py — VocabParallelEmbedding class",
        "pattern": "Vocab parallel: split vocab_size across TP ranks. Each rank holds [vocab/TP, hidden]. Mask + all_reduce.",
    },
    "TP_qkv": {
        "sglang": "sglang/srt/layers/attention/qkv_split.py — QKVParallelLinear (splits Q by heads, K/V by KV groups)",
        "vllm": "vllm/model_executor/layers/linear.py — QKVParallelLinear (fused QKV, column parallel)",
        "megatron": "megatron/core/tensor_parallel/layers.py — ColumnParallelLinear for Q/K/V",
        "pattern": "Column parallel: split Q heads [H/TP, d], K/V heads [KV/TP, d]. No comm in fwd.",
    },
    "TP_o_proj": {
        "sglang": "sglang/srt/layers/linear.py — RowParallelLinear for o_proj",
        "vllm": "vllm/model_executor/layers/linear.py — RowParallelLinear",
        "megatron": "megatron/core/tensor_parallel/layers.py — RowParallelLinear",
        "pattern": "Row parallel: split input dim [H/TP, d_out]. All_reduce in fwd (reduce_scatter with SP).",
    },
    "TP_mlp_gate_up": {
        "sglang": "sglang/srt/layers/linear.py — ColumnParallelLinear (fused gate+up via merged_linear)",
        "vllm": "vllm/model_executor/layers/linear.py — MergedColumnParallelLinear (gate+up)",
        "megatron": "megatron/core/tensor_parallel/layers.py — ColumnParallelLinear",
        "pattern": "Column parallel: split intermediate [inter/TP]. Fused gate+up: [2*inter/TP]. No comm in fwd.",
    },
    "TP_mlp_down": {
        "sglang": "sglang/srt/layers/linear.py — RowParallelLinear for down_proj",
        "vllm": "vllm/model_executor/layers/linear.py — RowParallelLinear",
        "megatron": "megatron/core/tensor_parallel/layers.py — RowParallelLinear",
        "pattern": "Row parallel: split [inter/TP, hidden]. All_reduce in fwd.",
    },
    "TP_lm_head": {
        "sglang": "sglang/srt/layers/logits_processor.py — parallel LM head + all_gather logits",
        "vllm": "vllm/model_executor/layers/logits_processor.py — LogitsProcessor with vocab parallel",
        "megatron": "megatron/core/tensor_parallel/cross_entropy.py — VocabParallelCrossEntropy",
        "pattern": "Column/vocab parallel: [hidden, vocab/TP]. No explicit all_reduce (fused cross-entropy).",
    },

    # ── Pipeline Parallel ──
    "PP_layer_split": {
        "sglang": "sglang/srt/managers/scheduler.py — layer_range for PP stages, send/recv activations",
        "vllm": "vllm/distributed/communication_op.py + pipeline_parallel — p2p send/recv between stages",
        "megatron": "megatron/core/pipeline_parallel/ — schedules (1F1B, interleaved), p2p comm",
        "pattern": "Split num_layers across PP stages. Activations passed via send/recv between adjacent stages.",
    },

    # ── Context Parallel ──
    "CP_ring_attention": {
        "sglang": "sglang/srt/layers/attention/cp.py — Ring attention with CP group send/recv KV",
        "vllm": "vllm/attention/backends/ — CP-aware attention backends (ring attention)",
        "megatron": "megatron/core/transformer/attention.py — CP attention with ring/cross-attention patterns",
        "pattern": "Split seq_len across CP ranks. Ring send/recv of KV blocks. Each rank computes local attention + gathers remote KV.",
    },

    # ── Expert Parallel ──
    "EP_dispatch": {
        "sglang": "sglang/srt/layers/moe/ — token dispatch via all-to-all, expert computation, combine",
        "vllm": "vllm/model_executor/layers/fused_moe/ — fused MoE with EP, all-to-all dispatch",
        "megatron": "megatron/core/transformer/moe/ — MoE layers with expert parallel all-to-all",
        "pattern": "Split num_experts across EP ranks. All-to-all dispatch: tokens→expert ranks, compute, all-to-all combine.",
    },

    # ── Sequence Parallel ──
    "SP_regions": {
        "sglang": "Not directly implemented (uses implicit SP via attention masking in some backends)",
        "vllm": "vllm/distributed/communication_op.py — reduce_scatter / all_gather for SP",
        "megatron": "megatron/core/tensor_parallel/mappings.py — scatter_to / gather_from sequence parallel region",
        "pattern": "Non-attention regions (norm, dropout) use [seq/TP, hidden]. All-gather before TP column-parallel, reduce-scatter after TP row-parallel.",
    },

    # ── Data Parallel ──
    "DP_grad_sync": {
        "sglang": "sglang/srt/managers/ — DP gradient all_reduce after backward",
        "vllm": "vllm/distributed/parallel_state.py — DP groups, gradient sync via all_reduce",
        "megatron": "megatron/core/parallel_state.py — DP groups, DDP gradient all-reduce",
        "pattern": "Full model replica on each DP rank. Gradients all-reduced after backward. No weight sharding.",
    },
}


def active_framework_refs(pc) -> list[tuple[str, dict]]:
    """Select references once for both Rich and plain-text entry points."""
    keys = []
    if pc.tp > 1:
        keys.extend(("TP_embedding", "TP_qkv", "TP_o_proj", "TP_mlp_gate_up", "TP_mlp_down", "TP_lm_head"))
    for enabled, key in ((pc.pp > 1, "PP_layer_split"), (pc.cp > 1, "CP_ring_attention"),
                         (pc.ep > 1, "EP_dispatch"), (pc.sp, "SP_regions"), (pc.dp > 1, "DP_grad_sync")):
        if enabled:
            keys.append(key)
    return [(key, get_framework_ref(key)) for key in keys]


def get_framework_ref(strategy: str) -> dict:
    """Get framework reference for a parallel strategy key."""
    return FRAMEWORK_REFS.get(strategy, {})
