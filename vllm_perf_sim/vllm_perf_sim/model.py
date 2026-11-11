"""Model shape, weight/KV memory, and transformer FLOP accounting."""
from dataclasses import dataclass
from math import ceil

from parallel_tool.config_parser import ModelConfig, parse_config

DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4, "fp8": 1, "int8": 1, "int4": 0.5}


@dataclass
class ModelGeometry:
    config: ModelConfig
    weight_dtype: str = "bf16"
    kv_dtype: str = "bf16"
    tp: int = 1
    pp: int = 1
    ep: int = 1

    def __post_init__(self):
        if self.weight_dtype not in DTYPE_BYTES or self.kv_dtype not in DTYPE_BYTES:
            raise ValueError(f"Supported dtypes: {', '.join(DTYPE_BYTES)}")
        for name in ("tp", "pp", "ep"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name.upper()} must be >= 1")
        if self.pp > self.config.num_hidden_layers:
            raise ValueError("PP cannot exceed the number of layers")

    @property
    def layers_per_stage_max(self) -> int:
        return ceil(self.config.num_hidden_layers / self.pp)

    def parameter_count(self, layers: int | None = None) -> int:
        c = self.config
        layers = c.num_hidden_layers if layers is None else layers
        h, qkv = c.hidden_size, c.qkv_dim
        attention = h * qkv + qkv * h
        if c.is_moe:
            expert = 3 * h * c.moe_intermediate_size
            mlp = c.num_experts * expert + h * c.num_experts
        else:
            mlp = 3 * h * c.intermediate_size if c.is_gated_mlp else 2 * h * c.intermediate_size
        per_layer = attention + mlp + 2 * h
        embeddings = c.vocab_size * h
        lm_head = 0 if c.tie_word_embeddings else embeddings
        return layers * per_layer + embeddings + lm_head + h

    @property
    def weight_bytes_total(self) -> int:
        return int(self.parameter_count() * DTYPE_BYTES[self.weight_dtype])

    @property
    def weight_bytes_per_gpu(self) -> int:
        c = self.config
        layers = self.layers_per_stage_max
        h, qkv = c.hidden_size, c.qkv_dim
        attention = h * qkv + qkv * h
        if c.is_moe:
            experts = ceil(c.num_experts / self.ep)
            mlp = experts * 3 * h * c.moe_intermediate_size / self.tp
            mlp += h * c.num_experts / self.tp
        else:
            width = c.intermediate_size
            mlp = (3 if c.is_gated_mlp else 2) * h * width / self.tp
        layer_params = layers * (attention / self.tp + mlp + 2 * h)
        # Conservative: embedding and untied LM head are both placed on this stage.
        global_params = c.vocab_size * h / self.tp
        if not c.tie_word_embeddings:
            global_params *= 2
        return int((layer_params + global_params + h) * DTYPE_BYTES[self.weight_dtype])

    @property
    def kv_bytes_per_token_per_gpu(self) -> float:
        c = self.config
        layers = self.layers_per_stage_max
        heads = max(1, ceil(c.kv_heads / self.tp))
        return 2 * layers * heads * c.effective_head_dim * DTYPE_BYTES[self.kv_dtype]

    def decode_flops_per_token(self, active_tokens: int = 1) -> float:
        """Weight-matrix FLOPs for a token, excluding attention score reads."""
        return 2 * self.parameter_count(self.layers_per_stage_max) / self.tp * active_tokens

    def attention_score_flops(self, query_tokens: int, context_tokens: int) -> float:
        c = self.config
        return (4 * self.layers_per_stage_max * c.num_attention_heads
                * c.effective_head_dim * query_tokens * context_tokens / self.tp)

    def moe_active_flops(self, tokens: int) -> float:
        """Expert FLOPs actually routed, not total expert parameters."""
        c = self.config
        if not c.is_moe:
            return 0.0
        return (2 * 3 * c.hidden_size * c.moe_intermediate_size
                * c.num_experts_per_tok * self.layers_per_stage_max * tokens / self.tp)

    def decode_weight_flops(self, tokens: int) -> float:
        c = self.config
        h, qkv = c.hidden_size, c.qkv_dim
        layers = self.layers_per_stage_max
        attention = 2 * (h * qkv + qkv * h) * layers
        if c.is_moe:
            expert = (2 * 3 * h * c.moe_intermediate_size * c.num_experts_per_tok * layers)
            router = 2 * h * c.num_experts * layers
            mlp = expert + router
        else:
            mlp = 2 * (3 if c.is_gated_mlp else 2) * h * c.intermediate_size * layers
        embedding_head = 2 * c.vocab_size * h
        return tokens * (attention + mlp + embedding_head) / self.tp


def load_geometry(config_path: str, **kwargs) -> ModelGeometry:
    return ModelGeometry(parse_config(config_path), **kwargs)
