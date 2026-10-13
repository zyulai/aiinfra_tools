"""
Parse HuggingFace model config.json and extract all architecture parameters
relevant to parallelism analysis.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Normalized model configuration extracted from HF config.json."""
    model_type: str = ""
    architectures: list = field(default_factory=list)

    # Core dimensions
    hidden_size: int = 0
    num_hidden_layers: int = 0
    vocab_size: int = 0
    max_position_embeddings: int = 0

    # Attention
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    attention_bias: bool = False

    # MLP
    intermediate_size: int = 0
    hidden_act: str = "silu"
    mlp_bias: bool = False

    # MoE
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0

    # Norm
    rms_norm_eps: float = 1e-6
    norm_type: str = "rms"  # rms, layernorm

    # Tying
    tie_word_embeddings: bool = False

    # Extra
    sliding_window: Optional[int] = None
    rope_theta: float = 10000.0
    original_max_position_embeddings: int = 0

    # Raw config for reference
    raw: dict = field(default_factory=dict)

    @property
    def kv_heads(self) -> int:
        return self.num_key_value_heads or self.num_attention_heads

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    @property
    def is_gated_mlp(self) -> bool:
        return self.hidden_act in ("silu", "gelu", "swiglu") and self.intermediate_size > 0

    @property
    def effective_head_dim(self) -> int:
        if self.head_dim > 0:
            return self.head_dim
        if self.num_attention_heads > 0:
            return self.hidden_size // self.num_attention_heads
        return 0

    @property
    def qkv_dim(self) -> int:
        """Total dimension for Q, K, V combined."""
        hd = self.effective_head_dim
        return (self.num_attention_heads + 2 * self.kv_heads) * hd

    def summary(self) -> str:
        lines = [
            f"Model type:        {self.model_type}",
            f"Architectures:     {self.architectures}",
            f"Hidden size:       {self.hidden_size}",
            f"Layers:            {self.num_hidden_layers}",
            f"Vocab size:        {self.vocab_size}",
            f"Attn heads:        {self.num_attention_heads}",
            f"KV heads:          {self.kv_heads}",
            f"Head dim:          {self.effective_head_dim}",
            f"Intermediate size: {self.intermediate_size}",
            f"MLP activation:    {self.hidden_act} (gated={self.is_gated_mlp})",
            f"MoE:               {self.is_moe}",
        ]
        if self.is_moe:
            lines.append(f"  Num experts:       {self.num_experts}")
            lines.append(f"  Experts per tok:   {self.num_experts_per_tok}")
            lines.append(f"  MoE inter size:    {self.moe_intermediate_size or self.intermediate_size}")
        lines.append(f"Tie embeddings:    {self.tie_word_embeddings}")
        return "\n".join(lines)


def parse_config(config_path: str) -> ModelConfig:
    """Parse a HuggingFace config.json file or directory containing one."""
    if os.path.isdir(config_path):
        config_path = os.path.join(config_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        raw = json.load(f)

    cfg = ModelConfig(raw=raw)
    cfg.model_type = raw.get("model_type", "")
    cfg.architectures = raw.get("architectures", [])

    cfg.hidden_size = raw.get("hidden_size", 0)
    cfg.num_hidden_layers = raw.get("num_hidden_layers", 0)
    cfg.vocab_size = raw.get("vocab_size", 0)
    cfg.max_position_embeddings = raw.get("max_position_embeddings", 0)

    cfg.num_attention_heads = raw.get("num_attention_heads", 0)
    cfg.num_key_value_heads = raw.get("num_key_value_heads", cfg.num_attention_heads)
    cfg.head_dim = raw.get("head_dim", 0)
    cfg.attention_bias = raw.get("attention_bias", False)

    cfg.intermediate_size = raw.get("intermediate_size", 0)
    cfg.hidden_act = raw.get("hidden_act", "silu")
    cfg.mlp_bias = raw.get("mlp_bias", False)

    # MoE — handle various config schemas
    cfg.num_experts = raw.get("num_experts", 0)
    cfg.num_experts_per_tok = raw.get("num_experts_per_tok", raw.get("num_activated_experts", 0))
    cfg.moe_intermediate_size = raw.get("moe_intermediate_size", 0)

    # Also check for decoder block MoE config (some models nest this)
    if not cfg.num_experts:
        for key in ("num_experts", "n_experts"):
            if key in raw:
                cfg.num_experts = raw[key]

    cfg.rms_norm_eps = raw.get("rms_norm_eps", raw.get("layer_norm_eps", 1e-6))
    cfg.norm_type = "rms" if "rms_norm_eps" in raw or "rms_norm_eps" in str(raw) else "layernorm"

    cfg.tie_word_embeddings = raw.get("tie_word_embeddings", False)
    cfg.sliding_window = raw.get("sliding_window")
    cfg.rope_theta = raw.get("rope_theta", 10000.0)
    cfg.original_max_position_embeddings = raw.get("original_max_position_embeddings", 0)

    # Validate
    if cfg.hidden_size == 0:
        raise ValueError("hidden_size not found in config")
    if cfg.num_attention_heads == 0:
        raise ValueError("num_attention_heads not found in config")

    return cfg
