#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KV Cache 大小计算器(按 vLLM 最新 main 分支源码逻辑,commit a55937f, 2026-10-25)

用法:
  python kv_calculator.py deepseek-ai/DeepSeek-V3                 # HF 模型 ID
  python kv_calculator.py ./config.json                           # 本地 config
  python kv_calculator.py meta-llama/Llama-3.1-70B-Instruct --tp 4 --kv-dtype fp8
  python kv_calculator.py Qwen/Qwen3-Next-80B-A3B-Instruct --seq 262144 --batch 4
  python kv_calculator.py deepseek-ai/DeepSeek-V3 --budget-gb 40 --seq 32768

计算口径与 vLLM V1 引擎一致:
  - 每层每 token 字节 = (head_size + head_size_v) * num_kv_heads * dtype字节
    [vllm/v1/kv_cache_interface.py AttentionSpec.state_content_size_bytes]
  - MLA 层: num_kv_heads=1, head_size=kv_lora_rank+qk_rope_head_dim, 无独立 V
    [vllm/model_executor/layers/attention/mla_attention.py:480]
  - MLA 的 latent 缓存跨 TP rank 复制(不除 TP);GQA 的 KV head 按 TP 切分
    [vllm/v1/kv_cache_interface.py AttentionSpec.max_tp_shards / kv_tp_replicas]
  - 滑动窗口层: 每 token 大小不变,但缓存 token 数被窗口上限截断
    [SlidingWindowSpec.max_admission_blocks_per_request]
  - 稀疏 MLA(DeepSeek-V4/GLM-5-next): tokens_per_state = compress_ratio/kpool,
    即每 ratio 个 token 才存一个 latent
  - 混合线性注意力层(GDN/KDA/Mamba): 无逐 token KV,改为每请求固定状态缓存,
    shape 公式来自 vllm/model_executor/layers/mamba/mamba_utils.py
  - 跨层 KV 共享(Gemma-3n/4 CLA): 被共享层不占显存
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 基础常量
# ---------------------------------------------------------------------------

DTYPE_BYTES = {  # kv-cache dtype -> 每元素字节数(数据部分)
    "auto": None, "float16": 2, "bfloat16": 2, "float32": 4,
    "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1,
    "int8": 1, "int8_per_token_head": 1, "fp8_per_token_head": 1,
    "int4_per_token_head": 0.5,           # 打包 2×int4/字节(scale 另计,很小)
    "nvfp4": 0.5625,                       # 4bit + 1/16 的 fp8 block scale
    "nvfp4_4over6": 0.5625,
    "turboquant_k8v4": 1.5,                # K=8bit, V=4bit (对 K+V 合计)
    "turboquant_4bit_nc": 0.5,
    "turboquant_k3v4_nc": 0.875,           # K=3bit, V=4bit
    "turboquant_3bit_nc": 0.375,
    "ultraquant_4bit": 0.5625,             # 4bit + UE8M0 组 scale
    # fp8_inc / fp8_ds_mla / nvfp4_ds_mla 为 MLA 专用布局,按 1/0.5 近似
    "fp8_inc": 1, "fp8_ds_mla": 1, "nvfp4_ds_mla": 0.5625,
}
# 注意: turboquant 系是 "K 用 X bit、V 用 Y bit" 的混合精度,上表已折算成
# (K+V)/2 的等效每元素字节数,直接乘 (head_size+head_size_v) 即可。

# 命令行常见缩写
DTYPE_ALIASES = {
    "bf16": "bfloat16", "fp16": "float16", "half": "float16",
    "fp32": "float32", "float": "float32", "e4m3": "fp8_e4m3",
    "e5m2": "fp8_e5m2", "uint8": "int8",
}

# 打包布局:这些 kv dtype 每 token 的内容不是 head_size 的简单倍数,
# vLLM 直接给出每 (head slot, state) 的字节数
# [vllm/v1/kv_cache_interface.py AttentionSpec.state_content_bytes]
#   mla_attention.py:1362-1367  -> 通用 MLA: fp8_ds_mla=656, nvfp4_ds_mla=352
#   models/deepseek_v4/attention.py:866-879 -> DSv4: fp8_ds_mla=584
MLA_PACKED_BYTES = {
    "fp8_ds_mla": 656, "nvfp4_ds_mla": 352,
    "turboquant_k8v4": None, "turboquant_4bit_nc": None,
    "turboquant_k3v4_nc": None, "turboquant_3bit_nc": None,
}
DSV4_PACKED_BYTES = {"fp8_ds_mla": 584, "nvfp4_ds_mla": 352}
# DSA indexer 缓存:uint8 存 fp8 值 + 每 128 元素 4 字节 fp32 scale
# [deepseek_v2.py:731-736] head_dim = index_head_dim + index_head_dim//128*4
INDEXER_QUANT_BLOCK = 128

# vLLM transformers_utils/model_arch_config_convertor.py:294-342 的 MLA 家族
MLA_MODEL_TYPES = {
    "axk1", "deepseek_v2", "deepseek_v3", "deepseek_v32", "deepseek_v4",
    "deepseek_v41", "dots3_note", "deepseek_mtp", "k3_dspark", "glm5_next",
    "glm5_next_text", "glm_moe_dsa", "glm4_moe_lite", "glm4_moe_lite_mtp",
    "kimi_k2", "kimi_linear", "longcat_flash", "pangu_ultra_moe",
    "pangu_ultra_moe_mtp", "bailing_hybrid", "bailing_hybrid_mtp",
    "bailing_hybrid_v3_mtp", "bailing_moe_v3", "bailing_moe_v3_mtp",
    "hy_v4", "hy_v4_mtp", "sarvam",
}
# bailing_moe_v3 / sarvam 不在 vLLM 的 is_deepseek_mla 列表里,但其模型实现
# (BailingMoeV3MLAAttention / SarvamMLAAttention)同样构建 MLAAttention。

KV_HEAD_FIELDS = [  # CONFIG_FIELD_ALIASES["total_num_kv_heads"]
    "num_key_value_heads", "num_kv_heads", "n_head_kv", "multi_query_group_num",
    "num_attention_groups",  # Step3p5
]

TEXT_CONFIG_KEYS = ["text_config", "language_config"]  # 多模态嵌套取文本主干

# 纯 encoder / embedding / 视觉塔:无自回归 KV cache
ENCODER_ONLY_TYPES = {
    "bert", "bert_with_rope", "roberta", "modernbert", "clip", "siglip",
    "siglip2navit", "intern_vit", "aimv2", "blip", "vision_tower",
    "colbert", "colpali", "colqwen", "colqwen3_5", "colultravbert",
    "jina_bert", "teratorch", "idefics2_vision_model", "timm_wrapper",
}

# 滑动窗口 "+1" 约定:Cohere 系窗口是"包含式半径",vLLM 会 +1 对齐训练约定
# (commandr.py:237-244, cohere2_moe.py:250-256, modernbert.py:99-102)
WINDOW_PLUS_ONE_TYPES = {"command_r", "cohere2", "cohere_compass", "modernbert"}


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


# ---------------------------------------------------------------------------
# 层规格
# ---------------------------------------------------------------------------

@dataclass
class LayerSpec:
    """一层的 KV 缓存规格(对应 vLLM 的 KVCacheSpec)"""
    kind: str = "full"          # full | swa | mla | sparse_mla | linear | mamba | cross | shared | encoder_only
    num_kv_heads: int = 0       # TP 切分前
    head_size: int = 0          # K 维度(MLA: kv_lora_rank+qk_rope; head_size_v=0)
    head_size_v: int = None     # V 维度(None == head_size;mimo_v2 等不同)
    window: int = None          # 滑动窗口 token 数
    tokens_per_state: int = 1   # >1: 稀疏压缩,每 N token 存一个状态
    indexer_head: int = 0       # DSA/dspark indexer 的 K 维度(每 token 单向量)
    indexer_kpool: int = 1      # indexer 的 kpool 压缩(GLM-5-next)
    tp_replicated: bool = False # MLA: latent 跨 TP 复制,不除 TP
    note: str = ""


@dataclass
class StateSpec:
    """线性注意力/Mamba 层的固定状态缓存(每请求一次,与长度无关)

    存 TP 之前的维度,dtype 分开记录:vLLM 的 mamba_cache_dtype 与
    mamba_ssm_cache_dtype 是两个独立开关,且不同层的默认值不同
    [mamba_utils.MambaStateDtypeCalculator]
      - GDN / Mamba1 / Mamba2: auto 时 conv 与 temporal 都用模型 dtype
      - KDA: auto 时 recurrent 恒定 fp32
    """
    kind: str = "gdn"           # gdn | kda | mamba1 | mamba2 | short_conv | linear
    conv_dim: int = 0           # TP 前的 conv 通道数
    conv_len: int = 0           # conv 状态长度(kernel-1)
    conv_elem: int = 2
    temporal_numel: int = 0     # TP 前的状态元素数
    temporal_elem: int = 2
    # Mamba2: n_groups < tp 时 conv 通道被扩到 tp(见 extra_groups_for_head_shards)
    groups: int = 0
    state_size: int = 0
    note: str = ""

    def _conv_dim_at_tp(self, tp: int) -> int:
        dim = self.conv_dim
        if self.groups and tp > self.groups:
            dim += 2 * (tp - self.groups) * self.state_size
        return dim

    @property
    def conv_bytes(self) -> int:
        return self.conv_dim * self.conv_len * self.conv_elem

    @property
    def temporal_bytes(self) -> int:
        return self.temporal_numel * self.temporal_elem

    @property
    def bytes_total(self) -> int:
        return self.conv_bytes + self.temporal_bytes

    def bytes_at_tp(self, tp: int) -> int:
        tp = max(1, tp)
        return max(1, self._conv_dim_at_tp(tp) // tp * self.conv_len * self.conv_elem
                   + self.temporal_numel // tp * self.temporal_elem)

    def describe(self) -> str:
        return (f"{self.note} → conv {self.conv_bytes:,} B + 状态 "
                f"{self.temporal_bytes:,} B (TP=1)")


@dataclass
class Geometry:
    """整个模型解析结果"""
    model_type: str = ""
    family: str = ""            # 描述性家族名
    layers: list = field(default_factory=list)   # list[LayerSpec]
    states: list = field(default_factory=list)   # list[StateSpec]
    cross_layers: int = 0       # enc-dec cross-attn 层数
    cross_kv_heads: int = 0
    cross_head_size: int = 0
    default_encoder_len: int = 0  # whisper 1500 / T5 类默认
    max_len: int = 0
    dtype: str = "bfloat16"
    notes: list = field(default_factory=list)

    def kv_layers(self):
        return [l for l in self.layers if l.kind in ("full", "swa", "mla", "sparse_mla")]


# ---------------------------------------------------------------------------
# config 获取
# ---------------------------------------------------------------------------

def load_config(source: str) -> dict:
    """source: HF repo id 或本地 config.json 路径。受限仓库自动尝试公开镜像。"""
    if os.path.isfile(source):
        with open(source, encoding="utf-8") as f:
            return json.load(f)
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        raise SystemExit("需要 huggingface_hub 才能拉取 HF 配置:pip install huggingface_hub")
    candidates = [source]
    name = source.split("/")[-1]
    for repo in ("unsloth/Meta-" + name, "unsloth/" + name,
                 "NousResearch/Meta-" + name, "NousResearch/" + name,
                 "mlx-community/" + name):
        if repo not in candidates:
            candidates.append(repo)
    last_err = None
    for repo in candidates:
        try:
            path = hf_hub_download(repo, "config.json")
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (GatedRepoError, RepositoryNotFoundError) as e:
            last_err = e
            continue
    raise SystemExit(f"无法获取 config.json(尝试过 {candidates}):{last_err}")


def resolve_text_config(cfg: dict) -> tuple[dict, dict]:
    """返回 (text_config, root_config)。取嵌套 text_config/language_config。"""
    root = cfg
    text = cfg
    for key in TEXT_CONFIG_KEYS:
        sub = cfg.get(key)
        if isinstance(sub, dict) and ("num_hidden_layers" in sub or "num_layers" in sub):
            text = sub
            break
    # Llama4 / GLM4.5-V 等还有更深层嵌套
    for key in TEXT_CONFIG_KEYS:
        sub = text.get(key) if isinstance(text, dict) else None
        if isinstance(sub, dict) and ("num_hidden_layers" in sub or "num_layers" in sub):
            text = sub
            break
    return text, root


def g(cfg: dict, *names, default=None):
    for n in names:
        v = cfg.get(n)
        if v is not None:
            return v
    return default


# ---------------------------------------------------------------------------
# 核心:把 config.json 解析成 Geometry(镜像 vLLM 各模型实现)
# ---------------------------------------------------------------------------

def _num_kv_heads(text: dict, default_heads: int) -> int:
    v = g(text, *KV_HEAD_FIELDS)
    return int(v) if v is not None else int(default_heads)


def _head_size(text: dict, num_heads: int, hidden: int) -> int:
    hd = g(text, "head_dim", "hidden_size_per_head", "attention_head_dim")  # zamba2
    if hd:
        return int(hd)
    if num_heads and hidden:
        return hidden // num_heads
    return 0


def _layer_types(text: dict, num_layers: int) -> list:
    lt = text.get("layer_types") or text.get("layers_block_type")
    if isinstance(lt, list) and lt:
        return lt
    # Nemotron-H: hybrid_override_pattern 用单字符标记层类型
    # M=Mamba, *=attention, -=MLP, E=MoE [nemotron_h.py:549-554]
    pattern = text.get("hybrid_override_pattern")
    if isinstance(pattern, str) and pattern:
        return ["full_attention" if c == "*"
                else ("mamba" if c == "M" else "mlp") for c in pattern]
    # Jamba: layers_block_type 是 HF 的派生 property(config.json 里没有),
    # "attention" 当 i % attn_layer_period == attn_layer_offset,其余 mamba
    # [transformers JambaConfig.layers_block_type]
    if text.get("attn_layer_period"):
        period = int(text["attn_layer_period"])
        offset = int(text.get("attn_layer_offset", 0))
        return ["full_attention" if i % period == offset else "mamba"
                for i in range(num_layers)]
    # Qwen3-Next 等:full_attention_interval=N → 每 N 层第 N 层 full,其余 linear
    # (transformers Qwen3NextConfig: full 当 (i+1)%interval==0)
    interval = g(text, "full_attention_interval", "attention_interval", default=None)
    if interval:
        return ["full_attention" if (i + 1) % int(interval) == 0 else "linear_attention"
                for i in range(num_layers)]
    # 旧 Gemma2/3 config:sliding_window_pattern=周期 P(如 6 = 5 滑动+1 全局,
    # 全局层在 i 满足 (i+1)%P==0;Gemma2 无该字段时默认 1:1 交错 P=2)
    # 与 transformers Gemma3TextConfig 推导一致(62层/p=6 → 52滑+10全)
    swp = text.get("sliding_window_pattern")
    mt = text.get("model_type", "")
    if swp is None and mt.startswith(("gemma2", "gemma3", "gemma3n", "gemma")):
        swp = 2
    if swp:
        return ["full_attention" if (i + 1) % int(swp) == 0 else "sliding_attention"
                for i in range(num_layers)]
    return []


def _is_mla(text: dict) -> bool:
    mt = text.get("model_type", "")
    if mt in MLA_MODEL_TYPES:
        if mt in ("deepseek_v4", "deepseek_v41", "glm5_next"):
            return text.get("kv_lora_rank") is not None or text.get("head_dim") is not None
        return text.get("kv_lora_rank") is not None
    # DeepSeek-V2 的 MHA 变体没有 kv_lora_rank → 走 GQA 分支
    return False


def _dsv4_compress_ratios(text: dict, num_layers: int) -> list:
    """vllm/transformers_utils/deepseek_v4_utils.py get_compress_ratios"""
    cr = text.get("compress_ratios")
    if isinstance(cr, list):
        return cr
    rates = text.get("compress_rates")
    if isinstance(rates, dict):
        lt = _layer_types(text, num_layers) or ["full_attention"] * num_layers
        ratios = [rates.get(t, 0) for t in lt]
        mtp = text.get("num_nextn_predict_layers", 0) or 0
        return ratios + [0] * mtp
    return []


def analyze(cfg: dict) -> Geometry:
    text, root = resolve_text_config(cfg)
    mt = text.get("model_type", root.get("model_type", ""))
    num_layers = int(g(text, "num_hidden_layers", "n_layer", "n_layers",
                       "num_layers", default=0))
    # longcat_flash: num_layers 是注意力子层维度,每个 decoder layer 有 2 个 MLA
    longcat_dual = (mt == "longcat_flash")
    if longcat_dual:
        num_layers = int(text.get("num_layers", 0))

    geo = Geometry(model_type=mt, max_len=int(g(
        text, "max_position_embeddings", "max_sequence_length", "model_max_length",
        "n_positions", "max_seq_len", "seq_length", default=0) or 0))
    geo.dtype = str(g(text, "torch_dtype", "dtype", default="bfloat16"))

    # ---- 纯 encoder / 视觉塔 ----
    if mt in ENCODER_ONLY_TYPES or (mt == "" and "num_hidden_layers" not in text):
        geo.family = f"{mt or 'unknown'}(encoder-only,无自回归 KV cache)"
        geo.notes.append("encoder-only 模型,prefill 一次前向,不保存 KV cache")
        return geo

    # ---- encoder-decoder(Whisper/T5/FunASR 等)----
    if mt in ("whisper", "funasr", "cohere_asr", "nemotron_parse", "parakeet",
              "voxtral", "ultravox"):
        return _analyze_encdec(geo, text, mt)

    # ---- MLA 家族 ----
    if _is_mla(text):
        return _analyze_mla(geo, text, mt, num_layers)

    # ---- 标准 GQA/MQA/混合(按 layer_types 或 sliding_window)----
    return _analyze_standard(geo, text, root, mt, num_layers)


def _analyze_encdec(geo, text, mt):
    geo.family = f"{mt}(encoder-decoder,cross-attention)"
    heads = int(g(text, "num_attention_heads", "decoder_attention_heads",
                  "n_heads", default=0))
    hidden = int(g(text, "hidden_size", "d_model", "embed_dim", default=0))
    head = _head_size(text, heads, hidden)
    # cohere_asr 从嵌套 transf_decoder 读
    td = text.get("transf_decoder", {}).get("config_dict", {}) if isinstance(text.get("transf_decoder"), dict) else {}
    if td:
        heads = int(td.get("num_attention_heads", heads))
        head = int(td.get("hidden_size", hidden)) // heads
    n_dec = int(g(text, "num_hidden_layers", "decoder_layers",
                  "num_decoder_layers", default=0))
    # Whisper: decoder 每层 self-attn + cross-attn;cross 缓存 encoder 输出
    geo.layers = [LayerSpec(kind="full", num_kv_heads=heads, head_size=head)
                  for _ in range(n_dec)]
    geo.cross_layers = n_dec
    geo.cross_kv_heads = heads
    geo.cross_head_size = head
    geo.default_encoder_len = int(g(text, "max_source_positions",
                                    "max_encoder_positions", default=0) or 0)
    geo.notes.append(f"decoder {n_dec} 层 self-attn 常规 KV;每层另有 1 个 "
                     f"cross-attn 缓存 encoder K/V(长度=音频/源文 token 数,"
                     f"config 默认 {geo.default_encoder_len or '未知,需手动指定'})")
    return geo


def _mla_latent_dims(text, mt):
    """返回 (kv_lora_rank, qk_rope_head_dim, head_dim_override)"""
    if mt in ("deepseek_v4", "deepseek_v41") and _dsv4_compress_ratios(text, 10**9):
        # 稀疏 MLA 变体直接用 head_dim 作为 latent 维度
        return None, None, int(text.get("head_dim", 0))
    kl = text.get("kv_lora_rank")
    qr = text.get("qk_rope_head_dim", 0) or 0
    return (int(kl), int(qr), None) if kl is not None else (None, None, int(text.get("head_dim", 0) or 0))


def _analyze_mla(geo, text, mt, num_layers):
    kv_lora, qk_rope, hd_override = _mla_latent_dims(text, mt)
    latent = hd_override if hd_override else (kv_lora + qk_rope)
    geo.family = f"{mt}(MLA:每层每 token 一个 {latent} 维 latent,无 K/V 分离,无 GQA)"

    ratios = _dsv4_compress_ratios(text, num_layers)
    lt = _layer_types(text, num_layers)
    kda_cfg = text.get("linear_attn_config") or {}
    kda_layers = set()
    if isinstance(kda_cfg, dict) and kda_cfg.get("kda_layers"):
        # kimi_linear.py: (layer_idx+1) in kda_layers → KDA 层
        kda_layers = {int(i) - 1 for i in kda_cfg["kda_layers"]}

    # indexer(DSA/dspark/GLM-5-next)
    # vLLM 仅在 is_v32 = hasattr(config, "index_topk") 时构建 Indexer
    # [deepseek_v2.py:1143] ;DSv4 只在 compress_ratio==4 的层建 indexer
    # [deepseek_v4/attention.py:289-302]
    idx_head = 0
    idx_kpool = 1
    if text.get("index_topk") is not None or text.get("index_head_dim"):
        idx_head = int(g(text, "index_head_dim", default=128) or 128)
    if mt in ("glm5_next", "glm5_next_text"):
        idx_head = int(g(text, "index_head_dim", default=128) or 128)
        idx_kpool = int(g(text, "index_kpool", default=1) or 1)

    # 部分层跳过 top-k/indexer,不建 indexer 缓存
    # [deepseek_v2.py:1145-1165]
    idx_freq = int(g(text, "index_topk_freq", default=1) or 1)
    idx_pattern = text.get("index_topk_pattern")
    idx_skip_offset = int(g(text, "index_skip_topk_offset", default=2) or 2)

    def _skip_indexer(layer_id: int) -> bool:
        if isinstance(idx_pattern, list) and 0 <= layer_id < len(idx_pattern):
            return idx_pattern[layer_id] == "S"
        if idx_pattern is None:
            return max(layer_id - idx_skip_offset + 1, 0) % idx_freq != 0
        return False

    dsv4 = mt in ("deepseek_v4", "deepseek_v41")
    # DSv4: 每层另有独立的 SWA latent 缓存(head_dim, 滑动窗口),与压缩缓存并存
    # [deepseek_v4/attention.py:345 DeepseekV4SWACache];compress_ratio<=1 的层
    # 只有 SWA 缓存,没有压缩缓存 [attention.py:856-859]
    dsv4_window = int(g(text, "sliding_window", "window_size", default=0) or 0)

    for i in range(num_layers):
        if i in kda_layers:
            geo.states.append(_kda_state(kda_cfg))
            continue
        # Kimi-Linear / bailing 等按 layer_types 分 MLA/KDA
        if lt and i < len(lt) and lt[i] in ("linear_attention", "kda", "recurrent"):
            if kda_cfg:
                geo.states.append(_kda_state(kda_cfg))
            else:  # bailing_moe_v3 的 recurrent 层
                geo.states.append(_bailing_v3_state(text))
            continue
        ratio = ratios[i] if i < len(ratios) else 0
        per_layer_idx = idx_kpool
        if dsv4:
            # 所有层都有 SWA latent 缓存窗口[attention.py:345]
            geo.layers.append(LayerSpec(
                kind="swa", num_kv_heads=1, head_size=latent,
                window=dsv4_window or None, tp_replicated=True,
                note="DSv4 SWA latent"))
            if ratio <= 1:
                continue
            per_layer_idx = int(ratio)   # DSv4 indexer 同样按 compress_ratio 压缩
        if ratio and ratio > 1:
            spec = LayerSpec(kind="sparse_mla", num_kv_heads=1, head_size=latent,
                             tokens_per_state=int(ratio),
                             tp_replicated=True,
                             note=f"压缩比 {ratio}:每 {ratio} token 存 1 个 latent")
        elif ratios and ratio == 0 and (text.get("sliding_window") or lt):
            w = int(text.get("sliding_window", 0) or 0)
            spec = LayerSpec(kind="swa", num_kv_heads=1, head_size=latent,
                             window=w or None, tp_replicated=True,
                             note="滑动窗口 MLA 层")
        else:
            w = int(g(text, "sliding_window", default=0) or 0)
            spec = LayerSpec(kind="mla", num_kv_heads=1, head_size=latent,
                             window=None, tp_replicated=True)
            if w and lt and i < len(lt) and lt[i] == "sliding_attention":
                spec = LayerSpec(kind="swa", num_kv_heads=1, head_size=latent,
                                 window=w, tp_replicated=True)
        # DSv4 只有 c4 层(compress_ratio==4)带 indexer;其余层按 skip 规则
        if (idx_head and (not dsv4 or ratio == 4)
                and not _skip_indexer(i)):
            spec.indexer_head = idx_head
            spec.indexer_kpool = per_layer_idx
        geo.layers.append(spec)

    if idx_head:
        idx_bytes = idx_head + idx_head // INDEXER_QUANT_BLOCK * 4
        geo.notes.append(f"DSA/dspark indexer:每 token 额外 {idx_bytes} B "
                         f"(uint8 存 {idx_head} 维 fp8 值 + fp32 scale)"
                         + (f",kpool={idx_kpool} 压缩" if idx_kpool > 1 else ""))
    # dots3_note: 额外的 SWA latent 缓存
    swa_lat = int(g(text, "swa_kv_lora_rank", default=0) or 0)
    if swa_lat:
        swa_rope = int(g(text, "swa_qk_rope_head_dim", default=0) or 0)
        w = int(g(text, "swa_sliding_window", "sliding_window", default=0) or 0)
        for l in geo.layers:
            if l.kind == "mla":
                l.note = (l.note + f";额外 SWA latent {swa_lat + swa_rope} 维,"
                          f"窗口 {w}").strip(";")
                l.head_size_v = -(swa_lat + swa_rope)  # 负值标记:额外加在窗口层
        geo.notes.append(f"dots3_note:全注意力层带额外 SWA latent "
                         f"({swa_lat}+{swa_rope} 维,窗口 {w})")
    if longcat := (mt == "longcat_flash"):
        geo.notes.append("longcat_flash:每个 decoder layer 含 2 个 MLA 子层,"
                         "num_layers 即为子层计数")
    return geo


def _kda_state(kda_cfg: dict) -> StateSpec:
    """Kimi KDA:conv (num_heads*head_dim+2*num_k*head_k, kernel-1)
    + recurrent (num_heads, head_dim, head_dim)  [mamba_utils.kda_state_shape]

    dtype 不同:conv 跟 mamba_cache_dtype,recurrent 在 auto 时恒为 fp32
    [mamba_utils.kda_state_dtype: auto -> torch.float32]
    """
    h = int(kda_cfg.get("num_heads", 0))
    d = int(kda_cfg.get("head_dim", 0))
    k = int(kda_cfg.get("conv_kernel_size", kda_cfg.get("short_conv_kernel_size", 4)))
    num_k = int(kda_cfg.get("num_k_heads", h) or h)
    head_k = int(kda_cfg.get("head_k_dim", d) or d)
    conv_dim = h * d + 2 * num_k * head_k
    return StateSpec("kda", conv_dim=conv_dim, conv_len=k - 1, conv_elem=2,
                     temporal_numel=h * d * d, temporal_elem=4,
                     note=f"KDA 状态: conv {conv_dim}×{k-1} + recurrent {h}×{d}×{d}")


def _bailing_v3_state(text: dict) -> StateSpec:
    h = int(g(text, "linear_num_heads", "kda_num_heads", default=0) or 0)
    d = int(g(text, "linear_head_dim", "kda_head_dim", default=0) or 0)
    if not h or not d:
        h = int(g(text, "num_attention_heads", default=0) or 0)
        d = int(g(text, "head_dim", default=0) or 0)
    k = int(g(text, "linear_conv_kernel_size", default=4) or 4)
    conv_dim = h * d * 3
    # bailing 的 KDA 走 kda_state_dtype:recurrent 默认 fp32
    return StateSpec("kda", conv_dim=conv_dim, conv_len=k - 1, conv_elem=2,
                     temporal_numel=h * d * d, temporal_elem=4,
                     note="bailing_v3 KDA 状态(字段近似)")


def _gdn_state(text: dict) -> StateSpec:
    """Qwen3-Next 等 GatedDeltaNet:
    conv (2*nk*dk + nv*dv, kernel-1) + temporal (nv, dv, dk)
    [mamba_utils.gated_delta_net_state_shape]

    GDN 的 temporal 走 mamba_ssm_cache_dtype,auto 时 = 模型 dtype
    [gated_delta_net_state_dtype -> _mamba_state_dtype]
    """
    nk = int(g(text, "linear_num_key_heads", default=0) or 0)
    nv = int(g(text, "linear_num_value_heads", default=0) or 0)
    dk = int(g(text, "linear_key_head_dim", default=0) or 0)
    dv = int(g(text, "linear_value_head_dim", default=0) or 0)
    k = int(g(text, "linear_conv_kernel_dim", default=4) or 4)
    if not (nk and nv):
        return StateSpec("gdn", note="GDN 字段缺失")
    conv_dim = 2 * nk * dk + nv * dv
    return StateSpec("gdn", conv_dim=conv_dim, conv_len=k - 1, conv_elem=2,
                     temporal_numel=nv * dv * dk, temporal_elem=2,
                     note=f"GDN 状态: conv {conv_dim}×{k-1} + temporal {nv}×{dv}×{dk}")


def _mamba1_state(text: dict) -> StateSpec:
    """Mamba-1(Jamba 用 state-spaces/mamba 的 MambaMixer):
    conv (intermediate, kernel-1) + ssm (intermediate, d_state)
    [mamba_utils.mamba1_state_shape];dtype 均跟 auto → 模型 dtype
    """
    hidden = int(g(text, "hidden_size", default=0) or 0)
    expand = int(g(text, "mamba_expand", "expand", default=2) or 2)
    d_inner = int(g(text, "mamba_d_ssm", "d_inner", default=0) or 0) or hidden * expand
    dstate = int(g(text, "mamba_d_state", "ssm_state_size", "d_state", default=16) or 16)
    dconv = int(g(text, "mamba_d_conv", "conv_kernel", "d_conv", default=4) or 4)
    return StateSpec("mamba1", conv_dim=d_inner, conv_len=dconv - 1, conv_elem=2,
                     temporal_numel=d_inner * dstate, temporal_elem=2,
                     note=f"Mamba1 状态: conv {d_inner}×{dconv-1} + ssm {d_inner}×{dstate}")


def _mamba2_state(text: dict) -> StateSpec:
    """[mamba_utils.mamba2_state_shape] conv (d_inner+2*n_groups*d_state, d_conv-1)
    + ssm (nheads, head_dim, d_state)

    n_groups 不整除 tp 时,conv 通道按 extra_groups_for_head_shards 扩到
    n_groups + (tp - n_groups),由 groups/state_size 在 TP 计算时处理。
    """
    hidden = int(g(text, "hidden_size", default=0) or 0)
    expand = int(g(text, "mamba_expand", "expand", default=2) or 2)
    d_inner = int(g(text, "mamba_d_ssm", "d_inner", default=0) or 0) or hidden * expand
    ngroups = int(g(text, "mamba_ngroups", "mamba_n_groups", "n_groups", "ngroups",
                    default=2) or 2)
    dstate = int(g(text, "mamba_d_state", "ssm_state_size", "d_state", default=128) or 128)
    dconv = int(g(text, "mamba_d_conv", "conv_kernel", "d_conv", default=4) or 4)
    nheads = int(g(text, "mamba_n_heads", "mamba_num_heads", "n_mamba_heads",
                   "mamba_nheads", "num_kernels", default=0) or 0)
    # 各模型字段名不同:Falcon-H1 用 mamba_n_heads/mamba_d_head,
    # Zamba2 用 n_mamba_heads/mamba_headdim,Nemotron-H 用 mamba_num_heads/mamba_head_dim
    headdim = int(g(text, "mamba_head_dim", "mamba_d_head", "mamba_headdim",
                    default=0) or 0)
    if not headdim:
        headdim = d_inner // nheads if nheads else 64
    if not nheads:
        nheads = d_inner // headdim if headdim else 0
    conv_dim = d_inner + 2 * ngroups * dstate
    return StateSpec("mamba2", conv_dim=conv_dim, conv_len=dconv - 1, conv_elem=2,
                     temporal_numel=nheads * headdim * dstate, temporal_elem=2,
                     groups=ngroups, state_size=dstate,
                     note=f"Mamba2 状态: conv {conv_dim}×{dconv-1} + ssm "
                          f"{nheads}×{headdim}×{dstate}")


def _short_conv_state(text: dict) -> StateSpec:
    # LFM2 的 conv_dim 是派生值,config.json 只给 hidden_size
    # [lfm2.py:443-453 用 hf_config.conv_dim,源码由 hidden_size 推导]
    dim = int(g(text, "conv_dim", default=0) or 0) or int(
        g(text, "hidden_size", default=0) or 0)
    k = int(g(text, "conv_L_cache", "conv_kernel_size", "conv_kernel", default=4) or 4)
    if not dim:
        return StateSpec("short_conv", note="conv 字段缺失")
    return StateSpec("short_conv", conv_dim=dim, conv_len=max(0, k - 1), conv_elem=2,
                     note=f"短卷积状态 {dim}×{k-1}")


LINEAR_LAYER_TOKENS = {"linear_attention", "linear", "kda", "recurrent", "gdn",
                       "mamba", "conv", "short_conv"}
# 既无逐 token KV 也无固定状态的纯 MLP/MoE 层(Nemotron-H 的 '-'/'E')
NON_KV_LAYER_TOKENS = {"mlp"}

# 并行混合结构:同一层里既有 attention 又有 SSM(不是二选一)
# Falcon-H1:每层 FalconH1ParallelHybrid = attn + mamba 并行相加
# [falcon_h1.py:339-359]
PARALLEL_HYBRID_TYPES = {"falcon_h1", "hybrid"}  # "hybrid" 是 Zamba2 的 layers_block_type


def _analyze_standard(geo, text, root, mt, num_layers):
    heads = int(g(text, "num_attention_heads", "n_heads", default=0) or 0)
    hidden = int(g(text, "hidden_size", "d_model", default=0) or 0)
    kvh = _num_kv_heads(text, heads)
    head = _head_size(text, heads, hidden)
    head_v = int(g(text, "v_head_dim", default=0) or 0) or None
    # Dbrx 特例
    if mt == "dbrx":
        kvh = int(text.get("attn_config", {}).get("kv_n_heads", kvh))
    # falcon 老 MQA
    if mt in ("falcon", "RefinedWeb", "RefinedWebModel"):
        if text.get("multi_query") and not text.get("new_decoder_architecture"):
            kvh = 1
    # step3_text: MQA 硬编码 num_kv_heads=1 (step3_text.py:237)
    if mt == "step3_text":
        kvh = 1
    # NemotronNAS: block_configs 的 n_heads_in_group
    if mt == "nemotron-nas":
        for blk in text.get("block_configs", []):
            att = blk.get("attention", {}) if isinstance(blk, dict) else {}
            if not att.get("no_op"):
                kvh = heads // int(att.get("n_heads_in_group", 1))
                break

    lt = _layer_types(text, num_layers)
    sw = int(g(text, "sliding_window", default=0) or 0)
    # Llama-4:no_rope_layers[i]==0 的是全局层,其余是 chunked local attention
    # [llama4.py:185,251-252 use_chunked_local_attn = not nope and attention_chunk_size]
    # 局部层每请求只保留 attention_chunk_size 个 token
    # [ChunkedLocalAttentionSpec.max_admission_blocks_per_request]
    chunk = int(g(text, "attention_chunk_size", default=0) or 0)
    no_rope = text.get("no_rope_layers")
    if chunk and isinstance(no_rope, list) and not lt:
        lt = ["full_attention" if v == 0 else "chunked_attention" for v in no_rope]
    # Qwen2 系列 config 里有 sliding_window 字段但默认关闭
    # (Qwen2.5-32B: sliding_window=131072 且 use_sliding_window=False)
    if text.get("use_sliding_window") is False:
        sw = 0
    plus1 = mt.split("-")[0] in WINDOW_PLUS_ONE_TYPES or mt in WINDOW_PLUS_ONE_TYPES
    # mimo_v2: 全局层无窗口,swa 层 window=sliding_window;head_v 可能 != head
    windowed_family = mt in (
        "gemma2", "gemma3", "gemma3n_text", "gemma4", "gemma4_text",
        "gemma", "gpt_oss", "mistral", "mixtral", "granite", "granitemoe",
        "granite_hybrid", "exaone", "exaone4", "exaone_moe", "exaone4_5",
        "cohere2_moe", "command_r", "cohere_compass", "qwen3_xpress",
        "mimo_v2", "mimo_v2_flash", "nemotron_h", "nemotron-h", "plamo3",
        "openpangu", "afmoe", "apertus", "laguna", "domyn_edge", "rnj1",
        "llama_text", "qwen3", "qwen3_moe", "embedding_gemma2", "step3p5",
        "muse_glimmer", "olmo_hybrid", "lfm2", "lfm2_moe", "qwen3_next",
        "qwen3_5", "qwen2", "hyperclovax", "alpamayo2_super",
    )
    is_hybrid = bool(lt and any(t in LINEAR_LAYER_TOKENS for t in lt)) or mt in (
        "qwen3_next", "lfm2", "lfm2_moe", "jamba", "zamba2", "falcon_h1",
        "nemotron_h", "nemotron-h", "granite_hybrid", "granitemoehybrid",
        "olmo_hybrid", "bailing_moe", "qwen3_5", "falcon_mamba", "mamba", "mamba2",
    )

    # 跨层 KV 共享(Gemma-3n / Gemma-4:最后 num_kv_shared_layers 层复用前面的 KV)
    n_shared = int(g(text, "num_kv_shared_layers", default=0) or 0)

    # Gemma-4:全注意力层用更宽的 head(global_head_dim + num_global_key_value_heads)
    # [transformers gemma4 modeling_gemma4.py:1137-1140]:
    #   head_dim = global_head_dim if not is_sliding else head_dim
    #   kv_heads = num_global_key_value_heads if (attention_k_eq_v and not sliding)
    #             else num_key_value_heads
    g_head = int(g(text, "global_head_dim", default=0) or 0)
    g_kvh = int(g(text, "num_global_key_value_heads", default=0) or 0)
    k_eq_v = bool(text.get("attention_k_eq_v"))

    # per-layer 窗口列表(plamo3 interleaved_sliding_window / exaone sliding_windows)
    win_list = None
    if isinstance(text.get("interleaved_sliding_window"), list):
        win_list = [w if w else None for w in text["interleaved_sliding_window"]]
    elif isinstance(text.get("sliding_windows"), list):
        win_list = [w if w else None for w in text["sliding_windows"]]

    for i in range(num_layers):
        layer_type = lt[i] if i < len(lt) else ""
        # ---- 并行混合层(Falcon-H1/Zamba2 hybrid):同层 attn 与 SSM 并存 ----
        # Falcon-H1 每层都是 attn+mamba 并行,不像 Jamba 那样按层二选一
        if mt == "falcon_h1":
            geo.states.append(_mamba2_state(text))
        elif layer_type == "hybrid":
            # Zamba2 hybrid 层 = 该层自己的 mamba + 共享 attention 块;
            # vLLM 为每个 hybrid 层各建一个 Attention(独立 KV)
            # [zamba2.py:174-198 "KV cache is unique for every attention layer"]
            geo.states.append(_mamba2_state(text))
            # Zamba2AttentionDecoderLayer 的 Attention() 没有传 num_kv_heads,
            # 因此回退成 MHA(num_attention_heads)
            a_heads = int(g(text, "num_attention_heads", default=0) or 0)
            a_head = int(g(text, "attention_head_dim", default=0) or 0) or head
            geo.layers.append(LayerSpec(kind="full", num_kv_heads=a_heads,
                                        head_size=a_head,
                                        note="Zamba2 共享注意力"))
            continue
        # ---- 纯线性注意力/Mamba 层:无逐 token KV ----
        elif layer_type in LINEAR_LAYER_TOKENS or (is_hybrid and layer_type == "" and not windowed_family):
            if mt in ("jamba", "zamba2", "nemotron_h", "nemotron-h",
                      "falcon_mamba", "mamba", "mamba2",
                      "granite_hybrid", "granitemoehybrid"):
                # Mamba-1(Jamba)与 Mamba-2 的状态形状不同
                geo.states.append(_mamba1_state(text) if mt == "jamba"
                                  else _mamba2_state(text))
            elif layer_type == "conv":
                geo.states.append(_short_conv_state(text))
            else:
                st = _gdn_state(text)
                if st.bytes_total == 0 and text.get("linear_attn_config"):
                    st = _kda_state(text["linear_attn_config"])
                geo.states.append(st)
            continue

        # ---- 纯 MLP/MoE 层(Nemotron-H 的 '-'/'E'):既无 KV 也无状态 ----
        if layer_type in NON_KV_LAYER_TOKENS:
            continue

        # ---- 跨层共享层:不占显存 ----
        if n_shared and i >= num_layers - n_shared:
            geo.layers.append(LayerSpec(kind="shared", num_kv_heads=kvh,
                                        head_size=head,
                                        note="CLA:复用同类型前序层的 KV"))
            continue

        # ---- 窗口判定 ----
        # 注意顺序:lt 为空(如 Mistral 只在 config 里给 int sliding_window)
        # 时 layer_type == "",必须先判 "全部层滑窗" 再落到通用分支
        w = None
        if win_list:
            w = win_list[i % len(win_list)]
        elif mt == "gpt_oss":
            w = sw if i % 2 == 0 else None      # gpt_oss.py:176 偶数层滑动
        elif sw and not lt:
            w = sw + 1 if plus1 else sw         # 全部层都是滑动窗(如 mistral)
        elif layer_type == "sliding_attention":
            w = sw + 1 if plus1 else sw
        elif isinstance(text.get("interleaved_sliding_window"), int):
            w = text["interleaved_sliding_window"]  # openpangu: 所有层
        elif layer_type == "chunked_attention" or "chunked" in layer_type:
            # chunked local attention:每请求只保留一个 chunk 的 token
            w = chunk or None
        elif layer_type in ("full_attention", "", "attention", "default", "dense",
                            "moe", "sparse"):
            w = None
        if mt == "domyn_edge" and w:
            w = w + 1
        # Gemma-4 的全局层用更宽的 head 与更少的 KV 头
        l_head, l_kvh = head, kvh
        if not w and g_head:
            l_head = g_head
            if k_eq_v and g_kvh:
                l_kvh = g_kvh
        spec = LayerSpec(kind="swa" if w else "full", num_kv_heads=l_kvh,
                         head_size=l_head, head_size_v=head_v, window=w,
                         note="滑窗层" if w else "")
        geo.layers.append(spec)

    fam = "GQA" if kvh < heads else ("MQA" if kvh == 1 else "MHA")
    if is_hybrid:
        fam += "+线性注意力混合"
    if any(l.kind == "swa" for l in geo.layers):
        fam += "+滑动窗口"
    if n_shared:
        fam += f"+跨层共享(后 {n_shared} 层复用)"
    geo.family = f"{mt}({fam}: {kvh} KV头 × {head} 维" + \
                 (f"(V:{head_v})" if head_v and head_v != head else "") + ")"
    if mt == "longcat_dual":
        pass
    return geo


# ---------------------------------------------------------------------------
# 计算
# ---------------------------------------------------------------------------

def effective_dtype_bytes(kv_dtype: str, model_dtype: str) -> float:
    """kv dtype 的每元素字节数。auto 跟随模型 dtype
    [vllm/utils/torch_utils.py:533-539 kv_cache_dtype_str_to_dtype]"""
    if kv_dtype in (None, "auto"):
        md = (model_dtype or "").lower()
        if "float32" in md or md in ("fp32", "f32"):
            return 4.0
        if "float16" in md or "bfloat16" in md or md in ("fp16", "bf16", "half"):
            return 2.0
        if "float8" in md or md in ("fp8",):
            return 1.0
        return 2.0
    kv_dtype = DTYPE_ALIASES.get(kv_dtype, kv_dtype)
    if kv_dtype not in DTYPE_BYTES:
        raise SystemExit(f"未知 kv dtype: {kv_dtype},可选 {sorted(DTYPE_BYTES)}")
    return DTYPE_BYTES[kv_dtype]


def normalize_dtype(kv_dtype: str) -> str:
    if kv_dtype is None:
        return "auto"
    return DTYPE_ALIASES.get(kv_dtype, kv_dtype)


def tp_kv_heads(l: LayerSpec, tp: int) -> int:
    if l.tp_replicated:
        return l.num_kv_heads        # MLA latent 不切分
    return max(1, l.num_kv_heads // tp)


def tp_kv_heads_checked(geo: Geometry, tp: int) -> list:
    """校验 TP 与 KV 头数是否兼容,返回警告列表。

    vLLM 的模型实现直接 assert [llama.py:148-155]:
      kv_heads >= tp 时要求 kv_heads % tp == 0;否则要求 tp % kv_heads == 0。
    计算器若静默用整除会给出实际起不来的配置,所以这里显式提示。
    """
    warned = set()
    for l in geo.layers:
        if l.kind in ("shared",) or l.tp_replicated or l.num_kv_heads <= 0:
            continue
        kvh = l.num_kv_heads
        if kvh in warned:
            continue
        if kvh >= tp:
            if kvh % tp:
                warned.add(kvh)
                yield (f"KV 头数 {kvh} 不能被 TP={tp} 整除,vLLM 会在加载时 assert 失败;"
                       f"本例按 {max(1, kvh // tp)} 头/卡估算")
        elif tp % kvh:
            warned.add(kvh)
            yield (f"KV 头数 {kvh} 小于 TP={tp} 且不能整除,vLLM 会在加载时 assert 失败;"
                   f"本例按每卡复制 {kvh} 头估算")


def state_bytes_tp(s: StateSpec, tp: int) -> int:
    return s.bytes_at_tp(tp)


def indexer_token_bytes(l: LayerSpec, kv_dtype: str) -> float:
    """DSA indexer 的每 token KV 字节。

    vLLM 用 torch.uint8 存 fp8 值,并在每 128 个元素后追加 4 字节 fp32 scale
    [deepseek_v2.py:731-736]:
        head_dim = index_head_dim + index_head_dim // 128 * 4
    GLM-5-next 再用 tokens_per_state=index_kpool 压缩 [glm5next attention.py:116-119]
    """
    dim = l.indexer_head + l.indexer_head // INDEXER_QUANT_BLOCK * 4
    return dim / max(1, l.indexer_kpool)


def layer_token_bytes(l: LayerSpec, tp: int, db: float, kv_dtype: str = "auto",
                      packed: bool = False) -> float:
    """单层每 token 的 KV 字节(TP 后、未按窗口截断)。"""
    kvh = tp_kv_heads(l, tp)
    if l.kind in ("mla", "sparse_mla"):
        if packed and kv_dtype in MLA_PACKED_BYTES and MLA_PACKED_BYTES[kv_dtype]:
            b = MLA_PACKED_BYTES[kv_dtype] * kvh
        else:
            b = l.head_size * kvh * db
    else:
        hv = (l.head_size_v
              if (l.head_size_v is not None and l.head_size_v > 0) else l.head_size)
        b = (l.head_size + hv) * kvh * db
    b = b / max(1, l.tokens_per_state)
    if l.indexer_head:
        b += indexer_token_bytes(l, kv_dtype)
    return b


@dataclass
class Result:
    per_token_bytes: int
    layers_desc: list
    state_bytes: int
    kv_at: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def bytes_at_length(geo: Geometry, L: int, tp: int = 1, kv_dtype: str = "auto",
                    encoder_len: int = None) -> int:
    """单序列在长度 L 时的 KV 总字节(含固定状态与窗口截断)。"""
    kv_dtype = normalize_dtype(kv_dtype)
    db = effective_dtype_bytes(kv_dtype, geo.dtype)
    dsv4 = geo.model_type in ("deepseek_v4", "deepseek_v41")
    packed_map = DSV4_PACKED_BYTES if dsv4 else MLA_PACKED_BYTES
    packed = kv_dtype in packed_map and packed_map[kv_dtype] is not None
    total = 0.0
    for l in geo.layers:
        if l.kind == "shared":
            continue
        b = layer_token_bytes(l, tp, db, kv_dtype, packed=packed)
        total += b * (min(L, (l.window or L)) if l.kind == "swa" else L)
    for l in geo.layers:
        if l.head_size_v is not None and l.head_size_v < 0:
            total += (-l.head_size_v) * db * min(L, l.window or L)
    if geo.cross_layers and encoder_len:
        total += (geo.cross_layers * geo.cross_kv_heads * geo.cross_head_size
                  * 2 * db * encoder_len)
    total += sum(state_bytes_tp(s, tp) for s in geo.states)
    return int(total)


def max_tokens_for_budget(geo: Geometry, budget_bytes: float, tp: int = 1,
                          kv_dtype: str = "auto", encoder_len: int = None,
                          hi: int = 1 << 24) -> int:
    """预算反算最大序列长度(二分;对滑窗/混合模型同样成立)。

    上限取模型 max_position_embeddings:纯滑窗模型在窗口之外容量不再增长,
    不设上限会得到远超模型最大长度的无意义数字。
    """
    if geo.max_len:
        hi = min(hi, int(geo.max_len))
    lo = 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if bytes_at_length(geo, mid, tp, kv_dtype, encoder_len) <= budget_bytes:
            lo = mid
        else:
            hi = mid - 1
    return lo


def compute(geo: Geometry, tp: int = 1, kv_dtype: str = "auto",
            seq_lens: tuple = (), encoder_len: int = None) -> Result:
    kv_dtype = normalize_dtype(kv_dtype)
    db = effective_dtype_bytes(kv_dtype, geo.dtype)
    dsv4 = geo.model_type in ("deepseek_v4", "deepseek_v41")
    packed_map = DSV4_PACKED_BYTES if dsv4 else MLA_PACKED_BYTES
    packed = kv_dtype in packed_map and packed_map[kv_dtype] is not None
    per_token = 0
    desc = []
    groups = {}
    for idx, l in enumerate(geo.layers):
        if l.kind == "shared":
            continue
        kvh = tp_kv_heads(l, tp)
        b = layer_token_bytes(l, tp, db, kv_dtype, packed=packed)
        per_token += b
        key = (l.kind, l.num_kv_heads, l.head_size, l.window, l.tokens_per_state,
               l.indexer_head, l.indexer_kpool,
               l.head_size_v if (l.head_size_v or 0) > 0 else None)
        if key not in groups:
            groups[key] = [0, 0.0, l]   # [层数, 每层每token字节, 样例层]
        groups[key][0] += 1
        groups[key][1] = b
    for (kind, kvh, hs, win, tps, ih, ik, _hsv), (n, b, l) in groups.items():
        tag = {"full": "全注意力", "swa": f"滑窗(w={win})", "mla": "MLA",
               "sparse_mla": f"稀疏MLA(×1/{tps})"}.get(kind, kind)
        hv = l.head_size_v if (l.head_size_v is not None and l.head_size_v > 0) else hs
        extra = f" + indexer {ih}d" + (f"/{ik}" if ik > 1 else "") if ih else ""
        print_dims = (f"{kvh}头×{hs}d" if kind in ("mla", "sparse_mla")
                      else f"{kvh}头×({hs}+{hv})d")
        desc.append(f"{n:>3} 层 {tag}: {print_dims}{extra} → "
                    f"每层每token {b:,.0f} B,合计 {b * n:,.0f} B/token")
    state_b = sum(state_bytes_tp(s, tp) for s in geo.states)
    res = Result(per_token_bytes=int(per_token), layers_desc=desc,
                 state_bytes=state_b, warnings=list(tp_kv_heads_checked(geo, tp)))
    for L in seq_lens:
        res.kv_at[L] = bytes_at_length(geo, L, tp, kv_dtype, encoder_len)
    return res


def fmt_b(n: float) -> str:
    n = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or u == "TB":
            return f"{n:.2f} {u}"
        n /= 1024
    return f"{n} B"


def main():
    ap = argparse.ArgumentParser(description="KV cache 大小计算器(按 vLLM 逻辑)")
    ap.add_argument("model", help="HF 模型 ID 或本地 config.json 路径")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel 大小")
    ap.add_argument("--kv-dtype", default="auto",
                    help="auto/fp8/int8/int4_per_token_head/nvfp4/...")
    ap.add_argument("--seq", type=int, nargs="*", default=[4096, 32768],
                    help="要计算的序列长度(可多个)")
    ap.add_argument("--batch", type=int, default=1, help="并发序列数")
    ap.add_argument("--budget-gb", type=float, default=None,
                    help="给定显存预算(GB),反算最大 token/并发")
    ap.add_argument("--encoder-len", type=int, default=None,
                    help="enc-dec 模型的 encoder 输入长度")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    cfg = load_config(args.model)
    geo = analyze(cfg)
    res = compute(geo, tp=args.tp, kv_dtype=args.kv_dtype,
                  seq_lens=tuple(args.seq), encoder_len=args.encoder_len)

    if args.json:
        out = {
            "model_type": geo.model_type, "family": geo.family,
            "kv_per_token_bytes": res.per_token_bytes,
            "state_bytes_per_seq": res.state_bytes,
            "kv_total_at_len": {str(k): v for k, v in res.kv_at.items()},
            "notes": geo.notes,
            "warnings": res.warnings,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    W = 78
    print("=" * W)
    print(f"模型: {args.model}")
    print(f"架构: {geo.family}")
    print(f"KV dtype: {args.kv_dtype}  |  TP: {args.tp}  |  "
          f"模型 dtype: {geo.dtype}")
    print("-" * W)
    if not geo.layers and not geo.states:
        print("该模型没有 KV cache(encoder-only / 状态模型)")
    for d in res.layers_desc:
        print("  " + d)
    for s in geo.states[:3]:
        print(f"  线性层状态: {s.note} → 每 seq 固定 "
              f"{fmt_b(state_bytes_tp(s, args.tp))}")
    if len(geo.states) > 3:
        print(f"  ...(共 {len(geo.states)} 个线性层,每 seq 合计 "
              f"{fmt_b(res.state_bytes)})")
    print("-" * W)
    print(f"每 token KV: {res.per_token_bytes:,} B = {fmt_b(res.per_token_bytes)}"
          f"  (TP={args.tp}, 单卡)")
    if res.state_bytes:
        print(f"每序列固定状态: {fmt_b(res.state_bytes)}")
    for L, v in sorted(res.kv_at.items()):
        line = f"长度 {L:>7,} token: 单序列 {fmt_b(v)}"
        if args.batch > 1:
            line += f"  |  {args.batch} 并发 {fmt_b(v * args.batch)}"
        print(line)
    for n in geo.notes:
        print("※ " + n)
    for w in res.warnings:
        print("⚠ " + w)
    # MHA 参照压缩率:同 head 数、按 qk_nope+qk_rope(K) 与 v_head_dim(V) 计
    kv_layers = geo.kv_layers()
    if kv_layers:
        text_cfg = resolve_text_config(cfg)[0]
        heads = int(g(text_cfg, "num_attention_heads", default=0) or 0)
        if heads:
            db2 = effective_dtype_bytes(args.kv_dtype, geo.dtype)
            mha = 0
            for l in kv_layers:
                hk = l.head_size
                if l.kind in ("mla", "sparse_mla"):
                    hk = int(g(text_cfg, "qk_nope_head_dim",
                               default=0) or 0) + int(g(text_cfg, "qk_rope_head_dim",
                               default=0) or 0) or l.head_size
                    hv = int(g(text_cfg, "v_head_dim", default=0) or 0) or l.head_size
                else:
                    hv = l.head_size_v or l.head_size
                mha += heads * (hk + hv) * db2 / l.tokens_per_state
            if mha:
                print("-" * W)
                print(f"对照同规模 MHA(不压缩): {int(mha):,} B/token, "
                      f"压缩率 {mha / max(1, res.per_token_bytes):.1f}×")
    if args.budget_gb:
        budget = args.budget_gb * 1024 ** 3
        # 二分反算,避免用"每 token 平均值"忽略滑窗截断
        cap = max_tokens_for_budget(geo, budget, args.tp, args.kv_dtype,
                                    args.encoder_len)
        print("-" * W)
        capped = " (已达模型最大长度)" if geo.max_len and cap >= geo.max_len else ""
        print(f"预算 {args.budget_gb} GB ≈ 单序列可容纳 {cap:,} token 的 KV{capped}")
        if args.seq:
            L = max(args.seq)
            per_seq = bytes_at_length(geo, L, args.tp, args.kv_dtype,
                                      args.encoder_len)
            conc = int(budget // max(1, per_seq))
            print(f"  · {L:,} token 序列每条约 {fmt_b(per_seq)} → 约 {conc} 条并发")
    print("=" * W)


if __name__ == "__main__":
    main()
