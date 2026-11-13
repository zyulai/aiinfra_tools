#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交叉验证:用已知公开数值断言引擎正确性"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
from kv_calculator import (analyze, compute, load_config,
                           max_tokens_for_budget)

CASES = [
    # (repo, 期望每token字节@bf16/TP1, 说明)
    ("meta-llama/Llama-3.1-70B-Instruct", 327680,
     "80层×8头×(128+128)×2B,社区公认 320 KiB/token"),
    ("Qwen/Qwen3-32B", 262144, "64层×8×(128+128)×2"),
    ("deepseek-ai/DeepSeek-V3", 70272,
     "61层 MLA 576维×2B,官方 ~70KB/token"),
    ("deepseek-ai/DeepSeek-R1", 70272, "同 V3"),
    ("deepseek-ai/DeepSeek-V2-0628", 69120, "60层 MLA 576×2B(V2 是 60 层)"),
    ("moonshotai/Kimi-K2-Instruct", 70272, "K2 = 61层 MLA 576"),
    ("deepseek-ai/DeepSeek-V3.2-Exp", 78324,
     "61×(576×2 MLA + 132 indexer:128B fp8值 + 4B fp32 scale)"),
    ("Qwen/Qwen3-Next-80B-A3B-Instruct", 24576,
     "12 full×2头×(256+256)×2B(36 层 GDN 无 KV)"),
    ("openai/gpt-oss-120b", 73728,
     "36层×8×(64+64)×2B(18滑窗128 + 18全局+sink)"),
    ("openai/gpt-oss-20b", 49152, "24层×8×(64+64)×2B"),
    ("google/gemma-3-27b-it", 507904, "62层×16×(128+128)×2B(52滑+10全)"),
    ("google/gemma-2-27b-it", 376832, "46层 1:1 交错"),
    ("MiniMaxAI/MiniMax-M2", 253952, "62层×8×(128+128)×2B"),
    ("zai-org/GLM-4.5", 376832, "92层×8×(128+128)×2B"),
    ("baidu/ERNIE-4.5-300B-A47B-PT", 221184, "54层×8×(128+128)×2B(head=8192/64)"),
]

ok = fail = 0
for repo, expect, why in CASES:
    try:
        geo = analyze(load_config(repo))
        res = compute(geo, tp=1, kv_dtype="auto")
        got = res.per_token_bytes
        mark = "✓" if got == expect else "✗"
        if got == expect:
            ok += 1
        else:
            fail += 1
        print(f"{mark} {repo.split('/')[-1]:36s} 期望 {expect:>7,}  实得 {got:>7,}  ({why})")
    except Exception as e:
        fail += 1
        print(f"✗ {repo}: {e}")

# 附加断言:Qwen3-Next 固定状态 & gpt-oss 长度截断
geo = analyze(load_config("Qwen/Qwen3-Next-80B-A3B-Instruct"))
res = compute(geo, seq_lens=(131072,))
assert res.state_bytes == 36 * (8192 * 3 + 32 * 128 * 128) * 2, "GDN 状态"
r2 = compute(analyze(load_config("openai/gpt-oss-120b")), seq_lens=(131072,))
expect2 = 18 * 2048 * 131072 + 18 * 2048 * 128
assert r2.kv_at[131072] == expect2, f"gpt-oss@128K {r2.kv_at[131072]:,} != {expect2:,}"
ok += 2
print(f"✓ Qwen3-Next GDN 固定状态 39,518,208 B/seq;gpt-oss-120b @128K 截断计算 {expect2:,} B")


# ---------------------------------------------------------------------------
# 回归用例:不依赖网络,直接喂 config dict
# 覆盖本次修正的解析分支(Mistral 全滑窗、Jamba/Zamba2 派生层类型、
# Falcon-H1 并行混合、indexer 压缩/跳过、kv dtype 别名、预算反算)
# ---------------------------------------------------------------------------
def _check(name, got, expect):
    global ok, fail
    if got == expect:
        ok += 1
        print(f"✓ {name}: {got:,}" if isinstance(got, int) else f"✓ {name}: {got}")
    else:
        fail += 1
        print(f"✗ {name}: 实得 {got!r} 期望 {expect!r}")


_MISTRAL = dict(model_type="mistral", hidden_size=4096, num_hidden_layers=32,
                num_attention_heads=32, num_key_value_heads=8,
                intermediate_size=14336, vocab_size=32000, head_dim=128,
                sliding_window=4096, max_position_embeddings=32768,
                torch_dtype="bfloat16")
_g = analyze(_MISTRAL)
_check("Mistral int sliding_window → 全部 32 层滑窗",
       sum(1 for l in _g.layers if l.kind == "swa"), 32)
_check("Mistral @32K(窗口 4096 截断)",
       compute(_g, seq_lens=(32768,)).kv_at[32768], 32 * 8 * 256 * 2 * 4096)
_check("Mistral 预算反算受 max_position_embeddings 约束",
       max_tokens_for_budget(_g, 8 * 1024 ** 3), 32768)

# Qwen2 系带 sliding_window 字段但 use_sliding_window=False
_qwen2 = dict(_MISTRAL, model_type="qwen2", sliding_window=131072,
              use_sliding_window=False)
_check("Qwen2 use_sliding_window=False → 不算滑窗",
       sum(1 for l in analyze(_qwen2).layers if l.kind == "swa"), 0)

# Jamba:layers_block_type 是 HF property,config.json 不含;
# 应按 attn_layer_period/offset 推导(i%8==4 为 attention → 4 层)
_jamba = dict(model_type="jamba", hidden_size=4096, num_hidden_layers=32,
              num_attention_heads=32, num_key_value_heads=8,
              intermediate_size=14336, vocab_size=65536, head_dim=128,
              attn_layer_period=8, attn_layer_offset=4,
              mamba_expand=2, mamba_d_state=16, mamba_d_conv=4,
              torch_dtype="bfloat16")
_gj = analyze(_jamba)
_check("Jamba attention 层数", len(_gj.layers), 4)
_check("Jamba Mamba 层数", len(_gj.states), 28)

# Falcon-H1:每层 attn + mamba 并行
_fh1 = dict(model_type="falcon_h1", hidden_size=5120, num_hidden_layers=72,
            num_attention_heads=20, num_key_value_heads=4, head_dim=128,
            intermediate_size=25600, vocab_size=65024, torch_dtype="bfloat16",
            mamba_expand=2, mamba_d_ssm=4096, mamba_n_heads=32,
            mamba_d_head=128, mamba_d_state=256, mamba_d_conv=4,
            mamba_n_groups=2)
_gf = analyze(_fh1)
_rf = compute(_gf)
_check("Falcon-H1 KV 层数 = 72(并行混合)", len(_gf.layers), 72)
_check("Falcon-H1 每 token", _rf.per_token_bytes, 72 * 4 * 256 * 2)
_check("Falcon-H1 每 seq 状态", _rf.state_bytes,
       72 * ((4096 + 2 * 2 * 256) * 3 * 2 + 32 * 128 * 256 * 2))
# TP=8:n_groups=2 < tp → conv 通道扩到 2+(8-2)=8 组后除 TP
_check("Falcon-H1 TP=8 每 seq 状态",
       compute(_gf, tp=8).state_bytes,
       72 * (((4096 + 2 * 8 * 256) // 8) * 3 * 2 + (32 // 8) * 128 * 256 * 2))

# DSA indexer:uint8 值 + 每 128 元素 4B fp32 scale → 132 B/token/层
_dsa = dict(model_type="deepseek_v32", hidden_size=7168, num_hidden_layers=61,
            num_attention_heads=128, num_key_value_heads=128, head_dim=128,
            kv_lora_rank=512, qk_rope_head_dim=64, v_head_dim=128,
            qk_nope_head_dim=128, index_head_dim=128, index_topk=2048,
            index_topk_freq=1, intermediate_size=18432, vocab_size=129280,
            torch_dtype="bfloat16")
_check("V3.2 DSA indexer 132 B/token",
       compute(analyze(_dsa)).per_token_bytes, 61 * (576 * 2 + 132))

# kv dtype 别名与 auto 跟随模型 dtype
_check("--kv-dtype bf16 别名可用",
       compute(_gf, kv_dtype="bf16").per_token_bytes, _rf.per_token_bytes)
_check("--kv-dtype fp16 别名可用",
       compute(_gf, kv_dtype="fp16").per_token_bytes, _rf.per_token_bytes)
_check("kv-dtype auto 跟随模型 bf16",
       compute(_gf, kv_dtype="auto").per_token_bytes, _rf.per_token_bytes)
_check("kv-dtype auto 跟随模型 fp32",
       compute(analyze(dict(_fh1, torch_dtype="float32")),
               kv_dtype="auto").per_token_bytes, _rf.per_token_bytes * 2)

# LFM2:conv_dim 不是 config 字段,状态应按 hidden_size 计
_lfm2 = dict(model_type="lfm2_moe", hidden_size=2048, num_hidden_layers=24,
             num_attention_heads=32, num_key_value_heads=8, head_dim=64,
             intermediate_size=7168, vocab_size=65536, torch_dtype="bfloat16",
             conv_L_cache=3,
             layer_types=["conv", "conv", "full_attention"] * 8)
_rl = compute(analyze(_lfm2))
_check("LFM2 固定状态(16 层 conv × 2048×2×2B)", _rl.state_bytes, 16 * 2048 * 2 * 2)
_check("LFM2 KV 层数", len(analyze(_lfm2).layers), 8)
# KDA recurrent 状态默认 fp32,mamba_ssm_cache_dtype=auto
_kda = dict(model_type="kimi_linear", hidden_size=4096, num_hidden_layers=20,
            num_attention_heads=32, num_key_value_heads=32, head_dim=128,
            kv_lora_rank=512, qk_rope_head_dim=64, qk_nope_head_dim=128,
            v_head_dim=128, intermediate_size=11008, vocab_size=163840,
            torch_dtype="bfloat16",
            linear_attn_config={"num_heads": 32, "head_dim": 128,
                                "short_conv_kernel_size": 4,
                                "kda_layers": list(range(1, 21)),
                                "full_attn_layers": []})
_check("KDA 状态 conv bf16 + recurrent fp32",
       compute(analyze(_kda)).state_bytes,
       20 * ((32 * 128 * 3) * 3 * 2 + 32 * 128 * 128 * 4))

# Gemma-4:全注意力层的 head 更宽、KV 头更少
# [transformers modeling_gemma4.py:1137-1140]
_gemma4 = dict(model_type="gemma4_text", hidden_size=64, num_hidden_layers=6,
               num_attention_heads=8, num_key_value_heads=4, head_dim=16,
               global_head_dim=32, num_global_key_value_heads=8,
               attention_k_eq_v=True, sliding_window=1024,
               intermediate_size=128, vocab_size=256, torch_dtype="bfloat16",
               layer_types=["sliding_attention"] * 5 + ["full_attention"])
_g4 = analyze(_gemma4)
_check("Gemma-4 滑窗层 head_dim=16/kv=4", _g4.layers[0].head_size, 16)
_check("Gemma-4 全局层取 global_head_dim=32", _g4.layers[5].head_size, 32)
_check("Gemma-4 全局层 KV 头取 num_global_key_value_heads=8",
       _g4.layers[5].num_kv_heads, 8)
_check("Gemma-4 每 token(5 滑 + 1 全)", compute(_g4).per_token_bytes,
       5 * 4 * 32 * 2 + 1 * 8 * 64 * 2)

# Llama-4:no_rope_layers==0 是全局层,其余 chunked local(只保留 chunk 个 token)
# [llama4.py:185,251-252; ChunkedLocalAttentionSpec]
_llama4 = dict(model_type="llama4_text", hidden_size=5120, num_hidden_layers=48,
               num_attention_heads=40, num_key_value_heads=8, head_dim=128,
               attention_chunk_size=8192, intermediate_size=8192,
               vocab_size=202048, torch_dtype="bfloat16",
               no_rope_layers=[1, 1, 1, 0] * 12)
_g4l = analyze(_llama4)
_check("Llama-4 全局层数 = 12",
       sum(1 for l in _g4l.layers if l.kind == "full"), 12)
_check("Llama-4 chunked 层被窗口截断",
       compute(_g4l, seq_lens=(1 << 20,)).kv_at[1 << 20],
       12 * 8 * 256 * 2 * (1 << 20) + 36 * 8 * 256 * 2 * 8192)

# TP 不整除 KV 头时 vLLM 会 assert 失败,计算器要给警告而不是静默取整
_llama = dict(model_type="llama", hidden_size=4096, num_hidden_layers=32,
              num_attention_heads=32, num_key_value_heads=8, head_dim=128,
              intermediate_size=14336, vocab_size=32000, torch_dtype="bfloat16")
_gl = analyze(_llama)
_check("TP=3 不能整除 8 KV 头 → 有警告", bool(compute(_gl, tp=3).warnings), True)
_check("TP=4 能整除 → 无警告", compute(_gl, tp=4).warnings, [])
_check("TP=16 > 8 KV 头且整除 → 无警告", compute(_gl, tp=16).warnings, [])
_check("KV 头 1(MQA)+ TP=8 → 无警告",
       compute(analyze(dict(_llama, num_key_value_heads=1)), tp=8).warnings, [])

print(f"\n{ok} 通过,{fail} 失败")
sys.exit(1 if fail else 0)
