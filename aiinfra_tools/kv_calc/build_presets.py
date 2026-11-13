#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""拉取主流模型 config,用 kv_calculator 引擎解析,生成 HTML 计算器的预设库"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from kv_calculator import (Geometry, LayerSpec, analyze, load_config,
                           resolve_text_config, g, state_bytes_tp)

PRESETS = [
    # ---- Meta / Llama ----
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    # ---- Qwen ----
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "Qwen/Qwen3.5-4B",
    # ---- DeepSeek ----
    "deepseek-ai/DeepSeek-V2-0628",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-V3.2-Exp",
    # ---- Kimi ----
    "moonshotai/Kimi-K2-Instruct",
    # ---- GLM ----
    "zai-org/GLM-4.5-Air",
    "zai-org/GLM-4.5",
    "zai-org/GLM-4.6",
    # ---- OpenAI / Google ----
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "google/gemma-2-9b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-27b-it",
    "google/gemma-3-1b-it",
    "google/gemma-3n-E4B-it",
    # ---- Mistral ----
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    # ---- MiniMax ----
    "MiniMaxAI/MiniMax-M2",
    # ---- Others ----
    "microsoft/phi-4",
    "LiquidAI/LFM2-8B-A1B",
    "tiiuae/Falcon-H1-34B-Instruct",
    "baidu/ERNIE-4.5-300B-A47B-PT",
]

LABELS = {
    "deepseek_v2": "DeepSeek-V2.5", "deepseek_v3": "DeepSeek-V3/R1",
    "deepseek_v32": "DeepSeek-V3.2(DSA)", "kimi_k2": "Kimi K2",
    "glm4_moe": "GLM-4.5/4.6", "glm4": "GLM-4", "llama": "Llama 3.x",
    "llama4_text": "Llama 4", "qwen2": "Qwen2.5", "qwen3": "Qwen3",
    "qwen3_moe": "Qwen3-MoE", "qwen3_next": "Qwen3-Next",
    "gpt_oss": "gpt-oss", "gemma2": "Gemma-2", "gemma3_text": "Gemma-3",
    "gemma3n_text": "Gemma-3n", "mistral": "Mistral", "minimax_m2": "MiniMax-M2",
    "phi3": "Phi-4", "lfm2": "LFM 2", "falcon_h1": "Falcon-H1",
    "nemotron": "Nemotron", "glm4_moe_lite": "GLM-MoE-Lite",
}


def to_preset(repo_id: str, geo: Geometry, cfg: dict) -> dict:
    text, _ = resolve_text_config(cfg)
    heads = int(g(text, "num_attention_heads", "n_heads", default=0) or 0)
    groups = {}
    for l in geo.layers:
        if l.kind == "shared":
            continue
        key = (l.kind, l.num_kv_heads, l.head_size,
               l.head_size_v if (l.head_size_v or 0) > 0 else l.head_size,
               l.window, l.tokens_per_state, l.indexer_head, l.indexer_kpool)
        groups[key] = groups.get(key, 0) + 1
    layers = [{"kind": k[0], "n": n, "kvh": k[1], "hs": k[2], "hsv": k[3],
               "win": k[4], "tps": k[5], "ih": k[6], "ik": k[7]}
              for k, n in groups.items()]
    return {
        "repo": repo_id,
        "label": repo_id.split("/")[-1],
        "family": geo.family,
        "family_key": LABELS.get(geo.model_type, geo.model_type),
        "max_len": geo.max_len,
        "dtype": geo.dtype,
        "heads": heads,
        "qk_nope": int(g(text, "qk_nope_head_dim", default=0) or 0),
        "qk_rope": int(g(text, "qk_rope_head_dim", default=0) or 0),
        "v_head": int(g(text, "v_head_dim", default=0) or 0),
        "layers": layers,
        "n_state_layers": len(geo.states),
        "state_bytes": sum(s.bytes_total for s in geo.states),
        # 状态形状(TP 前),前端据此按 TP 精确复算 conv/temporal
        "state_shape": ({
            "conv_dim": geo.states[0].conv_dim,
            "conv_len": geo.states[0].conv_len,
            "conv_elem": geo.states[0].conv_elem,
            "temporal_numel": geo.states[0].temporal_numel,
            "temporal_elem": geo.states[0].temporal_elem,
            "groups": geo.states[0].groups,
            "state_size": geo.states[0].state_size,
        } if geo.states else None),
        "state_desc": geo.states[0].note if geo.states else "",
        "shared_layers": sum(1 for l in geo.layers if l.kind == "shared"),
        "notes": geo.notes,
    }


def main():
    out = []
    failed = []
    for repo in PRESETS:
        try:
            cfg = load_config(repo)
            geo = analyze(cfg)
            p = to_preset(repo, geo, cfg)
            out.append(p)
            print(f"  ok  {repo:60s} {p['family'][:48]}")
        except SystemExit as e:
            failed.append((repo, str(e)[:80]))
            print(f"FAIL  {repo}: {str(e)[:80]}")
        except Exception as e:
            failed.append((repo, repr(e)[:80]))
            print(f"FAIL  {repo}: {repr(e)[:80]}")
    with open(os.path.join(os.path.dirname(__file__), "presets.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n{len(out)} presets → presets.json; {len(failed)} failed")


if __name__ == "__main__":
    main()
