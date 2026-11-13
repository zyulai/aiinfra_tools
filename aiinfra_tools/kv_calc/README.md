# KV Cache 显存计算器

按 **vLLM 最新 main 分支源码**(commit `a55937f`,2026-10-25,位于 `../vllm`)的计算口径,
为**全部模型架构**计算 KV cache 大小。重点处理各类 **KV 压缩机制**,逐一对照 vLLM 源码实现,
不凭记忆猜。

## 文件

| 文件 | 说明 |
|---|---|
| `kv_calculator.py` | 命令行/可导入计算引擎:传 HF 模型 ID 或本地 `config.json`,输出每 token KV、指定长度/并发总量、预算反算 |
| `kv_cache_calculator.html` | 离线交互式网页计算器(36 个主流模型预设,数据来自真实 config.json),双击即用 |
| `presets.json` | 预设库(由 `build_presets.py` 从 HF 拉取真实 config 生成) |
| `build_presets.py` | 重新拉取模型生成预设库 |
| `build_html.py` | 重新打包独立版 HTML |
| `verify_known.py` | 交叉验证脚本(17 项已知数值断言,全部通过) |
| `arch_registry.json` | vLLM 419 个 architecture 类名 → 实现文件映射(从 registry.py 提取) |

## 用法

```bash
python kv_calculator.py deepseek-ai/DeepSeek-V3                 # 直接算
python kv_calculator.py Qwen/Qwen3-Next-80B-A3B-Instruct --seq 131072 --batch 4
python kv_calculator.py meta-llama/Llama-3.1-70B-Instruct --tp 8 --kv-dtype fp8
python kv_calculator.py deepseek-ai/DeepSeek-V3 --budget-gb 40 --seq 32768   # 预算反算
python kv_calculator.py ./config.json                          # 本地 config
```

依赖:`pip install huggingface_hub`(受限仓库会自动尝试 unsloth 等公开镜像)。

## 计算口径(与 vLLM 一致,附源码依据)

- **通用层**:每层每 token 字节 = `(head_size_K + head_size_V) × num_kv_heads × dtype字节数`
  (`vllm/v1/kv_cache_interface.py` `AttentionSpec.state_content_size_bytes`)。
- **GQA/MQA/MHA**:`num_key_value_heads` 等字段别名见 `CONFIG_FIELD_ALIASES`
  (`vllm/transformers_utils/config.py:56`);缺省回退 MHA(=num_attention_heads)。
  TP 时 KV 头除以 TP(不足复制,`max(1, n//tp)`)。
- **MLA**(DeepSeek V2/V3/V3.2/V4、Kimi K2/K3/Linear、GLM-MoE-Lite、LongCat、
  Bailing、Pangu-Ultra、HY-V4、dots3、k3_dspark 等,名单见
  `vllm/transformers_utils/model_arch_config_convertor.py:294-342`):
  每层每 token = `(kv_lora_rank + qk_rope_head_dim) × 1 × dtype`,无独立 V
  (`mla_attention.py:480-485`)。**latent 跨 TP rank 复制,不除 TP**。
- **DSA 稀疏索引**(DeepSeek-V3.2、HY-V4、k3_dspark):每层每 token 额外
  `index_head_dim + index_head_dim//128*4` **字节**(uint8 存 fp8 值 + 每 128 元素
  4B fp32 scale,不是 `head_dim × kv_dtype`),见 `deepseek_v2.py:731-736`
  `DeepseekV32IndexerCache(head_dim=self.head_dim + self.head_dim//128*4, dtype=torch.uint8)`。
  仅当 config 含 `index_topk` 时才建 indexer(`deepseek_v2.py:1143`),
  且受 `index_topk_freq` / `index_topk_pattern` / `index_skip_topk_offset` 跳过规则约束。
- **GLM-5-next**:indexer K 每 `index_kpool` 个 token 存 1 个
  (`vllm/models/glm5next/common/attention.py:82-119`)。
- **DeepSeek-V4/V4.1 稀疏 MLA**:每层 `compress_ratio` 个 token 压成 1 个 latent
  (`deepseek_v4_utils.py get_compress_ratios`;ratio≤1 的层只有 **SWA latent** 缓存,
  没有压缩缓存,见 `models/deepseek_v4/attention.py:345,856-859`)。indexer 只在
  `compress_ratio==4` 的层建立(`attention.py:289-302`)。
- **滑动窗口**(Mistral、Gemma-2/3/3n/4、gpt-oss、Cohere、Granite、Exaone、
  Nemotron-H、plamo3、openpangu 等):每 token 大小不变,但该层缓存 token 数被
  窗口上限截断(预算按 `min(L, window)` 计)。交错 pattern 规则:
  - `layer_types` 列表优先(Gemma、Granite、Cohere2、LFM2、Qwen3.5…)
  - `hybrid_override_pattern`(Nemotron-H):`*`=attention、`M`=Mamba、`-`/`E`=MLP/MoE
  - `attn_layer_period`/`attn_layer_offset`(Jamba):`layers_block_type` 是 transformers
    的派生 property,**config.json 里没有**,必须按这两个字段推导
  - `full_attention_interval=N`(Qwen3-Next):`(i+1)%N==0` 为全注意力层,其余 GDN
  - `sliding_window_pattern=P`(Gemma):`(i+1)%P==0` 为全局层(Gemma3-27B:p=6 → 52滑+10全)
  - Mistral 系 config 只给 `sliding_window: int`,没有 `layer_types` → **所有层都是滑窗**
    (vLLM 的 `configs/mistral.py:191-193` 会补 `layer_types=["sliding_attention"]*N`)
  - `use_sliding_window: false`(Qwen2 系)时即使有 `sliding_window` 字段也不滑窗
  - gpt_oss:偶数层滑窗 128、奇数层全局+attention sink(`gpt_oss.py:176`,sink 是参数不占 KV)
  - Cohere 系窗口为包含式半径,vLLM 内部 +1(`commandr.py:237-244`)
  - plamo3:`interleaved_sliding_window` 逐层窗口列表;exaone-moe:`sliding_windows` 列表
  - **Llama-4 chunked local attention**:`no_rope_layers[i]==0` 是全局层,其余层用
    `attention_chunk_size` 作窗口(`llama4.py:185,251-252`;
    `ChunkedLocalAttentionSpec` 按 chunk 分配,不是全长)。漏算会把 1M 上下文高估 4 倍
    (Scout:206 GB → 52.7 GB)。
  - **Gemma-4 每层 head 不同**:全注意力层 `head_dim=global_head_dim`(512)、
    KV 头取 `num_global_key_value_heads`(`attention_k_eq_v` 时),滑窗层仍用
    `head_dim`/`num_key_value_heads`(256/16)。见 `modeling_gemma4.py:1137-1140`。
    统一按滑窗层几何算会高估约 9%。
  - **TP 整除校验**:vLLM 对 `kv_heads % tp` / `tp % kv_heads` 直接 assert
    (`llama.py:148-155`)。计算器现在给出警告(CLI 的 `⚠`、JSON 的 `warnings`),
    而不是静默取整出一个跑不起来的配置。
- **混合线性注意力**(Qwen3-Next/Qwen3.5、Kimi-Linear/K3、LFM-2、Jamba、Zamba2、
  Nemotron-H、bailing、OLMo-Hybrid…):线性层无逐 token KV,改为每请求固定状态缓存,
  shape 公式照抄 `vllm/model_executor/layers/mamba/mamba_utils.py MambaStateShapeCalculator`:
  - GDN(Qwen3-Next):conv `(2·nk·dk + nv·dv) × (k-1)` + temporal `nv × dv × dk`
  - KDA(Kimi):conv `(h·d + 2·nk·dk) × (k-1)` + recurrent `h × d × d`
  - Mamba2(Falcon-H1/Nemotron-H/Zamba2):conv `(d_inner + 2·n_groups·d_state) × (d_conv-1)`
    + ssm `nheads × head_dim × d_state`;Jamba 走 Mamba-1 形状
  - **dtype 分开算**:conv 跟 `mamba_cache_dtype`,temporal 跟 `mamba_ssm_cache_dtype`;
    两者 `auto` 时都等于模型 dtype,**唯独 KDA 的 recurrent 在 auto 时恒为 fp32**
    (`mamba_utils.py:133-149 kda_state_dtype`)
  - 状态维度按 TP 切分;Mamba2 的 `n_groups < tp` 时 conv 通道先扩到 `tp` 组再除
    (`mamba_utils.py:206-208 extra_groups_for_head_shards`)
- **并行混合结构**(Falcon-H1、Zamba2 的 `hybrid` 层):同一层里 **attention 与 SSM 并存**
  (Falcon-H1 每层 `FalconH1ParallelHybrid` 同时建 attn + mamba,`falcon_h1.py:339-359`;
  Zamba2 每个 hybrid 层各建一个 Attention,`zamba2.py:174-198`),不能按"二选一"处理。
- **跨层 KV 共享 / CLA**(Gemma-3n、Gemma-4):最后 `num_kv_shared_layers` 层复用同类型
  前序层的 KV,不占显存(`gemma3n.py:311-382`、`gemma4.py:470-489`)。
- **Kimi-Linear/K3 的 KDA 层**:`linear_attn_config.kda_layers`(1-based,`(i+1)` 判定)
  是 KDA,其余层是 NoPE-MLA(`vllm/models/kimi_k3/nvidia/model.py:716-780`)。
- **编码器-解码器**(Whisper/FunASR/Cohere-ASR 等):decoder 自注意力常规 KV + 每层
  cross-attention 缓存 encoder K/V(长度=音频/源文 token 数)。
- **纯编码器**(BERT/CLIP/SigLIP/vision tower):无 KV cache。
- **KV 量化**:`--kv-dtype` 支持 auto/bf16/fp16(2B)、fp8/int8(1B)、int4(0.5B)、
  nvfp4(≈0.56B 含 scale)及 turboquant 系等效字节数(见 `kv_calculator.py DTYPE_BYTES`)。
  `bf16`/`fp16`/`fp8` 等常见缩写会自动归一(`DTYPE_ALIASES`);`auto` 跟随模型 dtype
  (`torch_dtype`),与 vLLM `kv_cache_dtype_str_to_dtype` 一致。
  `fp8_ds_mla`/`nvfp4_ds_mla` 是打包布局,vLLM 直接给出每 token 字节数
  (通用 MLA 656/352 B,DSv4 584 B),不按 head_size 换算。
- **MTP/EAGLE draft 层**:默认不计(未开投机解码时不占 KV);`num_nextn_predict_layers`
  的几何与主模型对应层一致。

## 交叉验证(verify_known.py,45/45 通过)

| 模型 | 每 token KV(bf16, TP=1) | 依据 |
|---|---|---|
| Llama-3.1-70B | 327,680 B(320 KiB) | 80层×8×(128+128)×2 |
| Qwen3-32B | 262,144 B | 64层×8×(128+128)×2 |
| DeepSeek-V3 / R1 / Kimi-K2 | 70,272 B | 61层 MLA 576×2 |
| DeepSeek-V2-0628 | 69,120 B | 60层 MLA(V2 是 60 层) |
| DeepSeek-V3.2-Exp | 78,324 B | 61×(576×2 MLA + 132 indexer) |
| Qwen3-Next-80B | 24,576 B + 37.7MB/seq 固定状态 | 12 全注意力层 + 36 GDN |
| gpt-oss-120b | 73,728 B | 18滑窗(128)+18全局 |
| gpt-oss-20b | 49,152 B | 24层×8×(64+64)×2 |
| Gemma-3-27B | 507,904 B | 52滑(1024)+10全,16×128 |
| Gemma-2-27B | 376,832 B | 46层 1:1 交错,窗口 4096 |
| MiniMax-M2 | 253,952 B | 62层 GQA 8×128 |
| GLM-4.5 / 4.6 | 376,832 B | 92层 8×128 |
| ERNIE-4.5-300B | 221,184 B | 54层 8×128 |

离线回归用例(不联网):Mistral 全滑窗、Qwen2 `use_sliding_window=False`、Jamba 派生层类型、
Falcon-H1 并行混合(含 TP=8 group 扩展)、DSA indexer 132 B、KDA fp32 recurrent、
LFM2 短卷积、Gemma-4 分层 head 几何、Llama-4 chunked local、TP 整除警告、
kv dtype 别名与 auto 解析、预算反算上限。

另验证:GQA 在 TP=8 时每卡 ÷8(Llama-70B → 40KB),MLA 在 TP=4 时 latent 不切分
(DeepSeek-V3 仍 68.6KB/卡),gpt-oss@128K 滑窗截断(4,836,556,800 B)。

## 已知边界

- turboquant/ultraquant 等 新型量化的 scale 开销按等效字节数近似(nvfp4=0.5625B/元素)。
- bailing_v3 的 KDA 状态字段为近似(字段命名不稳定);Zamba2/NemotronNAS 等长尾结构
  按 config 字段尽力解析。
- 未收录的自定义架构回退为"标准 GQA + layer_types"通用规则,输出会标注家族名供人工核对。
- Whisper cross-attention 的 encoder 长度需 `--encoder-len` 指定(默认读
  `max_source_positions`,通常 1500 = 30 秒音频)。
