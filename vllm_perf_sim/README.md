# vLLM 大模型推理性能模拟器

一个**分析式、离散事件**的 vLLM 在线推理性能模拟器。输入 HF `config.json`、GPU、TP/PP、请求长度和到达率，输出与 `vllm bench serve` 相同口径的：

- Request / Output token / Total token throughput
- Peak output token throughput、Peak concurrent requests
- TTFT、TPOT、ITL、E2EL 的 mean / median / P90 / P95 / P99
- 每个 engine step 的 compute、weight read、attention、通信和 CPU 开销分解

> **这是容量规划和趋势分析工具，不是实测替代品。** 所有输出均为模拟值；默认效率参数是显式假设，不是从公开图表反推的“实测数据”。做采购或 SLO 决策前，请用目标机器上的真实 `vllm bench serve` 结果校准。

## 快速开始

在仓库根目录运行，无需安装第三方依赖：

```bash
python3 vllm_perf_sim/run.py \
  --config test_config.json \
  --gpu h100-sxm --num-gpus 1 --tp 1 \
  --num-prompts 100 --input-len 512 --output-len 128 \
  --request-rate inf
```

也可以把 `PYTHONPATH=.:vllm_perf_sim` 替换上面的快捷入口，使用 `python3 -m vllm_perf_sim`。

模拟 4 卡 TP 上的 70B 模型（传入对应模型配置）：

```bash
python3 vllm_perf_sim/run.py \
  --config /path/to/Llama-3-70B/config.json \
  --gpu h100-sxm --num-gpus 4 --tp 4 \
  --num-prompts 500 --input-len 202 --output-len 179 \
  --request-rate inf --save-json result.json --save-steps steps.json
```

查看支持的 GPU：

```bash
python3 vllm_perf_sim/run.py --config test_config.json --list-gpus
```

也可以安装成命令（仓库根目录需可导入 `parallel_tool`）：

```bash
pip install -e ./vllm_perf_sim
vllm-perf-sim --config test_config.json --gpu a100-80
```

## 模拟什么

1. **Continuous batching**：请求在每个 engine step 后动态加入/退出 running batch。
2. **Chunked prefill**：decode 优先，剩余 token budget 用于 prefill；长 prompt 分块执行。
3. **调度约束**：`max_num_batched_tokens`、`max_num_seqs`、客户端并发上限和 KV cache 容量。
4. **显存**：权重按 TP/PP/EP 分片，KV cache 按 GQA head 数、层数、KV dtype 估算。
5. **Roofline 时间模型**：矩阵乘 FLOPs、权重读取、随上下文增长的 KV 读取、TP all-reduce 和 CPU 调度开销取瓶颈/叠加。
6. **MoE**：decode 只计算被 router 选中的 experts，而不是全部 expert 参数。

模型结构的简化边界见下方“限制”。

## 主要参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--gpu` | `h100-sxm` | `h100-sxm/h100-pcie/h200/a100-80/a100-40/l40s/rtx4090/rtx5090` |
| `--num-gpus` | 1 | 总卡数，当前必须等于 `TP*PP` |
| `--tp / --pp / --ep` | 1 | 张量/流水线/专家并行 |
| `--dtype / --kv-dtype` | bf16 | 权重和 KV 数据类型 |
| `--num-prompts` | 100 | 请求数 |
| `--input-len / --output-len` | 512 / 128 | 平均输入/输出 token 数 |
| `--input-std / --output-std` | 0 | 长度高斯扰动标准差 |
| `--request-rate` | inf | Poisson 到达率（req/s）；inf 表示全部立即到达 |
| `--max-concurrency` | 无 | 客户端在途请求上限 |
| `--max-num-batched-tokens` | 8192 | 单次调度 token budget |
| `--max-num-seqs` | 256 | 最大 running sequences |
| `--gpu-memory-utilization` | 0.90 | 用于估算 KV 容量的显存比例 |
| `--kv-cache-gb` | 自动 | 直接指定每卡 KV 容量 |
| `--flops-efficiency` | GPU 默认 | 有效计算效率，建议实测校准 |
| `--memory-efficiency` | GPU 默认 | 有效显存带宽效率，建议实测校准 |
| `--base-overhead-ms` | GPU 默认 | 每个 engine step 的固定 CPU 开销 |
| `--per-sequence-overhead-ms` | GPU 默认 | 每条 running sequence 的 CPU 开销 |
| `--save-json` | 无 | 保存 bench-serve 风格指标和模拟元数据 |
| `--save-steps` | 无 | 保存每个 engine step 的详细分解 |

## 使用公开性能数据

`reference_data/sources.json` 保存了来源、访问日期、原始数值和适用范围：

- vLLM 官方 Benchmark CLI 文档的完整 8B 示例，用于验证指标格式与公式。文档未注明 GPU，**不用于 GPU 标定**。
- vLLM v0.6.0 性能文章的 A100/H100、8B/70B 和 ShareGPT/prefill-heavy/decode-heavy 场景。文章的绝对结果只在图表中，因此这里只记录场景和公开的相对提升，**不编造绝对数值**。
- GPU 数据手册中的显存、带宽和 dense Tensor Core 峰值。峰值不是 vLLM 实测吞吐。

## 用真实 bench serve 结果校准

先在实际环境运行：

```bash
vllm bench serve \
  --backend vllm --model /path/to/model \
  --dataset-name random --num-prompts 100 \
  --random-input-len 512 --random-output-len 128 \
  --request-rate inf --save-json --result-filename real.json
```

用相同模型/硬件/长度/到达率运行模拟器，比较 `duration`、`mean_ttft_ms`、`mean_tpot_ms` 和 `output_throughput`。`vllm_perf_sim.calibration.calibrate()` 可计算 workload-specific 修正系数。建议：

1. 先校准低并发 TPOT，主要调整 `--base-overhead-ms` / `--memory-efficiency`；
2. 再校准大 batch prefill，主要调整 `--flops-efficiency`；
3. 在不同并发、不同输入长度上交叉验证，不能只对一个点过拟合；
4. 将最终参数、vLLM 版本、GPU、模型和 workload 一起保存。

## 限制

- 不是 kernel/cycle 级模拟，未精确建模 FlashAttention、CUDA Graph、 sampler、detokenizer、prefix cache、speculative decoding 和具体通信拓扑。
- PP 当前按最大 stage 的串行步骤时间估计，不含完整 pipeline bubble；DP 聚合吞吐尚未实现。
- Poisson 到达 + 离散事件调度，不是对真实 HTTP 服务的网络模拟。
- 小模型的 CPU/launch overhead 和量化 kernel 的性能尤其需要实测校准。

## 测试

```bash
cd vllm_perf_sim
PYTHONPATH=..:. python3 -m unittest discover -s tests -v
```
