# Parallel Tool

Analyze how different parallel strategies (TP / PP / CP / EP / DP / SP) shard **every kernel** of any HuggingFace model.

Given an HF `config.json`, freely choose parallel combinations and get a detailed table showing:
- **Weight shape** per GPU (original → sharded)
- **Activation shape** per GPU (original → sharded)
- **Communication primitives** needed in forward / backward
- **Memory** per GPU
- **Pipeline stage assignment** for each layer
- **Framework implementation references** from SGLang / vLLm / Megatron-LM source code

## Directory Structure

```
parallel/
├── parallel_tool/             # ── Parallel analysis tool ──
│   ├── __init__.py            # Package init
│   ├── __main__.py            # python -m entry point
│   ├── main.py                # CLI: --config, --tp, --pp, --cp, --ep, --dp, --sp, --interactive, --compare, --json
│   ├── config_parser.py      # Parse HF config.json → ModelConfig
│   ├── model_kernels.py      # Build kernel list from ModelConfig
│   ├── parallel_engine.py    # ParallelConfig + compute_sharding() per kernel
│   ├── table_renderer.py     # Rich table rendering (with plain-text fallback)
│   └── framework_refs.py     # How SGLang/vLLM/Megatron implement each strategy
├── weight_gen/                # ── Random weight generator ──
│   ├── __init__.py            # Package init
│   ├── __main__.py            # python -m entry point
│   ├── main.py               # CLI: --config, --tp, --pp, --source-dir, --num-layers, --all-ranks, --benchmark
│   ├── weight_shaper.py      # Build weight name→shape mapping with TP/EP sharding
│   ├── weight_writer.py      # Generate random tensors → safetensors (multi-file + index.json)
│   ├── model_cloner.py       # Copy tokenizer/config files from local dir or HF Hub
│   └── benchmark.py          # Generate ready-to-run vLLM/SGLang/HF benchmark script
├── sglang/                    # Source (shallow clone)
├── vllm/                      # Source (shallow clone)
├── Megatron-LM/               # Source (shallow clone)
├── test_config.json          # Llama-3-8B-like test config
└── test_moe_config.json      # Qwen2-MoE test config
```

## Tool 2: Weight Generator (weight_gen)

Generate **random-weight models** with correct sharded shapes for speed benchmarking.
No need to download a full multi-GB model — just generate locally and test inference speed.

### Quick Start

```bash
# Full model, TP=4 pre-sharded random weights
python -m weight_gen.main --config config.json --tp 4 -o ./rand_model

# Truncated model (only 8 layers, for quick test)
python -m weight_gen.main --config config.json --tp 8 --num-layers 8 -o ./rand

# Copy tokenizer from local model dir
python -m weight_gen.main --config config.json --source-dir /path/to/model --tp 4 -o ./rand

# Download tokenizer from HuggingFace Hub
python -m weight_gen.main --config config.json --model-id meta-llama/Llama-3-8B --tp 4 -o ./rand

# Generate for ALL TP ranks (separate subdirs per rank)
python -m weight_gen.main --config config.json --tp 4 --all-ranks -o ./rand_tp4

# Generate benchmark script too
python -m weight_gen.main --config config.json --tp 4 -o ./rand --benchmark vllm

# Use fp16, zeros init (smallest possible)
python -m weight_gen.main --config config.json --tp 8 --dtype float16 --init zeros -o ./rand

# MoE model with TP+EP
python -m weight_gen.main --config moe_config.json --tp 2 --ep 4 -o ./rand_moe
```

### Options

| Option | Description |
|--------|-------------|
| `--config` | Path to HF config.json |
| `-o / --output` | Output directory |
| `--tp` | Tensor parallel size (pre-shard weights) |
| `--pp` | Pipeline parallel size |
| `--ep` | Expert parallel size (MoE) |
| `--source-dir` | Local model dir to copy tokenizer from |
| `--model-id` | HF Hub model ID to download tokenizer from |
| `--num-layers` | Truncate to N layers (speed test) |
| `--dtype` | bfloat16 / float16 / float32 |
| `--init` | normal / uniform / zeros / small |
| `--all-ranks` | Generate separate dirs for each TP rank |
| `--benchmark` | Generate vllm/sglang/hf benchmark script |
| `--max-file-gb` | Max safetensors file size (default 5GB) |

## Quick Start

```bash
# Basic: TP=4 on a model config
python -m parallel_tool.main --config path/to/config.json --tp 4

# 3D parallel: TP=4, PP=2, SP=on (8 GPUs)
python -m parallel_tool.main --config config.json --tp 4 --pp 2 --sp

# Context parallel + TP
python -m parallel_tool.main --config config.json --tp 4 --cp 2 --sp

# Expert parallel for MoE models
python -m parallel_tool.main --config moe_config.json --tp 2 --ep 4

# Full 5D: TP=4, PP=2, CP=2, DP=2, SP (32 GPUs)
python -m parallel_tool.main --config config.json --tp 4 --pp 2 --cp 2 --dp 2 --sp

# Interactive mode (choose strategies on the fly)
python -m parallel_tool.main --config config.json --interactive

# Compare multiple presets side by side
python -m parallel_tool.main --config config.json --compare tp2,tp4,tp8

# Export results to JSON
python -m parallel_tool.main --config config.json --tp 8 --json results.json

# Plain text tables (no rich dependency)
python -m parallel_tool.main --config config.json --tp 4 --plain

# Show only framework references
python -m parallel_tool.main --config config.json --tp 4 --refs-only

# List available presets
python -m parallel_tool.main --list-presets
```

## Interactive Sharding Interface

```bash
python run.py --config test_config.json
python run.py --config /path/to/model_directory
# Same interactive session through the package entry point:
python -m parallel_tool --config test_moe_config.json --interactive
```

The session keeps the current parallel configuration, precision, and view filters.
Choose **4** to edit dimensions, **5** to select a preset within a GPU budget,
and **6** to filter by category, layer, or pipeline stage and enable details.
Choose **1** to analyze, **3** to compare presets, or **2** to generate weights.
The single-GPU baseline is included in presets. GPU budget is a preset filter;
weight estimates alone do not establish whether a model fits on a GPU.

Tables adapt to terminal width; use `--width 80` to override it. Compact output
shows weight shapes and split dimensions. Activation shapes, forward/backward
communication, tensor names, and framework references are available on demand.
JSON export always includes all kernels, regardless of display filters.

```bash
# Inspect one pipeline stage and attention layer with full details:
python -m parallel_tool --config test_config.json --tp 4 --pp 2 \
  --stage 0 --layer 0 --category attention --details --width 80

# Include every layer and export the complete analysis:
python -m parallel_tool --config test_config.json --tp 4 --all-layers --json results.json

# Compare presets or explicit combinations:
python -m parallel_tool --config test_config.json --compare 1gpu,tp4,tp4_pp2
python -m parallel_tool --config test_config.json --compare 'tp=4,pp=2;tp=8'

# Change estimate precision and append framework references:
python -m parallel_tool --config test_config.json --tp 4 --dtype float32 --refs
```

**Memory scope:** weight bytes only, excluding activations, KV cache, gradients,
and workspace. Pipeline summaries report each stage and the maximum weight size
per GPU. Expert shapes describe one expert; byte totals include every expert
resident on the rank. Linear analysis shapes use `[in, out]`, while HF checkpoint
weights use `[out, in]`. The generator writes all PP layers for the selected
TP/EP shard; its file size is therefore different from a single PP stage's memory.

Generating a truncated model keeps the loaded model unchanged and writes the
selected layer count into the generated configuration. The generation plan is
shown after selecting precision and layers, before allocating tensors.

Run regression checks with:

```bash
python -m unittest discover -s tests -v
```

## Supported Parallel Strategies

| Strategy | Flag | What it shards | Comm Pattern |
|----------|------|----------------|--------------|
| **TP** (Tensor Parallel) | `--tp N` | Weights: column/row/head/vocab split | all_reduce / reduce_scatter |
| **PP** (Pipeline Parallel) | `--pp N` | Layers across stages | send/recv (p2p) |
| **CP** (Context Parallel) | `--cp N` | Sequence dimension | ring attention |
| **EP** (Expert Parallel) | `--ep N` | MoE experts | all_to_all |
| **DP** (Data Parallel) | `--dp N` | Full replica | all_reduce (grads) |
| **SP** (Sequence Parallel) | `--sp` | Sequence in non-attn regions | scatter/gather |

## Kernels Analyzed

Per transformer layer (Llama / Qwen / Mistral style):
- **Embedding**: token_embedding (vocab parallel)
- **Attention**: q_proj (head split), k_proj, v_proj, core_attention, o_proj (row parallel)
- **MLP**: gate_proj (column), up_proj (column), down_proj (row)
- **Norm**: input_norm, post_attn_norm, final_norm (replicated / SP-affected)
- **MoE**: router, expert_gate, expert_up, expert_down (expert parallel + optional TP)
- **LM Head**: lm_head (vocab parallel)

## Framework References

The tool cross-references how each strategy is implemented in:

- **SGLang**: `sglang/srt/layers/` — QKVParallelLinear, RowParallelLinear, VocabParallelEmbedding
- **vLLM**: `vllm/distributed/parallel_state.py` — TP/PP/CP/EP/DP groups; `vllm/model_executor/layers/` — parallel linear layers
- **Megatron-LM**: `megatron/core/tensor_parallel/` — ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding, mappings.py

## Validation

The tool checks divisibility constraints:
- All parallel sizes must be positive integers; invalid configurations are rejected
- TP must divide num_attention_heads, num_key_value_heads, vocab_size, and the FFN intermediate dimension (moe_intermediate_size for experts)
- PP must not exceed the layer count; uneven stage sizes produce a warning
- EP requires a MoE model and divides num_experts
- SP requires TP > 1
- CP without TP produces a backend-support warning

## Output Format

### Rich tables (default, requires `pip install rich`):
Colorized tables with per-kernel rows showing weight shape transitions, communication ops, memory.

### Plain text (`--plain`):
ASCII tables for environments without rich.

### JSON (`--json file.json`):
Complete structured output with all kernel sharding details.

### Comparison (`--compare`):
Side-by-side weight shapes across multiple parallel strategies.
