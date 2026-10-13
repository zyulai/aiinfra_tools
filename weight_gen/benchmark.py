"""
Benchmark script template — generates a ready-to-run benchmark script
that can test vLLM / SGLang / HuggingFace inference speed on the random-weight model.
"""
import os

# Use a placeholder approach to avoid brace conflicts
BENCHMARK_TEMPLATE = '''#!/usr/bin/env python3
"""
Auto-generated benchmark script — test inference speed on random-weight model.

Usage:
  python __SCRIPT_NAME__ --model-dir __MODEL_DIR__ --backend __BACKEND__ [options]

Backends:
  vllm    — use vLLM for inference (recommended for serving)
  sglang  — use SGLang for inference
  hf      — use HuggingFace transformers (baseline)
"""
import argparse
import time
import json
import sys


def benchmark_vllm(model_dir, tp, pp, num_prompts, max_tokens, dtype):
    """Benchmark with vLLM."""
    from vllm import LLM, SamplingParams
    
    llm = LLM(
        model=model_dir,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        dtype=dtype,
        trust_remote_code=True,
        enforce_eager=False,
        max_model_len=max_tokens + 1024,
    )
    
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
    )
    
    prompts = ["Hello, this is a test prompt for speed benchmarking. " * 10] * num_prompts
    
    # Warmup
    print("Warming up...")
    llm.generate(prompts[:1], sampling)
    
    # Benchmark
    print(f"Running {num_prompts} prompts x {max_tokens} tokens...")
    start = time.time()
    outputs = llm.generate(prompts, sampling)
    elapsed = time.time() - start
    
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / elapsed
    
    print(f"\\n=== vLLM Benchmark Results ===")
    print(f"  Model:        {model_dir}")
    print(f"  TP={tp}, PP={pp}, dtype={dtype}")
    print(f"  Prompts:      {num_prompts}")
    print(f"  Max tokens:   {max_tokens}")
    print(f"  Total time:   {elapsed:.2f}s")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Throughput:   {throughput:.1f} tok/s")
    
    return {"throughput": throughput, "elapsed": elapsed, "total_tokens": total_tokens}


def benchmark_sglang(model_dir, tp, pp, num_prompts, max_tokens, dtype):
    """Benchmark with SGLang."""
    import subprocess
    import requests
    
    server_cmd = [
        "python", "-m", "sglang.srt.server",
        "--model-path", model_dir,
        "--tp", str(tp),
        "--pp", str(pp),
        "--dtype", dtype,
        "--trust-remote-code",
        "--port", "30000",
    ]
    print(f"Starting SGLang server...")
    server = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    time.sleep(15)  # Wait for server to start
    
    try:
        url = "http://localhost:30000/generate"
        prompt = "Hello, this is a test. " * 10
        
        # Warmup
        requests.post(url, json={
            "text": prompt,
            "sampling_params": {"max_new_tokens": 16, "temperature": 0}
        })
        
        # Benchmark
        start = time.time()
        total_tokens = 0
        for _ in range(num_prompts):
            r = requests.post(url, json={
                "text": prompt,
                "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0}
            }).json()
            total_tokens += len(r.get("text", "").split())
        
        elapsed = time.time() - start
        throughput = total_tokens / elapsed
        
        print(f"\\n=== SGLang Benchmark Results ===")
        print(f"  Throughput: {throughput:.1f} tok/s")
        
        return {"throughput": throughput, "elapsed": elapsed}
    finally:
        server.terminate()
        server.wait()


def benchmark_hf(model_dir, tp, pp, num_prompts, max_tokens, dtype):
    """Benchmark with HuggingFace transformers."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    device_map = "auto"
    torch_dtype = getattr(torch, dtype)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch_dtype, device_map=device_map,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    prompt = "Hello, this is a test prompt for speed benchmarking. " * 10
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    
    # Benchmark
    print(f"Running {num_prompts} prompts x {max_tokens} tokens...")
    start = time.time()
    total_tokens = 0
    for _ in range(num_prompts):
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        total_tokens += out.shape[1] - inputs.input_ids.shape[1]
    elapsed = time.time() - start
    throughput = total_tokens / elapsed
    
    print(f"\\n=== HuggingFace Benchmark Results ===")
    print(f"  Throughput: {throughput:.1f} tok/s")
    
    return {"throughput": throughput, "elapsed": elapsed, "total_tokens": total_tokens}


def main():
    parser = argparse.ArgumentParser(description="Benchmark random-weight model inference speed")
    parser.add_argument("--model-dir", required=True, help="Path to model directory")
    parser.add_argument("--backend", choices=["vllm", "sglang", "hf"], default="__BACKEND__")
    parser.add_argument("--tp", type=int, default=__TP__, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=__PP__, help="Pipeline parallel size")
    parser.add_argument("--num-prompts", type=int, default=8, help="Number of prompts to generate")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max new tokens per prompt")
    parser.add_argument("--dtype", default="__DTYPE__", help="Data type")
    parser.add_argument("--output", type=str, help="Save results to JSON")
    
    args = parser.parse_args()
    
    print(f"Backend: {args.backend}")
    print(f"Model:   {args.model_dir}")
    print(f"TP={args.tp}, PP={args.pp}, dtype={args.dtype}")
    print()
    
    if args.backend == "vllm":
        results = benchmark_vllm(args.model_dir, args.tp, args.pp,
                                  args.num_prompts, args.max_tokens, args.dtype)
    elif args.backend == "sglang":
        results = benchmark_sglang(args.model_dir, args.tp, args.pp,
                                    args.num_prompts, args.max_tokens, args.dtype)
    elif args.backend == "hf":
        results = benchmark_hf(args.model_dir, args.tp, args.pp,
                                args.num_prompts, args.max_tokens, args.dtype)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
'''


def generate_benchmark_script(
    output_dir: str,
    model_dir: str,
    backend: str = "vllm",
    tp: int = 1,
    pp: int = 1,
    dtype: str = "bfloat16",
) -> str:
    """Generate a ready-to-run benchmark script."""
    script_name = f"benchmark_{backend}_tp{tp}_pp{pp}.py"
    script_path = os.path.join(output_dir, script_name)
    
    content = BENCHMARK_TEMPLATE
    content = content.replace("__SCRIPT_NAME__", script_name)
    content = content.replace("__MODEL_DIR__", model_dir)
    content = content.replace("__BACKEND__", backend)
    content = content.replace("__TP__", str(tp))
    content = content.replace("__PP__", str(pp))
    content = content.replace("__DTYPE__", dtype)
    
    with open(script_path, "w") as f:
        f.write(content)
    os.chmod(script_path, 0o755)
    
    return script_path
