"""
Weight writer — generate random weights and save as safetensors files.
Supports multi-file sharding with index.json.
"""
import json
import os
import math
from typing import Optional
import torch
from safetensors.torch import save_file

from .weight_shaper import WeightEntry, compute_total_bytes


def dtype_to_torch(dtype: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping.get(dtype, torch.bfloat16)


def generate_random_weights(
    entries: list[WeightEntry],
    seed: int = 42,
    init_strategy: str = "normal",
) -> dict[str, torch.Tensor]:
    """
    Generate random weight tensors for all entries.
    
    init_strategy:
      "normal"   — N(0, 0.02) like model initialization
      "uniform"  — uniform[-0.02, 0.02]
      "zeros"    — all zeros (smallest, just for shape testing)
      "small"    — small random values for numerical stability
    """
    torch.manual_seed(seed)
    tensors = {}
    
    for e in entries:
        dt = dtype_to_torch(e.dtype)
        
        if init_strategy == "normal":
            t = torch.randn(e.shape, dtype=dt) * 0.02
        elif init_strategy == "uniform":
            t = torch.empty(e.shape, dtype=dt).uniform_(-0.02, 0.02)
        elif init_strategy == "zeros":
            t = torch.zeros(e.shape, dtype=dt)
        elif init_strategy == "small":
            # Small values to avoid NaN in attention
            t = torch.randn(e.shape, dtype=dt) * 0.005
        else:
            t = torch.randn(e.shape, dtype=dt) * 0.02
        
        tensors[e.name] = t.contiguous()
    
    return tensors


def shard_weights_to_files(
    tensors: dict[str, torch.Tensor],
    output_dir: str,
    max_file_size_gb: float = 5.0,
    file_prefix: str = "model",
) -> tuple[list[str], dict]:
    """
    Save tensors into safetensors files, splitting at max_file_size_gb.
    Returns (list of filenames, index_dict).
    """
    max_bytes = int(max_file_size_gb * 1024**3)
    index = {"metadata": {"total_size": 0}, "weight_map": {}}
    
    current_batch = {}
    current_size = 0
    files_created = []
    file_idx = 0
    
    def flush_batch():
        nonlocal current_batch, current_size, file_idx
        if not current_batch:
            return
        fname = f"{file_prefix}-{file_idx:05d}-of-XXXXX.safetensors"
        # We'll rename later once we know total file count
        actual_fname = f"{file_prefix}-{file_idx:05d}.safetensors"
        fpath = os.path.join(output_dir, actual_fname)
        save_file(current_batch, fpath, metadata={"format": "pt"})
        files_created.append(actual_fname)
        for name in current_batch:
            index["weight_map"][name] = actual_fname
            index["metadata"]["total_size"] += current_batch[name].numel() * current_batch[name].element_size()
        file_idx += 1
        current_batch = {}
        current_size = 0
    
    for name, tensor in tensors.items():
        t_size = tensor.numel() * tensor.element_size()
        
        if current_size + t_size > max_bytes and current_batch:
            flush_batch()
        
        current_batch[name] = tensor
        current_size += t_size
    
    flush_batch()
    
    # Now rename files with proper "of-N" naming
    total_files = len(files_created)
    final_files = []
    for i, old_fname in enumerate(files_created):
        new_fname = f"{file_prefix}-{i:05d}-of-{total_files:05d}.safetensors"
        old_path = os.path.join(output_dir, old_fname)
        new_path = os.path.join(output_dir, new_fname)
        os.rename(old_path, new_path)
        final_files.append(new_fname)
        # Update index
        for name in index["weight_map"]:
            if index["weight_map"][name] == old_fname:
                index["weight_map"][name] = new_fname
    
    index["metadata"]["total_size"] = index["metadata"]["total_size"]
    return final_files, index


def save_single_file(
    tensors: dict[str, torch.Tensor],
    output_dir: str,
    filename: str = "model.safetensors",
) -> tuple[str, dict]:
    """Save all tensors to a single safetensors file."""
    fpath = os.path.join(output_dir, filename)
    save_file(tensors, fpath, metadata={"format": "pt"})
    index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
        "weight_map": {name: filename for name in tensors},
    }
    return filename, index


def write_index_json(output_dir: str, index: dict, filename: str = "model.safetensors.index.json"):
    """Write the safetensors index.json file."""
    index_path = os.path.join(output_dir, filename)
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
