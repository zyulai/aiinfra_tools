"""
Model cloner — pull tokenizer, config, and other auxiliary files
from the original model source (local path or HuggingFace Hub).
"""
import os
import shutil
import json
from typing import Optional


# Files to copy from the original model (excluding weights)
AUXILIARY_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
    "chat_template.json",
    "chat_template.jinja",
    "added_tokens.json",
    "config.json",
    "model.safetensors.index.json",  # We'll regenerate this
    "modellingconfiguration.json",
    "preprocessor_config.json",
    "processor_config.json",
    "configuration.json",
]

# Image/multimodal files
MULTIMODAL_FILES = [
    "preprocessor_config.json",
    "processor_config.json",
]


def clone_aux_files(
    source_dir: str,
    output_dir: str,
    skip_weights: bool = True,
    verbose: bool = True,
) -> list[str]:
    """
    Copy auxiliary files (tokenizer, configs) from source model dir.
    Returns list of copied file names.
    """
    copied = []
    
    if not os.path.isdir(source_dir):
        if verbose:
            print(f"  Source dir not found: {source_dir}")
        return copied
    
    for fname in os.listdir(source_dir):
        full_path = os.path.join(source_dir, fname)
        
        if not os.path.isfile(full_path):
            continue
        
        # Skip weight files if requested
        if skip_weights:
            if fname.endswith(".safetensors") or fname.endswith(".bin") or fname.endswith(".pt"):
                continue
            if fname.endswith(".gguf"):
                continue
        
        # Copy if it's an auxiliary file or any .json/.txt config
        if fname in AUXILIARY_FILES or fname.endswith((".json", ".txt", ".jinja")) or fname in ("tokenizer.model",):
            dest = os.path.join(output_dir, fname)
            shutil.copy2(full_path, dest)
            copied.append(fname)
            if verbose:
                print(f"  ✓ Copied {fname}")
    
    return copied


def download_aux_from_hub(
    model_id: str,
    output_dir: str,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> list[str]:
    """
    Download auxiliary files from HuggingFace Hub.
    Only downloads tokenizer/config files, not weights.
    """
    from huggingface_hub import snapshot_download
    
    allow_patterns = [
        "tokenizer*", "vocab*", "merges*", "special_tokens*",
        "generation_config*", "config.json", "chat_template*",
        "added_tokens*", "preprocessor_config*", "processor_config*",
        "*.txt", "*.jinja",
    ]
    
    if verbose:
        print(f"  Downloading aux files from HF Hub: {model_id}")
    
    local_path = snapshot_download(
        model_id,
        allow_patterns=allow_patterns,
        cache_dir=cache_dir,
    )
    
    # Now copy to output dir
    return clone_aux_files(local_path, output_dir, skip_weights=True, verbose=verbose)


def write_config(
    output_dir: str,
    config: dict,
    overrides: Optional[dict] = None,
    verbose: bool = True,
):
    """
    Write a modified config.json to the output dir.
    Overrides can adjust num_hidden_layers etc. for truncation.
    """
    cfg = dict(config)
    if overrides:
        cfg.update(overrides)
    
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    if verbose:
        print(f"  ✓ Wrote config.json")
        if overrides:
            print(f"    Overrides: {overrides}")
