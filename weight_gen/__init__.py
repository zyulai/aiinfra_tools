"""
Weight Generator — Generate random-weight models with correct sharded shapes.

Given an HF config + parallel strategy, produce a complete model directory with:
  - Random weights in safetensors format (full or TP-sharded per rank)
  - Copied tokenizer / config / generation files from the original model
  - model.safetensors.index.json for multi-file models

Purpose: benchmark inference speed without downloading a full multi-GB model.
"""
__version__ = "1.0.0"
