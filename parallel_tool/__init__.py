"""
Parallel Tool — Analyze how different parallel strategies shard every kernel of a model.

Given an HF model config, freely choose TP / PP / CP / EP / DP / SP strategies
and get a detailed table showing per-kernel weight & activation sharding,
communication primitives, and implementation references from SGLang / vLLM / Megatron-LM.
"""
__version__ = "1.0.0"
