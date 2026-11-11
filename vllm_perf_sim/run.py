#!/usr/bin/env python3
"""Run the vLLM performance simulator from the aiinfra_tools repository root."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "vllm_perf_sim")]

from vllm_perf_sim.cli import main

if __name__ == "__main__":
    main()
