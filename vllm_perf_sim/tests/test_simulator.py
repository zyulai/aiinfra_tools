import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vllm_perf_sim"))

from parallel_tool.config_parser import parse_config
from vllm_perf_sim.hardware import get_hardware
from vllm_perf_sim.model import ModelGeometry
from vllm_perf_sim.timing import TimingModel
from vllm_perf_sim.simulator import Simulator, SimRequest, make_workload
from vllm_perf_sim.metrics import calculate_metrics, percentile, render_report
from vllm_perf_sim.calibration import calibrate


class PerfSimTests(unittest.TestCase):
    def setUp(self):
        config = parse_config(str(ROOT / "test_config.json"))
        self.geometry = ModelGeometry(config)
        self.hardware = get_hardware("h100-sxm")
        self.timing = TimingModel(self.geometry, self.hardware)

    def test_llama_weight_and_kv_geometry(self):
        self.assertAlmostEqual(self.geometry.weight_bytes_per_gpu / 1024**3, 15.46, places=2)
        self.assertEqual(self.geometry.kv_bytes_per_token_per_gpu, 131072)

    def test_single_request_token_times(self):
        requests = [SimRequest(0, 0, 32, 4)]
        result = Simulator(self.geometry, self.timing, max_num_batched_tokens=8192).run(requests)
        req = result.requests[0]
        self.assertEqual(len(req.token_times), 4)
        self.assertAlmostEqual(req.first_token_s, req.token_times[0])
        self.assertAlmostEqual(req.end_s, req.token_times[-1])
        self.assertEqual(result.steps, 4)

    def test_chunked_prefill_and_decode_priority(self):
        requests = [SimRequest(0, 0, 100, 3), SimRequest(1, 0, 20, 2)]
        sim = Simulator(self.geometry, self.timing, max_num_batched_tokens=40, max_num_seqs=2)
        result = sim.run(requests)
        self.assertGreater(result.prefill_steps, 2)
        for req in result.requests:
            self.assertEqual(req.prefilled, req.input_len)
            self.assertEqual(req.decoded, req.output_len)
            self.assertEqual(len(req.token_times), req.output_len)

    def test_kv_capacity_blocks_concurrent_admission(self):
        # 128 KiB/token/GPU, so only ~2000 tokens fit.
        requests = [SimRequest(0, 0, 1500, 2), SimRequest(1, 0, 1500, 2)]
        sim = Simulator(self.geometry, self.timing, kv_cache_bytes=2000 * 131072)
        result = sim.run(requests)
        self.assertEqual(result.max_concurrency, 1)
        self.assertEqual(len(result.requests), 2)

    def test_poisson_arrivals_are_reproducible(self):
        a = make_workload(20, 32, 8, request_rate=10, seed=42)
        b = make_workload(20, 32, 8, request_rate=10, seed=42)
        self.assertEqual([r.arrival_s for r in a], [r.arrival_s for r in b])
        self.assertTrue(all(b.arrival_s >= a.arrival_s for a, b in zip(a, a[1:])))

    def test_metrics_conservation_and_report(self):
        requests = make_workload(10, 64, 8, request_rate=float("inf"))
        sim = Simulator(self.geometry, self.timing)
        metrics = calculate_metrics(sim.run(requests))
        self.assertEqual(metrics["total_input_tokens"], 640)
        self.assertEqual(metrics["total_output_tokens"], 80)
        self.assertAlmostEqual(metrics["output_throughput"], 80 / metrics["duration"])
        self.assertGreater(metrics["mean_e2el_ms"], metrics["mean_ttft_ms"])
        self.assertIn("Output token throughput (tok/s):", render_report(metrics))

    def test_percentile_linear_interpolation(self):
        self.assertEqual(percentile([0, 10], 50), 5)
        self.assertEqual(percentile([1], 99), 1)

    def test_moe_active_flops_less_than_all_experts(self):
        cfg = parse_config(str(ROOT / "test_moe_config.json"))
        moe = ModelGeometry(cfg, tp=1)
        active = moe.moe_active_flops(1)
        total = 2 * 3 * cfg.hidden_size * cfg.moe_intermediate_size * cfg.num_experts * cfg.num_hidden_layers
        self.assertLess(active, total)
        self.assertGreater(active, 0)

    def test_calibration_is_explicit(self):
        result = calibrate({"duration": 2, "request_throughput": 5,
                            "output_throughput": 100, "mean_tpot_ms": 20},
                           1, 10, 0.45, 0.75)
        self.assertEqual(result["duration_factor"], 2)
        self.assertEqual(result["tpot_factor"], 2)
        self.assertIn("warning", result)


if __name__ == "__main__":
    unittest.main()
