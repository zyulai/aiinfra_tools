"""Regression checks for sharding summaries and terminal interaction."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from parallel_tool.config_parser import ModelConfig, parse_config
from parallel_tool.parallel_engine import ParallelConfig
from parallel_tool.reporting import (
    analysis_as_dict, analyze_model, select_results, summarize_memory,
)
from parallel_tool.table_renderer import RICH_AVAILABLE, render_comparison, render_table
from weight_gen.weight_shaper import build_weight_map, compute_total_bytes

ROOT = Path(__file__).resolve().parents[1]


def generated_weight_bytes(cfg, pc, dtype="bfloat16"):
    entries = build_weight_map(cfg, pc, dtype=dtype)
    return compute_total_bytes(entries)


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.dense = parse_config(str(ROOT / "test_config.json"))
        self.moe = parse_config(str(ROOT / "test_moe_config.json"))

    def test_dense_summary_matches_checkpoint_shapes(self):
        for tp in (1, 2, 4, 8):
            with self.subTest(tp=tp):
                pc = ParallelConfig(tp=tp)
                summary = summarize_memory(self.dense, pc, analyze_model(self.dense, pc))
                self.assertEqual(summary.peak_weight_bytes, generated_weight_bytes(self.dense, pc))

    def test_moe_summary_counts_every_resident_expert_with_or_without_ep(self):
        for tp, ep in ((1, 1), (2, 1), (4, 1), (1, 3), (2, 4), (4, 5)):
            with self.subTest(tp=tp, ep=ep):
                pc = ParallelConfig(tp=tp, ep=ep)
                summary = summarize_memory(self.moe, pc, analyze_model(self.moe, pc))
                self.assertEqual(summary.peak_weight_bytes, generated_weight_bytes(self.moe, pc))
                self.assertEqual(summary.full_weight_bytes, generated_weight_bytes(self.moe, ParallelConfig()))

    def test_bias_weights_are_included(self):
        cfg = replace(self.dense, attention_bias=True, mlp_bias=True)
        pc = ParallelConfig(tp=4)
        summary = summarize_memory(cfg, pc, analyze_model(cfg, pc))
        self.assertEqual(summary.peak_weight_bytes, generated_weight_bytes(cfg, pc))

    def test_pipeline_stages_own_global_weights_and_partition_layers(self):
        cfg = replace(self.dense, num_hidden_layers=5)
        pc = ParallelConfig(tp=4, pp=2)
        results = analyze_model(cfg, pc)
        summary = summarize_memory(cfg, pc, results)
        entries = build_weight_map(cfg, pc)
        stage_bytes = [0, 0]
        for entry in entries:
            if ".layers." in entry.name:
                layer = int(entry.name.split(".layers.")[1].split(".")[0])
                stage = 0 if layer < 3 else 1
            else:
                stage = 0 if "embed_tokens" in entry.name else 1
            stage_bytes[stage] += entry.numel * 2
        self.assertEqual(summary.stage_weight_bytes, tuple(stage_bytes))
        self.assertEqual(sum(summary.stage_weight_bytes), generated_weight_bytes(cfg, pc))
        self.assertLess(summary.peak_weight_bytes, sum(summary.stage_weight_bytes))
        stages = {result.kernel.name: result.pp_stage for result in results}
        self.assertEqual(stages["token_embedding"], 0)
        self.assertEqual(stages["final_norm"], 1)
        self.assertEqual(stages["lm_head"], 1)
        self.assertEqual(stages["layer_2_q_proj"], 0)
        self.assertEqual(stages["layer_3_q_proj"], 1)

    def test_pipeline_boundaries_cover_even_uneven_and_one_layer_stages(self):
        for layers, expected in ((6, [0, 0, 1, 1, 2, 2]),
                                 (7, [0, 0, 0, 1, 1, 2, 2]),
                                 (3, [0, 1, 2])):
            with self.subTest(layers=layers):
                cfg = replace(self.dense, num_hidden_layers=layers)
                results = analyze_model(cfg, ParallelConfig(pp=3))
                actual = [result.pp_stage for result in results
                          if result.kernel.name.endswith("_q_proj")]
                self.assertEqual(actual, expected)

    def test_weightless_attention_has_zero_memory(self):
        for result in analyze_model(self.dense, ParallelConfig(tp=4)):
            if "core_attention" in result.kernel.name:
                self.assertEqual(result.memory_per_gpu_bytes, 0)
                self.assertEqual(result.total_memory_bytes, 0)

    def test_float32_doubles_weight_bytes(self):
        pc = ParallelConfig(tp=4, pp=2)
        bf16 = summarize_memory(self.dense, pc, analyze_model(self.dense, pc))
        fp32 = summarize_memory(self.dense, pc, analyze_model(self.dense, pc, "float32"))
        self.assertEqual(fp32.stage_weight_bytes, tuple(value * 2 for value in bf16.stage_weight_bytes))

    def test_tied_embeddings_are_not_counted_twice(self):
        cfg = replace(self.dense, tie_word_embeddings=True)
        pc = ParallelConfig(tp=4)
        summary = summarize_memory(cfg, pc, analyze_model(cfg, pc))
        self.assertEqual(summary.peak_weight_bytes, generated_weight_bytes(cfg, pc))


class MenuCacheTests(unittest.TestCase):
    @unittest.skipUnless(RICH_AVAILABLE, "Interactive runner requires rich")
    def test_repeated_estimates_reuse_cache_and_precision_has_its_own_entry(self):
        import run
        cfg = parse_config(str(ROOT / "test_config.json"))
        cache = {}
        with patch("run.analyze_model", wraps=analyze_model) as analyze:
            bf16 = run.estimate_weight_size(cfg, tp=4, pp=2, cache=cache)
            self.assertEqual(run.estimate_weight_size(cfg, tp=4, pp=2, cache=cache), bf16)
            fp32 = run.estimate_weight_size(cfg, tp=4, pp=2, dtype="float32", cache=cache)
            self.assertEqual(analyze.call_count, 2)
        self.assertEqual(fp32, bf16 * 2)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = parse_config(str(ROOT / "test_config.json"))

    def test_nonpositive_dimensions_are_rejected(self):
        for dimension in ("tp", "pp", "cp", "ep", "dp"):
            for value in (0, -1):
                with self.subTest(dimension=dimension, value=value):
                    pc = ParallelConfig(**{dimension: value})
                    self.assertTrue(pc.validate(self.cfg, include_warnings=False))
                    with self.assertRaises(ValueError):
                        analyze_model(self.cfg, pc)

    def test_uneven_pipeline_is_a_warning(self):
        pc = ParallelConfig(pp=3)
        self.assertFalse(pc.validate(self.cfg, include_warnings=False))
        self.assertTrue(pc.validation_warnings(self.cfg))
        self.assertEqual(len(summarize_memory(self.cfg, pc, analyze_model(self.cfg, pc)).stage_weight_bytes), 3)

    def test_pipeline_cannot_have_empty_stages(self):
        self.assertTrue(ParallelConfig(pp=33).validate(self.cfg, include_warnings=False))

    def test_moe_validation_uses_expert_intermediate_dimension(self):
        cfg = replace(self.cfg, num_experts=8, intermediate_size=14336, moe_intermediate_size=1409)
        self.assertTrue(ParallelConfig(tp=4).validate(cfg, include_warnings=False))


class RenderingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = parse_config(str(ROOT / "test_config.json"))
        self.pc = ParallelConfig(tp=4, pp=2)
        self.results = analyze_model(self.cfg, self.pc)

    def test_rendering_has_no_stdout_side_effects_and_respects_width(self):
        for use_rich in (False, True):
            for width in (48, 80, 120):
                with self.subTest(rich=use_rich, width=width):
                    captured = io.StringIO()
                    with contextlib.redirect_stdout(captured):
                        text = render_table(self.cfg, self.pc, self.results, use_rich=use_rich, width=width)
                    self.assertEqual(captured.getvalue(), "")
                    self.assertIn("Kernel Sharding", text)
                    if use_rich and RICH_AVAILABLE:
                        from rich.cells import cell_len
                        self.assertTrue(all(cell_len(line) <= width for line in text.splitlines()))
                    else:
                        self.assertTrue(all(len(line) <= width for line in text.splitlines()))

    def test_layer_and_category_filters_select_exact_rows(self):
        selected = select_results(self.cfg, self.pc, self.results, layer_idx=17, category="attention")
        self.assertEqual(len(selected), 5)
        self.assertTrue(all(result.kernel.layer_idx == 17 for result in selected))
        self.assertTrue(all(result.kernel.category == "attention" for result in selected))
        self.assertTrue(all(result.pp_stage == 1 for result in selected))

    def test_stage_filter_includes_only_its_own_global_modules(self):
        selected = select_results(self.cfg, self.pc, self.results, pp_stage=1, show_all=True)
        names = {result.kernel.name for result in selected}
        self.assertIn("lm_head", names)
        self.assertNotIn("token_embedding", names)
        self.assertTrue(all(result.pp_stage == 1 for result in selected))

    def test_full_view_and_json_keep_all_layers(self):
        self.assertLess(len(select_results(self.cfg, self.pc, self.results)), len(self.results))
        self.assertEqual(select_results(self.cfg, self.pc, self.results, show_all=True), self.results)
        exported = analysis_as_dict(self.cfg, self.pc, self.results)
        self.assertEqual(len(exported["kernels"]), len(self.results))
        self.assertEqual(len(exported["memory_summary"]["stage_weight_bytes"]), 2)

    def test_comparison_analyzes_each_plan_once(self):
        plans = [ParallelConfig(), ParallelConfig(tp=4), self.pc]
        captured = io.StringIO()
        with patch("parallel_tool.table_renderer.analyze_model", wraps=analyze_model) as analyze:
            with contextlib.redirect_stdout(captured):
                text = render_comparison(self.cfg, plans, width=80)
            self.assertEqual(analyze.call_count, len(plans))
        self.assertEqual(captured.getvalue(), "")
        self.assertIn("Peak weight/GPU", text)


class GenerationTests(unittest.TestCase):
    def test_truncation_updates_saved_config_without_mutating_loaded_model(self):
        # Stub only tensor allocation/writing, keeping the generation pipeline and
        # real config serializer so this test needs neither torch nor large RAM.
        writer = types.ModuleType("weight_gen.weight_writer")
        writer.generate_random_weights = Mock(return_value={})
        writer.save_single_file = Mock(return_value=("model.safetensors", {}))
        writer.shard_weights_to_files = Mock(return_value=([], {}))
        writer.write_index_json = Mock()
        spec = importlib.util.spec_from_file_location("weight_gen._test_generation", ROOT / "weight_gen/main.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"weight_gen.weight_writer": writer}):
            spec.loader.exec_module(module)
        raw = {"model_type": "llama", "hidden_size": 16, "num_hidden_layers": 4,
               "vocab_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
               "intermediate_size": 32}
        cfg = ModelConfig(model_type="llama", hidden_size=16, num_hidden_layers=4,
                          vocab_size=64, num_attention_heads=4, num_key_value_heads=2,
                          intermediate_size=32, raw=raw)
        with tempfile.TemporaryDirectory() as output:
            module.generate_model(cfg, ParallelConfig(), output, num_layers=2, dtype="float32", verbose=False)
            saved = json.loads((Path(output) / "config.json").read_text())
        self.assertEqual(saved["num_hidden_layers"], 2)
        self.assertEqual(cfg.num_hidden_layers, 4)
        self.assertEqual(cfg.raw["num_hidden_layers"], 4)
        entries = writer.generate_random_weights.call_args.args[0]
        self.assertTrue(all(entry.dtype == "float32" for entry in entries))
        self.assertFalse(any("layers.2." in entry.name for entry in entries))
        self.assertTrue(any("layers.1." in entry.name for entry in entries))


class CLITests(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run([sys.executable, "-m", "parallel_tool", "--config", "test_config.json", *args],
                              cwd=ROOT, capture_output=True, text=True, timeout=15)

    def test_filtered_cli_exports_full_analysis_once(self):
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "analysis.json")
            process = self.invoke("--tp", "4", "--pp", "2", "--layer", "0", "--plain", "--json", output)
            self.assertEqual(process.returncode, 0, process.stderr)
            exported = json.loads(Path(output).read_text())
        self.assertEqual(process.stdout.count("Kernel Sharding"), 1)
        self.assertIn("L0 q_proj", process.stdout)
        self.assertNotIn("L1 q_proj", process.stdout)
        self.assertTrue(any(row["layer_idx"] == 31 for row in exported["kernels"]))

    def test_invalid_cli_dimension_exits_with_error(self):
        process = self.invoke("--tp", "0")
        self.assertEqual(process.returncode, 2)
        self.assertIn("positive integer", process.stderr)

    @unittest.skipUnless(RICH_AVAILABLE, "Interactive runner requires rich")
    def test_moe_edit_and_filtered_analysis_in_narrow_terminal(self):
        inputs = "4\n2\n3\n1\n4\nn\n1\n6\nbfloat16\nmoe\n0\n0\nn\nn\n1\nn\nn\n0\n"
        process = subprocess.run(
            [sys.executable, "run.py", "--config", "test_moe_config.json", "--width", "48"],
            cwd=ROOT, input=inputs, capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("TP=2 | PP=3 | EP=4", process.stdout)
        self.assertIn("Expert shapes are per expert", process.stdout)
        self.assertIn("Showing 4 of 267 kernels", process.stdout)
        self.assertEqual(process.stdout.count("Kernel Sharding"), 1)
        self.assertNotIn("Traceback", process.stdout + process.stderr)

    @unittest.skipUnless(RICH_AVAILABLE, "Interactive runner requires rich")
    def test_preset_budget_edit_filters_compare_and_analyze_in_one_session(self):
        # Select TP=2 within four GPUs, turn on details for stage 0 / layer 0,
        # compare two presets, then analyze without losing the selected config.
        inputs = "5\n4\n2\n6\nbfloat16\nattention\n0\n0\nn\ny\n3\n1,2\n1\nn\nn\n0\n"
        process = subprocess.run([sys.executable, "run.py", "--config", "test_config.json", "--width", "80"],
                                 cwd=ROOT, input=inputs, capture_output=True, text=True, timeout=15)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("GPU budget: 4", process.stdout)
        self.assertIn("Strategy comparison", process.stdout)
        self.assertIn("HF tensor:", process.stdout)
        self.assertEqual(process.stdout.count("Kernel Sharding"), 1)
        self.assertNotIn("Traceback", process.stdout + process.stderr)


if __name__ == "__main__":
    unittest.main()
