import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


TIMING_COLUMNS = [
    "pt_moment_inference_ms",
    "event_feature_cube_ms",
    "adapter_correction_ms",
    "evidence_lookup_ms",
    "evidence_audit_ms",
    "residual_memory_retrieval_ms",
    "autoskill_retrieval_ms",
    "explanation_generation_ms",
    "total_pipeline_ms",
]


class RuntimeCostProfileTests(unittest.TestCase):
    def test_synthetic_smoke_writes_outputs_and_sample_rows(self):
        from experiments.profile_eafmas_runtime_cost import run_synthetic_smoke

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_synthetic_smoke(root, sample_anchors=3, seed=17, enable_qwenplus_live=False)

            required = [
                "tables/runtime_cost_rows.csv",
                "tables/runtime_cost_summary.csv",
                "tables/runtime_cost_summary.tex",
                "figures/runtime_cost_stacked_bar.png",
                "figures/runtime_cost_stacked_bar.pdf",
                "figures/runtime_cost_event_vs_nonevent_boxplot.png",
                "figures/runtime_cost_event_vs_nonevent_boxplot.pdf",
                "reports/runtime_cost_manifest.json",
                "reports/runtime_cost_interpretation.md",
            ]
            for rel in required:
                self.assertTrue((root / rel).is_file(), rel)

            rows = pd.read_csv(root / "tables" / "runtime_cost_rows.csv")
            self.assertEqual(3, len(rows))
            self.assertIn(True, set(rows["has_event"].astype(bool)))
            self.assertIn(False, set(rows["has_event"].astype(bool)))

    def test_timing_fields_are_nonnegative_or_na_for_disabled_qwenplus(self):
        from experiments.profile_eafmas_runtime_cost import run_synthetic_smoke

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_synthetic_smoke(root, sample_anchors=3, seed=23, enable_qwenplus_live=False)
            rows = pd.read_csv(root / "tables" / "runtime_cost_rows.csv")

            for col in TIMING_COLUMNS:
                self.assertTrue((pd.to_numeric(rows[col], errors="coerce") >= 0).all(), col)
            self.assertTrue(rows["qwenplus_summary_ms"].isna().all())
            self.assertEqual({0}, set(rows["qwenplus_api_calls"].astype(int)))
            self.assertEqual({"disabled"}, set(rows["qwenplus_status"]))

    def test_manifest_records_hardware_device_model_and_sample_protocol(self):
        from experiments.profile_eafmas_runtime_cost import run_synthetic_smoke

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_synthetic_smoke(
                root,
                sample_anchors=3,
                seed=31,
                enable_qwenplus_live=False,
                enable_local_vllm=True,
                llm_model="Qwen/Qwen3-8B",
            )
            manifest = json.loads((root / "reports" / "runtime_cost_manifest.json").read_text())

            self.assertEqual(3, manifest["sample_protocol"]["sample_anchors"])
            self.assertEqual(31, manifest["sample_protocol"]["seed"])
            self.assertIn("device", manifest["hardware"])
            self.assertEqual("Qwen/Qwen3-8B", manifest["models"]["local_vllm_model"])
            self.assertFalse(manifest["qwenplus"]["live_enabled"])
            self.assertIn("event", set(manifest["event_group_counts"]))
            self.assertIn("non_event", set(manifest["event_group_counts"]))

    def test_summary_contains_mean_median_p90_p95(self):
        from experiments.profile_eafmas_runtime_cost import run_synthetic_smoke

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_synthetic_smoke(root, sample_anchors=3, seed=37, enable_qwenplus_live=False)
            summary = pd.read_csv(root / "tables" / "runtime_cost_summary.csv")
            required_stats = {"mean", "median", "p90", "p95", "n"}
            self.assertTrue(required_stats.issubset(set(summary.columns)))
            self.assertIn("total_pipeline_ms", set(summary["metric"]))


if __name__ == "__main__":
    unittest.main()
