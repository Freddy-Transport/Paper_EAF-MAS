import inspect
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REQUIRED_MODES = {
    "pt_moment",
    "adapter_full_controller",
    "adapter_no_event_gate",
    "adapter_no_residual_support",
    "adapter_no_bound",
    "adapter_no_audit_guard",
    "adapter_always_apply",
    "adapter_random_gate",
}


class CalibrationMatchedAblationTests(unittest.TestCase):
    def test_synthetic_smoke_writes_required_outputs_and_modes(self):
        from experiments.run_calibration_matched_ablation import run_synthetic_smoke

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_synthetic_smoke(root, seed=17)

            required = [
                "predictions/calibration_ablation_rows.csv",
                "tables/calibration_ablation_summary.csv",
                "tables/calibration_ablation_summary.tex",
                "tables/calibration_ablation_paired_gain_ci.csv",
                "tables/calibration_ablation_metric_directions.csv",
                "figures/calibration_ablation_wape_bar.png",
                "figures/calibration_ablation_wape_bar.pdf",
                "figures/calibration_ablation_event_active_gain_forest.png",
                "figures/calibration_ablation_event_active_gain_forest.pdf",
                "figures/calibration_ablation_safety_regret.png",
                "figures/calibration_ablation_safety_regret.pdf",
                "reports/calibration_ablation_manifest.json",
                "reports/calibration_ablation_interpretation.md",
            ]
            for rel in required:
                self.assertTrue((root / rel).is_file(), rel)

            rows = pd.read_csv(root / "predictions" / "calibration_ablation_rows.csv")
            self.assertEqual(REQUIRED_MODES, set(rows["mode"]))
            counts = rows.groupby("mode")["anchor"].nunique().to_dict()
            self.assertEqual(1, len(set(counts.values())))

    def test_controller_invariants_and_metric_directions(self):
        from experiments.run_calibration_matched_ablation import run_synthetic_smoke

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_synthetic_smoke(root, seed=23)
            summary = pd.read_csv(root / "tables" / "calibration_ablation_summary.csv").set_index("mode")
            directions = pd.read_csv(root / "tables" / "calibration_ablation_metric_directions.csv")

            self.assertEqual(0, int(summary.loc["pt_moment", "active_correction_cells"]))
            self.assertGreaterEqual(
                float(summary.loc["adapter_no_bound", "max_abs_correction"]),
                float(summary.loc["adapter_full_controller", "max_abs_correction"]),
            )
            self.assertIn("direction", directions.columns)
            self.assertIn("event_active_gain", set(directions["metric"]))
            self.assertEqual(
                "higher_is_better",
                directions.set_index("metric").loc["event_active_gain", "direction"],
            )

    def test_controller_decision_helper_does_not_accept_actual(self):
        from experiments.run_calibration_matched_ablation import build_mode_correction

        self.assertNotIn("actual", inspect.signature(build_mode_correction).parameters)

    def test_random_gate_is_seed_reproducible(self):
        from experiments.run_calibration_matched_ablation import run_synthetic_smoke

        with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
            run_synthetic_smoke(Path(td1), seed=99)
            run_synthetic_smoke(Path(td2), seed=99)
            rows1 = pd.read_csv(Path(td1) / "predictions" / "calibration_ablation_rows.csv")
            rows2 = pd.read_csv(Path(td2) / "predictions" / "calibration_ablation_rows.csv")
            rand1 = rows1[rows1["mode"] == "adapter_random_gate"].reset_index(drop=True)
            rand2 = rows2[rows2["mode"] == "adapter_random_gate"].reset_index(drop=True)
            pd.testing.assert_frame_equal(rand1, rand2)


if __name__ == "__main__":
    unittest.main()
