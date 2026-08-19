import tempfile
import unittest
from pathlib import Path

import numpy as np


class EventPostTrainingTests(unittest.TestCase):
    def test_compute_residual_target_clamps_and_uses_eps(self):
        from event_post_training.sample_builder import compute_residual_target

        actual = np.array([[110.0, 80.0, 5.0]], dtype=np.float32)
        pred = np.array([[100.0, 100.0, 0.0]], dtype=np.float32)

        residual = compute_residual_target(actual, pred, residual_bound=0.10, eps=1.0)

        self.assertTrue(np.allclose(residual, [[0.10, -0.10, 0.10]], atol=1e-6))

    def test_anchor_splits_do_not_cross_boundaries_and_include_negative_samples(self):
        from event_post_training.sample_builder import build_anchor_plan

        plan = build_anchor_plan(
            n_rows=100,
            seq_len=8,
            horizon=4,
            train_rows=50,
            val_rows=20,
            event_start_indices={20, 60},
            stride=6,
            negative_ratio=1.0,
            max_windows=None,
        )

        by_split = {"train": [], "val": [], "test": []}
        for sample in plan:
            by_split[sample["split"]].append(sample)
            self.assertGreaterEqual(sample["history_start"], 0)
            self.assertLessEqual(sample["forecast_end"], sample["split_end"])
        self.assertTrue(any(s["has_event"] for s in by_split["train"]))
        self.assertTrue(any(not s["has_event"] for s in by_split["train"]))
        self.assertTrue(any(s["has_event"] for s in by_split["val"]))

    def test_resolve_lp_model_path_records_fallback(self):
        from event_post_training.config import resolve_lp_model_path

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            requested = root / "experiments" / "outputs" / "lp_top128"
            fallback = root / "experiments" / "outputs" / "lp_fallback_nyc_top128"
            fallback.mkdir(parents=True)
            (fallback / "lp_weights.pt").write_bytes(b"weights")

            resolved, source = resolve_lp_model_path(requested, project_root=root)

            self.assertEqual(resolved, fallback)
            self.assertEqual(source, "lp_fallback_nyc_top128")

    def test_prediction_csv_reconstructs_anchor_matrix(self):
        import pandas as pd
        from event_post_training.sample_builder import load_prediction_csv, prediction_for_anchor

        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "preds.csv"
            pd.DataFrame([
                {"anchor": 10, "channel_name": "a", "horizon_idx": 0, "moment_pred": 1.0},
                {"anchor": 10, "channel_name": "a", "horizon_idx": 1, "moment_pred": 2.0},
                {"anchor": 10, "channel_name": "b", "horizon_idx": 0, "moment_pred": 3.0},
                {"anchor": 10, "channel_name": "b", "horizon_idx": 1, "moment_pred": 4.0},
            ]).to_csv(csv_path, index=False)

            pred_df = load_prediction_csv(csv_path)
            pred = prediction_for_anchor(pred_df, 10, 2, ["a", "b"])

            self.assertTrue(np.allclose(pred, [[1.0, 2.0], [3.0, 4.0]]))

    def test_prediction_csv_builds_anchor_matrix_cache_for_reuse(self):
        import pandas as pd
        from event_post_training.sample_builder import load_prediction_csv, prediction_for_anchor

        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "preds.csv"
            pd.DataFrame([
                {"anchor": 10, "channel_idx": 0, "horizon_idx": 0, "moment_pred": 1.0},
                {"anchor": 10, "channel_idx": 0, "horizon_idx": 1, "moment_pred": 2.0},
                {"anchor": 10, "channel_idx": 1, "horizon_idx": 0, "moment_pred": 3.0},
                {"anchor": 10, "channel_idx": 1, "horizon_idx": 1, "moment_pred": 4.0},
                {"anchor": 14, "channel_idx": 0, "horizon_idx": 0, "moment_pred": 5.0},
                {"anchor": 14, "channel_idx": 0, "horizon_idx": 1, "moment_pred": 6.0},
                {"anchor": 14, "channel_idx": 1, "horizon_idx": 0, "moment_pred": 7.0},
                {"anchor": 14, "channel_idx": 1, "horizon_idx": 1, "moment_pred": 8.0},
            ]).to_csv(csv_path, index=False)

            pred_df = load_prediction_csv(csv_path)
            first = prediction_for_anchor(pred_df, 10, 2, ["a", "b"])
            second = prediction_for_anchor(pred_df, 14, 2, ["a", "b"])

            self.assertTrue(np.allclose(first, [[1.0, 2.0], [3.0, 4.0]]))
            self.assertTrue(np.allclose(second, [[5.0, 6.0], [7.0, 8.0]]))
            self.assertIn("_moment_prediction_matrix_cache", pred_df.attrs)
            self.assertEqual(len(pred_df.attrs["_moment_prediction_matrix_cache"]), 1)


if __name__ == "__main__":
    unittest.main()
