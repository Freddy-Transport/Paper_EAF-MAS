import json
import tempfile
import unittest
from pathlib import Path


def _write_mock_prediction(path: Path, mode: str = "event_adapter_frozen_moment") -> None:
    payload = {
        "request": {
            "date": "2023-06-19 10:00:00",
            "horizon": 3,
            "station_scope": "event_venue28",
            "mode": mode,
        },
        "numerical": {
            "raw_forecast": [[100.0, 110.0, 120.0], [50.0, 55.0, 60.0]],
            "ground_truth": [[100.0, 112.0, 121.0], [49.0, 54.0, 62.0]],
            "adjusted_forecast": [],
            "channel_names": ["N060__Times_Sq_42_St", "A013__49_St"],
            "timestamps": [
                "2023-06-19 10:00:00",
                "2023-06-19 11:00:00",
                "2023-06-19 12:00:00",
            ],
            "model_type": "LP-MOMENT",
        },
        "adjusted_forecast": [[100.0, 111.0, 121.0], [50.0, 55.0, 60.0]],
        "raw_metrics": {"mae": 1.5, "rmse": 2.0, "wape": 1.2, "smape": 1.1},
        "metrics": {"mae": 1.0, "rmse": 1.5, "wape": 0.8, "smape": 0.7},
        "event_window_metrics": {
            "status": "ok",
            "raw": {"wape": 2.0, "mae": 2.0},
            "adjusted": {"wape": 1.0, "mae": 1.0},
            "delta_raw_minus_adjusted": {"wape": 1.0, "mae": 1.0},
        },
        "adjusted_channel_metrics": {
            "status": "ok",
            "raw": {"wape": 2.5, "mae": 2.5},
            "adjusted": {"wape": 1.5, "mae": 1.5},
            "delta_raw_minus_adjusted": {"wape": 1.0, "mae": 1.0},
        },
        "decision": {
            "mode": mode,
            "adjusted_channels": ["N060__Times_Sq_42_St"],
            "abstain": False,
            "confidence": 0.8,
            "correction_bound": 0.05,
            "controller_allowed": True,
            "included_units": [
                {"station_channel": "N060__Times_Sq_42_St", "gate_score": 0.9, "reason": "near event"}
            ],
            "excluded_units": [
                {"station_channel": "A013__49_St", "exclusion_reason": "no event-station-channel evidence"}
            ],
            "correction_stats": {
                "max_abs_correction": 0.02,
                "mean_abs_correction": 0.01,
                "active_correction_cells": 2,
                "correction_bound": 0.05,
            },
        },
        "evidence": {
            "structured_events": [
                {
                    "title": "TSQ LIVE",
                    "event_time": "2023-06-19 11:00:00",
                    "impact_tier": "A",
                    "event_type": "Street Event",
                    "location": "Times Square",
                    "channel_name": "N060__Times_Sq_42_St",
                }
            ],
            "sources": [],
            "model_assisted_summaries": [{"summary": "Model-assisted evidence summary."}],
            "historical_event_cases": [
                {
                    "historical_event_title": "Past TSQ event",
                    "event_time": "2022-06-19",
                    "matched_station": "N060",
                    "residual_direction": "increase",
                    "median_lp_moment_correction": "+1.0%",
                    "split": "train_val_memory",
                }
            ],
            "local_residual_cases": [
                "[residual_case type=Street Event day=weekday rank=top64 correction=+1.0% best_score=1.0 merged_count=1] Statistics (N=8): p25=-0.2%, median=+1.0%, p75=+1.4%, std=1.0%"
            ],
            "evidence_audit": {
                "source_validity_score": 0.65,
                "geo_consistency_score": 0.9,
                "temporal_alignment_score": 0.95,
                "residual_support_score": 0.8,
                "evidence_validity_score": 0.85,
                "conflict_flags": [],
            },
            "has_major_event": True,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class VisualizationSchemaTests(unittest.TestCase):
    def test_default_method_aliases_use_pt_moment_for_paper_label(self):
        from experiments.visualization.eaf_viz.schema import default_method_aliases

        aliases = default_method_aliases()

        self.assertEqual(aliases["numerical_only"], "PT-MOMENT")
        self.assertNotIn("LP-MOMENT", set(aliases.values()))

    def test_method_aliases_and_case_tables_from_prediction_json(self):
        from experiments.visualization.eaf_viz.schema import build_processed_tables, default_method_aliases

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pred_dir = root / "case" / "predictions"
            pred_dir.mkdir(parents=True)
            _write_mock_prediction(pred_dir / "case.json")

            tables, report = build_processed_tables([root], method_aliases=default_method_aliases())

            self.assertEqual(tables["case_metrics"].iloc[0]["method_label"], "EAF-MAS-C")
            self.assertEqual(tables["case_metrics"].iloc[0]["wape_gain"], 0.4)
            self.assertEqual(len(tables["channel_metrics"]), 2)
            self.assertEqual(len(tables["horizon_forecast"]), 6)
            self.assertEqual(len(tables["event_audit"]), 1)
            self.assertEqual(len(tables["residual_memory"]), 2)
            self.assertIn("case_metrics", report)

    def test_horizon_rows_mark_forecast_time_event_window(self):
        from experiments.visualization.eaf_viz.schema import build_processed_tables, default_method_aliases

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pred_dir = root / "case" / "predictions"
            pred_dir.mkdir(parents=True)
            _write_mock_prediction(pred_dir / "case.json")

            tables, _ = build_processed_tables([root], method_aliases=default_method_aliases())
            horizon = tables["horizon_forecast"]

            event_rows = horizon[horizon["is_event_window"]]
            self.assertFalse(event_rows.empty)
            self.assertEqual(sorted(event_rows["timestamp"].unique().tolist()), [
                "2023-06-19 10:00:00",
                "2023-06-19 11:00:00",
                "2023-06-19 12:00:00",
            ])

    def test_case_metrics_include_localized_metric_fields(self):
        from experiments.visualization.eaf_viz.schema import build_processed_tables, default_method_aliases

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pred_dir = root / "case" / "predictions"
            pred_dir.mkdir(parents=True)
            _write_mock_prediction(pred_dir / "case.json")

            tables, _ = build_processed_tables([root], method_aliases=default_method_aliases())
            row = tables["case_metrics"].iloc[0]

            self.assertEqual(row["event_window_wape_gain"], 1.0)
            self.assertEqual(row["adjusted_channel_wape_gain"], 1.0)
            self.assertEqual(row["event_window_cell_count"], 0)

    def test_rag_explain_alias_is_eaf_mas_x(self):
        from experiments.visualization.eaf_viz.schema import method_label

        self.assertEqual(method_label("rag_explain"), "EAF-MAS-X")

    def test_formal_metric_rows_attach_method_aliases_and_baseline_deltas(self):
        from experiments.visualization.eaf_viz.schema import build_processed_tables, default_method_aliases

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            metric_path = root / "full_metric_rows.csv"
            metric_path.write_text(
                "split,anchor,date,mode,scope,wape,event_wape,non_event_wape,top128_wape,event_n,non_event_n\n"
                "test,1,2023-06-17 10:00:00,numerical_only,event_venue28,20.0,12.0,23.0,16.0,10,20\n"
                "test,1,2023-06-17 10:00:00,event_adapter_frozen_moment,event_venue28,18.0,11.0,23.2,15.8,10,20\n",
                encoding="utf-8",
            )

            tables, report = build_processed_tables(
                [],
                method_aliases=default_method_aliases(),
                formal_metric_paths=[metric_path],
            )

            formal = tables["formal_metrics"]
            adapter = formal[formal["mode"] == "event_adapter_frozen_moment"].iloc[0]
            self.assertEqual(adapter["method_label"], "EAF-MAS-C")
            self.assertAlmostEqual(float(adapter["wape_gain_vs_baseline"]), 2.0)
            self.assertAlmostEqual(float(adapter["event_wape_gain_vs_baseline"]), 1.0)
            self.assertTrue(bool(adapter["has_event_cells"]))
            self.assertIn("formal_metrics", report)


if __name__ == "__main__":
    unittest.main()
