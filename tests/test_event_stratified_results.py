import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


def _write_synthetic_inputs(root: Path) -> tuple[Path, Path, Path, Path]:
    metric_rows = []
    anchors = [
        (100, "2023-06-01 10:00:00"),
        (101, "2023-06-01 11:00:00"),
        (102, "2023-06-01 12:00:00"),
        (103, "2023-06-10 10:00:00"),
    ]
    for anchor, date in anchors:
        metric_rows.append(
            {
                "split": "test",
                "anchor": anchor,
                "date": date,
                "mode": "pt_moment",
                "event_active_wape": 20.0,
                "event_active_mae": 10.0,
                "event_active_n": 5 if anchor != 103 else 0,
                "non_event_wape": 15.0,
                "non_event_mae": 7.0,
                "non_event_n": 20,
            }
        )
        metric_rows.append(
            {
                "split": "test",
                "anchor": anchor,
                "date": date,
                "mode": "event_adapter_frozen",
                "event_active_wape": 18.0 if anchor != 102 else 22.0,
                "event_active_mae": 9.0,
                "event_active_n": 5 if anchor != 103 else 0,
                "non_event_wape": 15.1,
                "non_event_mae": 7.1,
                "non_event_n": 20,
            }
        )
    metric_csv = root / "full_metric_rows.csv"
    pd.DataFrame(metric_rows).to_csv(metric_csv, index=False)

    anchor_csv = root / "full_anchor_plan.csv"
    pd.DataFrame(
        [
            {
                "split": "test",
                "row_idx": anchor,
                "anchor": anchor,
                "date": date,
                "horizon": 192,
                "horizon_end_row_exclusive": anchor + 192,
                "horizon_end_time": str(pd.Timestamp(date) + pd.Timedelta(hours=191)),
            }
            for anchor, date in anchors
        ]
    ).to_csv(anchor_csv, index=False)

    events_json = root / "events.json"
    events_json.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "title": "Times Square Plaza Event",
                        "event_time": "2023-06-02 18:00:00",
                        "event_type": "Plaza Event",
                        "location": "Times Square: Father Duffy Square | 49 St",
                        "channel_name": "A013__49_St",
                        "station_rank": 20,
                    },
                    {
                        "title": "Central Park Routine Closure",
                        "event_time": "2023-06-01 19:00:00",
                        "event_type": "Special Event",
                        "location": "Central Park: Great Lawn | 81 St",
                        "channel_name": "N044__81_St",
                        "station_rank": 100,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    venue_json = root / "venue37_fusion_channels.json"
    venue_json.write_text(json.dumps({"channel_names": ["A013__49_St"]}), encoding="utf-8")
    return metric_csv, anchor_csv, events_json, venue_json


class EventStratifiedResultsTests(unittest.TestCase):
    def test_anchor_level_analysis_without_prediction_cache_writes_outputs(self):
        from experiments.analyze_event_stratified_results import run_event_stratified_analysis

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            metric_csv, anchor_csv, events_json, venue_json = _write_synthetic_inputs(root)
            out = root / "out"

            result = run_event_stratified_analysis(
                output_root=out,
                full_metric_rows=metric_csv,
                anchor_plan_csv=anchor_csv,
                events_json=events_json,
                venue_channels_json=venue_json,
                prediction_cache_npz=root / "missing_cache.npz",
                bootstrap_samples=100,
                min_group_n=3,
                random_seed=7,
            )

            self.assertFalse(result["manifest"]["cell_level_executed"])
            required = [
                "tables/event_type_gain_summary.csv",
                "tables/event_type_gain_summary.tex",
                "tables/impact_tier_gain_summary.csv",
                "tables/venue_group_gain_summary.csv",
                "tables/station_rank_group_gain_summary.csv",
                "figures/event_type_gain_heatmap.png",
                "figures/event_type_gain_heatmap.pdf",
                "figures/impact_tier_gain_forest.png",
                "figures/venue_group_gain_forest.png",
                "figures/event_count_vs_gain_scatter.png",
                "reports/event_stratified_manifest.json",
                "reports/event_stratified_interpretation.md",
            ]
            for rel in required:
                self.assertTrue((out / rel).is_file(), rel)

            summary = pd.read_csv(out / "tables" / "event_type_gain_summary.csv")
            self.assertFalse(summary.empty)
            for col in ["n_anchors", "gain_ci95_low", "gain_ci95_high", "small_n_flag"]:
                self.assertIn(col, summary.columns)
            self.assertEqual(4, int(summary["n_anchors"].sum()))
            self.assertIn("no_event", set(summary["dominant_event_type"]))
            self.assertTrue(summary.loc[summary["n_anchors"] < 3, "small_n_flag"].all())

    def test_missing_required_modes_raises_clear_error(self):
        from experiments.analyze_event_stratified_results import run_event_stratified_analysis

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            metric_csv, anchor_csv, events_json, venue_json = _write_synthetic_inputs(root)
            df = pd.read_csv(metric_csv)
            df = df[df["mode"] != "event_adapter_frozen"]
            df.to_csv(metric_csv, index=False)

            with self.assertRaisesRegex(ValueError, "requires modes"):
                run_event_stratified_analysis(
                    output_root=root / "out",
                    full_metric_rows=metric_csv,
                    anchor_plan_csv=anchor_csv,
                    events_json=events_json,
                    venue_channels_json=venue_json,
                )


if __name__ == "__main__":
    unittest.main()
