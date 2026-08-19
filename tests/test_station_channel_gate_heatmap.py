import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class StationChannelGateHeatmapTests(unittest.TestCase):
    def _write_synthetic_trace(self, root: Path, fmt: str = "parquet") -> Path:
        rows = [
            {
                "anchor": 1,
                "date": "2023-01-01 00:00:00",
                "horizon_idx": 0,
                "timestamp": "2023-01-01 00:00:00",
                "station_or_channel": "A001__Alpha_Station",
                "has_structured_event": True,
                "event_type": "Concert",
                "impact_tier": "A",
                "venue_name": "Arena",
                "station_matched": True,
                "channel_matched": True,
                "event_active": True,
                "audit_pass": True,
                "controller_allowed": True,
                "controller_abstained": False,
                "correction_applied": True,
                "bound_clipped": False,
                "reason_code": "allowed",
            },
            {
                "anchor": 1,
                "date": "2023-01-01 00:00:00",
                "horizon_idx": 1,
                "timestamp": "2023-01-01 01:00:00",
                "station_or_channel": "A001__Alpha_Station",
                "has_structured_event": True,
                "event_type": "Concert",
                "impact_tier": "A",
                "venue_name": "Arena",
                "station_matched": True,
                "channel_matched": True,
                "event_active": True,
                "audit_pass": True,
                "controller_allowed": True,
                "controller_abstained": False,
                "correction_applied": False,
                "bound_clipped": False,
                "reason_code": "allowed_no_numeric_effect",
            },
            {
                "anchor": 1,
                "date": "2023-01-01 00:00:00",
                "horizon_idx": 2,
                "timestamp": "2023-01-01 02:00:00",
                "station_or_channel": "B002__Beta_Station",
                "has_structured_event": True,
                "event_type": "Parade",
                "impact_tier": "B",
                "venue_name": "Street",
                "station_matched": True,
                "channel_matched": False,
                "event_active": False,
                "audit_pass": False,
                "controller_allowed": False,
                "controller_abstained": True,
                "correction_applied": False,
                "bound_clipped": False,
                "reason_code": "weak_residual",
            },
            {
                "anchor": 2,
                "date": "2023-01-02 00:00:00",
                "horizon_idx": 0,
                "timestamp": "2023-01-02 00:00:00",
                "station_or_channel": "C003__Tiny_Station",
                "has_structured_event": False,
                "event_type": "none",
                "impact_tier": "none",
                "venue_name": "no_event",
                "station_matched": False,
                "channel_matched": False,
                "event_active": False,
                "audit_pass": False,
                "controller_allowed": False,
                "controller_abstained": False,
                "correction_applied": False,
                "bound_clipped": False,
                "reason_code": "no_event",
            },
        ]
        path = root / f"gate_trace_rows.{fmt}"
        df = pd.DataFrame(rows)
        if fmt == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        return path

    def _write_channel_map(self, root: Path) -> Path:
        path = root / "channel_map.json"
        payload = {
            "top_n": 3,
            "channels": [
                {"rank": 1, "channel_name": "A001__Alpha_Station", "station_complex": "Alpha Station"},
                {"rank": 40, "channel_name": "B002__Beta_Station", "station_complex": "Beta Station"},
                {"rank": 90, "channel_name": "C003__Tiny_Station", "station_complex": "Tiny Station"},
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_synthetic_gate_trace_writes_tables_figures_and_manifest(self):
        from experiments.visualize_station_channel_gate_heatmap import run_station_channel_gate_heatmap

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace = self._write_synthetic_trace(root, "parquet")
            channel_map = self._write_channel_map(root)
            out = root / "out"
            run_station_channel_gate_heatmap(
                gate_trace=trace,
                channel_map_json=channel_map,
                output_root=out,
                top_k_stations=3,
                min_station_cells=2,
                heatmap_column="event_type",
            )

            required = [
                "tables/station_channel_gate_summary.csv",
                "tables/station_channel_gate_summary.tex",
                "tables/gate_action_by_station.csv",
                "tables/gate_action_by_tier.csv",
                "figures/station_channel_gate_heatmap.png",
                "figures/station_channel_gate_heatmap.pdf",
                "figures/gate_action_composition_by_station.png",
                "figures/gate_action_composition_by_station.pdf",
                "figures/gate_action_by_event_tier_heatmap.png",
                "figures/gate_action_by_event_tier_heatmap.pdf",
                "figures/correction_sparsity_by_station_rank.png",
                "figures/correction_sparsity_by_station_rank.pdf",
                "reports/station_channel_gate_heatmap_manifest.json",
                "reports/station_channel_gate_heatmap_interpretation.md",
            ]
            for rel in required:
                self.assertTrue((out / rel).is_file(), rel)

    def test_station_summary_invariants_actions_and_small_n_flag(self):
        from experiments.visualize_station_channel_gate_heatmap import run_station_channel_gate_heatmap

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace = self._write_synthetic_trace(root, "csv")
            channel_map = self._write_channel_map(root)
            out = root / "out"
            run_station_channel_gate_heatmap(
                gate_trace=trace,
                channel_map_json=channel_map,
                output_root=out,
                top_k_stations=3,
                min_station_cells=2,
            )

            summary = pd.read_csv(out / "tables" / "station_channel_gate_summary.csv")
            self.assertTrue((summary["corrected_cells"] <= summary["controller_allowed_cells"]).all())
            self.assertIn("small_n_flag", summary.columns)
            self.assertTrue(summary.loc[summary["station_or_channel"].eq("C003__Tiny_Station"), "small_n_flag"].iloc[0])

            action = pd.read_csv(out / "tables" / "gate_action_by_station.csv")
            expected_actions = {"corrected", "allowed_no_numeric_effect", "abstained", "unchanged_or_not_candidate"}
            self.assertTrue(expected_actions.issubset(set(action["gate_action"])))

            manifest = json.loads((out / "reports" / "station_channel_gate_heatmap_manifest.json").read_text())
            self.assertEqual(manifest["row_count"], 4)
            self.assertEqual(manifest["station_count"], 3)
            self.assertFalse(manifest["case_level_input"])


if __name__ == "__main__":
    unittest.main()
