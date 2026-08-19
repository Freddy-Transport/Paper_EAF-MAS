import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class VisualizationFig3AndSelectionTests(unittest.TestCase):
    def test_fig3_panel_tables_use_distinct_group_filters(self):
        from experiments.visualization.eaf_viz.plots_main import horizon_panel_tables

        hf = pd.DataFrame(
            [
                {
                    "case_id": "case_a",
                    "horizon_step": 1,
                    "observed": 100.0,
                    "raw_forecast": 110.0,
                    "adjusted_forecast": 105.0,
                    "is_event_window": True,
                    "is_adjusted_channel": True,
                },
                {
                    "case_id": "case_a",
                    "horizon_step": 2,
                    "observed": 100.0,
                    "raw_forecast": 120.0,
                    "adjusted_forecast": 110.0,
                    "is_event_window": False,
                    "is_adjusted_channel": True,
                },
                {
                    "case_id": "case_a",
                    "horizon_step": 1,
                    "observed": 200.0,
                    "raw_forecast": 210.0,
                    "adjusted_forecast": 210.0,
                    "is_event_window": True,
                    "is_adjusted_channel": False,
                },
                {
                    "case_id": "case_a",
                    "horizon_step": 2,
                    "observed": 200.0,
                    "raw_forecast": 230.0,
                    "adjusted_forecast": 230.0,
                    "is_event_window": False,
                    "is_adjusted_channel": False,
                },
            ]
        )

        panels = horizon_panel_tables(hf, bins=[1, 2])

        self.assertEqual(set(panels), {"overall", "event_window", "adjusted_channels", "non_adjusted_channels"})
        self.assertNotEqual(
            panels["overall"]["wape_gain"].round(6).tolist(),
            panels["event_window"]["wape_gain"].round(6).tolist(),
        )
        self.assertEqual(panels["non_adjusted_channels"]["wape_gain"].round(6).tolist(), [0.0, 0.0])

    def test_rolling_test_panel_table_uses_formal_metrics_not_case_count(self):
        from experiments.visualization.eaf_viz.plots_main import rolling_test_panel_table

        formal = pd.DataFrame(
            [
                {
                    "split": "test",
                    "anchor": i,
                    "anchor_time": f"2023-06-{17 + i:02d} 10:00:00",
                    "mode": "numerical_only",
                    "method_label": "PT-MOMENT",
                    "wape": 20.0 + i,
                    "event_wape": 12.0 + i,
                    "non_event_wape": 22.0 + i,
                    "top128_wape": 16.0 + i,
                    "event_n": 10,
                    "non_event_n": 20,
                }
                for i in range(3)
            ]
        )

        table = rolling_test_panel_table(formal)

        self.assertEqual(set(table["subset"]), {"event_venue28_overall", "event_active", "non_event", "top128_overall"})
        self.assertEqual(table["anchor"].nunique(), 3)
        self.assertEqual(len(table), 12)
        self.assertIn("PT-MOMENT", set(table["method_label"]))

    def test_fig3_gain_first_table_has_event_gain_and_paired_values(self):
        from experiments.visualization.eaf_viz.plots_main import fig3_gain_first_tables

        formal = pd.DataFrame(
            [
                {
                    "anchor": 1,
                    "anchor_time": "2023-06-17 10:00:00",
                    "mode": "numerical_only",
                    "wape": 18.0,
                    "event_wape": 14.0,
                    "non_event_wape": 20.0,
                    "top128_wape": 16.0,
                },
                {
                    "anchor": 1,
                    "anchor_time": "2023-06-17 10:00:00",
                    "mode": "event_adapter_frozen_moment",
                    "wape": 17.9,
                    "event_wape": 13.6,
                    "non_event_wape": 20.0,
                    "top128_wape": 15.95,
                },
            ]
        )

        gain_table, summary = fig3_gain_first_tables(formal)

        self.assertEqual(len(gain_table), 1)
        self.assertAlmostEqual(float(gain_table.iloc[0]["event_gain"]), 0.4)
        self.assertAlmostEqual(float(gain_table.iloc[0]["pt_event_wape"]), 14.0)
        self.assertAlmostEqual(float(gain_table.iloc[0]["adapter_event_wape"]), 13.6)
        self.assertIn("event-active", set(summary["subset_label"]))

    def test_selector_does_not_label_controller_allowed_case_as_abstention(self):
        from experiments.visualization.eaf_viz.case_selection import select_visualization_cases

        cases = pd.DataFrame(
            [
                {
                    "case_id": "allowed_case",
                    "anchor_time": "2023-06-19 10:00:00",
                    "mode": "event_adapter_frozen_moment",
                    "method_label": "EAF-MAS-C",
                    "calibration_decision": "apply",
                    "calibration_enabled": True,
                    "controller_participated": True,
                    "controller_allowed": True,
                    "abstain": False,
                    "final_matches_raw_forecast": False,
                    "max_applied_correction": 0.02,
                    "failed_evidence_dimensions": "",
                    "wape_gain": 0.1,
                    "event_window_wape_gain": 0.2,
                    "adjusted_channel_wape_gain": 0.3,
                    "has_historical_residual_memory": True,
                },
                {
                    "case_id": "blocked_case",
                    "anchor_time": "2023-06-20 10:00:00",
                    "mode": "event_adapter_frozen_moment",
                    "method_label": "EAF-MAS-C",
                    "calibration_decision": "abstain",
                    "calibration_enabled": True,
                    "controller_participated": True,
                    "controller_allowed": False,
                    "abstain": True,
                    "final_matches_raw_forecast": True,
                    "max_applied_correction": 0.0,
                    "failed_evidence_dimensions": "source_validity_score",
                    "adapter_proposal_max_abs_correction": 0.04,
                    "controller_abstention_reason": "source_validity_score",
                    "evidence_validity_score": 0.72,
                    "wape_gain": 0.0,
                    "event_window_wape_gain": 0.0,
                    "adjusted_channel_wape_gain": 0.0,
                    "has_historical_residual_memory": False,
                },
            ]
        )
        selected = select_visualization_cases(cases, pd.DataFrame(), pd.DataFrame())
        by_type = {row["case_type"]: row for row in selected}

        self.assertEqual(by_type["controller_abstention"]["case_id"], "blocked_case")
        self.assertEqual(by_type["localized_gain"]["case_id"], "allowed_case")

    def test_selector_rejects_rag_explain_and_prefers_residual_failure(self):
        from experiments.visualization.eaf_viz.case_selection import select_visualization_cases

        common = {
            "calibration_decision": "abstain",
            "abstain": True,
            "final_matches_raw_forecast": True,
            "max_applied_correction": 0.0,
            "adapter_proposal_max_abs_correction": 0.03,
            "wape_gain": 0.0,
            "event_window_wape_gain": 0.0,
            "adjusted_channel_wape_gain": 0.0,
            "has_historical_residual_memory": True,
        }
        cases = pd.DataFrame(
            [
                {
                    **common,
                    "case_id": "explanation_only",
                    "mode": "rag_explain",
                    "method_label": "EAF-MAS-X",
                    "calibration_enabled": False,
                    "controller_participated": False,
                    "controller_allowed": False,
                    "failed_evidence_dimensions": "source_validity_score",
                    "evidence_validity_score": 0.0,
                },
                {
                    **common,
                    "case_id": "source_failure",
                    "mode": "event_adapter_frozen_moment",
                    "method_label": "EAF-MAS-C",
                    "calibration_enabled": True,
                    "controller_participated": True,
                    "controller_allowed": False,
                    "failed_evidence_dimensions": "source_validity_score",
                    "evidence_validity_score": 0.2,
                },
                {
                    **common,
                    "case_id": "residual_failure",
                    "mode": "event_adapter_frozen_moment",
                    "method_label": "EAF-MAS-C",
                    "calibration_enabled": True,
                    "controller_participated": True,
                    "controller_allowed": False,
                    "failed_evidence_dimensions": "residual_support_score",
                    "evidence_validity_score": 0.8,
                },
            ]
        )

        selected = select_visualization_cases(cases, pd.DataFrame(), pd.DataFrame())
        by_type = {row["case_type"]: row for row in selected}

        self.assertEqual(by_type["controller_abstention"]["case_id"], "residual_failure")
        self.assertEqual(by_type["controller_abstention"]["mode"], "event_adapter_frozen_moment")
        self.assertFalse(by_type["controller_abstention"]["controller_allowed"])

    def test_selector_reports_missing_controller_abstention(self):
        from experiments.visualization.eaf_viz.case_selection import select_visualization_cases

        cases = pd.DataFrame(
            [
                {
                    "case_id": "allowed_case",
                    "anchor_time": "2023-06-19 10:00:00",
                    "calibration_decision": "apply",
                    "calibration_enabled": True,
                    "abstain": False,
                    "wape_gain": 0.1,
                    "event_window_wape_gain": 0.2,
                    "adjusted_channel_wape_gain": 0.3,
                    "has_historical_residual_memory": True,
                }
            ]
        )
        selected = select_visualization_cases(cases, pd.DataFrame(), pd.DataFrame())
        by_type = {row["case_type"]: row for row in selected}

        self.assertEqual(by_type["controller_abstention"]["status"], "missing_required_case")
        self.assertNotEqual(by_type["controller_abstention"].get("case_id"), "allowed_case")

    def test_selected_cases_json_is_written(self):
        from experiments.visualization.eaf_viz.case_selection import write_selected_visualization_cases

        cases = pd.DataFrame(
            [
                {
                    "case_id": "case_a",
                    "anchor_time": "2023-06-19 10:00:00",
                    "calibration_decision": "apply",
                    "calibration_enabled": True,
                    "abstain": False,
                    "wape_gain": 0.1,
                    "event_window_wape_gain": 0.2,
                    "adjusted_channel_wape_gain": 0.3,
                    "has_historical_residual_memory": True,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "selected_visualization_cases.json"
            write_selected_visualization_cases(cases, pd.DataFrame(), pd.DataFrame(), out)
            payload = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(payload["selection_policy"], "post_hoc_visualization_only")
        self.assertGreaterEqual(len(payload["cases"]), 3)

    def test_visualization_selector_does_not_use_near_zero_gain_for_localized_gain(self):
        from experiments.visualization.eaf_viz.case_selection import select_visualization_cases

        cases = pd.DataFrame(
            [
                {
                    "case_id": "weak_near_zero",
                    "anchor_time": "2023-05-09 10:00:00",
                    "calibration_decision": "apply",
                    "controller_allowed": True,
                    "adjusted_channel_wape_gain": 0.000010,
                    "event_window_wape_gain": 0.000023,
                    "wape_gain": 0.000002,
                    "has_historical_residual_memory": True,
                },
                {
                    "case_id": "strong_local",
                    "anchor_time": "2023-06-17 10:00:00",
                    "calibration_decision": "apply",
                    "controller_allowed": True,
                    "adjusted_channel_wape_gain": 0.16,
                    "event_window_wape_gain": 0.28,
                    "wape_gain": 0.09,
                    "has_historical_residual_memory": True,
                },
            ]
        )

        selected = select_visualization_cases(cases, pd.DataFrame(), pd.DataFrame())
        by_type = {row["case_type"]: row for row in selected}

        self.assertEqual(by_type["localized_gain"]["case_id"], "strong_local")
        self.assertGreaterEqual(by_type["localized_gain"]["adjusted_channel_wape_gain"], 0.10)


if __name__ == "__main__":
    unittest.main()
