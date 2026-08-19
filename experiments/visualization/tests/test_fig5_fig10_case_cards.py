import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


def _horizon_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "x_case",
                "method_label": "EAF-MAS-X",
                "horizon_step": 1,
                "observed": 100.0,
                "raw_forecast": 110.0,
                "adjusted_forecast": 110.0,
                "relative_correction": 0.0,
                "correction_bound": 0.05,
                "is_affected_channel": True,
                "is_adjusted_channel": False,
                "is_event_window": True,
            },
            {
                "case_id": "c_case",
                "method_label": "EAF-MAS-C",
                "horizon_step": 1,
                "observed": 100.0,
                "raw_forecast": 120.0,
                "adjusted_forecast": 114.0,
                "relative_correction": -0.05,
                "correction_bound": 0.05,
                "is_affected_channel": True,
                "is_adjusted_channel": True,
                "is_event_window": True,
            },
            {
                "case_id": "c_case",
                "method_label": "EAF-MAS-C",
                "horizon_step": 2,
                "observed": 100.0,
                "raw_forecast": 100.0,
                "adjusted_forecast": 100.0,
                "relative_correction": 0.0,
                "correction_bound": 0.05,
                "is_affected_channel": False,
                "is_adjusted_channel": False,
                "is_event_window": False,
            },
        ]
    )


def _case_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "x_case",
                "method_label": "EAF-MAS-X",
                "raw_wape": 10.0,
                "adjusted_wape": 10.0,
                "wape_gain": 0.0,
                "event_window_wape_gain": 0.0,
                "adjusted_channel_wape_gain": 0.0,
                "included_unit_wape_gain": 0.0,
                "calibration_decision": "abstain",
                "controller_allowed": False,
                "source_validity_score": 0.0,
                "geo_consistency_score": 0.8,
                "temporal_alignment_score": 0.9,
                "residual_support_score": 0.7,
                "evidence_validity_score": 0.6,
                "has_model_assisted_summary": True,
                "has_historical_residual_memory": True,
                "has_verified_external_evidence": False,
                "adjusted_channel_count": 0,
                "max_correction": 0.0,
            },
            {
                "case_id": "c_case",
                "method_label": "EAF-MAS-C",
                "raw_wape": 12.0,
                "adjusted_wape": 11.0,
                "wape_gain": 1.0,
                "event_window_wape_gain": 2.0,
                "adjusted_channel_wape_gain": 3.0,
                "included_unit_wape_gain": 3.5,
                "calibration_decision": "apply",
                "controller_allowed": True,
                "source_validity_score": 0.65,
                "geo_consistency_score": 0.9,
                "temporal_alignment_score": 0.95,
                "residual_support_score": 0.95,
                "evidence_validity_score": 0.87,
                "has_model_assisted_summary": True,
                "has_historical_residual_memory": True,
                "has_verified_external_evidence": False,
                "adjusted_channel_count": 1,
                "max_correction": 0.05,
            },
        ]
    )


def _channel_rows_for_gate() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "c_case",
                "station_name": "Times_Sq_42_St_N_Q_R_W_S_1_2_3_7_42_St_A_C_E_Bryant_Pk_B_D_F_M_5_Av",
                "is_affected_channel": True,
                "is_adjusted_channel": True,
                "is_excluded_channel": False,
                "gate_score": 0.92,
                "correction_max": 0.040,
                "correction_abs_mean": 0.010,
            },
            {
                "case_id": "c_case",
                "station_name": "34_St_Herald_Sq",
                "is_affected_channel": False,
                "is_adjusted_channel": False,
                "is_excluded_channel": True,
                "gate_score": None,
                "correction_max": 0.0,
                "correction_abs_mean": 0.0,
            },
        ]
    )


def _residual_rows_for_memory() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "c_case",
                "historical_event_name": "Past Street Festival",
                "historical_event_type": "Street Festival",
                "direction": "decrease",
                "median_correction_pct": -0.7,
                "n": None,
            },
            {
                "case_id": "c_case",
                "historical_event_name": None,
                "historical_event_type": "Street",
                "direction": None,
                "median_correction_pct": 0.5,
                "n": 9,
            },
        ]
    )


class Fig5Fig10AndCaseCardTests(unittest.TestCase):
    def test_correction_safety_summary_keeps_x_out_of_correction_subsets(self):
        from experiments.visualization.eaf_viz.plots_main import correction_safety_summary

        table = correction_safety_summary(_horizon_rows(), correction_bound=0.05)
        c_active = table[(table["method_label"] == "EAF-MAS-C") & (table["subset"] == "active_correction_cells")].iloc[0]
        x_active = table[(table["method_label"] == "EAF-MAS-X") & (table["subset"] == "active_correction_cells")].iloc[0]

        self.assertEqual(int(c_active["n_rows"]), 1)
        self.assertEqual(int(c_active["nonzero_correction_cells"]), 1)
        self.assertAlmostEqual(float(c_active["max_abs_relative_correction"]), 0.05)
        self.assertEqual(int(x_active["n_rows"]), 0)
        self.assertIn("EAF-MAS-X", set(table["method_label"]))

    def test_fig5_active_correction_plot_data_filters_zero_cells(self):
        from experiments.visualization.eaf_viz.plots_main import active_correction_cells_for_plot

        active = active_correction_cells_for_plot(_horizon_rows())

        self.assertEqual(len(active), 1)
        self.assertEqual(active.iloc[0]["method_label"], "EAF-MAS-C")
        self.assertAlmostEqual(float(active.iloc[0]["relative_correction"]), -0.05)

    def test_fig5_rolling_gain_distribution_uses_anchor_level_formal_metrics(self):
        from experiments.visualization.eaf_viz.plots_main import rolling_gain_distribution_table

        formal = pd.DataFrame(
            [
                {
                    "anchor": 1,
                    "anchor_time": "2023-06-17 10:00:00",
                    "mode": "event_adapter_frozen_moment",
                    "method_label": "EAF-MAS-C",
                    "wape_gain_vs_baseline": 0.2,
                    "event_wape_gain_vs_baseline": 0.4,
                    "non_event_wape_gain_vs_baseline": -0.1,
                    "top128_wape_gain_vs_baseline": 0.05,
                },
                {
                    "anchor": 2,
                    "anchor_time": "2023-06-18 10:00:00",
                    "mode": "rag_explain",
                    "method_label": "EAF-MAS-X",
                    "wape_gain_vs_baseline": 0.0,
                    "event_wape_gain_vs_baseline": 0.0,
                    "non_event_wape_gain_vs_baseline": 0.0,
                    "top128_wape_gain_vs_baseline": 0.0,
                },
            ]
        )

        table = rolling_gain_distribution_table(formal)

        self.assertEqual(table["anchor"].nunique(), 1)
        self.assertIn("event_wape_gain_vs_baseline", set(table["metric"]))
        self.assertAlmostEqual(float(table.loc[table["metric"] == "event_wape_gain_vs_baseline", "gain"].iloc[0]), 0.4)

    def test_fig5_safety_badges_treat_zero_bound_and_non_event_as_pass(self):
        from experiments.visualization.eaf_viz.plots_main import fig5_safety_badges

        formal = pd.DataFrame(
            [
                {
                    "anchor": 1,
                    "anchor_time": "2023-06-17 10:00:00",
                    "mode": "event_adapter_frozen_moment",
                    "non_event_wape_gain_vs_baseline": 0.0,
                    "event_wape_gain_vs_baseline": 0.2,
                }
            ]
        )

        badges = fig5_safety_badges(_horizon_rows(), formal, correction_bound=0.05)

        self.assertEqual(int(badges["bound_violation_count"]), 0)
        self.assertTrue(bool(badges["non_event_unchanged"]))
        self.assertGreater(float(badges["max_bound_use_pct"]), 0.0)

    def test_fig7_gate_summary_shortens_labels_and_counts_all_channel_states(self):
        from experiments.visualization.eaf_viz.plots_main import channel_gate_summary_tables

        ranked, flow = channel_gate_summary_tables(_channel_rows_for_gate(), top_k=20)

        self.assertIn("station_label", ranked.columns)
        self.assertNotIn("Times_Sq_42_St_N_Q_R_W_S_1_2_3_7", ranked.iloc[0]["station_label"])
        self.assertIn("correction_bps", ranked.columns)
        self.assertGreater(float(ranked.iloc[0]["correction_bps"]), 0.0)
        self.assertEqual(int(flow.loc[flow["decision_state"] == "adjusted", "count"].iloc[0]), 1)
        self.assertEqual(int(flow.loc[flow["decision_state"] == "excluded", "count"].iloc[0]), 1)
        self.assertIn("candidate_only", set(flow["decision_state"]))

    def test_fig9_memory_diagnostics_keeps_empty_skill_as_explicit_lifecycle_status(self):
        from experiments.visualization.eaf_viz.plots_main import memory_diagnostics_tables

        coverage, support, skill_status = memory_diagnostics_tables(
            _residual_rows_for_memory(),
            pd.DataFrame(),
        )

        self.assertFalse(coverage.empty)
        self.assertFalse(support.empty)
        self.assertEqual(skill_status.iloc[0]["lifecycle_status"], "no_promoted_skill")
        self.assertIn("no promoted residual-memory skill", skill_status.iloc[0]["message"])

    def test_fig6_evidence_audit_deduplicates_physical_events_and_reports_variance(self):
        from experiments.visualization.eaf_viz.plots_main import evidence_audit_diagnostics

        audit = pd.DataFrame(
            [
                {
                    "case_id": "case_a",
                    "anchor_time": "2023-06-19 10:00:00",
                    "event_name": "TSQ LIVE",
                    "event_start_time": "2023-06-20 17:00:00",
                    "venue": "Times Square",
                    "source_type": "model-assisted",
                    "source_validity_score": 0.65,
                    "geo_consistency_score": 0.9,
                    "temporal_alignment_score": 0.95,
                    "residual_support_score": 0.8,
                    "evidence_validity_score": 0.85,
                },
                {
                    "case_id": "case_a_duplicate_channel",
                    "anchor_time": "2023-06-19 10:00:00",
                    "event_name": "TSQ LIVE",
                    "event_start_time": "2023-06-20 17:00:00",
                    "venue": "Times Square",
                    "source_type": "model-assisted",
                    "source_validity_score": 0.65,
                    "geo_consistency_score": 0.9,
                    "temporal_alignment_score": 0.95,
                    "residual_support_score": 0.8,
                    "evidence_validity_score": 0.85,
                },
                {
                    "case_id": "case_b",
                    "anchor_time": "2023-06-21 10:00:00",
                    "event_name": "MSG concert",
                    "event_start_time": "2023-06-21 19:00:00",
                    "venue": "MSG",
                    "source_type": "structured",
                    "source_validity_score": 0.2,
                    "geo_consistency_score": 0.7,
                    "temporal_alignment_score": 0.8,
                    "residual_support_score": 0.4,
                    "evidence_validity_score": 0.5,
                },
            ]
        )

        rows, variance = evidence_audit_diagnostics(audit)

        self.assertEqual(len(rows), 2)
        self.assertIn("source_validity_score", set(variance["field"]))
        self.assertGreater(float(variance.loc[variance["field"] == "source_validity_score", "variance"].iloc[0]), 0.0)

    def test_fig6_heatmap_table_keeps_deduplicated_audit_rows(self):
        from experiments.visualization.eaf_viz.plots_main import evidence_audit_heatmap_table

        rows = pd.DataFrame(
            [
                {
                    "anchor_time": "2023-06-19 10:00:00",
                    "event_name": "TSQ LIVE",
                    "source_validity_score": 0.65,
                    "geo_consistency_score": 0.9,
                    "temporal_alignment_score": 0.95,
                    "semantic_consistency_score": 0.8,
                    "residual_support_score": 0.75,
                    "evidence_validity_score": 0.82,
                }
            ]
        )

        heatmap = evidence_audit_heatmap_table(rows)

        self.assertEqual(heatmap["matrix"].shape, (1, 6))
        self.assertIn("TSQ LIVE", heatmap["row_labels"][0])
        self.assertIn("source", heatmap["field_labels"][0])

    def test_fig6_temporal_alignment_marks_inside_forecast_horizon(self):
        from experiments.visualization.eaf_viz.plots_main import evidence_temporal_alignment_table

        rows = pd.DataFrame(
            [
                {
                    "anchor_time": "2023-06-19 10:00:00",
                    "event_start_time": "2023-06-20 17:00:00",
                    "event_name": "TSQ LIVE",
                },
                {
                    "anchor_time": "2023-06-19 10:00:00",
                    "event_start_time": "2023-07-01 10:00:00",
                    "event_name": "Outside Horizon",
                },
            ]
        )

        table = evidence_temporal_alignment_table(rows, horizon_hours=192)

        self.assertTrue(bool(table.iloc[0]["inside_horizon"]))
        self.assertFalse(bool(table.iloc[1]["inside_horizon"]))
        self.assertAlmostEqual(float(table.iloc[0]["event_offset_hours"]), 31.0)

    def test_fig6_variance_badge_explains_low_variance_limitation(self):
        from experiments.visualization.eaf_viz.plots_main import evidence_variance_badge_text

        variance = pd.DataFrame(
            [
                {"field": "source_validity_score", "variance": 0.0, "n": 8},
                {"field": "geo_consistency_score", "variance": 8.6e-5, "n": 8},
                {"field": "temporal_alignment_score", "variance": 0.0, "n": 8},
                {"field": "residual_support_score", "variance": 0.0, "n": 8},
                {"field": "evidence_validity_score", "variance": 3e-6, "n": 8},
            ]
        )

        text = evidence_variance_badge_text(variance, n_unique=8)

        self.assertIn("n_unique=8", text)
        self.assertIn("geo variance = 8.60e-05", text)
        self.assertIn("not a strong evidence-quality claim", text)

    def test_appendix_replaces_underpowered_wape_group_plots(self):
        from experiments.visualization.eaf_viz.plots_appendix import should_plot_underpowered_performance_group

        four_cases = pd.DataFrame(
            {
                "case_id": ["a", "b", "c", "d"],
                "impact_tier": ["A", "A", "A", "A"],
                "event_type": ["Street Festival", "Street Festival", "Street Event", "Plaza Partner Event"],
                "wape_gain": [1e-6, 0.0, 0.0, 0.0],
            }
        )
        richer_cases = pd.DataFrame(
            {
                "case_id": [f"case_{i}" for i in range(24)],
                "impact_tier": ["A", "B"] * 12,
                "event_type": ["Street Festival", "Sports"] * 12,
                "wape_gain": [0.1] * 24,
            }
        )

        self.assertFalse(should_plot_underpowered_performance_group(four_cases, "impact_tier"))
        self.assertFalse(should_plot_underpowered_performance_group(four_cases, "event_type"))
        self.assertTrue(should_plot_underpowered_performance_group(richer_cases, "impact_tier"))

    def test_fig2_evidence_composition_drops_verified_citation_metric(self):
        from experiments.visualization.eaf_viz.plots_main import fig2_evidence_source_composition

        table = fig2_evidence_source_composition(_case_rows())

        self.assertIn("model-assisted summary", set(table["metric"]))
        self.assertIn("historical residual memory", set(table["metric"]))
        self.assertNotIn("verified citation", set(table["metric"]))
        self.assertNotIn("missing citation", set(table["metric"]))

    def test_fig2_cell_ratio_table_sums_to_one(self):
        from experiments.visualization.eaf_viz.plots_main import fig2_cell_ratio_table

        summary = pd.DataFrame(
            [
                {"metric": "event cells", "value": 25},
                {"metric": "non-event cells", "value": 75},
            ]
        )

        ratio = fig2_cell_ratio_table(summary)

        self.assertAlmostEqual(float(ratio["ratio"].sum()), 1.0)
        self.assertAlmostEqual(float(ratio.loc[ratio["metric"] == "event cells", "ratio"].iloc[0]), 0.25)

    def test_fig4_selects_visible_local_intervention_window(self):
        from experiments.visualization.eaf_viz.plots_main import fig4_focus_window_for_plot

        cases = pd.DataFrame(
            [
                {
                    "case_id": "weak_case",
                    "method_label": "EAF-MAS-C",
                    "controller_allowed": True,
                    "adjusted_channel_wape_gain": 0.01,
                    "included_unit_wape_gain": 0.01,
                    "event_window_wape_gain": 0.0,
                },
                {
                    "case_id": "strong_case",
                    "method_label": "EAF-MAS-C",
                    "controller_allowed": True,
                    "adjusted_channel_wape_gain": 0.30,
                    "included_unit_wape_gain": 0.20,
                    "event_window_wape_gain": 0.10,
                },
            ]
        )
        rows = []
        for i in range(16):
            rows.append(
                {
                    "case_id": "weak_case",
                    "channel_id": "N001",
                    "horizon_step": i + 1,
                    "timestamp": f"2023-01-01 {i:02d}:00:00",
                    "raw_forecast": 100.0,
                    "adjusted_forecast": 101.0,
                    "observed": 110.0,
                    "relative_correction": 0.01,
                    "method_label": "EAF-MAS-C",
                    "is_event_window": True,
                }
            )
            correction = 40.0 if i == 9 else 5.0
            rows.append(
                {
                    "case_id": "strong_case",
                    "channel_id": "N060",
                    "horizon_step": i + 1,
                    "timestamp": f"2023-01-02 {i:02d}:00:00",
                    "raw_forecast": 1000.0,
                    "adjusted_forecast": 1000.0 + correction,
                    "observed": 1040.0,
                    "relative_correction": correction / 1000.0,
                    "method_label": "EAF-MAS-C",
                    "is_event_window": i >= 6,
                }
            )
        focus = fig4_focus_window_for_plot(cases, pd.DataFrame(rows), window_size=8)

        self.assertEqual(focus["case_id"], "strong_case")
        self.assertEqual(focus["channel_id"], "N060")
        self.assertEqual(len(focus["rows"]), 8)
        self.assertIn("2023-01-02 09:00:00", set(focus["rows"]["timestamp"]))
        self.assertGreater(float(focus["rows"]["adjusted_forecast"].max() - focus["rows"]["raw_forecast"].min()), 0.0)

    def test_fig4_builds_ape_reduction_heatmap_for_local_cells(self):
        from experiments.visualization.eaf_viz.plots_main import _robust_heatmap_limit, fig4_error_reduction_heatmap

        focus_rows = pd.DataFrame(
            [
                {"timestamp": "2023-01-02 08:00:00"},
                {"timestamp": "2023-01-02 09:00:00"},
            ]
        )
        hf = pd.DataFrame(
            [
                {
                    "case_id": "strong_case",
                    "channel_id": "N060",
                    "station_name": "Times_Sq_42_St_N_Q_R_W_S_1_2_3_7_42_St_A_C_E_Bryant_Pk_B_D_F_M_5_Av",
                    "timestamp": "2023-01-02 08:00:00",
                    "raw_forecast": 80.0,
                    "adjusted_forecast": 90.0,
                    "observed": 100.0,
                    "relative_correction": 0.125,
                    "method_label": "EAF-MAS-C",
                    "is_adjusted_channel": True,
                },
                {
                    "case_id": "strong_case",
                    "channel_id": "N060",
                    "station_name": "Times_Sq_42_St_N_Q_R_W_S_1_2_3_7_42_St_A_C_E_Bryant_Pk_B_D_F_M_5_Av",
                    "timestamp": "2023-01-02 09:00:00",
                    "raw_forecast": 120.0,
                    "adjusted_forecast": 110.0,
                    "observed": 100.0,
                    "relative_correction": -0.083,
                    "method_label": "EAF-MAS-C",
                    "is_adjusted_channel": True,
                },
                {
                    "case_id": "strong_case",
                    "channel_id": "A013",
                    "station_name": "49_St",
                    "timestamp": "2023-01-02 08:00:00",
                    "raw_forecast": 90.0,
                    "adjusted_forecast": 95.0,
                    "observed": 100.0,
                    "relative_correction": 0.056,
                    "method_label": "EAF-MAS-C",
                    "is_adjusted_channel": True,
                },
                {
                    "case_id": "strong_case",
                    "channel_id": "A013",
                    "station_name": "49_St",
                    "timestamp": "2023-01-02 09:00:00",
                    "raw_forecast": 90.0,
                    "adjusted_forecast": 80.0,
                    "observed": 100.0,
                    "relative_correction": -0.111,
                    "method_label": "EAF-MAS-C",
                    "is_adjusted_channel": True,
                },
            ]
        )

        heatmap = fig4_error_reduction_heatmap(hf, "strong_case", focus_rows, top_k=2)

        self.assertEqual(heatmap["value_label"], "raw APE - adjusted APE (percentage points)")
        self.assertEqual(heatmap["matrix"].shape, (2, 2))
        self.assertGreater(float(heatmap["matrix"][0, 0]), 0.0)
        self.assertLess(float(heatmap["matrix"][1, 1]), 0.0)
        self.assertIn("Times Sq-42 St / Bryant Pk", heatmap["channel_labels"][0])
        self.assertLessEqual(_robust_heatmap_limit(heatmap["matrix"].ravel()), 5.0)

    def test_ablation_materializer_creates_baseline_and_safety_variants(self):
        from experiments.visualization.eaf_viz.plots_main import materialize_ablation_pareto

        table = materialize_ablation_pareto(_case_rows())

        self.assertGreaterEqual(table["variant"].nunique(), 5)
        self.assertIn("PT-MOMENT", set(table["variant"]))
        self.assertNotIn("LP-MOMENT", set(table["variant"]))
        self.assertIn("EAF-MAS-C / summary-only evidence", set(table["variant"]))
        strict = table[table["variant"] == "EAF-MAS-C / summary-only evidence"].iloc[0]
        self.assertGreaterEqual(float(strict["calibration_coverage"]), 0.0)
        self.assertTrue(bool(strict["post_hoc_visualization_only"]))

    def test_ablation_uses_canonical_event_window_for_explanation_only_mode(self):
        from experiments.visualization.eaf_viz.plots_main import materialize_ablation_pareto

        cases = _case_rows()
        cases.loc[cases["method_label"] == "EAF-MAS-X", "event_window_raw_wape"] = 7.0
        cases.loc[cases["method_label"] == "EAF-MAS-C", "event_window_raw_wape"] = 21.0

        table = materialize_ablation_pareto(cases)
        lp = table[table["variant"] == "PT-MOMENT"].iloc[0]
        x = table[table["variant"] == "EAF-MAS-X"].iloc[0]

        self.assertEqual(float(x["event_window_wape"]), float(lp["event_window_wape"]))

    def test_case_card_extractor_uses_structured_llm_reasoning_without_markdown_duplication(self):
        from experiments.visualization.eaf_viz.case_cards import build_case_card_record

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "case"
            (root / "predictions").mkdir(parents=True)
            (root / "reports").mkdir()
            (root / "logs").mkdir()
            (root / "explanations").mkdir()
            prediction = {
                "request": {"date": "2023-06-19 10:00:00", "mode": "event_adapter_frozen_moment", "horizon": 192, "station_scope": "event_venue28"},
                "raw_metrics": {"wape": 12.0},
                "metrics": {"wape": 11.0},
                "decision": {"controller_allowed": True, "correction_bound": 0.05, "adjusted_channels": ["N060"], "included_units": [{"station_channel": "N060", "reason": "near event"}], "excluded_units": [{"station_channel": "A013", "exclusion_reason": "outside mask"}], "correction_stats": {"max_abs_correction": 0.04}},
                "evidence": {"historical_event_cases": [{"historical_event_title": "Past street event", "split": "train_val_memory", "residual_direction": "increase", "median_lp_moment_correction": "+1.0%"}]},
            }
            audit = {"source_validity_score": 0.65, "geo_consistency_score": 0.9, "temporal_alignment_score": 0.95, "residual_support_score": 0.8, "evidence_validity_score": 0.84}
            llm = {
                "result": {
                    "multi_hop_reasoning": "event -> venue -> station -> LP-MOMENT residual memory -> bounded correction",
                    "calibration_rationale": "LP-MOMENT raw forecast is bounded at 0.05",
                    "uncertainty_and_abstention": "weak citation but strong LP MOMENT memory",
                }
            }
            (root / "predictions" / "case.json").write_text(json.dumps(prediction), encoding="utf-8")
            (root / "reports" / "evidence_audit.json").write_text(json.dumps(audit), encoding="utf-8")
            (root / "logs" / "llm_explanation_event_adapter_frozen_moment.json").write_text(json.dumps(llm), encoding="utf-8")
            (root / "explanations" / "case.md").write_text("## Forecast-Time Model Reasoning\nduplicate\n\n## Forecast-Time Model Reasoning\nduplicate", encoding="utf-8")

            record = build_case_card_record(root, "Bounded Event Intervention")

        self.assertEqual(record["method_label"], "EAF-MAS-C")
        self.assertIn("event -> venue -> station", record["multi_hop_reasoning"])
        self.assertNotIn("duplicate", record["multi_hop_reasoning"])
        self.assertEqual(record["included_count"], 1)
        self.assertEqual(record["excluded_count"], 1)
        self.assertEqual(record["historical_memory_count"], 1)
        self.assertEqual(record["case_dir"], "case")
        self.assertIn("PT-MOMENT", record["multi_hop_reasoning"])
        self.assertIn("PT-MOMENT", record["calibration_rationale"])
        self.assertIn("PT-MOMENT", record["uncertainty"])
        self.assertNotIn("LP-MOMENT", record["multi_hop_reasoning"])
        self.assertNotIn("LP-MOMENT", record["calibration_rationale"])
        self.assertNotIn("LP MOMENT", record["uncertainty"])


if __name__ == "__main__":
    unittest.main()
