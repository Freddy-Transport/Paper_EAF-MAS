import json
import tempfile
import unittest
from pathlib import Path


def _payload(date: str, abstain: bool = False, boost: float = 0.0) -> dict:
    raw = [[100.0, 110.0, 120.0, 130.0]]
    adjusted = [[100.0, 110.0, 120.0, 130.0]]
    return {
        "request": {"date": date, "mode": "event_adapter_frozen_moment", "station_scope": "event_venue28"},
        "numerical": {
            "raw_forecast": raw,
            "ground_truth": raw,
            "channel_names": ["N060__Times_Sq_42_St"],
            "timestamps": ["h0", "h1", "h2", "h3"],
        },
        "adjusted_forecast": adjusted,
        "evidence": {
            "has_major_event": not abstain,
            "structured_events": [{"title": "TSQ LIVE", "event_type": "plaza programming", "impact_tier": "A"}],
            "model_assisted_summaries": [{"summary": "A scheduled plaza event may affect nearby station demand."}],
            "historical_event_cases": [
                {"event_type": "plaza programming", "day_type": "weekday", "station_rank_group": "top32", "residual_direction": "positive", "median_correction": 0.03, "iqr": 0.01, "n_eff": 8},
                {"event_type": "sports", "day_type": "weekend", "station_rank_group": "top128", "residual_direction": "negative", "median_correction": -0.02, "iqr": 0.04, "n_eff": 2},
            ],
            "local_residual_cases": ["case a", "case b"],
            "evidence_audit": {
                "source_validity_score": 0.65,
                "geo_consistency_score": 0.85,
                "temporal_alignment_score": 0.95,
                "semantic_consistency_score": 0.90,
                "residual_support_score": 0.80,
            },
        },
        "decision": {"abstain": abstain, "controller_allowed": not abstain, "adjusted_channels": [] if abstain else ["N060__Times_Sq_42_St"]},
        "explanation_quality": {
            "evidence_coverage": 0.55 + boost,
            "residual_case_relevance": 0.45 + boost,
            "multi_hop_completeness": 0.50 + boost,
            "groundedness": 0.50 + boost,
            "unsupported_claim_rate": max(0.0, 0.20 - boost),
            "leakage_free_rate": 1.0,
            "calibration_decision_consistent": True,
            "redundancy_rate": 0.25,
        },
        "explanation_markdown": "Forecast-time evidence links the event to Times Square and historical memory supports cautious reasoning.",
        "metrics": {"wape": 0.0},
        "raw_metrics": {"wape": 0.0},
    }


class AutoSkillEffectivenessStudyTests(unittest.TestCase):
    def test_residual_memory_coverage_audit_flags_missing_event_types(self):
        from experiments.run_autoskill_effectiveness_study import audit_residual_memory_coverage

        payloads = [
            {
                "request": {"date": "2023-06-01 10:00:00"},
                "evidence": {
                    "structured_events": [
                        {
                            "title": "Minor plaza activation",
                            "event_type": "Plaza Partner Event",
                            "impact_tier": "C",
                            "channel_name": "A013__49_St",
                        }
                    ],
                    "historical_event_cases": [],
                },
            }
        ]
        legacy_docs = [
            {
                "event_type": "Special Event",
                "impact_tier": "A",
                "rank_group": "top64",
                "day_type": "weekday",
                "median_correction": 0.01,
                "n_eff": 10,
            }
        ]

        audit = audit_residual_memory_coverage(payloads, legacy_docs, hit_threshold=0.55)

        self.assertTrue(audit["rebuild_required"])
        self.assertEqual(audit["window_count"], 1)
        self.assertEqual(audit["empty_retrieval_rate"], 1.0)
        self.assertLess(audit["hit_at_3"], 0.80)
        self.assertIn("Plaza Partner Event|C", audit["coverage_by_event_type_tier"])

    def test_build_coverage_residual_memory_v2_is_train_val_only_and_covers_no_event(self):
        from experiments.run_autoskill_effectiveness_study import build_coverage_residual_memory_v2

        events = [
            {
                "title": "Validation plaza case",
                "event_time": "2023-02-18 12:00:00",
                "event_type": "Plaza Partner Event",
                "impact_tier": "C",
                "station_rank": 45,
            },
            {
                "title": "Test-only event must not leak",
                "event_time": "2023-06-18 12:00:00",
                "event_type": "Street Festival",
                "impact_tier": "A",
                "station_rank": 20,
            },
        ]
        legacy_docs = [
            {
                "event_type": "Plaza Partner Event",
                "impact_tier": "C",
                "rank_group": "top64",
                "day_type": "weekend",
                "median_correction": -0.02,
                "iqr": 0.01,
                "n_eff": 8,
            }
        ]

        docs = build_coverage_residual_memory_v2(
            events,
            legacy_docs,
            train_start="2022-02-01 00:00:00",
            val_end="2023-02-28 23:00:00",
        )

        event_types = {doc["event_type"] for doc in docs}
        self.assertIn("Plaza Partner Event", event_types)
        self.assertIn("no_event", event_types)
        self.assertNotIn("Street Festival", event_types)
        self.assertTrue(all(doc["split"] in {"train", "val", "train_val_memory"} for doc in docs))
        for doc in docs:
            for key in ["median_correction", "iqr", "n_eff", "residual_direction", "source"]:
                self.assertIn(key, doc)

    def test_full_metric_rows_build_full_window_payloads(self):
        from experiments.run_autoskill_effectiveness_study import build_payloads_from_full_metric_rows

        rows = [
            {
                "split": "val",
                "anchor": "9152",
                "date": "2023-02-17 09:00:00",
                "mode": "numerical_only",
                "wape": "10.0",
                "event_wape": "8.0",
                "event_n": "4",
                "non_event_wape": "11.0",
            },
            {
                "split": "val",
                "anchor": "9152",
                "date": "2023-02-17 09:00:00",
                "mode": "event_adapter_frozen_moment",
                "wape": "9.5",
                "event_wape": "7.0",
                "event_n": "4",
                "non_event_wape": "11.0",
            },
            {
                "split": "test",
                "anchor": "12032",
                "date": "2023-06-17 10:00:00",
                "mode": "numerical_only",
                "wape": "20.0",
                "event_wape": "15.0",
                "event_n": "0",
                "non_event_wape": "20.0",
            },
            {
                "split": "test",
                "anchor": "12032",
                "date": "2023-06-17 10:00:00",
                "mode": "event_adapter_frozen_moment",
                "wape": "20.0",
                "event_wape": "15.0",
                "event_n": "0",
                "non_event_wape": "20.0",
            },
        ]
        events = [
            {
                "title": "Validation plaza case",
                "event_time": "2023-02-18 12:00:00",
                "event_type": "Plaza Partner Event",
                "impact_tier": "C",
                "channel_name": "A013__49_St",
            }
        ]
        residual_docs = [
            {
                "event_type": "Plaza Partner Event",
                "impact_tier": "C",
                "rank_group": "top64",
                "day_type": "weekday",
                "median_correction": 0.02,
                "iqr": 0.01,
                "n_eff": 8,
            }
        ]

        val_payloads = build_payloads_from_full_metric_rows(rows, "val", events, residual_docs)
        test_payloads = build_payloads_from_full_metric_rows(rows, "test", events, residual_docs)

        self.assertEqual(len(val_payloads), 1)
        self.assertEqual(len(test_payloads), 1)
        self.assertTrue(val_payloads[0]["evidence"]["has_major_event"])
        self.assertGreater(len(val_payloads[0]["evidence"]["historical_event_cases"]), 0)
        self.assertFalse(test_payloads[0]["evidence"]["has_major_event"])
        self.assertIn("raw_forecast", val_payloads[0]["numerical"])

    def test_effectiveness_study_writes_tables_figures_and_preserves_forecasts(self):
        from experiments.run_autoskill_effectiveness_study import run_autoskill_effectiveness_study

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            val_paths, test_paths = [], []
            for i in range(3):
                p = root / f"val_{i}.json"
                p.write_text(json.dumps(_payload(f"2023-05-0{i+1} 10:00:00", boost=0.1)), encoding="utf-8")
                val_paths.append(str(p))
            for i in range(4):
                p = root / f"test_{i}.json"
                p.write_text(json.dumps(_payload(f"2023-06-0{i+1} 10:00:00", abstain=(i % 2 == 1))), encoding="utf-8")
                test_paths.append(str(p))

            summary = run_autoskill_effectiveness_study(root / "out", val_prediction_json=val_paths, test_prediction_json=test_paths)

            self.assertEqual(summary["test_window_count"], 4)
            self.assertTrue(summary["forecast_arrays_identical"])
            self.assertGreaterEqual(summary["skill_lifecycle"]["val_promoted_count"], 1)
            self.assertEqual(summary["skill_lifecycle"]["test_promoted_count"], 0)
            self.assertGreaterEqual(summary["skill_lifecycle"]["test_active_count"], 1)
            self.assertTrue((root / "out" / "tables" / "skill_lifecycle.tex").is_file())
            self.assertTrue((root / "out" / "tables" / "explanation_quality_ablation.tex").is_file())
            self.assertTrue((root / "out" / "tables" / "residual_memory_relevance.tex").is_file())
            self.assertTrue((root / "out" / "tables" / "abstention_quality.tex").is_file())
            self.assertTrue((root / "out" / "figures" / "skill_evolution_timeline.png").is_file())
            self.assertTrue((root / "out" / "figures" / "explanation_quality_delta_forest.png").is_file())
            self.assertTrue((root / "out" / "figures" / "residual_memory_before_after.png").is_file())
            self.assertTrue((root / "out" / "figures" / "abstention_decision_matrix.png").is_file())
            self.assertTrue((root / "out" / "figures" / "pairwise_judge_winrate.png").is_file())

    def test_skill_improves_explanation_metrics_without_wape_claim(self):
        from experiments.run_autoskill_effectiveness_study import run_autoskill_effectiveness_study

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            val = root / "val.json"
            test = root / "test.json"
            val.write_text(json.dumps(_payload("2023-05-01 10:00:00", boost=0.1)), encoding="utf-8")
            test.write_text(json.dumps(_payload("2023-06-01 10:00:00")), encoding="utf-8")

            summary = run_autoskill_effectiveness_study(root / "out", val_prediction_json=[str(val)], test_prediction_json=[str(test)])
            deltas = summary["quality_delta_vs_no_skill"]

            self.assertGreater(deltas["full_skill"]["residual_case_relevance"], 0.0)
            self.assertGreater(deltas["full_skill"]["multi_hop_completeness"], 0.0)
            self.assertLessEqual(deltas["full_skill"]["unsupported_claim_rate"], 0.0)
            self.assertNotIn("wape", summary["quality_metrics"])

    def test_llm_judge_prompt_excludes_posthoc_metrics(self):
        from experiments.run_autoskill_effectiveness_study import build_pairwise_judge_prompt

        prompt = build_pairwise_judge_prompt("Explanation A", "Explanation B", {"wape": 12.3, "actual": [1, 2, 3]})

        self.assertIn("clarity", prompt.lower())
        self.assertIn("faithfulness", prompt.lower())
        self.assertNotIn("12.3", prompt)
        self.assertNotIn("actual", prompt.lower())
        self.assertNotIn("wape", prompt.lower())


if __name__ == "__main__":
    unittest.main()
