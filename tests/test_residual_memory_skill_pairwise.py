import json
import tempfile
import unittest
from pathlib import Path


def _payload(date: str, mode: str, skill_count: int, quality_boost: float = 0.0) -> dict:
    raw = [[100.0, 110.0, 120.0]]
    adjusted = [[100.0, 110.0, 120.0]]
    quality = {
        "evidence_coverage": 0.55 + quality_boost,
        "residual_case_relevance": 0.50 + quality_boost,
        "multi_hop_completeness": 0.50 + quality_boost,
        "groundedness": 0.45 + quality_boost,
        "unsupported_claim_rate": max(0.0, 0.20 - quality_boost),
        "leakage_free_rate": 1.0,
        "redundancy_rate": max(0.0, 0.20 - quality_boost),
        "skill_guidance_coverage": 1.0 if skill_count else 0.0,
        "selected_residual_memory_skill_count": skill_count,
        "calibration_decision_consistent": True,
    }
    return {
        "request": {"date": date, "mode": mode, "station_scope": "event_venue28"},
        "numerical": {"raw_forecast": raw, "ground_truth": raw, "channel_names": ["A013"], "timestamps": ["h0", "h1", "h2"]},
        "adjusted_forecast": adjusted,
        "metrics": {"wape": 0.0},
        "raw_metrics": {"wape": 0.0},
        "explanation_quality": quality,
        "evidence": {
            "has_major_event": True,
            "structured_events": [{"title": "Case Event", "event_type": "Plaza Event", "impact_tier": "A"}],
            "historical_event_cases": [{"historical_event_title": "History"}],
            "local_residual_cases": ["[residual_case type=Plaza Event day=weekday rank=top64 correction=+1.0%]"],
            "model_assisted_summaries": [{"summary": "Qwen-Plus summary"}],
        },
        "decision": {"abstain": True, "adjusted_channels": []},
    }


class ResidualMemorySkillPairwiseTests(unittest.TestCase):
    def test_pairwise_report_requires_matching_forecasts_and_writes_outputs(self):
        from experiments.run_residual_memory_skill_pairwise import run_pairwise_study

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inputs = {}
            for mode, skill_count, boost in [
                ("residual_rag_no_skill", 0, 0.0),
                ("residual_rag_static_skill", 1, 0.2),
                ("residual_rag_evolving_skill", 1, 0.3),
            ]:
                path = root / f"{mode}.json"
                path.write_text(json.dumps(_payload("2023-06-19 10:00:00", mode, skill_count, boost)), encoding="utf-8")
                inputs[mode] = [str(path)]

            summary = run_pairwise_study(
                output_root=root / "out",
                no_skill_prediction_json=inputs["residual_rag_no_skill"],
                static_skill_prediction_json=inputs["residual_rag_static_skill"],
                evolving_skill_prediction_json=inputs["residual_rag_evolving_skill"],
            )

            self.assertEqual(summary["window_count"], 1)
            self.assertTrue(summary["forecast_arrays_identical"])
            self.assertGreater(summary["mean_delta_vs_no_skill"]["residual_rag_static_skill"]["groundedness"], 0.0)
            self.assertGreater(summary["mean_delta_vs_no_skill"]["residual_rag_evolving_skill"]["skill_guidance_coverage"], 0.0)
            self.assertTrue((root / "out" / "tables" / "explanation_quality_pairwise.tex").is_file())
            self.assertTrue((root / "out" / "figures" / "skill_pairwise_quality_delta.png").is_file())
            self.assertTrue((root / "out" / "reports" / "residual_memory_skill_pairwise_summary.md").is_file())

    def test_pairwise_marks_forecast_mismatch_without_using_wape_as_skill_metric(self):
        from experiments.run_residual_memory_skill_pairwise import run_pairwise_study

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            no_skill = _payload("2023-06-19 10:00:00", "residual_rag_no_skill", 0, 0.0)
            static = _payload("2023-06-19 10:00:00", "residual_rag_static_skill", 1, 0.2)
            static["adjusted_forecast"] = [[101.0, 110.0, 120.0]]
            p0, p1 = root / "no.json", root / "static.json"
            p0.write_text(json.dumps(no_skill), encoding="utf-8")
            p1.write_text(json.dumps(static), encoding="utf-8")

            summary = run_pairwise_study(
                output_root=root / "out",
                no_skill_prediction_json=[str(p0)],
                static_skill_prediction_json=[str(p1)],
                evolving_skill_prediction_json=[str(p1)],
            )

            self.assertFalse(summary["forecast_arrays_identical"])
            self.assertIn("wape", summary["sanity_metrics"]["residual_rag_no_skill"])
            self.assertNotIn("wape", summary["quality_metrics"])


if __name__ == "__main__":
    unittest.main()
