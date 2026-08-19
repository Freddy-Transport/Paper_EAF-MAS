import unittest


class ExplanationQualityTests(unittest.TestCase):
    def test_evaluator_scores_citations_reasoning_and_abstain_consistency(self):
        from agents.explanation_evaluator import evaluate_explanation_quality

        markdown = """
## Event Evidence
### Accepted External Evidence
- [NYC DOT Plaza Event](https://www.nyc.gov/example): Herald Square event.

## Model Reasoning
The event links venue to station through walking distance, then maps event timing to hourly demand.

## Calibration Decision
The system abstains because evidence is not strong enough for bounded correction.
"""
        result = evaluate_explanation_quality(
            markdown=markdown,
            accepted_evidence=[{"url": "https://www.nyc.gov/example", "relevance_score": 0.9}],
            decision={"abstain": True, "reason": "evidence weak"},
            residual_cases=[
                "[residual_case type=Plaza Event day=weekday rank=top64 correction=+4.0%]",
                "[residual_case type=Plaza Event day=weekday rank=top64 correction=+4.0%]",
            ],
        )

        self.assertEqual(result["accepted_evidence_count"], 1)
        self.assertEqual(result["citation_coverage"], 1.0)
        self.assertTrue(result["station_event_linking"])
        self.assertTrue(result["multi_hop_reasoning"])
        self.assertTrue(result["calibration_decision_consistent"])
        self.assertFalse(result["residual_cases_deduplicated"])

    def test_evaluator_rejects_abstain_mismatch(self):
        from agents.explanation_evaluator import evaluate_explanation_quality

        markdown = "## Calibration Decision\nThe agent applies a positive correction."
        result = evaluate_explanation_quality(
            markdown=markdown,
            accepted_evidence=[],
            decision={"abstain": True},
            residual_cases=[],
        )

        self.assertFalse(result["calibration_decision_consistent"])
        self.assertEqual(result["citation_coverage"], 0.0)

    def test_evaluator_scores_residual_memory_skill_explanation_quality(self):
        from agents.explanation_evaluator import evaluate_explanation_quality

        markdown = """
## Event Evidence
### Forecast Window
- Anchor time: 2023-06-19 10:00:00
### Structured Future Events within Horizon
- TSQ LIVE | tier=A | stations=N060__Times_Sq_42_St
### Historical Event Memory from Train/Validation
- Times Square plaza case | split=train_val | station=N060__Times_Sq_42_St | direction=increase | historical residual=+7.0%
### Residual Pattern Evidence
- [residual_case type=Plaza Event day=weekday rank=top64 correction=+0.9%]

## Forecast-Time Model Reasoning
The current event maps from Times Square venue to a nearby station, then to evening forecast hours.
Similar train/validation cases show a positive residual pattern, so bounded calibration is plausible.

## Calibration Decision
The system applies a bounded correction.
"""
        result = evaluate_explanation_quality(
            markdown=markdown,
            accepted_evidence=[],
            decision={"abstain": False, "adjusted_channels": ["N060__Times_Sq_42_St"]},
            residual_cases=["[residual_case type=Plaza Event day=weekday rank=top64 correction=+0.9%]"],
            structured_events=[{"title": "TSQ LIVE", "event_type": "Plaza Event", "impact_tier": "A"}],
            historical_event_cases=[
                {
                    "historical_event_title": "Times Square plaza case",
                    "event_type": "Plaza Event",
                    "matched_station": "N060__Times_Sq_42_St",
                    "residual_direction": "increase",
                    "historical_baseline_residual": "+7.0%",
                }
            ],
        )

        self.assertGreaterEqual(result["evidence_coverage"], 0.75)
        self.assertGreaterEqual(result["residual_case_relevance"], 0.5)
        self.assertGreaterEqual(result["multi_hop_completeness"], 0.8)
        self.assertGreater(result["groundedness"], 0.0)
        self.assertEqual(result["unsupported_claim_rate"], 0.0)
        self.assertEqual(result["leakage_free_rate"], 1.0)
        self.assertTrue(result["calibration_decision_consistent"])


if __name__ == "__main__":
    unittest.main()
