import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class ForecastEvidenceAuditorTests(unittest.TestCase):
    def test_auditor_rejects_post_anchor_sources_and_scores_temporal_residual_support(self):
        from agents.forecast_evidence_auditor import ForecastEvidenceAuditor

        auditor = ForecastEvidenceAuditor()
        audit = auditor.audit(
            anchor_time="2023-06-19 10:00:00",
            horizon_hours=192,
            structured_events=[{
                "title": "TSQ LIVE: 45/46 Plaza Programming",
                "event_time": "2023-06-20 17:00:00",
                "location": "43/44 Broadway Pedestrian Plaza Times Square",
                "event_type": "Street Event",
                "impact_tier": "A",
                "channel_name": "N060__Times_Sq_42_St",
            }],
            evidence_sources=[{
                "title": "post-event recap",
                "url": "https://example.org/recap",
                "accepted": True,
                "source_time": "2023-06-21 12:00:00",
                "source_type": "official",
            }],
            channel_names=["N060__Times_Sq_42_St", "A022__34_St_Herald_Sq"],
            adjusted_channel_candidates=["N060__Times_Sq_42_St"],
            historical_event_cases=[{
                "historical_event_title": "Spring Broadway Fair",
                "split": "train_val_memory",
                "event_type": "Street Event",
                "day_type": "weekday",
                "rank_group": "top32",
                "residual_direction": "increase",
                "median_lp_moment_correction": 0.008,
                "iqr": 0.021,
                "n_eff": 26,
            }],
            residual_cases=["[residual_case type=Street Event day=weekday rank=top32 correction=+0.8%] Statistics (N=26): p25=-0.9%, median=+0.8%, p75=+1.2%"],
        )

        self.assertLess(audit["source_validity_score"], 0.60)
        self.assertGreaterEqual(audit["temporal_alignment_score"], 0.70)
        self.assertGreaterEqual(audit["residual_support_score"], 0.50)
        self.assertIn("source_time_after_anchor", audit["severe_conflict_flags"])
        self.assertTrue(audit["included_units"])
        self.assertTrue(audit["excluded_units"])


class CalibrationControllerTests(unittest.TestCase):
    def test_controller_allows_only_thresholded_in_mask_bounded_corrections(self):
        from agents.calibration_controller import CalibrationController

        controller = CalibrationController(correction_bound=0.05)
        raw = np.ones((2, 3), dtype=np.float32) * 100.0
        correction = np.array([[0.01, 0.02, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        proposed = raw * (1.0 + correction)
        high_audit = {
            "source_validity_score": 0.8,
            "geo_consistency_score": 0.8,
            "temporal_alignment_score": 0.9,
            "residual_support_score": 0.7,
            "severe_conflict_flags": [],
            "included_units": [{"station_channel": "N060__Times_Sq_42_St"}],
            "excluded_units": [{"station_channel": "A022__34_St_Herald_Sq", "exclusion_reason": "not in event mask"}],
        }

        allowed = controller.apply(
            raw_forecast=raw,
            proposed_adjusted=proposed,
            correction=correction,
            channel_names=["N060__Times_Sq_42_St", "A022__34_St_Herald_Sq"],
            adjusted_channel_candidates=["N060__Times_Sq_42_St"],
            event_channel_mask=np.array([True, False]),
            evidence_audit=high_audit,
        )
        self.assertTrue(allowed["controller_allowed"])
        self.assertFalse(allowed["abstain"])
        self.assertTrue(np.allclose(np.asarray(allowed["final_adjusted_forecast"]), proposed))
        self.assertEqual(allowed["adjusted_channels"], ["N060__Times_Sq_42_St"])
        self.assertGreater(allowed["correction_stats"]["max_abs_correction"], 0.0)

        low_audit = dict(high_audit, geo_consistency_score=0.2)
        blocked = controller.apply(
            raw_forecast=raw,
            proposed_adjusted=proposed,
            correction=correction,
            channel_names=["N060__Times_Sq_42_St", "A022__34_St_Herald_Sq"],
            adjusted_channel_candidates=["N060__Times_Sq_42_St"],
            event_channel_mask=np.array([True, False]),
            evidence_audit=low_audit,
        )
        self.assertFalse(blocked["controller_allowed"])
        self.assertTrue(blocked["abstain"])
        self.assertTrue(np.allclose(np.asarray(blocked["final_adjusted_forecast"]), raw))
        self.assertIn("geo_consistency_score", blocked["abstention_reason"])

        outside_mask = controller.apply(
            raw_forecast=raw,
            proposed_adjusted=proposed,
            correction=correction,
            channel_names=["N060__Times_Sq_42_St", "A022__34_St_Herald_Sq"],
            adjusted_channel_candidates=["A022__34_St_Herald_Sq"],
            event_channel_mask=np.array([True, False]),
            evidence_audit=high_audit,
        )
        self.assertFalse(outside_mask["controller_allowed"])
        self.assertIn("outside event-station-channel mask", outside_mask["abstention_reason"])


class ExplanationAuditStructureTests(unittest.TestCase):
    def test_markdown_contains_eight_sections_and_no_rejected_diagnostics(self):
        from agents.paper_workflow import (
            CalibrationDecisionSpec,
            EventEvidenceSpec,
            ForecastRequestSpec,
            NumericalForecastSpec,
            build_explanation_markdown,
        )

        request = ForecastRequestSpec(date="2023-06-19 10:00:00", horizon=192, station_scope="event_venue28", mode="event_adapter_frozen_moment")
        numerical = NumericalForecastSpec(raw_forecast=[[1.0, 2.0]], channel_names=["N060__Times_Sq_42_St"], timestamps=["2023-06-19 10:00:00", "2023-06-19 11:00:00"], model_type="LP-MOMENT")
        evidence = EventEvidenceSpec(
            structured_events=[{"title": "TSQ LIVE", "event_time": "2023-06-20 17:00:00", "source_type": "permit", "source_time": "2023-06-01 00:00:00"}],
            historical_event_cases=[{"historical_event_title": "Spring Broadway Fair", "split": "train", "residual_direction": "increase", "median_lp_moment_correction": 0.008, "iqr": 0.02, "n_eff": 26}],
            evidence_audit={
                "source_validity_score": 0.8,
                "geo_consistency_score": 0.8,
                "temporal_alignment_score": 0.9,
                "residual_support_score": 0.7,
                "conflict_flags": [],
                "included_units": [{"station_channel": "N060__Times_Sq_42_St", "reason": "Times Square station match", "gate_score": 0.9}],
                "excluded_units": [{"station_channel": "A022__34_St_Herald_Sq", "exclusion_reason": "outside event mask"}],
            },
            has_major_event=True,
        )
        decision = CalibrationDecisionSpec(
            mode="event_adapter_frozen_moment",
            adjusted_channels=["N060__Times_Sq_42_St"],
            abstain=False,
            confidence=0.8,
            reason="controller allowed bounded correction",
            correction_bound=0.05,
            controller_allowed=True,
            correction_stats={"max_abs_correction": 0.012, "mean_abs_correction": 0.004},
            included_units=[{"station_channel": "N060__Times_Sq_42_St", "reason": "Times Square station match"}],
            excluded_units=[{"station_channel": "A022__34_St_Herald_Sq", "exclusion_reason": "outside event mask"}],
        )

        markdown = build_explanation_markdown(
            request=request,
            evidence=evidence,
            decision=decision,
            metrics={"wape": 1.2},
            model_explanation="Fallback rationale",
            llm_markdown="Factual evidence: structured event.\n\nResidual analogues: train memory.\n\nCalibration rationale: bounded.\n\nUncertainty: source limits.\n\nRisk control: channel mask.",
            numerical=numerical,
            raw_metrics={"wape": 2.0},
            event_window_metrics={"wape": 1.1},
            adjusted_channel_metrics={"wape": 0.9},
            adapter_checkpoint="experiments/outputs/event_adapter_formal_frozen_real_lpmoment_test",
        )

        for heading in [
            "## 1. Forecast Request",
            "## 2. Raw Forecast",
            "## 3. Forecast-time Evidence Audit",
            "## 4. Candidate Event-Station-Channel Units",
            "## 5. Historical Residual Memory",
            "## 6. Calibration Controller",
            "## 7. Adjusted Forecast",
            "## 8. Explanation",
        ]:
            self.assertIn(heading, markdown)
        self.assertIn("why this event is related", markdown.lower())
        self.assertIn("Rejected Retrieval Diagnostics", markdown, "sanity check before replacement") if False else self.assertNotIn("Rejected Retrieval Diagnostics", markdown)

    def test_llm_payload_includes_audit_controller_and_units(self):
        from agents.forecast_explanation_agent import ForecastExplanationAgent

        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "llm.json"
            agent = ForecastExplanationAgent(base_url="http://127.0.0.1:9/v1", model="fake", timeout_s=0.1)
            agent.explain(
                forecast_request={"date": "2023-06-19 10:00:00"},
                structured_events=[{"title": "TSQ LIVE"}],
                retrieved_sources=[],
                station_scope={"scope": "event_venue28"},
                raw_forecast=[[1.0, 2.0]],
                channel_names=["N060__Times_Sq_42_St"],
                calibration_decision={"abstain": False, "adjusted_channels": ["N060__Times_Sq_42_St"]},
                rag_residual_context="residual context",
                evidence_audit={"source_validity_score": 0.8, "included_units": [{"station_channel": "N060__Times_Sq_42_St"}], "excluded_units": []},
                calibration_controller={"controller_allowed": True, "correction_stats": {"max_abs_correction": 0.01}},
                log_path=log_path,
            )
            payload = json.loads(log_path.read_text())["request_payload"]
            self.assertIn("evidence_audit", payload)
            self.assertIn("calibration_controller", payload)
            self.assertEqual(payload["evidence_audit"]["included_units"][0]["station_channel"], "N060__Times_Sq_42_St")


if __name__ == "__main__":
    unittest.main()
