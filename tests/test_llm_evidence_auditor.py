import unittest


class LLMEvidenceAuditScorerTests(unittest.TestCase):
    def test_llm_audit_scores_vary_by_event_item(self):
        from agents.forecast_evidence_auditor import ForecastEvidenceAuditor
        from agents.llm_evidence_auditor import LLMEvidenceAuditScorer

        def fake_completion(payload):
            rows = []
            for item in payload["items_to_score"]:
                title = (item.get("event_title") or "").lower()
                if "times square" in title or "tsq" in title:
                    rows.append({
                        "item_id": item["item_id"],
                        "source_relevance_score": 0.86,
                        "temporal_admissibility_score": 0.91,
                        "geo_station_consistency_score": 0.88,
                        "event_station_linking_score": 0.89,
                        "residual_memory_support_score": 0.72,
                        "conflict_severity": 0.08,
                        "confidence": 0.84,
                        "decision": "accept_for_calibration",
                        "rationale": "Times Square item links to supplied station and residual memory.",
                        "referenced_item_ids": [item["item_id"]],
                    })
                else:
                    rows.append({
                        "item_id": item["item_id"],
                        "source_relevance_score": 0.42,
                        "temporal_admissibility_score": 0.76,
                        "geo_station_consistency_score": 0.31,
                        "event_station_linking_score": 0.36,
                        "residual_memory_support_score": 0.18,
                        "conflict_severity": 0.41,
                        "confidence": 0.46,
                        "decision": "abstain",
                        "rationale": "The supplied station relation is weak.",
                        "referenced_item_ids": [item["item_id"]],
                    })
            return {"item_audits": rows, "aggregate": {}}

        audit = ForecastEvidenceAuditor().audit(
            anchor_time="2023-06-19 10:00:00",
            horizon_hours=192,
            structured_events=[
                {"title": "TSQ LIVE Times Square", "event_time": "2023-06-20 17:00:00", "location": "Times Square", "event_type": "Street Event", "impact_tier": "A", "channel_name": "N060__Times_Sq_42_St"},
                {"title": "Remote Park Maintenance", "event_time": "2023-06-21 12:00:00", "location": "Central Park", "event_type": "Maintenance", "impact_tier": "C", "channel_name": "N044__81_St_Museum_of_Natural_History"},
            ],
            evidence_sources=[{"title": "NYC permitted event", "accepted": True, "source_type": "official", "source_time": "2023-06-01 00:00:00"}],
            channel_names=["N060__Times_Sq_42_St", "N044__81_St_Museum_of_Natural_History"],
            adjusted_channel_candidates=["N060__Times_Sq_42_St"],
            historical_event_cases=[{"event_type": "Street Event", "median_lp_moment_correction": 0.01, "n_eff": 10}],
            residual_cases=["[residual_case type=Street Event day=weekday rank=top32 correction=+1.0%] Statistics (N=10): median=+1.0%"],
            llm_scorer=LLMEvidenceAuditScorer(completion_fn=fake_completion),
        )
        self.assertEqual(audit["audit_version"], "hard_gate_plus_llm_v3")
        scores = [row["geo_station_consistency_score"] for row in audit["llm_item_audits"]]
        self.assertGreater(max(scores) - min(scores), 0.30)
        self.assertGreater(audit["source_validity_score"], 0.0)
        self.assertIn(audit["llm_final_decision"], {"accept_for_calibration", "accept_for_explanation", "abstain"})

    def test_hard_gate_overrides_llm_acceptance_for_out_of_horizon_event(self):
        from agents.forecast_evidence_auditor import ForecastEvidenceAuditor
        from agents.llm_evidence_auditor import LLMEvidenceAuditScorer

        def overconfident_completion(payload):
            item = payload["items_to_score"][0]
            return {
                "item_audits": [{
                    "item_id": item["item_id"],
                    "source_relevance_score": 0.99,
                    "temporal_admissibility_score": 0.99,
                    "geo_station_consistency_score": 0.99,
                    "event_station_linking_score": 0.99,
                    "residual_memory_support_score": 0.99,
                    "conflict_severity": 0.0,
                    "confidence": 0.99,
                    "decision": "accept_for_calibration",
                    "rationale": "Overconfident but unsupported by hard gate.",
                    "referenced_item_ids": [item["item_id"]],
                }],
                "aggregate": {"decision": "accept_for_calibration"},
            }

        audit = ForecastEvidenceAuditor().audit(
            anchor_time="2023-06-19 10:00:00",
            horizon_hours=24,
            structured_events=[{"title": "Late Event", "event_time": "2023-06-25 17:00:00", "location": "Times Square", "event_type": "Street Event", "impact_tier": "A", "channel_name": "N060__Times_Sq_42_St"}],
            evidence_sources=[],
            channel_names=["N060__Times_Sq_42_St"],
            adjusted_channel_candidates=["N060__Times_Sq_42_St"],
            llm_scorer=LLMEvidenceAuditScorer(completion_fn=overconfident_completion),
        )
        self.assertIn("event_time_outside_horizon", audit["severe_conflict_flags"])
        self.assertEqual(audit["llm_final_decision"], "reject")
        self.assertLessEqual(audit["temporal_alignment_score"], 0.20)


if __name__ == "__main__":
    unittest.main()
