import json
import tempfile
import unittest
from pathlib import Path


def _forecast_payload(date: str = "2023-06-19 10:00:00", quality_boost: float = 0.0) -> dict:
    raw = [[100.0, 110.0, 120.0]]
    adjusted = [[100.0, 110.0, 120.0]]
    return {
        "request": {"date": date, "mode": "event_adapter_frozen_moment", "station_scope": "event_venue28"},
        "numerical": {
            "raw_forecast": raw,
            "ground_truth": raw,
            "channel_names": ["N060__Times_Sq_42_St"],
            "timestamps": ["h0", "h1", "h2"],
        },
        "adjusted_forecast": adjusted,
        "evidence": {
            "has_major_event": True,
            "structured_events": [
                {
                    "title": "TSQ LIVE",
                    "event_type": "plaza programming",
                    "impact_tier": "A",
                    "event_time": "2023-06-20 17:00:00",
                }
            ],
            "accepted_external_evidence": [],
            "model_assisted_summaries": [{"summary": "Model-assisted summary for a Times Square public event."}],
            "historical_event_cases": [
                {
                    "historical_event_title": "Earlier plaza programming",
                    "split": "val",
                    "event_type": "plaza programming",
                    "day_type": "weekday",
                    "station_rank_group": "top32",
                    "residual_direction": "positive",
                    "median_correction": 0.031,
                    "iqr": 0.012,
                    "n_eff": 8,
                }
            ],
            "local_residual_cases": [
                "[residual_case type=plaza programming day=weekday rank=top32 correction=+3.1% n_eff=8]"
            ],
            "evidence_audit": {
                "source_validity_score": 0.65,
                "geo_consistency_score": 0.82,
                "temporal_alignment_score": 0.95,
                "semantic_consistency_score": 0.86,
                "residual_support_score": 0.78,
            },
        },
        "decision": {"abstain": False, "controller_allowed": True, "adjusted_channels": ["N060__Times_Sq_42_St"]},
        "explanation_markdown": (
            "The current event is linked to the Times Square station. Historical train/validation "
            "memory shows similar plaza programming residual correction patterns. The controller "
            "applies bounded calibration."
        ),
        "explanation_quality": {
            "evidence_coverage": 0.70 + quality_boost,
            "residual_case_relevance": 0.72 + quality_boost,
            "multi_hop_completeness": 0.68 + quality_boost,
            "groundedness": 0.69 + quality_boost,
            "unsupported_claim_rate": max(0.0, 0.10 - quality_boost),
            "leakage_free_rate": 1.0,
            "decision_consistency": 1.0,
            "calibration_decision_consistent": True,
        },
        "metrics": {"wape": 17.0},
        "raw_metrics": {"wape": 17.0},
        "runtime": {"OPENAI_API_KEY": "sk-should-not-leak"},
    }


class AutoSkillEvolutionTests(unittest.TestCase):
    def test_experience_pool_records_replay_trace_and_redacts_secrets(self):
        from agents.autoskill_memory import ExperiencePool, build_replay_experience

        with tempfile.TemporaryDirectory() as td:
            pool = ExperiencePool(Path(td))
            experience = build_replay_experience(_forecast_payload(), split="val", experience_id="exp_tsq")
            pool.add(experience)

            rows = pool.load(split="val")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["experience_id"], "exp_tsq")
            self.assertEqual(rows[0]["split"], "val")
            self.assertIn("structured_event", rows[0])
            self.assertIn("historical_residual_memory", rows[0])
            self.assertIn("evidence_audit", rows[0])
            self.assertIn("controller_decision", rows[0])
            text = (Path(td) / "experiences_val.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("sk-should-not-leak", text)
            self.assertNotIn("OPENAI_API_KEY", text)

    def test_generates_four_domain_skills_with_autoskill_schema(self):
        from agents.autoskill_memory import REQUIRED_AUTOSKILL_SKILL_FIELDS, build_replay_experience, evolve_skills

        experiences = [build_replay_experience(_forecast_payload(quality_boost=0.1), split="val")]
        report = evolve_skills(experiences, split="val", mode="evolving_skill")
        categories = {skill["skill_category"] for skill in report["candidate_skills"]}

        self.assertEqual(
            {"routing_skill", "evidence_audit_skill", "residual_memory_skill", "abstention_skill"},
            categories,
        )
        self.assertGreaterEqual(report["promoted_count"], 1)
        self.assertTrue(report["forecast_arrays_identical"])
        for skill in report["candidate_skills"]:
            self.assertTrue(REQUIRED_AUTOSKILL_SKILL_FIELDS.issubset(skill))
            self.assertFalse(skill["validation_metrics"]["forecast_array_changed"])
            self.assertFalse(skill["memory_selection_policy"].get("use_for_prediction"))

    def test_evolving_skill_uses_experience_clusters_not_fixed_four_categories(self):
        from agents.autoskill_memory import REQUIRED_AUTOSKILL_SKILL_FIELDS, build_replay_experience, evolve_skills

        payloads = []
        for idx, (event_type, tier, source, residual, unsupported) in enumerate(
            [
                ("plaza programming", "A", 0.90, 0.80, 0.05),
                ("plaza programming", "A", 0.35, 0.75, 0.18),
                ("sport adult", "B", 0.80, 0.40, 0.08),
                ("street festival", "A", 0.72, 0.65, 0.12),
                ("routine construction", "C", 0.40, 0.30, 0.22),
                ("no event", "D", 1.00, 0.70, 0.03),
            ]
        ):
            payload = _forecast_payload(date=f"2023-06-{idx + 1:02d} 10:00:00")
            payload["evidence"]["structured_events"][0]["event_type"] = event_type
            payload["evidence"]["structured_events"][0]["impact_tier"] = tier
            payload["evidence"]["evidence_audit"]["source_validity_score"] = source
            payload["evidence"]["evidence_audit"]["residual_support_score"] = residual
            payload["explanation_quality"]["unsupported_claim_rate"] = unsupported
            payload["explanation_quality"]["residual_case_relevance"] = residual
            payloads.append(payload)
        experiences = [build_replay_experience(payload, split="val", experience_id=f"exp_{i}") for i, payload in enumerate(payloads)]

        report = evolve_skills(experiences, split="val", mode="evolving_skill")

        self.assertGreater(report["candidate_count"], 4)
        self.assertGreater(report["mutation_count"], report["candidate_count"])
        self.assertGreater(report["rejected_count"], 0)
        self.assertLess(report["promoted_count"], report["mutation_count"])
        self.assertTrue(all(REQUIRED_AUTOSKILL_SKILL_FIELDS.issubset(skill) for skill in report["candidate_skills"]))
        self.assertTrue(all(skill.get("experience_cluster") for skill in report["candidate_skills"]))

    def test_test_split_is_read_only_and_does_not_promote_or_mutate(self):
        from agents.autoskill_memory import build_replay_experience, evolve_skills

        experiences = [build_replay_experience(_forecast_payload(quality_boost=0.2), split="test")]
        report = evolve_skills(experiences, split="test", mode="evolving_skill")

        self.assertTrue(report["test_read_only"])
        self.assertEqual(report["promoted_count"], 0)
        self.assertEqual(report["mutation_count"], 0)
        self.assertTrue(all(not skill["promotion_status"]["promoted"] for skill in report["candidate_skills"]))
        self.assertIn("read-only", report["candidate_skills"][0]["promotion_status"]["reason"])

    def test_residual_memory_skill_selects_more_relevant_cases_without_changing_forecast(self):
        from agents.autoskill_memory import apply_residual_memory_skill

        cases = [
            {"event_type": "sports", "day_type": "weekend", "station_rank_group": "top128", "residual_direction": "negative", "n_eff": 2},
            {"event_type": "plaza programming", "day_type": "weekday", "station_rank_group": "top32", "residual_direction": "positive", "n_eff": 8},
            {"event_type": "concert", "day_type": "weekday", "station_rank_group": "top64", "residual_direction": "positive", "n_eff": 4},
        ]
        skill = {
            "skill_category": "residual_memory_skill",
            "trigger_condition": {
                "event_type": "plaza programming",
                "day_type": "weekday",
                "station_rank_group": "top32",
                "residual_direction": "positive",
            },
            "memory_selection_policy": {
                "select_cases_by": ["event_type", "day_type", "station_rank_group", "residual_direction", "n_eff"],
                "max_cases": 2,
                "use_for_prediction": False,
            },
        }

        selected, metrics = apply_residual_memory_skill(cases, skill)

        self.assertEqual(selected[0]["event_type"], "plaza programming")
        self.assertEqual(len(selected), 2)
        self.assertGreater(metrics["selected_relevance_mean"], metrics["baseline_relevance_mean"])
        self.assertFalse(metrics["forecast_array_changed"])


if __name__ == "__main__":
    unittest.main()
