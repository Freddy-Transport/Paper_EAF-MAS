import json
import tempfile
import unittest
from pathlib import Path


class KnowledgeBaseSkillFrameworkTests(unittest.TestCase):
    def test_manifest_redacts_api_key_and_records_split_policy(self):
        from agents.paper_knowledge_base import write_kb_manifest

        with tempfile.TemporaryDirectory() as td:
            path = write_kb_manifest(
                Path(td),
                kb_name="external_evidence_kb",
                source_files=["data/events.json"],
                split_policy="train_val_only",
                date_range={"train_end": "2023-01-27", "val_end": "2023-04-30"},
                doc_count=3,
                uses_qwenplus=True,
                extra={"api_key": "sk-secret", "OPENAI_API_KEY": "sk-secret"},
            )

            payload = json.loads(path.read_text(encoding="utf-8"))
            text = path.read_text(encoding="utf-8")
            self.assertEqual(payload["kb_name"], "external_evidence_kb")
            self.assertEqual(payload["split_policy"], "train_val_only")
            self.assertTrue(payload["uses_qwenplus"])
            self.assertNotIn("sk-secret", text)
            self.assertNotIn("OPENAI_API_KEY", text)

    def test_true_residual_cases_are_built_from_actual_moment_predictions(self):
        from agents.paper_knowledge_base import build_true_residual_cases

        prediction_rows = [
            {
                "anchor": "2023-01-01 00:00:00",
                "timestamp": "2023-01-01 01:00:00",
                "channel_name": "A013__49_St",
                "event_type": "Plaza Event",
                "impact_tier": "A",
                "rank_group": "top128",
                "day_type": "weekday",
                "moment_pred": 100.0,
                "actual": 110.0,
                "split": "train",
            },
            {
                "anchor": "2023-05-01 00:00:00",
                "timestamp": "2023-05-01 01:00:00",
                "channel_name": "A013__49_St",
                "event_type": "Plaza Event",
                "impact_tier": "A",
                "rank_group": "top128",
                "day_type": "weekday",
                "moment_pred": 100.0,
                "actual": 200.0,
                "split": "test",
            },
        ]

        docs, metas = build_true_residual_cases(prediction_rows, allowed_splits={"train", "val"})

        self.assertEqual(len(docs), 1)
        self.assertAlmostEqual(metas[0]["median_correction"], 0.10, places=4)
        self.assertEqual(metas[0]["source"], "true_lp_moment_residual")
        self.assertNotIn("200", docs[0])

    def test_skill_evolution_test_split_freezes_updates(self):
        from experiments.run_skill_evolution_study import (
            REQUIRED_SKILL_FIELDS,
            run_skill_evolution_study,
            should_update_skills,
        )

        self.assertTrue(should_update_skills("train", "evolving_skill"))
        self.assertTrue(should_update_skills("val", "evolving_skill"))
        self.assertFalse(should_update_skills("test", "evolving_skill"))
        self.assertFalse(should_update_skills("val", "frozen_skill"))
        self.assertFalse(should_update_skills("val", "no_skill"))

        with tempfile.TemporaryDirectory() as td:
            summary = run_skill_evolution_study(Path(td), split="test", skill_mode="evolving_skill")
            active = Path(td) / "models" / "skill_memory" / "active_skill_library.jsonl"
            records = [json.loads(line) for line in active.read_text(encoding="utf-8").splitlines() if line.strip()]

            self.assertFalse(summary["update_skills"])
            self.assertGreaterEqual(len(records), 1)
            for record in records:
                self.assertTrue(REQUIRED_SKILL_FIELDS.issubset(record))
            self.assertTrue((Path(td) / "predictions" / "skill_window_results.csv").is_file())

    def test_skill_evolution_val_generates_nonempty_promotion_report(self):
        from experiments.run_skill_evolution_study import REQUIRED_SKILL_FIELDS, run_skill_evolution_study

        with tempfile.TemporaryDirectory() as td:
            summary = run_skill_evolution_study(Path(td), split="val", skill_mode="evolving_skill")
            active = Path(td) / "models" / "skill_memory" / "active_skill_library.jsonl"
            candidates = Path(td) / "models" / "skill_memory" / "candidate_skill_records.jsonl"
            records = [json.loads(line) for line in candidates.read_text(encoding="utf-8").splitlines() if line.strip()]

            self.assertTrue(summary["update_skills"])
            self.assertGreaterEqual(len(records), 1)
            self.assertTrue(active.read_text(encoding="utf-8").strip())
            for record in records:
                self.assertTrue(REQUIRED_SKILL_FIELDS.issubset(record))
            self.assertIn("promoted_skills", summary)

    def test_skill_evolution_generates_typed_skill_categories(self):
        from experiments.run_skill_evolution_study import REQUIRED_SKILL_FIELDS, run_skill_evolution_study

        with tempfile.TemporaryDirectory() as td:
            summary = run_skill_evolution_study(Path(td), split="val", skill_mode="evolving_skill")
            candidates = Path(td) / "models" / "skill_memory" / "candidate_skill_records.jsonl"
            records = [json.loads(line) for line in candidates.read_text(encoding="utf-8").splitlines() if line.strip()]
            categories = {record.get("skill_category") for record in records}

            self.assertEqual(
                {"routing_skill", "evidence_skill", "abstention_skill", "calibration_skill", "residual_memory_skill"},
                categories,
            )
            self.assertEqual(summary["candidate_skills"], 5)
            for record in records:
                self.assertTrue(REQUIRED_SKILL_FIELDS.issubset(record))
                self.assertIn("skill_category", record)
            self.assertIn("residual_memory_skill", summary["promoted_by_category"])

    def test_skill_evolution_writes_autoskill_experience_and_promotion_trace(self):
        from agents.autoskill_memory import REQUIRED_AUTOSKILL_SKILL_FIELDS
        from experiments.run_skill_evolution_study import run_skill_evolution_study

        with tempfile.TemporaryDirectory() as td:
            summary = run_skill_evolution_study(Path(td), split="val", skill_mode="evolving_skill")
            root = Path(td)
            pool_path = root / "models" / "skill_memory" / "experience_pool" / "experiences_val.jsonl"
            trace_path = root / "reports" / "autoskill_evolution_trace.json"
            candidate_path = root / "models" / "skill_memory" / "autoskill_candidate_skills.jsonl"
            promoted_path = root / "models" / "skill_memory" / "autoskill_promoted_skills.jsonl"

            self.assertTrue(pool_path.is_file())
            self.assertTrue(trace_path.is_file())
            self.assertTrue(candidate_path.is_file())
            self.assertTrue(promoted_path.is_file())

            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            candidates = [json.loads(line) for line in candidate_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            promoted = [json.loads(line) for line in promoted_path.read_text(encoding="utf-8").splitlines() if line.strip()]

            self.assertEqual(trace["candidate_count"], 4)
            self.assertGreaterEqual(trace["mutation_count"], 1)
            self.assertGreaterEqual(trace["promoted_count"], 1)
            self.assertIn("autoskill", summary)
            for skill in candidates:
                self.assertTrue(REQUIRED_AUTOSKILL_SKILL_FIELDS.issubset(skill))
                self.assertFalse(skill["memory_selection_policy"].get("use_for_prediction"))
                self.assertFalse(skill["validation_metrics"]["forecast_array_changed"])
            for skill in promoted:
                self.assertTrue(skill["promotion_status"]["promoted"])

    def test_skill_evolution_test_split_loads_autoskill_champion_read_only(self):
        from experiments.run_skill_evolution_study import run_skill_evolution_study

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            val_summary = run_skill_evolution_study(root / "val", split="val", skill_mode="evolving_skill")
            champion = root / "val" / "models" / "skill_memory" / "autoskill_promoted_skills.jsonl"
            test_summary = run_skill_evolution_study(
                root / "test",
                split="test",
                skill_mode="evolving_skill",
                existing_skill_library=str(champion),
            )
            active = root / "test" / "models" / "skill_memory" / "autoskill_active_skill_library.jsonl"
            active_rows = [json.loads(line) for line in active.read_text(encoding="utf-8").splitlines() if line.strip()]

            self.assertGreater(val_summary["autoskill"]["promoted_count"], 0)
            self.assertFalse(test_summary["update_skills"])
            self.assertTrue(test_summary["autoskill"]["test_read_only"])
            self.assertEqual(test_summary["autoskill"]["promoted_count"], 0)
            self.assertEqual(test_summary["autoskill"]["loaded_existing_count"], len(active_rows))
            self.assertGreater(len(active_rows), 0)
            self.assertTrue(all(row["promotion_status"]["promoted"] for row in active_rows))

    def test_residual_memory_skill_promotes_on_explanation_quality_not_wape(self):
        from experiments.run_skill_evolution_study import build_typed_skill_candidates

        rows = [
            {
                "window_id": "w0",
                "station": "N060__Times_Sq_42_St",
                "event_type": "Plaza Event",
                "impact_tier": "A",
                "has_major_event": True,
                "wape_delta_raw_minus_adjusted": -0.4,
                "accepted_evidence_count": 0,
                "citation_coverage": 0.0,
                "calibration_decision_consistent": True,
                "evidence_coverage": 0.9,
                "residual_case_relevance": 0.8,
                "multi_hop_completeness": 1.0,
                "groundedness": 0.85,
                "leakage_free_rate": 1.0,
                "residual_memory_cases": 2,
            }
        ]

        records = build_typed_skill_candidates(rows, split="val", skill_mode="evolving_skill")
        residual_skill = [record for record in records if record["skill_category"] == "residual_memory_skill"][0]
        calibration_skill = [record for record in records if record["skill_category"] == "calibration_skill"][0]

        self.assertTrue(residual_skill["promoted"])
        self.assertFalse(calibration_skill["promoted"])
        self.assertEqual(residual_skill["calibration_action"]["policy"], "organize_residual_memory_for_explanation")
        self.assertIn("explanation_quality_delta", residual_skill["evidence_pattern"])

    def test_calibration_skill_promotes_only_on_positive_validation_delta(self):
        from experiments.run_skill_evolution_study import build_typed_skill_candidates

        harmful_rows = [
            {
                "window_id": "w0",
                "station": "A013__49_St",
                "event_type": "Plaza Event",
                "impact_tier": "A",
                "has_major_event": True,
                "wape_delta_raw_minus_adjusted": -0.5,
                "accepted_evidence_count": 1,
                "citation_coverage": 1.0,
                "calibration_decision_consistent": True,
            }
        ]

        records = build_typed_skill_candidates(harmful_rows, split="val", skill_mode="evolving_skill")
        calibration = [record for record in records if record["skill_category"] == "calibration_skill"][0]
        non_calibration = [record for record in records if record["skill_category"] != "calibration_skill"]

        self.assertFalse(calibration["promoted"])
        self.assertTrue(any(record["promoted"] for record in non_calibration))


if __name__ == "__main__":
    unittest.main()
