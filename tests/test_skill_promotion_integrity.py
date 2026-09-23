"""Synthetic replay contracts, not evidence of real explanation quality gains."""

import hashlib
import json
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

from agents import autoskill_memory as memory


def _experiences():
    return [
        {
            "experience_id": "val_1",
            "split": "val",
            "structured_event": {"event_type": "concert", "impact_tier": "A"},
            "explanation_scores": {
                "leakage_free_rate": 1.0,
                "unsupported_claim_rate": 0.0,
                "residual_case_relevance": 1.0,
                "multi_hop_completeness": 1.0,
                "decision_consistency": 1.0,
            },
            "evidence_audit": {
                "source_validity_score": 1.0,
                "geo_consistency_score": 1.0,
                "temporal_alignment_score": 1.0,
                "semantic_consistency_score": 1.0,
            },
            "historical_residual_memory": [{"event_type": "concert"}],
            "synthetic_numerical": {"raw": [[1.0, 2.0]], "adjusted": [[1.0, 2.1]]},
        },
    ]


def _measure(skill, experiences):
    # Fixture digests are computed from arrays, never from an invariance flag.
    before = {
        exp["experience_id"]: hashlib.sha256(
            json.dumps(exp["synthetic_numerical"], sort_keys=True).encode("utf-8")
        ).hexdigest()
        for exp in experiences
    }
    return {
        "metrics": {
            "leakage_free_rate": 1.0,
            "unsupported_claim_rate": 0.10,
            "residual_case_relevance": 0.70,
            "multi_hop_completeness": 0.70,
            "decision_consistency": 1.0,
            "evidence_coverage": 0.70,
            "groundedness": 0.70,
            "source_validity_score": 0.70,
            "geo_consistency_score": 0.70,
            "temporal_alignment_score": 0.70,
            "semantic_consistency_score": 0.70,
            "historical_memory_count": 1,
        },
        "replay_ids": list(before),
        "valid_sample_count": len(before),
        "numerical_digests_before": before,
        "numerical_digests_after": dict(before),
    }


class SkillPromotionIntegrityTests(unittest.TestCase):
    def test_callback_runs_for_baseline_and_every_original_and_mutation(self):
        evaluator = Mock(side_effect=_measure)
        report = memory.evolve_skills(_experiences(), "val", replay_evaluator=evaluator)
        skills = report["candidate_skills"] + report["mutated_skills"]
        self.assertEqual(evaluator.call_count, 1 + len(skills))
        self.assertIsNone(evaluator.call_args_list[0].args[0])
        self.assertEqual(report["candidate_count"], 4)
        self.assertEqual(report["mutation_count"], 12)
        evaluated_ids = []
        for call in evaluator.call_args_list[1:]:
            skill, experiences = call.args
            evaluated_ids.append(skill["skill_id"])
            self.assertEqual(experiences, _experiences())
            self.assertEqual(skill["validation_metrics"], {})
            self.assertIsNone(skill["validation_replay"])
            self.assertIsNone(skill["forecast_arrays_identical"])
            self.assertFalse(skill["promotion_status"]["promoted"])
        self.assertCountEqual(evaluated_ids, [skill["skill_id"] for skill in skills])
        self.assertEqual(len(set(evaluated_ids)), len(skills))
        self.assertEqual(report["promoted_count"], len(skills))
        self.assertEqual(report["promoted_count"] + report["rejected_count"], len(skills))
        self.assertTrue(report["forecast_arrays_identical"])
        for skill in skills:
            replay = skill["validation_replay"]
            self.assertEqual(replay["replay_ids"], ["val_1"])
            self.assertEqual(replay["valid_sample_count"], 1)
            self.assertEqual(skill["validation_metrics"], replay["metrics"])
            self.assertTrue(skill["forecast_arrays_identical"])
            self.assertTrue(all(delta == 0.0 for delta in skill["validation_comparison"]["metric_deltas"].values()))

    def test_no_callback_cannot_promote_cached_perfect_scores(self):
        with patch.object(memory, "_cluster_experiences") as cluster:
            with self.assertRaisesRegex(ValueError, "replay_evaluator"):
                memory.evolve_skills(_experiences(), "val")
            cluster.assert_not_called()

    def test_callback_is_executed_again_on_each_evolution(self):
        evaluator = Mock(side_effect=_measure)
        memory.evolve_skills(_experiences(), "val", replay_evaluator=evaluator)
        first_calls = evaluator.call_count
        memory.evolve_skills(_experiences(), "val", replay_evaluator=evaluator)
        self.assertEqual(evaluator.call_count, 2 * first_calls)

    def test_test_train_mixed_and_unlabelled_inputs_rejected_before_generation(self):
        for split, row_splits in (
            ("test", ["test"]), ("test", ["val"]), ("train", ["train"]),
            ("val", ["test"]), ("val", ["train"]), ("val", ["val", "test"]),
            ("val", ["val", "train"]), ("val", [None]), ("val", []),
        ):
            with self.subTest(split=split, row_splits=row_splits):
                rows = []
                for index, row_split in enumerate(row_splits):
                    row = _experiences()[0]
                    row.update(experience_id=f"row_{index}", split=row_split)
                    rows.append(row)
                evaluator = Mock(side_effect=_measure)
                with patch.object(memory, "_cluster_experiences") as cluster:
                    with self.assertRaises(ValueError):
                        memory.evolve_skills(rows, split, replay_evaluator=evaluator)
                    evaluator.assert_not_called()
                    cluster.assert_not_called()

    def test_missing_and_duplicate_experience_ids_rejected(self):
        for rows in ([{**_experiences()[0], "experience_id": None}], _experiences() * 2):
            evaluator = Mock(side_effect=_measure)
            with self.assertRaisesRegex(ValueError, "experience_id"):
                memory.evolve_skills(rows, "val", replay_evaluator=evaluator)
            evaluator.assert_not_called()

    def test_digest_mismatch_rejects_promotions_despite_purported_invariance(self):
        def changed(skill, experiences):
            result = _measure(skill, experiences)
            if skill is not None:
                result["numerical_digests_after"]["val_1"] = "changed-digest"
                result["forecast_arrays_identical"] = True
                result["forecast_array_changed"] = False
            return result

        report = memory.evolve_skills(_experiences(), "val", replay_evaluator=changed)
        self.assertEqual(report["promoted_count"], 0)
        self.assertFalse(report["forecast_arrays_identical"])
        for skill in report["candidate_skills"] + report["mutated_skills"]:
            self.assertIn("changed forecast arrays", skill["promotion_status"]["reason"])
            self.assertFalse(skill["forecast_arrays_identical"])
            self.assertIsNone(skill["validation_comparison"])

    def test_originals_do_not_inherit_successful_mutation_status(self):
        def measured(skill, experiences):
            result = _measure(skill, experiences)
            if skill is not None and skill["version"] == 1:
                result["metrics"]["unsupported_claim_rate"] = 0.50
            return result

        report = memory.evolve_skills(_experiences(), "val", replay_evaluator=measured)
        self.assertTrue(all(not skill["promotion_status"]["promoted"] for skill in report["candidate_skills"]))
        self.assertTrue(all(skill["promotion_status"]["promoted"] for skill in report["mutated_skills"]))

    def test_mutations_only_change_policy_and_clear_inherited_validation(self):
        report = memory.evolve_skills(_experiences(), "val", replay_evaluator=_measure)
        original = report["promoted_skills"][0]
        snapshot = deepcopy(original)
        for operator in (
            "adjust_memory_case_budget", "tighten_memory_case_budget",
            "broaden_trigger_for_recall", "strict_abstention_guard",
        ):
            with self.subTest(operator=operator):
                mutated = memory._mutate_candidate(original, 1, operator)
                self.assertEqual(original, snapshot)
                self.assertEqual(mutated["validation_metrics"], {})
                self.assertIsNone(mutated["validation_replay"])
                self.assertIsNone(mutated["validation_comparison"])
                self.assertIsNone(mutated["forecast_arrays_identical"])
                self.assertFalse(mutated["promotion_status"]["promoted"])
                self.assertEqual(mutated["evidence_pattern"], original["evidence_pattern"])
                allowed = {
                    "skill_id", "version", "memory_selection_policy", "abstention_rule",
                    "validation_metrics", "validation_replay", "validation_comparison",
                    "forecast_arrays_identical", "promotion_status",
                }
                for key in original.keys() - allowed:
                    self.assertEqual(mutated[key], original[key])
                self.assertFalse(memory._promotion_reason(mutated, "val", "evolving_skill")[0])

    def test_candidate_records_start_unevaluated_despite_cached_experience_scores(self):
        stats = memory._aggregate_experience_stats(_experiences())
        skill = memory._skill_record("routing_skill", stats, "val", "evolving_skill")
        self.assertEqual(skill["validation_metrics"], {})
        self.assertIsNone(skill["validation_replay"])
        self.assertIsNone(skill["forecast_arrays_identical"])
        self.assertFalse(memory._promotion_reason(skill, "val", "evolving_skill")[0])

    def test_invalid_provenance_and_nonpositive_sample_counts_fail_closed(self):
        invalid = (
            ("replay_ids", []), ("replay_ids", ["test_1"]),
            ("replay_ids", ["val_1", "val_1"]), ("replay_ids", [None]),
            ("valid_sample_count", 0), ("valid_sample_count", -1),
            ("valid_sample_count", 2), ("valid_sample_count", True),
            ("valid_sample_count", 1.0), ("valid_sample_count", None),
            ("numerical_digests_before", {}), ("numerical_digests_after", {}),
            ("numerical_digests_before", {"val_1": ""}),
            ("numerical_digests_after", {"val_1": None}),
            ("numerical_digests_after", {"foreign": "digest"}),
        )
        for key, value in invalid:
            with self.subTest(key=key, value=value):
                def malformed(skill, experiences):
                    result = _measure(skill, experiences)
                    if skill is not None:
                        result[key] = value
                    return result

                with self.assertRaises(ValueError):
                    memory.evolve_skills(_experiences(), "val", replay_evaluator=malformed)

    def test_missing_na_boolean_and_nonfinite_metrics_fail_closed(self):
        missing = object()
        for metric in ("leakage_free_rate", "unsupported_claim_rate", "source_validity_score",
                       "residual_case_relevance", "multi_hop_completeness", "decision_consistency",
                       "historical_memory_count"):
            for value in (missing, None, "NA", float("nan"), float("inf"), True):
                with self.subTest(metric=metric, value=value):
                    def malformed(skill, experiences):
                        result = _measure(skill, experiences)
                        if skill is not None:
                            if value is missing:
                                result["metrics"].pop(metric)
                            else:
                                result["metrics"][metric] = value
                        return result

                    with self.assertRaises(ValueError):
                        memory.evolve_skills(_experiences(), "val", replay_evaluator=malformed)

    def test_invalid_metric_ranges_fail_closed(self):
        for metric, value in (("leakage_free_rate", 1.1), ("unsupported_claim_rate", -0.1),
                              ("historical_memory_count", -1), ("historical_memory_count", 0.5)):
            with self.subTest(metric=metric, value=value):
                def malformed(skill, experiences):
                    result = _measure(skill, experiences)
                    result["metrics"][metric] = value
                    return result

                with self.assertRaises(ValueError):
                    memory.evolve_skills(_experiences(), "val", replay_evaluator=malformed)

    def test_none_is_allowed_only_for_no_skill_baseline(self):
        evaluator = Mock(return_value=None)
        with self.assertRaisesRegex(ValueError, "measurement dict"):
            memory.evolve_skills(_experiences(), "val", replay_evaluator=evaluator)
        self.assertEqual(evaluator.call_count, 2)

    def test_absent_baseline_does_not_invent_improvements(self):
        def without_baseline(skill, experiences):
            return None if skill is None else _measure(skill, experiences)

        report = memory.evolve_skills(_experiences(), "val", replay_evaluator=without_baseline)
        self.assertIsNone(report["no_skill_baseline"])
        self.assertGreater(report["promoted_count"], 0)
        self.assertTrue(all(skill["validation_comparison"] is None for skill in report["promoted_skills"]))

    def test_comparison_uses_measured_positive_and_negative_deltas_no_fixed_winner(self):
        def measured(skill, experiences):
            result = _measure(skill, experiences)
            result["metrics"]["evidence_coverage"] = 0.50
            if skill is not None:
                result["metrics"]["evidence_coverage"] = 0.60 if skill["version"] == 1 else 0.40
            return result

        report = memory.evolve_skills(_experiences(), "val", replay_evaluator=measured)
        for skill in report["candidate_skills"]:
            self.assertAlmostEqual(skill["validation_comparison"]["metric_deltas"]["evidence_coverage"], 0.10)
        for skill in report["mutated_skills"]:
            self.assertAlmostEqual(skill["validation_comparison"]["metric_deltas"]["evidence_coverage"], -0.10)

    def test_comparison_requires_same_replays_and_starting_numerical_state(self):
        for mismatch in ("sample_ids", "starting_state", "baseline_changed"):
            with self.subTest(mismatch=mismatch):
                experiences = _experiences()
                experiences.append({**deepcopy(experiences[0]), "experience_id": "val_2"})

                def measured(skill, rows):
                    result = _measure(skill, rows)
                    if skill is None:
                        if mismatch == "sample_ids":
                            result["replay_ids"] = ["val_1"]
                            result["valid_sample_count"] = 1
                            result["numerical_digests_before"].pop("val_2")
                            result["numerical_digests_after"].pop("val_2")
                        elif mismatch == "starting_state":
                            result["numerical_digests_before"]["val_1"] = "different"
                            result["numerical_digests_after"]["val_1"] = "different"
                        else:
                            result["numerical_digests_after"]["val_1"] = "changed"
                    return result

                report = memory.evolve_skills(experiences, "val", replay_evaluator=measured)
                self.assertTrue(all(skill["validation_comparison"] is None for skill in report["promoted_skills"]))
                if mismatch == "baseline_changed":
                    self.assertFalse(report["forecast_arrays_identical"])

    def test_hard_promotion_thresholds_use_only_measured_values(self):
        scenarios = (
            ({"leakage_free_rate": 0.99}, set()),
            ({"unsupported_claim_rate": 0.251}, set()),
            ({"unsupported_claim_rate": 0.25}, set(memory.DOMAIN_SKILL_CATEGORIES)),
            ({"residual_case_relevance": 0.49}, set(memory.DOMAIN_SKILL_CATEGORIES) - {"residual_memory_skill"}),
            ({"multi_hop_completeness": 0.49}, set(memory.DOMAIN_SKILL_CATEGORIES) - {"residual_memory_skill"}),
            ({"historical_memory_count": 0}, set(memory.DOMAIN_SKILL_CATEGORIES) - {"residual_memory_skill"}),
            ({"decision_consistency": 0.99}, set(memory.DOMAIN_SKILL_CATEGORIES) - {"abstention_skill"}),
            ({key: 0.59 for key in memory._CATEGORY_REPLAY_METRICS["evidence_audit_skill"]},
             set(memory.DOMAIN_SKILL_CATEGORIES) - {"evidence_audit_skill"}),
            ({**{key: 0.60 for key in memory._CATEGORY_REPLAY_METRICS["evidence_audit_skill"]},
              "residual_case_relevance": 0.50, "multi_hop_completeness": 0.50}, set(memory.DOMAIN_SKILL_CATEGORIES)),
        )
        for overrides, expected in scenarios:
            with self.subTest(overrides=overrides):
                def measured(skill, experiences):
                    result = _measure(skill, experiences)
                    result["metrics"].update(overrides)
                    return result

                report = memory.evolve_skills(_experiences(), "val", replay_evaluator=measured)
                self.assertEqual({skill["skill_category"] for skill in report["promoted_skills"]}, expected)

    def test_routing_still_requires_major_event_and_nonevolving_mode_cannot_promote(self):
        rows = _experiences()
        rows[0]["structured_event"] = {}
        report = memory.evolve_skills(rows, "val", replay_evaluator=_measure)
        self.assertNotIn("routing_skill", {skill["skill_category"] for skill in report["promoted_skills"]})
        report = memory.evolve_skills(_experiences(), "val", "static_skill", replay_evaluator=_measure)
        self.assertEqual(report["mutation_count"], 0)
        self.assertEqual(report["promoted_count"], 0)

    def test_policy_or_metric_changes_invalidate_existing_replay(self):
        report = memory.evolve_skills(_experiences(), "val", replay_evaluator=_measure)
        skill = deepcopy(report["promoted_skills"][0])
        skill["memory_selection_policy"]["max_cases"] += 1
        self.assertFalse(memory._promotion_reason(skill, "val", "evolving_skill")[0])
        skill = deepcopy(report["promoted_skills"][0])
        skill["validation_metrics"]["unsupported_claim_rate"] = 0.0
        self.assertFalse(memory._promotion_reason(skill, "val", "evolving_skill")[0])

    def test_callback_inputs_are_isolated_and_exceptions_are_not_swallowed(self):
        rows = _experiences()
        original = deepcopy(rows)

        def measured(skill, experiences):
            result = _measure(skill, experiences)
            experiences[0]["split"] = "test"
            if skill is not None:
                skill["memory_selection_policy"]["use_for_prediction"] = True
                skill["validation_metrics"]["leakage_free_rate"] = 0.0
            return result

        report = memory.evolve_skills(rows, "val", replay_evaluator=measured)
        self.assertEqual(rows, original)
        self.assertTrue(all(not skill["memory_selection_policy"]["use_for_prediction"] for skill in report["promoted_skills"]))
        evaluator = Mock(side_effect=RuntimeError("runner failed"))
        with self.assertRaisesRegex(RuntimeError, "runner failed"):
            memory.evolve_skills(rows, "val", replay_evaluator=evaluator)

    def test_missing_input_quality_is_unknown_not_perfect(self):
        experience = memory.build_replay_experience({}, "val", "missing")
        self.assertIsNone(experience["explanation_scores"]["leakage_free_rate"])
        self.assertIsNone(experience["explanation_scores"]["decision_consistency"])
        self.assertIsNone(experience["explanation_scores"]["unsupported_claim_rate"])
        stats = memory._aggregate_experience_stats([experience])
        self.assertIsNone(stats["quality_score"])
        self.assertIsNone(stats["quality_mean"]["leakage_free_rate"])
        cluster = memory._cluster_key(experience)
        self.assertEqual(cluster["source_quality_bucket"], "unknown")
        self.assertEqual(cluster["residual_support_bucket"], "unknown")
        self.assertEqual(cluster["unsupported_quality_bucket"], "unknown")

    def test_retrieval_diagnostics_are_not_independent_quality_or_invariance(self):
        cases = [{"event_type": "sports"}, {"event_type": "concert"}]
        selected, diagnostics = memory.apply_residual_memory_skill(cases, {
            "trigger_condition": {"event_type": "concert"},
            "memory_selection_policy": {"max_cases": 1},
        })
        self.assertEqual(selected, [cases[1]])
        self.assertNotIn("baseline_relevance_mean", diagnostics)
        self.assertNotIn("selected_relevance_mean", diagnostics)
        self.assertIn("not_independent_quality", diagnostics["score_kind"])
        self.assertGreater(diagnostics["selected_ranking_score_mean"], diagnostics["input_order_ranking_score_mean"])
        self.assertIsNone(diagnostics["forecast_array_changed"])
        _, empty = memory.apply_residual_memory_skill([], {})
        self.assertIsNone(empty["selected_ranking_score_mean"])


if __name__ == "__main__":
    unittest.main()
