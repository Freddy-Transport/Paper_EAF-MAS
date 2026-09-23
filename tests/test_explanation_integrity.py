"""Synthetic rule-proxy contracts; no data, models, or external calls."""

import copy
import json
import unittest

from agents.explanation_evaluator import METRICS, evaluate_explanation_quality


def output():
    return {
        "claims": [
            {"text": "The supplied event is associated with this station.",
             "evidence_ids": ["event-a"], "case_ids": []},
            {"text": "The historical record contains a residual observation.",
             "evidence_ids": [], "case_ids": ["history-a", "residual-a"]},
        ],
        "decision": "apply",
        "links": [
            {"from_id": "event-a", "to_id": "station-a", "relation": "event_station"},
            {"from_id": "station-a", "to_id": "time-a", "relation": "station_time"},
            {"from_id": "history-a", "to_id": "residual-a", "relation": "historical_residual"},
            {"from_id": "residual-a", "to_id": "decision", "relation": "residual_decision"},
        ],
        "uncertainty": "The references do not establish causality.",
    }


def inputs():
    return {
        "accepted_evidence": [],
        "structured_events": [{"evidence_id": "event-a", "station_id": "station-a", "time_id": "time-a", "event_type": "synthetic"}],
        "historical_event_cases": [{"case_id": "history-a", "split": "train", "channel_name": "station-a", "event_type": "synthetic"}],
        "residual_cases": [{"id": "residual-a", "split": "val", "channel_name": "station-a", "event_type": "synthetic"}],
        "decision": {"abstain": False},
    }


class ExplanationIntegrityTests(unittest.TestCase):
    def score(self, generated=None, **kwargs):
        arguments = inputs()
        arguments.update(kwargs)
        return evaluate_explanation_quality("", structured_output=output() if generated is None else generated, **arguments)

    def test_valid_output_and_metric_denominators(self):
        result = self.score()
        for metric in ("evidence_coverage", "residual_case_relevance", "groundedness",
                       "multi_hop_completeness", "decision_consistency"):
            self.assertEqual(result[metric], 1.0, metric)
        self.assertEqual(result["unsupported_claim_rate"], 0.0)
        self.assertEqual(result["redundancy_rate"], 0.0)
        self.assertIsNone(result["leakage_free_rate"])
        self.assertEqual(result["effective_counts"]["evidence_coverage"], 3)
        self.assertEqual(result["effective_counts"]["groundedness"], 2)
        self.assertEqual(result["effective_counts"]["multi_hop_completeness"], 4)
        self.assertEqual(result["effective_counts"]["leakage_free_rate"], 0)
        self.assertTrue(result["calibration_decision_consistent"])
        for metric in METRICS:
            self.assertEqual(result[metric + "_status"], result["metric_status"][metric])
            self.assertEqual(result[metric + "_denominator"], result["effective_counts"][metric])
        json.dumps(result, allow_nan=False)

    def test_empty_output_never_gets_credit_even_with_verified_input(self):
        empty = {"claims": [], "decision": "", "links": output()["links"], "uncertainty": ""}
        blank_claim = copy.deepcopy(empty)
        blank_claim["claims"] = [{"text": "  ", "evidence_ids": ["event-a"], "case_ids": []}]
        for generated in (None, empty, blank_claim):
            with self.subTest(generated=generated):
                result = evaluate_explanation_quality("", structured_output=generated,
                                                     forecast_time_input_verified=True, **inputs())
                for metric in METRICS:
                    self.assertIn(result[metric], (0.0, None), metric)
                self.assertEqual(result["output_status"], "empty")

    def test_input_keywords_and_selected_skills_do_not_earn_credit(self):
        generated = {"claims": [{"text": "Event station time historical residual correction.",
                                  "evidence_ids": [], "case_ids": []}],
                     "decision": "", "links": [], "uncertainty": "Unknown."}
        baseline = self.score(generated)
        variant = self.score(generated, selected_residual_memory_skills=[{"id": "skill-a"}],
                             structured_events=[{"id": "event-a", "title": "event station residual apply"}])
        for metric in METRICS:
            self.assertEqual(baseline[metric], variant[metric], metric)
        self.assertEqual(baseline["evidence_coverage"], 0.0)
        self.assertEqual(baseline["groundedness"], 0.0)
        self.assertEqual(baseline["unsupported_claim_rate"], 1.0)
        self.assertEqual(baseline["multi_hop_completeness"], 0.0)

    def test_actual_references_determine_coverage_and_grounding(self):
        generated = output()
        generated["claims"][1]["case_ids"] = ["unknown-case"]
        result = self.score(generated)
        self.assertAlmostEqual(result["evidence_coverage"], 1 / 3)
        self.assertEqual(result["groundedness"], 0.5)
        self.assertEqual(result["unsupported_claim_rate"], 0.5)
        self.assertEqual(result["residual_case_relevance"], 0.0)
        self.assertEqual(result["unresolved_ids"], ["unknown-case"])

    def test_one_valid_reference_does_not_hide_an_unresolved_reference(self):
        generated = output()
        generated["claims"][0]["evidence_ids"].append("missing-a")
        self.assertEqual(self.score(generated)["groundedness"], 0.5)

    def test_ids_are_exact_and_case_sensitive(self):
        generated = output()
        generated["claims"][0]["evidence_ids"] = ["EVENT-A"]
        self.assertEqual(self.score(generated)["groundedness"], 0.5)

    def test_markdown_citations_cannot_repair_structured_claims(self):
        generated = output()
        generated["claims"][0]["evidence_ids"] = []
        result = evaluate_explanation_quality("[event-a]", structured_output=generated, **inputs())
        self.assertAlmostEqual(result["evidence_coverage"], 2 / 3)
        self.assertEqual(result["groundedness"], 0.5)

    def test_case_references_require_case_provenance_not_event_id(self):
        generated = output()
        generated["claims"][1]["case_ids"] = ["event-a", "residual-a"]
        result = self.score(generated)
        self.assertEqual(result["residual_case_relevance"], 0.5)
        self.assertEqual(result["groundedness"], 0.5)

    def test_relevance_rejects_wrong_channel_event_and_test_cases(self):
        generated = output()
        generated["claims"][1]["case_ids"] = ["residual-a"]
        for change in ({"channel_name": "station-b"}, {"event_type": "different-event"}, {"split": "test"}):
            case = inputs()["residual_cases"][0]
            case.update(change)
            with self.subTest(change=change):
                result = self.score(generated, residual_cases=[case])
                self.assertEqual(result["residual_case_relevance"], 0.0)
                self.assertEqual(result["residual_case_relevance_denominator"], 1)
                self.assertEqual(result["case_reference_precision"], 1.0)
                self.assertEqual(result["groundedness"], 1.0)

    def test_missing_case_or_event_metadata_is_na_not_relevance(self):
        generated = output()
        generated["claims"][1]["case_ids"] = ["residual-a"]
        variants = []
        for key in ("channel_name", "event_type", "split"):
            case = inputs()["residual_cases"][0]
            del case[key]
            variants.append({"residual_cases": [case]})
        for key in ("station_id", "event_type"):
            event = inputs()["structured_events"][0]
            del event[key]
            variants.append({"structured_events": [event]})
        variants.extend([{"structured_events": []}, {"structured_events": None}])
        for override in variants:
            with self.subTest(override=override):
                result = self.score(generated, **override)
                self.assertIsNone(result["residual_case_relevance"])
                self.assertEqual(result["residual_case_relevance_status"], "unverifiable_case_context")
                self.assertEqual(result["residual_case_relevance_denominator"], 0)
                self.assertEqual(result["residual_case_relevance_unverifiable_count"], 1)
                self.assertEqual(result["case_reference_precision"], 1.0)

    def test_case_requires_channel_and_type_on_the_same_event(self):
        events = [{"id": "event-a", "channel_name": "station-a", "event_type": "other-type"},
                  {"id": "event-b", "channel_name": "station-b", "event_type": "synthetic"}]
        result = self.score(structured_events=events)
        self.assertEqual(result["residual_case_relevance"], 0.0)

    def test_context_matching_normalizes_event_type_and_supports_station_id(self):
        cases = [{"case_id": "residual-a", "split": "val", "station_id": "station-a", "event_type": " SYNTHETIC "}]
        self.assertEqual(self.score(residual_cases=cases)["residual_case_relevance"], 1.0)

    def test_partial_unknown_case_context_is_not_silently_excluded(self):
        case = inputs()["residual_cases"][0]
        del case["event_type"]
        result = self.score(residual_cases=[case])
        self.assertIsNone(result["residual_case_relevance"])
        self.assertEqual(result["residual_case_relevance_unverifiable_count"], 1)
        self.assertEqual(result["residual_case_relevance_checked_count"], 1)

    def test_conflicting_context_under_same_case_id_is_unverifiable(self):
        original = inputs()["residual_cases"][0]
        conflict = {**original, "channel_name": "station-b"}
        result = self.score(residual_cases=[original, conflict])
        self.assertIsNone(result["residual_case_relevance"])

    def test_missing_registry_and_legacy_cases_are_unverifiable(self):
        result = evaluate_explanation_quality("", structured_output=output())
        for metric in ("evidence_coverage", "groundedness", "unsupported_claim_rate", "multi_hop_completeness"):
            self.assertIsNone(result[metric], metric)
        result = self.score(residual_cases=["unidentified numerical memory"])
        self.assertIsNone(result["groundedness"])
        self.assertEqual(result["evidence_coverage_status"], "missing_evidence_ids")
        self.assertEqual(result["evidence_coverage_denominator"], 0)

    def test_explicit_empty_registry_does_not_support_claims(self):
        result = self.score(accepted_evidence=[], structured_events=[], residual_cases=[], historical_event_cases=[])
        self.assertIsNone(result["evidence_coverage"])
        self.assertEqual(result["groundedness"], 0.0)
        self.assertEqual(result["unsupported_claim_rate"], 1.0)

    def test_aliases_and_duplicate_sources_count_once(self):
        generated = output()
        generated["claims"][0]["evidence_ids"] = ["event-alias", "event-a"]
        rows = [{"id": "event-a", "evidence_id": "event-alias"}, {"id": "event-alias"}]
        result = self.score(generated, structured_events=rows)
        self.assertEqual(result["evidence_coverage"], 1.0)
        self.assertEqual(result["evidence_coverage_denominator"], 3)
        self.assertEqual(result["groundedness"], 1.0)

    def test_missing_relations_endpoints_and_wrong_types_earn_zero(self):
        for links in ([], [{"from_id": "event-a", "to_id": "station-a"}],
                      [{"from_id": "missing-a", "to_id": "station-a", "relation": "event_station"}],
                      [{"from_id": "residual-a", "to_id": "event-a", "relation": "event_station"}],
                      [{"from_id": "event-a", "to_id": "event-a", "relation": "event_station"}],
                      [{"from_id": [], "to_id": {}, "relation": []}], [None]):
            generated = output()
            generated["links"] = links
            with self.subTest(links=links):
                self.assertEqual(self.score(generated)["multi_hop_completeness"], 0.0)

    def test_relation_denominator_is_fixed_and_duplicates_do_not_help(self):
        generated = output()
        generated["links"] = generated["links"][:3] * 2
        self.assertEqual(self.score(generated)["multi_hop_completeness"], 0.75)

    def test_channel_name_and_typed_time_endpoints(self):
        result = self.score(structured_events=[{"id": "event-a", "channel_name": "station-a"}],
                            accepted_evidence=[{"id": "time-a", "kind": "time"}])
        self.assertEqual(result["multi_hop_completeness"], 1.0)

    def test_decision_matching_both_directions(self):
        for actual in ("apply", "abstain"):
            for expected in (True, False):
                generated = output()
                generated["decision"] = actual
                result = self.score(generated, decision={"abstain": expected})
                self.assertEqual(result["decision_consistency"], float((actual == "abstain") == expected))

    def test_all_controller_states_and_explicit_state_precedence(self):
        states = {"apply": "apply", "partial_apply": "apply", "abstain": "preserve",
                  "explain_only": "preserve", "no_event_keep_raw": "preserve"}
        for expected, expected_action in states.items():
            for actual, actual_action in states.items():
                generated = output()
                generated["decision"] = actual
                arguments = inputs()
                arguments["decision"] = {"decision": expected, "abstain": expected == "abstain"}
                result = evaluate_explanation_quality(json.dumps(generated), structured_output=generated, **arguments)
                with self.subTest(expected=expected, actual=actual):
                    self.assertEqual(result["decision_consistency"], float(actual_action == expected_action))
                    self.assertEqual(result["exact_decision_consistency"], float(actual == expected))
                    self.assertIn("residual_decision", result["valid_relations"])

    def test_unknown_explicit_state_never_falls_back_to_false_abstain_flag(self):
        result = self.score(decision={"decision": "unknown_state", "abstain": False})
        self.assertIsNone(result["decision_consistency"])
        self.assertIsNone(result["exact_decision_consistency"])

    def test_controller_state_contradictions_are_detected(self):
        generated = output()
        generated["decision"] = "partial_apply and no_event_keep_raw"
        for expected in ("partial_apply", "no_event_keep_raw"):
            result = self.score(generated, decision={"decision": expected, "abstain": False})
            self.assertEqual(result["decision_consistency"], 0.0)
            self.assertEqual(result["decision_consistency_status"], "contradictory_output")

    def test_contradiction_fails_for_both_expected_decisions(self):
        for expected in (True, False):
            generated = output()
            generated["decision"] = "apply and abstain"
            result = self.score(generated, decision={"abstain": expected})
            self.assertEqual(result["decision_consistency"], 0.0)
            self.assertEqual(result["decision_consistency_status"], "contradictory_output")
            self.assertNotIn("residual_decision", result["valid_relations"])

    def test_claim_and_markdown_cannot_contradict_structured_decision(self):
        generated = output()
        generated["claims"][0]["text"] = "We abstain from correction."
        self.assertEqual(self.score(generated)["decision_consistency"], 0.0)
        result = evaluate_explanation_quality("We abstain.", structured_output=output(), **inputs())
        self.assertEqual(result["decision_consistency"], 0.0)

    def test_missing_expected_decision_is_not_assumed_apply(self):
        self.assertIsNone(self.score(decision=None)["decision_consistency"])
        self.assertIsNone(self.score(decision={})["decision_consistency"])
        generated = output()
        generated["decision"] = "not apply"
        self.assertEqual(self.score(generated, decision={"abstain": True})["decision_consistency"], 1.0)

    def test_missing_output_decision_is_unavailable(self):
        generated = output()
        generated["decision"] = ""
        result = self.score(generated)
        self.assertIsNone(result["decision_consistency"])
        self.assertEqual(result["decision_consistency_status"], "missing_output_decision")
        self.assertEqual(result["decision_consistency_denominator"], 0)
        self.assertNotIn("residual_decision", result["valid_relations"])

    def test_leakage_requires_external_boolean_attestation(self):
        for value, expected in ((None, None), (True, 1.0), (False, 0.0), ("true", None), (1, None)):
            result = self.score(forecast_time_input_verified=value)
            self.assertEqual(result["leakage_free_rate"], expected)
        generated = output()
        generated["forecast_time_input_verified"] = True
        generated["claims"][0]["text"] = "No future information was used."
        self.assertIsNone(self.score(generated)["leakage_free_rate"])

    def test_failed_generation_never_scores_stale_output(self):
        for status in ("failed", "timeout", "skipped", "invalid_json"):
            result = self.score(generation_status=status, forecast_time_input_verified=True)
            for metric in METRICS:
                self.assertIsNone(result[metric], metric)
                self.assertEqual(result[metric + "_status"], "generation_failed")
                self.assertEqual(result["effective_counts"][metric], 0)

    def test_success_status_does_not_add_bonus(self):
        baseline = self.score()
        for status in ("success", "ok", "completed", "generated"):
            result = self.score(generation_status=status)
            for metric in METRICS:
                self.assertEqual(baseline[metric], result[metric])

    def test_malformed_structured_output_is_na_not_repaired(self):
        bad_claims = output()
        bad_claims["claims"][0]["evidence_ids"] = "event-a"
        for generated in ({}, [], "not-json", bad_claims):
            result = self.score(generated)
            for metric in METRICS:
                self.assertIsNone(result[metric], metric)
                self.assertEqual(result[metric + "_status"], "invalid_structured_output")

    def test_redundancy_comes_from_output_not_duplicate_input_cases(self):
        generated = output()
        generated["claims"].append(copy.deepcopy(generated["claims"][0]))
        generated["claims"][-1]["text"] = "  THE supplied event is associated with this station.  "
        self.assertAlmostEqual(self.score(generated)["redundancy_rate"], 1 / 3)
        result = self.score(residual_cases=inputs()["residual_cases"] * 2)
        self.assertEqual(result["redundancy_rate"], 0.0)

    def test_old_positional_signature_and_explicit_markdown_ids(self):
        result = evaluate_explanation_quality("[event-a] [case:history-a] [evidence:residual-a]\nDecision: apply",
                                             [], {"abstain": False}, [{"id": "residual-a"}],
                                             [{"id": "event-a"}], [{"id": "history-a"}], [])
        self.assertEqual(result["evidence_coverage"], 1.0)
        self.assertEqual(result["decision_consistency"], 1.0)
        self.assertIsNone(result["groundedness"])
        self.assertIsNone(result["multi_hop_completeness"])
        self.assertIsNone(result["leakage_free_rate"])

    def test_inputs_and_generated_output_are_not_mutated(self):
        generated, arguments = output(), inputs()
        before = copy.deepcopy((generated, arguments))
        evaluate_explanation_quality("", structured_output=generated, **arguments)
        self.assertEqual((generated, arguments), before)


if __name__ == "__main__":
    unittest.main()
