"""Small fictional fixtures; never inputs for formal experiment reporting."""
import copy
import json
import unittest
import numpy as np
import torch

from agents.audit_inputs import assert_forecast_time, digest
from agents.formal_workflow import numerical_context, calibrate, validate_memory, run_numerical, FORMAL_FEATURES, FEATURE_SCHEMA
from agents.skillbench import evaluate_modes, aggregate, MODES, LocalJSONGenerator, select_skills, explanation_payload, GenerationSchemaError
from agents.event_adapter import EventResidualAdapter
from experiments.run_formal_evaluation import ablation_rows, summarize


def fixture():
    record = dict(request_id="fictional-request", forecast_origin="2030-01-01T00:00:00", split="test",
        model_identity="fictional-baseline", raw_forecast=[[10.,20.],[30.,40.]], outcomes=[[11.,21.],[30.,40.]],
        channel_names=["fictional-east", "fictional-west"], timestamps=["2030-01-01T00:00:00","2030-01-01T01:00:00"],
        channel_meta={}, station_relations=[], venue_channels=["fictional-east"],
        event_active_mask=[[True,True],[False,False]], groups={"event_type":"fictional-exhibit"},
        events=[dict(id="event-x",channel_name="fictional-east",event_time="2030-01-01T00:00:00",
                     event_type="fictional-exhibit",impact_tier="A",duration_hours=1)],
        evidence_sources=[dict(id="source-x",url="https://example.invalid/event",source_type="official",
                               published_at="2029-12-01T00:00:00")], model_assisted_summaries=[])
    memory = [dict(case_id="case-x",split="train",support_end="2029-01-01T00:00:00",channel_name="fictional-east",
        event_type="fictional-exhibit",median_correction=.02,iqr=.01,n_eff=7,prediction_source_sha256="a"*64)]
    config = dict(audit_thresholds=dict(source_threshold=.2,geo_threshold=.2,temporal_threshold=.2,residual_threshold=.2),
        gate_weights={k:.2 for k in ("source_validity_score","geo_consistency_score","temporal_alignment_score","semantic_consistency_score","residual_support_score")},
        secondary_threshold=.4,core_threshold=.7,rho_full=.04,rho_partial=.01, random_seed=4)
    return record,memory,config


class FormalWorkflowIntegrityTests(unittest.TestCase):
    def test_true_memory_changes_support_not_event_attributes(self):
        r,m,c=fixture()
        a=numerical_context(r,m,c)
        b=numerical_context(r,[],c)
        self.assertTrue(a["mask"].any())
        self.assertFalse(b["mask"].any())
        self.assertEqual(a["R_num"]["case_ids"][0][0],["case-x"])
        self.assertFalse(a["mask"][1].any())

    def test_future_or_test_residual_cases_cannot_enter(self):
        r,m,c=fixture()
        m[0]["support_end"]="2031-01-01"
        self.assertEqual(validate_memory(m,r["forecast_origin"]),[])
        m[0]["split"]="test"
        with self.assertRaises(ValueError): validate_memory(m,r["forecast_origin"])

    def test_six_modes_fixed_numerical_state_and_no_fixed_bonus(self):
        r,m,c=fixture(); context=numerical_context(r,m,c)
        state=calibrate(r,context,np.full((2,2),.02),c)
        before=digest(state)
        def generator(payload):
            output={"claims":[{"text":"A linked event exists.","evidence_ids":["event-x"],"case_ids":[]}],
                    "links":[],"decision":payload["decision"],"uncertainty":"Limited source support."}
            return json.dumps(output),output
        library=[dict(skill_id="fictional-skill",created_split="val",skill_category="residual_memory_skill",
                      promotion_status={"promoted":True},trigger_condition={"event_type":"fictional-exhibit"},
                      memory_selection_policy={"max_cases":1})]
        rows=evaluate_modes(r,state,m,library,generator)
        self.assertEqual(len(rows),len(MODES))
        self.assertTrue(all(x["generation_status"]=="success" for x in rows),rows)
        self.assertEqual(len({json.dumps(x["quality"],sort_keys=True) for x in rows}),1)
        self.assertEqual(digest(state),before)
        self.assertEqual(len({x["numerical_state_digest"] for x in rows}),1)

    def test_failed_generation_has_no_fallback_or_perfect_score(self):
        r,m,c=fixture(); context=numerical_context(r,m,c)
        state=calibrate(r,context,np.zeros((2,2)),c)
        def failed(payload): raise RuntimeError("fictional backend unavailable")
        rows=evaluate_modes(r,state,m,[],failed)
        for value in aggregate(rows).values():
            self.assertEqual(value["failures"],1)
            self.assertIsNone(value["metrics"]["groundedness"]["mean"])

    def test_retrospective_fields_and_remote_backend_rejected(self):
        with self.assertRaises(ValueError): assert_forecast_time({"evidence":[{"actual_wape":1}]})
        with self.assertRaises(ValueError): LocalJSONGenerator("https://example.invalid/v1","fake")

    def test_old_adapter_schema_rejected(self):
        r,m,c=fixture(); adapter=EventResidualAdapter(16)
        with self.assertRaisesRegex(ValueError,"schema"):
            run_numerical(r,m,c,adapter,{"feature_names":[]})

    def test_same_proposal_ablation_no_bound_and_non_event_preservation(self):
        r,m,c=fixture(); adapter=EventResidualAdapter(len(FORMAL_FEATURES))
        adapter.output_transform="identity"
        with torch.no_grad(): adapter.net[-1].bias.fill_(.2)
        context,state=run_numerical(r,m,c,adapter,{"feature_schema":FEATURE_SCHEMA,"feature_names":FORMAL_FEATURES,"output_transform":"identity"})
        rows=ablation_rows(r,context,state,adapter,c,0)
        indexed={x["variant"]:x for x in rows}
        self.assertTrue({"no_event_gate", "no_residual_support", "no_audit_guard"}.issubset(indexed))
        self.assertEqual(indexed["full_controller"]["non_event_degradation"],0)
        self.assertGreater(indexed["no_bound"]["max_relative_correction"],indexed["full_controller"]["max_relative_correction"])
        self.assertEqual(indexed["random_gate"]["corrected_cells"],indexed["full_controller"]["corrected_cells"])
        self.assertTrue(summarize(rows))

    def test_record_not_mutated_and_case_hash_provenance_required(self):
        r,m,c=fixture(); original=copy.deepcopy(r)
        numerical_context(r,m,c)
        self.assertEqual(r,original)
        del m[0]["prediction_source_sha256"]
        with self.assertRaises(ValueError): numerical_context(r,m,c)

    def test_test_created_skills_are_rejected(self):
        r,m,c=fixture()
        skill=dict(created_split="test",skill_category="abstention_skill",promotion_status={"promoted":True})
        with self.assertRaises(ValueError): select_skills([skill],"full_autoskill",r["events"])

    def test_prompt_fields_preserve_policy_but_reject_outcomes_and_future_sources(self):
        r,m,c=fixture(); state={"decision":"abstain"}
        skill=dict(skill_id="fictional",skill_category="abstention_skill",abstention_rule={"policy":"explain preservation"})
        payload=explanation_payload(r,state,m,[skill],"full_autoskill")
        self.assertIn("abstention_rule",payload["skill_instructions"][0])
        r["evidence_sources"][0]["outcomes"]=[1]
        with self.assertRaises(ValueError): explanation_payload(r,state,m,[],"no_skill")
        del r["evidence_sources"][0]["outcomes"]
        r["evidence_sources"][0]["published_at"]="2031-01-01T00:00:00"
        with self.assertRaises(ValueError): explanation_payload(r,state,m,[],"no_skill")

    def test_malformed_schema_is_recorded_as_failure(self):
        r,m,c=fixture(); state={"decision":"abstain"}
        def invalid(payload):
            output={"claims":[],"decision":17,"links":[],"uncertainty":None}
            return json.dumps(output),output
        row=evaluate_modes(r,state,m,[],invalid,modes=["no_skill"])[0]
        self.assertEqual(row["generation_status"],"schema_failed")
        self.assertTrue(row["raw_response"])

    def test_unparseable_response_is_preserved(self):
        r,m,c=fixture()
        def invalid(payload):
            raise GenerationSchemaError("invalid JSON", "fictional malformed response {")
        row=evaluate_modes(r,{"decision":"abstain"},m,[],invalid,modes=["no_skill"])[0]
        self.assertEqual(row["generation_status"],"schema_failed")
        self.assertEqual(row["raw_response"],"fictional malformed response {")
        self.assertIsNone(row["quality"]["groundedness"])


if __name__=="__main__": unittest.main()
