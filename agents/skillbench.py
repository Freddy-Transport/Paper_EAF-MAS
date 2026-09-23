"""Real-output, mode-blind SkillBench execution; no synthetic formal inputs."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from datetime import datetime
from urllib.parse import urlparse
import urllib.request

from agents.audit_inputs import assert_forecast_time, digest
from agents.autoskill_memory import apply_residual_memory_skill
from agents.explanation_evaluator import evaluate_explanation_quality

MODES = ("no_skill", "static_prompt", "residual_memory", "evidence_audit", "abstention", "full_autoskill")
METRICS = ("evidence_coverage", "residual_case_relevance", "multi_hop_completeness", "groundedness",
           "unsupported_claim_rate", "redundancy_rate", "decision_consistency", "leakage_free_rate")
SYSTEM = '''Produce a forecast-time explanation as JSON with these fields:
claims: a list of objects with text, evidence_ids, case_ids;
decision: the supplied fixed decision;
links: a list of {from_id, to_id, relation}, using supplied identifiers;
uncertainty: a string.
Use relation names event_station, station_time, historical_residual, residual_decision.
Use the reserved identifier decision as the decision-link endpoint.
Keep evidence identifiers and provenance. Claims must reference their supporting items.
Use only supplied forecast-time inputs. Historical residuals are train/validation cases.
Describe uncertainty explicitly. Return JSON only.'''

DISPLAY_FIELDS = {
    "event": {"id", "evidence_id", "title", "event_type", "event_category", "event_time", "duration_hours",
              "channel_name", "station_id", "time_id", "venue_name", "location", "impact_tier", "published_at", "available_at"},
    "source": {"id", "evidence_id", "url", "title", "source_type", "published_at", "source_time",
               "availability_time", "available_at", "snippet", "summary", "text", "accepted"},
    "case": {"case_id", "split", "support_end", "channel_name", "event_type", "median_correction", "iqr",
             "n_eff", "prediction_source_sha256", "day_type", "station_rank_group", "residual_direction"},
    "relation": {"id", "evidence_id", "channel_name", "station_id", "time_id", "event_id", "relation", "distance_m", "entity_type"},
}


def display_rows(rows, kind):
    result = []
    for row in rows:
        assert_forecast_time(row)
        selected = {k:deepcopy(v) for k,v in row.items() if k in DISPLAY_FIELDS[kind]}
        if any(isinstance(v,(dict,list)) for v in selected.values()):
            raise ValueError("Evidence display fields must be scalar; normalize nested fields explicitly")
        result.append(selected)
    return result


class GenerationSchemaError(ValueError):
    def __init__(self, message, raw_response):
        super().__init__(message)
        self.raw_response = raw_response


class LocalJSONGenerator:
    def __init__(self, base_url, model, *, timeout=120, max_tokens=2048):
        parsed = urlparse(base_url)
        if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Private evaluation permits a local backend only")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.timeout, self.max_tokens = model, timeout, max_tokens

    def __call__(self, payload):
        body = dict(model=self.model, messages=[{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(payload, allow_nan=False)}], temperature=0,
            max_tokens=self.max_tokens, response_format={"type": "json_object"})
        request = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.getenv("LOCAL_LLM_API_KEY", "EMPTY")})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            text = json.load(response)["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError) as exc:
            raise GenerationSchemaError("Explanation is not valid JSON", text) from exc
        if not isinstance(parsed, dict) or not {"claims", "decision", "links", "uncertainty"}.issubset(parsed):
            raise GenerationSchemaError("Explanation schema missing required fields", text)
        if not isinstance(parsed["claims"], list) or not isinstance(parsed["links"], list):
            raise GenerationSchemaError("Explanation schema has invalid fields", text)
        return text, parsed


def select_skills(library, mode, events):
    categories = {"residual_memory": {"residual_memory_skill"}, "evidence_audit": {"evidence_audit_skill"},
                  "abstention": {"abstention_skill"}, "full_autoskill": None}
    if mode not in categories:
        return []
    selected = []
    types = {str(e.get("event_type") or e.get("event_category") or "").lower() for e in events}
    for skill in library:
        if not skill.get("promotion_status", {}).get("promoted"):
            continue
        if skill.get("created_split") != "val" or any(skill.get(k) == "test" for k in ("split", "source_split")):
            raise ValueError("Explicit validation-created skill provenance is required")
        allowed = categories[mode]
        if allowed is not None and skill["skill_category"] not in allowed:
            continue
        trigger = skill.get("trigger_condition", {})
        if trigger.get("has_major_event") and not events:
            continue
        if trigger.get("event_type") and str(trigger["event_type"]).lower() not in types:
            continue
        selected.append(deepcopy(skill))
    return selected


def explanation_payload(record, state, cases, skills, mode):
    analogues = deepcopy(cases[:3])
    for skill in skills:
        if skill["skill_category"] == "residual_memory_skill":
            analogues, _ = apply_residual_memory_skill(cases, skill)
            break
    instructions = [{k: deepcopy(s[k]) for k in ("skill_id", "skill_category", "trigger_condition",
        "action", "reasoning_template", "memory_selection_policy", "abstention_rule") if k in s} for s in skills]
    payload = {"forecast_origin": record["forecast_origin"], "structured_events": display_rows(record["events"], "event"),
        "evidence": display_rows(record["evidence_sources"] + record["model_assisted_summaries"], "source"),
        "station_relations": display_rows(record["station_relations"], "relation"), "historical_cases": display_rows(analogues,"case"),
        "decision": state["decision"], "decision_id": "decision",
        "skill_instructions": instructions}
    if mode == "static_prompt":
        payload["organization"] = "Organize by evidence, historical cases, decision and uncertainty."
    assert_forecast_time(payload)
    origin = datetime.fromisoformat(record["forecast_origin"])
    for row in payload["evidence"] + payload["structured_events"]:
        for key in ("published_at", "source_time", "availability_time", "available_at"):
            if row.get(key) and datetime.fromisoformat(str(row[key])) > origin:
                raise ValueError("Supplied evidence timestamp is later than forecast origin")
    return payload


def evaluate_modes(record, state, cases, library, generator, *, modes=MODES):
    state_hash, library_hash = digest(state), digest(library)
    rows = []
    for mode in modes:
        if mode not in MODES:
            raise ValueError("Unknown skill mode")
        skills = select_skills(library, mode, record["events"])
        payload = explanation_payload(record, state, cases, skills, mode)
        text, output, error = "", None, None
        try:
            text, output = generator(deepcopy(payload))
            quality = evaluate_explanation_quality(text,
                accepted_evidence=payload["evidence"] + payload["station_relations"],
                decision={"decision": state["decision"], "abstain": state["decision"] == "abstain",
                          "id": payload["decision_id"]},
                residual_cases=payload["historical_cases"], structured_events=payload["structured_events"],
                historical_event_cases=payload["historical_cases"], structured_output=output,
                generation_status="success", forecast_time_input_verified=True)
            status = "schema_failed" if quality.get("structured_output_status") == "invalid" else "success"
            if status != "success": error = "Generated explanation has an invalid structured schema"
        except Exception as exc:
            quality, status, error = {k: None for k in METRICS}, "failed", type(exc).__name__ + ": " + str(exc)
            if isinstance(exc, GenerationSchemaError):
                text, status = exc.raw_response, "schema_failed"
        if digest(state) != state_hash or digest(library) != library_hash:
            raise RuntimeError("Explanation stage mutated numerical state or promoted library")
        rows.append({"request_id": record["request_id"], "mode": mode, "generation_status": status,
                     "error": error, "raw_response": text, "structured_output": output,
                     "post_constraint_output": output, "constraint_rewrite_applied": False,
                     "input_digest": digest(payload), "numerical_state_digest": state_hash,
                     "library_digest": library_hash, "quality": quality})
    return rows


def aggregate(rows):
    result = {}
    for mode in dict.fromkeys(row["mode"] for row in rows):
        group = [r for r in rows if r["mode"] == mode]
        metrics = {}
        for metric in METRICS:
            values = [r["quality"].get(metric) for r in group if r["generation_status"] == "success"]
            values = [v for v in values if v is not None]
            metrics[metric] = {"mean": sum(values) / len(values) if values else None, "n": len(values)}
        result[mode] = {"requests": len(group), "failures": sum(r["generation_status"] != "success" for r in group),
                        "metrics": metrics}
    return result
