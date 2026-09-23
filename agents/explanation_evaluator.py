"""Mode-blind rule proxies for actual explanation output.

Grounding means reference traceability, not factual entailment. Missing
verification is None (JSON null), never an inferred passing score. This module
does not call models, access the network, or inspect experimental mode names.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence


RELATIONS = {
    "event_station": ("event", "station"),
    "station_time": ("station", "time"),
    "historical_residual": ("historical", "residual"),
    "residual_decision": ("residual", "decision"),
}
METRICS = (
    "evidence_coverage", "residual_case_relevance", "multi_hop_completeness",
    "groundedness", "unsupported_claim_rate", "redundancy_rate",
    "decision_consistency", "leakage_free_rate",
)
DECISION_ACTIONS = {
    "apply": "apply", "partial_apply": "apply",
    "abstain": "preserve", "explain_only": "preserve", "no_event_keep_raw": "preserve",
}


def _norm(value):
    return re.sub(r"\s+", " ", value).strip().lower() if isinstance(value, str) else ""


def _ids(row):
    if not isinstance(row, Mapping):
        return set()
    return {row[key].strip() for key in ("id", "evidence_id", "case_id")
            if isinstance(row.get(key), str) and row[key].strip()}


def _references(value):
    return (isinstance(value, list)
            and all(isinstance(item, str) and item.strip() for item in value))


def _actions(text):
    """Explicit action vocabulary only; not a semantic consistency judge."""
    text = _norm(text)
    negative_apply = bool(re.search(r"\b(?:do not|does not|don't|not to|not) apply\b", text))
    text = re.sub(r"\b(?:do not|does not|don't|not to|not) apply\b", "", text)
    # Negating abstention is not by itself an explicit apply decision.
    text = re.sub(r"\b(?:do not|does not|don't|not to|not) abstain\b", "", text)
    actions = set()
    if negative_apply or re.search(r"\b(?:abstain|abstains|abstained|explain_only|no_event_keep_raw|no correction|keep raw)\b", text):
        actions.add("preserve")
    if re.search(r"\b(?:apply|partial_apply|applies|applied)\b", text):
        actions.add("apply")
    return actions


def _registry(groups):
    records, roles = [], {}
    missing_ids = False
    for rows, role in groups:
        for row in rows or []:
            aliases = _ids(row)
            if not aliases:
                missing_ids = True
                continue
            merged = set(aliases)
            for existing in list(records):
                if existing & merged:
                    merged.update(existing)
                    records.remove(existing)
            records.append(merged)
            for identifier in aliases:
                roles.setdefault(identifier, set()).add(role)
                for key in ("kind", "type", "entity_type"):
                    kind = row.get(key)
                    if isinstance(kind, str) and kind in {"event", "station", "time", "historical", "residual"}:
                        roles[identifier].add(kind)
            for key, endpoint_role in (("station_id", "station"), ("channel_name", "station"),
                                       ("time_id", "time")):
                identifier = row.get(key)
                if isinstance(identifier, str) and identifier.strip():
                    roles.setdefault(identifier.strip(), set()).add(endpoint_role)
    for aliases in records:
        record_roles = set().union(*(roles[identifier] for identifier in aliases))
        for identifier in aliases:
            roles[identifier].update(record_roles)
    return records, roles, missing_ids


def _context_fields(row):
    channels = {row[key].strip() for key in ("channel_name", "station_id")
                if isinstance(row.get(key), str) and row[key].strip()}
    event_type = _norm(row.get("event_type") or row.get("event_category"))
    return channels, event_type


def _case_context_match(case, events):
    """Return 1/0/None for an exact-metadata analogue, never infer missing data."""
    split = _norm(case.get("split"))
    if not split:
        return None
    if split not in {"train", "val"}:
        return 0
    channels, event_type = _context_fields(case)
    if not channels or not event_type or not events:
        return None
    incomplete = False
    for event in events:
        if not isinstance(event, Mapping):
            incomplete = True
            continue
        event_channels, current_type = _context_fields(event)
        if not event_channels or not current_type:
            incomplete = True
        elif channels & event_channels and event_type == current_type:
            return 1
    return None if incomplete else 0


def evaluate_explanation_quality(
    markdown: str,
    accepted_evidence: Sequence[dict] | None = None,
    decision: dict | None = None,
    residual_cases: Sequence[object] | None = None,
    structured_events: Sequence[dict] | None = None,
    historical_event_cases: Sequence[dict] | None = None,
    selected_residual_memory_skills: Sequence[dict] | None = None,
    *,
    structured_output: Mapping | None = None,
    generation_status: str | None = None,
    forecast_time_input_verified: bool | None = None,
) -> dict[str, object]:
    """Score actual output, preserving the original seven positional arguments.

    structured_output: {claims: [{text, evidence_ids, case_ids}], decision: str,
    links: [{from_id, to_id, relation}], uncertainty: str}. Pass the raw generated
    object without repairing claims/decisions from input. IDs are case-sensitive.
    id/evidence_id/case_id are aliases of one supplied source. Collections assign
    event/historical/residual roles; optional kind/type/entity_type supports typed
    station/time records. station_id/channel_name and time_id register additional
    endpoint IDs. An unambiguous output action enables the reserved endpoint
    'decision'. Endpoint existence/type is a proxy, NOT causal or semantic proof.

    Metrics (each numerator and effective denominator is returned):
    - evidence_coverage: distinct cited source records / supplied source records.
    - residual_case_relevance: metadata-matched case IDs / unique cited case IDs.
      Each must resolve to a supplied historical/residual record with split=train
      or val, share an exact channel_name/station_id with at least one supplied
      event, and match that SAME event's event_type (event_category fallback;
      case/whitespace normalized). Unknown IDs, explicit non-train/val splits,
      and complete-but-mismatching metadata score zero. Missing/ambiguous case
      metadata or event context makes the metric NA, with per-ID matches and
      unverifiable counts. This is a context-matching proxy, NOT causal relevance.
      case_reference_precision separately reports ID traceability only.
    - groundedness: nonempty claims with >=1 reference and ALL IDs resolved to
      their correct reference class / claims; unsupported_claim_rate complements it.
    - redundancy_rate: repeated normalized claim text / claims.
    - multi_hop_completeness: valid typed relation kinds / four required kinds.
    - decision_consistency: apply/partial_apply map to apply; abstain/explain_only/
      no_event_keep_raw map to preserve. Explicit expected decision takes priority
      over the legacy abstain flag. Conflicting actions fail for either expectation.
      exact_decision_consistency separately compares recognized exact state names;
      it is NA for prose-only decisions, missing states, or invalid/failed output.
    - leakage_free_rate: caller's forecast_time_input_verified attestation / 1.
      Supply ONLY the external recursive input guard result, never a generated
      self-attestation. Scope is provided inputs, not real-world truth/availability.

    Markdown-only compatibility calls use explicit [ID], [evidence:ID], [case:ID]
    references for coverage; unverifiable claim/link metrics remain NA. Structured
    output is authoritative for those metrics. Markdown actions are also checked
    for contradictions. Action parsing is conservative and vocabulary-based.
    No mode, input keyword, source title, retrieval score, or selected skill bonus.

    NA is None with *_status, *_numerator=None, *_denominator=0. metric_status,
    metric_denominators and effective_counts provide aggregate-friendly maps.
    Failed generation suppresses scores even with stale supplied output. Accepted
    generation statuses: success, ok, completed, generated. None evaluates supplied
    output without attesting generation success. Legacy keys use new semantics.
    """
    result = {
        "evaluator": "mode_blind_rule_proxy_v1",
        "grounding_scope": "reference_traceability_not_factual_entailment",
        "leakage_scope": "provided_inputs_only_external_guard_attestation",
        "generation_status": generation_status or "unspecified",
        "metric_status": {}, "metric_denominators": {}, "effective_counts": {},
        "exact_decision_consistency": None,
        "exact_decision_consistency_status": "unavailable",
    }

    def put(name, numerator=None, denominator=0, status="unverifiable"):
        result[name] = numerator / denominator if numerator is not None and denominator else None
        result[name + "_status"] = status
        result[name + "_numerator"] = numerator
        result[name + "_denominator"] = denominator
        result["metric_status"][name] = status
        result["metric_denominators"][name] = denominator
        result["effective_counts"][name] = denominator

    for metric in METRICS:
        put(metric)
    put("case_reference_precision")

    groups = ((accepted_evidence, "evidence"), (structured_events, "event"),
              (historical_event_cases, "historical"), (residual_cases, "residual"))
    records, roles, missing_ids = _registry(groups)
    known = set().union(*records) if records else set()
    case_ids = {key for key in known if roles[key] & {"historical", "residual"}}
    registry_available = any(rows is not None for rows, _ in groups)
    text = markdown if isinstance(markdown, str) else ""
    supplied = structured_output is not None
    obj = structured_output if isinstance(structured_output, Mapping) else {}
    claims = obj.get("claims")
    claims_valid = isinstance(claims, list) and all(
        isinstance(claim, Mapping) and isinstance(claim.get("text"), str)
        and _references(claim.get("evidence_ids")) and _references(claim.get("case_ids"))
        for claim in claims)
    schema_valid = (claims_valid and isinstance(obj.get("decision"), str)
                    and isinstance(obj.get("links"), list)
                    and isinstance(obj.get("uncertainty"), str))
    result["structured_output_status"] = (
        "valid" if schema_valid else "invalid" if supplied else "not_supplied")
    refs, case_refs = set(), []
    if claims_valid:
        for claim in claims:
            if claim["text"].strip():
                refs.update(item.strip() for item in claim["evidence_ids"] + claim["case_ids"])
                case_refs.extend(item.strip() for item in claim["case_ids"])
    elif not supplied:
        refs = set(re.findall(r"\[(?:evidence:|case:)?([\w.:-]+)\]", text))
    result["referenced_ids"] = sorted(refs)
    result["unresolved_ids"] = sorted(refs - known)
    result["unidentifiable_source_records"] = missing_ids
    result["accepted_evidence_count"] = len(accepted_evidence or [])
    result["selected_residual_memory_skill_count"] = len(selected_residual_memory_skills or [])

    failed = generation_status is not None and _norm(generation_status) not in {
        "success", "ok", "completed", "generated"}
    empty = not text.strip() and not (
        claims_valid and any(claim["text"].strip() for claim in claims)
        or isinstance(obj.get("decision"), str) and obj["decision"].strip()
        or isinstance(obj.get("uncertainty"), str) and obj["uncertainty"].strip())
    result["output_status"] = "generation_failed" if failed else "empty" if empty else "present"
    if failed or (supplied and not schema_valid) or empty:
        for metric in (*METRICS, "case_reference_precision"):
            put(metric, status=("generation_failed" if failed else
                                "invalid_structured_output" if supplied and not schema_valid else "empty_output"))
    else:
        if missing_ids or not registry_available:
            put("evidence_coverage", status="missing_evidence_ids" if missing_ids else "missing_evidence_registry")
        elif records:
            put("evidence_coverage", sum(bool(aliases & refs) for aliases in records), len(records),
                "scored_reference_proxy")
        else:
            put("evidence_coverage", status="no_evidence")

        if claims_valid and claims:
            if registry_available and not missing_ids:
                grounded = sum(bool(claim["text"].strip())
                               and bool(claim["evidence_ids"] + claim["case_ids"])
                               and all(item.strip() in known for item in claim["evidence_ids"])
                               and all(item.strip() in case_ids for item in claim["case_ids"])
                               for claim in claims)
                put("groundedness", grounded, len(claims), "scored_reference_proxy")
                put("unsupported_claim_rate", len(claims) - grounded, len(claims), "scored_reference_proxy")
            else:
                for metric in ("groundedness", "unsupported_claim_rate"):
                    put(metric, status="missing_evidence_registry" if not registry_available else "missing_evidence_ids")
            normalized = [_norm(claim["text"]) for claim in claims]
            put("redundancy_rate", len(normalized) - len(set(normalized)), len(normalized), "scored_exact_text_proxy")
        else:
            for metric in ("groundedness", "unsupported_claim_rate", "redundancy_rate"):
                put(metric, status="no_claims" if claims_valid else "unstructured_output")

        if case_refs and registry_available and not missing_ids:
            unique_case_refs = set(case_refs)
            put("case_reference_precision", len(unique_case_refs & case_ids), len(unique_case_refs), "scored_reference_proxy")
            case_rows = list(historical_event_cases or []) + list(residual_cases or [])
            matches = {}
            for identifier in sorted(unique_case_refs):
                if identifier not in case_ids:
                    matches[identifier] = 0
                    continue
                aliases = next(record for record in records if identifier in record)
                matching_rows = [row for row in case_rows if _ids(row) & aliases]
                values = {_case_context_match(row, structured_events) for row in matching_rows}
                # Conflicting records under one ID do not establish provenance.
                matches[identifier] = next(iter(values)) if len(values) == 1 else None
            unknown_count = sum(value is None for value in matches.values())
            result["case_context_matches"] = matches
            result["residual_case_relevance_unverifiable_count"] = unknown_count
            result["residual_case_relevance_checked_count"] = len(matches) - unknown_count
            if unknown_count:
                put("residual_case_relevance", status="unverifiable_case_context")
            else:
                put("residual_case_relevance", sum(matches.values()), len(matches), "scored_context_match_proxy")
        else:
            for metric in ("residual_case_relevance", "case_reference_precision"):
                put(metric, status=("no_case_references" if not case_refs else
                                    "missing_evidence_registry" if not registry_available else "missing_evidence_ids"))

        output_actions = _actions(obj.get("decision", "")) if supplied else set()
        claim_actions = set().union(*(_actions(claim["text"]) for claim in claims)) if claims_valid else set()
        actions = output_actions | _actions(text) | claim_actions
        expected = None
        expected_state = None
        output_state = _norm(obj.get("decision")) if supplied else None
        if isinstance(decision, Mapping):
            if "decision" in decision:
                # Never turn an explicit preserve/unknown state into apply via a
                # legacy boolean computed as (state == 'abstain').
                expected_state = _norm(decision["decision"])
                expected = DECISION_ACTIONS.get(expected_state)
            elif isinstance(decision.get("abstain"), bool):
                expected = "preserve" if decision["abstain"] else "apply"
        if expected_state in DECISION_ACTIONS and output_state in DECISION_ACTIONS:
            result["exact_decision_consistency"] = float(expected_state == output_state and len(actions) == 1)
            result["exact_decision_consistency_status"] = "scored_exact_state_proxy"
        if len(actions) > 1:
            put("decision_consistency", 0, 1, "contradictory_output")
        elif expected is None:
            put("decision_consistency", status="missing_expected_decision")
        elif not actions or (supplied and not output_actions):
            put("decision_consistency", status="missing_output_decision")
        else:
            put("decision_consistency", int(actions == {expected}), 1, "scored_explicit_action_proxy")

        valid_relations, invalid_links = set(), 0
        if supplied:
            if len(output_actions) == 1 and len(actions) == 1:
                roles["decision"] = {"decision"}
            for link in obj["links"]:
                if not isinstance(link, Mapping):
                    invalid_links += 1
                    continue
                relation = link.get("relation")
                endpoints = (link.get("from_id"), link.get("to_id"))
                if (not isinstance(relation, str) or relation not in RELATIONS
                        or not all(isinstance(item, str) for item in endpoints)):
                    invalid_links += 1
                    continue
                source_role, target_role = RELATIONS[relation]
                if (source_role in roles.get(endpoints[0], set())
                        and target_role in roles.get(endpoints[1], set())
                        and endpoints[0] != endpoints[1]):
                    valid_relations.add(relation)
                else:
                    invalid_links += 1
            if obj["links"] and (missing_ids or not registry_available):
                put("multi_hop_completeness", status="missing_evidence_ids" if missing_ids else "missing_evidence_registry")
            else:
                put("multi_hop_completeness", len(valid_relations), len(RELATIONS), "scored_typed_link_proxy")
        else:
            put("multi_hop_completeness", status="unstructured_output")
        result["valid_relations"] = sorted(valid_relations)
        result["invalid_link_count"] = invalid_links

        if type(forecast_time_input_verified) is bool:
            put("leakage_free_rate", int(forecast_time_input_verified), 1,
                "provided_inputs_guard_passed" if forecast_time_input_verified else "provided_inputs_guard_failed")
        else:
            put("leakage_free_rate", status="missing_input_guard_attestation")

    # Compatibility names retain new proxy/NA semantics, not old keyword scores.
    result["citation_coverage"] = result["evidence_coverage"]
    result["evidence_relevance_mean"] = None
    result["skill_guidance_coverage"] = None
    relations = result.get("valid_relations", [])
    result["station_event_linking"] = ("event_station" in relations
                                      if result["multi_hop_completeness"] is not None else None)
    result["multi_hop_reasoning"] = (result["multi_hop_completeness"] == 1.0
                                     if result["multi_hop_completeness"] is not None else None)
    consistent = result["decision_consistency"]
    result["calibration_decision_consistent"] = bool(consistent) if consistent is not None else None
    result["abstention_correct"] = (result["calibration_decision_consistent"]
                                    if isinstance(decision, Mapping) and decision.get("abstain") is True else None)
    result["residual_cases_deduplicated"] = (len(case_refs) == len(set(case_refs))
                                            if case_refs and not failed else None)
    return result
