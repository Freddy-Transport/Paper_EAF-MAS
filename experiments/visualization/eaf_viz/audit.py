"""Evidence audit helpers for visualization."""

from __future__ import annotations

from typing import Any, Dict, List


AUDIT_SCORE_FIELDS = [
    "source_validity_score",
    "geo_consistency_score",
    "temporal_alignment_score",
    "semantic_consistency_score",
    "residual_support_score",
    "evidence_validity_score",
]


def audit_scores(evidence: Dict[str, Any]) -> Dict[str, Any]:
    audit = evidence.get("evidence_audit") or {}
    return {field: audit.get(field) for field in AUDIT_SCORE_FIELDS}


def conflict_flags(evidence: Dict[str, Any]) -> List[str]:
    audit = evidence.get("evidence_audit") or {}
    flags = []
    flags.extend(audit.get("conflict_flags") or [])
    flags.extend(audit.get("severe_conflict_flags") or [])
    return [str(x) for x in flags]


def event_rows(case_id: str, payload: Dict[str, Any]) -> List[dict]:
    request = payload.get("request") or {}
    evidence = payload.get("evidence") or {}
    decision = payload.get("decision") or {}
    scores = audit_scores(evidence)
    rows = []
    for idx, event in enumerate(evidence.get("structured_events") or []):
        rows.append(
            {
                "case_id": case_id,
                "anchor_time": request.get("date"),
                "event_id": f"{case_id}_event_{idx}",
                "event_name": event.get("title"),
                "event_type": event.get("event_type") or event.get("event_category"),
                "event_start_time": event.get("event_time") or event.get("start_time"),
                "event_end_time": event.get("end_time"),
                "venue": event.get("venue_name"),
                "event_location_text": event.get("location"),
                "matched_station_ids": event.get("station_complex_id"),
                "matched_station_names": event.get("channel_name"),
                "impact_tier": event.get("impact_tier"),
                "source_type": _source_type(evidence),
                "source_quality": _source_quality(evidence),
                "source_timestamp": _first_source_value(evidence, "source_time"),
                "retrieval_timestamp": _first_source_value(evidence, "retrieved_at"),
                "known_before_anchor": _first_source_value(evidence, "source_time_status") == "known_before_anchor",
                "post_event_source_flag": "possible_post_event_report" in ",".join(conflict_flags(evidence)),
                "citation_quality_flag": bool(evidence.get("sources")),
                "model_assisted_flag": bool(evidence.get("model_assisted_summaries")),
                "conflict_flags": ";".join(conflict_flags(evidence)),
                "calibration_candidate": bool(decision.get("adjusted_channels")),
                "event_decision": "abstain" if decision.get("abstain") else "apply",
                "event_decision_reason": decision.get("reason"),
                **scores,
            }
        )
    return rows


def _first_source_value(evidence: Dict[str, Any], key: str):
    for src in evidence.get("sources") or []:
        if src.get(key):
            return src.get(key)
    return None


def _source_type(evidence: Dict[str, Any]) -> str:
    if evidence.get("sources"):
        return "citation-quality"
    if evidence.get("model_assisted_summaries"):
        return "model-assisted"
    return "structured"


def _source_quality(evidence: Dict[str, Any]) -> str:
    if evidence.get("sources"):
        return "verified"
    if evidence.get("model_assisted_summaries"):
        return "non-citable-summary"
    return "structured-only"
