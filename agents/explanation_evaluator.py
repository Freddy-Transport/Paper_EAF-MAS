"""Lightweight forecast explanation quality metrics.

These metrics are deliberately mechanical. They are not a substitute for human
evaluation, but they make the paper experiments audit whether each agent did
its job: cite evidence when available, explain station-event links, keep
forecast-time reasoning separate, and stay consistent with abstention.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Sequence


def _norm_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


def _has_any(text: str, terms: Sequence[str]) -> bool:
    return any(term in text for term in terms)


def _residual_cases_unique(residual_cases: Iterable[object]) -> bool:
    seen = set()
    for case in residual_cases or []:
        key = _norm_text(case)
        if key in seen:
            return False
        seen.add(key)
    return True


def _forecast_time_text(markdown: str) -> str:
    text = str(markdown or "")
    return re.split(r"\n##\s*Post-hoc Metrics\b", text, flags=re.I)[0].lower()


def _event_terms(rows: Sequence[dict] | None) -> set[str]:
    terms: set[str] = set()
    for row in rows or []:
        for key in ("event_type", "event_category", "title", "impact_tier"):
            value = str(row.get(key) or "").lower()
            for token in re.findall(r"[a-z0-9]+", value):
                if len(token) >= 3:
                    terms.add(token)
    return terms


def _residual_case_relevance(
    markdown: str,
    residual_cases: Sequence[object] | None,
    structured_events: Sequence[dict] | None,
    historical_event_cases: Sequence[dict] | None,
) -> float:
    text = _norm_text(markdown)
    if not residual_cases and not historical_event_cases:
        return 0.0
    score = 0.0
    possible = 0.0
    terms = _event_terms(structured_events)
    if terms:
        possible += 1.0
        if any(term in text for term in terms):
            score += 1.0
    if historical_event_cases:
        possible += 1.0
        if _has_any(text, ["train/validation", "historical", "memory", "similar"]):
            score += 1.0
    if residual_cases:
        possible += 1.0
        if _has_any(text, ["residual", "correction", "lp-moment"]):
            score += 1.0
    return float(score / possible) if possible else 0.0


def _multi_hop_completeness(text: str) -> float:
    required = [
        ["event", "plaza", "concert", "parade", "festival", "game"],
        ["venue", "location", "times square", "herald", "station", "nearby"],
        ["hour", "forecast horizon", "timing", "time"],
        ["train/validation", "historical", "similar"],
        ["residual", "lp-moment", "correction pattern"],
        ["calibration", "correction", "abstain", "bounded"],
    ]
    hits = sum(1 for group in required if _has_any(text, group))
    return float(hits / len(required))


def evaluate_explanation_quality(
    markdown: str,
    accepted_evidence: Sequence[dict] | None = None,
    decision: Dict | None = None,
    residual_cases: Sequence[object] | None = None,
    structured_events: Sequence[dict] | None = None,
    historical_event_cases: Sequence[dict] | None = None,
    selected_residual_memory_skills: Sequence[dict] | None = None,
) -> Dict[str, object]:
    """Return automatic explanation metrics for one forecast artifact."""
    text = _norm_text(markdown)
    accepted = list(accepted_evidence or [])
    urls = [str(row.get("url") or "").strip() for row in accepted if row.get("url")]
    citation_coverage = 1.0 if urls and any(url.lower() in text for url in urls) else 0.0

    has_station_link = _has_any(text, ["station", "channel", "subway", "nearby"]) and _has_any(
        text, ["venue", "location", "distance", "walking", "herald", "times square"]
    )
    has_time_link = _has_any(text, ["hour", "hourly", "time", "timing", "forecast horizon", "demand"])
    has_event_link = _has_any(text, ["event", "concert", "plaza", "parade", "game", "venue"])
    multi_hop_score = _multi_hop_completeness(text)
    multi_hop = has_station_link and has_time_link and has_event_link

    abstain = bool((decision or {}).get("abstain"))
    mentions_abstain = _has_any(text, ["abstain", "no correction", "keep raw", "retained"])
    mentions_apply = _has_any(text, ["applies", "apply a", "positive correction", "negative correction", "calibration is applied"])
    if abstain:
        consistent = mentions_abstain
    else:
        consistent = mentions_apply or _has_any(text, ["calibration", "correction", "adjustment"])

    relevance_values = []
    for row in accepted:
        try:
            relevance_values.append(float(row.get("relevance_score")))
        except Exception:
            pass

    evidence_sections = [
        "structured future events within horizon" in text or bool(structured_events),
        "historical event memory from train/validation" in text or bool(historical_event_cases),
        "residual pattern evidence" in text or bool(residual_cases),
        bool(urls) and any(url.lower() in text for url in urls) or "model-assisted summary" in text,
    ]
    evidence_coverage = float(sum(1 for item in evidence_sections if item) / len(evidence_sections))
    residual_relevance = _residual_case_relevance(markdown, residual_cases, structured_events, historical_event_cases)
    forecast_text = _forecast_time_text(markdown)
    leakage_free = not _has_any(forecast_text, ["ground truth", "post-hoc", "post hoc", "actual wape", "realized ridership"])
    unsupported_patterns = 0
    if not urls and _has_any(forecast_text, ["according to", "cited source", "external citation", "url evidence"]):
        unsupported_patterns += 1
    if _has_any(forecast_text, ["will definitely", "certainly increase", "guaranteed"]):
        unsupported_patterns += 1
    unsupported_claim_rate = float(min(1.0, unsupported_patterns / 2.0))
    total_residual = len(list(residual_cases or []))
    unique_residual = len({_norm_text(case) for case in residual_cases or []})
    redundancy_rate = float(1.0 - unique_residual / total_residual) if total_residual else 0.0
    groundedness = float(
        (
            evidence_coverage
            + residual_relevance
            + (1.0 if consistent else 0.0)
            + (1.0 - unsupported_claim_rate)
        )
        / 4.0
    )
    skill_count = len(list(selected_residual_memory_skills or []))
    skill_guidance_coverage = 0.0
    if skill_count:
        skill_guidance_coverage = 1.0 if _has_any(text, ["residual-memory skill", "residual_memory_skill", "skill guidance"]) else 0.0

    return {
        "accepted_evidence_count": len(accepted),
        "citation_coverage": float(citation_coverage),
        "evidence_relevance_mean": float(sum(relevance_values) / len(relevance_values)) if relevance_values else 0.0,
        "station_event_linking": bool(has_station_link),
        "multi_hop_reasoning": bool(multi_hop),
        "evidence_coverage": evidence_coverage,
        "residual_case_relevance": residual_relevance,
        "multi_hop_completeness": multi_hop_score,
        "groundedness": groundedness,
        "unsupported_claim_rate": unsupported_claim_rate,
        "leakage_free_rate": 1.0 if leakage_free else 0.0,
        "redundancy_rate": redundancy_rate,
        "selected_residual_memory_skill_count": skill_count,
        "skill_guidance_coverage": skill_guidance_coverage,
        "calibration_decision_consistent": bool(consistent),
        "abstention_correct": bool(consistent) if abstain else None,
        "residual_cases_deduplicated": _residual_cases_unique(residual_cases or []),
    }
