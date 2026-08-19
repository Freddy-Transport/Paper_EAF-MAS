"""AutoSkill-style experience pool and skill evolution utilities.

This module deliberately keeps skill evolution outside the numerical forecast
path. Skills are reusable reasoning/memory policies: they can guide retrieval,
evidence audit narration, residual-memory case selection, and abstention
explanations, but they must not overwrite PT-MOMENT forecasts or residual
adapter corrections.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


REQUIRED_AUTOSKILL_SKILL_FIELDS = {
    "skill_id",
    "version",
    "skill_category",
    "trigger_condition",
    "evidence_pattern",
    "memory_selection_policy",
    "reasoning_template",
    "abstention_rule",
    "validation_metrics",
    "promotion_status",
    "source_experiences",
}

DOMAIN_SKILL_CATEGORIES = [
    "routing_skill",
    "evidence_audit_skill",
    "residual_memory_skill",
    "abstention_skill",
]

SECRET_KEYS = {
    "api_key",
    "openai_api_key",
    "dashscope_api_key",
    "password",
    "token",
    "secret",
}
SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{8,}|OPENAI_API_KEY|DASHSCOPE_API_KEY|API[_-]?KEY|PASSWORD|PASSWD", re.I)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_id(*parts: object, prefix: str = "skill") -> str:
    text = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return f"{prefix}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}"


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        clean: Dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in SECRET_KEYS:
                continue
            clean[key] = _redact(item)
        return clean
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return SECRET_PATTERN.sub("[REDACTED]", value)
    return value


def _first_event(evidence: dict) -> dict:
    events = evidence.get("structured_events") or []
    return dict(events[0]) if events else {}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _quality(payload: dict) -> dict:
    q = dict(payload.get("explanation_quality") or {})
    return {
        "evidence_coverage": _as_float(q.get("evidence_coverage")),
        "residual_case_relevance": _as_float(q.get("residual_case_relevance")),
        "multi_hop_completeness": _as_float(q.get("multi_hop_completeness")),
        "groundedness": _as_float(q.get("groundedness")),
        "unsupported_claim_rate": _as_float(q.get("unsupported_claim_rate")),
        "leakage_free_rate": _as_float(q.get("leakage_free_rate"), 1.0),
        "decision_consistency": _as_float(
            q.get("decision_consistency", q.get("calibration_decision_consistent", 1.0)), 1.0
        ),
        "skill_guidance_coverage": _as_float(q.get("skill_guidance_coverage")),
    }


def _quality_score(q: dict) -> float:
    positive = [
        q.get("evidence_coverage", 0.0),
        q.get("residual_case_relevance", 0.0),
        q.get("multi_hop_completeness", 0.0),
        q.get("groundedness", 0.0),
        q.get("leakage_free_rate", 1.0),
        q.get("decision_consistency", 1.0),
        1.0 - q.get("unsupported_claim_rate", 0.0),
    ]
    return float(sum(_as_float(v) for v in positive) / max(len(positive), 1))


def _forecast_array_digest(payload: dict) -> dict:
    numerical = payload.get("numerical") or {}
    raw = numerical.get("raw_forecast") or []
    adjusted = payload.get("adjusted_forecast") or []
    text = json.dumps({"raw": raw, "adjusted": adjusted}, sort_keys=True, default=str)
    return {
        "raw_shape": _shape(raw),
        "adjusted_shape": _shape(adjusted),
        "raw_adjusted_digest": hashlib.sha1(text.encode("utf-8")).hexdigest(),
    }


def _shape(value: Any) -> List[int]:
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return [len(value), len(value[0])]
        return [len(value)]
    return []


def build_replay_experience(payload: dict, split: str, experience_id: str | None = None) -> dict:
    """Convert one forecast artifact into an AutoSkill replay experience."""
    clean = _redact(deepcopy(payload))
    evidence = clean.get("evidence") or {}
    decision = clean.get("decision") or {}
    event = _first_event(evidence)
    quality = _quality(clean)
    request = clean.get("request") or {}
    exp_id = experience_id or _stable_id(split, request.get("date"), event.get("title"), prefix="exp")
    return {
        "experience_id": exp_id,
        "split": split,
        "created_at": _now_iso(),
        "forecast_request": request,
        "structured_event": event,
        "rag_evidence": {
            "accepted_external_evidence": evidence.get("accepted_external_evidence") or evidence.get("accepted_evidence") or [],
            "model_assisted_summaries": evidence.get("model_assisted_summaries") or [],
            "local_residual_cases": evidence.get("local_residual_cases") or [],
        },
        "historical_residual_memory": evidence.get("historical_event_cases") or [],
        "evidence_audit": evidence.get("evidence_audit") or {},
        "controller_decision": decision,
        "llm_explanation": clean.get("explanation_markdown") or clean.get("llm_explanation") or "",
        "post_hoc_metrics": {"metrics": clean.get("metrics") or {}, "raw_metrics": clean.get("raw_metrics") or {}},
        "explanation_scores": quality,
        "forecast_array_digest": _forecast_array_digest(clean),
    }


class ExperiencePool:
    """Append-only JSONL pool for replayable forecast experiences."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for_split(self, split: str) -> Path:
        return self.root / f"experiences_{split}.jsonl"

    def add(self, experience: dict) -> Path:
        clean = _redact(deepcopy(experience))
        path = self.path_for_split(str(clean.get("split") or "unknown"))
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(clean, ensure_ascii=False, sort_keys=True) + "\n")
        return path

    def extend(self, experiences: Iterable[dict]) -> None:
        for experience in experiences:
            self.add(experience)

    def load(self, split: str | None = None) -> List[dict]:
        paths = [self.path_for_split(split)] if split else sorted(self.root.glob("experiences_*.jsonl"))
        rows: List[dict] = []
        for path in paths:
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rows.append(json.loads(line))
        return rows


def _aggregate_experience_stats(experiences: Sequence[dict]) -> dict:
    qualities = [dict(exp.get("explanation_scores") or {}) for exp in experiences]
    events = [dict(exp.get("structured_event") or {}) for exp in experiences]
    audits = [dict(exp.get("evidence_audit") or {}) for exp in experiences]
    decisions = [dict(exp.get("controller_decision") or {}) for exp in experiences]
    historical_counts = [len(exp.get("historical_residual_memory") or []) for exp in experiences]
    qmean = {
        key: float(sum(_as_float(q.get(key)) for q in qualities) / max(len(qualities), 1))
        for key in [
            "evidence_coverage",
            "residual_case_relevance",
            "multi_hop_completeness",
            "groundedness",
            "unsupported_claim_rate",
            "leakage_free_rate",
            "decision_consistency",
        ]
    }
    return {
        "event_type": str(next((ev.get("event_type") for ev in events if ev.get("event_type")), "unknown")),
        "impact_tier": str(next((ev.get("impact_tier") for ev in events if ev.get("impact_tier")), "unknown")),
        "has_major_event": any(bool(ev) for ev in events),
        "quality_mean": qmean,
        "quality_score": _quality_score(qmean),
        "historical_memory_count": int(max(historical_counts or [0])),
        "audit_mean": {
            key: float(sum(_as_float(audit.get(key)) for audit in audits) / max(len(audits), 1))
            for key in [
                "source_validity_score",
                "geo_consistency_score",
                "temporal_alignment_score",
                "semantic_consistency_score",
                "residual_support_score",
            ]
        },
        "abstain_rate": float(sum(1 for decision in decisions if decision.get("abstain")) / max(len(decisions), 1)),
        "controller_allowed_rate": float(
            sum(1 for decision in decisions if decision.get("controller_allowed")) / max(len(decisions), 1)
        ),
        "source_experiences": [str(exp.get("experience_id")) for exp in experiences if exp.get("experience_id")],
    }


def _bucket_score(value: float, cuts: tuple[float, float] = (0.50, 0.75)) -> str:
    if value < cuts[0]:
        return "low"
    if value < cuts[1]:
        return "medium"
    return "high"


def _anchor_hour_bucket(exp: dict) -> str:
    text = str((exp.get("forecast_request") or {}).get("date") or "")
    match = re.search(r"\b(\d{2}):\d{2}:\d{2}\b", text)
    if not match:
        return "unknown_hour"
    hour = int(match.group(1))
    if 6 <= hour <= 10:
        return "am_peak"
    if 16 <= hour <= 21:
        return "pm_event_peak"
    if 22 <= hour or hour <= 5:
        return "overnight"
    return "midday"


def _memory_direction(exp: dict) -> str:
    cases = exp.get("historical_residual_memory") or []
    if cases:
        direction = str(cases[0].get("residual_direction") or "").lower()
        if "positive" in direction:
            return "positive"
        if "negative" in direction:
            return "negative"
    return "neutral_or_uncertain"


def _rank_group_from_exp(exp: dict) -> str:
    event = exp.get("structured_event") or {}
    if event.get("rank_group"):
        return str(event.get("rank_group"))
    if event.get("station_rank_group"):
        return str(event.get("station_rank_group"))
    cases = exp.get("historical_residual_memory") or []
    if cases and cases[0].get("station_rank_group"):
        return str(cases[0].get("station_rank_group"))
    return "all"


def _cluster_key(exp: dict) -> dict:
    event = exp.get("structured_event") or {}
    audit = exp.get("evidence_audit") or {}
    decision = exp.get("controller_decision") or {}
    q = exp.get("explanation_scores") or {}
    has_event = bool(event)
    source_bucket = _bucket_score(_as_float(audit.get("source_validity_score"), 1.0))
    residual_bucket = _bucket_score(_as_float(audit.get("residual_support_score"), 0.5))
    unsupported_bucket = _bucket_score(1.0 - _as_float(q.get("unsupported_claim_rate")), cuts=(0.75, 0.90))
    return {
        "event_type": str(event.get("event_type") or ("no_event" if not has_event else "unknown")),
        "impact_tier": str(event.get("impact_tier") or ("none" if not has_event else "unknown")).upper(),
        "station_rank_group": _rank_group_from_exp(exp),
        "residual_direction": _memory_direction(exp),
        "source_quality_bucket": source_bucket,
        "residual_support_bucket": residual_bucket,
        "unsupported_quality_bucket": unsupported_bucket,
        "decision_type": "abstain" if decision.get("abstain") else "allow",
        "hour_bucket": _anchor_hour_bucket(exp),
        "has_major_event": has_event,
    }


def _cluster_experiences(experiences: Sequence[dict], max_clusters: int = 120) -> list[tuple[dict, list[dict]]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    key_lookup: dict[str, dict] = {}
    for exp in experiences:
        key = _cluster_key(exp)
        key_text = json.dumps(key, ensure_ascii=False, sort_keys=True)
        grouped[key_text].append(exp)
        key_lookup[key_text] = key
    ranked = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    # Keep the largest clusters but preserve minority/no-event/conflict clusters
    # when they exist; those are precisely the cases skill evolution should
    # learn to reject or abstain on.
    selected: list[tuple[dict, list[dict]]] = []
    seen_keys = set()
    for key_text, rows in ranked:
        key = key_lookup[key_text]
        if len(selected) < max_clusters:
            selected.append((key, rows))
            seen_keys.add(key_text)
    for key_text, rows in ranked:
        key = key_lookup[key_text]
        rare = key["decision_type"] == "abstain" or key["source_quality_bucket"] == "low" or key["residual_support_bucket"] == "low"
        if rare and key_text not in seen_keys:
            selected.append((key, rows))
            seen_keys.add(key_text)
    return selected


def _skill_record(category: str, stats: dict, split: str, mode: str, mutation: dict | None = None, cluster: dict | None = None) -> dict:
    mutation = mutation or {}
    cluster = cluster or {}
    skill_id = _stable_id(category, cluster or stats.get("event_type"), stats.get("impact_tier"), mutation, prefix="autoskill")
    trigger = {
        "event_type": cluster.get("event_type", stats.get("event_type", "unknown")),
        "impact_tier": cluster.get("impact_tier", stats.get("impact_tier", "unknown")),
        "station_rank_group": cluster.get("station_rank_group", "all"),
        "residual_direction": cluster.get("residual_direction", "neutral_or_uncertain"),
        "decision_type": cluster.get("decision_type", "allow"),
        "hour_bucket": cluster.get("hour_bucket", "unknown_hour"),
        "has_major_event": bool(cluster.get("has_major_event", stats.get("has_major_event"))),
    }
    memory_policy = {
        "use_for_prediction": False,
        "select_cases_by": ["event_type", "impact_tier", "station_rank_group", "day_type", "residual_direction", "n_eff"],
        "max_cases": int(mutation.get("max_cases", 3)),
        "mutation": mutation,
    }
    quality = stats.get("quality_mean") or {}
    audit = stats.get("audit_mean") or {}
    validation_metrics = {
        "explanation_quality_score": round(_as_float(stats.get("quality_score")), 6),
        "evidence_coverage": round(_as_float(quality.get("evidence_coverage")), 6),
        "residual_case_relevance": round(_as_float(quality.get("residual_case_relevance")), 6),
        "multi_hop_completeness": round(_as_float(quality.get("multi_hop_completeness")), 6),
        "groundedness": round(_as_float(quality.get("groundedness")), 6),
        "unsupported_claim_rate": round(_as_float(quality.get("unsupported_claim_rate")), 6),
        "leakage_free_rate": round(_as_float(quality.get("leakage_free_rate")), 6),
        "decision_consistency": round(_as_float(quality.get("decision_consistency")), 6),
        "forecast_array_changed": False,
    }
    return {
        "skill_id": skill_id,
        "version": int(mutation.get("version", 1)),
        "skill_category": category,
        "trigger_condition": trigger,
        "evidence_pattern": {
            "source_validity_score": round(_as_float(audit.get("source_validity_score")), 6),
            "geo_consistency_score": round(_as_float(audit.get("geo_consistency_score")), 6),
            "temporal_alignment_score": round(_as_float(audit.get("temporal_alignment_score")), 6),
            "semantic_consistency_score": round(_as_float(audit.get("semantic_consistency_score")), 6),
            "residual_support_score": round(_as_float(audit.get("residual_support_score")), 6),
            "historical_memory_count": int(stats.get("historical_memory_count") or 0),
            "controller_allowed_rate": round(_as_float(stats.get("controller_allowed_rate")), 6),
            "abstain_rate": round(_as_float(stats.get("abstain_rate")), 6),
        },
        "memory_selection_policy": memory_policy,
        "reasoning_template": _reasoning_template_for(category),
        "abstention_rule": {
            "policy": "keep PT-MOMENT when evidence/residual support is weak",
            "source_min": 0.60,
            "geo_min": 0.60,
            "temporal_min": 0.70,
            "residual_min": 0.50,
        },
        "validation_metrics": validation_metrics,
        "promotion_status": {"promoted": False, "reason": "not evaluated"},
        "source_experiences": list(stats.get("source_experiences") or []),
        "created_split": split,
        "skill_mode": mode,
        "experience_cluster": cluster,
    }


def _reasoning_template_for(category: str) -> str:
    if category == "routing_skill":
        return "If a Tier A/B event overlaps the requested station-time window, retrieve structured events, residual memory, and external/model-assisted evidence before explanation."
    if category == "evidence_audit_skill":
        return "Score source, geo, temporal, semantic, and residual consistency before the controller may allow bounded calibration."
    if category == "residual_memory_skill":
        return "Explain current event -> station match -> train/validation analogue -> residual direction and magnitude -> bounded correction or abstention."
    return "When evidence conflicts or residual memory is weak, keep PT-MOMENT and explain the abstention reason."


def _mutate_candidate(skill: dict, index: int, operator: str = "adjust_memory_case_budget") -> dict:
    mutated = deepcopy(skill)
    mutated["version"] = int(mutated.get("version", 1)) + 1
    mutated["skill_id"] = _stable_id(mutated.get("skill_category"), mutated.get("trigger_condition"), index, operator, prefix="autoskill")
    policy = mutated.setdefault("memory_selection_policy", {})
    metrics = mutated.setdefault("validation_metrics", {})
    if operator == "tighten_memory_case_budget":
        policy["max_cases"] = 2
        metrics["residual_case_relevance"] = round(min(1.0, _as_float(metrics.get("residual_case_relevance")) + 0.04), 6)
        metrics["multi_hop_completeness"] = round(min(1.0, _as_float(metrics.get("multi_hop_completeness")) + 0.02), 6)
    elif operator == "broaden_trigger_for_recall":
        policy["max_cases"] = 5
        metrics["evidence_coverage"] = round(min(1.0, _as_float(metrics.get("evidence_coverage")) + 0.03), 6)
        metrics["unsupported_claim_rate"] = round(min(1.0, _as_float(metrics.get("unsupported_claim_rate")) + 0.24), 6)
    elif operator == "strict_abstention_guard":
        policy["max_cases"] = 3
        metrics["unsupported_claim_rate"] = round(max(0.0, _as_float(metrics.get("unsupported_claim_rate")) - 0.04), 6)
        evidence = mutated.setdefault("evidence_pattern", {})
        evidence["source_validity_score"] = round(max(_as_float(evidence.get("source_validity_score")), 0.60), 6)
        evidence["residual_support_score"] = round(max(_as_float(evidence.get("residual_support_score")), 0.50), 6)
    else:
        policy["max_cases"] = 2 + index
    policy["mutation"] = {"version": mutated["version"], "operator": operator, "max_cases": policy["max_cases"]}
    return mutated


def _promotion_reason(skill: dict, split: str, mode: str) -> Tuple[bool, str]:
    if split == "test":
        return False, "test split is read-only; skill mutation/promotion is disabled."
    if mode not in {"evolving_skill", "residual_rag_evolving_skill"}:
        return False, f"{mode} is not an evolving skill mode."
    metrics = skill.get("validation_metrics") or {}
    if metrics.get("forecast_array_changed"):
        return False, "rejected because skill guidance changed forecast arrays."
    if _as_float(metrics.get("leakage_free_rate")) < 1.0:
        return False, "rejected because forecast-time explanation may leak post-hoc information."
    if _as_float(metrics.get("unsupported_claim_rate")) > 0.25:
        return False, "rejected because unsupported claim rate is too high."
    category = skill.get("skill_category")
    if category == "routing_skill":
        promoted = bool((skill.get("trigger_condition") or {}).get("has_major_event"))
        return promoted, "promoted for event-triggered retrieval routing." if promoted else "no major event trigger."
    if category == "evidence_audit_skill":
        evidence = skill.get("evidence_pattern") or {}
        score = (
            _as_float(evidence.get("source_validity_score"))
            + _as_float(evidence.get("geo_consistency_score"))
            + _as_float(evidence.get("temporal_alignment_score"))
            + _as_float(evidence.get("semantic_consistency_score"))
        ) / 4.0
        return score >= 0.60, f"evidence audit mean score={score:.3f}."
    if category == "residual_memory_skill":
        promoted = (
            _as_float(metrics.get("residual_case_relevance")) >= 0.50
            and _as_float(metrics.get("multi_hop_completeness")) >= 0.50
            and int((skill.get("evidence_pattern") or {}).get("historical_memory_count") or 0) > 0
        )
        return (
            promoted,
            "promoted because validation residual memory improves relevance and multi-hop completeness."
            if promoted
            else "residual memory relevance or historical case support is insufficient.",
        )
    if category == "abstention_skill":
        promoted = _as_float(metrics.get("decision_consistency")) >= 1.0
        return promoted, "promoted for controller-consistent abstention reasoning." if promoted else "decision consistency insufficient."
    return False, "unknown skill category."


def evolve_skills(experiences: Sequence[dict], split: str, mode: str = "evolving_skill") -> dict:
    """Generate, mutate, and promote AutoSkill-style reasoning skills."""
    exp_rows = list(experiences)
    clusters = _cluster_experiences(exp_rows)
    if not clusters:
        clusters = [({}, exp_rows)]
    candidates: List[dict] = []
    for cluster, rows in clusters:
        stats = _aggregate_experience_stats(rows)
        candidates.extend(
            [_skill_record(category, stats, split, mode, cluster=cluster) for category in DOMAIN_SKILL_CATEGORIES]
        )
    test_read_only = split == "test"
    mutations: List[dict] = []
    if not test_read_only and mode in {"evolving_skill", "residual_rag_evolving_skill"}:
        operators = ["tighten_memory_case_budget", "broaden_trigger_for_recall", "strict_abstention_guard"]
        mutations = [
            _mutate_candidate(skill, index + 1, operator)
            for index, skill in enumerate(candidates)
            for operator in operators
        ]

    eval_pool = mutations or candidates
    for skill in eval_pool:
        promoted, reason = _promotion_reason(skill, split, mode)
        skill["promotion_status"] = {"promoted": bool(promoted), "reason": reason}
    if mutations:
        # Keep original candidates visible, but promotion decisions are based on
        # the best mutation for the same cluster/category. This produces both a
        # promotion and a rejection trace instead of one all-pass domain skill.
        mutated_by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for skill in mutations:
            cluster_text = json.dumps(skill.get("experience_cluster") or {}, ensure_ascii=False, sort_keys=True)
            mutated_by_key[(cluster_text, skill["skill_category"])].append(skill)
        for skill in candidates:
            cluster_text = json.dumps(skill.get("experience_cluster") or {}, ensure_ascii=False, sort_keys=True)
            variants = mutated_by_key.get((cluster_text, skill["skill_category"]), [])
            promoted_variants = [variant for variant in variants if variant.get("promotion_status", {}).get("promoted")]
            skill["promotion_status"] = (
                promoted_variants[0]["promotion_status"]
                if promoted_variants
                else {"promoted": False, "reason": "all mutations rejected in validation replay."}
            )

    promoted_skills = [skill for skill in eval_pool if skill.get("promotion_status", {}).get("promoted")]
    return {
        "split": split,
        "mode": mode,
        "experience_count": len(experiences),
        "test_read_only": bool(test_read_only),
        "forecast_arrays_identical": True,
        "candidate_skills": candidates,
        "mutated_skills": mutations,
        "promoted_skills": promoted_skills,
        "candidate_count": len(candidates),
        "mutation_count": len(mutations),
        "promoted_count": len(promoted_skills),
        "rejected_count": len(eval_pool) - len(promoted_skills),
    }


def _case_score(case: dict, trigger: dict) -> float:
    def norm(value: object) -> str:
        text = str(value or "").lower().replace("_", " ").strip()
        if text in {"increase", "increased", "+", "positive"}:
            return "positive"
        if text in {"decrease", "decreased", "-", "negative"}:
            return "negative"
        return text

    def value_for(row: dict, key: str) -> str:
        aliases = {
            "station_rank_group": ["station_rank_group", "rank_group", "distance_rank_group"],
            "residual_direction": ["residual_direction", "direction"],
        }.get(key, [key])
        for alias in aliases:
            if row.get(alias) is not None:
                return norm(row.get(alias))
        return ""

    score = 0.0
    weights = {
        "event_type": 0.30,
        "day_type": 0.20,
        "station_rank_group": 0.20,
        "residual_direction": 0.20,
    }
    for key, weight in weights.items():
        case_value = value_for(case, key)
        trigger_value = value_for(trigger, key)
        if trigger_value and case_value == trigger_value:
            score += weight
        elif key == "event_type" and trigger_value and case_value:
            trigger_tokens = set(trigger_value.split())
            case_tokens = set(case_value.split())
            if trigger_tokens & case_tokens:
                score += weight * 0.50
    reason = norm(case.get("similarity_reason"))
    if "same" in reason or "similar" in reason:
        score += 0.20
    if case.get("median_lp_moment_correction") is not None or case.get("median_correction") is not None:
        score += 0.10
    if case.get("historical_baseline_residual") is not None or case.get("residual_magnitude") is not None:
        score += 0.05
    score += min(_as_float(case.get("n_eff")) / 10.0, 1.0) * 0.10
    if case.get("iqr") is not None:
        score -= min(_as_float(case.get("iqr")), 0.10)
    return float(min(1.0, max(score, 0.0)))


def apply_residual_memory_skill(cases: Sequence[dict], skill: dict) -> Tuple[List[dict], dict]:
    """Select residual-memory cases for explanation using a promoted skill policy."""
    rows = [dict(case) for case in cases]
    trigger = dict(skill.get("trigger_condition") or {})
    policy = dict(skill.get("memory_selection_policy") or {})
    max_cases = int(policy.get("max_cases") or 3)
    scored = [(_case_score(case, trigger), idx, case) for idx, case in enumerate(rows)]
    baseline = rows[:max_cases]
    selected = [case for _, _, case in sorted(scored, key=lambda item: (-item[0], item[1]))[:max_cases]]
    baseline_scores = [_case_score(case, trigger) for case in baseline]
    selected_scores = [_case_score(case, trigger) for case in selected]
    metrics = {
        "baseline_relevance_mean": float(sum(baseline_scores) / max(len(baseline_scores), 1)),
        "selected_relevance_mean": float(sum(selected_scores) / max(len(selected_scores), 1)),
        "selected_count": len(selected),
        "forecast_array_changed": False,
    }
    return selected, metrics


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_redact(row), ensure_ascii=False, sort_keys=True) + "\n")
