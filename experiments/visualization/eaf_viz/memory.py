"""Residual and skill memory parsers for visualization."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List


def parse_residual_case_text(case_id: str, text: str) -> Dict[str, Any]:
    def match(pattern: str, default=None):
        m = re.search(pattern, text or "", flags=re.I)
        return m.group(1).strip() if m else default

    return {
        "case_id": case_id,
        "retrieved_case_id": match(r"\[residual_case\s+([^\\]]+)\]", ""),
        "historical_event_type": match(r"type=([^ ]+)", "unknown"),
        "day_type": match(r"day=([^ ]+)", "unknown"),
        "station_rank_group": match(r"rank=([^ ]+)", "unknown"),
        "median_correction_pct": _to_float(match(r"correction=([+-]?[0-9.]+)%", "")),
        "p25": _to_float(match(r"p25=([+-]?[0-9.]+)%", "")),
        "median": _to_float(match(r"median=([+-]?[0-9.]+)%", "")),
        "p75": _to_float(match(r"p75=([+-]?[0-9.]+)%", "")),
        "std": _to_float(match(r"std=([+-]?[0-9.]+)%", "")),
        "n": _to_int(match(r"N=([0-9]+)", "")),
        "best_score": _to_float(match(r"best_score=([0-9.]+)", "")),
        "merged_count": _to_int(match(r"merged_count=([0-9]+)", "")),
    }


def residual_memory_rows(case_id: str, evidence: dict) -> List[dict]:
    rows: List[dict] = []
    for idx, item in enumerate(evidence.get("historical_event_cases") or []):
        rows.append(
            {
                "case_id": case_id,
                "retrieved_case_id": f"historical_{idx}",
                "memory_split": item.get("split", "train_val_memory"),
                "historical_event_name": item.get("historical_event_title") or item.get("title"),
                "historical_event_date": item.get("event_time") or item.get("date"),
                "historical_event_type": item.get("event_type"),
                "historical_station": item.get("matched_station") or item.get("channel_name"),
                "station_rank_group": item.get("rank_group"),
                "day_type": item.get("day_type"),
                "direction": item.get("residual_direction"),
                "historical_residual_pct": _to_float(item.get("historical_baseline_residual")),
                "median_correction_pct": _to_float(item.get("median_lp_moment_correction")),
            }
        )
    for item in evidence.get("local_residual_cases") or []:
        rows.append(parse_residual_case_text(case_id, str(item)))
    return rows


def skill_memory_rows(skills: Iterable[dict]) -> List[dict]:
    rows = []
    for skill in skills:
        trigger = skill.get("trigger_condition") or {}
        action = skill.get("explanation_action") or skill.get("calibration_action") or {}
        rows.append(
            {
                "skill_id": skill.get("skill_id"),
                "category": skill.get("skill_category") or skill.get("category"),
                "trigger_event_type": trigger.get("event_type"),
                "trigger_impact_tier": trigger.get("impact_tier"),
                "policy": action.get("policy"),
                "select_cases_by": ",".join(action.get("select_cases_by") or []),
                "promoted_from_split": skill.get("promoted_from_split"),
                "validation_score": skill.get("validation_score"),
                "test_read_only": skill.get("test_read_only"),
                "usage_count": skill.get("usage_count", 0),
                "explanation_coverage_gain": skill.get("explanation_coverage_gain"),
                "relevance_gain": skill.get("relevance_gain"),
                "leakage_free_reasoning_flag": skill.get("leakage_free_reasoning_flag"),
            }
        )
    return rows


def _to_float(value):
    if value in (None, ""):
        return None
    text = str(value).strip().replace("%", "")
    try:
        return float(text)
    except Exception:
        return None


def _to_int(value):
    try:
        return int(value)
    except Exception:
        return None
