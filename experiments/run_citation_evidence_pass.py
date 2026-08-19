#!/usr/bin/env python3
"""Run a citation-quality Evidence-RAG pass over saved representative cases.

This script does not rerun numerical forecasting. It reads saved prediction JSON
files, selects Tier A/B structured events, and optionally calls DashScope native
search through EvidenceResearchAgent when runtime credentials are present.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from agents.evidence_research_agent import EvidenceResearchAgent


REQUIRED_ENV = ("OPENAI_BASE_URL", "OPENAI_API_KEY", "LLM_MODEL")
PUBLIC_EVENT_TERMS = {
    "tsq live": 3.0,
    "plaza programming": 2.2,
    "times square": 1.8,
    "broadway": 1.0,
    "parade": 2.5,
    "festival": 2.0,
    "concert": 2.0,
    "marathon": 2.5,
    "game": 1.5,
    "performance": 1.4,
}
LOW_CITATION_VALUE_TERMS = {
    "cheese": -2.0,
    "promo": -1.5,
    "sampling": -1.5,
    "giveaway": -1.5,
    "brand": -1.0,
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _prediction_files(run_root: Path, max_cases: int) -> List[Path]:
    paths = sorted((run_root / "cases_vllm").glob("*/predictions/*.json"))
    return paths[: max(0, int(max_cases))]


def _citation_event_score(event: Mapping) -> float:
    tier = str(event.get("impact_tier", "") or "").upper()
    score = 10.0 if tier == "A" else 8.0 if tier == "B" else 0.0
    text = " ".join(
        str(event.get(k, "") or "").lower()
        for k in ("title", "location", "venue_name", "event_type", "event_category")
    )
    for term, weight in PUBLIC_EVENT_TERMS.items():
        if term in text:
            score += weight
    for term, weight in LOW_CITATION_VALUE_TERMS.items():
        if term in text:
            score += weight
    if str(event.get("event_category", "") or "").lower() not in {"", "other"}:
        score += 1.0
    if event.get("channel_name") or event.get("station_complex"):
        score += 0.3
    return score


def _high_value_events(prediction: Mapping, max_events: int = 1) -> List[dict]:
    candidates = []
    seen = set()
    for event in ((prediction.get("evidence") or {}).get("structured_events") or []):
        tier = str(event.get("impact_tier", "") or "").upper()
        should_adjust = bool(event.get("should_adjust_forecast"))
        key = "|".join(
            str(event.get(k, "") or "")[:120]
            for k in ("title", "event_time", "location", "venue_name")
        )
        if tier not in {"A", "B"} or not should_adjust or key in seen:
            continue
        seen.add(key)
        item = dict(event)
        item.setdefault("event_score", 1.0 if tier == "A" else 0.85)
        item["citation_event_score"] = round(_citation_event_score(item), 4)
        candidates.append(item)
    candidates.sort(key=lambda row: (-float(row.get("citation_event_score", 0.0)), str(row.get("event_time", ""))))
    return candidates[: max(0, int(max_events))]


def _station_names(prediction: Mapping) -> List[str]:
    numerical = prediction.get("numerical") or {}
    names = numerical.get("channel_names") or prediction.get("channel_names") or []
    if names:
        return [str(x) for x in names]
    events = ((prediction.get("evidence") or {}).get("structured_events") or [])
    out = []
    for event in events:
        name = event.get("channel_name") or event.get("station_complex") or event.get("station_complex_id")
        if name and name not in out:
            out.append(str(name))
    return out


def env_ready(env: Mapping[str, str] | None = None) -> bool:
    env = env or os.environ
    return all(bool(env.get(k)) for k in REQUIRED_ENV)


def run_citation_pass(
    run_root: str | Path,
    max_cases: int = 3,
    max_events_per_case: int = 1,
    qwenplus_max_events: int = 3,
    qwenplus_min_event_score: float = 0.75,
    cache_only: bool = False,
    env: Mapping[str, str] | None = None,
) -> dict:
    run_root = Path(run_root)
    reports = run_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    evidence_root = run_root / "external_evidence_kb" / "citation_qwen_plus_cache"
    env = env or os.environ

    if not env_ready(env):
        summary = {
            "status": "skipped",
            "reason": "missing runtime environment variables for DashScope/Qwen-Plus citation search",
            "required_env_present": {k: bool(env.get(k)) for k in REQUIRED_ENV},
            "accepted_external_evidence_count": 0,
            "api_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "written_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (reports / "citation_evidence_status.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    agent = EvidenceResearchAgent(
        cache_dir=evidence_root,
        api_key=env.get("OPENAI_API_KEY", ""),
        base_url=env.get("OPENAI_BASE_URL", ""),
        model=env.get("LLM_MODEL", "qwen-plus"),
        cache_only=cache_only,
    )
    case_rows = []
    total_stats = {"api_calls": 0, "cache_hits": 0, "cache_misses": 0, "accepted_evidence": 0, "rejected_evidence": 0}
    calls_remaining = max(0, int(qwenplus_max_events))
    for path in _prediction_files(run_root, max_cases):
        prediction = _read_json(path)
        target_date = str((prediction.get("request") or {}).get("date", ""))
        selected_events = _high_value_events(prediction, max_events=max_events_per_case)
        if calls_remaining <= 0:
            selected_events = []
        selected_events = selected_events[:calls_remaining]
        result = agent.research_events(
            selected_events,
            target_date=target_date,
            station_names=_station_names(prediction),
            max_events=min(max_events_per_case, calls_remaining),
            min_score=qwenplus_min_event_score,
        )
        calls_remaining -= int(result.stats.get("api_calls", 0))
        for key in total_stats:
            total_stats[key] += int(result.stats.get(key, 0))
        case_report = {
            "case_prediction": str(path),
            "target_date": target_date,
            "selected_event_count": len(selected_events),
            "stats": result.stats,
            "accepted": result.accepted,
            "rejected": result.rejected,
            "model_assisted_summaries": result.summaries,
        }
        case_rows.append(case_report)

    summary = {
        "status": "completed",
        "run_root": str(run_root),
        "case_count": len(case_rows),
        "max_cases": int(max_cases),
        "max_events_per_case": int(max_events_per_case),
        "qwenplus_max_events": int(qwenplus_max_events),
        "qwenplus_min_event_score": float(qwenplus_min_event_score),
        "cache_only": bool(cache_only),
        "stats": total_stats,
        "accepted_external_evidence_count": total_stats["accepted_evidence"],
        "cases": case_rows,
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (reports / "citation_evidence_status.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", default="autotemp/ccfa_formal_rolling_20260607_115014")
    parser.add_argument("--max_cases", type=int, default=3)
    parser.add_argument("--max_events_per_case", type=int, default=1)
    parser.add_argument("--qwenplus_max_events", type=int, default=3)
    parser.add_argument("--qwenplus_min_event_score", type=float, default=0.75)
    parser.add_argument("--cache_only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_citation_pass(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
