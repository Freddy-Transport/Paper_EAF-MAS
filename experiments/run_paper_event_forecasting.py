#!/usr/bin/env python3
"""Run the paper-grade event-aware explainable forecasting workflow."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.calibration_controller import CalibrationController  # noqa: E402
from agents.event_adapter import apply_adapter_to_forecast, channel_meta_by_name, load_channel_map, load_events_json  # noqa: E402
from agents.forecast_evidence_auditor import ForecastEvidenceAuditor  # noqa: E402
from agents.event_agent import EventAnalysisAgent  # noqa: E402
from agents.event_relevance_agent import EventRelevanceAgent  # noqa: E402
from agents.event_retrieval_agent import EventRetrievalAgent  # noqa: E402
from agents.evidence_research_agent import EvidenceResearchAgent  # noqa: E402
from agents.explanation_evaluator import evaluate_explanation_quality  # noqa: E402
from agents.forecast_explanation_agent import ForecastExplanationAgent  # noqa: E402
from agents.numerical_agent import NumericalPredictionAgent  # noqa: E402
from agents.paper_workflow import (  # noqa: E402
    CalibrationDecisionSpec,
    EventEvidenceSpec,
    ForecastRequestSpec,
    ForecastResultSpec,
    NumericalForecastSpec,
    build_explanation_markdown,
    compute_metrics,
    deduplicate_residual_cases,
    display_station_name,
    ensure_paper_run_dir,
    extract_historical_event_cases,
    split_residual_case_blocks,
    write_forecast_artifacts,
)
from agents.paper_knowledge_base import write_layered_kb_manifests  # noqa: E402
from agents.prediction_fusion import PredictionFusion  # noqa: E402
from agents.rag_pipeline import RAGPipeline  # noqa: E402
from agents.schemas import FinalPrediction  # noqa: E402
from agents.skill_extractor import SkillExtractor  # noqa: E402
from agents.skill_library import SkillLibrary  # noqa: E402
from event_post_training.config import resolve_lp_model_path  # noqa: E402


MODES = {
    "numerical_only",
    "rag_explain",
    "event_adapter_frozen_moment",
    "event_adapter_peft_moment",
    "full_skill_agent",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=sorted(MODES), default="numerical_only")
    p.add_argument("--target_date", required=True)
    p.add_argument("--station_scope", choices=["top128", "event_venue28"], default="event_venue28")
    p.add_argument("--traffic_csv", default="data/nyc_top128_station_hourly_flow.csv")
    p.add_argument("--events_json", default="data/nyc_top128_station_events.json")
    p.add_argument("--channel_map", default="data/nyc_top128_channel_map.json")
    p.add_argument("--venue_station_map", default="data/venue_station_map.json")
    p.add_argument("--fusion_channels", default="data/venue37_fusion_channels.json")
    p.add_argument("--lp_model_path", default="experiments/outputs/lp_top128")
    p.add_argument("--event_adapter_path", default="experiments/outputs/event_adapter_formal_frozen")
    p.add_argument("--event_adapter_peft_path", default="experiments/outputs/event_adapter_formal_peft")
    p.add_argument("--knowledge_base_dir", default="agents/knowledge_base_residual")
    p.add_argument("--output_root", default=None)
    p.add_argument("--llm_base_url", default="http://localhost:8000/v1")
    p.add_argument("--llm_model", default="Qwen/Qwen3-8B")
    p.add_argument("--llm_api_key", default="EMPTY")
    p.add_argument("--device", default="auto")
    p.add_argument("--horizon", type=int, default=192)
    p.add_argument("--enable_online_retrieval", action="store_true")
    p.add_argument("--enable_qwenplus_search", action="store_true")
    p.add_argument("--qwenplus_max_events", type=int, default=3)
    p.add_argument("--qwenplus_min_event_score", type=float, default=0.75)
    p.add_argument("--qwenplus_cache_only", action="store_true")
    p.add_argument("--qwenplus_force_refresh", action="store_true")
    p.add_argument("--require_qwenplus_live", action="store_true")
    p.add_argument("--kb_root", default=None)
    p.add_argument("--max_retrieval_events", type=int, default=8)
    p.add_argument("--max_analysis_events", type=int, default=8)
    p.add_argument("--max_explanation_events", type=int, default=12)
    p.add_argument("--allow_missing_adapter_passthrough", action="store_true")
    p.add_argument("--update_skills", action="store_true")
    p.add_argument("--residual_memory_skill_library", default=None)
    p.add_argument("--include_focused_prediction", action="store_true")
    p.add_argument("--focus_center", default="")
    p.add_argument("--focus_start", default="")
    p.add_argument("--focus_end", default="")
    p.add_argument("--focus_hours_before", type=int, default=4)
    p.add_argument("--focus_hours_after", type=int, default=4)
    p.add_argument("--focus_channel", action="append", default=[])
    p.add_argument("--focus_station_query", action="append", default=[])
    p.add_argument("--interactive_focus", action="store_true")
    return p.parse_args()


def validate_qwenplus_live_policy(args: argparse.Namespace) -> None:
    """Fail early for formal evidence runs that require a paid/live Qwen-Plus call."""
    if not getattr(args, "require_qwenplus_live", False):
        return
    if not getattr(args, "enable_qwenplus_search", False):
        raise RuntimeError("--require_qwenplus_live requires --enable_qwenplus_search")
    if getattr(args, "qwenplus_cache_only", False):
        raise RuntimeError("--require_qwenplus_live cannot be combined with --qwenplus_cache_only")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for --require_qwenplus_live")


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_channel_names(traffic_csv: Path) -> List[str]:
    cols = pd.read_csv(traffic_csv, nrows=1).columns.tolist()
    return [c for c in cols if c != "date"]


def load_scope_indices(scope: str, channel_names: Sequence[str], fusion_channels: Path) -> Tuple[List[int], dict]:
    if scope == "top128":
        return list(range(len(channel_names))), {"scope": "top128", "n": len(channel_names)}
    payload = json.loads(fusion_channels.read_text(encoding="utf-8"))
    requested = list(payload.get("channel_names", []))
    name_to_idx = {name: i for i, name in enumerate(channel_names)}
    names = [name for name in requested if name in name_to_idx]
    return [name_to_idx[name] for name in names], {
        "scope": "event_venue28",
        "requested_venue_total": int(payload.get("venue37_total", len(requested))),
        "n": len(names),
        "missing_from_top128": payload.get("missing_from_top_n", []),
    }


def subset_prediction(pred, indices: Sequence[int]):
    pred.forecast = [pred.forecast[i] for i in indices]
    pred.channel_names = [pred.channel_names[i] for i in indices]
    if pred.ground_truth:
        pred.ground_truth = [pred.ground_truth[i] for i in indices]
    return pred


def filter_events(events, prediction, channel_map_path: Path):
    relevance = EventRelevanceAgent(channel_map_path=channel_map_path, relevance_threshold=0.45, max_events=80)
    filtered, rows = relevance.filter_events(
        events,
        forecast_timestamps=prediction.forecast_timestamps,
        target_channels=prediction.channel_names,
    )
    has_major = any(getattr(ev, "impact_tier", "C") in ("A", "B") for ev in filtered)
    return filtered, rows, has_major


def _confidence_score(value) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except Exception:
        return {"high": 1.0, "medium": 0.6, "low": 0.25}.get(str(value).lower(), 0.0)


def rank_events_for_reasoning(events) -> List:
    tier_score = {"A": 4, "B": 3, "C": 2, "D": 1}

    def key(ev):
        tier = str(getattr(ev, "impact_tier", "C") or "C").upper()
        confidence = _confidence_score(getattr(ev, "confidence", None))
        title = str(getattr(ev, "title", "") or "")
        location = str(getattr(ev, "location", "") or "")
        return (
            tier_score.get(tier, 0),
            confidence,
            1 if title else 0,
            1 if location else 0,
            title,
        )

    return sorted(events, key=key, reverse=True)


def _parse_dt(value: str):
    if not value:
        return None
    text = str(value)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[:19], fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def event_within_forecast_window(event, target_date: str, horizon: int) -> bool:
    anchor = _parse_dt(target_date)
    when = _parse_dt(str(getattr(event, "event_time", "") or ""))
    if anchor is None or when is None:
        return True
    return anchor <= when < anchor + timedelta(hours=int(horizon))


def _event_value(event: Any, key: str, default: Any = "") -> Any:
    if isinstance(event, dict):
        return event.get(key, default)
    return getattr(event, key, default)


def filter_events_for_focus_window(events: Sequence[Any], focus_window: dict) -> List[Any]:
    """Keep only events whose scheduled time falls inside the user-facing viewport."""
    start = _parse_dt(str(focus_window.get("start", "") or ""))
    end = _parse_dt(str(focus_window.get("end", "") or ""))
    if start is None or end is None:
        return list(events)
    out: List[Any] = []
    for event in events:
        when = _parse_dt(str(_event_value(event, "event_time") or _event_value(event, "start_time") or _event_value(event, "date") or ""))
        if when is not None and start <= when <= end:
            out.append(event)
    return out


def split_evidence_for_forecast_time(rows: Sequence[dict]) -> Tuple[List[dict], List[dict]]:
    accepted: List[dict] = []
    rejected: List[dict] = []
    for row in rows:
        item = dict(row)
        if item.get("accepted") is True and item.get("url") and item.get("source_time_status") == "known_before_anchor":
            accepted.append(item)
        else:
            if item.get("accepted") is True and item.get("url") and item.get("source_time_status") != "known_before_anchor":
                item["accepted"] = False
                reason = item.get("rejected_reason") or ""
                extra = "source_time_unknown" if not item.get("source_time_status") else f"source_time_{item.get('source_time_status')}"
                item["rejected_reason"] = "; ".join(x for x in [reason, extra] if x)
            rejected.append(item)
    return accepted, rejected


def _metric_block(actual: np.ndarray, raw: np.ndarray, adjusted: np.ndarray, channel_mask=None, hour_mask=None) -> dict:
    if actual.size == 0 or raw.size == 0 or adjusted.size == 0:
        return {"status": "not_applicable_missing_ground_truth"}
    c = min(actual.shape[0], raw.shape[0], adjusted.shape[0])
    h = min(actual.shape[1], raw.shape[1], adjusted.shape[1])
    if c == 0 or h == 0:
        return {"status": "not_applicable_empty"}
    actual = actual[:c, :h]
    raw = raw[:c, :h]
    adjusted = adjusted[:c, :h]
    if channel_mask is None:
        channel_mask = np.ones(c, dtype=bool)
    else:
        channel_mask = np.asarray(channel_mask, dtype=bool)[:c]
    if hour_mask is None:
        hour_mask = np.ones(h, dtype=bool)
    else:
        hour_mask = np.asarray(hour_mask, dtype=bool)[:h]
    if not bool(channel_mask.any()):
        return {"status": "not_applicable_no_channels"}
    if not bool(hour_mask.any()):
        return {"status": "not_applicable_no_hours"}
    a = actual[np.ix_(channel_mask, hour_mask)]
    r = raw[np.ix_(channel_mask, hour_mask)]
    z = adjusted[np.ix_(channel_mask, hour_mask)]
    return {
        "status": "ok",
        "channel_count": int(channel_mask.sum()),
        "hour_count": int(hour_mask.sum()),
        "cell_count": int(channel_mask.sum() * hour_mask.sum()),
        "raw": compute_metrics(a, r),
        "adjusted": compute_metrics(a, z),
        "delta_raw_minus_adjusted": {
            "wape": float(compute_metrics(a, r).get("wape", 0.0) - compute_metrics(a, z).get("wape", 0.0)),
            "mae": float(compute_metrics(a, r).get("mae", 0.0) - compute_metrics(a, z).get("mae", 0.0)),
        },
    }


def _event_window_mask(timestamps: Sequence[str], structured_events: Sequence[dict], pre_hours: int = 3, post_hours: int = 6) -> np.ndarray:
    parsed_ts = [_parse_dt(str(ts)) for ts in timestamps]
    mask = np.zeros(len(parsed_ts), dtype=bool)
    for event in structured_events:
        when = _parse_dt(str(event.get("event_time") or event.get("start_time") or event.get("date") or ""))
        if when is None:
            continue
        start = when - timedelta(hours=int(pre_hours))
        end = when + timedelta(hours=int(post_hours))
        for i, ts in enumerate(parsed_ts):
            if ts is not None and start <= ts <= end:
                mask[i] = True
    return mask


def _channel_mask_from_names(channel_names: Sequence[str], selected_names: Sequence[str]) -> np.ndarray:
    wanted = {str(x) for x in selected_names if x}
    return np.asarray([name in wanted for name in channel_names], dtype=bool)


def compute_local_forecast_metrics(
    ground_truth,
    raw_forecast,
    adjusted_forecast,
    channel_names: Sequence[str],
    timestamps: Sequence[str],
    structured_events: Sequence[dict],
    adjusted_channels: Sequence[str],
    included_units: Sequence[dict],
    controller_allowed: bool,
) -> dict:
    if not ground_truth:
        return {
            "event_window_metrics": {"status": "not_applicable_missing_ground_truth"},
            "adjusted_channel_metrics": {"status": "not_applicable_missing_ground_truth"},
            "included_unit_metrics": {"status": "not_applicable_missing_ground_truth"},
        }
    actual = np.asarray(ground_truth, dtype=np.float32)
    raw = np.asarray(raw_forecast, dtype=np.float32)
    adjusted = np.asarray(adjusted_forecast, dtype=np.float32)

    event_mask = _event_window_mask(timestamps, structured_events)
    event_window_metrics = _metric_block(actual, raw, adjusted, hour_mask=event_mask)
    if event_window_metrics.get("status") != "ok":
        event_window_metrics["status"] = "not_applicable_no_event_window"

    if not controller_allowed:
        adjusted_channel_metrics = _metric_block(actual, raw, raw)
        adjusted_channel_metrics["status"] = "not_applicable_abstained"
    else:
        ch_mask = _channel_mask_from_names(channel_names, adjusted_channels)
        adjusted_channel_metrics = _metric_block(actual, raw, adjusted, channel_mask=ch_mask)
        if adjusted_channel_metrics.get("status") != "ok":
            adjusted_channel_metrics["status"] = "not_applicable_no_adjusted_channels"

    included_names = []
    for row in included_units:
        name = row.get("station_channel") or row.get("channel") or row.get("station")
        if name:
            included_names.append(str(name))
    included_mask = _channel_mask_from_names(channel_names, included_names)
    included_unit_metrics = _metric_block(actual, raw, adjusted, channel_mask=included_mask)
    if included_unit_metrics.get("status") != "ok":
        included_unit_metrics["status"] = "not_applicable_no_included_units"
    return {
        "event_window_metrics": event_window_metrics,
        "adjusted_channel_metrics": adjusted_channel_metrics,
        "included_unit_metrics": included_unit_metrics,
    }


def _norm_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _load_channel_records(channel_map_path: Path, scoped_channel_names: Sequence[str]) -> Dict[str, dict]:
    payload = json.loads(channel_map_path.read_text(encoding="utf-8")) if channel_map_path.is_file() else {}
    rows = payload.get("channels", []) if isinstance(payload, dict) else payload
    by_name = {str(row.get("channel_name")): dict(row) for row in rows or [] if row.get("channel_name")}
    out = {}
    for rank, name in enumerate(scoped_channel_names, 1):
        row = dict(by_name.get(name, {}))
        row.setdefault("channel_name", name)
        row.setdefault("station_complex_id", name.split("__", 1)[0] if "__" in name else name)
        row.setdefault("station_complex", name.split("__", 1)[-1] if "__" in name else name)
        row.setdefault("station_name_clean", row.get("station_complex"))
        row.setdefault("rank", rank)
        out[name] = row
    return out


def _load_venue_station_map(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("venues", []) if isinstance(payload, dict) else payload)


def resolve_focus_channels(
    scoped_channel_names: Sequence[str],
    channel_map_path: Path,
    venue_station_map_path: Path,
    focus_channels: Sequence[str],
    station_queries: Sequence[str],
    fallback_channels: Sequence[str],
    max_channels: int = 6,
) -> List[dict]:
    records = _load_channel_records(channel_map_path, scoped_channel_names)
    venues = _load_venue_station_map(venue_station_map_path)
    selected: Dict[str, dict] = {}

    def add(name: str, matched_by: str, query: str, reason: str) -> None:
        if name not in records or name in selected:
            return
        row = dict(records[name])
        row.update({"matched_by": matched_by, "query": query, "event_relevance_reason": reason})
        selected[name] = row

    lower_to_name = {name.lower(): name for name in scoped_channel_names}
    id_to_names: Dict[str, List[str]] = {}
    for name, row in records.items():
        sid = str(row.get("station_complex_id") or name.split("__", 1)[0])
        id_to_names.setdefault(sid.lower(), []).append(name)

    for query in focus_channels or []:
        q = str(query or "").strip()
        if not q:
            continue
        if q in records:
            add(q, "exact_channel", q, "explicit channel selected by user")
            continue
        if q.lower() in lower_to_name:
            add(lower_to_name[q.lower()], "exact_channel", q, "explicit channel selected by user")
            continue
        for name in id_to_names.get(q.lower(), []):
            add(name, "station_complex_id", q, "station complex id selected by user")

    for query in station_queries or []:
        q_norm = _norm_text(query)
        if not q_norm:
            continue
        matched_venue_ids: List[str] = []
        for venue in venues:
            aliases = [_norm_text(x) for x in venue.get("aliases", [])]
            if any(q_norm == alias or q_norm in alias or alias in q_norm for alias in aliases if alias):
                matched_venue_ids.extend(str(x).lower() for x in venue.get("station_complex_ids", []))
        for sid in matched_venue_ids:
            for name in id_to_names.get(sid, []):
                add(name, "venue_alias", query, f"venue query maps to station complex {sid.upper()}")
        scored = []
        for name, row in records.items():
            haystack = _norm_text(" ".join(str(row.get(k, "")) for k in ["channel_name", "station_complex_id", "station_complex", "station_name_clean"]))
            q_tokens = set(q_norm.split())
            h_tokens = set(haystack.split())
            overlap = len(q_tokens & h_tokens)
            if q_norm in haystack or overlap >= max(1, min(2, len(q_tokens))):
                scored.append((overlap, int(row.get("rank") or 999), name))
        for _, _, name in sorted(scored, key=lambda item: (-item[0], item[1]))[:max_channels]:
            add(name, "station_query", query, "station query matches station/channel metadata")

    for name in fallback_channels or []:
        add(str(name), "fallback", str(name), "fallback from auditor included units or adjusted channels")
    if not selected:
        for name in list(scoped_channel_names)[:max_channels]:
            add(name, "top_scope_fallback", "", "fallback top channel in station scope")
    rows = sorted(selected.values(), key=lambda row: int(row.get("rank") or 999))
    return rows[:max_channels]


def resolve_focus_time_window(
    timestamps: Sequence[str],
    focus_start: str,
    focus_end: str,
    focus_center: str,
    hours_before: int,
    hours_after: int,
    structured_events: Sequence[dict],
    fallback_anchor: str,
) -> dict:
    parsed = [_parse_dt(str(ts)) for ts in timestamps]
    valid = [ts for ts in parsed if ts is not None]
    min_ts = min(valid) if valid else _parse_dt(fallback_anchor)
    max_ts = max(valid) if valid else _parse_dt(fallback_anchor)
    start = _parse_dt(focus_start)
    end = _parse_dt(focus_end)
    center = _parse_dt(focus_center)
    source = "explicit_start_end" if start and end else "explicit_center"
    if not center and not (start and end):
        for event in structured_events:
            center = _parse_dt(str(event.get("event_time") or event.get("start_time") or event.get("date") or ""))
            if center:
                source = "structured_event_time"
                break
    if not center and not (start and end):
        center = _parse_dt(fallback_anchor) or min_ts
        source = "anchor_fallback"
    if not (start and end):
        start = center - timedelta(hours=int(hours_before))
        end = center + timedelta(hours=int(hours_after))
    if min_ts and start < min_ts:
        start = min_ts
    if max_ts and end > max_ts:
        end = max_ts
    indices = [i for i, ts in enumerate(parsed) if ts is not None and start <= ts <= end]
    return {
        "start": start.isoformat(sep=" ") if start else "",
        "end": end.isoformat(sep=" ") if end else "",
        "center": center.isoformat(sep=" ") if center else "",
        "source": source,
        "hours_before": int(hours_before),
        "hours_after": int(hours_after),
        "horizon_indices": indices,
        "status": "ok" if indices else "not_applicable_no_timestamps",
    }


def build_focused_prediction(
    raw_forecast,
    adjusted_forecast,
    ground_truth,
    channel_names: Sequence[str],
    timestamps: Sequence[str],
    resolved_channels: Sequence[dict],
    focus_window: dict,
    controller_allowed: bool,
) -> dict:
    raw = np.asarray(raw_forecast, dtype=np.float32)
    adjusted = np.asarray(adjusted_forecast, dtype=np.float32)
    actual = np.asarray(ground_truth, dtype=np.float32) if ground_truth else None
    name_to_idx = {name: i for i, name in enumerate(channel_names)}
    channel_indices = [name_to_idx[row["channel_name"]] for row in resolved_channels if row.get("channel_name") in name_to_idx]
    hour_indices = [int(i) for i in focus_window.get("horizon_indices", [])]
    rows: List[dict] = []
    reason_by_name = {row.get("channel_name"): row.get("event_relevance_reason", row.get("matched_by", "")) for row in resolved_channels}
    meta_by_name = {row.get("channel_name"): row for row in resolved_channels}
    for ci in channel_indices:
        for hi in hour_indices:
            if ci >= raw.shape[0] or hi >= raw.shape[1]:
                continue
            row = {
                "timestamp": timestamps[hi] if hi < len(timestamps) else str(hi),
                "horizon_idx": hi,
                "channel_name": channel_names[ci],
                "display_station_name": display_station_name(channel_names[ci], meta_by_name.get(channel_names[ci], {})),
                "raw_forecast": float(raw[ci, hi]),
                "adjusted_forecast": float(adjusted[ci, hi]),
                "correction": float(adjusted[ci, hi] - raw[ci, hi]),
                "controller_allowed": bool(controller_allowed),
                "event_relevance_reason": reason_by_name.get(channel_names[ci], ""),
            }
            if actual is not None and ci < actual.shape[0] and hi < actual.shape[1]:
                row["actual"] = float(actual[ci, hi])
                row["raw_error"] = float(abs(actual[ci, hi] - raw[ci, hi]))
                row["adjusted_error"] = float(abs(actual[ci, hi] - adjusted[ci, hi]))
            rows.append(row)
    metrics = {"status": "not_available"}
    if actual is not None and channel_indices and hour_indices:
        c_mask = np.zeros(raw.shape[0], dtype=bool)
        h_mask = np.zeros(raw.shape[1], dtype=bool)
        c_mask[channel_indices] = True
        h_mask[hour_indices] = True
        metrics = _metric_block(actual, raw, adjusted, channel_mask=c_mask, hour_mask=h_mask)
    return {
        "focused_request": {"controller_allowed": bool(controller_allowed)},
        "focused_resolved_channels": list(resolved_channels),
        "focused_time_window": dict(focus_window),
        "focused_forecast_rows": rows,
        "focused_forecast_metrics": metrics,
    }


def focused_context_for_llm(focused_prediction: dict) -> dict:
    if not focused_prediction:
        return {}
    all_rows = list(focused_prediction.get("focused_forecast_rows") or [])
    by_channel: dict[str, List[dict]] = {}
    for row in all_rows:
        by_channel.setdefault(str(row.get("channel_name", "")), []).append(row)
    resolved = list(focused_prediction.get("focused_resolved_channels") or [])
    label_by_channel = {
        str(row.get("channel_name")): display_station_name(row.get("channel_name"), row, max_len=80)
        for row in resolved
    }
    sampled_rows: List[dict] = []
    # Keep the prompt compact: full focused rows are saved as CSV/JSON, while
    # the LLM only needs representative station-hour values for reasoning.
    for channel_name in list(by_channel.keys())[:1]:
        channel_rows = by_channel[channel_name]
        if len(channel_rows) <= 3:
            sampled_rows.extend(channel_rows)
            continue
        mid = len(channel_rows) // 2
        keep_positions = sorted({0, mid, len(channel_rows) - 1})
        sampled_rows.extend(channel_rows[i] for i in keep_positions if 0 <= i < len(channel_rows))
    rows = []
    for row in sampled_rows[:3]:
        channel_name = str(row.get("channel_name") or "")
        rows.append(
            {
                "timestamp": row.get("timestamp"),
                "station_label": label_by_channel.get(channel_name, display_station_name(channel_name, max_len=80)),
                "raw_forecast": round(float(row.get("raw_forecast", 0.0) or 0.0), 2),
                "adjusted_forecast": round(float(row.get("adjusted_forecast", 0.0) or 0.0), 2),
                "correction": round(float(row.get("correction", 0.0) or 0.0), 3),
                "controller_allowed": row.get("controller_allowed"),
                "event_relevance_reason": str(row.get("event_relevance_reason") or "")[:120],
            }
        )
    corrections = [float(row.get("correction", 0.0) or 0.0) for row in all_rows]
    raw_values = [float(row.get("raw_forecast", 0.0) or 0.0) for row in all_rows]
    adjusted_values = [float(row.get("adjusted_forecast", 0.0) or 0.0) for row in all_rows]
    forecast_time_summary = {
        "row_count": len(all_rows),
        "channel_count": len(by_channel),
        "sampled_row_count": len(rows),
        "raw_forecast_min": round(min(raw_values), 2) if raw_values else None,
        "raw_forecast_max": round(max(raw_values), 2) if raw_values else None,
        "adjusted_forecast_min": round(min(adjusted_values), 2) if adjusted_values else None,
        "adjusted_forecast_max": round(max(adjusted_values), 2) if adjusted_values else None,
        "max_abs_correction": round(max((abs(x) for x in corrections), default=0.0), 3),
        "nonzero_correction_rows": sum(1 for x in corrections if abs(x) > 1e-6),
    }
    return {
        "focused_time_window": focused_prediction.get("focused_time_window") or {},
        "focused_resolved_channels": [
            {
                "station_label": display_station_name(row.get("channel_name"), row, max_len=80),
                "matched_by": row.get("matched_by"),
                "event_relevance_reason": str(row.get("event_relevance_reason") or "")[:120],
            }
            for row in resolved[:2]
        ],
        "forecast_time_summary": forecast_time_summary,
        "full_rows_saved_to_artifacts": True,
        "forecast_time_rows": rows,
    }


def maybe_prompt_focus_args(args: argparse.Namespace) -> None:
    if not args.interactive_focus:
        return
    if not (args.focus_center or (args.focus_start and args.focus_end)):
        args.focus_center = input("Focus center time (YYYY-MM-DD HH:MM:SS, blank=event/anchor default): ").strip()
    if not (args.focus_channel or args.focus_station_query):
        text = input("Focus station/channel query, comma-separated (blank=auto): ").strip()
        if text:
            args.focus_station_query = [x.strip() for x in text.split(",") if x.strip()]


def load_residual_memory_skills(path: str | None, max_skills: int = 5) -> List[dict]:
    if not path:
        return []
    p = resolve_path(path)
    if not p.is_file():
        return []
    skills: List[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            skill = json.loads(line)
        except json.JSONDecodeError:
            continue
        if skill.get("skill_category") == "residual_memory_skill" and skill.get("promoted") is True:
            skills.append(skill)
        if len(skills) >= max_skills:
            break
    return skills


def analyze_events(events, args, rag_cutoff_date: str):
    rag = None
    kb_dir = resolve_path(args.knowledge_base_dir)
    if kb_dir.exists():
        rag = RAGPipeline(knowledge_base_dir=str(kb_dir))
    agent = EventAnalysisAgent(
        api_key=args.llm_api_key,
        base_url=args.llm_base_url,
        model=args.llm_model,
        rag_pipeline=rag,
    )
    return agent.analyze(events, _progress=True, rag_cutoff_date=rag_cutoff_date)


def build_rag_residual_context(events, args, cutoff_date: str) -> str:
    kb_dir = resolve_path(args.knowledge_base_dir)
    if not kb_dir.exists() or not events:
        return ""
    try:
        rag = RAGPipeline(knowledge_base_dir=str(kb_dir))
        chunks = []
        for event in events[:4]:
            ctx = rag.retrieve_context(event, cutoff_date=cutoff_date)
            if ctx:
                chunks.append(ctx[:1200])
        return "\n\n".join(chunks)
    except Exception as exc:
        return f"RAG residual context unavailable: {type(exc).__name__}: {exc}"


def adapter_adjust(mode: str, args, prediction, events, device: str):
    adapter_dir = resolve_path(args.event_adapter_path if mode == "event_adapter_frozen_moment" else args.event_adapter_peft_path)
    raw = np.asarray(prediction.forecast, dtype=np.float32)
    if not (adapter_dir / "event_adapter.pt").is_file():
        if args.allow_missing_adapter_passthrough:
            return prediction.forecast, [], f"Adapter checkpoint missing at {adapter_dir}; passthrough enabled.", np.zeros_like(raw), str(adapter_dir)
        raise FileNotFoundError(f"Adapter checkpoint missing: {adapter_dir / 'event_adapter.pt'}")
    meta = channel_meta_by_name(load_channel_map(resolve_path(args.channel_map)))
    adjusted, correction = apply_adapter_to_forecast(
        prediction.forecast,
        prediction.forecast_timestamps,
        prediction.channel_names,
        events,
        meta,
        adapter_dir,
        device=device,
    )
    changed = np.abs(correction).max(axis=1) > 1e-6
    channels = [prediction.channel_names[i] for i, yes in enumerate(changed) if yes]
    reason = f"Residual adapter proposed correction from {adapter_dir}; max correction={float(np.abs(correction).max()):.4f}."
    return adjusted.tolist(), channels, reason, correction.astype(np.float32), str(adapter_dir)


def write_preference_record(run_dir: Path, mode: str, metrics_raw: dict, metrics_adjusted: dict, explanation: str) -> None:
    if not metrics_raw or not metrics_adjusted:
        return
    delta = metrics_raw.get("wape", 0.0) - metrics_adjusted.get("wape", 0.0)
    row = {
        "mode": mode,
        "metric": "wape",
        "metric_delta_raw_minus_adjusted": delta,
        "auto_preference": "adjusted" if delta > 0 else "raw",
        "explanation_preview": explanation[:500],
    }
    path = run_dir / "preferences" / "preference_records.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def maybe_update_skill_library(run_dir: Path, final_prediction: FinalPrediction, channel_map_path: Path) -> None:
    library = SkillLibrary(str(run_dir / "models" / "skill_library.jsonl"))
    extractor = SkillExtractor(library, channel_map_path=str(channel_map_path))
    trace_id = f"paper_{int(time.time())}"
    skills = extractor.extract(final_prediction, trace_id=trace_id)
    (run_dir / "reports" / "skill_updates.json").write_text(
        json.dumps([s.model_dump() for s in skills], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _scan_secret_leaks(run_dir: Path) -> List[str]:
    leaks: List[str] = []
    secret_pattern = re.compile(r"sk-[A-Za-z0-9_-]{12,}")
    for path in run_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() in {".png", ".pt", ".npz"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "OPENAI_API_KEY" in text or secret_pattern.search(text):
            leaks.append(str(path.relative_to(run_dir)))
    return leaks


def _residual_duplicate_count(explanation: str) -> int:
    keys = []
    in_section = False
    for line in explanation.splitlines():
        if line.startswith("### Local Residual RAG Cases") or line.startswith("### Residual Pattern Evidence"):
            in_section = True
            continue
        if in_section and line.startswith("### "):
            break
        if "[residual_case " in line:
            match = re.search(r"type=([^ ](?:.*?)) day=([^ ]+) rank=([^ ]+) correction=([^ ]+)", line)
            if match:
                keys.append("|".join(match.groups()))
    return len(keys) - len(set(keys))


def write_self_check(
    run_dir: Path,
    result: ForecastResultSpec,
    raw_metrics: dict,
    adjusted_metrics: dict,
    explanation: str,
    evidence: EventEvidenceSpec,
    explanation_quality: dict | None = None,
    model_assisted_summaries: list[dict] | None = None,
    qwenplus_stats: dict | None = None,
) -> dict:
    checks = []

    def add(name: str, ok: bool, level: str = "fail", detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "level": level, "detail": detail})

    for artifact_name, artifact_path in result.artifacts.items():
        path = Path(artifact_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        add(f"artifact_exists:{artifact_name}", path.is_file(), detail=str(path))

    fig_path = Path(result.artifacts.get("hourly_curve_png", ""))
    if not fig_path.is_absolute():
        fig_path = PROJECT_ROOT / fig_path
    meta_path = fig_path.with_suffix(".meta.json")
    meta = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    add("figure_meta_exists", meta_path.is_file(), detail=str(meta_path))
    if meta.get("raw_adjusted_overlap"):
        add("raw_visibility_note", "raw == adjusted" in str(meta.get("raw_adjusted_note", "")), detail=str(meta.get("raw_adjusted_note", "")))
    else:
        add("raw_visibility_note", bool(meta), detail="raw and adjusted differ or meta available")

    dup_count = _residual_duplicate_count(explanation)
    add("residual_cases_deduplicated", dup_count == 0, detail=f"duplicate_count={dup_count}")
    quality = explanation_quality or {}
    add(
        "explanation_calibration_consistent",
        bool(quality.get("calibration_decision_consistent", True)),
        detail=str(quality.get("calibration_decision_consistent", "")),
    )
    add(
        "explanation_station_event_linking",
        bool(quality.get("station_event_linking", False)) or not evidence.has_major_event,
        level="warn",
        detail=f"station_event_linking={quality.get('station_event_linking')}",
    )
    add(
        "explanation_multi_hop_reasoning",
        bool(quality.get("multi_hop_reasoning", False)) or not evidence.has_major_event,
        level="warn",
        detail=f"multi_hop_reasoning={quality.get('multi_hop_reasoning')}",
    )
    add("no_secret_leak", not _scan_secret_leaks(run_dir), detail="scanned text artifacts")
    if result.decision.abstain:
        add(
            "abstain_explanation_consistent",
            "No numerical event calibration was applied" in explanation,
            detail="abstain requires explicit no-calibration explanation",
        )
    qwen_stats = qwenplus_stats or {}
    selected_events = int(qwen_stats.get("selected_events") or 0)
    summary_count = len(model_assisted_summaries or [])
    if selected_events > 0:
        add(
            "model_assisted_summary_present",
            summary_count > 0,
            level="fail",
            detail=f"selected_events={selected_events}, model_assisted_summaries={summary_count}",
        )
    raw_wape = float(raw_metrics.get("wape", 0.0) or 0.0)
    adjusted_wape = float(adjusted_metrics.get("wape", 0.0) or 0.0)
    allowed = raw_wape * 1.02 if raw_wape else adjusted_wape
    add(
        "overall_wape_not_degraded_gt_2pct",
        adjusted_wape <= allowed + 1e-9,
        detail=f"raw={raw_wape:.6f}, adjusted={adjusted_wape:.6f}, allowed={allowed:.6f}",
    )

    failed = [row for row in checks if not row["ok"] and row["level"] == "fail"]
    warnings = [row for row in checks if not row["ok"] and row["level"] == "warn"]
    payload = {
        "status": "fail" if failed else "pass",
        "failed_checks": failed,
        "warnings": warnings,
        "checks": checks,
        "explanation_quality": quality,
    }
    (run_dir / "reports" / "self_check.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Experiment Self Check", "", f"- status: `{payload['status']}`", f"- warnings: `{len(warnings)}`", ""]
    for row in checks:
        mark = "PASS" if row["ok"] else row["level"].upper()
        lines.append(f"- {mark}: `{row['name']}` {row.get('detail', '')}")
    (run_dir / "reports" / "self_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    maybe_prompt_focus_args(args)
    validate_qwenplus_live_policy(args)
    device = "cuda:0" if args.device == "auto" else args.device
    if args.device == "auto":
        try:
            import torch
            if not torch.cuda.is_available():
                device = "cpu"
        except Exception:
            device = "cpu"

    run_dir = ensure_paper_run_dir(
        args.output_root
        or PROJECT_ROOT / "autotemp" / f"paper_event_forecasting_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    traffic_csv = resolve_path(args.traffic_csv)
    events_json = resolve_path(args.events_json)
    channel_map = resolve_path(args.channel_map)
    venue_station_map = resolve_path(args.venue_station_map)
    fusion_channels = resolve_path(args.fusion_channels)
    channel_names = load_channel_names(traffic_csv)
    scope_indices, scope_meta = load_scope_indices(args.station_scope, channel_names, fusion_channels)
    resolved_lp, lp_source = resolve_lp_model_path(resolve_path(args.lp_model_path), PROJECT_ROOT)

    manifest = {
        "mode": args.mode,
        "target_date": args.target_date,
        "station_scope": args.station_scope,
        "scope_meta": scope_meta,
        "llm_base_url": args.llm_base_url,
        "llm_model": args.llm_model,
        "resolved_lp_model_path": str(resolved_lp),
        "lp_source": lp_source,
        "device": device,
        "max_retrieval_events": args.max_retrieval_events,
        "max_analysis_events": args.max_analysis_events,
        "max_explanation_events": args.max_explanation_events,
        "enable_qwenplus_search": bool(args.enable_qwenplus_search),
        "qwenplus_max_events": int(args.qwenplus_max_events),
        "qwenplus_min_event_score": float(args.qwenplus_min_event_score),
        "qwenplus_cache_only": bool(args.qwenplus_cache_only),
        "qwenplus_force_refresh": bool(args.qwenplus_force_refresh),
        "require_qwenplus_live": bool(args.require_qwenplus_live),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    kb_root = resolve_path(args.kb_root) if args.kb_root else run_dir / "knowledge"
    kb_manifests = write_layered_kb_manifests(kb_root, uses_qwenplus=args.enable_qwenplus_search)

    agent = NumericalPredictionAgent(
        model_path=str(PROJECT_ROOT / "__paper_no_gca__"),
        device=device,
        forecast_horizon=args.horizon,
        n_channels=len(channel_names),
        channel_names=channel_names,
        lp_model_path=str(resolved_lp),
        lp_data_path=str(traffic_csv),
        allow_lp_training_fallback=False,
    )
    prediction = agent.predict_for_date(str(traffic_csv), args.target_date, forecast_horizon=args.horizon)
    prediction = subset_prediction(prediction, scope_indices)
    events = load_events_json(events_json) if events_json.is_file() else []
    filtered_events, relevance_rows, has_major = filter_events(events, prediction, channel_map)
    ranked_events = rank_events_for_reasoning(filtered_events)
    horizon_events = [ev for ev in ranked_events if event_within_forecast_window(ev, args.target_date, args.horizon)]
    focus_requested = bool(
        args.include_focused_prediction
        or args.focus_center
        or args.focus_start
        or args.focus_end
        or args.focus_channel
        or args.focus_station_query
        or args.interactive_focus
    )
    preliminary_focus_window: Dict[str, Any] = {}
    preliminary_focus_channels: List[dict] = []
    focus_aligned_events = list(horizon_events)
    if focus_requested:
        preliminary_focus_window = resolve_focus_time_window(
            prediction.forecast_timestamps,
            focus_start=args.focus_start,
            focus_end=args.focus_end,
            focus_center=args.focus_center,
            hours_before=args.focus_hours_before,
            hours_after=args.focus_hours_after,
            structured_events=[ev.model_dump() for ev in horizon_events],
            fallback_anchor=args.target_date,
        )
        focus_aligned_events = filter_events_for_focus_window(horizon_events, preliminary_focus_window)
        preliminary_focus_channels = resolve_focus_channels(
            prediction.channel_names,
            channel_map,
            venue_station_map,
            focus_channels=args.focus_channel,
            station_queries=args.focus_station_query,
            fallback_channels=[],
        )
    has_major = any(getattr(ev, "impact_tier", "C") in ("A", "B") for ev in focus_aligned_events)
    retrieval_events = focus_aligned_events[: max(0, args.max_retrieval_events)]
    analysis_events = focus_aligned_events[: max(0, args.max_analysis_events)]
    explanation_events = focus_aligned_events[: max(0, args.max_explanation_events)]
    retrieval_channel_names = [row.get("channel_name") for row in preliminary_focus_channels if row.get("channel_name")] or prediction.channel_names

    retrieval = EventRetrievalAgent(
        cache_dir=run_dir / "retrieval_cache_v2",
        enabled=args.enable_online_retrieval,
    )
    evidence_rows = retrieval.retrieve_for_events(
        retrieval_events,
        args.target_date,
        retrieval_channel_names,
        max_results_per_event=4,
    )
    evidence_source_rows_all = [row.to_dict() for row in evidence_rows]
    evidence_model_summaries = []
    qwenplus_stats = {
        "selected_events": 0,
        "api_calls": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "accepted_evidence": 0,
        "rejected_evidence": 0,
    }
    if args.enable_qwenplus_search:
        qwenplus = EvidenceResearchAgent(
            cache_dir=kb_root / "external_evidence_kb" / "qwen_plus_cache",
            cache_only=args.qwenplus_cache_only,
            force_refresh=args.qwenplus_force_refresh,
        )
        qwenplus_result = qwenplus.research_events(
            retrieval_events,
            args.target_date,
            retrieval_channel_names,
            max_events=args.qwenplus_max_events,
            min_score=args.qwenplus_min_event_score,
            horizon_hours=args.horizon,
        )
        evidence_source_rows_all = evidence_source_rows_all + qwenplus_result.to_sources()
        evidence_model_summaries = qwenplus_result.summaries
        qwenplus_stats = qwenplus_result.stats
    accepted_source_rows, rejected_source_rows = split_evidence_for_forecast_time(evidence_source_rows_all)
    (run_dir / "reports" / "retrieval_rejection_diagnostics.json").write_text(
        json.dumps(
            {
                "rejected_count": len(rejected_source_rows),
                "rejected_sources": rejected_source_rows,
                "policy": "Rejected retrieval sources are diagnostics only and are not passed to Qwen/vLLM or rendered in explanation Markdown.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    impacts = []
    proposed_adjusted = prediction.forecast
    adjusted = prediction.forecast
    proposed_adjusted_channels: List[str] = []
    adjusted_channels: List[str] = []
    correction_matrix = np.zeros_like(np.asarray(prediction.forecast, dtype=np.float32))
    adapter_checkpoint = ""
    reason = "Numerical baseline used directly."
    adapter_reason = reason
    abstain = True

    if args.mode == "numerical_only":
        pass
    elif args.mode == "rag_explain":
        if analysis_events:
            impacts = analyze_events(analysis_events, args, prediction.forecast_timestamps[0][:10])
            reason = "RAG/LLM explanation generated; numerical forecast left unchanged for ablation."
        abstain = True
    elif args.mode in ("event_adapter_frozen_moment", "event_adapter_peft_moment"):
        if has_major:
            proposed_adjusted, proposed_adjusted_channels, adapter_reason, correction_matrix, adapter_checkpoint = adapter_adjust(args.mode, args, prediction, filtered_events, device)
            adjusted = proposed_adjusted
            adjusted_channels = list(proposed_adjusted_channels)
            reason = adapter_reason
            abstain = len(adjusted_channels) == 0
        else:
            reason = "No Tier A/B event after relevance filtering; adapter abstained."
            abstain = True
    elif args.mode == "full_skill_agent":
        if has_major and analysis_events:
            impacts = analyze_events(analysis_events, args, prediction.forecast_timestamps[0][:10])
            fused = PredictionFusion(fusion_channel_names=set(prediction.channel_names)).fuse(prediction, impacts)
            adjusted = fused.adjusted_forecast
            adjusted_channels = fused.channels_adjusted
            reason = "Full event RAG plus skill-compatible fusion applied."
            abstain = len(adjusted_channels) == 0
        else:
            reason = "No Tier A/B event after relevance filtering; full agent abstained."
            abstain = True

    raw_metrics = compute_metrics(prediction.ground_truth, prediction.forecast)
    request = ForecastRequestSpec(
        date=args.target_date,
        horizon=args.horizon,
        station_scope=args.station_scope,
        mode=args.mode,
        retrieval_policy="online_cached" if args.enable_online_retrieval else "cache_record_only",
    )
    rag_residual_context = build_rag_residual_context(explanation_events, args, prediction.forecast_timestamps[0][:10])
    residual_cases = deduplicate_residual_cases(split_residual_case_blocks(rag_residual_context))
    historical_event_cases = extract_historical_event_cases(rag_residual_context)
    dedup_rag_residual_context = "\n\n".join(residual_cases)
    selected_residual_memory_skills = load_residual_memory_skills(args.residual_memory_skill_library)
    evidence_audit = ForecastEvidenceAuditor().audit(
        anchor_time=args.target_date,
        horizon_hours=args.horizon,
        structured_events=[ev.model_dump() for ev in explanation_events],
        evidence_sources=accepted_source_rows,
        model_assisted_summaries=evidence_model_summaries,
        channel_names=prediction.channel_names,
        adjusted_channel_candidates=proposed_adjusted_channels,
        historical_event_cases=historical_event_cases,
        residual_cases=residual_cases,
    )
    controller_result = {
        "controller_allowed": False,
        "abstain": bool(abstain),
        "final_adjusted_forecast": adjusted,
        "adjusted_channels": adjusted_channels,
        "abstention_reason": reason,
        "correction_stats": {"max_abs_correction": float(np.abs(correction_matrix).max()) if correction_matrix.size else 0.0, "mean_abs_correction": float(np.abs(correction_matrix).mean()) if correction_matrix.size else 0.0, "active_correction_cells": int((np.abs(correction_matrix) > 1e-8).sum()) if correction_matrix.size else 0, "correction_bound": 0.05 if "adapter" in args.mode else 0.10},
        "audit_scores": {k: evidence_audit.get(k) for k in ["source_validity_score", "geo_consistency_score", "temporal_alignment_score", "residual_support_score"]},
        "included_units": evidence_audit.get("included_units", []),
        "excluded_units": evidence_audit.get("excluded_units", []),
    }
    if args.mode in ("event_adapter_frozen_moment", "event_adapter_peft_moment"):
        event_channel_mask = np.abs(correction_matrix).max(axis=1) > 1e-8 if correction_matrix.size else np.zeros(len(prediction.channel_names), dtype=bool)
        controller_result = CalibrationController(correction_bound=0.05).apply(
            raw_forecast=prediction.forecast,
            proposed_adjusted=proposed_adjusted,
            correction=correction_matrix,
            channel_names=prediction.channel_names,
            adjusted_channel_candidates=proposed_adjusted_channels,
            event_channel_mask=event_channel_mask,
            evidence_audit=evidence_audit,
        )
        adjusted = controller_result["final_adjusted_forecast"]
        adjusted_channels = controller_result["adjusted_channels"]
        abstain = bool(controller_result["abstain"])
        reason = adapter_reason if controller_result["controller_allowed"] else f"Calibration controller abstained: {controller_result['abstention_reason']}"
        if focus_requested and controller_result.get("controller_allowed"):
            raw_arr = np.asarray(prediction.forecast, dtype=np.float32)
            corr = np.asarray(correction_matrix, dtype=np.float32)
            focus_names = [row.get("channel_name") for row in preliminary_focus_channels if row.get("channel_name")] or list(adjusted_channels)
            channel_mask = _channel_mask_from_names(prediction.channel_names, focus_names)
            hour_mask = np.zeros(raw_arr.shape[1], dtype=bool)
            for idx in preliminary_focus_window.get("horizon_indices", []):
                if 0 <= int(idx) < len(hour_mask):
                    hour_mask[int(idx)] = True
            if channel_mask.any() and hour_mask.any():
                masked_corr = np.where(channel_mask[:, None] & hour_mask[None, :], corr, 0.0)
                adjusted = (raw_arr + masked_corr).tolist()
                active_by_channel = np.abs(masked_corr).max(axis=1) > 1e-8
                adjusted_channels = [name for name, active in zip(prediction.channel_names, active_by_channel) if active]
                controller_result["final_adjusted_forecast"] = adjusted
                controller_result["adjusted_channels"] = adjusted_channels
                controller_result["focus_window_aligned"] = True
                controller_result["focus_window"] = preliminary_focus_window
                controller_result["focused_adjusted_channels"] = adjusted_channels
                controller_result["correction_stats"] = {
                    "max_abs_correction": float(np.abs(masked_corr).max()) if masked_corr.size else 0.0,
                    "mean_abs_correction": float(np.abs(masked_corr).mean()) if masked_corr.size else 0.0,
                    "active_correction_cells": int((np.abs(masked_corr) > 1e-8).sum()) if masked_corr.size else 0,
                    "correction_bound": 0.05,
                }
                if not adjusted_channels:
                    abstain = True
                    controller_result["abstain"] = True
                    controller_result["controller_allowed"] = False
                    controller_result["abstention_reason"] = "no active correction inside focus window"
                    reason = "Calibration controller abstained: no active correction inside focus window"
            else:
                adjusted = prediction.forecast
                adjusted_channels = []
                abstain = True
                controller_result["final_adjusted_forecast"] = adjusted
                controller_result["adjusted_channels"] = []
                controller_result["controller_allowed"] = False
                controller_result["abstain"] = True
                controller_result["focus_window_aligned"] = True
                controller_result["abstention_reason"] = "focus window has no matched correction cells"
                reason = "Calibration controller abstained: focus window has no matched correction cells"
    (run_dir / "reports" / "evidence_audit.json").write_text(json.dumps(evidence_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "reports" / "calibration_controller.json").write_text(json.dumps(controller_result, ensure_ascii=False, indent=2), encoding="utf-8")
    evidence = EventEvidenceSpec(
        sources=accepted_source_rows,
        rejected_sources=rejected_source_rows,
        structured_events=[ev.model_dump() for ev in explanation_events],
        local_residual_cases=residual_cases,
        historical_event_cases=historical_event_cases,
        model_assisted_summaries=evidence_model_summaries,
        selected_residual_memory_skills=selected_residual_memory_skills,
        evidence_audit=evidence_audit,
        has_major_event=has_major,
    )
    decision = CalibrationDecisionSpec(
        mode=args.mode,
        adjusted_channels=adjusted_channels,
        abstain=abstain,
        confidence=0.8 if adjusted_channels else 0.0,
        reason=reason,
        correction_bound=0.05 if "adapter" in args.mode else 0.10,
        controller_allowed=bool(controller_result.get("controller_allowed")),
        audit_scores=dict(controller_result.get("audit_scores") or {}),
        included_units=list(controller_result.get("included_units") or []),
        excluded_units=list(controller_result.get("excluded_units") or []),
        correction_stats=dict(controller_result.get("correction_stats") or {}),
    )
    adjusted_metrics = compute_metrics(prediction.ground_truth, adjusted)
    local_metric_groups = compute_local_forecast_metrics(
        prediction.ground_truth,
        prediction.forecast,
        adjusted,
        prediction.channel_names,
        prediction.forecast_timestamps,
        [ev.model_dump() for ev in explanation_events],
        adjusted_channels,
        decision.included_units,
        controller_allowed=bool(controller_result.get("controller_allowed")),
    )
    focused_prediction: Dict[str, Any] = {}
    if focus_requested:
        fallback_focus_channels = []
        for unit in decision.included_units:
            name = unit.get("station_channel") or unit.get("channel") or unit.get("station")
            if name:
                fallback_focus_channels.append(str(name))
        fallback_focus_channels.extend(adjusted_channels)
        resolved_focus_channels = resolve_focus_channels(
            prediction.channel_names,
            channel_map,
            venue_station_map,
            focus_channels=args.focus_channel,
            station_queries=args.focus_station_query,
            fallback_channels=fallback_focus_channels,
        )
        focus_window = preliminary_focus_window or resolve_focus_time_window(
            prediction.forecast_timestamps,
            focus_start=args.focus_start,
            focus_end=args.focus_end,
            focus_center=args.focus_center,
            hours_before=args.focus_hours_before,
            hours_after=args.focus_hours_after,
            structured_events=[ev.model_dump() for ev in explanation_events],
            fallback_anchor=args.target_date,
        )
        focused_prediction = build_focused_prediction(
            raw_forecast=prediction.forecast,
            adjusted_forecast=adjusted,
            ground_truth=prediction.ground_truth,
            channel_names=prediction.channel_names,
            timestamps=prediction.forecast_timestamps,
            resolved_channels=resolved_focus_channels,
            focus_window=focus_window,
            controller_allowed=bool(controller_result.get("controller_allowed")),
        )
        focused_prediction["focused_request"].update(
            {
                "include_focused_prediction": True,
                "focus_center_arg": args.focus_center,
                "focus_start_arg": args.focus_start,
                "focus_end_arg": args.focus_end,
                "focus_hours_before": int(args.focus_hours_before),
                "focus_hours_after": int(args.focus_hours_after),
                "focus_channel_args": list(args.focus_channel or []),
                "focus_station_query_args": list(args.focus_station_query or []),
            }
        )
    explanation_agent = ForecastExplanationAgent(
        base_url=args.llm_base_url,
        model=args.llm_model,
        api_key=args.llm_api_key,
    )
    explanation_result = explanation_agent.explain(
        forecast_request=request.__dict__,
        structured_events=[ev.model_dump() for ev in explanation_events],
        retrieved_sources=accepted_source_rows,
        station_scope=scope_meta,
        raw_forecast=prediction.forecast,
        channel_names=prediction.channel_names,
        calibration_decision=decision.__dict__,
        rag_residual_context=dedup_rag_residual_context,
        historical_event_cases=historical_event_cases,
        selected_residual_memory_skills=selected_residual_memory_skills,
        model_assisted_summaries=evidence_model_summaries,
        evidence_audit=evidence_audit,
        calibration_controller=controller_result,
        focused_forecast_context=focused_context_for_llm(focused_prediction),
        log_path=run_dir / "logs" / f"llm_explanation_{args.mode}.json",
    )
    model_explanation = (
        PredictionFusion().fuse(prediction, impacts).explanation
        if impacts and args.mode == "rag_explain"
        else reason
    )
    explanation = build_explanation_markdown(
        request,
        evidence,
        decision,
        adjusted_metrics,
        model_explanation,
        llm_markdown=explanation_result.markdown,
        numerical=NumericalForecastSpec(
            raw_forecast=prediction.forecast,
            channel_names=prediction.channel_names,
            timestamps=prediction.forecast_timestamps,
            ground_truth=prediction.ground_truth,
            model_type=getattr(agent, "model_type", "LP-MOMENT"),
        ),
        raw_metrics=raw_metrics,
        event_window_metrics=local_metric_groups["event_window_metrics"],
        adjusted_channel_metrics=local_metric_groups["adjusted_channel_metrics"],
        included_unit_metrics=local_metric_groups["included_unit_metrics"],
        focused_prediction=focused_prediction,
        adapter_checkpoint=adapter_checkpoint,
    )
    accepted_external = [src for src in evidence.sources if src.get("accepted") is True and src.get("url")]
    explanation_quality = evaluate_explanation_quality(
        markdown=explanation,
        accepted_evidence=accepted_external,
        decision=decision.__dict__,
        residual_cases=residual_cases,
        structured_events=[ev.model_dump() for ev in explanation_events],
        historical_event_cases=historical_event_cases,
        selected_residual_memory_skills=selected_residual_memory_skills,
    )
    if raw_metrics and adjusted_metrics:
        explanation_quality["wape_delta_raw_minus_adjusted"] = float(
            raw_metrics.get("wape", 0.0) - adjusted_metrics.get("wape", 0.0)
        )
    result = ForecastResultSpec(
        request=request,
        numerical=NumericalForecastSpec(
            raw_forecast=prediction.forecast,
            channel_names=prediction.channel_names,
            timestamps=prediction.forecast_timestamps or [],
            ground_truth=prediction.ground_truth,
            model_type=agent.model_type,
        ),
        evidence=evidence,
        decision=decision,
        adjusted_forecast=adjusted,
        explanation_markdown=explanation,
        metrics=adjusted_metrics,
        raw_metrics=raw_metrics,
        event_window_metrics=local_metric_groups["event_window_metrics"],
        adjusted_channel_metrics=local_metric_groups["adjusted_channel_metrics"],
        included_unit_metrics=local_metric_groups["included_unit_metrics"],
        focused_prediction=focused_prediction,
        explanation_quality=explanation_quality,
    )
    stem = f"{args.target_date.replace(':', '').replace(' ', '_')}_{args.station_scope}_{args.mode}"
    write_forecast_artifacts(result, run_dir, stem=stem)
    write_preference_record(run_dir, args.mode, raw_metrics, adjusted_metrics, explanation)

    if args.update_skills and args.mode == "full_skill_agent" and impacts:
        final_prediction = FinalPrediction(
            raw_forecast=prediction.forecast,
            adjusted_forecast=adjusted,
            events_considered=impacts,
            explanation=explanation,
            channel_names=prediction.channel_names,
            ground_truth=prediction.ground_truth,
            forecast_timestamps=prediction.forecast_timestamps,
            channels_adjusted=adjusted_channels,
            channels_passthrough=[ch for ch in prediction.channel_names if ch not in adjusted_channels],
        )
        maybe_update_skill_library(run_dir, final_prediction, channel_map)

    self_check = write_self_check(
        run_dir,
        result,
        raw_metrics,
        adjusted_metrics,
        explanation,
        evidence,
        explanation_quality,
        model_assisted_summaries=evidence_model_summaries,
        qwenplus_stats=qwenplus_stats,
    )
    summary = {
        "run_dir": str(run_dir),
        "mode": args.mode,
        "raw_metrics": raw_metrics,
        "adjusted_metrics": adjusted_metrics,
        "event_window_metrics": local_metric_groups["event_window_metrics"],
        "adjusted_channel_metrics": local_metric_groups["adjusted_channel_metrics"],
        "included_unit_metrics": local_metric_groups["included_unit_metrics"],
        "focused_prediction": focused_prediction,
        "has_major_event": has_major,
        "adjusted_channels": adjusted_channels,
        "llm_explanation_parsed": explanation_result.parsed,
        "llm_explanation_fallback_reason": explanation_result.fallback_reason,
        "qwenplus_stats": qwenplus_stats,
        "explanation_quality": explanation_quality,
        "selected_residual_memory_skills": len(selected_residual_memory_skills),
        "evidence_audit": evidence_audit,
        "calibration_controller": controller_result,
        "kb_manifests": kb_manifests,
        "self_check": self_check,
        "artifacts": result.artifacts,
    }
    (run_dir / "reports" / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
