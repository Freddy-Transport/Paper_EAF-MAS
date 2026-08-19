#!/usr/bin/env python3
"""CCF-A full-stack final experiment orchestrator for EAF-MAS.

The entry keeps the paper claim boundaries explicit:
PT-MOMENT is the numerical backbone; the frozen residual adapter provides sparse
bounded event-channel correction; Qwen/RAG/AutoSkill improve evidence,
explanation, routing, residual-memory organization, and safe abstention.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.evidence_research_agent import EvidenceResearchAgent  # noqa: E402
from experiments.run_full_autoskill_skillbench import (  # noqa: E402
    EventWindowIndex,
    build_full_hourly_anchor_plan,
    run_full_autoskill_skillbench,
)

REQUIRED_QWENPLUS_ENV = ("OPENAI_BASE_URL", "OPENAI_API_KEY", "LLM_MODEL")
DEFAULT_TRAIN_ROWS = 12 * 30 * 24
DEFAULT_VAL_ROWS = 4 * 30 * 24
MAJOR_EVENT_TYPE_TERMS = {
    "parade",
    "street festival",
    "single block festival",
    "athletic race",
    "tour",
    "concert",
    "plaza event",
    "plaza partner event",
    "street event",
    "sport - adult",
}
MEDIUM_EVENT_TYPE_TERMS = {
    "special event",
    "production event",
    "block party",
    "religious event",
    "stationary demonstration",
    "open street",
}
QWENPLUS_LIVE_EVENT_TYPES = {
    "parade",
    "street festival",
    "single block festival",
    "athletic race / tour",
    "plaza event",
    "plaza partner event",
    "street event",
    "open street partner event",
    "block party",
}
QWENPLUS_LIVE_TEXT_TERMS = {
    "times square",
    "tsq live",
    "madison square garden",
    "yankee stadium",
    "citi field",
    "us open",
    "marathon",
    "10k",
    "nyrr",
    "concert",
    "festival",
    "parade",
}
QWENPLUS_EXCLUDED_TYPES = {
    "farmers market",
    "sport - youth",
    "sidewalk sale",
    "health fair",
}
HIGH_VALUE_TEXT_TERMS = {
    "times square",
    "madison square garden",
    "msg",
    "yankee stadium",
    "central park",
    "broadway",
    "marathon",
    "festival",
    "parade",
}
LOW_VALUE_TEXT_TERMS = {
    "covid",
    "testing site",
    "test site",
    "health fair",
    "clinic",
    "winter closure",
    "closure",
    "lawn closure",
    "routine",
    "setup",
    "load in",
    "load out",
    "maintenance",
}


def ensure_dirs(root: Path) -> None:
    for sub in [
        "reports",
        "tables",
        "figures",
        "predictions",
        "external_evidence/qwenplus_live_cache",
        "logs",
        "paper_assets",
    ]:
        (root / sub).mkdir(parents=True, exist_ok=True)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_events(path: str | Path) -> List[dict]:
    payload = read_json(path)
    rows = payload.get("events", payload) if isinstance(payload, dict) else payload
    return [dict(row) for row in rows or []]


def read_traffic_dates(path: str | Path) -> List[str]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        return [row[0] for row in reader if row]


def normalize_event_for_fullstack(event: Mapping[str, Any]) -> dict:
    """Infer paper-audit event fields when the raw NYC event record lacks them."""
    out = dict(event)
    event_type = str(out.get("event_type") or "").strip()
    text = " ".join(
        str(out.get(k, "") or "").lower()
        for k in ("title", "content", "location", "venue_name", "event_type")
    )
    event_type_l = event_type.lower()
    tier = str(out.get("impact_tier") or "").upper()
    low_value = any(term in text for term in LOW_VALUE_TEXT_TERMS)
    if tier not in {"A", "B", "C", "D"}:
        if low_value:
            tier = "C"
        elif any(term in event_type_l for term in MAJOR_EVENT_TYPE_TERMS) or any(term in text for term in HIGH_VALUE_TEXT_TERMS):
            tier = "A"
        elif any(term in event_type_l for term in MEDIUM_EVENT_TYPE_TERMS):
            tier = "B"
        else:
            tier = "C"
    elif low_value and tier in {"A", "B"}:
        tier = "C"
    out["impact_tier"] = tier
    if not out.get("relevance_score"):
        try:
            rank = float(out.get("station_rank", 128) or 128)
        except Exception:
            rank = 128.0
        base = 0.92 if out.get("channel_name") or out.get("station_complex_id") else 0.65
        rank_bonus = max(0.0, min(0.12, (128.0 - rank) / 128.0 * 0.12))
        tier_bonus = {"A": 0.06, "B": 0.03, "C": 0.0, "D": -0.08}.get(tier, 0.0)
        out["relevance_score"] = round(max(0.0, min(0.99, base + rank_bonus + tier_bonus)), 4)
    if not out.get("venue_name"):
        out["venue_name"] = str(out.get("location") or "").split("|")[0].strip()
    return out


def event_physical_key(event: Mapping[str, Any]) -> str:
    venue = str(event.get("venue_name") or "").strip()
    if not venue:
        venue = str(event.get("location") or "").split("|")[0].strip()
    return "|".join(
        str(value or "")[:120].strip().lower()
        for value in (event.get("title"), event.get("event_time"), venue)
    )


def is_qwenplus_live_candidate(event: Mapping[str, Any]) -> bool:
    event_type = str(event.get("event_type") or "").lower().strip()
    text = " ".join(str(event.get(k, "") or "").lower() for k in ("title", "location", "venue_name", "content"))
    if any(term in text for term in LOW_VALUE_TEXT_TERMS) or event_type in QWENPLUS_EXCLUDED_TYPES:
        return False
    if event_type in QWENPLUS_LIVE_EVENT_TYPES:
        return True
    if any(term in text for term in QWENPLUS_LIVE_TEXT_TERMS):
        return True
    if event_type == "sport - adult" and any(term in text for term in {"stadium", "madison square garden", "citi field", "us open", "yankee"}):
        return True
    return False


def validate_real_lp_adapter_manifest(path: str | Path | None) -> dict:
    if path is None:
        return {"valid": False, "status": "not_provided"}
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "training_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"adapter training manifest not found: {manifest_path}")
    payload = read_json(manifest_path)
    problems = []
    if payload.get("real_lp_moment_predictions") is not True:
        problems.append("real_lp_moment_predictions is not true")
    if payload.get("deterministic_fallback_used") is not False:
        problems.append("deterministic_fallback_used is not false")
    if payload.get("moment_head_training") is not False:
        problems.append("moment_head_training is not false")
    sources = payload.get("prediction_sources") or []
    if "moment_predictions_csv" not in sources:
        problems.append("prediction_sources does not include moment_predictions_csv")
    if problems:
        raise ValueError("Adapter manifest is not a real LP-MOMENT frozen-adapter run: " + "; ".join(problems))
    return {
        "valid": True,
        "manifest_path": str(manifest_path),
        "real_lp_moment_predictions": True,
        "deterministic_fallback_used": False,
        "moment_head_training": False,
        "prediction_sources": list(sources),
        "sample_counts": payload.get("sample_counts", {}),
        "best_val_loss": payload.get("best_val_loss"),
        "validation_calibration": payload.get("validation_calibration", {}),
    }


def require_qwenplus_live_env(enabled: bool) -> dict:
    if not enabled:
        return {"enabled": False, "env_present": False}
    missing = [key for key in REQUIRED_QWENPLUS_ENV if not os.environ.get(key)]
    if missing:
        raise RuntimeError(
            "Qwen-Plus live evidence was requested, but runtime env is missing: "
            + ", ".join(missing)
            + ". Refusing to generate fake summaries."
        )
    return {
        "enabled": True,
        "env_present": True,
        "base_url_present": bool(os.environ.get("OPENAI_BASE_URL")),
        "api_key_present": bool(os.environ.get("OPENAI_API_KEY")),
        "model": os.environ.get("LLM_MODEL", ""),
    }


def collect_unique_high_value_events(
    anchors: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    horizon: int,
    max_events: int | None = None,
) -> List[dict]:
    index = EventWindowIndex([normalize_event_for_fullstack(event) for event in events])
    selected: Dict[str, dict] = {}
    for anchor in anchors:
        anchor_date = str(anchor.get("date") or "")
        for event in index.events_for_anchor(anchor_date, horizon=horizon, max_events=None):
            normalized = normalize_event_for_fullstack(event)
            if str(normalized.get("impact_tier") or "").upper() not in {"A", "B"}:
                continue
            if float(normalized.get("relevance_score", 0.0) or 0.0) < 0.75:
                continue
            if not is_qwenplus_live_candidate(normalized):
                continue
            key = event_physical_key(normalized)
            if key not in selected:
                normalized["first_anchor_date"] = anchor_date
                normalized["physical_event_key"] = key
                selected[key] = normalized
            if max_events is not None and max_events >= 0 and len(selected) >= max_events:
                return list(selected.values())
    return list(selected.values())


def collect_qwenplus_live_evidence(
    output_root: str | Path,
    anchors: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    horizon: int = 192,
    enabled: bool = False,
    max_live_events: int | None = None,
    min_score: float = 0.75,
    force_refresh: bool = True,
    client_factory: Callable[..., object] | None = None,
) -> dict:
    root = Path(output_root)
    ensure_dirs(root)
    env_status = require_qwenplus_live_env(enabled)
    summary_path = root / "reports" / "qwenplus_live_evidence_summary.json"
    if not enabled:
        summary = {
            "enabled": False,
            "status": "disabled",
            "stats": {"api_calls": 0, "cache_hits": 0, "cache_misses": 0, "summary_count": 0, "summary_used_for_explanation": 0},
            "unique_high_value_physical_events": 0,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    unique_events = collect_unique_high_value_events(anchors, events, horizon=horizon, max_events=max_live_events)
    cache_dir = root / "external_evidence" / "qwenplus_live_cache"
    concurrency = max(1, int(os.environ.get("QWENPLUS_LIVE_CONCURRENCY", "6") or 6))
    print(
        f"[qwenplus-summary] unique_events={len(unique_events)} concurrency={concurrency} force_refresh={bool(force_refresh)}",
        flush=True,
    )
    totals = defaultdict(int)
    event_rows: List[dict] = []

    def _research_one(event: Mapping[str, Any]) -> dict:
        agent = EvidenceResearchAgent(
            cache_dir=cache_dir,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL", ""),
            model=os.environ.get("LLM_MODEL", "qwen-plus"),
            force_refresh=force_refresh,
            client_factory=client_factory,
        )
        target_date = str(event.get("first_anchor_date") or event.get("event_time") or "")
        station_names = [str(event.get("channel_name") or event.get("station_complex") or "event_venue28")]
        result = agent.research_events(
            [event],
            target_date=target_date,
            station_names=station_names,
            max_events=1,
            min_score=min_score,
            horizon_hours=horizon,
        )
        return {
            "physical_event_key": event.get("physical_event_key"),
            "title": event.get("title"),
            "event_time": event.get("event_time"),
            "first_anchor_date": target_date,
            "impact_tier": event.get("impact_tier"),
            "source_type": "model-assisted",
            "model_assisted_summary_count": len(result.summaries),
            "summary_used_for_explanation": any(bool(row.get("summary_used_for_explanation", True)) for row in result.summaries),
            "stats": result.stats,
            "summaries": result.summaries,
        }

    if unique_events:
        completed = 0
        with ThreadPoolExecutor(max_workers=min(concurrency, len(unique_events))) as executor:
            future_to_event = {executor.submit(_research_one, event): event for event in unique_events}
            for future in as_completed(future_to_event):
                row = future.result()
                completed += 1
                for key, value in (row.get("stats") or {}).items():
                    if key in {"selected_events", "api_calls", "cache_hits", "cache_misses", "summary_count", "summary_used_for_explanation"}:
                        totals[key] += int(value or 0)
                event_rows.append(row)
                if completed % 25 == 0 or completed == len(unique_events):
                    print(f"[qwenplus-summary] {completed}/{len(unique_events)} completed", flush=True)
    summary = {
        "enabled": True,
        "status": "completed",
        "env": {k: v for k, v in env_status.items() if k != "api_key_present"} | {"api_key_present": True},
        "unique_high_value_physical_events": len(unique_events),
        "max_live_events": max_live_events,
        "min_score": min_score,
        "force_refresh": bool(force_refresh),
        "stats": dict(totals),
        "events": event_rows,
        "claim_boundary": "Qwen-Plus generates model-assisted event summaries for explanation; it is not a URL citation retriever or numerical forecaster.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (root / "tables" / "qwenplus_live_evidence_events.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["physical_event_key", "title", "event_time", "first_anchor_date", "impact_tier", "model_assisted_summary_count", "summary_used_for_explanation"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in event_rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return summary


def _secret_scan_text(root: Path) -> dict:
    import re

    patterns = [
        ("dashscope_or_openai_key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
        ("api_key_env_name", re.compile(r"OPENAI_API_KEY")),
        ("ssh_password_marker", re.compile(r"(?i)(ssh[_-]?password|password\\s*[:=])")),
    ]
    hits = []
    for path in root.rglob("*"):
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for name, pattern in patterns:
            if pattern.search(text):
                hits.append({"path": str(path), "pattern": name})
                break
    return {"passed": not hits, "hits": hits[:20], "hit_count": len(hits)}


def _write_anchor_tables(root: Path, val_anchors: Sequence[dict], test_anchors: Sequence[dict]) -> None:
    with (root / "reports" / "full_anchor_plan.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["split", "row_idx", "anchor", "date", "horizon", "horizon_end_row_exclusive", "horizon_end_time"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in list(val_anchors) + list(test_anchors):
            writer.writerow({field: row.get(field, "") for field in fields})


def _run_numeric_fullstack(
    root: Path,
    traffic_csv: str | Path,
    events_json: str | Path,
    channel_map: str | Path,
    fusion_channels: str | Path,
    lp_model_path: str | Path,
    adapter_path: str | Path,
    test_anchors: Sequence[dict],
    device: str,
    save_prediction_cache: bool,
) -> dict:
    import pandas as pd
    from agents.event_adapter import apply_adapter_to_forecast, build_event_feature_cube, channel_meta_by_name, load_channel_map, load_events_json
    from agents.numerical_agent import NumericalPredictionAgent
    from event_post_training.config import EventPostTrainingConfig, resolve_lp_model_path
    from experiments.run_full_paper_results import load_channel_names, load_venue_indices, metrics

    cfg = EventPostTrainingConfig(
        traffic_csv=Path(traffic_csv),
        events_json=Path(events_json),
        channel_map_path=Path(channel_map),
        fusion_channels_path=Path(fusion_channels),
        device=device,
    )
    df = pd.read_csv(cfg.traffic_csv, parse_dates=["date"])
    channel_names = load_channel_names(Path(traffic_csv))
    venue_indices, venue_meta = load_venue_indices(Path(fusion_channels), channel_names)
    events = load_events_json(events_json)
    channel_meta = channel_meta_by_name(load_channel_map(channel_map))
    lp_dir, lp_source = resolve_lp_model_path(Path(lp_model_path), PROJECT_ROOT)
    agent = NumericalPredictionAgent(
        model_path=str(root / "__no_gca_model_for_fullstack__"),
        device=device,
        forecast_horizon=cfg.horizon,
        n_channels=len(channel_names),
        channel_names=list(channel_names),
        lp_model_path=str(lp_dir),
        lp_data_path=str(traffic_csv),
        allow_lp_training_fallback=False,
    )
    modes = ["pt_moment", "event_adapter_frozen"]
    rows: List[dict] = []
    horizon_abs_error = {mode: np.zeros(cfg.horizon, dtype=np.float64) for mode in modes}
    horizon_actual_sum = {mode: np.zeros(cfg.horizon, dtype=np.float64) for mode in modes}
    event_gain_rows: List[dict] = []
    cache_raw: List[np.ndarray] = []
    cache_adjusted: List[np.ndarray] = []
    cache_actual: List[np.ndarray] = []
    adapter_path = Path(adapter_path)
    for idx, anchor_row in enumerate(test_anchors, 1):
        anchor = int(anchor_row["anchor"])
        target_date = str(df["date"].iloc[anchor])
        pred = agent.predict_for_date(str(traffic_csv), target_date, forecast_horizon=cfg.horizon)
        raw = np.asarray(pred.forecast, dtype=np.float32)
        actual = np.asarray(pred.ground_truth, dtype=np.float32)
        timestamps = list(pred.forecast_timestamps or [str(x) for x in df["date"].iloc[anchor: anchor + cfg.horizon]])
        adjusted, correction = apply_adapter_to_forecast(raw, timestamps, channel_names, events, channel_meta, adapter_path, device=device)
        adjusted = np.asarray(adjusted, dtype=np.float32)
        event_features = build_event_feature_cube(raw, timestamps, channel_names, events, channel_meta)
        event_mask = event_features[..., 0] > 0
        predictions = {
            "pt_moment": raw,
            "event_adapter_frozen": adjusted,
        }
        venue_actual = actual[list(venue_indices)]
        venue_mask = event_mask[list(venue_indices)]
        raw_event_wape = None
        adjusted_event_wape = None
        for mode, arr in predictions.items():
            top = metrics(actual, arr)
            venue = metrics(venue_actual, arr[list(venue_indices)])
            event_m = metrics(venue_actual, arr[list(venue_indices)], venue_mask)
            non_event_m = metrics(venue_actual, arr[list(venue_indices)], ~venue_mask)
            rows.append({
                "split": "test",
                "anchor": anchor,
                "date": target_date,
                "mode": mode,
                "top128_wape": top["wape"],
                "top128_mae": top["mae"],
                "event_venue28_wape": venue["wape"],
                "event_venue28_mae": venue["mae"],
                "event_active_wape": event_m["wape"],
                "event_active_mae": event_m["mae"],
                "event_active_n": event_m["n"],
                "non_event_wape": non_event_m["wape"],
                "non_event_mae": non_event_m["mae"],
                "non_event_n": non_event_m["n"],
                "max_abs_correction": float(np.max(np.abs(adjusted - raw))),
                "active_correction_cells": int(np.sum(np.abs(adjusted - raw) > 1e-5)),
            })
            horizon_abs_error[mode] += np.sum(np.abs(arr - actual), axis=0)
            horizon_actual_sum[mode] += np.sum(np.abs(actual), axis=0)
            if mode == "pt_moment":
                raw_event_wape = rows[-1]["event_active_wape"]
            elif mode == "event_adapter_frozen":
                adjusted_event_wape = rows[-1]["event_active_wape"]
        if raw_event_wape is not None and adjusted_event_wape is not None:
            event_gain_rows.append({"anchor": anchor, "date": target_date, "event_active_wape_gain": float(raw_event_wape - adjusted_event_wape)})
        if save_prediction_cache:
            cache_raw.append(raw.astype(np.float16))
            cache_adjusted.append(adjusted.astype(np.float16))
            cache_actual.append(actual.astype(np.float16))
        if idx % 25 == 0 or idx == len(test_anchors):
            print(f"[numeric] {idx}/{len(test_anchors)} anchor={anchor} date={target_date}", flush=True)
    table_path = root / "predictions" / "full_metric_rows.csv"
    with table_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    per_horizon = []
    for mode in modes:
        wape_by_h = horizon_abs_error[mode] / np.maximum(horizon_actual_sum[mode], 1.0) * 100.0
        for h, value in enumerate(wape_by_h):
            per_horizon.append({"mode": mode, "horizon_idx": h, "wape": float(value)})
    with (root / "tables" / "per_horizon_wape.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "horizon_idx", "wape"])
        writer.writeheader(); writer.writerows(per_horizon)
    if event_gain_rows:
        with (root / "tables" / "event_active_gain_by_anchor.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(event_gain_rows[0].keys()))
            writer.writeheader(); writer.writerows(event_gain_rows)
    summary = {}
    for mode in modes:
        ms = [row for row in rows if row["mode"] == mode]
        summary[mode] = {key: float(np.mean([float(row[key]) for row in ms])) for key in ["top128_wape", "event_venue28_wape", "event_active_wape", "non_event_wape"]}
        summary[mode]["n_windows"] = len(ms)
    if save_prediction_cache and cache_raw:
        np.savez_compressed(
            root / "predictions" / "full_test_prediction_cache.npz",
            raw=np.stack(cache_raw),
            adjusted=np.stack(cache_adjusted),
            actual=np.stack(cache_actual),
        )
    return {
        "status": "completed",
        "lp_source": lp_source,
        "n_test_anchors": len(test_anchors),
        "summary": summary,
        "full_metric_rows": str(table_path),
        "venue_meta": venue_meta,
        "prediction_cache_written": bool(save_prediction_cache and cache_raw),
    }



def _inverse_transform_batch(scaler, pred_norm: np.ndarray) -> np.ndarray:
    """Inverse-transform batched MOMENT output shaped (B, C, H)."""
    b, c, h = pred_norm.shape
    rows = pred_norm.transpose(0, 2, 1).reshape(b * h, c)
    inv = scaler.inverse_transform(rows).reshape(b, h, c).transpose(0, 2, 1)
    return inv.astype(np.float32)


def _run_numeric_fullstack_batch(
    root: Path,
    traffic_csv: str | Path,
    events_json: str | Path,
    channel_map: str | Path,
    fusion_channels: str | Path,
    lp_model_path: str | Path,
    adapter_path: str | Path,
    test_anchors: Sequence[dict],
    device: str,
    save_prediction_cache: bool,
    batch_size: int = 8,
) -> dict:
    import pandas as pd
    import torch
    from sklearn.preprocessing import StandardScaler
    from agents.event_adapter import (
        apply_correction,
        build_event_feature_cube,
        channel_meta_by_name,
        load_adapter,
        load_channel_map,
        load_events_json,
    )
    from agents.numerical_agent import NumericalPredictionAgent
    from event_post_training.config import EventPostTrainingConfig, resolve_lp_model_path
    from experiments.run_full_paper_results import load_channel_names, load_venue_indices, metrics

    cfg = EventPostTrainingConfig(
        traffic_csv=Path(traffic_csv),
        events_json=Path(events_json),
        channel_map_path=Path(channel_map),
        fusion_channels_path=Path(fusion_channels),
        device=device,
    )
    df = pd.read_csv(cfg.traffic_csv, parse_dates=["date"])
    channel_names = load_channel_names(Path(traffic_csv))
    venue_indices, venue_meta = load_venue_indices(Path(fusion_channels), channel_names)
    events = load_events_json(events_json)
    channel_meta = channel_meta_by_name(load_channel_map(channel_map))
    data_df = df.drop(columns=["date"]).infer_objects(copy=False).interpolate(method="cubic")
    data_np = data_df.to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    scaler.fit(data_np[: cfg.train_rows])
    data_norm = scaler.transform(data_np).astype(np.float32)
    lp_dir, lp_source = resolve_lp_model_path(Path(lp_model_path), PROJECT_ROOT)
    agent = NumericalPredictionAgent(
        model_path=str(root / "__no_gca_model_for_fullstack_batch__"),
        device=device,
        forecast_horizon=cfg.horizon,
        n_channels=len(channel_names),
        channel_names=list(channel_names),
        lp_model_path=str(lp_dir),
        lp_data_path=str(traffic_csv),
        allow_lp_training_fallback=False,
    )
    adapter, _ = load_adapter(adapter_path, device=device)
    seq_len = cfg.seq_len
    horizon = cfg.horizon
    modes = ["pt_moment", "event_adapter_frozen"]
    rows: List[dict] = []
    event_gain_rows: List[dict] = []
    horizon_abs_error = {mode: np.zeros(horizon, dtype=np.float64) for mode in modes}
    horizon_actual_sum = {mode: np.zeros(horizon, dtype=np.float64) for mode in modes}
    cache_raw: List[np.ndarray] = []
    cache_adjusted: List[np.ndarray] = []
    cache_actual: List[np.ndarray] = []

    for start in range(0, len(test_anchors), max(1, int(batch_size))):
        batch = list(test_anchors[start : start + max(1, int(batch_size))])
        anchor_ids = [int(row["anchor"]) for row in batch]
        histories = np.stack([data_norm[a - seq_len : a].T for a in anchor_ids], axis=0)
        x = torch.tensor(histories, dtype=torch.float32, device=device)
        input_mask = torch.ones(len(batch), seq_len, dtype=torch.float32, device=device)
        with torch.no_grad():
            output = agent.model(x_enc=x, input_mask=input_mask)
            pred_norm = output.forecast.detach().cpu().numpy().astype(np.float32)
        raw_batch = _inverse_transform_batch(scaler, pred_norm)
        for b_idx, anchor in enumerate(anchor_ids):
            target_date = str(df["date"].iloc[anchor])
            timestamps = [str(ts) for ts in df["date"].iloc[anchor : anchor + horizon].tolist()]
            actual = data_np[anchor : anchor + horizon].T.astype(np.float32)
            raw = raw_batch[b_idx]
            features = build_event_feature_cube(raw, timestamps, channel_names, events, channel_meta)
            adjusted, correction = apply_correction(adapter, raw, features, device=device)
            adjusted = np.asarray(adjusted, dtype=np.float32)
            event_mask = features[..., 0] > 0
            predictions = {
                "pt_moment": raw,
                "event_adapter_frozen": adjusted,
            }
            venue_actual = actual[list(venue_indices)]
            venue_mask = event_mask[list(venue_indices)]
            raw_event_wape = None
            adjusted_event_wape = None
            for mode, arr in predictions.items():
                top = metrics(actual, arr)
                venue = metrics(venue_actual, arr[list(venue_indices)])
                event_m = metrics(venue_actual, arr[list(venue_indices)], venue_mask)
                non_event_m = metrics(venue_actual, arr[list(venue_indices)], ~venue_mask)
                row = {
                    "split": "test",
                    "anchor": anchor,
                    "date": target_date,
                    "mode": mode,
                    "top128_wape": top["wape"],
                    "top128_mae": top["mae"],
                    "event_venue28_wape": venue["wape"],
                    "event_venue28_mae": venue["mae"],
                    "event_active_wape": event_m["wape"],
                    "event_active_mae": event_m["mae"],
                    "event_active_n": event_m["n"],
                    "non_event_wape": non_event_m["wape"],
                    "non_event_mae": non_event_m["mae"],
                    "non_event_n": non_event_m["n"],
                    "max_abs_correction": float(np.max(np.abs(adjusted - raw))),
                    "active_correction_cells": int(np.sum(np.abs(adjusted - raw) > 1e-5)),
                }
                rows.append(row)
                horizon_abs_error[mode] += np.sum(np.abs(arr - actual), axis=0)
                horizon_actual_sum[mode] += np.sum(np.abs(actual), axis=0)
                if mode == "pt_moment":
                    raw_event_wape = row["event_active_wape"]
                elif mode == "event_adapter_frozen":
                    adjusted_event_wape = row["event_active_wape"]
            if raw_event_wape is not None and adjusted_event_wape is not None:
                event_gain_rows.append({"anchor": anchor, "date": target_date, "event_active_wape_gain": float(raw_event_wape - adjusted_event_wape)})
            if save_prediction_cache:
                cache_raw.append(raw.astype(np.float16)); cache_adjusted.append(adjusted.astype(np.float16)); cache_actual.append(actual.astype(np.float16))
        done = min(start + len(batch), len(test_anchors))
        print(f"[numeric-batch] {done}/{len(test_anchors)} anchors", flush=True)
    table_path = root / "predictions" / "full_metric_rows.csv"
    with table_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    per_horizon = []
    for mode in modes:
        wape_by_h = horizon_abs_error[mode] / np.maximum(horizon_actual_sum[mode], 1.0) * 100.0
        for h, value in enumerate(wape_by_h):
            per_horizon.append({"mode": mode, "horizon_idx": h, "wape": float(value)})
    with (root / "tables" / "per_horizon_wape.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "horizon_idx", "wape"])
        writer.writeheader(); writer.writerows(per_horizon)
    with (root / "tables" / "event_active_gain_by_anchor.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["anchor", "date", "event_active_wape_gain"])
        writer.writeheader(); writer.writerows(event_gain_rows)
    summary = {}
    for mode in modes:
        ms = [row for row in rows if row["mode"] == mode]
        summary[mode] = {key: float(np.mean([float(row[key]) for row in ms])) for key in ["top128_wape", "event_venue28_wape", "event_active_wape", "non_event_wape"]}
        summary[mode]["n_windows"] = len(ms)
    if save_prediction_cache and cache_raw:
        np.savez_compressed(root / "predictions" / "full_test_prediction_cache.npz", raw=np.stack(cache_raw), adjusted=np.stack(cache_adjusted), actual=np.stack(cache_actual))
    return {"status": "completed", "lp_source": lp_source, "n_test_anchors": len(test_anchors), "summary": summary, "full_metric_rows": str(table_path), "venue_meta": venue_meta, "prediction_cache_written": bool(save_prediction_cache and cache_raw), "batch_size": int(batch_size)}

def _make_paper_asset_manifest(root: Path, numeric: dict, skillbench: dict, qwenplus: dict) -> dict:
    figures_main = []
    figures_appendix = []
    for path in sorted((root / "figures").glob("*.png")):
        name = path.name
        if any(key in name for key in ["coverage", "lifecycle", "quality", "rank", "safety", "judge"]):
            figures_main.append(str(path))
        else:
            figures_appendix.append(str(path))
    manifest = {
        "section_4_1": {
            "purpose": "full test split and forecasting statistics",
            "numeric_summary": numeric.get("summary", {}),
            "recommended_tables": [str(root / "predictions" / "full_metric_rows.csv"), str(root / "tables" / "per_horizon_wape.csv")],
        },
        "section_4_2": {
            "purpose": "focused case studies only; do not use as dataset-level evidence",
            "recommended_source": "Qwen-Plus live case cards generated separately from full split statistics",
        },
        "section_4_3": {
            "purpose": "AutoSkill effectiveness and residual-memory organization",
            "recommended_figures": figures_main,
            "skillbench_summary": skillbench,
        },
        "section_4_4": {
            "purpose": "evidence audit, source quality, explanation faithfulness, abstention safety",
            "qwenplus_summary": qwenplus,
            "limitation": "citation-quality URL evidence and audit-score variance must be reported honestly.",
        },
        "appendix": {"diagnostic_figures": figures_appendix},
    }
    (root / "paper_assets" / "paper_asset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def run_ccfa_fullstack_final(
    output_root: str | Path,
    traffic_csv: str | Path = PROJECT_ROOT / "data/nyc_top128_station_hourly_flow.csv",
    events_json: str | Path = PROJECT_ROOT / "data/nyc_top128_station_events.json",
    channel_map: str | Path = PROJECT_ROOT / "data/nyc_top128_channel_map.json",
    fusion_channels: str | Path = PROJECT_ROOT / "data/venue37_fusion_channels.json",
    residual_kb_preview: str | Path | None = PROJECT_ROOT / "agents/knowledge_base_residual_venue37/documents_preview.json",
    adapter_manifest: str | Path | None = PROJECT_ROOT / "experiments/outputs/event_adapter_formal_frozen_real_lpmoment_20260612_084207/training_manifest.json",
    adapter_path: str | Path = PROJECT_ROOT / "experiments/outputs/event_adapter_formal_frozen_real_lpmoment_20260612_084207",
    lp_model_path: str | Path = PROJECT_ROOT / "experiments/outputs/lp_fallback_nyc_top128",
    horizon: int = 192,
    train_rows: int = DEFAULT_TRAIN_ROWS,
    val_rows: int = DEFAULT_VAL_ROWS,
    expected_val_anchors: int | None = 2689,
    expected_test_anchors: int | None = 576,
    run_numeric: bool = True,
    run_skillbench: bool = True,
    enable_qwenplus_live: bool = False,
    qwenplus_max_live_events: int | None = None,
    enable_local_vllm: bool = True,
    llm_base_url: str = "http://127.0.0.1:8000/v1",
    llm_model: str = "Qwen/Qwen3-8B",
    device: str = "cuda:0",
    save_prediction_cache: bool = True,
) -> dict:
    root = Path(output_root)
    ensure_dirs(root)
    traffic_csv = Path(traffic_csv)
    events_json = Path(events_json)
    val_anchors, test_anchors = build_full_hourly_anchor_plan(traffic_csv, horizon=horizon, train_rows=train_rows, val_rows=val_rows)
    if expected_val_anchors is not None and len(val_anchors) != int(expected_val_anchors):
        raise ValueError(f"Expected {expected_val_anchors} validation anchors, got {len(val_anchors)}")
    if expected_test_anchors is not None and len(test_anchors) != int(expected_test_anchors):
        raise ValueError(f"Expected {expected_test_anchors} test anchors, got {len(test_anchors)}")
    _write_anchor_tables(root, val_anchors, test_anchors)
    adapter_status = validate_real_lp_adapter_manifest(adapter_manifest) if adapter_manifest else {"valid": False, "status": "skipped"}
    events = read_events(events_json)
    qwenplus = collect_qwenplus_live_evidence(
        root,
        anchors=val_anchors + test_anchors,
        events=events,
        horizon=horizon,
        enabled=enable_qwenplus_live,
        max_live_events=qwenplus_max_live_events,
        force_refresh=True,
    )
    skillbench = {"status": "skipped"}
    if run_skillbench:
        if residual_kb_preview is None:
            raise ValueError("residual_kb_preview is required when run_skillbench=True")
        skillbench = run_full_autoskill_skillbench(
            output_root=root / "autoskill_skillbench_full",
            traffic_csv=traffic_csv,
            events_json=events_json,
            residual_kb_preview=residual_kb_preview,
            station_scope="event_venue28",
            anchor_policy="exhaustive_hourly",
            horizon=horizon,
            train_rows=train_rows,
            val_rows=val_rows,
            expected_val_anchors=expected_val_anchors,
            expected_test_anchors=expected_test_anchors,
            enable_qwenplus_live=False,
            enable_local_vllm_explanations=enable_local_vllm,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
        )
    numeric = {"status": "skipped"}
    if run_numeric:
        numeric = _run_numeric_fullstack_batch(
            root=root,
            traffic_csv=traffic_csv,
            events_json=events_json,
            channel_map=channel_map,
            fusion_channels=fusion_channels,
            lp_model_path=lp_model_path,
            adapter_path=adapter_path,
            test_anchors=test_anchors,
            device=device,
            save_prediction_cache=save_prediction_cache,
        )
    paper_assets = _make_paper_asset_manifest(root, numeric, skillbench, qwenplus)
    manifest = {
        "output_root": str(root),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "anchors": {"val_count": len(val_anchors), "test_count": len(test_anchors), "horizon": horizon, "train_rows": train_rows, "val_rows": val_rows},
        "adapter_manifest": adapter_status,
        "qwenplus_live": qwenplus,
        "local_vllm": {"enabled": bool(enable_local_vllm), "base_url": llm_base_url, "model": llm_model},
        "numeric_fullstack": numeric,
        "autoskill_skillbench": skillbench,
        "paper_assets": paper_assets,
        "claim_boundary": {
            "pt_moment_backbone": True,
            "moment_head_training": False,
            "frozen_residual_adapter_only": True,
            "skill_changes_forecast_arrays": False,
            "rag_skill_primary_role": "explainability_traceability_memory_routing_safe_abstention",
        },
    }
    secret_scan = _secret_scan_text(root)
    manifest["secret_scan"] = secret_scan
    (root / "reports" / "ccfa_fullstack_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="autotemp/ccfa_fullstack_final_" + time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--traffic_csv", default="data/nyc_top128_station_hourly_flow.csv")
    parser.add_argument("--events_json", default="data/nyc_top128_station_events.json")
    parser.add_argument("--channel_map", default="data/nyc_top128_channel_map.json")
    parser.add_argument("--fusion_channels", default="data/venue37_fusion_channels.json")
    parser.add_argument("--residual_kb_preview", default="agents/knowledge_base_residual_venue37/documents_preview.json")
    parser.add_argument("--adapter_manifest", default="experiments/outputs/event_adapter_formal_frozen_real_lpmoment_20260612_084207/training_manifest.json")
    parser.add_argument("--adapter_path", default="experiments/outputs/event_adapter_formal_frozen_real_lpmoment_20260612_084207")
    parser.add_argument("--lp_model_path", default="experiments/outputs/lp_fallback_nyc_top128")
    parser.add_argument("--horizon", type=int, default=192)
    parser.add_argument("--train_rows", type=int, default=DEFAULT_TRAIN_ROWS)
    parser.add_argument("--val_rows", type=int, default=DEFAULT_VAL_ROWS)
    parser.add_argument("--expected_val_anchors", type=int, default=2689)
    parser.add_argument("--expected_test_anchors", type=int, default=576)
    parser.add_argument("--skip_numeric", action="store_true")
    parser.add_argument("--skip_skillbench", action="store_true")
    parser.add_argument("--enable_qwenplus_live", action="store_true")
    parser.add_argument("--qwenplus_max_live_events", type=int, default=-1, help="-1 means all selected unique events")
    parser.add_argument("--disable_local_vllm", action="store_true")
    parser.add_argument("--llm_base_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm_model", default="Qwen/Qwen3-8B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no_prediction_cache", action="store_true")
    return parser.parse_args()


def _resolve_project_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def main() -> None:
    args = parse_args()
    max_live = None if args.qwenplus_max_live_events is None or args.qwenplus_max_live_events < 0 else args.qwenplus_max_live_events
    manifest = run_ccfa_fullstack_final(
        output_root=_resolve_project_path(args.output_root),
        traffic_csv=_resolve_project_path(args.traffic_csv),
        events_json=_resolve_project_path(args.events_json),
        channel_map=_resolve_project_path(args.channel_map),
        fusion_channels=_resolve_project_path(args.fusion_channels),
        residual_kb_preview=_resolve_project_path(args.residual_kb_preview),
        adapter_manifest=_resolve_project_path(args.adapter_manifest),
        adapter_path=_resolve_project_path(args.adapter_path),
        lp_model_path=_resolve_project_path(args.lp_model_path),
        horizon=args.horizon,
        train_rows=args.train_rows,
        val_rows=args.val_rows,
        expected_val_anchors=args.expected_val_anchors,
        expected_test_anchors=args.expected_test_anchors,
        run_numeric=not args.skip_numeric,
        run_skillbench=not args.skip_skillbench,
        enable_qwenplus_live=args.enable_qwenplus_live,
        qwenplus_max_live_events=max_live,
        enable_local_vllm=not args.disable_local_vllm,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        device=args.device,
        save_prediction_cache=not args.no_prediction_cache,
    )
    print(json.dumps({"output_root": manifest["output_root"], "anchors": manifest["anchors"], "secret_scan": manifest["secret_scan"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
