"""Normalize EAF-MAS v2 outputs into visualization-ready tables."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .audit import audit_scores, event_rows
from .io import find_prediction_jsons, load_json
from .memory import residual_memory_rows, skill_memory_rows
from .metrics import metric_bundle, relative_correction, wape


def default_method_aliases() -> Dict[str, str]:
    return {
        "numerical_only": "PT-MOMENT",
        "rag_explain": "EAF-MAS-X",
        "event_adapter_frozen_moment": "EAF-MAS-C",
    }


def method_label(mode: str, aliases: Dict[str, str] | None = None) -> str:
    return (aliases or default_method_aliases()).get(mode, mode)


def build_processed_tables(
    input_roots: Iterable[str | Path],
    method_aliases: Dict[str, str] | None = None,
    neutral_threshold: float = 0.01,
    formal_metric_paths: Iterable[str | Path] | None = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    aliases = method_aliases or default_method_aliases()
    prediction_files = find_prediction_jsons(input_roots)
    case_rows: List[dict] = []
    channel_rows: List[dict] = []
    horizon_rows: List[dict] = []
    audit_rows: List[dict] = []
    residual_rows: List[dict] = []
    skill_rows: List[dict] = []
    warnings: List[str] = []

    for path in prediction_files:
        try:
            payload = load_json(path)
        except Exception as exc:
            warnings.append(f"failed to parse {path}: {type(exc).__name__}: {exc}")
            continue
        case_id = _case_id(path, payload)
        case_rows.append(_case_row(case_id, payload, aliases, path))
        channel_rows.extend(_channel_rows(case_id, payload, aliases, neutral_threshold))
        horizon_rows.extend(_horizon_rows(case_id, payload, aliases))
        audit_rows.extend(event_rows(case_id, payload))
        residual_rows.extend(residual_memory_rows(case_id, payload.get("evidence") or {}))
        skill_rows.extend(skill_memory_rows((payload.get("evidence") or {}).get("selected_residual_memory_skills") or []))

    tables = {
        "case_metrics": pd.DataFrame(case_rows),
        "channel_metrics": pd.DataFrame(channel_rows),
        "horizon_forecast": pd.DataFrame(horizon_rows),
        "event_audit": pd.DataFrame(audit_rows),
        "residual_memory": pd.DataFrame(residual_rows),
        "skill_memory": pd.DataFrame(skill_rows),
        "formal_metrics": load_formal_metric_rows(formal_metric_paths or [], aliases),
    }
    report = {
        "prediction_files": [_display_path(p) for p in prediction_files],
        "warnings": warnings,
        "case_metrics": f"{len(case_rows)} rows",
        "channel_metrics": f"{len(channel_rows)} rows",
        "horizon_forecast": f"{len(horizon_rows)} rows",
        "event_audit": f"{len(audit_rows)} rows",
        "residual_memory": f"{len(residual_rows)} rows",
        "skill_memory": f"{len(skill_rows)} rows",
        "formal_metrics": f"{len(tables['formal_metrics'])} rows",
    }
    return tables, report


def load_formal_metric_rows(paths: Iterable[str | Path], aliases: Dict[str, str] | None = None) -> pd.DataFrame:
    """Load rolling/full-test metric rows exported by run_full_paper_results.py."""
    rows = []
    for path_like in paths:
        path = Path(path_like)
        if not path.is_file():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        df = df.copy()
        df["source_file"] = _display_path(path)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True, sort=False)
    if "mode" in out:
        out["method_label"] = out["mode"].map(lambda value: method_label(str(value), aliases))
    if "date" in out:
        out["anchor_time"] = out["date"]
    if {"event_n", "non_event_n"}.issubset(out.columns):
        out["has_event_cells"] = pd.to_numeric(out["event_n"], errors="coerce").fillna(0) > 0
        out["total_eval_cells"] = (
            pd.to_numeric(out["event_n"], errors="coerce").fillna(0)
            + pd.to_numeric(out["non_event_n"], errors="coerce").fillna(0)
        )
    out = _attach_baseline_deltas(out)
    return out


def _attach_baseline_deltas(df: pd.DataFrame, baseline_mode: str = "numerical_only") -> pd.DataFrame:
    if df.empty or not {"anchor", "mode"}.issubset(df.columns):
        return df
    out = df.copy()
    keys = ["split", "scope", "anchor"]
    keys = [key for key in keys if key in out.columns]
    metric_cols = ["wape", "event_wape", "non_event_wape", "top128_wape", "mae", "event_mae", "non_event_mae"]
    baseline = out[out["mode"].astype(str) == baseline_mode][keys + [c for c in metric_cols if c in out]].copy()
    if baseline.empty:
        return out
    rename = {col: f"baseline_{col}" for col in metric_cols if col in baseline}
    baseline = baseline.rename(columns=rename)
    out = out.merge(baseline, on=keys, how="left")
    for col in metric_cols:
        base = f"baseline_{col}"
        if col in out and base in out:
            out[f"{col}_gain_vs_baseline"] = pd.to_numeric(out[base], errors="coerce") - pd.to_numeric(out[col], errors="coerce")
    return out


def load_station_metadata(channel_map_path: str | Path) -> pd.DataFrame:
    path = Path(channel_map_path)
    if not path.is_file():
        return pd.DataFrame()
    data = load_json(path)
    rows = []
    for ch in data.get("channels", []):
        rows.append(
            {
                "station_id": ch.get("station_complex_id"),
                "channel_id": ch.get("channel_name"),
                "station_name": ch.get("station_complex") or ch.get("station_name_clean"),
                "latitude": ch.get("latitude") or ch.get("lat"),
                "longitude": ch.get("longitude") or ch.get("lon"),
                "line": ch.get("daytime_routes") or ch.get("routes"),
                "complex": ch.get("station_complex"),
                "station_rank": ch.get("rank"),
                "station_group": _rank_group(ch.get("rank")),
            }
        )
    return pd.DataFrame(rows)


def load_event_metadata(events_path: str | Path) -> pd.DataFrame:
    path = Path(events_path)
    if not path.is_file():
        return pd.DataFrame()
    data = load_json(path)
    events = data.get("events") if isinstance(data, dict) else data
    rows = []
    for idx, ev in enumerate(events or []):
        rows.append(
            {
                "event_id": ev.get("event_id") or f"event_{idx}",
                "event_name": ev.get("title") or ev.get("event_name"),
                "event_type": ev.get("event_type") or ev.get("event_category"),
                "impact_tier": ev.get("impact_tier"),
                "start_time": ev.get("event_time") or ev.get("start_time"),
                "end_time": ev.get("end_time"),
                "venue": ev.get("venue_name"),
                "latitude": ev.get("latitude"),
                "longitude": ev.get("longitude"),
                "location_text": ev.get("location"),
                "source_type": ev.get("source_type"),
                "source_quality": ev.get("source_quality"),
            }
        )
    return pd.DataFrame(rows)


def write_processed_tables(tables: Dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)


def write_schema_report(report: Dict[str, Any], out_md: str | Path) -> None:
    lines = ["# Visualization Data Schema Report", ""]
    for key, value in report.items():
        if key == "prediction_files":
            lines.append("## Prediction files")
            for p in value:
                lines.append(f"- `{p}`")
        elif key == "warnings":
            lines.append("## Warnings")
            if value:
                for item in value:
                    lines.append(f"- {item}")
            else:
                lines.append("- none")
        else:
            lines.append(f"- `{key}`: {value}")
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _case_id(path: Path, payload: dict) -> str:
    request = payload.get("request") or {}
    base = f"{request.get('date','unknown')}|{request.get('mode','unknown')}|{path.parent.parent.name}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
    return f"{path.parent.parent.name}_{digest}"


def _physical_event_key(event: dict) -> str | None:
    """Return the canonical key used to join one physical event across artifacts."""
    if not event:
        return None
    title = re.sub(r"\s+", " ", str(event.get("title") or "").strip().lower())
    event_time = str(event.get("event_time") or event.get("start_time") or "").strip()[:16]
    location = str(event.get("location") or event.get("event_location") or "").split(" | ", 1)[0]
    location = re.sub(r"\s+", " ", location.strip().lower())
    if not title or not event_time or not location:
        return None
    return f"{title}|{event_time}|{location}"


def _max_applied_relative_correction(payload: dict) -> tuple[float | None, bool]:
    numerical = payload.get("numerical") or {}
    raw_values = numerical.get("raw_forecast")
    final_values = payload.get("adjusted_forecast")
    if raw_values is None or final_values is None:
        return None, False
    raw = np.asarray(raw_values, dtype=float)
    final = np.asarray(final_values, dtype=float)
    if raw.shape != final.shape or raw.size == 0:
        return None, False
    max_applied = float(np.max(np.abs(relative_correction(raw, final))))
    return max_applied, bool(np.allclose(raw, final, rtol=0.0, atol=1e-8, equal_nan=True))


def _case_row(case_id: str, payload: dict, aliases: Dict[str, str], path: Path) -> dict:
    request = payload.get("request") or {}
    evidence = payload.get("evidence") or {}
    decision = payload.get("decision") or {}
    raw = payload.get("raw_metrics") or {}
    adj = payload.get("metrics") or {}
    audit = audit_scores(evidence)
    audit_detail = evidence.get("evidence_audit") or {}
    thresholds = audit_detail.get("thresholds") or {}
    correction_stats = decision.get("correction_stats") or {}
    events = evidence.get("structured_events") or []
    first_event = events[0] if events else {}
    raw_wape = raw.get("wape")
    adjusted_wape = adj.get("wape")
    gain = _safe_sub(raw_wape, adjusted_wape)
    event_window = _localized_metric_fields(payload, "event_window_metrics", "event_window")
    adjusted_channel = _localized_metric_fields(payload, "adjusted_channel_metrics", "adjusted_channel")
    included_unit = _localized_metric_fields(payload, "included_unit_metrics", "included_unit")
    mode = request.get("mode")
    calibration_enabled = mode == "event_adapter_frozen_moment"
    controller_participated = bool(
        calibration_enabled
        and isinstance(decision.get("controller_allowed"), bool)
        and isinstance(decision.get("audit_scores"), dict)
        and isinstance(correction_stats, dict)
        and correction_stats
        and isinstance(audit_detail, dict)
        and thresholds
    )
    max_applied_correction, final_matches_raw = _max_applied_relative_correction(payload)
    failed_dimensions = []
    for field in (
        "residual_support_score",
        "geo_consistency_score",
        "source_validity_score",
        "temporal_alignment_score",
    ):
        score = audit_detail.get(field)
        threshold = thresholds.get(field)
        if score is not None and threshold is not None and float(score) < float(threshold):
            failed_dimensions.append(field)
    severe_flags = audit_detail.get("severe_conflict_flags") or []
    if severe_flags:
        failed_dimensions.append("severe_conflict")
    controller_reason = str(decision.get("reason") or "")
    controller_reason = re.sub(r"^Calibration controller abstained:\s*", "", controller_reason).strip()
    return {
        "case_id": case_id,
        "input_file": _display_path(path),
        "anchor_time": request.get("date"),
        "mode": mode,
        "method_label": method_label(mode, aliases),
        "station_scope": request.get("station_scope"),
        "horizon": request.get("horizon"),
        "event_id": f"{case_id}_event_0" if first_event else None,
        "event_name": first_event.get("title"),
        "event_type": first_event.get("event_type") or first_event.get("event_category"),
        "impact_tier": first_event.get("impact_tier"),
        "physical_event_key": _physical_event_key(first_event),
        "raw_mae": raw.get("mae"),
        "raw_rmse": raw.get("rmse"),
        "raw_wape": raw_wape,
        "raw_smape": raw.get("smape"),
        "adjusted_mae": adj.get("mae"),
        "adjusted_rmse": adj.get("rmse"),
        "adjusted_wape": adjusted_wape,
        "adjusted_smape": adj.get("smape"),
        "wape_gain": gain,
        "relative_wape_gain": gain / max(float(raw_wape or 0.0), 1e-8) if gain is not None else None,
        "calibration_enabled": calibration_enabled,
        "controller_participated": controller_participated,
        "calibration_decision": "abstain" if decision.get("abstain") else "apply",
        "abstain": bool(decision.get("abstain")),
        "controller_allowed": decision.get("controller_allowed"),
        "controller_abstention_reason": controller_reason or None,
        "confidence": decision.get("confidence"),
        "correction_bound": decision.get("correction_bound") or correction_stats.get("correction_bound"),
        "max_correction": correction_stats.get("max_abs_correction"),
        "adapter_proposal_max_abs_correction": correction_stats.get("max_abs_correction"),
        "adapter_proposal_active_cells": correction_stats.get("active_correction_cells"),
        "max_applied_correction": max_applied_correction,
        "final_matches_raw_forecast": final_matches_raw,
        "failed_evidence_dimensions": ",".join(failed_dimensions),
        "severe_conflict_flags": ",".join(str(flag) for flag in severe_flags),
        **{f"threshold_{field}": thresholds.get(field) for field in thresholds},
        "adjusted_channel_count": len(decision.get("adjusted_channels") or []),
        "excluded_channel_count": len(decision.get("excluded_units") or []),
        "has_verified_external_evidence": bool(evidence.get("sources")),
        "has_model_assisted_summary": bool(evidence.get("model_assisted_summaries")),
        "has_historical_residual_memory": bool(evidence.get("historical_event_cases")),
        "geo_conflict_flag": _has_flag(evidence, "geo"),
        "source_conflict_flag": _has_flag(evidence, "source"),
        "day_type_match_level": None,
        **event_window,
        **adjusted_channel,
        **included_unit,
        **audit,
    }


def _channel_rows(case_id: str, payload: dict, aliases: Dict[str, str], neutral_threshold: float) -> List[dict]:
    request = payload.get("request") or {}
    numerical = payload.get("numerical") or {}
    decision = payload.get("decision") or {}
    evidence = payload.get("evidence") or {}
    raw = np.asarray(numerical.get("raw_forecast") or [], dtype=float)
    actual = np.asarray(numerical.get("ground_truth") or [], dtype=float)
    adjusted = np.asarray(payload.get("adjusted_forecast") or raw, dtype=float)
    names = numerical.get("channel_names") or []
    adjusted_set = set(decision.get("adjusted_channels") or [])
    included = {r.get("station_channel"): r for r in decision.get("included_units") or []}
    excluded = {r.get("station_channel"): r for r in decision.get("excluded_units") or []}
    events = evidence.get("structured_events") or []
    first_event = events[0] if events else {}
    rows = []
    for i, name in enumerate(names):
        if i >= raw.shape[0] or i >= actual.shape[0] or i >= adjusted.shape[0]:
            continue
        raw_metrics = metric_bundle(actual[i], raw[i])
        adj_metrics = metric_bundle(actual[i], adjusted[i])
        corr = adjusted[i] - raw[i]
        bound = decision.get("correction_bound") or (decision.get("correction_stats") or {}).get("correction_bound") or 0.05
        gain = raw_metrics["wape"] - adj_metrics["wape"]
        rows.append(
            {
                "case_id": case_id,
                "anchor_time": request.get("date"),
                "mode": request.get("mode"),
                "method_label": method_label(request.get("mode"), aliases),
                "station_id": name.split("__", 1)[0],
                "channel_id": name,
                "station_name": name.split("__", 1)[-1] if "__" in name else name,
                "station_group": None,
                "event_id": f"{case_id}_event_0" if first_event else None,
                "event_name": first_event.get("title"),
                "event_type": first_event.get("event_type") or first_event.get("event_category"),
                "impact_tier": first_event.get("impact_tier"),
                "is_affected_channel": name in included,
                "is_adjusted_channel": name in adjusted_set,
                "is_excluded_channel": name in excluded,
                "gate_score": (included.get(name) or {}).get("gate_score"),
                "exclusion_reason": (excluded.get(name) or {}).get("exclusion_reason"),
                "raw_wape": raw_metrics["wape"],
                "adjusted_wape": adj_metrics["wape"],
                "wape_gain": gain,
                "raw_mae": raw_metrics["mae"],
                "adjusted_mae": adj_metrics["mae"],
                "correction_mean": float(np.mean(corr)),
                "correction_max": float(np.max(np.abs(corr))),
                "correction_abs_mean": float(np.mean(np.abs(corr))),
                "correction_bound": bound,
                "bound_utilization": float(np.max(np.abs(relative_correction(raw[i], adjusted[i]))) / max(float(bound), 1e-8)),
                "beneficial": gain > neutral_threshold,
                "harmful": gain < -neutral_threshold,
            }
        )
    return rows


def _horizon_rows(case_id: str, payload: dict, aliases: Dict[str, str]) -> List[dict]:
    request = payload.get("request") or {}
    numerical = payload.get("numerical") or {}
    decision = payload.get("decision") or {}
    raw = np.asarray(numerical.get("raw_forecast") or [], dtype=float)
    actual = np.asarray(numerical.get("ground_truth") or [], dtype=float)
    adjusted = np.asarray(payload.get("adjusted_forecast") or raw, dtype=float)
    timestamps = numerical.get("timestamps") or []
    names = numerical.get("channel_names") or []
    adjusted_set = set(decision.get("adjusted_channels") or [])
    included_set = {r.get("station_channel") for r in decision.get("included_units") or [] if r.get("station_channel")}
    bound = decision.get("correction_bound") or (decision.get("correction_stats") or {}).get("correction_bound") or 0.05
    event_window_mask = _event_window_mask(payload, timestamps)
    rows = []
    for i, name in enumerate(names):
        if i >= raw.shape[0] or i >= actual.shape[0] or i >= adjusted.shape[0]:
            continue
        for h in range(min(raw.shape[1], actual.shape[1], adjusted.shape[1], len(timestamps))):
            obs = float(actual[i, h])
            r = float(raw[i, h])
            a = float(adjusted[i, h])
            rows.append(
                {
                    "case_id": case_id,
                    "anchor_time": request.get("date"),
                    "timestamp": timestamps[h],
                    "horizon_step": h + 1,
                    "mode": request.get("mode"),
                    "method_label": method_label(request.get("mode"), aliases),
                    "station_id": name.split("__", 1)[0],
                    "channel_id": name,
                    "station_name": name.split("__", 1)[-1] if "__" in name else name,
                    "observed": obs,
                    "raw_forecast": r,
                    "adjusted_forecast": a,
                    "raw_error": r - obs,
                    "adjusted_error": a - obs,
                    "raw_abs_error": abs(r - obs),
                    "adjusted_abs_error": abs(a - obs),
                    "correction": a - r,
                    "relative_correction": (a - r) / max(abs(r), 1e-8),
                    "correction_bound": bound,
                    "is_event_window": bool(event_window_mask[h]) if h < len(event_window_mask) else False,
                    "is_affected_channel": name in adjusted_set or name in included_set,
                    "is_adjusted_channel": name in adjusted_set,
                }
            )
    return rows


def _localized_metric_fields(payload: dict, key: str, prefix: str) -> dict:
    group = payload.get(key) or {}
    raw = group.get("raw") or {}
    adjusted = group.get("adjusted") or {}
    delta = group.get("delta_raw_minus_adjusted") or {}
    raw_wape = raw.get("wape")
    adjusted_wape = adjusted.get("wape")
    raw_mae = raw.get("mae")
    adjusted_mae = adjusted.get("mae")
    return {
        f"{prefix}_status": group.get("status"),
        f"{prefix}_channel_count": group.get("channel_count", 0),
        f"{prefix}_hour_count": group.get("hour_count", 0),
        f"{prefix}_cell_count": group.get("cell_count", 0),
        f"{prefix}_raw_wape": raw_wape,
        f"{prefix}_adjusted_wape": adjusted_wape,
        f"{prefix}_wape_gain": delta.get("wape", _safe_sub(raw_wape, adjusted_wape)),
        f"{prefix}_raw_mae": raw_mae,
        f"{prefix}_adjusted_mae": adjusted_mae,
        f"{prefix}_mae_gain": delta.get("mae", _safe_sub(raw_mae, adjusted_mae)),
    }


def _event_window_mask(payload: dict, timestamps: List[str]) -> List[bool]:
    if not timestamps:
        return []
    ts = pd.to_datetime(pd.Series(timestamps), errors="coerce")
    mask = pd.Series([False] * len(ts))
    evidence = payload.get("evidence") or {}
    for event in evidence.get("structured_events") or []:
        start = _parse_event_datetime(
            event.get("event_time")
            or event.get("start_time")
            or event.get("start_datetime")
            or _extract_content_datetime(event.get("content"), "start_datetime")
        )
        if start is None:
            continue
        end = _parse_event_datetime(
            event.get("end_time")
            or event.get("end_datetime")
            or _extract_content_datetime(event.get("content"), "end_datetime")
        )
        if end is None or end < start:
            end = start
        window_start = start - pd.Timedelta(hours=3)
        window_end = end + pd.Timedelta(hours=6)
        mask = mask | ((ts >= window_start) & (ts <= window_end))
    return mask.fillna(False).astype(bool).tolist()


def _parse_event_datetime(value) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    try:
        parsed = pd.to_datetime(value, errors="coerce")
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    return parsed


def _extract_content_datetime(content: str | None, key: str) -> str | None:
    if not content:
        return None
    match = re.search(rf"{re.escape(key)}=([^;]+)", str(content))
    return match.group(1).strip() if match else None


def _safe_sub(a, b):
    try:
        return round(float(a) - float(b), 12)
    except Exception:
        return None


def _has_flag(evidence: dict, token: str) -> bool:
    audit = evidence.get("evidence_audit") or {}
    flags = " ".join(str(x) for x in (audit.get("conflict_flags") or []) + (audit.get("severe_conflict_flags") or []))
    return token.lower() in flags.lower()


def _rank_group(rank) -> str | None:
    try:
        r = int(rank)
    except Exception:
        return None
    if r <= 28:
        return "event_venue28"
    if r <= 64:
        return "top64"
    return "top128"


def _display_path(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(Path.cwd()))
    except Exception:
        parts = path.parts
        if "0206moment" in parts:
            idx = parts.index("0206moment")
            return str(Path(*parts[idx + 1 :]))
        return path.name
