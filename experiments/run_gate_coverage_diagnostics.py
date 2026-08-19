#!/usr/bin/env python3
"""Dataset-level Event-Station-Channel gate coverage diagnostics.

This script records forecast-time gate/controller states without changing the
PT-MOMENT raw forecast, the frozen residual adapter proposal, or any Skill
memory.  It is a diagnostic trace for correction scope and abstention reasons.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_calibration_matched_ablation import (  # noqa: E402
    ControllerThresholds,
    _audit_pass,
    _predict_adapter_correction,
    _residual_pass,
    compute_controller_scores,
    filter_events_for_window,
    prepare_timed_events,
)


TRACE_COLUMNS = [
    "anchor",
    "date",
    "horizon_idx",
    "timestamp",
    "station_or_channel",
    "has_structured_event",
    "event_type",
    "impact_tier",
    "venue_name",
    "station_matched",
    "channel_matched",
    "event_active",
    "residual_support_score",
    "source_validity_score",
    "geo_consistency_score",
    "temporal_alignment_score",
    "semantic_consistency_score",
    "audit_pass",
    "controller_allowed",
    "controller_partial_apply",
    "controller_abstained",
    "correction_applied",
    "correction_value",
    "abs_correction",
    "bound_value",
    "bound_clipped",
    "raw_forecast",
    "adjusted_forecast",
    "forecast_array_changed_by_skill",
    "reason_code",
]


FUNNEL_STAGES = [
    "all_cells",
    "event_candidate_cells",
    "station_channel_matched_cells",
    "residual_supported_cells",
    "audit_passed_cells",
    "controller_allowed_cells",
    "corrected_cells",
    "clipped_cells",
    "abstained_cells",
]


REASON_ORDER = [
    "no_event",
    "no_station_match",
    "no_channel_match",
    "weak_residual",
    "weak_source",
    "weak_geo",
    "weak_temporal",
    "allowed",
    "clipped",
    "allowed_no_numeric_effect",
]


def ensure_dirs(root: Path) -> None:
    for sub in ("predictions", "tables", "figures", "reports", "logs"):
        (root / sub).mkdir(parents=True, exist_ok=True)


def _norm_text(value: object, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    text = str(value).strip()
    return text if text else default


def _event_field(event: Mapping[str, object], names: Sequence[str], default: str = "") -> str:
    for name in names:
        if name in event:
            value = _norm_text(event.get(name), "")
            if value:
                return value
    return default


def _event_tier_priority(tier: str) -> int:
    order = {"A": 4, "B": 3, "C": 2, "D": 1}
    return order.get(str(tier).upper(), 0)


def _channel_tokens(channel_name: str) -> set[str]:
    text = channel_name.replace("__", "_").replace("/", "_").replace("-", "_")
    return {tok.lower() for tok in text.split("_") if tok and not tok.isdigit()}


def _extract_channel_station_fields(channel_name: str, channel_meta: Mapping[str, Mapping[str, object]]) -> Tuple[str, str]:
    meta = channel_meta.get(channel_name, {}) if isinstance(channel_meta, Mapping) else {}
    station_id = _event_field(meta, ["station_complex_id", "station_id", "complex_id"], "")
    station_name = _event_field(meta, ["station_name", "stop_name", "complex_name", "name"], "")
    return station_id, station_name


def _event_station_match(event: Mapping[str, object], channel_name: str, channel_meta: Mapping[str, Mapping[str, object]]) -> bool:
    station_id, station_name = _extract_channel_station_fields(channel_name, channel_meta)
    event_station = _event_field(event, ["station_complex_id", "station_id", "matched_station_id"], "")
    event_station_name = _event_field(event, ["matched_station", "station_name", "nearest_station", "matched_station_name"], "")
    event_channel = _event_field(event, ["channel_name", "station_channel", "matched_channel"], "")
    if event_channel and event_channel == channel_name:
        return True
    if station_id and event_station and station_id == event_station:
        return True
    if station_name and event_station_name and station_name.lower() in event_station_name.lower():
        return True
    if event_station_name:
        event_tokens = _channel_tokens(event_station_name)
        channel_tokens = _channel_tokens(channel_name)
        if len(event_tokens & channel_tokens) >= 2:
            return True
    return False


def _event_channel_match(event: Mapping[str, object], channel_name: str, channel_meta: Mapping[str, Mapping[str, object]]) -> bool:
    event_channel = _event_field(event, ["channel_name", "station_channel", "matched_channel"], "")
    if event_channel and event_channel == channel_name:
        return True
    return _event_station_match(event, channel_name, channel_meta)


def _event_time(event: Mapping[str, object]) -> pd.Timestamp | None:
    raw = event.get("event_time") or event.get("start_time") or event.get("date")
    if not raw:
        return None
    try:
        return pd.Timestamp(raw)
    except Exception:
        return None


def build_cell_event_attribution(
    *,
    timestamps: Sequence[object],
    channel_names: Sequence[str],
    events: Sequence[Mapping[str, object]],
    channel_meta: Mapping[str, Mapping[str, object]],
    event_active_mask: np.ndarray,
    event_window_before_hours: float = 3.0,
    event_window_after_hours: float = 6.0,
) -> Dict[str, np.ndarray]:
    """Attribute each channel-hour cell to a dominant forecast-time event.

    The attribution is diagnostic.  It does not decide correction by itself;
    controller decisions continue to use event feature scores and thresholds.
    """
    n_channels = len(channel_names)
    horizon = len(timestamps)
    ts = pd.to_datetime(pd.Series(list(timestamps))).to_numpy(dtype="datetime64[ns]")

    has_event = np.zeros((n_channels, horizon), dtype=bool)
    station_matched = np.zeros((n_channels, horizon), dtype=bool)
    channel_matched = np.zeros((n_channels, horizon), dtype=bool)
    event_type = np.full((n_channels, horizon), "none", dtype=object)
    impact_tier = np.full((n_channels, horizon), "none", dtype=object)
    venue_name = np.full((n_channels, horizon), "no_event", dtype=object)
    best_score = np.full((n_channels, horizon), -1.0, dtype=np.float32)

    for event in events:
        et = _event_time(event)
        if et is None:
            continue
        start = np.datetime64(et - pd.Timedelta(hours=float(event_window_before_hours)))
        end = np.datetime64(et + pd.Timedelta(hours=float(event_window_after_hours)))
        temporal_mask = (ts >= start) & (ts <= end)
        if not bool(temporal_mask.any()):
            continue
        typ = _event_field(event, ["event_type", "category", "type"], "event")
        tier = _event_field(event, ["impact_tier", "tier"], "unknown").upper()
        venue = _event_field(event, ["venue_name", "venue", "location_name"], "unknown venue")
        base_score = float(_event_tier_priority(tier))
        for c_idx, channel in enumerate(channel_names):
            st_match = _event_station_match(event, channel, channel_meta)
            ch_match = _event_channel_match(event, channel, channel_meta)
            score = base_score + (2.0 if ch_match else 0.0) + (1.0 if st_match else 0.0)
            target = np.where(temporal_mask & (score > best_score[c_idx]))[0]
            if target.size == 0:
                continue
            has_event[c_idx, target] = True
            station_matched[c_idx, target] = station_matched[c_idx, target] | bool(st_match)
            channel_matched[c_idx, target] = channel_matched[c_idx, target] | bool(ch_match)
            event_type[c_idx, target] = typ
            impact_tier[c_idx, target] = tier
            venue_name[c_idx, target] = venue
            best_score[c_idx, target] = score

    # Feature-cube event_active is the most faithful event-channel mask.  Keep
    # attribution station/channel flags at least consistent with it.
    event_active = np.asarray(event_active_mask, dtype=bool)
    station_matched = station_matched | event_active
    channel_matched = channel_matched | event_active
    has_event = has_event | event_active
    event_type[has_event & (event_type == "none")] = "event"
    impact_tier[has_event & (impact_tier == "none")] = "unknown"
    venue_name[has_event & (venue_name == "no_event")] = "structured_event"

    return {
        "has_structured_event": has_event,
        "event_type": event_type,
        "impact_tier": impact_tier,
        "venue_name": venue_name,
        "station_matched": station_matched,
        "channel_matched": channel_matched,
    }


def assign_reason_codes(
    *,
    has_structured_event: np.ndarray,
    station_matched: np.ndarray,
    channel_matched: np.ndarray,
    residual_pass: np.ndarray,
    scores: Mapping[str, np.ndarray],
    thresholds: ControllerThresholds,
    audit_pass: np.ndarray,
    controller_allowed: np.ndarray,
    correction_applied: np.ndarray,
    bound_clipped: np.ndarray,
) -> np.ndarray:
    reason = np.full(has_structured_event.shape, "allowed_no_numeric_effect", dtype=object)
    reason[~has_structured_event] = "no_event"
    reason[has_structured_event & ~station_matched] = "no_station_match"
    reason[has_structured_event & station_matched & ~channel_matched] = "no_channel_match"
    reason[has_structured_event & station_matched & channel_matched & ~residual_pass] = "weak_residual"
    candidate = has_structured_event & station_matched & channel_matched & residual_pass
    reason[candidate & (scores["source_validity_score"] < thresholds.source_validity)] = "weak_source"
    reason[candidate & (scores["source_validity_score"] >= thresholds.source_validity) & (scores["geo_consistency_score"] < thresholds.geo_consistency)] = "weak_geo"
    reason[
        candidate
        & (scores["source_validity_score"] >= thresholds.source_validity)
        & (scores["geo_consistency_score"] >= thresholds.geo_consistency)
        & (scores["temporal_alignment_score"] < thresholds.temporal_alignment)
    ] = "weak_temporal"
    reason[controller_allowed & ~correction_applied] = "allowed_no_numeric_effect"
    reason[controller_allowed & correction_applied] = "allowed"
    reason[controller_allowed & correction_applied & bound_clipped] = "clipped"
    # If an unlisted semantic/evidence condition blocks audit, keep a generic
    # forecast-time weak_source code rather than adding unsupported categories.
    blocked_by_other_audit = (
        candidate
        & ~audit_pass
        & (scores["source_validity_score"] >= thresholds.source_validity)
        & (scores["geo_consistency_score"] >= thresholds.geo_consistency)
        & (scores["temporal_alignment_score"] >= thresholds.temporal_alignment)
    )
    reason[blocked_by_other_audit] = "weak_source"
    return reason


def build_trace_frame(
    *,
    anchor: int,
    date: str,
    timestamps: Sequence[object],
    channel_names: Sequence[str],
    attribution: Mapping[str, np.ndarray],
    event_active: np.ndarray,
    scores: Mapping[str, np.ndarray],
    thresholds: ControllerThresholds,
    correction: np.ndarray,
    unbounded_correction: np.ndarray,
    raw_forecast: np.ndarray,
    residual_bound: float,
) -> pd.DataFrame:
    n_channels = len(channel_names)
    horizon = len(timestamps)
    residual_ok = _residual_pass(scores, thresholds)
    audit_ok = _audit_pass(scores, thresholds)
    controller_allowed = np.asarray(event_active, dtype=bool) & residual_ok & audit_ok
    correction = np.where(controller_allowed, np.clip(np.asarray(correction, dtype=np.float32), -residual_bound, residual_bound), 0.0)
    abs_correction = np.abs(correction)
    correction_applied = abs_correction > 1e-8
    bound_clipped = (np.abs(unbounded_correction) > float(residual_bound) + 1e-8) & controller_allowed & correction_applied
    adjusted = np.asarray(raw_forecast, dtype=np.float32) * (1.0 + correction)
    reason = assign_reason_codes(
        has_structured_event=attribution["has_structured_event"],
        station_matched=attribution["station_matched"],
        channel_matched=attribution["channel_matched"],
        residual_pass=residual_ok,
        scores=scores,
        thresholds=thresholds,
        audit_pass=audit_ok,
        controller_allowed=controller_allowed,
        correction_applied=correction_applied,
        bound_clipped=bound_clipped,
    )

    channel_grid = np.repeat(np.asarray(channel_names, dtype=object), horizon)
    horizon_grid = np.tile(np.arange(horizon, dtype=np.int16), n_channels)
    timestamp_grid = np.tile([str(pd.Timestamp(ts)) for ts in timestamps], n_channels)

    def flat(name: str) -> np.ndarray:
        return np.asarray(attribution[name]).reshape(-1)

    df = pd.DataFrame(
        {
            "anchor": np.full(n_channels * horizon, int(anchor), dtype=np.int32),
            "date": np.full(n_channels * horizon, str(date), dtype=object),
            "horizon_idx": horizon_grid,
            "timestamp": timestamp_grid,
            "station_or_channel": channel_grid,
            "has_structured_event": flat("has_structured_event").astype(bool),
            "event_type": flat("event_type"),
            "impact_tier": flat("impact_tier"),
            "venue_name": flat("venue_name"),
            "station_matched": flat("station_matched").astype(bool),
            "channel_matched": flat("channel_matched").astype(bool),
            "event_active": np.asarray(event_active, dtype=bool).reshape(-1),
            "residual_support_score": scores["residual_support_score"].reshape(-1).astype(np.float32),
            "source_validity_score": scores["source_validity_score"].reshape(-1).astype(np.float32),
            "geo_consistency_score": scores["geo_consistency_score"].reshape(-1).astype(np.float32),
            "temporal_alignment_score": scores["temporal_alignment_score"].reshape(-1).astype(np.float32),
            "semantic_consistency_score": scores["semantic_consistency_score"].reshape(-1).astype(np.float32),
            "audit_pass": audit_ok.reshape(-1).astype(bool),
            "controller_allowed": controller_allowed.reshape(-1).astype(bool),
            "controller_partial_apply": (controller_allowed & bound_clipped).reshape(-1).astype(bool),
            "controller_abstained": (attribution["has_structured_event"] & ~controller_allowed).reshape(-1).astype(bool),
            "correction_applied": correction_applied.reshape(-1).astype(bool),
            "correction_value": correction.reshape(-1).astype(np.float32),
            "abs_correction": abs_correction.reshape(-1).astype(np.float32),
            "bound_value": np.full(n_channels * horizon, float(residual_bound), dtype=np.float32),
            "bound_clipped": bound_clipped.reshape(-1).astype(bool),
            "raw_forecast": np.asarray(raw_forecast, dtype=np.float32).reshape(-1),
            "adjusted_forecast": adjusted.reshape(-1).astype(np.float32),
            "forecast_array_changed_by_skill": np.zeros(n_channels * horizon, dtype=bool),
            "reason_code": reason.reshape(-1),
        }
    )
    return df[TRACE_COLUMNS]


class GateTraceAggregator:
    def __init__(self) -> None:
        self.stage_counts: Counter[str] = Counter()
        self.reason_counts: Counter[str] = Counter()
        self.tier_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        self.anchor_count = 0

    def update(self, trace: pd.DataFrame) -> None:
        self.anchor_count += int(trace["anchor"].nunique())
        has_event = trace["has_structured_event"].to_numpy(dtype=bool)
        matched = trace["station_matched"].to_numpy(dtype=bool) & trace["channel_matched"].to_numpy(dtype=bool)
        residual_supported = matched & (
            trace["residual_support_score"].to_numpy(dtype=float)
            >= 0.0  # already encoded in controller_allowed; exact threshold reflected by reason/controller columns
        )
        # For the funnel, residual-supported means not blocked by weak_residual.
        residual_supported = matched & (trace["reason_code"].to_numpy(dtype=object) != "weak_residual")
        audit_passed = residual_supported & trace["audit_pass"].to_numpy(dtype=bool)
        allowed = trace["controller_allowed"].to_numpy(dtype=bool)
        corrected = trace["correction_applied"].to_numpy(dtype=bool)
        clipped = trace["bound_clipped"].to_numpy(dtype=bool)
        abstained = trace["controller_abstained"].to_numpy(dtype=bool)

        self.stage_counts["all_cells"] += int(len(trace))
        self.stage_counts["event_candidate_cells"] += int(has_event.sum())
        self.stage_counts["station_channel_matched_cells"] += int((has_event & matched).sum())
        self.stage_counts["residual_supported_cells"] += int((has_event & residual_supported).sum())
        self.stage_counts["audit_passed_cells"] += int((has_event & audit_passed).sum())
        self.stage_counts["controller_allowed_cells"] += int(allowed.sum())
        self.stage_counts["corrected_cells"] += int(corrected.sum())
        self.stage_counts["clipped_cells"] += int(clipped.sum())
        self.stage_counts["abstained_cells"] += int(abstained.sum())
        self.stage_counts["non_event_corrected_cells"] += int((~has_event & corrected).sum())

        self.reason_counts.update(trace["reason_code"].astype(str).tolist())
        for tier, group in trace.groupby("impact_tier", dropna=False):
            key = str(tier)
            g_has_event = group["has_structured_event"].to_numpy(dtype=bool)
            g_allowed = group["controller_allowed"].to_numpy(dtype=bool)
            g_corrected = group["correction_applied"].to_numpy(dtype=bool)
            g_clipped = group["bound_clipped"].to_numpy(dtype=bool)
            g_abstained = group["controller_abstained"].to_numpy(dtype=bool)
            self.tier_counts[key]["all_cells"] += int(len(group))
            self.tier_counts[key]["event_candidate_cells"] += int(g_has_event.sum())
            self.tier_counts[key]["controller_allowed_cells"] += int(g_allowed.sum())
            self.tier_counts[key]["corrected_cells"] += int(g_corrected.sum())
            self.tier_counts[key]["clipped_cells"] += int(g_clipped.sum())
            self.tier_counts[key]["abstained_cells"] += int(g_abstained.sum())

    def manifest_stats(self) -> dict:
        total = max(1, int(self.stage_counts["all_cells"]))
        non_event = total - int(self.stage_counts["event_candidate_cells"])
        non_event_corrections = int(self.stage_counts.get("non_event_corrected_cells", 0))
        return {
            "anchor_count": int(self.anchor_count),
            "total_cell_count": int(self.stage_counts["all_cells"]),
            "non_event_correction_ratio": float(non_event_corrections / non_event) if non_event else 0.0,
            "correction_sparsity": float(self.stage_counts["corrected_cells"] / total),
        }


def aggregate_trace_frame(trace: pd.DataFrame) -> GateTraceAggregator:
    agg = GateTraceAggregator()
    agg.update(trace)
    return agg


def write_trace(root: Path, trace_frames: Iterable[pd.DataFrame], trace_format: str = "parquet", allow_csv_fallback: bool = False) -> Tuple[str, GateTraceAggregator]:
    pred_dir = root / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    agg = GateTraceAggregator()
    trace_format = str(trace_format).lower()

    if trace_format == "parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception as exc:
            if not allow_csv_fallback:
                raise RuntimeError("pyarrow is required for default parquet trace output; pass --trace_format csv for explicit fallback") from exc
            trace_format = "csv"
        else:
            out_path = pred_dir / "gate_trace_rows.parquet"
            writer = None
            try:
                for frame in trace_frames:
                    frame = frame[TRACE_COLUMNS]
                    agg.update(frame)
                    table = pa.Table.from_pandas(frame, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
                    writer.write_table(table)
            finally:
                if writer is not None:
                    writer.close()
            return str(out_path), agg

    out_path = pred_dir / "gate_trace_rows.csv"
    first = True
    for frame in trace_frames:
        frame = frame[TRACE_COLUMNS]
        agg.update(frame)
        frame.to_csv(out_path, mode="w" if first else "a", header=first, index=False)
        first = False
    return str(out_path), agg


def _funnel_df(agg: GateTraceAggregator) -> pd.DataFrame:
    rows = []
    total = max(1, int(agg.stage_counts["all_cells"]))
    prev = None
    for stage in FUNNEL_STAGES:
        count = int(agg.stage_counts[stage])
        rows.append(
            {
                "stage": stage,
                "count": count,
                "ratio_of_all": float(count / total),
                "ratio_of_previous": float(count / prev) if prev and stage != "abstained_cells" else np.nan,
            }
        )
        if stage != "abstained_cells":
            prev = max(1, count)
    return pd.DataFrame(rows)


def _reason_df(agg: GateTraceAggregator) -> pd.DataFrame:
    total = max(1, int(agg.stage_counts["all_cells"]))
    candidate = max(1, int(agg.stage_counts["event_candidate_cells"]))
    rows = []
    for reason in REASON_ORDER:
        count = int(agg.reason_counts.get(reason, 0))
        rows.append(
            {
                "reason_code": reason,
                "cells": count,
                "ratio_of_all": float(count / total),
                "ratio_of_candidate": float(count / candidate),
            }
        )
    extras = sorted(set(agg.reason_counts) - set(REASON_ORDER))
    for reason in extras:
        count = int(agg.reason_counts[reason])
        rows.append(
            {
                "reason_code": reason,
                "cells": count,
                "ratio_of_all": float(count / total),
                "ratio_of_candidate": float(count / candidate),
            }
        )
    return pd.DataFrame(rows)


def _tier_df(agg: GateTraceAggregator) -> pd.DataFrame:
    rows = []
    for tier, counts in sorted(agg.tier_counts.items(), key=lambda kv: (str(kv[0]) == "none", str(kv[0]))):
        all_cells = max(1, int(counts["all_cells"]))
        candidate = max(1, int(counts["event_candidate_cells"]))
        rows.append(
            {
                "impact_tier": tier,
                "all_cells": int(counts["all_cells"]),
                "event_candidate_cells": int(counts["event_candidate_cells"]),
                "controller_allowed_cells": int(counts["controller_allowed_cells"]),
                "corrected_cells": int(counts["corrected_cells"]),
                "clipped_cells": int(counts["clipped_cells"]),
                "abstained_cells": int(counts["abstained_cells"]),
                "candidate_ratio": float(counts["event_candidate_cells"] / all_cells),
                "allowed_ratio_of_candidate": float(counts["controller_allowed_cells"] / candidate),
                "corrected_ratio_of_candidate": float(counts["corrected_cells"] / candidate),
                "clipped_ratio_of_corrected": float(counts["clipped_cells"] / max(1, int(counts["corrected_cells"]))),
                "abstained_ratio_of_candidate": float(counts["abstained_cells"] / candidate),
            }
        )
    return pd.DataFrame(rows)


def write_tables(root: Path, agg: GateTraceAggregator) -> Dict[str, pd.DataFrame]:
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    funnel = _funnel_df(agg)
    reason = _reason_df(agg)
    tier = _tier_df(agg)
    funnel.to_csv(tables / "gate_funnel_summary.csv", index=False)
    reason.to_csv(tables / "gate_reason_code_summary.csv", index=False)
    tier.to_csv(tables / "gate_by_event_tier_summary.csv", index=False)
    funnel.to_latex(tables / "gate_funnel_summary.tex", index=False, float_format=lambda x: f"{x:.4f}", escape=True)
    reason.to_latex(tables / "gate_reason_code_summary.tex", index=False, float_format=lambda x: f"{x:.4f}", escape=True)
    tier.to_latex(tables / "gate_by_event_tier_summary.tex", index=False, float_format=lambda x: f"{x:.4f}", escape=True)
    return {"funnel": funnel, "reason": reason, "tier": tier}


def write_figures(root: Path, tables: Mapping[str, pd.DataFrame]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    funnel = tables["funnel"].copy()
    labels = [s.replace("_", " ") for s in funnel["stage"]]
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    y = np.arange(len(funnel))
    ax.barh(y, funnel["count"], color=["#4e79a7" if s != "abstained_cells" else "#c44e52" for s in funnel["stage"]])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Station-channel-hour cells")
    ax.set_title("Dataset-level Event-Station-Channel gate coverage funnel")
    total = max(1, int(funnel.loc[funnel["stage"] == "all_cells", "count"].iloc[0]))
    for yi, (_, row) in enumerate(funnel.iterrows()):
        ax.text(row["count"], yi, f"  {int(row['count']):,} ({row['count']/total:.2%})", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "gate_coverage_funnel.png", dpi=220)
    fig.savefig(fig_dir / "gate_coverage_funnel.pdf")
    plt.close(fig)

    reason = tables["reason"].copy()
    reason = reason[reason["cells"] > 0].sort_values("cells", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.35 * len(reason) + 1.5)))
    ax.barh(reason["reason_code"], reason["cells"], color="#59a14f")
    ax.set_xlabel("Cells")
    ax.set_title("Abstention and application reason-code distribution")
    for yi, (_, row) in enumerate(reason.iterrows()):
        ax.text(row["cells"], yi, f"  {int(row['cells']):,}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "gate_reason_code_bar.png", dpi=220)
    fig.savefig(fig_dir / "gate_reason_code_bar.pdf")
    plt.close(fig)

    tier = tables["tier"].copy()
    if tier.empty:
        tier = pd.DataFrame({"impact_tier": ["none"], "candidate_ratio": [0.0], "allowed_ratio_of_candidate": [0.0], "corrected_ratio_of_candidate": [0.0], "abstained_ratio_of_candidate": [0.0]})
    heat_cols = ["candidate_ratio", "allowed_ratio_of_candidate", "corrected_ratio_of_candidate", "abstained_ratio_of_candidate"]
    matrix = tier[heat_cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, max(3.8, 0.42 * len(tier) + 1.8)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=max(0.01, float(np.nanmax(matrix))))
    ax.set_xticks(np.arange(len(heat_cols)))
    ax.set_xticklabels([c.replace("_", "\n") for c in heat_cols], fontsize=8)
    ax.set_yticks(np.arange(len(tier)))
    ax.set_yticklabels(tier["impact_tier"].astype(str))
    ax.set_title("Gate coverage by event impact tier")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1%}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="Ratio")
    fig.tight_layout()
    fig.savefig(fig_dir / "gate_by_tier_heatmap.png", dpi=220)
    fig.savefig(fig_dir / "gate_by_tier_heatmap.pdf")
    plt.close(fig)


def write_interpretation(root: Path, agg: GateTraceAggregator, manifest: Mapping[str, object]) -> None:
    funnel = _funnel_df(agg).set_index("stage")
    reason = _reason_df(agg).sort_values("cells", ascending=False)
    total = max(1, int(funnel.loc["all_cells", "count"]))
    corrected = int(funnel.loc["corrected_cells", "count"])
    candidate = int(funnel.loc["event_candidate_cells", "count"])
    abstained = int(funnel.loc["abstained_cells", "count"])
    top_reasons = reason[(reason["cells"] > 0) & (~reason["reason_code"].isin(["allowed", "clipped", "allowed_no_numeric_effect"]))].head(5)
    lines = [
        "# Gate Coverage Diagnostics Interpretation",
        "",
        "This diagnostic traces Event-Station-Channel gating over the matched test split. It does not call Qwen-Plus, local LLMs, or AutoSkill, and it does not change forecast arrays through Skill memory.",
        "",
        "## Scope Control",
        f"- Total station-channel-hour cells traced: `{total:,}`.",
        f"- Event candidate cells: `{candidate:,}` (`{candidate / total:.2%}` of all cells).",
        f"- Corrected cells: `{corrected:,}` (`{corrected / total:.4%}` of all cells).",
        f"- Abstained candidate cells: `{abstained:,}`.",
        f"- Non-event correction ratio: `{float(manifest.get('non_event_correction_ratio', 0.0)):.6f}`.",
        "",
        "## Main Abstention Reasons",
    ]
    if top_reasons.empty:
        lines.append("- No non-application reason codes were observed.")
    else:
        for _, row in top_reasons.iterrows():
            lines.append(f"- `{row['reason_code']}`: `{int(row['cells']):,}` cells.")
    lines.extend(
        [
            "",
            "## Paper Use",
            "- The funnel and reason-code bar are suitable for the main paper if the corrected-cell ratio is sparse and non-event correction remains near zero.",
            "- The tier heatmap is better suited for appendix unless the paper text needs detailed tier-level diagnostics.",
            "",
            "## Claim Boundary",
            "- This supports the claim that EAF-MAS constrains bounded calibration to forecast-time eligible event-channel cells.",
            "- It does not support a claim that RAG, AutoSkill, or LLM agents directly improve numerical forecasts.",
        ]
    )
    (root / "reports" / "gate_coverage_interpretation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(root: Path, manifest: MutableMapping[str, object], agg: GateTraceAggregator, trace_path: str, trace_format: str, tables: Mapping[str, pd.DataFrame]) -> dict:
    stats = agg.manifest_stats()
    non_event_rows = int(stats["total_cell_count"] - agg.stage_counts["event_candidate_cells"])
    non_event_corr = int(agg.stage_counts.get("non_event_corrected_cells", 0))
    manifest.update(
        {
            "status": "completed",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "trace_path": str(Path(trace_path).relative_to(root) if str(trace_path).startswith(str(root)) else trace_path),
            "trace_format": trace_format,
            "trace_compression": "zstd" if trace_format == "parquet" else None,
            "full_anchor_count": int(stats["anchor_count"]),
            "total_cell_count": int(stats["total_cell_count"]),
            "non_event_cells": non_event_rows,
            "non_event_correction_cells": int(non_event_corr),
            "non_event_correction_ratio": float(stats["non_event_correction_ratio"]),
            "non_event_unchanged_rate": float(1.0 - stats["non_event_correction_ratio"]),
            "correction_sparsity": float(stats["correction_sparsity"]),
            "forecast_array_changed_by_skill": False,
            "llm_or_qwenplus_used": False,
            "autoskill_updated": False,
            "test_time_training_or_promotion": False,
            "outputs": {
                "trace": str(Path(trace_path).relative_to(root) if str(trace_path).startswith(str(root)) else trace_path),
                "tables": [
                    "tables/gate_funnel_summary.csv",
                    "tables/gate_reason_code_summary.csv",
                    "tables/gate_by_event_tier_summary.csv",
                ],
                "figures": [
                    "figures/gate_coverage_funnel.png",
                    "figures/gate_reason_code_bar.png",
                    "figures/gate_by_tier_heatmap.png",
                ],
            },
        }
    )
    (root / "reports" / "gate_coverage_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return dict(manifest)


def load_reference_parity(run_root: str | Path) -> dict:
    """Read existing paper run metrics for provenance-only parity metadata."""
    rows_path = Path(run_root) / "predictions" / "full_metric_rows.csv"
    if not rows_path.exists():
        return {"available": False, "reason": f"missing {rows_path}"}
    try:
        rows = pd.read_csv(rows_path)
    except Exception as exc:
        return {"available": False, "reason": f"failed to read reference rows: {exc}"}
    if "mode" not in rows.columns:
        return {"available": False, "reason": "reference rows missing mode column"}
    ref = rows[rows["mode"] == "event_adapter_frozen"]
    raw = rows[rows["mode"] == "pt_moment"]
    if ref.empty:
        return {"available": False, "reason": "event_adapter_frozen mode missing"}
    out = {
        "available": True,
        "source": str(rows_path),
        "reference_mode": "event_adapter_frozen",
        "reference_anchor_count": int(ref["anchor"].nunique()) if "anchor" in ref.columns else int(len(ref)),
        "note": "Reference metrics are read for provenance only; gate decisions in this script do not use actual values.",
    }
    for col in ["event_active_wape", "active_correction_cells", "max_abs_correction", "non_event_wape"]:
        if col in ref.columns:
            out[f"reference_mean_{col}"] = float(pd.to_numeric(ref[col], errors="coerce").mean())
    if not raw.empty and "event_active_wape" in raw.columns and "event_active_wape" in ref.columns:
        out["reference_mean_event_active_gain_vs_pt_moment"] = float(
            pd.to_numeric(raw["event_active_wape"], errors="coerce").mean()
            - pd.to_numeric(ref["event_active_wape"], errors="coerce").mean()
        )
    return out


def finalize_outputs(root: Path, agg: GateTraceAggregator, trace_path: str, trace_format: str, manifest: MutableMapping[str, object]) -> dict:
    tables = write_tables(root, agg)
    write_figures(root, tables)
    final_manifest = write_manifest(root, manifest, agg, trace_path, trace_format, tables)
    write_interpretation(root, agg, final_manifest)
    return {"tables": tables, "manifest": final_manifest}


def run_synthetic_smoke(output_root: str | Path, trace_format: str = "parquet", seed: int = 13) -> dict:
    root = Path(output_root)
    ensure_dirs(root)
    rng = np.random.default_rng(seed)
    n_anchors, n_channels, horizon = 3, 4, 8
    channel_names = [f"S{i:03d}__Synthetic_Channel_{i}" for i in range(n_channels)]
    thresholds = ControllerThresholds()
    frames: List[pd.DataFrame] = []

    for a in range(n_anchors):
        timestamps = pd.date_range(f"2023-06-{1+a:02d} 10:00:00", periods=horizon, freq="h")
        raw = rng.uniform(100.0, 200.0, size=(n_channels, horizon)).astype(np.float32)
        event_active = np.zeros((n_channels, horizon), dtype=bool)
        event_active[:2, 2:6] = True
        scores = {name: np.full((n_channels, horizon), 0.85, dtype=np.float32) for name in [
            "source_validity_score",
            "geo_consistency_score",
            "temporal_alignment_score",
            "semantic_consistency_score",
            "residual_support_score",
            "evidence_validity_score",
        ]}
        scores["residual_support_score"][1, 3] = 0.05
        scores["source_validity_score"][0, 5] = 0.05
        scores["evidence_validity_score"][0, 5] = 0.05
        correction = np.zeros_like(raw, dtype=np.float32)
        correction[event_active] = -0.04
        correction[0, 2] = -0.05
        unbounded = correction.copy()
        unbounded[0, 2] = -0.12
        attribution = {
            "has_structured_event": np.zeros((n_channels, horizon), dtype=bool),
            "event_type": np.full((n_channels, horizon), "none", dtype=object),
            "impact_tier": np.full((n_channels, horizon), "none", dtype=object),
            "venue_name": np.full((n_channels, horizon), "no_event", dtype=object),
            "station_matched": np.zeros((n_channels, horizon), dtype=bool),
            "channel_matched": np.zeros((n_channels, horizon), dtype=bool),
        }
        attribution["has_structured_event"][:, 2:6] = True
        attribution["event_type"][:, 2:6] = "concert"
        attribution["impact_tier"][:, 2:6] = "A"
        attribution["venue_name"][:, 2:6] = "Synthetic Arena"
        attribution["station_matched"][:3, 2:6] = True
        attribution["channel_matched"][:2, 2:6] = True
        frames.append(
            build_trace_frame(
                anchor=1000 + a,
                date=str(timestamps[0]),
                timestamps=list(timestamps),
                channel_names=channel_names,
                attribution=attribution,
                event_active=event_active,
                scores=scores,
                thresholds=thresholds,
                correction=correction,
                unbounded_correction=unbounded,
                raw_forecast=raw,
                residual_bound=0.05,
            )
        )

    trace_path, agg = write_trace(root, frames, trace_format=trace_format, allow_csv_fallback=False)
    manifest = {
        "status": "synthetic_smoke",
        "synthetic": True,
        "random_seed": int(seed),
        "claim_boundary": "Synthetic smoke validates output contracts and gate invariants only.",
        "authoritative_root_read_only": True,
    }
    return finalize_outputs(root, agg, trace_path, trace_format, manifest)


def run_full_diagnostics(args: argparse.Namespace) -> dict:
    import torch
    from sklearn.preprocessing import StandardScaler

    from agents.event_adapter import build_event_feature_cube, channel_meta_by_name, load_adapter, load_channel_map, load_events_json
    from agents.numerical_agent import NumericalPredictionAgent
    from event_post_training.config import resolve_lp_model_path
    from experiments.run_ccfa_fullstack_final import _inverse_transform_batch, validate_real_lp_adapter_manifest
    from experiments.run_full_autoskill_skillbench import build_full_hourly_anchor_plan
    from experiments.run_full_paper_results import load_channel_names

    root = Path(args.output_root)
    ensure_dirs(root)
    traffic_csv = Path(args.traffic_csv)
    df = pd.read_csv(traffic_csv, parse_dates=["date"])
    channel_names = load_channel_names(traffic_csv)
    events = load_events_json(args.events_json)
    timed_events = prepare_timed_events(events)
    channel_meta = channel_meta_by_name(load_channel_map(args.channel_map))
    val_anchors, test_anchors = build_full_hourly_anchor_plan(traffic_csv, horizon=args.horizon, train_rows=args.train_rows, val_rows=args.val_rows)
    if args.max_test_anchors:
        test_anchors = test_anchors[: int(args.max_test_anchors)]
    if args.expected_test_anchors is not None and not args.max_test_anchors and len(test_anchors) != int(args.expected_test_anchors):
        raise ValueError(f"Expected {args.expected_test_anchors} test anchors, got {len(test_anchors)}")

    data_df = df.drop(columns=["date"]).infer_objects(copy=False).interpolate(method="cubic")
    data_np = data_df.to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    scaler.fit(data_np[: args.train_rows])
    data_norm = scaler.transform(data_np).astype(np.float32)

    lp_dir, lp_source = resolve_lp_model_path(Path(args.lp_model_path), PROJECT_ROOT)
    agent = NumericalPredictionAgent(
        model_path=str(root / "__no_gca_model_for_gate_diagnostics__"),
        device=args.device,
        forecast_horizon=args.horizon,
        n_channels=len(channel_names),
        channel_names=list(channel_names),
        lp_model_path=str(lp_dir),
        lp_data_path=str(traffic_csv),
        allow_lp_training_fallback=False,
    )
    adapter_manifest = Path(args.adapter_manifest) if args.adapter_manifest else Path(args.adapter_path) / "training_manifest.json"
    adapter_status = validate_real_lp_adapter_manifest(adapter_manifest)
    adapter, adapter_config = load_adapter(args.adapter_path, device=args.device)
    thresholds = ControllerThresholds(
        source_validity=args.source_validity_threshold,
        geo_consistency=args.geo_consistency_threshold,
        temporal_alignment=args.temporal_alignment_threshold,
        evidence_validity=args.evidence_validity_threshold,
        residual_support=args.residual_support_threshold,
    )

    seq_len = int(args.seq_len)

    def frame_iter() -> Iterable[pd.DataFrame]:
        processed = 0
        for start in range(0, len(test_anchors), max(1, int(args.batch_size))):
            batch = list(test_anchors[start : start + max(1, int(args.batch_size))])
            anchor_ids = [int(row["anchor"]) for row in batch]
            histories = np.stack([data_norm[a - seq_len : a].T for a in anchor_ids], axis=0)
            x = torch.tensor(histories, dtype=torch.float32, device=args.device)
            input_mask = torch.ones(len(batch), seq_len, dtype=torch.float32, device=args.device)
            with torch.no_grad():
                output = agent.model(x_enc=x, input_mask=input_mask)
                pred_norm = output.forecast.detach().cpu().numpy().astype(np.float32)
            raw_batch = _inverse_transform_batch(scaler, pred_norm)

            for b_idx, anchor in enumerate(batch):
                anchor_id = int(anchor["anchor"])
                timestamps = [pd.Timestamp(ts) for ts in df["date"].iloc[anchor_id : anchor_id + args.horizon].tolist()]
                raw = raw_batch[b_idx].astype(np.float32)
                filtered_events = filter_events_for_window(
                    timed_events,
                    pd.Timestamp(timestamps[0]),
                    pd.Timestamp(timestamps[-1]),
                    back_hours=args.event_prefilter_back_hours,
                    forward_hours=args.event_prefilter_forward_hours,
                )
                features = build_event_feature_cube(raw, [str(ts) for ts in timestamps], channel_names, filtered_events, channel_meta)
                event_active = features[..., 0] > 0.0
                scores = compute_controller_scores(features)
                proposal = _predict_adapter_correction(adapter, features, args.device, force_event_active=None, disable_bound=False)
                unbounded = _predict_adapter_correction(adapter, features, args.device, force_event_active=None, disable_bound=True)
                attribution = build_cell_event_attribution(
                    timestamps=timestamps,
                    channel_names=channel_names,
                    events=filtered_events,
                    channel_meta=channel_meta,
                    event_active_mask=event_active,
                    event_window_before_hours=args.event_window_before_hours,
                    event_window_after_hours=args.event_window_after_hours,
                )
                processed += 1
                if processed % max(1, int(args.progress_every)) == 0 or processed == len(test_anchors):
                    print(f"[gate_coverage] processed {processed}/{len(test_anchors)} anchors", flush=True)
                yield build_trace_frame(
                    anchor=anchor_id,
                    date=str(anchor.get("date") or timestamps[0]),
                    timestamps=timestamps,
                    channel_names=channel_names,
                    attribution=attribution,
                    event_active=event_active,
                    scores=scores,
                    thresholds=thresholds,
                    correction=proposal,
                    unbounded_correction=unbounded,
                    raw_forecast=raw,
                    residual_bound=args.residual_bound,
                )

    trace_path, agg = write_trace(root, frame_iter(), trace_format=args.trace_format, allow_csv_fallback=args.allow_csv_fallback)
    manifest = {
        "synthetic": False,
        "run_root": str(args.run_root),
        "run_root_read_only": True,
        "traffic_csv": str(traffic_csv),
        "events_json": str(args.events_json),
        "channel_map": str(args.channel_map),
        "fusion_channels": str(args.fusion_channels),
        "lp_model_path": str(lp_dir),
        "lp_source": lp_source,
        "adapter_path": str(args.adapter_path),
        "adapter_config": adapter_config,
        "adapter_manifest_status": adapter_status,
        "horizon": int(args.horizon),
        "train_rows": int(args.train_rows),
        "val_rows": int(args.val_rows),
        "test_anchor_count": int(len(test_anchors)),
        "expected_test_anchors": args.expected_test_anchors,
        "max_test_anchors": args.max_test_anchors,
        "controller_thresholds": asdict(thresholds),
        "residual_bound": float(args.residual_bound),
        "event_prefilter": {
            "timed_event_count": int(len(timed_events)),
            "back_hours": float(args.event_prefilter_back_hours),
            "forward_hours": float(args.event_prefilter_forward_hours),
            "event_window_before_hours": float(args.event_window_before_hours),
            "event_window_after_hours": float(args.event_window_after_hours),
            "uses_actual": False,
        },
        "decision_actual_leakage_guard": "Trace fields and controller states are computed from forecast-time features only; actual arrays are not loaded for decisions.",
        "full_controller_parity_reference": load_reference_parity(args.run_root),
    }
    return finalize_outputs(root, agg, trace_path, args.trace_format, manifest)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Run dataset-level Event-Station-Channel gate coverage diagnostics.")
    parser.add_argument("--run_root", default=str(PROJECT_ROOT / "autotemp" / "ccfa_fullstack_summary_only_20260619_123041"))
    parser.add_argument("--output_root", default=str(PROJECT_ROOT / "autotemp" / f"revision_trc_tits_ccfa_{timestamp}" / "gate_coverage_diagnostics"))
    parser.add_argument("--traffic_csv", default=str(PROJECT_ROOT / "data" / "nyc_top128_station_hourly_flow.csv"))
    parser.add_argument("--events_json", default=str(PROJECT_ROOT / "data" / "nyc_top128_station_events.json"))
    parser.add_argument("--channel_map", default=str(PROJECT_ROOT / "data" / "nyc_top128_channel_map.json"))
    parser.add_argument("--fusion_channels", default=str(PROJECT_ROOT / "data" / "venue37_fusion_channels.json"))
    parser.add_argument("--lp_model_path", default=str(PROJECT_ROOT / "experiments" / "outputs" / "lp_fallback_nyc_top128"))
    parser.add_argument("--adapter_path", default=str(PROJECT_ROOT / "experiments" / "outputs" / "event_adapter_formal_frozen_real_lpmoment_20260612_084207"))
    parser.add_argument("--adapter_manifest", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--horizon", type=int, default=192)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--train_rows", type=int, default=12 * 30 * 24)
    parser.add_argument("--val_rows", type=int, default=4 * 30 * 24)
    parser.add_argument("--expected_test_anchors", type=int, default=576)
    parser.add_argument("--max_test_anchors", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--trace_format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--allow_csv_fallback", action="store_true")
    parser.add_argument("--residual_bound", type=float, default=0.05)
    parser.add_argument("--event_prefilter_back_hours", type=float, default=72.0)
    parser.add_argument("--event_prefilter_forward_hours", type=float, default=2.0)
    parser.add_argument("--event_window_before_hours", type=float, default=3.0)
    parser.add_argument("--event_window_after_hours", type=float, default=6.0)
    parser.add_argument("--progress_every", type=int, default=32)
    parser.add_argument("--source_validity_threshold", type=float, default=0.20)
    parser.add_argument("--geo_consistency_threshold", type=float, default=0.20)
    parser.add_argument("--temporal_alignment_threshold", type=float, default=0.50)
    parser.add_argument("--evidence_validity_threshold", type=float, default=0.20)
    parser.add_argument("--residual_support_threshold", type=float, default=0.20)
    parser.add_argument("--synthetic_smoke", action="store_true")
    parser.add_argument("--random_seed", type=int, default=13)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.synthetic_smoke:
        run_synthetic_smoke(args.output_root, trace_format=args.trace_format, seed=args.random_seed)
    else:
        run_full_diagnostics(args)


if __name__ == "__main__":
    main()
