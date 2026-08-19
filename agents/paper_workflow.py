"""Paper-oriented workflow schema and artifact writers."""

from __future__ import annotations

import json
import re
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def display_station_name(channel_name: Any, meta: Optional[Dict[str, Any]] = None, max_len: int = 72) -> str:
    """Return a paper-facing station label while preserving raw IDs elsewhere."""
    meta = meta or {}
    label = str(meta.get("station_complex") or meta.get("station_name_clean") or "")
    if not label:
        raw = str(channel_name or "")
        label = raw.split("__", 1)[-1] if "__" in raw else raw
        label = label.replace("_St_Penn_Station", " St-Penn Station")
        label = label.replace("_St_Herald_Sq", " St-Herald Sq")
        label = label.replace("_Sts_Rockefeller_Center", " Sts-Rockefeller Center")
        label = label.replace("_Av_Barclays_Ctr", " Av-Barclays Ctr")
        label = label.replace("_", " ")
    label = re.sub(r"\s+", " ", label).strip()
    if ("Times Sq-42 St" in label or "Times Sq 42 St" in label) and "Bryant" in label:
        label = "Times Sq-42 St / 42 St / Bryant Pk / 5 Av"
    if len(label) > max_len:
        label = label[: max_len - 1].rstrip() + "…"
    return label or str(channel_name or "unknown station")


@dataclass
class ForecastRequestSpec:
    date: str
    horizon: int
    station_scope: str
    mode: str
    retrieval_policy: str = "online_cached"


@dataclass
class NumericalForecastSpec:
    raw_forecast: List[List[float]]
    channel_names: List[str]
    timestamps: List[str]
    ground_truth: Optional[List[List[float]]] = None
    model_type: str = ""


@dataclass
class EventEvidenceSpec:
    sources: List[Dict[str, Any]] = field(default_factory=list)
    rejected_sources: List[Dict[str, Any]] = field(default_factory=list)
    structured_events: List[Dict[str, Any]] = field(default_factory=list)
    local_residual_cases: List[str] = field(default_factory=list)
    historical_event_cases: List[Dict[str, Any]] = field(default_factory=list)
    model_assisted_summaries: List[Dict[str, Any]] = field(default_factory=list)
    selected_residual_memory_skills: List[Dict[str, Any]] = field(default_factory=list)
    evidence_audit: Dict[str, Any] = field(default_factory=dict)
    has_major_event: bool = False


@dataclass
class CalibrationDecisionSpec:
    mode: str
    adjusted_channels: List[str] = field(default_factory=list)
    abstain: bool = False
    confidence: float = 0.0
    reason: str = ""
    correction_bound: float = 0.0
    controller_allowed: bool = False
    audit_scores: Dict[str, Any] = field(default_factory=dict)
    included_units: List[Dict[str, Any]] = field(default_factory=list)
    excluded_units: List[Dict[str, Any]] = field(default_factory=list)
    correction_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ForecastResultSpec:
    request: ForecastRequestSpec
    numerical: NumericalForecastSpec
    evidence: EventEvidenceSpec
    decision: CalibrationDecisionSpec
    adjusted_forecast: List[List[float]]
    explanation_markdown: str
    metrics: Dict[str, float] = field(default_factory=dict)
    raw_metrics: Dict[str, float] = field(default_factory=dict)
    event_window_metrics: Dict[str, Any] = field(default_factory=dict)
    adjusted_channel_metrics: Dict[str, Any] = field(default_factory=dict)
    included_unit_metrics: Dict[str, Any] = field(default_factory=dict)
    focused_prediction: Dict[str, Any] = field(default_factory=dict)
    explanation_quality: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sum(np.abs(actual - pred)) / max(float(np.sum(np.abs(actual))), 1.0) * 100.0)


def smape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(actual) + np.abs(pred), 1.0)
    return float(np.mean(2.0 * np.abs(pred - actual) / denom) * 100.0)


def compute_metrics(actual: Optional[Sequence[Sequence[float]]], pred: Sequence[Sequence[float]]) -> Dict[str, float]:
    if actual is None:
        return {}
    a = np.asarray(actual, dtype=np.float32)
    p = np.asarray(pred, dtype=np.float32)
    h = min(a.shape[-1], p.shape[-1])
    c = min(a.shape[0], p.shape[0])
    if c == 0 or h == 0:
        return {}
    a = a[:c, :h]
    p = p[:c, :h]
    return {"mae": mae(a, p), "rmse": rmse(a, p), "wape": wape(a, p), "smape": smape(a, p)}


def ensure_paper_run_dir(root: str | Path) -> Path:
    root = Path(root)
    for sub in [
        "predictions",
        "models",
        "retrieval_cache",
        "figures",
        "explanations",
        "reports",
        "logs",
        "preferences",
        "tables",
    ]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def write_forecast_artifacts(result: ForecastResultSpec, run_dir: str | Path, stem: str) -> ForecastResultSpec:
    run_dir = Path(run_dir)
    ensure_paper_run_dir(run_dir)
    json_path = run_dir / "predictions" / f"{stem}.json"
    md_path = run_dir / "explanations" / f"{stem}.md"
    fig_path = run_dir / "figures" / f"{stem}_hourly_curves.png"
    focused_fig_path = run_dir / "figures" / f"{stem}_focused_adjusted_channels.png"
    result.artifacts.update({
        "prediction_json": str(json_path),
        "explanation_md": str(md_path),
        "hourly_curve_png": str(fig_path),
        "focused_hourly_curve_png": str(focused_fig_path),
    })
    md_path.write_text(result.explanation_markdown, encoding="utf-8")
    plot_hourly_curves(result, fig_path)
    plot_focused_adjusted_channels(result, focused_fig_path)
    write_focused_prediction_artifacts(result, run_dir, stem)
    json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _arrays_almost_equal(a: np.ndarray, b: np.ndarray, tol: float = 1e-4) -> bool:
    if a.shape != b.shape:
        return False
    return bool(np.max(np.abs(a - b)) <= tol) if a.size else True


def plot_hourly_curves(result: ForecastResultSpec, out_png: str | Path, max_channels: int = 4) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = np.asarray(result.numerical.raw_forecast, dtype=np.float32)
    adj = np.asarray(result.adjusted_forecast, dtype=np.float32)
    actual = np.asarray(result.numerical.ground_truth, dtype=np.float32) if result.numerical.ground_truth else None
    if raw.ndim != 2 or raw.size == 0:
        return
    if adj.shape != raw.shape:
        adj = raw.copy()
    totals = actual.sum(axis=1) if actual is not None and actual.ndim == 2 else raw.sum(axis=1)
    ranked = np.argsort(-totals)[:max_channels]
    event_offsets = _event_offsets_for_plot(result)
    overlap = _arrays_almost_equal(raw, adj)
    delta_matrix = adj - raw
    abs_delta = np.abs(delta_matrix)
    max_delta = float(np.max(abs_delta)) if raw.size else 0.0
    mean_delta = float(np.mean(abs_delta)) if raw.size else 0.0
    active_cells = int((abs_delta > 1e-6).sum()) if raw.size else 0
    nrows = max(1, len(ranked)) * 2
    height_ratios = []
    for _ in ranked:
        height_ratios.extend([2.4, 0.9])
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(13, max(4.4, 3.6 * len(ranked))),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if nrows == 2:
        axes = list(axes)
    for panel_i, idx in enumerate(ranked):
        ax = axes[2 * panel_i]
        delta_ax = axes[2 * panel_i + 1]
        x = np.arange(raw.shape[1])
        if actual is not None and idx < actual.shape[0]:
            ax.plot(actual[idx], label="actual", linewidth=1.4, color="#9ca3af", alpha=0.9, zorder=1)
        adjusted_label = "event-aware adjusted"
        if overlap:
            adjusted_label = "event-aware adjusted (same as raw)"
        ax.plot(adj[idx], label=adjusted_label, linewidth=1.6, color="#f97316", alpha=0.95, zorder=2)
        ax.plot(raw[idx], label="LP-MOMENT raw", linewidth=1.7, color="#2563eb", linestyle="--", alpha=0.98, zorder=4)
        delta = np.abs(adj[idx] - raw[idx])
        if float(delta.max()) > 1e-4:
            ax.fill_between(
                x,
                raw[idx],
                adj[idx],
                color="#7c3aed",
                alpha=0.12,
                label="bounded correction area (adjusted - raw)",
            )
        delta_ax.axhline(0.0, color="#6b7280", linewidth=0.8, alpha=0.8)
        delta_ax.plot(delta_matrix[idx], color="#7c3aed", linewidth=1.1, alpha=0.9)
        delta_ax.fill_between(x, 0.0, delta_matrix[idx], color="#7c3aed", alpha=0.18)
        delta_ax.set_ylabel("adj-raw", fontsize=8, color="#6d28d9")
        delta_ax.tick_params(axis="y", labelsize=7, colors="#6d28d9")
        delta_ax.grid(axis="y", alpha=0.16)
        delta_ax.spines["top"].set_visible(False)
        delta_ax.spines["right"].set_visible(False)
        if float(delta.max()) <= 1e-4:
            delta_ax.text(
                0.99,
                0.78,
                "raw == adjusted\ncorrection rejected or abstained",
                transform=delta_ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#1d4ed8",
                bbox={"facecolor": "white", "edgecolor": "#bfdbfe", "alpha": 0.86, "pad": 3},
            )
        for event_offset, event_label in event_offsets:
            if 0 <= event_offset < raw.shape[1]:
                ax.axvline(event_offset, color="#dc2626", linestyle="--", linewidth=0.9, alpha=0.75)
                delta_ax.axvline(event_offset, color="#dc2626", linestyle="--", linewidth=0.75, alpha=0.55)
                ymax = ax.get_ylim()[1]
                ax.text(
                    event_offset + 0.4,
                    ymax * 0.94,
                    event_label,
                    color="#991b1b",
                    fontsize=7,
                    rotation=90,
                    va="top",
                    ha="left",
                )
        name = result.numerical.channel_names[idx] if idx < len(result.numerical.channel_names) else f"channel_{idx}"
        title = name[:80]
        if actual is not None and idx < actual.shape[0]:
            raw_wape = wape(actual[idx : idx + 1], raw[idx : idx + 1])
            adj_wape = wape(actual[idx : idx + 1], adj[idx : idx + 1])
            title = f"{title} | WAPE raw {raw_wape:.1f}% -> adjusted {adj_wape:.1f}%"
        if overlap:
            title = f"{title} | no correction applied"
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_ylabel("flow")
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if overlap:
            ax.text(
                0.99,
                0.88,
                "raw == adjusted\n(abstain/no calibration)",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#1d4ed8",
                bbox={"facecolor": "white", "edgecolor": "#bfdbfe", "alpha": 0.86, "pad": 3},
                zorder=5,
            )
        delta_ax.text(
            0.01,
            0.78,
            f"max |delta|={float(delta.max()):.2f}, mean |delta|={float(delta.mean()):.2f}",
            transform=delta_ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#4c1d95",
        )
    axes[-1].set_xlabel("forecast horizon hour")
    handles, labels = axes[0].get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    axes[0].legend(dedup.values(), dedup.keys(), fontsize=8, ncol=min(4, len(dedup)))
    fig.suptitle(f"{result.request.mode} | {result.request.date}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    meta = {
        "raw_adjusted_overlap": overlap,
        "raw_adjusted_note": "raw == adjusted (abstain/no calibration)" if overlap else "raw and adjusted differ",
        "max_abs_adjusted_minus_raw": max_delta,
        "mean_abs_adjusted_minus_raw": mean_delta,
        "active_correction_cells": active_cells,
        "controller_allowed": bool(result.decision.controller_allowed),
        "abstain_reason": result.decision.reason if result.decision.abstain else "",
        "adjusted_channels": list(result.decision.adjusted_channels),
        "figure_checks": (
            ["raw == adjusted; no correction applied; abstention/no-calibration annotation shown"]
            if overlap
            else ["bounded correction area (adjusted - raw)", "independent adjusted-minus-raw delta subplot"]
        ),
    }
    Path(out_png).with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_focused_adjusted_channels(result: ForecastResultSpec, out_png: str | Path, max_channels: int = 4) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = np.asarray(result.numerical.raw_forecast, dtype=np.float32)
    adj = np.asarray(result.adjusted_forecast, dtype=np.float32)
    actual = np.asarray(result.numerical.ground_truth, dtype=np.float32) if result.numerical.ground_truth else None
    if raw.ndim != 2 or raw.size == 0:
        return
    if adj.shape != raw.shape:
        adj = raw.copy()
    channel_names = list(result.numerical.channel_names)
    wanted = set(result.decision.adjusted_channels or [])
    candidate_indices = [i for i, name in enumerate(channel_names) if name in wanted]
    if not candidate_indices:
        deltas = np.max(np.abs(adj - raw), axis=1)
        candidate_indices = [int(i) for i in np.argsort(-deltas)[:max_channels] if float(deltas[i]) > 1e-6]
    out_png = Path(out_png)
    if not candidate_indices:
        fig, ax = plt.subplots(1, 1, figsize=(9, 2.8))
        ax.axis("off")
        ax.text(
            0.5,
            0.55,
            "No adjusted channels\nraw == adjusted, correction rejected by controller",
            ha="center",
            va="center",
            fontsize=12,
            color="#1d4ed8",
        )
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        out_png.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "focused_channels": [],
                    "controller_allowed": bool(result.decision.controller_allowed),
                    "abstain_reason": result.decision.reason,
                    "raw_adjusted_overlap": True,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return
    candidate_indices = candidate_indices[:max_channels]
    fig, axes = plt.subplots(len(candidate_indices), 1, figsize=(13, max(3.2, 2.7 * len(candidate_indices))), sharex=True)
    if len(candidate_indices) == 1:
        axes = [axes]
    x = np.arange(raw.shape[1])
    for ax, idx in zip(axes, candidate_indices):
        if actual is not None and idx < actual.shape[0]:
            ax.plot(actual[idx], label="actual", linewidth=1.2, color="#9ca3af", alpha=0.85, zorder=1)
        ax.plot(adj[idx], label="event-aware adjusted", linewidth=1.5, color="#f97316", alpha=0.96, zorder=2)
        ax.plot(raw[idx], label="LP-MOMENT raw", linewidth=1.6, color="#2563eb", linestyle="--", alpha=0.98, zorder=4)
        delta = adj[idx] - raw[idx]
        ax2 = ax.twinx()
        ax2.plot(delta, label="adjusted - raw", color="#7c3aed", linewidth=0.95, alpha=0.75)
        ax2.axhline(0, color="#6b7280", linewidth=0.7, alpha=0.6)
        ax2.set_ylabel("delta", fontsize=8, color="#6d28d9")
        ax2.tick_params(axis="y", labelsize=7, colors="#6d28d9")
        name = channel_names[idx] if idx < len(channel_names) else f"channel_{idx}"
        title = f"{name[:80]} | max |delta|={float(np.max(np.abs(delta))):.2f}, mean |delta|={float(np.mean(np.abs(delta))):.2f}"
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_ylabel("flow")
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("forecast horizon hour")
    handles, labels = axes[0].get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    axes[0].legend(dedup.values(), dedup.keys(), fontsize=8, ncol=min(4, len(dedup)))
    fig.suptitle(f"Focused adjusted channels | {result.request.date}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    out_png.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "focused_channels": [channel_names[i] for i in candidate_indices],
                "controller_allowed": bool(result.decision.controller_allowed),
                "abstain_reason": result.decision.reason if result.decision.abstain else "",
                "max_abs_adjusted_minus_raw": float(np.max(np.abs(adj - raw))),
                "active_correction_cells": int((np.abs(adj - raw) > 1e-6).sum()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def write_focused_prediction_artifacts(result: ForecastResultSpec, run_dir: str | Path, stem: str) -> None:
    focused = result.focused_prediction or {}
    if not focused:
        return
    run_dir = Path(run_dir)
    tables_dir = run_dir / "tables"
    figures_dir = run_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    json_path = tables_dir / f"{stem}_focused_forecast_rows.json"
    csv_path = tables_dir / f"{stem}_focused_forecast_rows.csv"
    fig_path = figures_dir / f"{stem}_focused_station_time_forecast.png"
    summary_path = run_dir / "reports" / "focused_forecast_summary.json"
    rows = list(focused.get("focused_forecast_rows") or [])
    json_path.write_text(json.dumps(focused, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    plot_focused_station_time_forecast(focused, fig_path)
    summary = {
        "focused_request": focused.get("focused_request") or {},
        "focused_time_window": focused.get("focused_time_window") or {},
        "focused_resolved_channels": focused.get("focused_resolved_channels") or [],
        "focused_forecast_metrics": focused.get("focused_forecast_metrics") or {},
        "row_count": len(rows),
        "artifacts": {
            "focused_forecast_json": str(json_path),
            "focused_forecast_csv": str(csv_path),
            "focused_station_time_png": str(fig_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    result.artifacts.update(
        {
            "focused_forecast_json": str(json_path),
            "focused_forecast_csv": str(csv_path),
            "focused_station_time_png": str(fig_path),
            "focused_forecast_summary": str(summary_path),
        }
    )


def plot_focused_station_time_forecast(focused: Dict[str, Any], out_png: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(focused.get("focused_forecast_rows") or [])
    out_png = Path(out_png)
    if not rows:
        fig, ax = plt.subplots(1, 1, figsize=(8, 2.6))
        ax.axis("off")
        ax.text(0.5, 0.5, "No focused station-time rows selected", ha="center", va="center", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        return
    channels = []
    for row in rows:
        ch = row.get("channel_name")
        if ch not in channels:
            channels.append(ch)
    fig, axes = plt.subplots(len(channels), 1, figsize=(11.5, max(3.0, 2.4 * len(channels))), sharex=False)
    if len(channels) == 1:
        axes = [axes]
    for ax, ch in zip(axes, channels):
        ch_rows = [row for row in rows if row.get("channel_name") == ch]
        x = np.arange(len(ch_rows))
        labels = [str(row.get("timestamp", ""))[5:16] for row in ch_rows]
        raw = [float(row.get("raw_forecast", 0.0) or 0.0) for row in ch_rows]
        adjusted = [float(row.get("adjusted_forecast", 0.0) or 0.0) for row in ch_rows]
        actual_vals = [row.get("actual") for row in ch_rows]
        ax.plot(x, raw, label="LP-MOMENT raw", color="#2563eb", linestyle="--", linewidth=1.8)
        ax.plot(x, adjusted, label="event-aware adjusted", color="#f97316", linewidth=1.8)
        if any(v is not None for v in actual_vals):
            actual = [float(v) if v is not None else np.nan for v in actual_vals]
            ax.plot(x, actual, label="actual (post-hoc)", color="#9ca3af", linewidth=1.4)
        ax.bar(x, np.asarray(adjusted) - np.asarray(raw), label="adjusted - raw", color="#7c3aed", alpha=0.16)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(display_station_name(ch, ch_rows[0] if ch_rows else {}, max_len=80), fontsize=9, loc="left")
        ax.set_ylabel("flow")
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(fontsize=8, ncol=4)
    window = focused.get("focused_time_window") or {}
    fig.suptitle(f"Focused station-time forecast | {window.get('start', '')} to {window.get('end', '')}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _event_offsets_for_plot(result: ForecastResultSpec) -> List[tuple[int, str]]:
    try:
        start = datetime.fromisoformat(str(result.request.date).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return []
    offsets: List[tuple[int, str]] = []
    seen = set()
    for event in result.evidence.structured_events[:8]:
        when = event.get("event_time") or event.get("start_time") or event.get("date")
        if not when:
            continue
        try:
            event_time = datetime.fromisoformat(str(when).replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            continue
        offset = int(round((event_time - start).total_seconds() / 3600.0))
        title = str(event.get("title") or "event")[:18]
        key = (offset, title)
        if key in seen:
            continue
        seen.add(key)
        offsets.append((offset, title))
    return offsets


def _parse_residual_case(case: str) -> Dict[str, Any]:
    text = str(case or "")
    def match(pattern: str, default: str = "") -> str:
        m = re.search(pattern, text, flags=re.I)
        return m.group(1).strip() if m else default

    event_type = match(r"type=([^\]\n]+?)\s+day=", match(r"Event type:\s*([^|]+)", "unknown"))
    day_type = match(r"day=([^\]\n\s]+)", match(r"Day:\s*([^\n|]+)", "unknown"))
    rank = match(r"rank=([^\]\n\s]+)", match(r"Station rank:\s*([^|]+)", "unknown"))
    correction = match(r"correction=([+-]?\d+(?:\.\d+)?%)", match(r"median correction needed\s*=\s*([+-]?\d+(?:\.\d+)?%)", "unknown"))
    score_text = match(r"score=([0-9.]+)", "0")
    try:
        score = float(score_text)
    except ValueError:
        score = 0.0
    stats = ""
    stats_match = re.search(r"(Statistics\s*\([^)]*\):\s*[^\n]+)", text, flags=re.I)
    if stats_match:
        stats = stats_match.group(1).strip()
        baseline_start = stats.find(" Baseline excess")
        if baseline_start > 0:
            stats = stats[:baseline_start].strip()
    baseline = match(r"(Baseline excess[^\n]+)", "")
    if baseline:
        second = baseline.find(" Baseline excess", 1)
        if second > 0:
            baseline = baseline[:second].strip()
    direction = "neutral"
    if correction.startswith("+"):
        direction = "increase"
    elif correction.startswith("-"):
        direction = "decrease"
    key = "|".join([event_type.lower(), day_type.lower(), rank.lower(), direction, correction])
    return {
        "key": key,
        "event_type": event_type,
        "day_type": day_type,
        "rank": rank,
        "correction": correction,
        "score": score,
        "stats": stats,
        "baseline": baseline,
        "text": text,
    }


def split_residual_case_blocks(text_or_cases: Sequence[str] | str) -> List[str]:
    if isinstance(text_or_cases, str):
        texts = [text_or_cases]
    else:
        texts = [str(item or "") for item in text_or_cases]
    blocks: List[str] = []
    for text in texts:
        for chunk in re.split(r"(?=\[correction_rag=\d+)", text):
            chunk = chunk.strip()
            if chunk:
                blocks.append(chunk)
    return blocks


def deduplicate_residual_cases(cases: Sequence[str], max_cases: int = 3) -> List[str]:
    grouped: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for case in split_residual_case_blocks(cases):
        if not str(case or "").strip():
            continue
        parsed = _parse_residual_case(case)
        key = parsed["key"]
        if key not in grouped:
            grouped[key] = {**parsed, "merged_count": 1, "best_score": parsed["score"]}
            order.append(key)
            continue
        grouped[key]["merged_count"] += 1
        if parsed["score"] > grouped[key]["best_score"]:
            grouped[key].update({k: v for k, v in parsed.items() if k not in {"key"}})
            grouped[key]["best_score"] = parsed["score"]

    rows = sorted((grouped[k] for k in order), key=lambda item: item.get("best_score", 0.0), reverse=True)
    out: List[str] = []
    for item in rows[:max_cases]:
        parts = [
            f"[residual_case type={item['event_type']} day={item['day_type']} rank={item['rank']} correction={item['correction']} "
            f"best_score={item['best_score']:.3f} merged_count={item['merged_count']}]",
        ]
        if item.get("stats"):
            parts.append(str(item["stats"]))
        if item.get("baseline"):
            parts.append(str(item["baseline"]))
        out.append(" ".join(parts))
    return out


def extract_historical_event_cases(text_or_cases: Sequence[str] | str, max_cases: int = 3) -> List[Dict[str, Any]]:
    """Extract readable train/validation examples from residual-RAG documents."""
    blocks = split_residual_case_blocks(text_or_cases)
    out: List[Dict[str, Any]] = []
    seen = set()
    for block in blocks:
        event_type = re.search(r"Event type:\s*([^|\\n]+)", block, flags=re.I)
        rank = re.search(r"Station rank:\s*([^|\\n]+)", block, flags=re.I)
        day = re.search(r"Day:\s*([^|\\n]+)", block, flags=re.I)
        correction = re.search(r"median correction needed\s*=\s*([+-]?\d+(?:\.\d+)?%)", block, flags=re.I)
        examples = re.search(r"Example instances:\s*(.*)", block, flags=re.I | re.S)
        if not examples:
            continue
        for title, date, station, baseline_resid in re.findall(
            r'"([^"]+)"\s*\(([^,]+),\s*([^,]+),\s*bl_resid=([+-]?\d+(?:\.\d+)?%)\)',
            examples.group(1),
        ):
            key = (title, date, station)
            if key in seen:
                continue
            seen.add(key)
            direction = "increase" if baseline_resid.startswith("+") else "decrease"
            out.append(
                {
                    "split": "train_val_memory",
                    "historical_event_title": title,
                    "event_time": date,
                    "matched_station": station,
                    "event_type": event_type.group(1).strip() if event_type else "unknown",
                    "rank_group": rank.group(1).strip() if rank else "unknown",
                    "day_type": day.group(1).strip() if day else "unknown",
                    "residual_direction": direction,
                    "historical_baseline_residual": baseline_resid,
                    "median_lp_moment_correction": correction.group(1) if correction else "unknown",
                    "similarity_reason": "same residual-RAG event type, station-rank group, and day-type pattern from train/validation memory",
                }
            )
            if len(out) >= max_cases:
                return out
    return out


def _structured_event_score(event: Dict[str, Any]) -> float:
    text = " ".join(
        str(event.get(k, "") or "").lower()
        for k in ("title", "location", "venue_name", "event_type", "event_category")
    )
    score = 10.0 if str(event.get("impact_tier", "") or "").upper() == "A" else 0.0
    for term, weight in {
        "tsq live": 3.0,
        "plaza programming": 2.2,
        "times square": 1.8,
        "broadway": 1.0,
        "parade": 2.5,
        "festival": 2.0,
        "concert": 2.0,
        "marathon": 2.5,
    }.items():
        if term in text:
            score += weight
    for term, weight in {"cheese": -2.0, "promo": -1.5, "sampling": -1.5, "giveaway": -1.5}.items():
        if term in text:
            score += weight
    return score


def summarize_structured_events(events: Sequence[Dict[str, Any]], max_events: int = 8) -> List[str]:
    grouped: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for ev in events:
        base_location = str(ev.get("location") or ev.get("venue_name") or "").split("|")[0].strip()
        key = "|".join(
            [
                str(ev.get("title", "") or "")[:160],
                str(ev.get("event_time", "") or "")[:80],
                base_location[:160],
            ]
        )
        if key not in grouped:
            grouped[key] = {**ev, "_stations": [], "_score": _structured_event_score(ev)}
            order.append(key)
        station = ev.get("channel_name") or ev.get("station_complex_id") or ev.get("station_complex")
        if station and station not in grouped[key]["_stations"]:
            grouped[key]["_stations"].append(station)
        grouped[key]["_score"] = max(float(grouped[key].get("_score", 0.0)), _structured_event_score(ev))
    rows = sorted((grouped[k] for k in order), key=lambda item: (-float(item.get("_score", 0.0)), str(item.get("event_time", ""))))
    out: List[str] = []
    for ev in rows[:max_events]:
        title = ev.get("title") or "Untitled event"
        when = ev.get("event_time") or ""
        tier = ev.get("impact_tier") or ""
        stations = list(ev.get("_stations") or [])
        station_preview = ", ".join(display_station_name(x) for x in stations[:3]) or "(none)"
        location = str(ev.get("location") or ev.get("venue_name") or "")[:160]
        out.append(
            f"- **{title}** is scheduled for {when or 'an unspecified time'} near {location or 'the target area'}. "
            f"It is treated as tier {tier or 'unknown'} evidence and is linked to {len(stations)} station/channel unit(s), "
            f"including {station_preview}. (station_count={len(stations)})"
        )
    return out


def summarize_historical_event_cases(cases: Sequence[Dict[str, Any]], max_cases: int = 3) -> List[str]:
    out: List[str] = []
    for case in list(cases)[:max_cases]:
        title = case.get("historical_event_title") or case.get("title") or "Historical event"
        when = case.get("event_time") or case.get("date") or "unknown date"
        station = case.get("matched_station") or case.get("channel_name") or "unknown station"
        direction = case.get("residual_direction") or "unknown"
        baseline = case.get("historical_baseline_residual") or case.get("residual_magnitude") or "unknown"
        correction = case.get("median_lp_moment_correction") or "unknown"
        reason = case.get("similarity_reason") or "train/validation historical memory"
        out.append(
            f"- In {case.get('split', 'train_val')} memory, **{title}** ({when}) matched {display_station_name(station)}. "
            f"The historical residual direction was **{direction}** with residual {baseline}; "
            f"the median LP-MOMENT correction was {correction}. This case is used because {reason}. "
            f"(median LP-MOMENT correction={correction})"
        )
    return out


def summarize_residual_memory_skills(skills: Sequence[Dict[str, Any]], max_skills: int = 3) -> List[str]:
    out: List[str] = []
    for skill in list(skills)[:max_skills]:
        action = skill.get("explanation_action") or skill.get("calibration_action") or {}
        trigger = skill.get("trigger_condition") or {}
        select_by = action.get("select_cases_by") or []
        if isinstance(select_by, list):
            select_text = ", ".join(str(x) for x in select_by)
        else:
            select_text = str(select_by)
        out.append(
            f"- {skill.get('skill_id', 'residual_memory_skill')} | category={skill.get('skill_category', 'unknown')} | "
            f"trigger={json.dumps(trigger, ensure_ascii=False)} | policy={action.get('policy', 'organize_residual_memory')} | "
            f"select_cases_by={select_text or '(not specified)'} | reason={skill.get('reason', '')}"
        )
    return out


def _forecast_window_lines(request: ForecastRequestSpec) -> List[str]:
    try:
        start = datetime.fromisoformat(str(request.date).replace("Z", "+00:00")).replace(tzinfo=None)
        end = start + timedelta(hours=int(request.horizon))
        return [
            f"- Anchor time: {start.isoformat(sep=' ')}",
            f"- Horizon: {int(request.horizon)} hours",
            f"- Horizon end: {end.isoformat(sep=' ')}",
            "- Event policy: scheduled future events inside the horizon are allowed only if known at forecast time.",
            "- Source policy: post-event reports or sources published after the anchor are not forecast-time evidence.",
        ]
    except Exception:
        return [
            f"- Anchor time: {request.date}",
            f"- Horizon: {request.horizon} hours",
            "- Event policy: scheduled future events inside the horizon are allowed only if known at forecast time.",
        ]


def _fmt_metric(value: Any) -> str:
    try:
        number = float(value)
        if number != 0.0 and abs(number) < 0.001:
            return "<0.001" if number > 0 else ">-0.001"
        return f"{number:.3f}"
    except Exception:
        return "n/a"


def _metric_scope_line(label: str, payload: Dict[str, Any]) -> str:
    if not payload:
        return f"- {label}: not_applicable"
    status = payload.get("status", "ok")
    if status != "ok":
        detail = []
        if "channel_count" in payload:
            detail.append(f"channels={payload.get('channel_count')}")
        if "hour_count" in payload:
            detail.append(f"hours={payload.get('hour_count')}")
        extra = f" ({', '.join(detail)})" if detail else ""
        return f"- {label}: {status}{extra}"
    raw = payload.get("raw") or {}
    adjusted = payload.get("adjusted") or {}
    delta = payload.get("delta_raw_minus_adjusted") or {}
    return (
        f"- {label}: raw WAPE={_fmt_metric(raw.get('wape'))}, adjusted WAPE={_fmt_metric(adjusted.get('wape'))}, "
        f"raw MAE={_fmt_metric(raw.get('mae'))}, adjusted MAE={_fmt_metric(adjusted.get('mae'))}, "
        f"delta(raw-adjusted) WAPE={_fmt_metric(delta.get('wape'))}, "
        f"cells={payload.get('cell_count', 'n/a')}"
    )


def _local_metric_lines(
    raw_metrics: Dict[str, Any],
    adjusted_metrics: Dict[str, Any],
    event_window_metrics: Dict[str, Any],
    adjusted_channel_metrics: Dict[str, Any],
    included_unit_metrics: Dict[str, Any],
) -> List[str]:
    return [
        (
            f"- Overall 192h Raw WAPE: {_fmt_metric(raw_metrics.get('wape'))}; "
            f"Overall 192h Adjusted WAPE: {_fmt_metric(adjusted_metrics.get('wape'))}; "
            f"Overall 192h Raw MAE: {_fmt_metric(raw_metrics.get('mae'))}; "
            f"Overall 192h Adjusted MAE: {_fmt_metric(adjusted_metrics.get('mae'))}"
        ),
        _metric_scope_line("Event-window subset", event_window_metrics),
        _metric_scope_line("Adjusted-channel subset", adjusted_channel_metrics),
        _metric_scope_line("Included-unit subset", included_unit_metrics),
    ]


def _focused_prediction_markdown(focused: Dict[str, Any]) -> str:
    if not focused:
        return ""
    rows = list(focused.get("focused_forecast_rows") or [])
    window = focused.get("focused_time_window") or {}
    channels = focused.get("focused_resolved_channels") or []
    metrics = focused.get("focused_forecast_metrics") or {}
    channel_names = [display_station_name(row.get("channel_name", ""), row) for row in channels]
    channel_text = ", ".join(channel_names[:6]) or "(none)"
    controller_allowed = any(bool(row.get("controller_allowed")) for row in rows)
    has_event = bool(focused.get("focused_request", {}).get("has_focus_window_events", True))
    decision_text = "bounded correction is permitted by the controller" if controller_allowed else "the raw LP-MOMENT forecast is kept"

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("channel_name") or "unknown"), []).append(row)

    outlook_lines: List[str] = []
    max_abs_delta = 0.0
    for channel_name, ch_rows in list(grouped.items())[:6]:
        ordered = sorted(ch_rows, key=lambda item: str(item.get("timestamp", "")))
        if not ordered:
            continue
        start = ordered[0]
        end = ordered[-1]
        peak = max(ordered, key=lambda item: float(item.get("adjusted_forecast", item.get("raw_forecast", 0.0)) or 0.0))
        raw_total = sum(float(item.get("raw_forecast", 0.0) or 0.0) for item in ordered)
        adj_total = sum(float(item.get("adjusted_forecast", 0.0) or 0.0) for item in ordered)
        ch_max_delta = max(abs(float(item.get("correction", 0.0) or 0.0)) for item in ordered)
        max_abs_delta = max(max_abs_delta, ch_max_delta)
        direction = "higher than" if adj_total > raw_total else "lower than" if adj_total < raw_total else "the same as"
        station = display_station_name(channel_name, ordered[0])
        outlook_lines.append(
            "- **{station}** starts at {start_raw:.1f} riders/hour, peaks at {peak_adj:.1f} around {peak_time}, "
            "and ends at {end_adj:.1f}. Across the selected window, the adjusted total is {direction} raw LP-MOMENT "
            "by {delta_total:.1f} riders, with max hourly correction {max_delta:.2f}.".format(
                station=station,
                start_raw=float(start.get("raw_forecast", 0.0) or 0.0),
                peak_adj=float(peak.get("adjusted_forecast", peak.get("raw_forecast", 0.0)) or 0.0),
                peak_time=str(peak.get("timestamp", "")),
                end_adj=float(end.get("adjusted_forecast", end.get("raw_forecast", 0.0)) or 0.0),
                direction=direction,
                delta_total=abs(adj_total - raw_total),
                max_delta=ch_max_delta,
            )
        )
    if not outlook_lines:
        outlook_lines.append("- No station-hour rows were selected for the focused window.")

    relation_lines: List[str] = []
    for row in channels[:6]:
        station = display_station_name(row.get("channel_name", ""), row)
        reason = row.get("event_relevance_reason") or row.get("matched_by") or "selected by the focus-window event gate"
        relation_lines.append(f"- **{station}** is considered relevant because {reason}.")
    if not relation_lines:
        relation_lines.append("- No station/channel passed the focused event-station gate.")

    local_status = metrics.get("status", "not_available")
    raw_wape = _fmt_metric((metrics.get("raw") or {}).get("wape"))
    adj_wape = _fmt_metric((metrics.get("adjusted") or {}).get("wape"))
    raw_mae = _fmt_metric((metrics.get("raw") or {}).get("mae"))
    adj_mae = _fmt_metric((metrics.get("adjusted") or {}).get("mae"))
    negligible_note = ""
    try:
        wape_delta = abs(float((metrics.get("raw") or {}).get("wape", 0.0)) - float((metrics.get("adjusted") or {}).get("wape", 0.0)))
    except Exception:
        wape_delta = 0.0
    if max_abs_delta < 1.0 or wape_delta < 0.001:
        negligible_note = (
            "\n- The controller allowed the decision evidence-wise, but the numerical correction is negligible "
            f"within this local window (max hourly delta {max_abs_delta:.2f} riders/hour)."
        )

    excluded_count = int(focused.get("focused_request", {}).get("excluded_channel_count", 0) or 0)
    if excluded_count:
        excluded_text = f"{excluded_count} channel(s) were outside the focused event-station mask or lacked enough audit support."
    else:
        excluded_text = "Channels not listed in this focused answer are kept at the raw LP-MOMENT forecast unless separately selected by the event-station mask."

    lines = [
        "## 8. Operational Answer for the Focus Window",
        "",
        "The full 192-hour LP-MOMENT forecast remains stored in JSON/CSV. This section only answers the traffic operator's focused question for the selected station-time window.",
        "",
        "### Operator question",
        "",
        f"The traffic operator asked for {channel_text} from {window.get('start', 'n/a')} to {window.get('end', 'n/a')} "
        f"(centered at {window.get('center', 'n/a')}). The system should report whether event-aware calibration is justified for this local operating window.",
        "",
        "### Short answer",
        "",
        f"The focused-window answer is: {decision_text}. The event gate is {'active' if has_event else 'inactive'} for this window, "
        f"and the local metric status is `{local_status}`.{negligible_note}",
        "",
        "### Local ridership outlook",
        "",
        *outlook_lines,
        "",
        "### Why this station/channel is relevant",
        "",
        *relation_lines,
        "",
        "### Residual-memory support",
        "",
        "Historical train/validation residual memory is used as context for the correction direction and uncertainty; the complete matched cases remain in the Historical Residual Memory section above.",
        "",
        "### Correction and risk control",
        "",
        f"The controller uses the predefined correction bound and event-station mask before changing any station-hour value. In this focused window, the maximum hourly correction is {max_abs_delta:.2f} riders/hour.",
        "",
        "### Why other channels are excluded",
        "",
        excluded_text,
        "",
        "### Post-hoc local evaluation",
        "",
        f"- Local metric status: {local_status}",
        f"- Local raw WAPE: {raw_wape}",
        f"- Local adjusted WAPE: {adj_wape}",
        f"- Local raw MAE: {raw_mae}",
        f"- Local adjusted MAE: {adj_mae}",
    ]
    return "\n".join(lines).rstrip()


def _normalize_explanation_reasoning(text: str) -> str:
    """Render section 8 as five human-readable explanation blocks."""
    clean = (text or "").strip().replace("LP-MOMENT", "PT-MOMENT").replace("LP MOMENT", "PT-MOMENT")
    labels = ["Factual evidence", "Residual analogues", "Calibration rationale", "Uncertainty", "Risk control"]
    if not clean:
        clean = "No additional local-LLM reasoning was available for this case."
    for label in labels:
        clean = re.sub(rf"(?im)^\s*#+\s*{re.escape(label)}\s*$", f"### {label}", clean)
        clean = re.sub(rf"(?im)^\s*{re.escape(label)}\s*:\s*", f"### {label}\n", clean)
    if all(f"### {label}" in clean for label in labels):
        if "why this event is related" not in clean.lower():
            clean = clean.replace(
                "### Factual evidence\n",
                "### Factual evidence\n"
                "This block explains why this event is related to the included station/channel units.\n",
                1,
            )
        return clean
    return "\n\n".join(
        [
            "### Factual evidence\n"
            "The forecast-time evidence consists of structured event records, Qwen-Plus model-assisted summaries when available, and the Evidence Auditor scores shown above; this block explains why this event is related to the included station/channel units.",
            "### Residual analogues\n"
            "The historical residual memory section lists train/validation cases used to reason about whether similar events tended to require upward, downward, or no correction.",
            "### Calibration rationale\n"
            + clean,
            "### Uncertainty\n"
            "Uncertainty is handled by requiring station-event alignment, residual support, and controller approval before applying any bounded correction.",
            "### Risk control\n"
            "Channels outside the event-station mask remain at the PT-MOMENT raw forecast, and the correction magnitude is capped by the predefined bound.",
        ]
    )


def build_explanation_markdown(
    request: ForecastRequestSpec,
    evidence: EventEvidenceSpec,
    decision: CalibrationDecisionSpec,
    metrics: Dict[str, float],
    model_explanation: str,
    llm_markdown: str = "",
    numerical: Optional[NumericalForecastSpec] = None,
    raw_metrics: Optional[Dict[str, float]] = None,
    event_window_metrics: Optional[Dict[str, Any]] = None,
    adjusted_channel_metrics: Optional[Dict[str, Any]] = None,
    included_unit_metrics: Optional[Dict[str, Any]] = None,
    focused_prediction: Optional[Dict[str, Any]] = None,
    adapter_checkpoint: str = "",
) -> str:
    event_lines = summarize_structured_events(evidence.structured_events, max_events=8) or ["- No structured event record available."]
    historical_lines = summarize_historical_event_cases(evidence.historical_event_cases, max_cases=5) or ["- No train/validation historical event case was attached."]
    skill_lines = summarize_residual_memory_skills(evidence.selected_residual_memory_skills, max_skills=3) or ["- No validation-promoted residual-memory skill was attached."]
    residual_lines = [f"- {case}" for case in deduplicate_residual_cases(evidence.local_residual_cases, max_cases=5)] or ["- No local residual RAG case was attached."]
    summary_lines = []
    for item in evidence.model_assisted_summaries[:4]:
        summary = str(item.get("summary", "") or "").strip()
        if summary:
            summary_lines.append(f"- {summary[:500]}")
    if not summary_lines:
        summary_lines.append("- No model-assisted evidence summary available.")
    audit = evidence.evidence_audit or {}
    included_units = decision.included_units or audit.get("included_units") or []
    excluded_units = decision.excluded_units or audit.get("excluded_units") or []
    audit_score_lines = [
        f"- source_validity_score: {float(audit.get('source_validity_score', 0.0) or 0.0):.3f}",
        f"- geo_consistency_score: {float(audit.get('geo_consistency_score', 0.0) or 0.0):.3f}",
        f"- temporal_alignment_score: {float(audit.get('temporal_alignment_score', 0.0) or 0.0):.3f}",
        f"- semantic_consistency_score: {float(audit.get('semantic_consistency_score', 0.0) or 0.0):.3f}",
        f"- residual_support_score: {float(audit.get('residual_support_score', 0.0) or 0.0):.3f}",
        f"- evidence_validity_score: {float(audit.get('evidence_validity_score', 0.0) or 0.0):.3f}",
        f"- conflict_flags: {', '.join(audit.get('conflict_flags') or []) or '(none)'}",
        f"- severe_conflict_flags: {', '.join(audit.get('severe_conflict_flags') or []) or '(none)'}",
    ]
    def _unit_lines(rows, included=True):
        if not rows:
            return ["- (none)"]
        out = []
        for row in rows[:12]:
            ch = row.get("station_channel") or row.get("channel") or row.get("station") or "unknown"
            if included:
                out.append(
                    f"- **{display_station_name(ch)}** is included because {row.get('reason','the evidence audit linked it to the event')}. "
                    f"Relation: {row.get('relation','event-station match')}; gate score: {row.get('gate_score','n/a')}."
                )
            else:
                out.append(
                    f"- **{display_station_name(ch)}** is excluded because {row.get('exclusion_reason','the audit/controller did not find enough support')}."
                )
        return out
    raw_metric_text = json.dumps(raw_metrics or {}, ensure_ascii=False)
    adjusted_metric_text = json.dumps(metrics or {}, ensure_ascii=False)
    metric_text = json.dumps(metrics, ensure_ascii=False, indent=2) if metrics else "Ground truth unavailable."
    raw_summary = "not available"
    if numerical and numerical.raw_forecast:
        raw = np.asarray(numerical.raw_forecast, dtype=np.float32)
        if raw.size:
            raw_summary = f"channels={raw.shape[0]}, horizon={raw.shape[1]}, mean={float(raw.mean()):.2f}, max={float(raw.max()):.2f}"
    raw_model_label = ((numerical.model_type if numerical else "") or "PT-MOMENT").replace("LP-MOMENT", "PT-MOMENT")
    correction_stats = decision.correction_stats or {}
    correction_stats_text = json.dumps(correction_stats, ensure_ascii=False)
    local_metric_lines = _local_metric_lines(
        raw_metrics=raw_metrics or {},
        adjusted_metrics=metrics or {},
        event_window_metrics=event_window_metrics or {},
        adjusted_channel_metrics=adjusted_channel_metrics or {},
        included_unit_metrics=included_unit_metrics or {},
    )
    # The focused station-time rows remain available in CSV/JSON artifacts, but
    # the paper-facing explanation follows the earlier audited 8-section layout.
    focused_section = ""
    explanation_heading = "## 8. Explanation"
    reasoning = _normalize_explanation_reasoning(llm_markdown.strip() if llm_markdown else model_explanation)
    return f"""# Event-Aware Forecast Explanation

## 1. Forecast Request

- Anchor time: {request.date}
- Forecast horizon: {request.horizon} hours
- Mode: {request.mode}
- Station scope: {request.station_scope}
- Retrieval policy: {request.retrieval_policy}
- Calibration enabled: {not decision.abstain}

## 2. Raw Forecast

- Raw model: {raw_model_label}
- Raw metrics: {raw_metric_text}
- Raw forecast summary: {raw_summary}

## 3. Forecast-time Evidence Audit

Major event detected: {evidence.has_major_event}

### Forecast Window

{chr(10).join(_forecast_window_lines(request))}

### Structured Future Events within Horizon

{chr(10).join(event_lines)}

### Model-Assisted Summary (Non-citable)

{chr(10).join(summary_lines)}

### Geo-temporal consistency

{chr(10).join(audit_score_lines)}

## 4. Candidate Event-Station-Channel Units

### Included

{chr(10).join(_unit_lines(included_units, included=True))}

### Excluded

{chr(10).join(_unit_lines(excluded_units, included=False))}

## 5. Historical Residual Memory

### Historical Event Memory from Train/Validation

{chr(10).join(historical_lines)}

### Residual-Memory Skill Guidance

{chr(10).join(skill_lines)}

### Residual Pattern Evidence

{chr(10).join(residual_lines)}

### Residual support summary

```json
{json.dumps(audit.get('residual_support_summary') or {}, ensure_ascii=False, indent=2)}
```

## 6. Calibration Controller

### Calibration Decision

- Calibration decision: {'allow bounded correction' if decision.controller_allowed and not decision.abstain else 'abstain / keep raw PT-MOMENT'}
- Confidence: {decision.confidence:.2f}
- Correction bound: {decision.correction_bound:.3f}
- Max correction: {float(correction_stats.get('max_abs_correction', 0.0) or 0.0):.4f}
- Adjusted station/channel units: {', '.join(display_station_name(x) for x in decision.adjusted_channels) or '(none)'}
- Excluded station/channel units: {len(excluded_units)}
- Abstention reason if any: {decision.reason if decision.abstain else '(none)'}
- Controller allowed: {decision.controller_allowed}

## 7. Adjusted Forecast

- Adjusted model: {request.mode}
- Adapter checkpoint: {adapter_checkpoint or '(none recorded)'}
{chr(10).join(local_metric_lines)}
- Correction magnitude statistics: {correction_stats_text}

{focused_section}

{explanation_heading}

{reasoning}

## Post-hoc Metrics

```json
{metric_text}
```
"""
