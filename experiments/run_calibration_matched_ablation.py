#!/usr/bin/env python3
"""Matched calibration ablation for EAF-MAS controller variants.

This experiment keeps the numerical backbone fixed.  It recomputes the same
test anchors, raw PT-MOMENT forecasts, adapter proposals, and actual arrays for
all controller variants, then evaluates only forecast-time controller choices.
It does not call LLMs, update skills, or tune on the test split.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODES: List[str] = [
    "pt_moment",
    "adapter_full_controller",
    "adapter_no_event_gate",
    "adapter_no_residual_support",
    "adapter_no_bound",
    "adapter_no_audit_guard",
    "adapter_always_apply",
    "adapter_random_gate",
]

METRIC_DIRECTIONS: Dict[str, str] = {
    "top128_wape": "lower_is_better",
    "top128_mae": "lower_is_better",
    "top128_rmse": "lower_is_better",
    "event_venue28_wape": "lower_is_better",
    "event_venue28_mae": "lower_is_better",
    "event_active_wape": "lower_is_better",
    "event_active_mae": "lower_is_better",
    "non_event_wape": "lower_is_better",
    "non_event_mae": "lower_is_better",
    "event_active_gain": "higher_is_better",
    "positive_gain_rate": "higher_is_better",
    "harmful_anchor_rate": "lower_is_better",
    "worst_regret": "lower_is_better",
    "max_abs_correction": "lower_is_safer",
    "mean_abs_correction": "lower_is_safer",
    "active_correction_cells": "diagnostic",
    "bound_clipped_cells": "diagnostic",
}


@dataclass(frozen=True)
class ControllerThresholds:
    source_validity: float = 0.20
    geo_consistency: float = 0.20
    temporal_alignment: float = 0.50
    evidence_validity: float = 0.20
    residual_support: float = 0.20


def ensure_dirs(root: Path) -> None:
    for sub in ("predictions", "tables", "figures", "reports", "logs"):
        (root / sub).mkdir(parents=True, exist_ok=True)


def _as_event_dict(event) -> dict:
    if hasattr(event, "model_dump"):
        return event.model_dump()
    if isinstance(event, dict):
        return dict(event)
    return {
        key: getattr(event, key)
        for key in dir(event)
        if not key.startswith("_") and not callable(getattr(event, key))
    }


def _event_timestamp(event) -> pd.Timestamp | None:
    row = _as_event_dict(event)
    raw = row.get("event_time") or row.get("start_time") or row.get("date")
    if not raw:
        return None
    try:
        return pd.Timestamp(raw)
    except Exception:
        return None


def prepare_timed_events(events: Iterable[object]) -> List[Tuple[dict, pd.Timestamp]]:
    rows: List[Tuple[dict, pd.Timestamp]] = []
    for event in events or []:
        row = _as_event_dict(event)
        ts = _event_timestamp(row)
        if ts is not None:
            rows.append((row, ts))
    return rows


def filter_events_for_window(
    timed_events: Sequence[Tuple[dict, pd.Timestamp]],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    back_hours: float = 72.0,
    forward_hours: float = 2.0,
) -> List[dict]:
    start = pd.Timestamp(window_start) - pd.Timedelta(hours=float(back_hours))
    end = pd.Timestamp(window_end) + pd.Timedelta(hours=float(forward_hours))
    return [event for event, event_time in timed_events if start <= event_time <= end]


def _nanmean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def _wape(actual: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if mask is None:
        selected = np.ones(actual.shape, dtype=bool)
    else:
        selected = np.asarray(mask, dtype=bool)
    if selected.shape != actual.shape:
        selected = np.broadcast_to(selected, actual.shape)
    if int(selected.sum()) == 0:
        return float("nan")
    denom = float(np.abs(actual[selected]).sum())
    if denom <= 1e-9:
        return float("nan")
    return float(np.abs(actual[selected] - pred[selected]).sum() / denom * 100.0)


def _mae(actual: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    selected = np.ones(actual.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if selected.shape != actual.shape:
        selected = np.broadcast_to(selected, actual.shape)
    return float(np.abs(actual[selected] - pred[selected]).mean()) if int(selected.sum()) else float("nan")


def _rmse(actual: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    selected = np.ones(actual.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if selected.shape != actual.shape:
        selected = np.broadcast_to(selected, actual.shape)
    return float(np.sqrt(np.square(actual[selected] - pred[selected]).mean())) if int(selected.sum()) else float("nan")


def metric_bundle(actual: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> dict:
    selected = np.ones(actual.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if selected.shape != actual.shape:
        selected = np.broadcast_to(selected, actual.shape)
    return {
        "wape": _wape(actual, pred, selected),
        "mae": _mae(actual, pred, selected),
        "rmse": _rmse(actual, pred, selected),
        "n": int(selected.sum()),
    }


def compute_controller_scores(features: np.ndarray) -> Dict[str, np.ndarray]:
    """Derive forecast-time controller scores from event features only.

    The score proxy intentionally avoids actual values.  It is a diagnostic
    controller layer over the existing event feature cube, not a learned judge.
    """
    from agents.event_adapter import FEATURE_NAMES

    idx = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    event_active = np.clip(features[..., idx["event_active"]], 0.0, 1.0)
    confidence = np.clip(features[..., idx["confidence_score"]], 0.0, 1.0)
    tier = np.clip(features[..., idx["tier_score"]], 0.0, 1.0)
    station_match = np.clip(features[..., idx["station_match"]], 0.0, 1.0)
    distance = np.clip(features[..., idx["distance_score"]], 0.0, 1.0)
    spillover = np.clip(features[..., idx["spillover_strength"]], 0.0, 1.0)
    direction = np.clip(np.abs(features[..., idx["expected_direction"]]), 0.0, 1.0)

    source = confidence
    geo = np.maximum(station_match, distance)
    temporal = np.maximum(event_active, spillover)
    semantic = np.maximum(tier, direction)
    residual = np.clip(0.40 * tier + 0.25 * confidence + 0.20 * station_match + 0.15 * spillover, 0.0, 1.0)
    evidence = np.clip((source + geo + temporal + semantic) / 4.0, 0.0, 1.0)
    return {
        "source_validity_score": source.astype(np.float32),
        "geo_consistency_score": geo.astype(np.float32),
        "temporal_alignment_score": temporal.astype(np.float32),
        "semantic_consistency_score": semantic.astype(np.float32),
        "residual_support_score": residual.astype(np.float32),
        "evidence_validity_score": evidence.astype(np.float32),
    }


def _audit_pass(scores: Mapping[str, np.ndarray], thresholds: ControllerThresholds) -> np.ndarray:
    return (
        (scores["source_validity_score"] >= thresholds.source_validity)
        & (scores["geo_consistency_score"] >= thresholds.geo_consistency)
        & (scores["temporal_alignment_score"] >= thresholds.temporal_alignment)
        & (scores["evidence_validity_score"] >= thresholds.evidence_validity)
    )


def _residual_pass(scores: Mapping[str, np.ndarray], thresholds: ControllerThresholds) -> np.ndarray:
    return scores["residual_support_score"] >= thresholds.residual_support


def build_mode_correction(
    *,
    mode: str,
    proposal_correction: np.ndarray,
    no_gate_proposal_correction: np.ndarray,
    unbounded_correction: np.ndarray,
    event_mask: np.ndarray,
    scores: Mapping[str, np.ndarray],
    thresholds: ControllerThresholds = ControllerThresholds(),
    residual_bound: float = 0.05,
    random_seed: int = 13,
    anchor_index: int = 0,
) -> Tuple[np.ndarray, dict]:
    """Return a correction matrix for a controller ablation mode.

    This helper deliberately has no ``actual`` argument: controller decisions
    are forecast-time choices and evaluation happens after the fact.
    """
    if mode not in MODES:
        raise ValueError(f"unknown calibration ablation mode: {mode}")

    proposal = np.asarray(proposal_correction, dtype=np.float32)
    no_gate = np.asarray(no_gate_proposal_correction, dtype=np.float32)
    unbounded = np.asarray(unbounded_correction, dtype=np.float32)
    event_gate = np.asarray(event_mask, dtype=bool)
    audit = _audit_pass(scores, thresholds)
    residual = _residual_pass(scores, thresholds)
    full_allowed = event_gate & audit & residual

    if mode == "pt_moment":
        allowed = np.zeros_like(event_gate, dtype=bool)
        source = proposal
        correction = np.zeros_like(proposal, dtype=np.float32)
    elif mode == "adapter_full_controller":
        allowed = full_allowed
        source = proposal
        correction = np.where(allowed, np.clip(source, -residual_bound, residual_bound), 0.0)
    elif mode == "adapter_no_event_gate":
        allowed = audit & residual
        source = no_gate
        correction = np.where(allowed, np.clip(source, -residual_bound, residual_bound), 0.0)
    elif mode == "adapter_no_residual_support":
        allowed = event_gate & audit
        source = proposal
        correction = np.where(allowed, np.clip(source, -residual_bound, residual_bound), 0.0)
    elif mode == "adapter_no_audit_guard":
        allowed = event_gate & residual
        source = proposal
        correction = np.where(allowed, np.clip(source, -residual_bound, residual_bound), 0.0)
    elif mode == "adapter_no_bound":
        allowed = full_allowed
        source = unbounded
        correction = np.where(allowed, source, 0.0)
    elif mode == "adapter_always_apply":
        allowed = np.ones_like(event_gate, dtype=bool)
        source = no_gate
        correction = np.clip(source, -residual_bound, residual_bound)
    elif mode == "adapter_random_gate":
        source = no_gate
        rng = np.random.default_rng(int(random_seed) + int(anchor_index))
        flat = np.arange(event_gate.size)
        count = int(full_allowed.sum())
        mask = np.zeros(event_gate.size, dtype=bool)
        if count > 0:
            chosen = rng.choice(flat, size=min(count, event_gate.size), replace=False)
            mask[chosen] = True
        allowed = mask.reshape(event_gate.shape)
        correction = np.where(allowed, np.clip(source, -residual_bound, residual_bound), 0.0)

    correction = np.asarray(correction, dtype=np.float32)
    active = np.abs(correction) > 1e-8
    bound_basis = unbounded if mode in {"adapter_full_controller", "adapter_no_residual_support", "adapter_no_audit_guard", "adapter_no_bound"} else source
    bound_clipped = (np.abs(bound_basis) > float(residual_bound) + 1e-8) & np.asarray(allowed, dtype=bool)
    meta = {
        "allowed_cells": int(np.asarray(allowed, dtype=bool).sum()),
        "active_correction_cells": int(active.sum()),
        "bound_clipped_cells": int(bound_clipped.sum()),
        "max_abs_correction": float(np.max(np.abs(correction))) if correction.size else 0.0,
        "mean_abs_correction": float(np.mean(np.abs(correction[active]))) if int(active.sum()) else 0.0,
    }
    return correction, meta


def evaluate_anchor(
    *,
    split: str,
    anchor: int,
    date: str,
    mode: str,
    raw: np.ndarray,
    adjusted: np.ndarray,
    actual: np.ndarray,
    correction: np.ndarray,
    event_mask: np.ndarray,
    venue_indices: Sequence[int],
    correction_meta: Mapping[str, float],
) -> dict:
    venue_mask = np.zeros_like(actual, dtype=bool)
    for idx in venue_indices:
        if 0 <= int(idx) < venue_mask.shape[0]:
            venue_mask[int(idx), :] = True
    non_event_mask = ~np.asarray(event_mask, dtype=bool)
    top = metric_bundle(actual, adjusted)
    venue = metric_bundle(actual, adjusted, venue_mask)
    event = metric_bundle(actual, adjusted, event_mask)
    non_event = metric_bundle(actual, adjusted, non_event_mask)
    raw_event = _wape(actual, raw, event_mask)
    raw_top = _wape(actual, raw)
    raw_non_event = _wape(actual, raw, non_event_mask)
    return {
        "split": split,
        "anchor": int(anchor),
        "date": str(date),
        "mode": mode,
        "top128_wape": top["wape"],
        "top128_mae": top["mae"],
        "top128_rmse": top["rmse"],
        "event_venue28_wape": venue["wape"],
        "event_venue28_mae": venue["mae"],
        "event_active_wape": event["wape"],
        "event_active_mae": event["mae"],
        "non_event_wape": non_event["wape"],
        "non_event_mae": non_event["mae"],
        "event_active_n": event["n"],
        "non_event_n": non_event["n"],
        "event_active_gain": raw_event - event["wape"] if np.isfinite(raw_event) and np.isfinite(event["wape"]) else float("nan"),
        "top128_gain": raw_top - top["wape"] if np.isfinite(raw_top) and np.isfinite(top["wape"]) else float("nan"),
        "non_event_degradation": non_event["wape"] - raw_non_event if np.isfinite(raw_non_event) and np.isfinite(non_event["wape"]) else float("nan"),
        "max_abs_correction": correction_meta.get("max_abs_correction", 0.0),
        "mean_abs_correction": correction_meta.get("mean_abs_correction", 0.0),
        "active_correction_cells": int(correction_meta.get("active_correction_cells", 0)),
        "bound_clipped_cells": int(correction_meta.get("bound_clipped_cells", 0)),
    }


def summarize_rows(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for mode, group in rows.groupby("mode", sort=False):
        valid_gain = group["event_active_gain"].replace([np.inf, -np.inf], np.nan).dropna()
        active_cells = group["active_correction_cells"].sum()
        if active_cells:
            mean_abs = float((group["mean_abs_correction"] * group["active_correction_cells"]).sum() / active_cells)
        else:
            mean_abs = 0.0
        worst_regret = float(np.maximum(-valid_gain, 0.0).max()) if len(valid_gain) else float("nan")
        out.append(
            {
                "mode": mode,
                "anchor_count": int(group["anchor"].nunique()),
                "top128_wape": _nanmean(group["top128_wape"]),
                "top128_mae": _nanmean(group["top128_mae"]),
                "top128_rmse": _nanmean(group["top128_rmse"]),
                "event_venue28_wape": _nanmean(group["event_venue28_wape"]),
                "event_venue28_mae": _nanmean(group["event_venue28_mae"]),
                "event_active_wape": _nanmean(group["event_active_wape"]),
                "event_active_mae": _nanmean(group["event_active_mae"]),
                "non_event_wape": _nanmean(group["non_event_wape"]),
                "non_event_mae": _nanmean(group["non_event_mae"]),
                "event_active_n": int(group["event_active_n"].sum()),
                "non_event_n": int(group["non_event_n"].sum()),
                "event_active_gain": _nanmean(group["event_active_gain"]),
                "top128_gain": _nanmean(group["top128_gain"]),
                "non_event_degradation": _nanmean(group["non_event_degradation"]),
                "max_abs_correction": float(group["max_abs_correction"].max()),
                "mean_abs_correction": mean_abs,
                "active_correction_cells": int(active_cells),
                "bound_clipped_cells": int(group["bound_clipped_cells"].sum()),
                "positive_gain_rate": float((valid_gain > 0).mean()) if len(valid_gain) else float("nan"),
                "harmful_anchor_rate": float((valid_gain < 0).mean()) if len(valid_gain) else float("nan"),
                "worst_regret": worst_regret,
            }
        )
    return pd.DataFrame(out)


def bootstrap_gain_ci(rows: pd.DataFrame, samples: int = 2000, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for mode, group in rows.groupby("mode", sort=False):
        values = group["event_active_gain"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float64)
        if len(values) == 0:
            out.append({"mode": mode, "mean_event_active_gain": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0})
            continue
        boots = []
        for _ in range(int(samples)):
            idx = rng.integers(0, len(values), len(values))
            boots.append(float(np.mean(values[idx])))
        out.append(
            {
                "mode": mode,
                "mean_event_active_gain": float(values.mean()),
                "ci_low": float(np.percentile(boots, 2.5)),
                "ci_high": float(np.percentile(boots, 97.5)),
                "n": int(len(values)),
            }
        )
    return pd.DataFrame(out)


def write_metric_directions(root: Path) -> pd.DataFrame:
    df = pd.DataFrame([{"metric": metric, "direction": direction} for metric, direction in METRIC_DIRECTIONS.items()])
    df.to_csv(root / "tables" / "calibration_ablation_metric_directions.csv", index=False)
    return df


def write_latex(summary: pd.DataFrame, root: Path) -> None:
    cols = [
        "mode",
        "top128_wape",
        "event_venue28_wape",
        "event_active_wape",
        "non_event_wape",
        "event_active_gain",
        "positive_gain_rate",
        "harmful_anchor_rate",
        "worst_regret",
        "max_abs_correction",
    ]
    latex = summary[cols].to_latex(index=False, float_format=lambda x: f"{x:.4f}", escape=True)
    (root / "tables" / "calibration_ablation_summary.tex").write_text(latex, encoding="utf-8")


def _clean_label(mode: str) -> str:
    return {
        "pt_moment": "PT-MOMENT",
        "adapter_full_controller": "Full controller",
        "adapter_no_event_gate": "No event gate",
        "adapter_no_residual_support": "No residual support",
        "adapter_no_bound": "No bound",
        "adapter_no_audit_guard": "No audit guard",
        "adapter_always_apply": "Always apply",
        "adapter_random_gate": "Random gate",
    }.get(mode, mode)


def write_figures(summary: pd.DataFrame, ci: pd.DataFrame, root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [_clean_label(m) for m in summary["mode"]]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(12, 5.2))
    width = 0.21
    for offset, col, name in [
        (-1.5 * width, "top128_wape", "Top128"),
        (-0.5 * width, "event_venue28_wape", "venue28"),
        (0.5 * width, "event_active_wape", "event-active"),
        (1.5 * width, "non_event_wape", "non-event"),
    ]:
        ax.bar(x + offset, summary[col], width=width, label=name)
    ax.set_ylabel("WAPE (lower is better)")
    ax.set_title("Calibration matched ablation: WAPE by evaluation subset")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend(frameon=False, ncol=4)
    ax.text(
        0.01,
        0.98,
        "PT-MOMENT is the raw numerical backbone; ablations are diagnostic variants.",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(root / "figures" / "calibration_ablation_wape_bar.png", dpi=220)
    fig.savefig(root / "figures" / "calibration_ablation_wape_bar.pdf")
    plt.close(fig)

    ci = ci.set_index("mode").reindex(summary["mode"]).reset_index()
    y = np.arange(len(ci))
    means = ci["mean_event_active_gain"].to_numpy(dtype=float)
    lows = ci["ci_low"].to_numpy(dtype=float)
    highs = ci["ci_high"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axvline(0.0, color="#333333", linewidth=1.0)
    ax.errorbar(means, y, xerr=[means - lows, highs - means], fmt="o", color="#1f77b4", ecolor="#8bb7df", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels([_clean_label(m) for m in ci["mode"]])
    ax.set_xlabel("Event-active WAPE gain vs PT-MOMENT (higher is better)")
    ax.set_title("Event-active paired gain with bootstrap 95% CI")
    ax.text(
        0.01,
        0.02,
        "Full controller is the proposed bounded calibration; variants diagnose removed safeguards.",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(root / "figures" / "calibration_ablation_event_active_gain_forest.png", dpi=220)
    fig.savefig(root / "figures" / "calibration_ablation_event_active_gain_forest.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    panels = [
        ("harmful_anchor_rate", "Harmful anchor rate", "lower is safer"),
        ("worst_regret", "Worst event-active regret", "lower is safer"),
        ("non_event_degradation", "Non-event WAPE degradation", "lower is safer"),
        ("max_abs_correction", "Max |correction|", "bounded risk"),
    ]
    for ax, (col, title, ylabel) in zip(axes.ravel(), panels):
        ax.bar(x, summary[col], color="#c55a11" if col in {"harmful_anchor_rate", "worst_regret"} else "#4472c4")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        if col == "max_abs_correction":
            ax.axhline(0.05, color="#444444", linestyle="--", linewidth=1.0, label="nominal bound")
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Calibration safety and regret diagnostics")
    fig.tight_layout()
    fig.savefig(root / "figures" / "calibration_ablation_safety_regret.png", dpi=220)
    fig.savefig(root / "figures" / "calibration_ablation_safety_regret.pdf")
    plt.close(fig)


def write_interpretation(summary: pd.DataFrame, root: Path, parity: Mapping[str, float] | None = None) -> None:
    by_mode = summary.set_index("mode")
    lines = [
        "# Calibration Matched Ablation Interpretation",
        "",
        "This diagnostic compares controller variants with the same PT-MOMENT raw forecast, actual array, adapter checkpoint, and test anchors.",
        "",
        "## Supported Claims",
    ]
    full_gain = float(by_mode.loc["adapter_full_controller", "event_active_gain"]) if "adapter_full_controller" in by_mode.index else float("nan")
    always_non_event = float(by_mode.loc["adapter_always_apply", "non_event_degradation"]) if "adapter_always_apply" in by_mode.index else float("nan")
    no_bound_max = float(by_mode.loc["adapter_no_bound", "max_abs_correction"]) if "adapter_no_bound" in by_mode.index else float("nan")
    full_max = float(by_mode.loc["adapter_full_controller", "max_abs_correction"]) if "adapter_full_controller" in by_mode.index else float("nan")
    lines.append(f"- Full controller event-active gain vs PT-MOMENT: `{full_gain:.4f}` WAPE points.")
    if np.isfinite(always_non_event) and always_non_event > 0:
        lines.append(
            f"- Always-apply non-event WAPE degradation: `{always_non_event:.4f}`. "
            "This supports event gating as a numerical safety component."
        )
    else:
        lines.append(
            f"- Always-apply non-event WAPE degradation: `{always_non_event:.4f}`. "
            "This run does not show non-event numerical harm, although always-apply still perturbs many cells outside the event mask."
        )
    if np.isfinite(no_bound_max) and np.isfinite(full_max) and no_bound_max > full_max:
        lines.append(
            f"- No-bound max absolute correction: `{no_bound_max:.4f}` vs full controller `{full_max:.4f}`. "
            "This supports correction bounding as risk control."
        )
    else:
        lines.append(
            f"- No-bound max absolute correction: `{no_bound_max:.4f}` vs full controller `{full_max:.4f}`. "
            "This run does not provide strong evidence for a bound-control claim."
        )
    identical_guards = []
    for guard_mode in ["adapter_no_event_gate", "adapter_no_residual_support", "adapter_no_audit_guard"]:
        if guard_mode in by_mode.index and "adapter_full_controller" in by_mode.index:
            diff = abs(float(by_mode.loc[guard_mode, "event_active_gain"]) - full_gain)
            if diff < 1e-9:
                identical_guards.append(guard_mode)
    if identical_guards:
        lines.append(
            "- The following guard ablations are numerically identical to full controller in this run: "
            f"`{', '.join(identical_guards)}`. Treat these as evidence that the current proxy thresholds are permissive, not as proof of guard necessity."
        )
    lines.extend(
        [
            "",
            "## Claims To Avoid",
            "- This experiment does not show that RAG, AutoSkill, or LLM agents improve numerical accuracy.",
            "- Diagnostic ablations such as no-bound and always-apply are not deployable models.",
            "- If audit or residual-support ablations are close to the full controller, report them as weak evidence for those guards rather than overstating necessity.",
            "",
            "## Recommended Paper Placement",
            "- Main paper: event-active gain forest and safety-risk plot.",
            "- Appendix: full WAPE grouped bar, random-gate sanity baseline, and complete metric table.",
        ]
    )
    if parity:
        lines.extend(
            [
                "",
                "## Frozen Adapter Parity Diagnostic",
                f"- Mean absolute event-active WAPE difference between `adapter_full_controller` and frozen reference: `{parity.get('mean_abs_event_active_wape_diff', float('nan')):.6f}`.",
                f"- Max absolute event-active WAPE difference: `{parity.get('max_abs_event_active_wape_diff', float('nan')):.6f}`.",
                "- If this is non-zero, it reflects the added diagnostic guard logic rather than a new trained model.",
            ]
        )
    (root / "reports" / "calibration_ablation_interpretation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(root: Path, rows: pd.DataFrame, manifest: dict, bootstrap_samples: int = 2000, seed: int = 13, parity: Mapping[str, float] | None = None) -> dict:
    ensure_dirs(root)
    rows.to_csv(root / "predictions" / "calibration_ablation_rows.csv", index=False)
    summary = summarize_rows(rows)
    summary.to_csv(root / "tables" / "calibration_ablation_summary.csv", index=False)
    ci = bootstrap_gain_ci(rows, samples=bootstrap_samples, seed=seed)
    ci.to_csv(root / "tables" / "calibration_ablation_paired_gain_ci.csv", index=False)
    write_metric_directions(root)
    write_latex(summary, root)
    write_figures(summary, ci, root)
    write_interpretation(summary, root, parity=parity)
    manifest = dict(manifest)
    manifest.update(
        {
            "modes": MODES,
            "row_count": int(len(rows)),
            "anchor_count": int(rows["anchor"].nunique()) if len(rows) else 0,
            "bootstrap_samples": int(bootstrap_samples),
            "metric_directions": METRIC_DIRECTIONS,
            "outputs": {
                "rows": "predictions/calibration_ablation_rows.csv",
                "summary": "tables/calibration_ablation_summary.csv",
                "ci": "tables/calibration_ablation_paired_gain_ci.csv",
                "figures": [
                    "figures/calibration_ablation_wape_bar.png",
                    "figures/calibration_ablation_event_active_gain_forest.png",
                    "figures/calibration_ablation_safety_regret.png",
                ],
            },
        }
    )
    if parity:
        manifest["adapter_frozen_reference_parity"] = dict(parity)
    (root / "reports" / "calibration_ablation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"summary": summary, "ci": ci, "manifest": manifest}


def run_synthetic_smoke(output_root: str | Path, seed: int = 13) -> dict:
    root = Path(output_root)
    ensure_dirs(root)
    rng = np.random.default_rng(seed)
    rows: List[dict] = []
    n_anchors, n_channels, horizon = 4, 4, 6
    venue_indices = [0, 1]
    thresholds = ControllerThresholds()
    for a in range(n_anchors):
        actual = rng.uniform(80.0, 160.0, size=(n_channels, horizon)).astype(np.float32)
        raw = actual.copy()
        event_mask = np.zeros((n_channels, horizon), dtype=bool)
        event_mask[:2, 2:5] = True
        raw[event_mask] = actual[event_mask] * 1.08
        raw[~event_mask] = actual[~event_mask] * 0.99
        proposal = np.zeros_like(actual, dtype=np.float32)
        proposal[event_mask] = -0.05
        no_gate = np.full_like(actual, 0.035, dtype=np.float32)
        no_gate[event_mask] = -0.05
        unbounded = np.zeros_like(actual, dtype=np.float32)
        unbounded[event_mask] = -0.12
        unbounded[~event_mask] = 0.12
        score_base = np.full_like(actual, 0.85, dtype=np.float32)
        scores = {
            "source_validity_score": score_base.copy(),
            "geo_consistency_score": score_base.copy(),
            "temporal_alignment_score": score_base.copy(),
            "semantic_consistency_score": score_base.copy(),
            "residual_support_score": score_base.copy(),
            "evidence_validity_score": score_base.copy(),
        }
        for mode in MODES:
            corr, meta = build_mode_correction(
                mode=mode,
                proposal_correction=proposal,
                no_gate_proposal_correction=no_gate,
                unbounded_correction=unbounded,
                event_mask=event_mask,
                scores=scores,
                thresholds=thresholds,
                residual_bound=0.05,
                random_seed=seed,
                anchor_index=a,
            )
            adjusted = raw * (1.0 + corr)
            rows.append(
                evaluate_anchor(
                    split="test",
                    anchor=1000 + a,
                    date=f"2023-06-{1 + a:02d} 10:00:00",
                    mode=mode,
                    raw=raw,
                    adjusted=adjusted,
                    actual=actual,
                    correction=corr,
                    event_mask=event_mask,
                    venue_indices=venue_indices,
                    correction_meta=meta,
                )
            )
    manifest = {
        "status": "synthetic_smoke",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "synthetic": True,
        "random_seed": int(seed),
        "claim_boundary": "Synthetic smoke validates output contracts only.",
    }
    return write_outputs(root, pd.DataFrame(rows), manifest, bootstrap_samples=200, seed=seed)


def _predict_adapter_correction(adapter, features: np.ndarray, device: str, *, force_event_active: float | None = None, disable_bound: bool = False) -> np.ndarray:
    import torch
    from agents.event_adapter import FEATURE_NAMES

    feats = np.asarray(features, dtype=np.float32).copy()
    if force_event_active is not None:
        feats[..., FEATURE_NAMES.index("event_active")] = float(force_event_active)
    flat = feats.reshape(-1, feats.shape[-1])
    with torch.no_grad():
        x = torch.tensor(flat, dtype=torch.float32, device=device)
        if disable_bound:
            out = adapter.net(x).squeeze(-1)
            corr = torch.tanh(out).detach().cpu().numpy().reshape(feats.shape[:2])
            event_gate = np.clip(feats[..., FEATURE_NAMES.index("event_active")], 0.0, 1.0)
            corr = corr * event_gate
        else:
            corr = adapter.to(device)(x).detach().cpu().numpy().reshape(feats.shape[:2])
    blend = float(getattr(adapter, "blend_factor", 1.0))
    corr = corr * blend
    if not disable_bound:
        corr = np.clip(corr, -float(adapter.max_correction), float(adapter.max_correction))
    return corr.astype(np.float32)


def run_full_ablation(args: argparse.Namespace) -> dict:
    import torch
    from sklearn.preprocessing import StandardScaler
    from agents.event_adapter import build_event_feature_cube, channel_meta_by_name, load_adapter, load_channel_map, load_events_json
    from agents.numerical_agent import NumericalPredictionAgent
    from event_post_training.config import resolve_lp_model_path
    from experiments.run_ccfa_fullstack_final import _inverse_transform_batch, validate_real_lp_adapter_manifest
    from experiments.run_full_autoskill_skillbench import build_full_hourly_anchor_plan
    from experiments.run_full_paper_results import load_channel_names, load_venue_indices

    root = Path(args.output_root)
    ensure_dirs(root)
    provenance_root = Path(args.authoritative_root)
    adapter_manifest = Path(args.adapter_manifest) if args.adapter_manifest else Path(args.adapter_path) / "training_manifest.json"
    adapter_status = validate_real_lp_adapter_manifest(adapter_manifest)

    traffic_csv = Path(args.traffic_csv)
    df = pd.read_csv(traffic_csv, parse_dates=["date"])
    channel_names = load_channel_names(traffic_csv)
    venue_indices, venue_meta = load_venue_indices(Path(args.fusion_channels), channel_names)
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
        model_path=str(root / "__no_gca_model_for_calibration_ablation__"),
        device=args.device,
        forecast_horizon=args.horizon,
        n_channels=len(channel_names),
        channel_names=list(channel_names),
        lp_model_path=str(lp_dir),
        lp_data_path=str(traffic_csv),
        allow_lp_training_fallback=False,
    )
    adapter, adapter_config = load_adapter(args.adapter_path, device=args.device)
    seq_len = int(args.seq_len)
    thresholds = ControllerThresholds(
        source_validity=args.source_validity_threshold,
        geo_consistency=args.geo_consistency_threshold,
        temporal_alignment=args.temporal_alignment_threshold,
        evidence_validity=args.evidence_validity_threshold,
        residual_support=args.residual_support_threshold,
    )
    rows: List[dict] = []
    parity_diffs: List[float] = []

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
            timestamps = [str(ts) for ts in df["date"].iloc[anchor_id : anchor_id + args.horizon].tolist()]
            raw = raw_batch[b_idx]
            actual = data_np[anchor_id : anchor_id + args.horizon].T.astype(np.float32)
            filtered_events = filter_events_for_window(
                timed_events,
                pd.Timestamp(timestamps[0]),
                pd.Timestamp(timestamps[-1]),
                back_hours=args.event_prefilter_back_hours,
                forward_hours=args.event_prefilter_forward_hours,
            )
            features = build_event_feature_cube(raw, timestamps, channel_names, filtered_events, channel_meta)
            event_mask = features[..., 0] > 0.0
            scores = compute_controller_scores(features)
            proposal = _predict_adapter_correction(adapter, features, args.device, force_event_active=None, disable_bound=False)
            no_gate_proposal = _predict_adapter_correction(adapter, features, args.device, force_event_active=1.0, disable_bound=False)
            unbounded = _predict_adapter_correction(adapter, features, args.device, force_event_active=None, disable_bound=True)

            reference_adjusted = raw * (1.0 + proposal)
            reference_event_wape = _wape(actual, reference_adjusted, event_mask)

            for mode in MODES:
                corr, meta = build_mode_correction(
                    mode=mode,
                    proposal_correction=proposal,
                    no_gate_proposal_correction=no_gate_proposal,
                    unbounded_correction=unbounded,
                    event_mask=event_mask,
                    scores=scores,
                    thresholds=thresholds,
                    residual_bound=args.residual_bound,
                    random_seed=args.random_seed,
                    anchor_index=anchor_id,
                )
                adjusted = raw * (1.0 + corr)
                row = evaluate_anchor(
                    split="test",
                    anchor=anchor_id,
                    date=str(anchor.get("date") or df["date"].iloc[anchor_id]),
                    mode=mode,
                    raw=raw,
                    adjusted=adjusted,
                    actual=actual,
                    correction=corr,
                    event_mask=event_mask,
                    venue_indices=venue_indices,
                    correction_meta=meta,
                )
                rows.append(row)
                if mode == "adapter_full_controller":
                    parity_diffs.append(abs(float(row["event_active_wape"]) - float(reference_event_wape)) if np.isfinite(reference_event_wape) else 0.0)
            processed += 1
            if processed % max(1, int(args.progress_every)) == 0 or processed == len(test_anchors):
                print(f"[calibration_ablation] processed {processed}/{len(test_anchors)} anchors", flush=True)

    rows_df = pd.DataFrame(rows)
    parity = {
        "reference": "adapter_frozen_reference_apply_correction_equivalent",
        "mean_abs_event_active_wape_diff": float(np.mean(parity_diffs)) if parity_diffs else float("nan"),
        "max_abs_event_active_wape_diff": float(np.max(parity_diffs)) if parity_diffs else float("nan"),
    }
    manifest = {
        "status": "completed",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "synthetic": False,
        "authoritative_root": str(provenance_root),
        "authoritative_root_read_only": True,
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
        "controller_thresholds": thresholds.__dict__,
        "residual_bound": float(args.residual_bound),
        "decision_actual_leakage_guard": "controller helper has no actual argument; actual is only passed to evaluation.",
        "llm_or_qwenplus_used": False,
        "autoskill_updated": False,
        "test_time_training_or_promotion": False,
        "event_prefilter": {
            "timed_event_count": int(len(timed_events)),
            "back_hours": float(args.event_prefilter_back_hours),
            "forward_hours": float(args.event_prefilter_forward_hours),
            "uses_actual": False,
        },
        "venue_meta": venue_meta,
    }
    return write_outputs(root, rows_df, manifest, bootstrap_samples=args.bootstrap_samples, seed=args.random_seed, parity=parity)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run matched calibration ablations for EAF-MAS.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--output_root", default=str(PROJECT_ROOT / "autotemp" / f"revision_trc_tits_ccfa_{timestamp}" / "calibration_ablation"))
    parser.add_argument("--authoritative_root", default=str(PROJECT_ROOT / "autotemp" / "ccfa_fullstack_summary_only_20260619_123041"))
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
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--random_seed", type=int, default=13)
    parser.add_argument("--residual_bound", type=float, default=0.05)
    parser.add_argument("--event_prefilter_back_hours", type=float, default=72.0)
    parser.add_argument("--event_prefilter_forward_hours", type=float, default=2.0)
    parser.add_argument("--progress_every", type=int, default=32)
    parser.add_argument("--source_validity_threshold", type=float, default=0.20)
    parser.add_argument("--geo_consistency_threshold", type=float, default=0.20)
    parser.add_argument("--temporal_alignment_threshold", type=float, default=0.50)
    parser.add_argument("--evidence_validity_threshold", type=float, default=0.20)
    parser.add_argument("--residual_support_threshold", type=float, default=0.20)
    parser.add_argument("--synthetic_smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.synthetic_smoke:
        run_synthetic_smoke(args.output_root, seed=args.random_seed)
    else:
        run_full_ablation(args)


if __name__ == "__main__":
    main()
