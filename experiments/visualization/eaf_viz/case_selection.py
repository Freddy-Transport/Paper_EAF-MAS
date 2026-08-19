"""Post-hoc visualization case selection for EAF-MAS figures.

The selector is deliberately descriptive: it never changes model outputs,
thresholds, controller decisions, or skill memories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


CASE_TYPES = [
    "localized_gain",
    "event_window_harm",
    "controller_abstention",
    "historical_memory_supported_correction",
]
MIN_LOCALIZED_GAIN = 0.10


def select_visualization_cases(
    case_metrics: pd.DataFrame,
    horizon_forecast: pd.DataFrame | None = None,
    channel_metrics: pd.DataFrame | None = None,
) -> List[dict]:
    if case_metrics.empty:
        return [_missing_row(case_type, "no case_metrics rows available") for case_type in CASE_TYPES]

    cases = case_metrics.copy()
    selected = [
        _select_localized_gain(cases),
        _select_event_window_harm(cases),
        _select_controller_abstention(cases),
        _select_historical_memory_supported(cases),
    ]
    return selected


def write_selected_visualization_cases(
    case_metrics: pd.DataFrame,
    horizon_forecast: pd.DataFrame,
    channel_metrics: pd.DataFrame,
    out_path: str | Path,
) -> List[dict]:
    cases = select_visualization_cases(case_metrics, horizon_forecast, channel_metrics)
    payload = {
        "selection_policy": "post_hoc_visualization_only",
        "note": (
            "Cases are selected from completed test outputs for visualization and qualitative audit only. "
            "They must not be used to update model weights, controller thresholds, residual memory, or skill promotion."
        ),
        "cases": cases,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    missing = [row for row in cases if row.get("status") == "missing_required_case"]
    missing_report = out.with_name("missing_visualization_cases.md")
    if missing:
        lines = [
            "# Missing Visualization Case Types",
            "",
            "The selector did not fabricate missing categories. The following case types require new matched runs before they can be plotted as empirical evidence:",
            "",
        ]
        for row in missing:
            lines.append(f"- `{row.get('case_type')}`: {row.get('selection_reason')}")
        lines.append("")
        lines.append("Recommended action: rerun `experiments/run_paper_event_forecasting.py` on additional test anchors and include both `rag_explain` and `event_adapter_frozen_moment` outputs in `viz_config.yaml`.")
        missing_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif missing_report.exists():
        missing_report.unlink()
    return cases


def _select_localized_gain(cases: pd.DataFrame) -> dict:
    score_col = "adjusted_channel_wape_gain" if "adjusted_channel_wape_gain" in cases else "wape_gain"
    ranked = _sort_numeric(cases, score_col, ascending=False)
    if ranked.empty:
        return _missing_row("localized_gain", f"no numeric {score_col} values available")
    visible = ranked[ranked["_selection_score"] >= MIN_LOCALIZED_GAIN]
    if visible.empty:
        return _missing_row(
            "localized_gain",
            f"no candidate reaches visible localized gain threshold {MIN_LOCALIZED_GAIN:.2f} on {score_col}",
        )
    return _selected_row(
        "localized_gain",
        visible.head(1),
        f"largest {score_col}; highlights correction effect on adjusted station/channel units",
        score_col,
    )


def _select_event_window_harm(cases: pd.DataFrame) -> dict:
    score_col = "event_window_wape_gain" if "event_window_wape_gain" in cases else "wape_gain"
    ranked = _sort_numeric(cases, score_col, ascending=True)
    if ranked.empty:
        return _missing_row("event_window_harm", f"no numeric {score_col} values available")
    row = ranked.iloc[0]
    if _to_float(row.get(score_col), default=0.0) >= 0.0:
        return _missing_row("event_window_harm", "no case shows negative event-window WAPE gain")
    return _row_payload(
        "event_window_harm",
        row,
        "selected",
        f"most negative {score_col}; useful for failure analysis and risk-control discussion",
        score_col,
    )


def _select_controller_abstention(cases: pd.DataFrame) -> dict:
    required = {
        "mode",
        "method_label",
        "calibration_enabled",
        "controller_participated",
        "controller_allowed",
        "abstain",
        "calibration_decision",
        "final_matches_raw_forecast",
        "max_applied_correction",
        "failed_evidence_dimensions",
    }
    missing = sorted(required.difference(cases.columns))
    if missing:
        return _missing_row(
            "controller_abstention",
            "strict Controller-abstention fields unavailable: " + ", ".join(missing),
        )

    # This is intentionally conjunctive.  Explanation-only EAF-MAS-X cases can
    # have an unchanged forecast and an abstain-like label, but the Calibration
    # Controller did not participate in those runs.
    mask = (
        cases["mode"].fillna("").astype(str).eq("event_adapter_frozen_moment")
        & cases["method_label"].fillna("").astype(str).eq("EAF-MAS-C")
        & cases["calibration_enabled"].map(_is_true)
        & cases["controller_participated"].map(_is_true)
        & cases["controller_allowed"].map(_is_false)
        & cases["abstain"].map(_is_true)
        & cases["calibration_decision"].fillna("").astype(str).str.lower().eq("abstain")
        & cases["final_matches_raw_forecast"].map(_is_true)
        & (pd.to_numeric(cases["max_applied_correction"], errors="coerce").fillna(float("inf")) <= 1e-8)
        & cases["failed_evidence_dimensions"].fillna("").astype(str).str.strip().ne("")
    )
    candidates = cases[mask].copy()
    if candidates.empty:
        return _missing_row(
            "controller_abstention",
            "no completed EAF-MAS-C case satisfies all Controller-abstention invariants",
        )

    failed = candidates["failed_evidence_dimensions"].fillna("").astype(str)
    candidates["_failure_priority"] = np.select(
        [
            failed.str.contains("residual_support_score", regex=False),
            failed.str.contains("source_validity_score", regex=False),
            failed.str.contains("geo_consistency_score", regex=False),
            failed.str.contains("severe_conflict", regex=False),
        ],
        [0, 1, 2, 3],
        default=4,
    )
    proposal = (
        pd.to_numeric(candidates["adapter_proposal_max_abs_correction"], errors="coerce").fillna(0.0)
        if "adapter_proposal_max_abs_correction" in candidates
        else pd.Series(0.0, index=candidates.index)
    )
    candidates["_proposal_priority"] = (proposal <= 1e-8).astype(int)
    candidates["_evidence_rank"] = (
        pd.to_numeric(candidates["evidence_validity_score"], errors="coerce").fillna(float("inf"))
        if "evidence_validity_score" in candidates
        else pd.Series(float("inf"), index=candidates.index)
    )
    candidates = candidates.sort_values(
        ["_failure_priority", "_proposal_priority", "_evidence_rank"],
        ascending=[True, True, True],
    )
    row = candidates.iloc[0]
    reason = (
        "strict EAF-MAS-C Controller abstention; failed "
        f"{row.get('failed_evidence_dimensions')}; nonzero adapter proposal was rejected and PT-MOMENT raw retained"
    )
    return _row_payload("controller_abstention", row, "selected", reason, "evidence_validity_score")


def _select_historical_memory_supported(cases: pd.DataFrame) -> dict:
    if "has_historical_residual_memory" not in cases:
        return _missing_row("historical_memory_supported_correction", "has_historical_residual_memory column unavailable")
    candidates = cases[cases["has_historical_residual_memory"].fillna(False).astype(bool)]
    if candidates.empty:
        return _missing_row("historical_memory_supported_correction", "no case contains historical residual memory")
    score_col = "included_unit_wape_gain" if "included_unit_wape_gain" in candidates else "wape_gain"
    return _selected_row(
        "historical_memory_supported_correction",
        _sort_numeric(candidates, score_col, ascending=False).head(1),
        f"historical memory available and high {score_col}; use to explain residual-memory reasoning",
        score_col,
    )


def _selected_row(case_type: str, rows: pd.DataFrame, reason: str, score_col: str) -> dict:
    if rows.empty:
        return _missing_row(case_type, f"no candidate rows for {score_col}")
    return _row_payload(case_type, rows.iloc[0], "selected", reason, score_col)


def _row_payload(case_type: str, row: pd.Series, status: str, reason: str, score_col: str | None) -> dict:
    payload = {
        "case_type": case_type,
        "status": status,
        "case_id": row.get("case_id"),
        "input_file": row.get("input_file"),
        "anchor_time": row.get("anchor_time"),
        "mode": row.get("mode"),
        "method_label": row.get("method_label"),
        "event_name": row.get("event_name"),
        "event_type": row.get("event_type"),
        "impact_tier": row.get("impact_tier"),
        "selection_reason": reason,
        "post_hoc_visualization_only": True,
    }
    if score_col:
        payload["selection_metric"] = score_col
        payload["selection_metric_value"] = _to_float(row.get(score_col))
    for col in [
        "wape_gain",
        "event_window_wape_gain",
        "adjusted_channel_wape_gain",
        "included_unit_wape_gain",
        "calibration_enabled",
        "calibration_decision",
        "abstain",
        "controller_allowed",
        "controller_participated",
        "controller_abstention_reason",
        "failed_evidence_dimensions",
        "adapter_proposal_max_abs_correction",
        "adapter_proposal_active_cells",
        "max_applied_correction",
        "final_matches_raw_forecast",
        "physical_event_key",
        "evidence_validity_score",
    ]:
        if col in row.index:
            payload[col] = _json_scalar(row.get(col))
    return payload


def _missing_row(case_type: str, reason: str) -> dict:
    return {
        "case_type": case_type,
        "status": "missing_required_case",
        "selection_reason": reason,
        "post_hoc_visualization_only": True,
    }


def _sort_numeric(df: pd.DataFrame, col: str, ascending: bool) -> pd.DataFrame:
    if col not in df:
        return pd.DataFrame(columns=df.columns)
    ranked = df.copy()
    ranked["_selection_score"] = pd.to_numeric(ranked[col], errors="coerce")
    return ranked.dropna(subset=["_selection_score"]).sort_values("_selection_score", ascending=ascending)


def _to_float(value, default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _json_scalar(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bool, str, int, float)):
        return value
    return str(value)


def _is_true(value) -> bool:
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")


def _is_false(value) -> bool:
    return value is False or (isinstance(value, str) and value.strip().lower() == "false")
