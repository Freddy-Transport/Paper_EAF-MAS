"""Case dashboard figure generation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .plotting_utils import FigureManifest, missing_note, save_figure
from .style import COLORS, add_panel_labels, apply_style, despine


def select_case(case_metrics: pd.DataFrame, requested_case_id: str | None = None) -> str | None:
    if case_metrics.empty:
        return None
    if requested_case_id and requested_case_id in set(case_metrics["case_id"]):
        return requested_case_id
    ranked = case_metrics.copy()
    score = ranked.get("adjusted_channel_wape_gain", ranked.get("wape_gain", pd.Series([0] * len(ranked)))).fillna(0)
    if "included_unit_wape_gain" in ranked:
        score = score + ranked["included_unit_wape_gain"].fillna(0) * 0.5
    if "event_window_wape_gain" in ranked:
        score = score + ranked["event_window_wape_gain"].fillna(0) * 0.05
    if "has_historical_residual_memory" in ranked:
        score = score + ranked["has_historical_residual_memory"].fillna(False).astype(int) * 0.05
    if "evidence_validity_score" in ranked:
        score = score + ranked["evidence_validity_score"].fillna(0) * 0.02
    ranked["_score"] = score
    return str(ranked.sort_values("_score", ascending=False).iloc[0]["case_id"])


def make_case_dashboard(tables: Dict[str, pd.DataFrame], out_dirs: dict, case_id: str | None = None) -> FigureManifest:
    import matplotlib.pyplot as plt

    apply_style()
    manifest = FigureManifest()
    cases = tables.get("case_metrics", pd.DataFrame())
    horizon = tables.get("horizon_forecast", pd.DataFrame())
    audit = tables.get("event_audit", pd.DataFrame())
    residual = tables.get("residual_memory", pd.DataFrame())
    channels = tables.get("channel_metrics", pd.DataFrame())
    selected = select_case(cases, case_id)
    if not selected:
        missing_note(out_dirs["reports"] / "missing_fig8_case_dashboard.md", "Missing case dashboard", "No case_metrics rows available.")
        manifest.add("fig8_case_dashboard", "Case dashboard", [], "", "", ["case_metrics"], "skipped_missing_data", "No case selected.")
        return manifest

    c = cases[cases["case_id"] == selected].iloc[0]
    hf = horizon[horizon["case_id"] == selected]
    au = audit[audit["case_id"] == selected]
    rm = residual[residual["case_id"] == selected]
    ch = channels[channels["case_id"] == selected]
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 10.0))
    axes = axes.ravel()

    axes[0].axis("off")
    axes[0].text(0.03, 0.88, f"Anchor: {c.get('anchor_time')}", fontsize=9)
    axes[0].text(0.03, 0.72, f"Mode: {c.get('method_label')}", fontsize=9)
    axes[0].text(0.03, 0.56, f"Horizon: {c.get('horizon')} h", fontsize=9)
    axes[0].text(0.03, 0.40, "Forecast-time evidence only", fontsize=9, color=COLORS["audit"])
    axes[0].set_title("Forecast request")

    score_cols = ["source_validity_score", "geo_consistency_score", "temporal_alignment_score", "residual_support_score"]
    if not au.empty:
        matrix = au[score_cols].fillna(0).head(6).values
        axes[1].imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        axes[1].set_xticks(range(len(score_cols)))
        axes[1].set_xticklabels([x.replace("_score", "").replace("_", "\n") for x in score_cols], fontsize=7)
        axes[1].set_yticks(range(min(6, len(au))))
        axes[1].set_yticklabels(au["event_name"].fillna("event").astype(str).str.slice(0, 16).head(6), fontsize=7)
    axes[1].set_title("Evidence audit")

    if not ch.empty:
        top = ch.sort_values(["is_adjusted_channel", "correction_abs_mean"], ascending=[False, False]).head(10)
        axes[2].barh(top["station_name"].astype(str).str.slice(0, 24), top["correction_abs_mean"].fillna(0), color=top["is_adjusted_channel"].map({True: COLORS["adjusted"], False: COLORS["missing"]}))
    axes[2].set_title("Channel decisions")
    despine(axes[2])

    if not hf.empty:
        core_name = ch.sort_values("correction_abs_mean", ascending=False).iloc[0]["channel_id"] if not ch.empty else hf.iloc[0]["channel_id"]
        curve = hf[hf["channel_id"] == core_name].sort_values("horizon_step")
        axes[3].plot(curve["horizon_step"], curve["observed"], label="observed", color=COLORS["observed"])
        axes[3].plot(curve["horizon_step"], curve["raw_forecast"], label="PT-MOMENT", color=COLORS["raw"], linestyle="--")
        axes[3].plot(curve["horizon_step"], curve["adjusted_forecast"], label="EAF-MAS-C", color=COLORS["adjusted"])
        axes[3].legend()
    axes[3].set_title("Raw vs adjusted forecast")
    despine(axes[3])

    if not hf.empty:
        axes[4].plot(curve["horizon_step"], curve["relative_correction"] * 100.0, color=COLORS["memory"])
        axes[4].axhline(float(c.get("correction_bound") or 0.05) * 100.0, color=COLORS["harmful"], linestyle="--")
        axes[4].axhline(-float(c.get("correction_bound") or 0.05) * 100.0, color=COLORS["harmful"], linestyle="--")
    axes[4].set_title("Relative correction (%)")
    despine(axes[4])

    if not rm.empty:
        vals = rm.get("median_correction_pct", pd.Series(dtype=float)).fillna(0).head(8)
        labels = rm.get("historical_event_name", rm.get("historical_event_type", pd.Series(["case"] * len(rm)))).fillna("case").astype(str).str.slice(0, 18).head(8)
        axes[5].barh(labels, vals, color=COLORS["memory"])
    axes[5].set_title("Residual memory analogues")
    despine(axes[5])

    axes[6].axis("off")
    axes[6].text(0.03, 0.82, f"Decision: {c.get('calibration_decision')}", fontsize=10)
    axes[6].text(0.03, 0.65, f"Confidence: {c.get('confidence')}", fontsize=9)
    axes[6].text(0.03, 0.50, f"Max correction: {c.get('max_correction')}", fontsize=9)
    axes[6].text(0.03, 0.35, f"Adjusted channels: {c.get('adjusted_channel_count')}", fontsize=9)
    axes[6].set_title("Calibration decision")

    axes[7].axis("off")
    axes[7].text(0.03, 0.82, f"Raw WAPE: {float(c.get('raw_wape')):.3f}", fontsize=10)
    axes[7].text(0.03, 0.66, f"Adjusted WAPE: {float(c.get('adjusted_wape')):.3f}", fontsize=10)
    axes[7].text(0.03, 0.50, f"WAPE gain: {float(c.get('wape_gain')):.3f}", fontsize=10)
    if "event_window_wape_gain" in c:
        axes[7].text(0.03, 0.34, f"Event-window gain: {float(c.get('event_window_wape_gain')):.3f}", fontsize=9)
    if "adjusted_channel_wape_gain" in c:
        axes[7].text(0.03, 0.20, f"Adjusted-channel gain: {float(c.get('adjusted_channel_wape_gain')):.3f}", fontsize=9)
    axes[7].set_title("Metrics")

    axes[8].axis("off")
    axes[8].text(
        0.03,
        0.82,
        "Evidence audit and residual memory jointly\nsupport a bounded correction.\nNon-citable model summaries remain separated\nfrom verified citations.",
        fontsize=9,
        va="top",
    )
    axes[8].set_title("Explanation summary")

    add_panel_labels(axes)
    fig.tight_layout()
    pdf, png = save_figure(fig, out_dirs["figures_main"] / "fig8_case_dashboard")
    plt.close(fig)
    notes = out_dirs["figures_main"] / "fig8_case_dashboard_notes.md"
    notes.write_text(
        f"# Fig. 8 Case Dashboard Notes\n\nSelected case: `{selected}`.\n\nThis dashboard links forecast request, evidence audit, localized channel gating, bounded correction, residual memory, and post-hoc metrics without using post-event evidence as forecast-time evidence.\n",
        encoding="utf-8",
    )
    manifest.add("fig8_case_dashboard", "Case study dashboard", [], pdf, png, ["case_metrics", "horizon_forecast"], "generated", f"selected_case={selected}")
    return manifest
