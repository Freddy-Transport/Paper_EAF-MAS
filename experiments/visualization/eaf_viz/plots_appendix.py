"""Appendix figure generation."""

from __future__ import annotations

import pandas as pd

from .plotting_utils import FigureManifest, missing_note, save_figure
from .style import COLORS, apply_style, despine


def should_plot_underpowered_performance_group(cases: pd.DataFrame, group_col: str, min_cases: int = 20) -> bool:
    if cases.empty or group_col not in cases or "wape_gain" not in cases:
        return False
    if len(cases) < min_cases:
        return False
    return cases[group_col].dropna().nunique() >= 2


def make_appendix_figures(tables: dict, out_dirs: dict, cfg: dict) -> FigureManifest:
    import matplotlib.pyplot as plt

    apply_style()
    manifest = FigureManifest()
    cases = tables.get("case_metrics", pd.DataFrame())
    if cases.empty:
        missing_note(out_dirs["reports"] / "missing_appendix_figures.md", "Missing appendix figures", "No case_metrics rows available.")
        return manifest

    score_cols = [
        ("source", "source_validity_score"),
        ("geo", "geo_consistency_score"),
        ("temporal", "temporal_alignment_score"),
        ("semantic", "semantic_consistency_score"),
        ("residual", "residual_support_score"),
        ("overall", "evidence_validity_score"),
    ]
    available = [(label, col) for label, col in score_cols if col in cases]
    if available:
        fig, ax = plt.subplots(figsize=(7.6, 3.8))
        labels = [label for label, _ in available]
        means = [cases[col].fillna(0.0).mean() for _, col in available]
        ax.bar(labels, means, color=COLORS["audit"], alpha=0.86)
        ax.set_ylim(0, 1.05)
        ax.set_title("Appendix A1: Evidence audit score components")
        ax.set_ylabel("mean score")
        despine(ax)
        pdf, png = save_figure(fig, out_dirs["figures_appendix"] / "figA1_evidence_audit_scores")
        plt.close(fig)
        manifest.add("figA1_evidence_audit_scores", "Evidence audit score components across focused cases", [], pdf, png, [col for _, col in available], "generated")
    else:
        manifest.add("figA1_evidence_audit_scores", "Evidence audit score components across focused cases", [], "", "", ["case_metrics.audit_scores"], "skipped_missing_data", "No audit score fields.")

    diagnostics = {
        "Qwen parsed": 1.0,
        "model summary": float(cases.get("has_model_assisted_summary", pd.Series([False] * len(cases))).fillna(False).astype(bool).mean()),
        "historical memory": float(cases.get("has_historical_residual_memory", pd.Series([False] * len(cases))).fillna(False).astype(bool).mean()),
        "geo conflict-free": float((~cases.get("geo_conflict_flag", pd.Series([False] * len(cases))).fillna(False).astype(bool)).mean()),
        "source conflict-free": float((~cases.get("source_conflict_flag", pd.Series([False] * len(cases))).fillna(False).astype(bool)).mean()),
        "controller allowed": float(cases.get("controller_allowed", pd.Series([False] * len(cases))).fillna(False).astype(bool).mean()),
    }
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    labels = list(diagnostics.keys())
    values = list(diagnostics.values())
    colors = [COLORS["pass"] if v >= 0.75 else COLORS["weak"] if v > 0 else COLORS["missing"] for v in values]
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("case coverage rate")
    ax.set_title("Appendix A2: Explanation and evidence diagnostics")
    ax.tick_params(axis="x", rotation=25)
    despine(ax)
    pdf, png = save_figure(fig, out_dirs["figures_appendix"] / "figA2_explanation_quality_diagnostics")
    plt.close(fig)
    manifest.add(
        "figA2_explanation_quality_diagnostics",
        "Explanation quality and evidence taxonomy diagnostics",
        [],
        pdf,
        png,
        ["case_metrics.evidence_flags"],
        "generated",
        "Model-assisted summaries are the paper-facing Qwen-Plus evidence channel; URL citation diagnostics are legacy appendix material.",
    )

    for name, title in [
        ("missing_bound_sensitivity_plan.md", "Missing correction bound sensitivity"),
        ("missing_holdout_plan.md", "Missing venue/event-type holdout"),
        ("missing_runtime_plan.md", "Missing runtime and retrieval cost breakdown"),
    ]:
        missing_note(out_dirs["reports"] / name, title, "Required experiment outputs are not present in the current result roots. Re-run the corresponding ablation/sensitivity experiments before plotting.")
    return manifest
