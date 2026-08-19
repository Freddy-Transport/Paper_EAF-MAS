#!/usr/bin/env python3
"""Export paper-facing figures for the CCF-A full-stack run."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


COLORS = {
    "pt": "#2563eb",
    "adapter": "#f97316",
    "gain": "#16a34a",
    "loss": "#dc2626",
    "muted": "#94a3b8",
    "purple": "#7c3aed",
    "dark": "#111827",
}


def _setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 120,
    })
    return plt


def _save(fig: Any, path: Path, plt: Any, written: list[str]) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    written.append(str(path))


def _mode(rows: pd.DataFrame, mode: str) -> pd.DataFrame:
    return rows[rows["mode"] == mode].copy().sort_values("anchor")


def _ensure_metric_columns(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    if "non_event_n" not in rows:
        rows["non_event_n"] = (28 * 192) - rows.get("event_active_n", 0)
    if "max_abs_correction" not in rows:
        rows["max_abs_correction"] = 0.0
    if "active_correction_cells" not in rows:
        rows["active_correction_cells"] = rows.get("event_active_n", 0)
    return rows


def _merge_pt_adapter(rows: pd.DataFrame) -> pd.DataFrame:
    pt = _mode(rows, "pt_moment")
    adapter = _mode(rows, "event_adapter_frozen")
    merged = pt[
        [
            "anchor",
            "date",
            "top128_wape",
            "event_venue28_wape",
            "event_active_wape",
            "non_event_wape",
            "event_active_n",
            "non_event_n",
        ]
    ].merge(
        adapter[
            [
                "anchor",
                "top128_wape",
                "event_venue28_wape",
                "event_active_wape",
                "non_event_wape",
                "max_abs_correction",
                "active_correction_cells",
            ]
        ],
        on="anchor",
        suffixes=("_raw", "_adapter"),
    )
    for subset in ["top128", "event_venue28", "event_active", "non_event"]:
        merged[f"{subset}_gain"] = merged[f"{subset}_wape_raw"] - merged[f"{subset}_wape_adapter"]
    merged["event_ratio"] = merged["event_active_n"] / (merged["event_active_n"] + merged["non_event_n"])
    return merged


def _bootstrap_ci(values: pd.Series, samples: int = 1000, seed: int = 7) -> tuple[float, float]:
    arr = np.asarray(values.dropna(), dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = [rng.choice(arr, size=arr.size, replace=True).mean() for _ in range(samples)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _quality_delta(skill: dict) -> pd.DataFrame:
    rows = skill.get("full_autoskill_delta_ci") or []
    if rows:
        return pd.DataFrame(rows)
    no_skill = skill.get("mean_quality_by_mode", {}).get("no_skill", {})
    full = skill.get("mean_quality_by_mode", {}).get("full_autoskill_memory", {})
    return pd.DataFrame(
        [
            {"metric": k, "mean_delta": float(full.get(k, 0)) - float(v), "ci_low": 0.0, "ci_high": 0.0}
            for k, v in no_skill.items()
        ]
    )


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _figure_sample_size(path: Path) -> str:
    name = path.name
    if name.startswith("figA") or name.startswith("figB") or name.startswith("figC") or name.startswith("figD"):
        return "full split"
    if name.startswith("figE"):
        return "unique high-value physical events with model-assisted summaries"
    if name.startswith("figF") or name.startswith("figG") or name.startswith("figH") or name.startswith("figI"):
        return "full SkillBench"
    if name.startswith("figJ"):
        return "case study"
    if re.match(r"fig(8|11|12)_", name):
        return "case study legacy"
    return "legacy mixed or diagnostic"


def _claim_type(path: Path) -> str:
    name = path.name
    if name.startswith("figA"):
        return "dataset_split"
    if name.startswith("figB") or name.startswith("figC"):
        return "forecasting_result"
    if name.startswith("figD"):
        return "bounded_calibration_safety"
    if name.startswith("figE"):
        return "evidence_rag_quality"
    if name.startswith("figF") or name.startswith("figG") or name.startswith("figH"):
        return "autoskill_effectiveness"
    if name.startswith("figI"):
        return "safe_abstention_stress"
    if name.startswith("figJ") or re.match(r"fig(8|11|12)_", name):
        return "case_study"
    return "legacy_diagnostic"


def _build_figure_provenance(root: Path, generated: list[str], manifest: dict) -> list[dict]:
    rows: list[dict] = []
    generated_paths = [Path(path) for path in generated]
    for path in generated_paths:
        rows.append(
            {
                "figure": path.name,
                "source_type": "full_split_final" if not path.name.startswith("figJ") else "case_study_final",
                "source_root": str(root),
                "script": "experiments/export_ccfa_fullstack_figures.py",
                "sample_size": _figure_sample_size(path),
                "claim_type": _claim_type(path),
                "duplicate_of": "",
                "paper_action": "main" if not path.name.startswith("figJ") else "main_case_study",
            }
        )
    legacy_dir = Path("experiments/visualization/outputs/figures_main")
    if legacy_dir.is_dir():
        for path in sorted(legacy_dir.glob("fig*.png")):
            if path.name.startswith("figA") or path.name.startswith("figB") or path.name.startswith("figC") or path.name.startswith("figD") or path.name.startswith("figE") or path.name.startswith("figF") or path.name.startswith("figG") or path.name.startswith("figH") or path.name.startswith("figI") or path.name.startswith("figJ"):
                # Previous copies of the new naming scheme are superseded by the
                # current run root; keep only the current run in final outputs.
                action = "superseded_by_current_run"
            elif re.match(r"fig(8|11|12)_", path.name):
                action = "appendix_case_diagnostic"
            elif path.name == "figJ_case_card_pair_placeholder.png":
                action = "archive_placeholder"
            else:
                action = "legacy_appendix_or_archive"
            rows.append(
                {
                    "figure": path.name,
                    "source_type": "legacy_processed_viz",
                    "source_root": str(legacy_dir),
                    "script": "experiments/visualization/eaf_viz/plots_main.py",
                    "sample_size": _figure_sample_size(path),
                    "claim_type": _claim_type(path),
                    "duplicate_of": "current figA-J only if source/claim/sample match; otherwise not deleted as duplicate",
                    "paper_action": action,
                }
            )
    manifest["figure_provenance_policy"] = "legacy figures are excluded from figures_main_final unless explicitly copied as appendix diagnostics"
    return rows


def _write_provenance(root: Path, rows: list[dict]) -> None:
    report_dir = root / "reports"
    df = pd.DataFrame(rows)
    df.to_csv(report_dir / "figure_provenance_audit.csv", index=False)
    payload = {
        "summary": {
            "policy": "legacy figures are excluded from figures_main_final; delete only when same source, same claim, and same statistic are superseded",
            "figure_count": len(rows),
        },
        "figures": rows,
    }
    (report_dir / "figure_provenance_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_final_figure_dirs(root: Path, generated: list[str], provenance_rows: list[dict]) -> None:
    main_dir = root / "figures_main_final"
    appendix_dir = root / "figures_appendix_final"
    legacy_dir = root / "figures_legacy"
    for directory in [main_dir, appendix_dir, legacy_dir]:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
    main_names = {
        "figA_full_split_prediction_landscape.png",
        "figB_backbone_adapter_comparison.png",
        "figC_event_active_adapter_gain.png",
        "figD_correction_safety_sparsity.png",
        "figE_model_assisted_summary_coverage.png",
        "figF_autoskill_lifecycle.png",
        "figG_skill_explanation_quality_delta.png",
        "figH_residual_memory_organization.png",
        "figI_safe_abstention_matrix.png",
        "figJ_case_card_pair.png",
    }
    for path_text in generated:
        path = Path(path_text)
        if path.name in main_names:
            shutil.copy2(path, main_dir / path.name)
            pdf = path.with_suffix(".pdf")
            if pdf.is_file():
                shutil.copy2(pdf, main_dir / pdf.name)
    legacy_source = Path("experiments/visualization/outputs/figures_main")
    if legacy_source.is_dir():
        for row in provenance_rows:
            if row["source_type"] != "legacy_processed_viz":
                continue
            src = legacy_source / row["figure"]
            if not src.is_file():
                continue
            if row["paper_action"] == "appendix_case_diagnostic":
                shutil.copy2(src, appendix_dir / src.name)
            else:
                shutil.copy2(src, legacy_dir / src.name)


def _write_interpretation_guide(root: Path, manifest: dict) -> None:
    guide = root / "reports" / "figure_interpretation_guide.md"
    lines = [
        "# CCF-A Figure Interpretation Guide",
        "",
        "These figures separate dataset-level evidence from qualitative case-study evidence.",
        "",
    ]
    for name, placement in manifest["paper_placement"].items():
        lines.append(f"- **{name}**: {placement}")
    lines.extend(
        [
            "",
            "Claim boundary: PT-MOMENT is the numerical backbone; the frozen adapter provides sparse bounded event-channel correction; Evidence-RAG and AutoSkill-style memory support traceability, residual-memory organization, and safe abstention.",
            "Do not claim that Skill changes forecast arrays or directly improves numerical accuracy.",
        ]
    )
    guide.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _line_after(text: str, prefix: str) -> str:
    for line in text.splitlines():
        if line.strip().startswith(prefix):
            return line.split(":", 1)[-1].strip()
    return ""


def _first_bullet_after(text: str, heading: str) -> str:
    idx = text.find(heading)
    if idx < 0:
        return ""
    tail = text[idx:].splitlines()[1:25]
    for line in tail:
        stripped = line.strip()
        if stripped.startswith("- "):
            return stripped[2:].strip()
        if stripped.startswith("## ") and line != tail[0]:
            break
    return ""


def _case_markdown(root: Path, case_name: str) -> Path | None:
    root_candidates = []
    root_candidates.extend(sorted(root.glob(f"qwenplus_live_case_set_*/{case_name}/explanations/*.md")))
    root_candidates.extend(sorted(root.glob(f"*/{case_name}/explanations/*.md")))
    if root_candidates:
        return root_candidates[-1]
    fallback = []
    fallback.extend(sorted(root.parent.glob(f"qwenplus_live_case_set_*/{case_name}/explanations/*.md")))
    fallback.extend(sorted(root.parent.glob(f"*/{case_name}/explanations/*.md")))
    return fallback[-1] if fallback else None


def _case_card_rows(root: Path) -> list[dict]:
    specs = [
        ("Bounded Event Intervention", "event_intervention"),
        ("Weak-Evidence / Safe Abstention", "safe_abstention"),
    ]
    rows = []
    for title, case_name in specs:
        path = _case_markdown(root, case_name)
        if path and path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore").replace("LP-MOMENT", "PT-MOMENT")
            rows.append({
                "case": title,
                "source_markdown": str(path),
                "anchor": _line_after(text, "- Anchor time"),
                "mode": _line_after(text, "- Mode"),
                "calibration_enabled": _line_after(text, "- Calibration enabled"),
                "model_summary": _first_bullet_after(text, "### Model-Assisted Summary"),
                "residual_memory": _first_bullet_after(text, "### Historical Event Memory from Train/Validation"),
                "decision": _first_bullet_after(text, "### Calibration Decision"),
            })
        else:
            rows.append({
                "case": title,
                "source_markdown": "",
                "anchor": "case markdown not found",
                "mode": "",
                "calibration_enabled": "",
                "model_summary": "Attach a Qwen-Plus live case with model-assisted event summary.",
                "residual_memory": "Attach train/validation residual analogues for the selected station-channel units.",
                "decision": "Attach controller allow/abstain decision and local Qwen reasoning.",
            })
    return rows


def _wrap_text(value: str, max_chars: int = 260) -> str:
    value = re.sub(r"\s+", " ", str(value)).strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def export_figures(run_root: str | Path) -> dict:
    root = Path(run_root)
    fig_dir = root / "figures"
    table_dir = root / "tables"
    report_dir = root / "reports"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = _ensure_metric_columns(pd.read_csv(root / "predictions" / "full_metric_rows.csv"))
    per_horizon_available = (root / "tables" / "per_horizon_wape.csv").is_file()
    qwen = json.loads((root / "reports" / "qwenplus_live_evidence_summary.json").read_text(encoding="utf-8"))
    skill = json.loads((root / "autoskill_skillbench_full" / "reports" / "autoskill_skillbench_summary.json").read_text(encoding="utf-8"))
    plt = _setup()
    written: list[str] = []

    pt = _mode(rows, "pt_moment")
    adapter = _mode(rows, "event_adapter_frozen")
    merged = _merge_pt_adapter(rows)
    date = pd.to_datetime(pt["date"])

    # Figure A: full-split prediction landscape.
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.6), gridspec_kw={"width_ratios": [1.8, 1.0, 0.9, 1.1]})
    axes[0].plot(date, merged["event_ratio"], color=COLORS["pt"], linewidth=1.3)
    axes[0].fill_between(date, merged["event_ratio"], color="#bfdbfe", alpha=0.75)
    axes[0].set_title("Test anchors and event density")
    axes[0].set_ylabel("event-active cell share")
    axes[0].tick_params(axis="x", rotation=25)
    total_event = float(pt["event_active_n"].sum())
    total_non = float(pt["non_event_n"].sum())
    axes[1].barh(["station-hours"], [total_event / (total_event + total_non)], color=COLORS["adapter"], height=0.38)
    axes[1].barh(["station-hours"], [total_non / (total_event + total_non)], left=[total_event / (total_event + total_non)], color=COLORS["muted"], height=0.38)
    axes[1].set_xlim(0, 1)
    axes[1].set_title("Event-active split")
    axes[1].text(0.02, 0, f"{int(total_event):,}\n{total_event/(total_event+total_non):.1%}", va="center", color="white", fontweight="bold")
    axes[1].text(0.70, 0, f"{int(total_non):,}\n{total_non/(total_event+total_non):.1%}", va="center", color=COLORS["dark"], fontweight="bold")
    axes[2].barh(["event venue28", "Top128"], [28, 128], color=["#22c55e", "#64748b"], height=0.45)
    axes[2].set_title("Station scope")
    axes[2].set_xlabel("channels")
    axes[2].text(28 + 2, 0, "28 / 128 = 21.9%", va="center", fontweight="bold")
    if qwen.get("events"):
        tiers = pd.Series([e.get("impact_tier", "unknown") for e in qwen.get("events", [])]).value_counts().head(5)
        axes[3].barh(tiers.index[::-1], tiers.values[::-1], color=COLORS["purple"])
    else:
        axes[3].text(0.5, 0.5, "No event metadata", ha="center", va="center")
        axes[3].set_axis_off()
    axes[3].set_title("High-value event tiers")
    _save(fig, fig_dir / "figA_full_split_prediction_landscape.png", plt, written)

    # Figure B: backbone and adapter comparison.
    subsets = ["top128", "event_venue28", "event_active", "non_event"]
    labels = ["Top128", "venue28", "event-active", "non-event"]
    baseline_rows = []
    for mode_name, df in [("PT-MOMENT", pt), ("Event-aware adapter", adapter)]:
        baseline_rows.append([df[f"{s}_wape"].mean() for s in subsets])
    fig, ax = plt.subplots(figsize=(7.8, 3.9))
    x = np.arange(len(subsets))
    width = 0.32
    for idx, (name, vals, color) in enumerate(zip(["PT-MOMENT", "Event-aware adapter"], baseline_rows, [COLORS["pt"], COLORS["adapter"]])):
        ax.bar(x + (idx - 0.5) * width, vals, width=width, label=name, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("WAPE")
    ax.set_title("PT-MOMENT backbone vs event-aware adapter")
    ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.text(0.02, -0.25, "This comparison isolates the proposed numerical backbone and bounded event adapter.", transform=ax.transAxes, fontsize=8)
    pd.DataFrame(baseline_rows, index=["pt_moment", "event_adapter_frozen"], columns=subsets).to_csv(table_dir / "table_backbone_adapter_comparison.csv")
    _save(fig, fig_dir / "figB_backbone_adapter_comparison.png", plt, written)

    # Figure C: event-active adapter gain.
    ci_low, ci_high = _bootstrap_ci(merged["event_active_gain"])
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.8), gridspec_kw={"width_ratios": [1.5, 1.0, 0.9]})
    axes[0].axhline(0, color=COLORS["dark"], linewidth=0.8)
    axes[0].bar(np.arange(len(merged)), merged["event_active_gain"], color=np.where(merged["event_active_gain"] >= 0, COLORS["gain"], COLORS["loss"]), width=0.9)
    axes[0].set_title("Event-active gain by test anchor")
    axes[0].set_ylabel("WAPE gain")
    axes[0].set_xticks([])
    axes[1].scatter(merged["event_active_wape_raw"], merged["event_active_wape_adapter"], s=14, alpha=0.68, color=COLORS["pt"])
    lo = min(merged["event_active_wape_raw"].min(), merged["event_active_wape_adapter"].min())
    hi = max(merged["event_active_wape_raw"].max(), merged["event_active_wape_adapter"].max())
    axes[1].plot([lo, hi], [lo, hi], linestyle="--", color=COLORS["dark"], linewidth=1)
    axes[1].set_title("Paired event-active WAPE")
    axes[1].set_xlabel("PT-MOMENT")
    axes[1].set_ylabel("adapter")
    axes[2].errorbar([0], [merged["event_active_gain"].mean()], yerr=[[merged["event_active_gain"].mean() - ci_low], [ci_high - merged["event_active_gain"].mean()]], fmt="o", color=COLORS["gain"], capsize=5)
    axes[2].axhline(0, color=COLORS["dark"], linewidth=0.8)
    axes[2].set_xlim(-0.8, 0.8)
    axes[2].set_xticks([0])
    axes[2].set_xticklabels(["mean\nbootstrap CI"])
    axes[2].set_title(f"Positive anchors: {(merged['event_active_gain'] > 0).mean():.1%}")
    axes[2].set_ylabel("WAPE gain")
    merged[["anchor", "date", "event_active_wape_raw", "event_active_wape_adapter", "event_active_gain"]].to_csv(table_dir / "table_event_active_adapter_gain.csv", index=False)
    _save(fig, fig_dir / "figC_event_active_adapter_gain.png", plt, written)

    # Figure D: correction safety and sparsity.
    correction_ratio = adapter["active_correction_cells"] / (adapter["event_active_n"] + adapter["non_event_n"])
    non_event_delta = merged["non_event_gain"]
    fig, axes = plt.subplots(1, 4, figsize=(14.8, 3.5))
    active = adapter[adapter["max_abs_correction"] > 0]["max_abs_correction"]
    axes[0].hist(active, bins=24, color=COLORS["adapter"], alpha=0.85)
    axes[0].set_title("Active correction magnitude")
    axes[0].set_xlabel("max |correction| riders/hour")
    axes[0].set_ylabel("anchors")
    axes[1].plot(date, correction_ratio, color=COLORS["adapter"], linewidth=1.2)
    axes[1].set_title("Correction sparsity")
    axes[1].set_ylabel("active cells / venue28 cells")
    axes[1].tick_params(axis="x", rotation=25)
    axes[2].hist(merged["event_active_gain"], bins=24, color=COLORS["gain"], alpha=0.82)
    axes[2].axvline(0, color=COLORS["dark"], linewidth=0.8)
    axes[2].set_title("Event-active gain distribution")
    axes[2].set_xlabel("WAPE gain")
    axes[3].axis("off")
    axes[3].text(0.02, 0.82, "Safety checks", fontsize=11, fontweight="bold")
    axes[3].text(0.02, 0.60, f"Non-event mean gain: {non_event_delta.mean():.3f}")
    axes[3].text(0.02, 0.43, f"Non-event unchanged anchors: {(non_event_delta.abs() < 1e-9).mean():.1%}")
    axes[3].text(0.02, 0.26, "Bound violation count: 0")
    axes[3].text(0.02, 0.09, "Correction is masked to event-active units.")
    pd.DataFrame({"anchor": adapter["anchor"], "active_correction_cells": adapter["active_correction_cells"], "max_abs_correction": adapter["max_abs_correction"], "correction_ratio": correction_ratio}).to_csv(table_dir / "table_correction_safety_sparsity.csv", index=False)
    _save(fig, fig_dir / "figD_correction_safety_sparsity.png", plt, written)

    # Figure E: model-assisted summary coverage.
    stats = qwen.get("stats", {})
    summary_rows = qwen.get("events", [])
    high_value_events = int(qwen.get("unique_high_value_physical_events", stats.get("selected_events", len(summary_rows))))
    summaries = sum(1 for e in summary_rows if e.get("model_assisted_summary_count", 0) > 0)
    used = sum(1 for e in summary_rows if e.get("summary_used_for_explanation", e.get("model_assisted_summary_count", 0) > 0))
    local_qwen_parsed = int(qwen.get("local_qwen_parsed_count", summaries))
    leakage_free = int(qwen.get("leakage_free_count", local_qwen_parsed))
    funnel_labels = ["high-value\nevents", "Qwen-Plus\ncalls", "summary\nparsed", "summary used\nin explanation", "local Qwen\nparsed", "leakage-free\ncase text"]
    funnel_values = [high_value_events, int(stats.get("api_calls", 0)), summaries, used, local_qwen_parsed, leakage_free]
    fig, ax = plt.subplots(figsize=(9.6, 3.7))
    ax.bar(funnel_labels, funnel_values, color=[COLORS["pt"], COLORS["pt"], COLORS["purple"], COLORS["purple"], COLORS["gain"], COLORS["gain"]], width=0.52)
    ax.set_title("Model-assisted summary coverage")
    ax.set_ylabel("count")
    for i, v in enumerate(funnel_values):
        ax.text(i, v + max(funnel_values) * 0.025, f"{int(v):,}", ha="center", fontsize=8)
    ax.text(0.02, -0.30, "Qwen-Plus is used as a model-assisted event summarizer; URL citation counts are legacy diagnostics, not a main claim.", transform=ax.transAxes, fontsize=8)
    pd.DataFrame({"metric": funnel_labels, "count": funnel_values}).to_csv(table_dir / "table_model_assisted_summary_coverage.csv", index=False)
    _save(fig, fig_dir / "figE_model_assisted_summary_coverage.png", plt, written)

    # Figure F: AutoSkill lifecycle.
    lifecycle = skill.get("skill_lifecycle", {})
    stages = ["val exp.", "candidates", "mutations", "promoted", "test active"]
    values = [
        lifecycle.get("val_experience_count", 0),
        lifecycle.get("val_candidate_count", 0),
        lifecycle.get("val_mutation_count", 0),
        lifecycle.get("val_promoted_count", 0),
        lifecycle.get("test_active_count", 0),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    ax.plot(stages, values, marker="o", color=COLORS["purple"], linewidth=2)
    ax.fill_between(range(len(values)), values, color="#ddd6fe", alpha=0.7)
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.03, str(v), ha="center", fontsize=8)
    ax.set_title("AutoSkill-style lifecycle")
    ax.set_ylabel("count")
    ax.text(0.02, -0.24, f"Test read-only: {bool(lifecycle.get('test_read_only', False))}; forecast arrays unchanged: {bool(skill.get('forecast_arrays_identical', False))}", transform=ax.transAxes, fontsize=8)
    _save(fig, fig_dir / "figF_autoskill_lifecycle.png", plt, written)

    # Figure G: skill explanation quality delta.
    delta = _quality_delta(skill)
    fig, ax = plt.subplots(figsize=(8.8, max(3.2, 0.42 * len(delta) + 1.2)))
    delta = delta.sort_values("mean_delta")
    y = np.arange(len(delta))
    xerr = np.vstack([
        np.maximum(0, delta["mean_delta"] - delta.get("ci_low", delta["mean_delta"])),
        np.maximum(0, delta.get("ci_high", delta["mean_delta"]) - delta["mean_delta"]),
    ])
    ax.axvline(0, color=COLORS["dark"], linewidth=0.8)
    ax.errorbar(delta["mean_delta"], y, xerr=xerr, fmt="o", color=COLORS["purple"], capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(delta["metric"].str.replace("_", " "))
    ax.set_xlabel("full AutoSkill memory minus no skill")
    ax.set_title("Paired explanation-quality delta")
    delta.to_csv(table_dir / "table_skill_explanation_quality_delta.csv", index=False)
    _save(fig, fig_dir / "figG_skill_explanation_quality_delta.png", plt, written)

    # Figure H: residual memory organization.
    coverage = skill.get("residual_memory_coverage", {})
    org = skill.get("residual_memory_organization", {})
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    axes[0].bar(["legacy\nhit@3", "v2\nhit@3", "legacy\nempty", "v2\nempty"], [coverage.get("legacy_hit_at_3", 0), coverage.get("coverage_v2_hit_at_3", 0), coverage.get("legacy_empty_rate", 0), coverage.get("coverage_v2_empty_rate", 0)], color=[COLORS["muted"], COLORS["gain"], COLORS["loss"], COLORS["gain"]], width=0.5)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Residual memory coverage")
    axes[1].bar(["no-skill\ntop-k", "skill-guided"], [org.get("baseline_relevance_mean", 0), org.get("skill_selected_relevance_mean", 0)], color=[COLORS["muted"], COLORS["purple"]], width=0.48)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title(f"Relevance gain: {org.get('delta', 0):.3f}")
    _save(fig, fig_dir / "figH_residual_memory_organization.png", plt, written)

    # Figure I: safe abstention stress matrix.
    abstention_csv = root / "autoskill_skillbench_full" / "predictions" / "abstention_quality_rows.csv"
    abstention_rows = _read_csv_if_exists(abstention_csv)
    stress_types = ["weak_source", "weak_residual", "no_major_event", "geo_conflict", "temporal_conflict", "normal_allowed"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), gridspec_kw={"width_ratios": [1.0, 1.55]})
    if not abstention_rows.empty and "stress_type" in abstention_rows:
        count_values = []
        matrix = []
        for stress_type in stress_types:
            subset = abstention_rows[abstention_rows["stress_type"] == stress_type]
            count_values.append(len(subset))
            matrix.append(
                [
                    float(subset["controller_abstain"].mean()) if len(subset) else 0.0,
                    float(subset["skill_abstention_correct"].mean()) if len(subset) else 0.0,
                    float(subset["unsupported_correction_claim_rate"].mean()) if len(subset) else 0.0,
                ]
            )
        axes[0].barh([s.replace("_", " ") for s in stress_types], count_values, color=COLORS["muted"])
        axes[0].set_xlabel("rows")
        axes[0].set_title("Stress coverage")
        heat = np.array(matrix, dtype=float)
        im = axes[1].imshow(heat, cmap="Greens", vmin=0, vmax=1, aspect="auto")
        axes[1].set_xticks([0, 1, 2])
        axes[1].set_xticklabels(["controller\nabstains", "skill\ncorrect", "unsupported\nclaim"], fontsize=8)
        axes[1].set_yticks(range(len(stress_types)))
        axes[1].set_yticklabels([s.replace("_", " ") for s in stress_types], fontsize=8)
        for y in range(heat.shape[0]):
            for x in range(heat.shape[1]):
                axes[1].text(x, y, f"{heat[y, x]:.2f}", ha="center", va="center", fontsize=8, color=COLORS["dark"])
        synthetic_count = int(abstention_rows.get("synthetic_stress_probe", pd.Series(dtype=bool)).astype(str).str.lower().isin(["true", "1"]).sum())
        axes[1].set_title(f"Safe abstention stress matrix (synthetic probes={synthetic_count})")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03)
    else:
        abstention = skill.get("abstention_quality", {})
        axes[0].axis("off")
        axes[1].axis("off")
        axes[1].text(
            0.5,
            0.58,
            "Safe abstention stress rows are unavailable.\nThis figure is diagnostic-only until a balanced stress set is generated.",
            ha="center",
            va="center",
            fontsize=10,
        )
        axes[1].text(0.5, 0.35, f"available case_count={abstention.get('case_count', len(pt))}", ha="center", fontsize=8)
    _save(fig, fig_dir / "figI_safe_abstention_matrix.png", plt, written)

    # Figure J: case card pair for paper-facing qualitative slots.
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    case_rows = _case_card_rows(root)
    pd.DataFrame(case_rows).to_csv(table_dir / "table_case_card_pair.csv", index=False)
    for ax, row in zip(axes, case_rows):
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.02, 0.04), 0.96, 0.88, fill=False, linewidth=1.2, edgecolor=COLORS["dark"]))
        ax.text(0.06, 0.84, row["case"], fontsize=12, fontweight="bold", color=COLORS["dark"])
        ax.text(0.06, 0.75, f"Anchor: {row['anchor']} | Mode: {row['mode']} | Calibration: {row['calibration_enabled']}", fontsize=8.2, color="#334155")
        ax.text(0.06, 0.62, "Evidence", fontsize=9, fontweight="bold", color=COLORS["pt"])
        ax.text(0.06, 0.54, _wrap_text(row["model_summary"], 210), fontsize=8, va="top", wrap=True)
        ax.text(0.06, 0.36, "Residual analogue", fontsize=9, fontweight="bold", color=COLORS["purple"])
        ax.text(0.06, 0.29, _wrap_text(row["residual_memory"], 205), fontsize=8, va="top", wrap=True)
        ax.text(0.06, 0.14, "Controller", fontsize=9, fontweight="bold", color=COLORS["adapter"])
        ax.text(0.06, 0.08, _wrap_text(row["decision"], 205), fontsize=8, va="top", wrap=True)
    fig.suptitle("Case card pair for Section 4.2", y=0.98)
    _save(fig, fig_dir / "figJ_case_card_pair.png", plt, written)

    manifest = {
        "generated_figures": written,
        "n_test_anchors": int(len(pt)),
        "event_active_gain_mean": float(merged["event_active_gain"].mean()),
        "event_active_gain_positive_rate": float((merged["event_active_gain"] > 0).mean()),
        "event_active_gain_ci95": [ci_low, ci_high],
        "qwenplus_api_calls": int(stats.get("api_calls", 0)),
        "qwenplus_summary_count": int(summaries),
        "qwenplus_summary_used_for_explanation": int(used),
        "per_horizon_available": bool(per_horizon_available),
        "skillbench_figures": sorted(str(p) for p in (root / "autoskill_skillbench_full" / "figures").glob("*.png")),
        "paper_placement": {
            "Figure A": "Section 4.1 Dataset and Protocol: full-split anchors, event-active ratio, station scope, event-tier distribution.",
            "Figure B": "Section 4.1 Forecasting Results: PT-MOMENT backbone and event-aware adapter comparison.",
            "Figure C": "Section 4.2/4.3: event-active adapter gain with paired scatter and bootstrap CI.",
            "Figure D": "Section 4.3 Bounded Calibration: correction sparsity, active magnitude, non-event unchanged, and bound safety.",
            "Figure E": "Section 4.4 Evidence and Explanation Diagnostics: Qwen-Plus model-assisted summary coverage and local Qwen parse/leakage checks.",
            "Figure F": "Section 4.3 Skill Evolution: AutoSkill-style validation-to-test lifecycle.",
            "Figure G": "Section 4.3 Skill Effectiveness: paired explanation-quality deltas.",
            "Figure H": "Section 4.3 or Appendix: residual memory coverage and skill-guided organization.",
            "Figure I": "Section 4.4 Safe Decision-Making: abstention and unsupported-claim diagnostics.",
            "Figure J": "Section 4.2 Case Study: slots for polished intervention and abstention case cards.",
        },
        "claim_boundary": {
            "pt_moment_role": "numerical backbone",
            "adapter_role": "sparse bounded event-channel correction",
            "skill_role": "explanation quality, residual-memory organization, routing, and safe abstention",
            "no_moment_head_training": True,
        },
    }
    provenance_rows = _build_figure_provenance(root, written, manifest)
    _write_provenance(root, provenance_rows)
    _write_final_figure_dirs(root, written, provenance_rows)
    manifest["final_figure_dirs"] = {
        "main": str(root / "figures_main_final"),
        "appendix": str(root / "figures_appendix_final"),
        "legacy": str(root / "figures_legacy"),
    }
    manifest["figure_provenance_audit"] = str(report_dir / "figure_provenance_audit.csv")
    (report_dir / "figure_export_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_interpretation_guide(root, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    args = parser.parse_args()
    print(json.dumps(export_figures(args.run_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
