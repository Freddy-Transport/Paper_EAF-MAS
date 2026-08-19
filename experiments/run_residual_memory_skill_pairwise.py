#!/usr/bin/env python3
"""Pairwise explanation-quality study for residual-memory skill modes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


QUALITY_METRICS = [
    "evidence_coverage",
    "residual_case_relevance",
    "multi_hop_completeness",
    "groundedness",
    "unsupported_claim_rate",
    "leakage_free_rate",
    "redundancy_rate",
    "skill_guidance_coverage",
    "calibration_decision_consistent",
]
SANITY_METRICS = ["wape", "raw_wape", "adjusted_wape"]
MODES = ["residual_rag_no_skill", "residual_rag_static_skill", "residual_rag_evolving_skill"]


def _load_payloads(paths: Sequence[str]) -> List[dict]:
    payloads = []
    for path in paths:
        p = Path(path)
        if p.is_file():
            payloads.append(json.loads(p.read_text(encoding="utf-8")))
    return payloads


def _window_key(payload: dict) -> str:
    req = payload.get("request") or {}
    return str(req.get("date") or req.get("target_date") or payload.get("window_id") or "")


def _quality(payload: dict) -> dict:
    q = dict(payload.get("explanation_quality") or {})
    decision = payload.get("decision") or {}
    for key in QUALITY_METRICS:
        if key not in q:
            if key == "calibration_decision_consistent":
                q[key] = True
            else:
                q[key] = 0.0
    q["calibration_decision_consistent"] = 1.0 if bool(q.get("calibration_decision_consistent")) else 0.0
    q["selected_residual_memory_skill_count"] = float(q.get("selected_residual_memory_skill_count") or 0.0)
    q["abstain"] = 1.0 if bool(decision.get("abstain")) else 0.0
    return q


def _sanity(payload: dict) -> dict:
    metrics = payload.get("metrics") or {}
    raw_metrics = payload.get("raw_metrics") or {}
    return {
        "wape": float(metrics.get("wape", 0.0) or 0.0),
        "raw_wape": float(raw_metrics.get("wape", metrics.get("raw_wape", 0.0)) or 0.0),
        "adjusted_wape": float(metrics.get("wape", 0.0) or 0.0),
    }


def _array_equal(a: object, b: object) -> bool:
    aa = np.asarray(a or [], dtype=float)
    bb = np.asarray(b or [], dtype=float)
    return aa.shape == bb.shape and bool(np.allclose(aa, bb, atol=1e-6))


def _mode_map(no_skill: Sequence[str], static_skill: Sequence[str], evolving_skill: Sequence[str]) -> Dict[str, Dict[str, dict]]:
    out: Dict[str, Dict[str, dict]] = {mode: {} for mode in MODES}
    for mode, paths in [
        ("residual_rag_no_skill", no_skill),
        ("residual_rag_static_skill", static_skill),
        ("residual_rag_evolving_skill", evolving_skill),
    ]:
        for payload in _load_payloads(paths):
            key = _window_key(payload)
            if key:
                out[mode][key] = payload
    return out


def _mean(rows: Sequence[dict], metric: str) -> float:
    values = [float(row.get(metric) or 0.0) for row in rows]
    return float(np.mean(values)) if values else 0.0


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    fields = ["window", "mode"] + QUALITY_METRICS + SANITY_METRICS + ["selected_residual_memory_skill_count"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_tex(path: Path, summary: dict) -> None:
    rows = []
    for mode in ["residual_rag_static_skill", "residual_rag_evolving_skill"]:
        delta = summary["mean_delta_vs_no_skill"].get(mode, {})
        rows.append(
            [
                mode.replace("_", "\\_"),
                f"{delta.get('evidence_coverage', 0.0):.3f}",
                f"{delta.get('residual_case_relevance', 0.0):.3f}",
                f"{delta.get('multi_hop_completeness', 0.0):.3f}",
                f"{delta.get('groundedness', 0.0):.3f}",
                f"{delta.get('skill_guidance_coverage', 0.0):.3f}",
                f"{delta.get('unsupported_claim_rate', 0.0):.3f}",
            ]
        )
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Paired explanation-quality deltas over the no-skill residual-RAG baseline. Positive values are better except unsupported-claim rate.}",
        "\\label{tab:residual_memory_pairwise}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Mode & Evidence & ResidualRel. & Multi-hop & Grounded & SkillGuide & Unsup. \\\\",
        "\\midrule",
    ]
    lines.extend(" & ".join(row) + " \\\\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_figure(path: Path, summary: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["evidence_coverage", "residual_case_relevance", "multi_hop_completeness", "groundedness", "skill_guidance_coverage"]
    modes = ["residual_rag_static_skill", "residual_rag_evolving_skill"]
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    for i, mode in enumerate(modes):
        values = [summary["mean_delta_vs_no_skill"].get(mode, {}).get(metric, 0.0) for metric in metrics]
        ax.bar(x + (i - 0.5) * width, values, width=width, label=mode.replace("residual_rag_", ""))
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Evidence", "ResidualRel.", "Multi-hop", "Grounded", "SkillGuide"], rotation=15, ha="right")
    ax.set_ylabel("Delta vs no skill")
    ax.set_title("Residual-memory skill improves explanation organization metrics")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_md(path: Path, summary: dict) -> None:
    lines = [
        "# Residual-Memory Skill Pairwise Explanation Study",
        "",
        f"- window_count: `{summary['window_count']}`",
        f"- forecast_arrays_identical: `{summary['forecast_arrays_identical']}`",
        "- interpretation: skill modes are evaluated on explanation quality, not forecasting accuracy.",
        "",
        "## Mean Quality",
        "",
    ]
    for mode, values in summary["mean_quality_by_mode"].items():
        lines.append(f"### {mode}")
        for metric in QUALITY_METRICS:
            lines.append(f"- {metric}: `{values.get(metric, 0.0):.4f}`")
        lines.append("")
    lines.append("## Delta vs No Skill")
    for mode, values in summary["mean_delta_vs_no_skill"].items():
        lines.append(f"### {mode}")
        for metric, value in values.items():
            lines.append(f"- {metric}: `{value:.4f}`")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_pairwise_study(
    output_root: str | Path,
    no_skill_prediction_json: Sequence[str],
    static_skill_prediction_json: Sequence[str],
    evolving_skill_prediction_json: Sequence[str],
) -> dict:
    run_dir = Path(output_root)
    for sub in ["reports", "tables", "figures", "predictions"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    payloads = _mode_map(no_skill_prediction_json, static_skill_prediction_json, evolving_skill_prediction_json)
    common_windows = sorted(set(payloads["residual_rag_no_skill"]) & set(payloads["residual_rag_static_skill"]) & set(payloads["residual_rag_evolving_skill"]))
    rows: List[dict] = []
    forecast_arrays_identical = True
    for window in common_windows:
        base = payloads["residual_rag_no_skill"][window]
        for mode in MODES:
            payload = payloads[mode][window]
            if not _array_equal(base.get("numerical", {}).get("raw_forecast"), payload.get("numerical", {}).get("raw_forecast")):
                forecast_arrays_identical = False
            if not _array_equal(base.get("adjusted_forecast"), payload.get("adjusted_forecast")):
                forecast_arrays_identical = False
            row = {"window": window, "mode": mode}
            row.update(_quality(payload))
            row.update(_sanity(payload))
            rows.append(row)
    _write_csv(run_dir / "predictions" / "pairwise_quality_rows.csv", rows)

    by_mode = {mode: [row for row in rows if row["mode"] == mode] for mode in MODES}
    mean_quality = {mode: {metric: _mean(mode_rows, metric) for metric in QUALITY_METRICS} for mode, mode_rows in by_mode.items()}
    mean_quality["residual_rag_no_skill"]["selected_residual_memory_skill_count"] = _mean(by_mode["residual_rag_no_skill"], "selected_residual_memory_skill_count")
    for mode in ["residual_rag_static_skill", "residual_rag_evolving_skill"]:
        mean_quality[mode]["selected_residual_memory_skill_count"] = _mean(by_mode[mode], "selected_residual_memory_skill_count")
    delta = {}
    base = mean_quality.get("residual_rag_no_skill", {})
    for mode in ["residual_rag_static_skill", "residual_rag_evolving_skill"]:
        delta[mode] = {metric: mean_quality.get(mode, {}).get(metric, 0.0) - base.get(metric, 0.0) for metric in QUALITY_METRICS}
        delta[mode]["selected_residual_memory_skill_count"] = mean_quality.get(mode, {}).get("selected_residual_memory_skill_count", 0.0) - base.get("selected_residual_memory_skill_count", 0.0)

    sanity = {mode: {metric: _mean(mode_rows, metric) for metric in SANITY_METRICS} for mode, mode_rows in by_mode.items()}
    summary = {
        "window_count": len(common_windows),
        "modes": MODES,
        "quality_metrics": QUALITY_METRICS,
        "sanity_metrics": sanity,
        "forecast_arrays_identical": bool(forecast_arrays_identical),
        "mean_quality_by_mode": mean_quality,
        "mean_delta_vs_no_skill": delta,
    }
    (run_dir / "reports" / "residual_memory_skill_pairwise_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_md(run_dir / "reports" / "residual_memory_skill_pairwise_summary.md", summary)
    _write_tex(run_dir / "tables" / "explanation_quality_pairwise.tex", summary)
    _write_figure(run_dir / "figures" / "skill_pairwise_quality_delta.png", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default=f"autotemp/residual_memory_skill_pairwise_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--no_skill_prediction_json", action="append", default=[])
    parser.add_argument("--static_skill_prediction_json", action="append", default=[])
    parser.add_argument("--evolving_skill_prediction_json", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_pairwise_study(
        output_root=args.output_root,
        no_skill_prediction_json=args.no_skill_prediction_json,
        static_skill_prediction_json=args.static_skill_prediction_json,
        evolving_skill_prediction_json=args.evolving_skill_prediction_json,
    )
    print(json.dumps({"run_dir": str(args.output_root), **summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
