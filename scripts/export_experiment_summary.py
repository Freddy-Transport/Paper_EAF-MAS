#!/usr/bin/env python3
"""
Export experiment summary (Table 1), JSON metrics, and paper figures
from traces_experiment/.

Usage:
  python scripts/export_experiment_summary.py \
      --trace-dir traces_experiment \
      --outdir figures_experiment \
      --csv data/nyc_top128_station_hourly_flow.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualize_agent import COLORS, _collect_events_for_channel, load_ground_truth


def _run_type(path: Path) -> str:
    return "major" if "major" in path.stem else "control"


def _parse_date(path: Path, trace: dict) -> str:
    return trace.get("target_date") or re.search(r"\d{4}-\d{2}-\d{2}", path.stem).group(0)


def mae(pred: np.ndarray, truth: np.ndarray) -> float | None:
    mask = ~np.isnan(truth)
    if mask.sum() == 0:
        return None
    return float(np.mean(np.abs(pred[mask] - truth[mask])))


def mean_adj_pct(raw: np.ndarray, adj: np.ndarray) -> float:
    return float(np.mean(np.abs(adj - raw) / (np.abs(raw) + 1e-6)) * 100)


def event_direction_stats(trace: dict, ch: str, gt: np.ndarray, ts: pd.DatetimeIndex, raw: np.ndarray) -> dict:
    """Per-event direction correctness vs GT residual in event window."""
    ts_start, ts_end = ts[0], ts[-1]
    events = _collect_events_for_channel(trace, ch, ts_start, ts_end)
    n_total = len(events)
    if n_total == 0:
        return {
            "n_events": 0,
            "n_scored": 0,
            "n_direction_correct": 0,
            "direction_accuracy": None,
        }

    n_correct = 0
    n_scored = 0
    for ev in events:
        if ev["direction"] == "neutral":
            continue
        ev_t = ev["time"]
        dur_h = max(1, int(ev["duration"]))
        win = (ts >= ev_t) & (ts < ev_t + pd.Timedelta(hours=dur_h))
        idx_w = np.where(win)[0]
        if len(idx_w) == 0:
            continue
        m_valid = ~np.isnan(gt[idx_w]) & (raw[idx_w] > 0)
        if m_valid.sum() == 0:
            continue
        gt_delta = float(
            np.mean((gt[idx_w][m_valid] - raw[idx_w][m_valid]) / (raw[idx_w][m_valid] + 1e-6))
        )
        n_scored += 1
        correct = (gt_delta > 0.01 and ev["direction"] == "increase") or (
            gt_delta < -0.01 and ev["direction"] == "decrease"
        )
        if correct:
            n_correct += 1

    acc = n_correct / n_scored if n_scored else None
    return {
        "n_events": n_total,
        "n_scored": n_scored,
        "n_direction_correct": n_correct,
        "direction_accuracy": acc,
    }


def summarize_trace(path: Path, csv_path: str) -> dict:
    with open(path) as f:
        trace = json.load(f)

    ch = list(trace["raw_forecast_target"].keys())[0]
    raw = np.array(trace["raw_forecast_target"][ch], dtype=float)
    adj = np.array(trace["adjusted_forecast_target"][ch], dtype=float)
    ts_list = trace["forecast_timestamps"]
    n = min(24, len(raw))
    raw, adj = raw[:n], adj[:n]
    ts = pd.to_datetime(ts_list[:n])

    gt = load_ground_truth(csv_path, ch, ts_list[:n])
    if gt is not None:
        gt = gt[:n]

    m_r = mae(raw, gt) if gt is not None else None
    m_a = mae(adj, gt) if gt is not None else None
    impr = (m_r - m_a) / m_r * 100 if m_r and m_a and m_r > 0 else None

    run_type = _run_type(path)
    dir_stats = (
        event_direction_stats(trace, ch, gt, ts, raw)
        if gt is not None
        else {"n_events": len(trace.get("events_considered", [])), "direction_accuracy": None}
    )

    return {
        "run": int(re.search(r"run(\d+)", path.stem).group(1)),
        "trace_file": path.name,
        "run_type": run_type,
        "date": _parse_date(path, trace),
        "raw_mae": round(m_r, 1) if m_r is not None else None,
        "adj_mae": round(m_a, 1) if m_a is not None else None,
        "mae_improvement_pct": round(impr, 2) if impr is not None else None,
        "mean_adj_pct": round(mean_adj_pct(raw, adj), 2),
        "n_events_considered": len(trace.get("events_considered", [])),
        **{k: dir_stats[k] for k in ("n_events", "n_scored", "n_direction_correct", "direction_accuracy")},
    }


def aggregate(runs: list[dict], run_type: str) -> dict:
    subset = [r for r in runs if r["run_type"] == run_type]
    if not subset:
        return {}

    def avg(key):
        vals = [r[key] for r in subset if r.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    accs = [r["direction_accuracy"] for r in subset if r.get("direction_accuracy") is not None]
    return {
        "n_runs": len(subset),
        "mean_raw_mae": avg("raw_mae"),
        "mean_adj_mae": avg("adj_mae"),
        "mean_mae_improvement_pct": avg("mae_improvement_pct"),
        "mean_adj_pct": avg("mean_adj_pct"),
        "mean_direction_accuracy": round(sum(accs) / len(accs), 3) if accs else None,
        "total_events": sum(r.get("n_events_considered", 0) for r in subset),
    }


def write_table1_tex(path: Path, runs: list[dict], major: dict, control: dict, pooled: dict):
    lines = [
        r"% Auto-generated by scripts/export_experiment_summary.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ten-run experiment on Times Sq-42 St (N060): major event days vs.\ control days.}",
        r"\label{tab:experiment10}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Run & Type & Date & Raw MAE & Adj MAE & $\Delta$MAE (\%) & Dir.\ Acc. \\",
        r"\midrule",
    ]
    for r in sorted(runs, key=lambda x: x["run"]):
        acc = (
            f"{r['direction_accuracy']*100:.0f}\\%"
            if r.get("direction_accuracy") is not None
            else "---"
        )
        impr = f"{r['mae_improvement_pct']:+.1f}" if r.get("mae_improvement_pct") is not None else "0.0"
        lines.append(
            f"{r['run']:02d} & {r['run_type']} & {r['date']} & {r['raw_mae']:.1f} & {r['adj_mae']:.1f} & {impr} & {acc} \\\\"
        )
    maj_acc = (
        f"{major['mean_direction_accuracy']*100:.0f}\\%"
        if major.get("mean_direction_accuracy") is not None
        else "---"
    )
    lines += [
        r"\midrule",
        f"Major mean & major & --- & {major['mean_raw_mae']:.1f} & {major['mean_adj_mae']:.1f} & "
        f"{major['mean_mae_improvement_pct']:+.1f} & {maj_acc} \\\\",
        f"Control mean & control & --- & {control['mean_raw_mae']:.1f} & {control['mean_adj_mae']:.1f} & "
        f"{control['mean_mae_improvement_pct']:+.1f} & --- \\\\",
        f"Pooled & all & --- & {pooled['mean_raw_mae']:.1f} & {pooled['mean_adj_mae']:.1f} & "
        f"{pooled['mean_mae_improvement_pct']:+.1f} & --- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def fig_major_control_bar(runs: list[dict], outpath: Path):
    major = [r for r in runs if r["run_type"] == "major"]
    control = [r for r in runs if r["run_type"] == "control"]

    groups = ["Major (n=5)", "Control (n=5)"]
    raw_means = [
        np.mean([r["raw_mae"] for r in major]),
        np.mean([r["raw_mae"] for r in control]),
    ]
    adj_means = [
        np.mean([r["adj_mae"] for r in major]),
        np.mean([r["adj_mae"] for r in control]),
    ]

    x = np.arange(len(groups))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - w / 2, raw_means, w, label="Raw (MOMENT-LP)", color=COLORS["raw"])
    b2 = ax.bar(x + w / 2, adj_means, w, label="Agent-Adjusted", color=COLORS["adjusted"])
    ax.set_ylabel("Mean MAE (entries/h)")
    ax.set_title("Major vs Control Days — Mean MAE (Times Sq-42 St)")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 15, f"{h:.0f}", ha="center", fontsize=9)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {outpath}")


def fig_per_run_delta(runs: list[dict], outpath: Path):
    runs = sorted(runs, key=lambda x: x["run"])
    labels = [f"R{r['run']:02d}\n{r['date'][5:]}" for r in runs]
    deltas = [r.get("mae_improvement_pct") or 0.0 for r in runs]
    colors = []
    for r, d in zip(runs, deltas):
        if r["run_type"] == "control":
            colors.append("#95A5A6")
        elif d > 0:
            colors.append("#27AE60")
        else:
            colors.append("#E74C3C")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    bars = ax.bar(labels, deltas, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("MAE improvement (%)  [(Raw − Adj) / Raw]")
    ax.set_title("Per-Run MAE Change — Green = major win, Red = major loss, Gray = control")
    for bar, r in zip(bars, runs):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + (0.3 if h >= 0 else -0.8),
            f"{h:+.1f}%",
            ha="center",
            fontsize=8,
            color="black",
        )
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#27AE60", label="Major (improved)"),
            Patch(facecolor="#E74C3C", label="Major (degraded)"),
            Patch(facecolor="#95A5A6", label="Control (no adjustment)"),
        ],
        loc="upper right",
        fontsize=8,
    )
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", default="traces_experiment")
    parser.add_argument("--outdir", default="figures_experiment")
    parser.add_argument("--csv", default="data/nyc_top128_station_hourly_flow.csv")
    parser.add_argument("--summary-json", default="traces_experiment/experiment_summary.json")
    args = parser.parse_args()

    trace_dir = ROOT / args.trace_dir
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    paths = sorted(trace_dir.glob("trace_run*.json"))
    if not paths:
        print(f"No traces in {trace_dir}")
        sys.exit(1)

    print(f"Summarizing {len(paths)} traces...")
    runs = [summarize_trace(p, str(ROOT / args.csv)) for p in paths]
    major_agg = aggregate(runs, "major")
    control_agg = aggregate(runs, "control")
    pooled_agg = aggregate(runs, "major")  # placeholder
    all_raw = [r["raw_mae"] for r in runs if r["raw_mae"]]
    all_adj = [r["adj_mae"] for r in runs if r["adj_mae"]]
    pooled_agg = {
        "n_runs": len(runs),
        "mean_raw_mae": round(sum(all_raw) / len(all_raw), 2),
        "mean_adj_mae": round(sum(all_adj) / len(all_adj), 2),
        "mean_mae_improvement_pct": round(
            sum(r.get("mae_improvement_pct") or 0 for r in runs) / len(runs), 2
        ),
    }

    summary = {
        "query": "Times Sq-42 St",
        "station": "N060",
        "n_runs": len(runs),
        "runs": runs,
        "aggregate": {"major": major_agg, "control": control_agg, "pooled": pooled_agg},
        "claims": {
            "C1_control_no_harm": all(
                r["mean_adj_pct"] == 0 and r["raw_mae"] == r["adj_mae"]
                for r in runs
                if r["run_type"] == "control"
            ),
            "C3_major_improved_fraction": sum(
                1 for r in runs if r["run_type"] == "major" and (r.get("mae_improvement_pct") or 0) > 0
            )
            / max(1, sum(1 for r in runs if r["run_type"] == "major")),
        },
    }

    summary_path = ROOT / args.summary_json
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {summary_path}")

    tex_path = outdir / "table1_experiment.tex"
    write_table1_tex(tex_path, runs, major_agg, control_agg, pooled_agg)

    fig_major_control_bar(runs, outdir / "fig_major_control_bar.png")
    fig_per_run_delta(runs, outdir / "fig_per_run_delta.png")

    # Markdown table for quick reference
    md_lines = ["# Table 1 — Experiment Summary\n", "| Run | Type | Date | Raw MAE | Adj MAE | ΔMAE% | Dir. Acc. |", "|-----|------|------|---------|---------|-------|-----------|"]
    for r in sorted(runs, key=lambda x: x["run"]):
        acc = f"{r['direction_accuracy']*100:.0f}%" if r.get("direction_accuracy") is not None else "—"
        impr = f"{r['mae_improvement_pct']:+.1f}" if r.get("mae_improvement_pct") is not None else "0.0"
        md_lines.append(
            f"| {r['run']:02d} | {r['run_type']} | {r['date']} | {r['raw_mae']} | {r['adj_mae']} | {impr} | {acc} |"
        )
    (outdir / "table1_experiment.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"  ✓ {outdir / 'table1_experiment.md'}")


if __name__ == "__main__":
    main()
