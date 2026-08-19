#!/usr/bin/env python3
"""Venue37 Fusion 前后预测对比 — 自动选取有事件日期/站点，区分线型可视化."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FRAMEWORK = Path("/root/autodl-tmp/0206moment")
DATA_ROOT = Path("/root/autodl-tmp/纽约地铁数据处理")
sys.path.insert(0, str(FRAMEWORK))

from agents.config import load_env_file

load_env_file()

# ── 绘图样式（MOMENT 虚线 / Fusion 实线 / GT 点划线+标记）────────────────────
STYLE = {
    "moment": {
        "color": "#4A6FA5",
        "ls": "--",
        "lw": 2.0,
        "alpha": 0.95,
        "zorder": 2,
        "label": "MOMENT (numerical)",
    },
    "fusion": {
        "color": "#D35400",
        "ls": "-",
        "lw": 2.4,
        "alpha": 0.98,
        "zorder": 4,
        "label": "Fusion (adjusted)",
    },
    "gt": {
        "color": "#1B7F3A",
        "ls": "-",
        "lw": 2.2,
        "alpha": 0.9,
        "zorder": 3,
        "marker": "o",
        "ms": 3.5,
        "markevery": 4,
        "label": "Ground Truth",
    },
    "event_span": {"color": "#F39C12", "alpha": 0.18},
    "event_line": {"color": "#E67E22", "lw": 1.5, "ls": ":"},
}


def mae_mape(pred: list[float], gt: list[float]) -> tuple[float, float]:
    p, g = np.array(pred, dtype=float), np.array(gt, dtype=float)
    mask = np.isfinite(p) & np.isfinite(g) & (g > 0)
    if mask.sum() == 0:
        return float("nan"), float("nan")
    err = np.abs(p[mask] - g[mask])
    return float(err.mean()), float((err / g[mask]).mean() * 100)


def _parse_ts(ts_list: list[str]) -> pd.DatetimeIndex:
    return pd.to_datetime(ts_list)


def plot_forecast_panel(
    ax: plt.Axes,
    ts: pd.DatetimeIndex,
    raw: np.ndarray,
    adj: np.ndarray,
    gt: np.ndarray | None,
    title: str,
    event_times: list[pd.Timestamp] | None = None,
    event_labels: list[str] | None = None,
    show_legend: bool = True,
) -> None:
    """绘制单通道预测对比，线型/颜色明确区分."""
    sm, sf, sg = STYLE["moment"], STYLE["fusion"], STYLE["gt"]
    ax.plot(ts, raw, **{k: sm[k] for k in ("color", "ls", "lw", "alpha", "zorder")}, label=sm["label"])
    ax.plot(ts, adj, **{k: sf[k] for k in ("color", "ls", "lw", "alpha", "zorder")}, label=sf["label"])
    if gt is not None and np.any(np.isfinite(gt)):
        ax.plot(
            ts, gt,
            color=sg["color"], ls=sg["ls"], lw=sg["lw"], alpha=sg["alpha"], zorder=sg["zorder"],
            marker=sg["marker"], markersize=sg["ms"], markevery=sg["markevery"],
            label=sg["label"],
        )

    if event_times:
        for i, ev_t in enumerate(event_times):
            if ts[0] <= ev_t <= ts[-1]:
                ax.axvline(ev_t, **STYLE["event_line"])
                ax.axvspan(ev_t, ev_t + pd.Timedelta(hours=4), **STYLE["event_span"])
                if event_labels and i < len(event_labels):
                    ax.annotate(
                        event_labels[i][:40],
                        xy=(ev_t, ax.get_ylim()[1]),
                        xytext=(4, -8),
                        textcoords="offset points",
                        fontsize=7,
                        color=STYLE["event_line"]["color"],
                        rotation=0,
                        va="top",
                    )

    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_ylabel("Ridership / h", fontsize=8)
    ax.grid(alpha=0.25, ls=":")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    if show_legend:
        ax.legend(loc="upper left", fontsize=7, framealpha=0.92)


def pick_event_cases(
    events_path: Path,
    fusion_channels: set[str],
    channel_map: dict[str, dict],
    n_cases: int = 5,
    test_start: str = "2023-03-17",
    test_end: str = "2023-06-15",
) -> list[dict]:
    """从验真事件中选取有 channel、高 lift 的 (预测起始日, 通道, 事件) 组合."""
    if events_path.suffix == ".json":
        events = json.loads(events_path.read_text(encoding="utf-8")).get("events", [])
        df = pd.DataFrame(events)
    else:
        df = pd.read_csv(events_path)

    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df = df[df["event_time"].notna()]
    df = df[(df["event_time"] >= test_start) & (df["event_time"] <= test_end)]

    sid_to_ch = {m["station_complex_id"]: m["channel_name"] for m in channel_map.values()}

    if "channel_name" not in df.columns or df["channel_name"].isna().all():
        df["channel_name"] = df["station_complex_id"].map(sid_to_ch)

    df = df[df["channel_name"].isin(fusion_channels)]
    df["abs_lift"] = pd.to_numeric(df.get("lift_pct", 0), errors="coerce").abs()
    df = df.sort_values("abs_lift", ascending=False)

    cases: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for _, row in df.iterrows():
        ch = str(row["channel_name"])
        ev_t = row["event_time"]
        # 预测窗口起始：事件前 2 天，确保事件落在 192h 窗口中部
        pred_start = (ev_t - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        key = (pred_start, ch)
        if key in seen:
            continue
        seen.add(key)
        title = str(row.get("title", "") or row.get("event_name", "") or "Event")
        if title == "nan" or not title.strip():
            title = str(row.get("event_type", "Event"))
        cases.append(
            {
                "target_date": pred_start,
                "channel_name": ch,
                "station_complex": row.get("station_complex", ch),
                "event_time": ev_t.strftime("%Y-%m-%d %H:%M:%S"),
                "event_title": title[:80],
                "lift_pct": float(row.get("lift_pct", 0) or 0),
                "impact_tier": str(row.get("impact_tier", "")),
            }
        )
        if len(cases) >= n_cases:
            break
    return cases


def _events_in_window(events_json: Path, ch: str, ts_start: pd.Timestamp, ts_end: pd.Timestamp) -> tuple[list, list]:
    payload = json.loads(events_json.read_text(encoding="utf-8"))
    times, labels = [], []
    for ev in payload.get("events", []):
        if ev.get("channel_name") != ch:
            continue
        try:
            et = pd.to_datetime(ev.get("event_time", ""))
        except Exception:
            continue
        if ts_start <= et <= ts_end:
            times.append(et)
            labels.append(str(ev.get("title", "") or ev.get("event_type", ""))[:50])
    return times, labels


def run_comparison(
    target_date: str,
    horizon: int,
    device: str,
    outdir: Path,
    focus_channels: list[str] | None = None,
    case_meta: dict | None = None,
) -> dict:
    from agents.orchestrator import DynamicOrchestrator

    fusion_cfg = json.loads((FRAMEWORK / "data" / "venue37_fusion_channels.json").read_text())
    fusion_channels = fusion_cfg["channel_names"]
    station_ids = fusion_cfg["station_complex_ids"]
    fusion_set = set(fusion_channels)

    channel_map_list = json.loads((FRAMEWORK / "data" / "nyc_top128_channel_map.json").read_text())["channels"]
    all_ch_names = [ch["channel_name"] for ch in channel_map_list]
    ch_meta = {ch["channel_name"]: ch for ch in channel_map_list}

    events_json = DATA_ROOT / "outputs/venue37_analysis/venue37_verified_events.json"
    kb_dir = FRAMEWORK / "agents/knowledge_base_venue37"
    residual_kb = FRAMEWORK / "agents/knowledge_base_residual_venue37"

    import os
    from agents import config as cfg

    if residual_kb.is_dir():
        os.environ["RESIDUAL_KB_DIR"] = str(residual_kb)
        cfg.RESIDUAL_KB_DIR = str(residual_kb)

    orchestrator = DynamicOrchestrator.from_config(
        device=device,
        channel_names=all_ch_names,
        n_channels=len(all_ch_names),
        knowledge_base_dir=str(kb_dir),
        channel_map_path=str(FRAMEWORK / "data/nyc_top128_channel_map.json"),
        max_llm_events_per_window=10,
        fusion_channel_names=fusion_channels,
        event_station_whitelist=station_ids,
        lp_data_path=str(FRAMEWORK / "data/nyc_top128_station_hourly_flow.csv"),
        lp_model_path=str(FRAMEWORK / "experiments/outputs/lp_fallback_nyc_top128"),
        freeze_skills=True,
    )

    result = orchestrator.run(
        data_path=str(FRAMEWORK / "data/nyc_top128_station_hourly_flow.csv"),
        event_source=str(events_json),
        forecast_horizon=192,
        n_channels=len(all_ch_names),
        fusion_channel_names=fusion_channels,
        event_station_whitelist=station_ids,
        target_date=target_date,
        update_skills=False,
    )

    h = min(horizon, len(result.forecast_timestamps or []))
    ts = _parse_ts(result.forecast_timestamps[:h])
    name_to_idx = {n: i for i, n in enumerate(result.channel_names or [])}

    plot_channels = focus_channels or fusion_channels
    plot_channels = [c for c in plot_channels if c in fusion_set and c in name_to_idx]

    rows = []
    raw_dict, adj_dict, gt_dict = {}, {}, {}

    for ch in fusion_channels:
        if ch not in name_to_idx:
            continue
        idx = name_to_idx[ch]
        raw = np.array(result.raw_forecast[idx][:h], dtype=float)
        adj = np.array(result.adjusted_forecast[idx][:h], dtype=float)
        gt = (
            np.array(result.ground_truth[idx][:h], dtype=float)
            if result.ground_truth
            else None
        )
        raw_dict[ch] = raw
        adj_dict[ch] = adj
        if gt is not None:
            gt_dict[ch] = gt
        changed = bool(np.any(np.abs(raw - adj) > 1e-4))
        mae_r, mape_r = mae_mape(raw.tolist(), gt.tolist()) if gt is not None else (float("nan"), float("nan"))
        mae_a, mape_a = mae_mape(adj.tolist(), gt.tolist()) if gt is not None else (float("nan"), float("nan"))
        meta = ch_meta.get(ch, {})
        rows.append(
            {
                "channel_name": ch,
                "station_complex": meta.get("station_complex", ch),
                "fusion_adjusted": changed,
                "mae_raw": round(mae_r, 2),
                "mae_adj": round(mae_a, 2),
                "mae_delta": round(mae_a - mae_r, 2),
            }
        )

    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir / "venue37_fusion_comparison_metrics.csv", index=False)

    # ── 有事件站点：全局 + 事件窗口放大 ──
    event_focus = plot_channels[:6]
    if case_meta and case_meta.get("channel_name") in name_to_idx:
        event_focus = [case_meta["channel_name"]]

    n = len(event_focus)
    fig, axes = plt.subplots(n, 2, figsize=(16, 3.2 * n), gridspec_kw={"width_ratios": [2.2, 1.2]})
    if n == 1:
        axes = np.array([axes])

    for i, ch in enumerate(event_focus):
        raw, adj = raw_dict[ch], adj_dict[ch]
        gt = gt_dict.get(ch)
        ev_times, ev_labels = _events_in_window(events_json, ch, ts[0], ts[-1])
        if case_meta and case_meta.get("channel_name") == ch:
            ev_t = pd.to_datetime(case_meta["event_time"])
            if ev_t not in ev_times:
                ev_times = [ev_t] + ev_times
                ev_labels = [case_meta.get("event_title", "")] + ev_labels

        short = ch.split("__", 1)[-1][:45]
        diff_note = " [Fusion≠MOMENT]" if np.any(np.abs(raw - adj) > 1e-4) else " [no adj]"
        plot_forecast_panel(
            axes[i, 0], ts, raw, adj, gt,
            title=f"{short}{diff_note}",
            event_times=ev_times,
            event_labels=ev_labels,
            show_legend=(i == 0),
        )

        # 右列：事件窗口 ±24h 放大
        ax_z = axes[i, 1]
        if ev_times:
            ev_t = ev_times[0]
            mask = (ts >= ev_t - pd.Timedelta(hours=24)) & (ts <= ev_t + pd.Timedelta(hours=24))
            if mask.sum() >= 2:
                plot_forecast_panel(
                    ax_z, ts[mask], raw[mask], adj[mask],
                    gt[mask] if gt is not None else None,
                    title=f"Zoom ±24h @ {ev_t.strftime('%m/%d %H:%M')}",
                    event_times=[ev_t],
                    show_legend=False,
                )
            else:
                ax_z.text(0.5, 0.5, "Event outside\nforecast window", ha="center", va="center", transform=ax_z.transAxes)
                ax_z.axis("off")
        else:
            ax_z.text(0.5, 0.5, "No event in\nthis window", ha="center", va="center", transform=ax_z.transAxes)
            ax_z.axis("off")

    title_suffix = ""
    if case_meta:
        title_suffix = (
            f" | Event: {case_meta.get('event_title', '')[:50]} "
            f"(lift {case_meta.get('lift_pct', 0):+.1f}%)"
        )
    fig.suptitle(
        f"Venue37 Forecast — start {target_date}, horizon {h}h{title_suffix}",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(outdir / "fusion_event_station_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "target_date": target_date,
        "horizon_hours": h,
        "focus_channels": event_focus,
        "channels_with_fusion_change": sum(1 for r in rows if r["fusion_adjusted"]),
        "events_considered": len(result.events_considered),
        "channels_adjusted": result.channels_adjusted,
        "case": case_meta,
    }
    (outdir / "venue37_fusion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return summary


def run_auto_cases(n_cases: int, horizon: int, device: str, outdir: Path) -> list[dict]:
    fusion_cfg = json.loads((FRAMEWORK / "data" / "venue37_fusion_channels.json").read_text())
    fusion_set = set(fusion_cfg["channel_names"])
    ch_meta = {
        ch["channel_name"]: ch
        for ch in json.loads((FRAMEWORK / "data/nyc_top128_channel_map.json").read_text())["channels"]
    }
    events_path = DATA_ROOT / "outputs/venue37_analysis/venue37_verified_events.csv"
    cases = pick_event_cases(events_path, fusion_set, ch_meta, n_cases=n_cases)

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "selected_event_cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summaries = []
    for i, case in enumerate(cases, 1):
        slug = re.sub(r"[^A-Za-z0-9]+", "_", case["channel_name"])[:40]
        case_dir = outdir / f"case_{i:02d}_{case['target_date']}_{slug}"
        print(f"\n=== Case {i}/{len(cases)}: {case['event_title']} @ {case['channel_name']} ===")
        summaries.append(
            run_comparison(
                case["target_date"],
                horizon,
                device,
                case_dir,
                focus_channels=[case["channel_name"]],
                case_meta=case,
            )
        )
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Compare MOMENT vs Fusion with event-aware plots.")
    parser.add_argument("--date", default=None, help="Forecast window start (YYYY-MM-DD)")
    parser.add_argument("--channel", default=None, help="Focus channel_name")
    parser.add_argument("--horizon", type=int, default=192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--auto", action="store_true", help="Auto-pick top event cases from verified events")
    parser.add_argument("--n-cases", type=int, default=5, help="Number of auto-picked cases")
    parser.add_argument("--outdir", default=str(FRAMEWORK / "figures_venue37" / "fusion_event_cases"))
    args = parser.parse_args()

    outdir = Path(args.outdir)

    if args.auto:
        summaries = run_auto_cases(args.n_cases, args.horizon, args.device, outdir)
        print(json.dumps(summaries, ensure_ascii=False, indent=2, default=str))
        print(f"\nAuto comparison -> {outdir}/case_*/fusion_event_station_comparison.png")
        return

    if not args.date:
        args.date = "2023-05-20"
    focus = [args.channel] if args.channel else None
    summary = run_comparison(args.date, args.horizon, args.device, outdir, focus_channels=focus)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"\nOutputs -> {outdir}/fusion_event_station_comparison.png")


if __name__ == "__main__":
    main()
