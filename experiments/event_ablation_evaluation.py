#!/usr/bin/env python3
"""Evaluate numerical_only vs event-aware adapter modes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.event_adapter import apply_adapter_to_forecast, build_event_feature_cube, channel_meta_by_name, load_channel_map, load_events_json
from event_post_training.config import EventPostTrainingConfig
from event_post_training.sample_builder import baseline_prediction, event_indices_for_dates, load_prediction_csv, prediction_for_anchor


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def wape(y, p):
    return float(np.sum(np.abs(y - p)) / max(np.sum(np.abs(y)), 1.0) * 100.0)


def masked_metrics(actual, pred, mask):
    if mask is None or not np.asarray(mask).any():
        return {"mae": mae(actual, pred), "wape": wape(actual, pred), "n": int(actual.size)}
    return {"mae": mae(actual[mask], pred[mask]), "wape": wape(actual[mask], pred[mask]), "n": int(np.asarray(mask).sum())}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traffic_csv", default="data/nyc_top128_station_hourly_flow.csv")
    p.add_argument("--events_json", default="data/nyc_top128_station_events.json")
    p.add_argument("--channel_map", default="data/nyc_top128_channel_map.json")
    p.add_argument("--station_panel", default="hub_events")
    p.add_argument("--modes", default="numerical_only,event_adapter_frozen_moment,event_adapter_peft_moment")
    p.add_argument("--event_adapter_path", default="experiments/outputs/event_adapter_top128")
    p.add_argument("--event_adapter_peft_path", default="experiments/outputs/event_adapter_peft_top128")
    p.add_argument("--select_worst_event_windows", type=int, default=0)
    p.add_argument("--save_target_forecasts", action="store_true")
    p.add_argument("--plot_dir", default="autodl_outputs/figures_event_adapter")
    p.add_argument("--output", default="autodl_outputs/full_ablation_event_adapter.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--moment_predictions_csv", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    try:
        import torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"
    cfg = EventPostTrainingConfig(traffic_csv=PROJECT_ROOT / args.traffic_csv, events_json=PROJECT_ROOT / args.events_json, channel_map_path=PROJECT_ROOT / args.channel_map, device=device)
    df = pd.read_csv(cfg.traffic_csv, parse_dates=["date"])
    channel_names = [c for c in df.columns if c != "date"]
    values = df[channel_names].infer_objects(copy=False).interpolate(method="linear").ffill().bfill().values.astype(np.float32)
    events = load_events_json(cfg.events_json) if cfg.events_json.is_file() else []
    prediction_df = load_prediction_csv(PROJECT_ROOT / args.moment_predictions_csv) if args.moment_predictions_csv else None
    event_idx = event_indices_for_dates(events, list(df["date"]))
    anchors = []
    test_start = min(cfg.train_rows + cfg.val_rows, len(df))
    start = max(test_start + cfg.seq_len, cfg.seq_len)
    stop = len(df) - cfg.horizon
    for anchor in range(start, stop + 1, 24):
        if any(anchor <= e < anchor + cfg.horizon for e in event_idx):
            anchors.append(anchor)
    if not anchors:
        anchors = list(range(start, min(stop + 1, start + 24 * 5), 24))
    meta = channel_meta_by_name(load_channel_map(cfg.channel_map_path))
    if args.select_worst_event_windows:
        scored = []
        for anchor in anchors[:40]:
            actual = values[anchor:anchor+cfg.horizon].T
            pred = prediction_for_anchor(prediction_df, anchor, cfg.horizon, channel_names)
            if pred is None:
                pred = baseline_prediction(values, anchor, cfg.horizon)
            timestamps = [str(ts) for ts in df["date"].iloc[anchor:anchor+cfg.horizon]]
            features = build_event_feature_cube(pred, timestamps, channel_names, events, meta)
            event_mask = features[..., 0] > 0
            scored.append((masked_metrics(actual, pred, event_mask)["wape"], anchor))
        anchors = [a for _, a in sorted(scored, reverse=True)[: args.select_worst_event_windows]]

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    rows = []
    curves = []
    plot_dir = PROJECT_ROOT / args.plot_dir
    plot_dir.mkdir(parents=True, exist_ok=True)
    for anchor in anchors:
        actual = values[anchor:anchor+cfg.horizon].T.astype(np.float32)
        raw = prediction_for_anchor(prediction_df, anchor, cfg.horizon, channel_names)
        if raw is None:
            raw = baseline_prediction(values, anchor, cfg.horizon)
        timestamps = [str(ts) for ts in df["date"].iloc[anchor:anchor+cfg.horizon]]
        event_features = build_event_feature_cube(raw, timestamps, channel_names, events, meta)
        event_mask = event_features[..., 0] > 0
        non_event_mask = ~event_mask
        for mode in modes:
            pred = raw
            adapter_dir = None
            if mode == "event_adapter_frozen_moment":
                adapter_dir = PROJECT_ROOT / args.event_adapter_path
            elif mode == "event_adapter_peft_moment":
                adapter_dir = PROJECT_ROOT / args.event_adapter_peft_path
            if adapter_dir is not None and (adapter_dir / "event_adapter.pt").is_file():
                pred, _ = apply_adapter_to_forecast(raw, timestamps, channel_names, events, meta, adapter_dir, device=device)
            event_metrics = masked_metrics(actual, pred, event_mask)
            non_event_metrics = masked_metrics(actual, pred, non_event_mask)
            rows.append({
                "anchor": int(anchor), "date": timestamps[0], "mode": mode,
                "mae": mae(actual, pred), "wape": wape(actual, pred),
                "event_mae": event_metrics["mae"], "event_wape": event_metrics["wape"], "event_n": event_metrics["n"],
                "non_event_mae": non_event_metrics["mae"], "non_event_wape": non_event_metrics["wape"], "non_event_n": non_event_metrics["n"],
            })
            if args.save_target_forecasts:
                # Save event-active channels first; fall back to highest-volume channels.
                totals = actual.sum(axis=1)
                active_channels = np.where(event_mask.any(axis=1))[0]
                ranked = active_channels[np.argsort(-totals[active_channels])] if len(active_channels) else np.argsort(-totals)
                for idx in ranked[:2]:
                    curves.append({
                        "anchor": int(anchor), "date": timestamps[0], "mode": mode, "channel": channel_names[int(idx)],
                        "timestamps": timestamps,
                        "actual": actual[int(idx)].tolist(),
                        "prediction": pred[int(idx)].tolist(),
                    })
    summary = {
        m: {
            "mae": float(np.mean([r["mae"] for r in rows if r["mode"] == m])),
            "wape": float(np.mean([r["wape"] for r in rows if r["mode"] == m])),
            "event_mae": float(np.mean([r["event_mae"] for r in rows if r["mode"] == m])),
            "event_wape": float(np.mean([r["event_wape"] for r in rows if r["mode"] == m])),
            "non_event_mae": float(np.mean([r["non_event_mae"] for r in rows if r["mode"] == m])),
            "non_event_wape": float(np.mean([r["non_event_wape"] for r in rows if r["mode"] == m])),
        } for m in modes
    }
    payload = {"modes": modes, "anchors": anchors, "rows": rows, "summary": summary, "curves": curves}
    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if curves:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for c in curves[:12]:
            plt.figure(figsize=(10, 3))
            plt.plot(c["actual"], label="actual")
            plt.plot(c["prediction"], label=c["mode"])
            plt.title(f"{c['channel'][:50]} {c['date']} {c['mode']}")
            plt.legend()
            plt.tight_layout()
            safe = f"{c['date']}_{c['mode']}_{abs(hash(c['channel'])) % 10000}.png"
            plt.savefig(plot_dir / safe, dpi=140)
            plt.close()
    print(json.dumps({"summary": summary, "output": str(out), "plot_dir": str(plot_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
