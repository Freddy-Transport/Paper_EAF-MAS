from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd


def compute_traffic_evidence(
    traffic_df: pd.DataFrame,
    channel_names: Sequence[str],
    event_time: str,
    horizon_hours: int = 4,
    train_end_idx: int | None = None,
    low_volume_threshold: float = 50.0,
) -> Dict:
    df = traffic_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    event_ts = pd.Timestamp(event_time)
    matches = np.where(df["date"].to_numpy() >= np.datetime64(event_ts))[0]
    if len(matches) == 0:
        return {"available": False, "reason": "event_time_out_of_range", "n_channels": len(channel_names)}
    start = int(matches[0])
    stop = min(start + int(horizon_hours), len(df))
    if stop <= start:
        return {"available": False, "reason": "empty_event_window", "n_channels": len(channel_names)}
    channels = [c for c in channel_names if c in df.columns]
    if not channels:
        return {"available": False, "reason": "no_matching_channels", "n_channels": 0}
    train_stop = min(start, train_end_idx if train_end_idx is not None else start)
    hist = df.iloc[:train_stop]
    actual = df.loc[start: stop - 1, channels].astype(float).to_numpy()
    actual_sum = float(np.nansum(actual))
    actual_mean = float(np.nanmean(actual)) if actual.size else 0.0

    if hist.empty:
        baseline_vals = np.full((1, len(channels)), np.nan)
    else:
        same = hist[(hist["date"].dt.hour == event_ts.hour) & (hist["date"].dt.dayofweek == event_ts.dayofweek)]
        if len(same) < 4:
            same = hist[hist["date"].dt.hour == event_ts.hour]
        if len(same) < 4:
            same = hist
        baseline_vals = same[channels].astype(float).to_numpy()
    finite_baseline = baseline_vals[np.isfinite(baseline_vals)] if baseline_vals.size else np.asarray([])
    baseline_mean = float(np.mean(finite_baseline)) if finite_baseline.size else 0.0
    baseline_std = float(np.std(finite_baseline)) if finite_baseline.size else 0.0
    denom = max(abs(baseline_mean), 1.0)
    delta_pct = (actual_mean - baseline_mean) / denom
    z_score = (actual_mean - baseline_mean) / max(baseline_std, 1.0)
    return {
        "available": True,
        "event_time": str(event_ts),
        "window_start_idx": start,
        "window_hours": int(stop - start),
        "n_channels": len(channels),
        "actual_sum": actual_sum,
        "actual_mean": actual_mean,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "delta_pct": float(delta_pct),
        "z_score": float(z_score),
        "low_volume_warning": bool(actual_sum < low_volume_threshold),
    }
