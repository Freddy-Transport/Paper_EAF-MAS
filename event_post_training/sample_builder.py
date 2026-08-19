"""Sample construction for event-aware residual training."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from agents.event_adapter import build_event_feature_cube, channel_meta_by_name, load_channel_map, load_events_json
from event_post_training.config import RESIDUAL_EPS


def compute_residual_target(actual: np.ndarray, moment_pred: np.ndarray, residual_bound: float = 0.10, eps: float = RESIDUAL_EPS) -> np.ndarray:
    denom = np.maximum(moment_pred.astype(np.float32), float(eps))
    residual = (actual.astype(np.float32) - moment_pred.astype(np.float32)) / denom
    return np.clip(residual, -float(residual_bound), float(residual_bound)).astype(np.float32)


def load_prediction_csv(path: Optional[str | Path]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _first_existing(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {c.lower(): c for c in columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _prediction_matrix_cache(
    prediction_df: pd.DataFrame,
    horizon: int,
    channel_names: Sequence[str],
) -> Optional[Dict[int, np.ndarray]]:
    cols = list(prediction_df.columns)
    anchor_col = _first_existing(cols, ["anchor", "forecast_anchor", "forecast_start_idx"])
    step_col = _first_existing(cols, ["horizon_idx", "step", "hour_offset", "t"])
    channel_col = _first_existing(cols, ["channel_name", "channel", "station", "station_channel"])
    channel_idx_col = _first_existing(cols, ["channel_idx", "channel_index"])
    pred_col = _first_existing(cols, ["moment_pred", "prediction", "pred", "forecast", "raw_moment"])
    if not anchor_col or not step_col or not pred_col or (not channel_col and not channel_idx_col):
        return None

    root = prediction_df.attrs.setdefault("_moment_prediction_matrix_cache", {})
    cache_key = (
        int(horizon),
        tuple(str(name) for name in channel_names),
        str(anchor_col),
        str(step_col),
        str(channel_col or ""),
        str(channel_idx_col or ""),
        str(pred_col),
    )
    if cache_key in root:
        return root[cache_key]

    name_to_idx = {str(name): i for i, name in enumerate(channel_names)}
    matrices: Dict[int, np.ndarray] = {}
    numeric_steps = pd.to_numeric(prediction_df[step_col], errors="coerce")
    numeric_preds = pd.to_numeric(prediction_df[pred_col], errors="coerce")
    if channel_idx_col:
        numeric_channels = pd.to_numeric(prediction_df[channel_idx_col], errors="coerce")
    else:
        numeric_channels = prediction_df[channel_col].astype(str).map(name_to_idx)

    indexed = prediction_df[[anchor_col]].copy()
    indexed["_step"] = numeric_steps
    indexed["_channel_idx"] = numeric_channels
    indexed["_pred"] = numeric_preds

    for anchor_value, rows in indexed.groupby(anchor_col, sort=False):
        try:
            anchor_int = int(anchor_value)
        except Exception:
            continue
        out = np.full((len(channel_names), int(horizon)), np.nan, dtype=np.float32)
        steps = rows["_step"].to_numpy(dtype=np.float64, copy=False)
        channels = rows["_channel_idx"].to_numpy(dtype=np.float64, copy=False)
        preds = rows["_pred"].to_numpy(dtype=np.float64, copy=False)
        valid = (
            np.isfinite(steps)
            & np.isfinite(channels)
            & np.isfinite(preds)
            & (steps >= 0)
            & (steps < int(horizon))
            & (channels >= 0)
            & (channels < len(channel_names))
        )
        if valid.any():
            out[channels[valid].astype(np.int64), steps[valid].astype(np.int64)] = preds[valid].astype(np.float32)
        if not np.isnan(out).any():
            matrices[anchor_int] = out

    root[cache_key] = matrices
    return matrices


def prediction_for_anchor(
    prediction_df: Optional[pd.DataFrame],
    anchor: int,
    horizon: int,
    channel_names: Sequence[str],
) -> Optional[np.ndarray]:
    """Read an external MOMENT prediction CSV in a long format when available.

    Supported columns are flexible aliases around:
    anchor/forecast_anchor, horizon_idx/step, channel_name/channel_idx, and
    prediction/moment_pred. Missing or partial rows return None and let callers
    use the deterministic cache fallback.
    """
    if prediction_df is None or prediction_df.empty:
        return None
    matrices = _prediction_matrix_cache(prediction_df, horizon, channel_names)
    if not matrices:
        return None
    pred = matrices.get(int(anchor))
    if pred is None:
        return None
    return pred.astype(np.float32, copy=True)


def build_anchor_plan(
    n_rows: int,
    seq_len: int,
    horizon: int,
    train_rows: int,
    val_rows: int,
    event_start_indices: Set[int],
    stride: int = 24,
    negative_ratio: float = 1.0,
    max_windows: Optional[int] = None,
) -> List[dict]:
    splits = [
        ("train", 0, min(train_rows, n_rows)),
        ("val", min(train_rows, n_rows), min(train_rows + val_rows, n_rows)),
        ("test", min(train_rows + val_rows, n_rows), n_rows),
    ]
    samples: List[dict] = []
    for split, split_start, split_end in splits:
        candidates = []
        start = max(split_start + seq_len, seq_len)
        stop = split_end - horizon
        if stop < start:
            continue
        for anchor in range(start, stop + 1, max(1, stride)):
            has_event = any(anchor <= e < anchor + horizon for e in event_start_indices)
            candidates.append({
                "split": split,
                "anchor": anchor,
                "history_start": anchor - seq_len,
                "forecast_start": anchor,
                "forecast_end": anchor + horizon,
                "split_start": split_start,
                "split_end": split_end,
                "has_event": bool(has_event),
            })
        event_rows = [c for c in candidates if c["has_event"]]
        neg_rows = [c for c in candidates if not c["has_event"]]
        n_neg = int(max(len(event_rows), 1) * max(0.0, negative_ratio)) if event_rows else min(len(neg_rows), 4)
        selected = event_rows + neg_rows[:n_neg]
        if max_windows is not None:
            selected = selected[: max_windows]
        samples.extend(selected)
    return samples


def event_indices_for_dates(events: Iterable, dates: Sequence[pd.Timestamp]) -> Set[int]:
    by_date = {pd.Timestamp(ts).strftime("%Y-%m-%d"): i for i, ts in enumerate(dates)}
    out: Set[int] = set()
    for ev in events:
        raw = getattr(ev, "event_time", None) if not isinstance(ev, dict) else ev.get("event_time")
        if not raw:
            continue
        try:
            key = pd.Timestamp(raw).strftime("%Y-%m-%d")
        except Exception:
            continue
        if key in by_date:
            out.add(by_date[key])
    return out


def events_for_window(events: Iterable, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> List:
    selected = []
    start_day = (pd.Timestamp(start_ts) - pd.Timedelta(hours=6)).date()
    end_day = (pd.Timestamp(end_ts) + pd.Timedelta(hours=24)).date()
    for ev in events:
        raw = getattr(ev, "event_time", None) if not isinstance(ev, dict) else ev.get("event_time")
        if not raw:
            continue
        try:
            day = pd.Timestamp(raw).date()
        except Exception:
            continue
        if start_day <= day <= end_day:
            selected.append(ev)
    return selected


def baseline_prediction(values: np.ndarray, anchor: int, horizon: int) -> np.ndarray:
    # Deterministic MOMENT-cache fallback: previous week if possible, otherwise previous day.
    lag = 24 * 7 if anchor >= 24 * 7 else 24
    start = max(0, anchor - lag)
    block = values[start : start + horizon]
    if len(block) < horizon:
        block = values[anchor - horizon : anchor]
    if len(block) < horizon:
        last = values[max(0, anchor - 1) : anchor]
        block = np.repeat(last, horizon, axis=0)
    return block[:horizon].T.astype(np.float32)


def save_moment_cache(samples_by_split: Dict[str, List[dict]], cache_dir: str | Path) -> None:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    manifest = []
    for split, samples in samples_by_split.items():
        for sample in samples:
            name = f"{split}_anchor_{int(sample['anchor'])}.npz"
            np.savez_compressed(
                cache / name,
                pred=sample["pred"],
                actual=sample["actual"],
                timestamps=np.asarray(sample["timestamps"]),
                channel_names=np.asarray(sample["channel_names"]),
            )
            manifest.append({"split": split, "anchor": int(sample["anchor"]), "file": name})
    (cache / "manifest.json").write_text(__import__("json").dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def build_training_arrays(
    traffic_csv: str | Path,
    events_json: str | Path,
    channel_map_path: str | Path,
    seq_len: int,
    horizon: int,
    train_rows: int,
    val_rows: int,
    residual_bound: float,
    stride: int = 24,
    negative_ratio: float = 1.0,
    max_windows: Optional[int] = None,
    moment_predictions_csv: Optional[str | Path] = None,
    splits: Sequence[str] = ("train", "val", "test"),
    strict_prediction_csv: bool = False,
    allow_deterministic_fallback: bool = True,
) -> Dict[str, List[dict]]:
    df = pd.read_csv(traffic_csv, parse_dates=["date"])
    channel_names = [c for c in df.columns if c != "date"]
    values = df[channel_names].infer_objects(copy=False).interpolate(method="linear").ffill().bfill().values.astype(np.float32)
    events = load_events_json(events_json) if Path(events_json).is_file() else []
    prediction_df = load_prediction_csv(moment_predictions_csv)
    if strict_prediction_csv and prediction_df is None:
        raise ValueError("strict_prediction_csv requires a complete moment_predictions_csv")
    event_idx = event_indices_for_dates(events, list(df["date"]))
    plan = build_anchor_plan(len(df), seq_len, horizon, train_rows, val_rows, event_idx, stride=stride, negative_ratio=negative_ratio, max_windows=max_windows)
    allowed_splits = set(splits)
    channel_meta = channel_meta_by_name(load_channel_map(channel_map_path))
    by_split: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    for item in plan:
        if item["split"] not in allowed_splits:
            continue
        anchor = item["anchor"]
        actual = values[anchor : anchor + horizon].T.astype(np.float32)
        pred = prediction_for_anchor(prediction_df, anchor, horizon, channel_names)
        prediction_source = "moment_predictions_csv" if pred is not None else "deterministic_lag_cache"
        if pred is None:
            if strict_prediction_csv:
                raise ValueError(f"Missing complete moment_predictions_csv rows for split={item['split']} anchor={anchor}")
            if not allow_deterministic_fallback:
                raise ValueError(f"Deterministic lag fallback disabled and no complete moment_predictions_csv rows for split={item['split']} anchor={anchor}")
            pred = baseline_prediction(values, anchor, horizon)
        timestamps = [str(ts) for ts in df["date"].iloc[anchor : anchor + horizon]]
        residual = compute_residual_target(actual, pred, residual_bound=residual_bound)
        window_events = events_for_window(events, df["date"].iloc[anchor], df["date"].iloc[anchor + horizon - 1])
        features = build_event_feature_cube(pred, timestamps, channel_names, window_events, channel_meta)
        by_split[item["split"]].append({**item, "pred": pred, "actual": actual, "target": residual, "features": features, "timestamps": timestamps, "channel_names": channel_names, "prediction_source": prediction_source})
    return by_split
