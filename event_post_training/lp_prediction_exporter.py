"""Export real LP-MOMENT forecasts for residual-adapter training.

This module deliberately uses NumericalPredictionAgent and cached LP-MOMENT
weights. It does not use deterministic lag predictions and does not train or
modify MOMENT weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from agents.numerical_agent import NumericalPredictionAgent
from event_post_training.config import EventPostTrainingConfig, resolve_lp_model_path
from event_post_training.sample_builder import build_anchor_plan, event_indices_for_dates
from agents.event_adapter import load_events_json


def build_prediction_anchor_plan(
    traffic_csv: str | Path,
    events_json: str | Path,
    seq_len: int,
    horizon: int,
    train_rows: int,
    val_rows: int,
    stride: int = 24,
    negative_ratio: float = 1.0,
    max_windows: Optional[int] = None,
    splits: Sequence[str] = ("train", "val"),
) -> Tuple[pd.DataFrame, List[dict]]:
    df = pd.read_csv(traffic_csv, parse_dates=["date"])
    events = load_events_json(events_json) if Path(events_json).is_file() else []
    event_idx = event_indices_for_dates(events, list(df["date"]))
    plan = build_anchor_plan(
        len(df),
        seq_len,
        horizon,
        train_rows,
        val_rows,
        event_idx,
        stride=stride,
        negative_ratio=negative_ratio,
        max_windows=max_windows,
    )
    allowed = set(splits)
    return df, [row for row in plan if row["split"] in allowed]


def export_lp_moment_predictions(
    agent,
    traffic_csv: str | Path,
    channel_names: Sequence[str],
    anchors: Sequence[dict],
    horizon: int,
    out_csv: str | Path,
) -> Tuple[pd.DataFrame, dict]:
    df = pd.read_csv(traffic_csv, parse_dates=["date"])
    rows = []
    for item in anchors:
        anchor = int(item["anchor"])
        split = str(item.get("split", "unknown"))
        if anchor < 0 or anchor >= len(df):
            raise ValueError(f"anchor out of range: {anchor}")
        target_date = str(df["date"].iloc[anchor])
        prediction = agent.predict_for_date(str(traffic_csv), target_date, forecast_horizon=int(horizon))
        forecast = np.asarray(prediction.forecast, dtype=np.float32)
        actual = np.asarray(prediction.ground_truth, dtype=np.float32)
        timestamps = list(prediction.forecast_timestamps or [str(x) for x in df["date"].iloc[anchor : anchor + horizon]])
        if forecast.ndim != 2 or actual.ndim != 2 or forecast.shape[1] < horizon or actual.shape[1] < horizon or len(timestamps) < horizon:
            raise ValueError(f"prediction horizon too short for anchor {anchor}: forecast={forecast.shape}, actual={actual.shape}, timestamps={len(timestamps)}")
        if forecast.shape[0] < len(channel_names) or actual.shape[0] < len(channel_names):
            raise ValueError(f"prediction channel count too short for anchor {anchor}: forecast={forecast.shape}, expected={len(channel_names)}")
        for c_idx, channel_name in enumerate(channel_names):
            for h in range(int(horizon)):
                rows.append({
                    "split": split,
                    "anchor": anchor,
                    "date": target_date,
                    "channel_name": str(channel_name),
                    "channel_idx": int(c_idx),
                    "horizon_idx": int(h),
                    "timestamp": str(timestamps[h]),
                    "moment_pred": float(forecast[c_idx, h]),
                    "actual": float(actual[c_idx, h]),
                })
    out = pd.DataFrame(rows, columns=["split", "anchor", "date", "channel_name", "channel_idx", "horizon_idx", "timestamp", "moment_pred", "actual"])
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    manifest = {
        "prediction_source": "real_lp_moment",
        "row_count": int(len(out)),
        "anchor_count": int(len(anchors)),
        "splits": sorted({str(row.get("split", "unknown")) for row in anchors}),
        "horizon": int(horizon),
        "channel_count": int(len(channel_names)),
        "output_csv": str(out_csv),
    }
    return out, manifest


def init_lp_agent_for_export(
    cfg: EventPostTrainingConfig,
    resolved_lp_model_path: str | Path,
    device: str,
) -> NumericalPredictionAgent:
    resolved = Path(resolved_lp_model_path)
    if not (resolved / "lp_weights.pt").is_file():
        raise FileNotFoundError(f"required LP-MOMENT weights not found: {resolved / 'lp_weights.pt'}")
    no_gca_path = Path(cfg.output_dir) / "__no_gca_model_for_adapter_export__"
    return NumericalPredictionAgent(
        model_path=str(no_gca_path),
        device=device,
        forecast_horizon=cfg.horizon,
        n_channels=len(cfg.channel_names),
        channel_names=list(cfg.channel_names),
        lp_model_path=str(resolved),
        lp_data_path=str(cfg.traffic_csv),
        lp_max_epoch=1,
        lp_batch_size=8,
        allow_lp_training_fallback=False,
    )


def export_predictions_for_adapter_training(
    cfg: EventPostTrainingConfig,
    resolved_lp_model_path: str | Path,
    out_csv: str | Path,
    stride: int = 24,
    negative_ratio: float = 1.0,
    max_windows: Optional[int] = None,
    device: str = "cuda:0",
    splits: Sequence[str] = ("train", "val"),
) -> dict:
    df, anchors = build_prediction_anchor_plan(
        cfg.traffic_csv,
        cfg.events_json,
        cfg.seq_len,
        cfg.horizon,
        cfg.train_rows,
        cfg.val_rows,
        stride=stride,
        negative_ratio=negative_ratio,
        max_windows=max_windows,
        splits=splits,
    )
    agent = init_lp_agent_for_export(cfg, resolved_lp_model_path, device=device)
    _, manifest = export_lp_moment_predictions(
        agent=agent,
        traffic_csv=cfg.traffic_csv,
        channel_names=cfg.channel_names,
        anchors=anchors,
        horizon=cfg.horizon,
        out_csv=out_csv,
    )
    manifest.update({
        "traffic_csv": str(cfg.traffic_csv),
        "events_json": str(cfg.events_json),
        "resolved_lp_model_path": str(resolved_lp_model_path),
        "real_lp_moment_predictions": True,
        "deterministic_fallback_used": False,
        "anchor_plan_count": int(len(anchors)),
    })
    manifest_path = Path(out_csv).with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest
