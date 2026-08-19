#!/usr/bin/env python3
"""Error-focused event adapter experiment for venue28 Top128 intersections.

This script intentionally refuses to use the deterministic lag fallback as a
stand-in for numerical_only. It loads the cached LP MOMENT model, exports real
forecasts for test anchors, mines worst venue-channel periods, then trains a
frozen residual adapter and a PEFT-head+adapter variant on train/val windows.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.event_adapter import (  # noqa: E402
    build_event_feature_cube,
    channel_meta_by_name,
    load_channel_map,
    load_events_json,
    save_adapter,
    adapter_config,
    build_adapter_from_config,
)
from agents.numerical_agent import NumericalPredictionAgent  # noqa: E402
from event_post_training.config import EventPostTrainingConfig, resolve_lp_model_path  # noqa: E402
from event_post_training.sample_builder import build_anchor_plan, event_indices_for_dates, events_for_window  # noqa: E402
from event_post_training.trainer import configure_trainable_parameters  # noqa: E402


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))


def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sum(np.abs(actual - pred)) / max(float(np.sum(np.abs(actual))), 1.0) * 100.0)


def ensure_run_dir(path: Path) -> None:
    for sub in ["predictions", "error_mining", "models", "figures", "reports", "logs"]:
        (path / sub).mkdir(parents=True, exist_ok=True)


def load_traffic(path: Path) -> Tuple[pd.DataFrame, List[str], np.ndarray]:
    df = pd.read_csv(path, parse_dates=["date"])
    channel_names = [c for c in df.columns if c != "date"]
    values = df[channel_names].infer_objects(copy=False).interpolate(method="linear").ffill().bfill().values.astype(np.float32)
    return df, channel_names, values


def load_venue_indices(path: Path, channel_names: Sequence[str]) -> Tuple[List[str], List[int], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    requested = list(payload.get("channel_names", []))
    name_to_idx = {name: i for i, name in enumerate(channel_names)}
    venue_names = [name for name in requested if name in name_to_idx]
    venue_indices = [name_to_idx[name] for name in venue_names]
    meta = {
        "requested_venue_total": int(payload.get("venue37_total", len(requested))),
        "top128_intersection_count": len(venue_indices),
        "source_intersection_count": int(payload.get("intersection_count", len(venue_indices))),
        "missing_from_top_n": payload.get("missing_from_top_n", []),
    }
    return venue_names, venue_indices, meta


def choose_test_anchors(cfg: EventPostTrainingConfig, events: Sequence, dates: Sequence[pd.Timestamp], max_anchors: int, stride: int) -> List[int]:
    event_idx = event_indices_for_dates(events, dates)
    test_start = min(cfg.train_rows + cfg.val_rows, len(dates))
    start = max(test_start + cfg.seq_len, cfg.seq_len)
    stop = len(dates) - cfg.horizon
    anchors = []
    for anchor in range(start, stop + 1, max(1, stride)):
        if any(anchor <= e < anchor + cfg.horizon for e in event_idx):
            anchors.append(anchor)
    if not anchors:
        anchors = list(range(start, stop + 1, max(1, stride)))
    return anchors[: max_anchors]


def init_lp_agent(cfg: EventPostTrainingConfig, channel_names: Sequence[str], resolved_lp: Path, device: str) -> NumericalPredictionAgent:
    if not (resolved_lp / "lp_weights.pt").is_file():
        raise FileNotFoundError(f"required LP weights not found: {resolved_lp / 'lp_weights.pt'}")
    no_gca_path = PROJECT_ROOT / "__no_gca_model_for_event_error_focus__"
    return NumericalPredictionAgent(
        model_path=str(no_gca_path),
        device=device,
        forecast_horizon=cfg.horizon,
        n_channels=len(channel_names),
        channel_names=list(channel_names),
        lp_model_path=str(resolved_lp),
        lp_data_path=str(cfg.traffic_csv),
        lp_max_epoch=1,
        lp_batch_size=8,
    )


def export_real_moment_predictions(
    agent: NumericalPredictionAgent,
    cfg: EventPostTrainingConfig,
    df: pd.DataFrame,
    channel_names: Sequence[str],
    venue_indices: Sequence[int],
    anchors: Sequence[int],
    out_csv: Path,
) -> pd.DataFrame:
    rows = []
    for i, anchor in enumerate(anchors, 1):
        target_date = str(df["date"].iloc[anchor])
        pred = agent.predict_for_date(str(cfg.traffic_csv), target_date, forecast_horizon=cfg.horizon)
        forecast = np.asarray(pred.forecast, dtype=np.float32)
        actual = np.asarray(pred.ground_truth, dtype=np.float32)
        timestamps = list(pred.forecast_timestamps or [str(x) for x in df["date"].iloc[anchor:anchor + cfg.horizon]])
        if forecast.shape[1] < cfg.horizon or actual.shape[1] < cfg.horizon:
            raise ValueError(f"prediction horizon too short for anchor {anchor}: forecast={forecast.shape}, actual={actual.shape}")
        for c_idx in venue_indices:
            for h in range(cfg.horizon):
                rows.append({
                    "anchor": int(anchor),
                    "date": target_date,
                    "channel_name": channel_names[c_idx],
                    "channel_idx": int(c_idx),
                    "horizon_idx": int(h),
                    "timestamp": timestamps[h],
                    "moment_pred": float(forecast[c_idx, h]),
                    "actual": float(actual[c_idx, h]),
                })
        print(f"[export] {i}/{len(anchors)} anchor={anchor} date={target_date}", flush=True)
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out_csv, index=False)
    return pred_df


def mine_worst_periods(pred_df: pd.DataFrame, out_dir: Path, min_actual_sum: float, top_k: int) -> List[dict]:
    rows = []
    for (anchor, channel_name, channel_idx), group in pred_df.groupby(["anchor", "channel_name", "channel_idx"]):
        group = group.sort_values("horizon_idx")
        for block_start in range(0, int(group["horizon_idx"].max()) + 1, 24):
            block = group[(group["horizon_idx"] >= block_start) & (group["horizon_idx"] < block_start + 24)]
            if block.empty:
                continue
            actual = block["actual"].to_numpy(dtype=np.float32)
            pred = block["moment_pred"].to_numpy(dtype=np.float32)
            actual_sum = float(np.sum(np.abs(actual)))
            if actual_sum < min_actual_sum:
                continue
            rows.append({
                "anchor": int(anchor),
                "date": str(block["date"].iloc[0]),
                "channel_name": str(channel_name),
                "channel_idx": int(channel_idx),
                "block_start": int(block_start),
                "block_end": int(block_start + len(block) - 1),
                "start_timestamp": str(block["timestamp"].iloc[0]),
                "end_timestamp": str(block["timestamp"].iloc[-1]),
                "actual_sum": actual_sum,
                "mae": mae(actual, pred),
                "wape": wape(actual, pred),
            })
    ranked = sorted(rows, key=lambda r: (r["wape"], r["mae"]), reverse=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ranked).to_csv(out_dir / "top_station_periods.csv", index=False)
    (out_dir / "top_station_periods.json").write_text(json.dumps(ranked[:top_k], ensure_ascii=False, indent=2), encoding="utf-8")
    make_error_heatmap(pd.DataFrame(ranked), out_dir / "venue28_error_heatmap.png")
    return ranked[:top_k]


def make_error_heatmap(ranked_df: pd.DataFrame, out_png: Path) -> None:
    if ranked_df.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top_channels = ranked_df.groupby("channel_name")["wape"].mean().sort_values(ascending=False).head(12).index
    pivot = ranked_df[ranked_df["channel_name"].isin(top_channels)].pivot_table(
        index="channel_name", columns="block_start", values="wape", aggfunc="mean"
    ).fillna(0.0)
    plt.figure(figsize=(10, max(4, 0.35 * len(pivot))))
    plt.imshow(pivot.values, aspect="auto", cmap="magma")
    plt.colorbar(label="WAPE")
    plt.yticks(range(len(pivot.index)), [x[:45] for x in pivot.index], fontsize=7)
    plt.xticks(range(len(pivot.columns)), pivot.columns.astype(str), fontsize=8)
    plt.xlabel("192h horizon block start")
    plt.title("Venue28 numerical_only error heatmap")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def make_training_samples(cfg: EventPostTrainingConfig, events: Sequence, dates: Sequence[pd.Timestamp], max_windows: int, negative_ratio: float) -> Dict[str, List[dict]]:
    event_idx = event_indices_for_dates(events, dates)
    plan = build_anchor_plan(
        len(dates), cfg.seq_len, cfg.horizon, cfg.train_rows, cfg.val_rows, event_idx,
        stride=24, negative_ratio=negative_ratio, max_windows=max_windows,
    )
    by_split = {"train": [], "val": [], "test": []}
    for item in plan:
        by_split[item["split"]].append(item)
    return by_split


@dataclass
class TrainResult:
    adapter: torch.nn.Module
    model: torch.nn.Module
    manifest: dict
    history: List[dict]


def _torch_window(values_norm: np.ndarray, anchor: int, cfg: EventPostTrainingConfig, device: str) -> torch.Tensor:
    hist = values_norm[anchor - cfg.seq_len: anchor].T[np.newaxis, :, :]
    return torch.tensor(hist, dtype=torch.float32, device=device)


def _actual_window(values_raw: np.ndarray, anchor: int, cfg: EventPostTrainingConfig, device: str) -> torch.Tensor:
    actual = values_raw[anchor: anchor + cfg.horizon].T[np.newaxis, :, :]
    return torch.tensor(actual, dtype=torch.float32, device=device)


def _inverse_torch(pred_norm: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return pred_norm * scale.view(1, -1, 1) + mean.view(1, -1, 1)


def train_moment_event_adapter(
    base_model: torch.nn.Module,
    mode: str,
    cfg: EventPostTrainingConfig,
    df: pd.DataFrame,
    values_raw: np.ndarray,
    channel_names: Sequence[str],
    venue_indices: Sequence[int],
    events: Sequence,
    channel_meta: Dict[str, dict],
    out_dir: Path,
    epochs: int,
    max_windows: int,
    negative_ratio: float,
    lr_adapter: float,
    lr_head: float,
    residual_bound: float,
    device: str,
) -> TrainResult:
    scaler = StandardScaler()
    scaler.fit(values_raw[: cfg.train_rows])
    values_norm = scaler.transform(values_raw).astype(np.float32)
    mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)
    input_mask = torch.ones(1, cfg.seq_len, dtype=torch.float32, device=device)
    samples = make_training_samples(cfg, events, list(df["date"]), max_windows=max_windows, negative_ratio=negative_ratio)
    adapter = build_adapter_from_config(adapter_config(mode=mode, hidden_dim=64, max_correction=residual_bound)).to(device)

    model = base_model
    model.to(device)
    model.train(mode == "peft_moment")
    if mode == "peft_moment":
        trainable = configure_trainable_parameters(model, adapter, mode="peft_moment")
        params = [
            {"params": adapter.parameters(), "lr": lr_adapter},
            {"params": [p for p in model.head.parameters() if p.requires_grad], "lr": lr_head},
        ]
    elif mode == "frozen_moment":
        for p in model.parameters():
            p.requires_grad = False
        for p in adapter.parameters():
            p.requires_grad = True
        trainable = [f"adapter.{name}" for name, _ in adapter.named_parameters()]
        params = [{"params": adapter.parameters(), "lr": lr_adapter}]
    else:
        raise ValueError(f"unsupported training mode: {mode}")
    optimizer = torch.optim.AdamW(params)
    venue_mask_np = np.zeros((len(channel_names), cfg.horizon), dtype=np.float32)
    venue_mask_np[list(venue_indices), :] = 1.0
    venue_mask = torch.tensor(venue_mask_np[np.newaxis, :, :], dtype=torch.float32, device=device)
    history = []
    best_state = None
    best_val = float("inf")

    for epoch in range(epochs):
        train_loss = _run_train_split(model, adapter, optimizer, samples["train"], cfg, df, values_norm, values_raw, channel_names, events, channel_meta, mean, scale, input_mask, venue_mask, device, mode)
        val_loss, val_metrics = _eval_split(model, adapter, samples["val"], cfg, df, values_norm, values_raw, channel_names, events, channel_meta, mean, scale, input_mask, venue_mask, device)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_metrics}
        history.append(row)
        print(f"[{mode}] epoch={epoch} train_loss={train_loss:.5f} val_wape={val_metrics.get('venue_wape', math.nan):.3f}", flush=True)
        if val_loss <= best_val:
            best_val = val_loss
            best_state = {
                "adapter": {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()},
                "head": {k: v.detach().cpu().clone() for k, v in getattr(model, "head").state_dict().items()} if hasattr(model, "head") else None,
            }
    if best_state:
        adapter.load_state_dict(best_state["adapter"])
        if mode == "peft_moment" and best_state["head"] is not None:
            model.head.load_state_dict(best_state["head"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dict = adapter_config(mode=mode, hidden_dim=64, max_correction=residual_bound)
    manifest = {
        "mode": mode,
        "scope": "venue28_top128_intersection",
        "epochs": epochs,
        "sample_counts": {k: len(v) for k, v in samples.items()},
        "trainable_parameters": trainable,
        "history": history,
        "best_val_loss": best_val,
        "residual_bound": residual_bound,
        "true_moment_head_training": bool(mode == "peft_moment"),
    }
    save_adapter(adapter, out_dir, cfg_dict, manifest)
    if mode == "peft_moment" and hasattr(model, "head"):
        torch.save(model.head.state_dict(), out_dir / "peft_head.pt")
    (out_dir / "training_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return TrainResult(adapter=adapter, model=model, manifest=manifest, history=history)


def _forward_raw(model, x, input_mask, mean, scale, train_head: bool) -> torch.Tensor:
    if train_head:
        out = model(x_enc=x, input_mask=input_mask)
    else:
        with torch.no_grad():
            out = model(x_enc=x, input_mask=input_mask)
    return _inverse_torch(out.forecast.float(), mean, scale)


def _adapter_adjust(adapter, raw: torch.Tensor, features_np: np.ndarray, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    feats = torch.tensor(features_np.reshape(-1, features_np.shape[-1]), dtype=torch.float32, device=device)
    corr = adapter(feats).reshape(raw.shape[1], raw.shape[2]).unsqueeze(0)
    adjusted = raw * (1.0 + corr)
    return adjusted, corr


def _run_train_split(model, adapter, optimizer, split_samples, cfg, df, values_norm, values_raw, channel_names, events, channel_meta, mean, scale, input_mask, venue_mask, device, mode) -> float:
    adapter.train(True)
    model.train(mode == "peft_moment")
    total = 0.0
    for item in split_samples:
        anchor = int(item["anchor"])
        x = _torch_window(values_norm, anchor, cfg, device)
        actual = _actual_window(values_raw, anchor, cfg, device)
        raw = _forward_raw(model, x, input_mask, mean, scale, train_head=(mode == "peft_moment"))
        timestamps = [str(ts) for ts in df["date"].iloc[anchor: anchor + cfg.horizon]]
        window_events = events_for_window(events, df["date"].iloc[anchor], df["date"].iloc[anchor + cfg.horizon - 1])
        features_np = build_event_feature_cube(raw.detach().cpu().numpy()[0], timestamps, channel_names, window_events, channel_meta)
        adjusted, corr = _adapter_adjust(adapter, raw, features_np, device)
        event_mask = torch.tensor((features_np[..., 0] > 0).astype(np.float32)[np.newaxis, :, :], dtype=torch.float32, device=device)
        weights = venue_mask * (0.25 + 2.75 * event_mask)
        per_cell = torch.nn.functional.smooth_l1_loss(adjusted, actual, reduction="none")
        loss = (per_cell * weights).sum() / torch.clamp(weights.sum(), min=1.0) + 1e-3 * (corr ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu())
    return total / max(len(split_samples), 1)


def _eval_split(model, adapter, split_samples, cfg, df, values_norm, values_raw, channel_names, events, channel_meta, mean, scale, input_mask, venue_mask, device) -> Tuple[float, dict]:
    adapter.eval()
    model.eval()
    losses = []
    actual_all, pred_all = [], []
    with torch.no_grad():
        for item in split_samples:
            anchor = int(item["anchor"])
            x = _torch_window(values_norm, anchor, cfg, device)
            actual = _actual_window(values_raw, anchor, cfg, device)
            raw = _forward_raw(model, x, input_mask, mean, scale, train_head=False)
            timestamps = [str(ts) for ts in df["date"].iloc[anchor: anchor + cfg.horizon]]
            window_events = events_for_window(events, df["date"].iloc[anchor], df["date"].iloc[anchor + cfg.horizon - 1])
            features_np = build_event_feature_cube(raw.detach().cpu().numpy()[0], timestamps, channel_names, window_events, channel_meta)
            adjusted, _ = _adapter_adjust(adapter, raw, features_np, device)
            weights = venue_mask
            loss = (torch.nn.functional.smooth_l1_loss(adjusted, actual, reduction="none") * weights).sum() / torch.clamp(weights.sum(), min=1.0)
            losses.append(float(loss.detach().cpu()))
            actual_all.append(actual.detach().cpu().numpy()[0][venue_mask.cpu().numpy()[0] > 0].reshape(len(np.where(venue_mask.cpu().numpy()[0][:, 0] > 0)[0]), cfg.horizon))
            pred_all.append(adjusted.detach().cpu().numpy()[0][venue_mask.cpu().numpy()[0] > 0].reshape(len(np.where(venue_mask.cpu().numpy()[0][:, 0] > 0)[0]), cfg.horizon))
    if actual_all:
        a = np.concatenate(actual_all, axis=1)
        p = np.concatenate(pred_all, axis=1)
        metrics = {"venue_mae": mae(a, p), "venue_wape": wape(a, p)}
    else:
        metrics = {"venue_mae": None, "venue_wape": None}
    return float(np.mean(losses)) if losses else float("inf"), metrics


def evaluate_stress_periods(
    frozen: TrainResult,
    peft: TrainResult,
    cfg: EventPostTrainingConfig,
    df: pd.DataFrame,
    values_raw: np.ndarray,
    channel_names: Sequence[str],
    venue_indices: Sequence[int],
    events: Sequence,
    channel_meta: Dict[str, dict],
    top_periods: Sequence[dict],
    pred_df: pd.DataFrame,
    out_dir: Path,
    device: str,
) -> dict:
    scaler = StandardScaler().fit(values_raw[: cfg.train_rows])
    values_norm = scaler.transform(values_raw).astype(np.float32)
    mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)
    input_mask = torch.ones(1, cfg.seq_len, dtype=torch.float32, device=device)
    rows = []
    curve_payload = []
    for period in top_periods[:2]:
        anchor = int(period["anchor"])
        c_idx = int(period["channel_idx"])
        block_start = int(period["block_start"])
        h_slice = slice(block_start, block_start + 24)
        timestamps = [str(ts) for ts in df["date"].iloc[anchor: anchor + cfg.horizon]]
        actual_full = values_raw[anchor: anchor + cfg.horizon].T.astype(np.float32)
        raw_full = pred_df[(pred_df["anchor"] == anchor)].pivot(index="channel_idx", columns="horizon_idx", values="moment_pred")
        raw = np.zeros_like(actual_full)
        for idx in venue_indices:
            if idx in raw_full.index:
                raw[idx] = raw_full.loc[idx].sort_index().to_numpy(dtype=np.float32)
        raw_block = raw[c_idx, h_slice]
        actual_block = actual_full[c_idx, h_slice]
        window_events = events_for_window(events, df["date"].iloc[anchor], df["date"].iloc[anchor + cfg.horizon - 1])
        features_np = build_event_feature_cube(raw, timestamps, channel_names, window_events, channel_meta)
        frozen_adj, _ = _numpy_adapter_predict(frozen.adapter, raw, features_np, device)
        peft_raw = _predict_model_raw(peft.model, cfg, values_norm, anchor, mean, scale, input_mask, device)[0]
        peft_features = build_event_feature_cube(peft_raw, timestamps, channel_names, window_events, channel_meta)
        peft_adj, _ = _numpy_adapter_predict(peft.adapter, peft_raw, peft_features, device)
        preds = {
            "numerical_only": raw_block,
            "event_adapter_frozen_moment": frozen_adj[c_idx, h_slice],
            "event_adapter_peft_moment": peft_adj[c_idx, h_slice],
        }
        for mode, pred in preds.items():
            rows.append({
                "anchor": anchor,
                "date": period["date"],
                "channel_name": period["channel_name"],
                "channel_idx": c_idx,
                "block_start": block_start,
                "mode": mode,
                "mae": mae(actual_block, pred),
                "wape": wape(actual_block, pred),
            })
        curve_payload.append({
            "period": period,
            "timestamps": timestamps,
            "actual": actual_full[c_idx].tolist(),
            "numerical_only": raw[c_idx].tolist(),
            "event_adapter_frozen_moment": frozen_adj[c_idx].tolist(),
            "event_adapter_peft_moment": peft_adj[c_idx].tolist(),
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "stress_metrics.csv", index=False)
    (out_dir / "stress_curves.json").write_text(json.dumps(curve_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    make_stress_figures(curve_payload, out_dir)
    summary = {mode: {"mae": float(np.mean([r["mae"] for r in rows if r["mode"] == mode])), "wape": float(np.mean([r["wape"] for r in rows if r["mode"] == mode]))} for mode in sorted({r["mode"] for r in rows})}
    return {"rows": rows, "summary": summary, "curves": curve_payload}


def _predict_model_raw(model, cfg, values_norm, anchor, mean, scale, input_mask, device) -> np.ndarray:
    model.eval()
    x = _torch_window(values_norm, anchor, cfg, device)
    with torch.no_grad():
        raw = _forward_raw(model, x, input_mask, mean, scale, train_head=False)
    return raw.detach().cpu().numpy()


def _numpy_adapter_predict(adapter, raw_np: np.ndarray, features_np: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray]:
    adapter.eval()
    with torch.no_grad():
        raw = torch.tensor(raw_np[np.newaxis, :, :], dtype=torch.float32, device=device)
        adjusted, corr = _adapter_adjust(adapter, raw, features_np, device)
    return adjusted.detach().cpu().numpy()[0].astype(np.float32), corr.detach().cpu().numpy()[0].astype(np.float32)


def make_stress_figures(curves: Sequence[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for i, curve in enumerate(curves, 1):
        period = curve["period"]
        plt.figure(figsize=(12, 4))
        xs = np.arange(len(curve["actual"]))
        plt.plot(xs, curve["actual"], label="actual", linewidth=2)
        plt.plot(xs, curve["numerical_only"], label="numerical_only", alpha=0.85)
        plt.plot(xs, curve["event_adapter_frozen_moment"], label="frozen_adapter", alpha=0.85)
        plt.plot(xs, curve["event_adapter_peft_moment"], label="peft_head_adapter", alpha=0.85)
        plt.axvspan(period["block_start"], period["block_end"], color="orange", alpha=0.15, label="worst 24h block")
        plt.title(f"{period['channel_name'][:70]} | anchor {period['anchor']} | {period['date']}")
        plt.xlabel("horizon hour")
        plt.ylabel("ridership")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"stress_curve_{i}_anchor{period['anchor']}_ch{period['channel_idx']}.png", dpi=150)
        plt.close()


def write_summary(run_dir: Path, venue_meta: dict, anchors: Sequence[int], top_periods: Sequence[dict], frozen_manifest: dict, peft_manifest: dict, stress: dict, elapsed_s: float) -> None:
    numerical = stress["summary"].get("numerical_only", {})
    frozen = stress["summary"].get("event_adapter_frozen_moment", {})
    peft = stress["summary"].get("event_adapter_peft_moment", {})
    passed = False
    if numerical:
        base = numerical.get("wape", float("inf"))
        passed = (frozen.get("wape", float("inf")) < base) or (peft.get("wape", float("inf")) < base)
    text = f"""# Event Adapter Error-Focus Experiment Summary

Run directory: `{run_dir}`

## Scope

- Station set: venue37/Top128 intersection, actual n={venue_meta['top128_intersection_count']}.
- Original venue total recorded in source file: {venue_meta['requested_venue_total']}.
- Test anchors exported with real LP-MOMENT: {list(map(int, anchors))}.
- Deterministic lag fallback: not used.

## Worst Numerical-Only Periods

Top 2 selected periods:

```json
{json.dumps(list(top_periods[:2]), ensure_ascii=False, indent=2)}
```

## Stress Metrics

```json
{json.dumps(stress['summary'], ensure_ascii=False, indent=2)}
```

## Training Manifests

Frozen adapter:
```json
{json.dumps(frozen_manifest, ensure_ascii=False, indent=2)[:4000]}
```

PEFT-head adapter:
```json
{json.dumps(peft_manifest, ensure_ascii=False, indent=2)[:4000]}
```

## Acceptance

Passed: **{passed}**

Criterion: frozen or PEFT-head adapter must improve selected stress-window WAPE over numerical_only without relying on test-set tuning.

Elapsed seconds: {elapsed_s:.1f}
"""
    (run_dir / "reports" / "experiment_summary.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--traffic_csv", default="data/nyc_top128_station_hourly_flow.csv")
    p.add_argument("--events_json", default="data/nyc_top128_station_events.json")
    p.add_argument("--channel_map", default="data/nyc_top128_channel_map.json")
    p.add_argument("--fusion_channels", default="data/venue37_fusion_channels.json")
    p.add_argument("--lp_model_path", default="experiments/outputs/lp_top128")
    p.add_argument("--device", default="auto")
    p.add_argument("--max_test_anchors", type=int, default=4)
    p.add_argument("--max_windows", type=int, default=20)
    p.add_argument("--epochs_frozen", type=int, default=2)
    p.add_argument("--epochs_peft", type=int, default=1)
    p.add_argument("--negative_ratio", type=float, default=2.0)
    p.add_argument("--residual_bound", type=float, default=0.05)
    p.add_argument("--min_actual_sum", type=float, default=50.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    device = "cuda:0" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    run_dir = Path(args.run_dir).resolve()
    ensure_run_dir(run_dir)
    cfg = EventPostTrainingConfig(
        traffic_csv=PROJECT_ROOT / args.traffic_csv,
        events_json=PROJECT_ROOT / args.events_json,
        channel_map_path=PROJECT_ROOT / args.channel_map,
        fusion_channels_path=PROJECT_ROOT / args.fusion_channels,
        lp_model_path=PROJECT_ROOT / args.lp_model_path,
        output_dir=run_dir / "models",
        residual_bound=args.residual_bound,
        device=device,
    )
    df, channel_names, values = load_traffic(cfg.traffic_csv)
    venue_names, venue_indices, venue_meta = load_venue_indices(cfg.fusion_channels_path, channel_names)
    events = load_events_json(cfg.events_json) if cfg.events_json.is_file() else []
    channel_meta = channel_meta_by_name(load_channel_map(cfg.channel_map_path))
    resolved_lp, lp_source = resolve_lp_model_path(cfg.lp_model_path, PROJECT_ROOT)
    run_manifest = {
        "run_dir": str(run_dir),
        "device": device,
        "requested_lp_model_path": str(cfg.lp_model_path),
        "resolved_lp_model_path": str(resolved_lp),
        "lp_source": lp_source,
        "venue_meta": venue_meta,
        "deterministic_lag_fallback_used": False,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        agent = init_lp_agent(cfg, channel_names, resolved_lp, device)
    except Exception as exc:
        (run_dir / "logs" / "real_lp_load_error.txt").write_text(repr(exc), encoding="utf-8")
        raise

    anchors = choose_test_anchors(cfg, events, list(df["date"]), args.max_test_anchors, stride=24)
    pred_csv = run_dir / "predictions" / "moment_predictions_test_venue28.csv"
    pred_df = export_real_moment_predictions(agent, cfg, df, channel_names, venue_indices, anchors, pred_csv)
    top_periods = mine_worst_periods(pred_df, run_dir / "error_mining", args.min_actual_sum, top_k=10)
    if not top_periods:
        raise RuntimeError("no top error periods found after filtering")

    # Train separate models so PEFT does not mutate the frozen baseline model.
    frozen_model = agent.model
    frozen = train_moment_event_adapter(
        frozen_model, "frozen_moment", cfg, df, values, channel_names, venue_indices, events, channel_meta,
        run_dir / "models" / "frozen_moment", args.epochs_frozen, args.max_windows, args.negative_ratio,
        lr_adapter=1e-3, lr_head=0.0, residual_bound=args.residual_bound, device=device,
    )
    peft_agent = init_lp_agent(cfg, channel_names, resolved_lp, device)
    peft = train_moment_event_adapter(
        peft_agent.model, "peft_moment", cfg, df, values, channel_names, venue_indices, events, channel_meta,
        run_dir / "models" / "peft_moment", args.epochs_peft, args.max_windows, args.negative_ratio,
        lr_adapter=3e-4, lr_head=1e-5, residual_bound=args.residual_bound, device=device,
    )
    stress = evaluate_stress_periods(
        frozen, peft, cfg, df, values, channel_names, venue_indices, events, channel_meta,
        top_periods[:2], pred_df, run_dir / "figures", device,
    )
    (run_dir / "reports" / "stress_metrics_summary.json").write_text(json.dumps(stress["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(run_dir, venue_meta, anchors, top_periods, frozen.manifest, peft.manifest, stress, time.time() - t0)
    print(json.dumps({
        "run_dir": str(run_dir),
        "predictions_csv": str(pred_csv),
        "top_periods": str(run_dir / "error_mining" / "top_station_periods.json"),
        "stress_summary": stress["summary"],
        "summary_report": str(run_dir / "reports" / "experiment_summary.md"),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
