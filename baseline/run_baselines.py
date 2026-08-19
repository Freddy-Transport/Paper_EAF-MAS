"""Baseline forecasting workspace for NYC Top128 hourly ridership.

This script is intentionally independent from the EAF-MAS event/reasoning
pipeline. It evaluates numerical forecasting baselines on the same full-test
anchor protocol used by the paper: lookback=512, horizon=192, train rows=8640,
validation rows=2880, and all complete test anchors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


MODEL_CHOICES = {
    "seasonal_week",
    "last_day",
    "dlinear",
    "nlinear",
    "patchtst",
    "itransformer",
}


@dataclass
class Protocol:
    data_path: str
    lookback: int
    horizon: int
    train_rows: int
    val_rows: int
    stride: int
    seed: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ridership(path: str | Path) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(path, parse_dates=["date"])
    channels = [c for c in df.columns if c != "date"]
    df = df.sort_values("date").reset_index(drop=True)
    # Keep the model input regular without using test information in scaling.
    df[channels] = df[channels].infer_objects(copy=False).interpolate(method="linear").ffill().bfill()
    return df, channels


def split_bounds(n_rows: int, train_rows: int, val_rows: int) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    train_end = min(train_rows, n_rows)
    val_end = min(train_rows + val_rows, n_rows)
    return (0, train_end), (train_end, val_end), (val_end, n_rows)


def anchors_for_split(start: int, end: int, lookback: int, horizon: int, stride: int) -> List[int]:
    # Validation/test anchors may use historical context from the preceding
    # split; only the forecast target block is constrained to the split.
    first = max(start, lookback)
    last = end - horizon
    if last < first:
        return []
    return list(range(first, last + 1, max(1, stride)))


class WindowDataset(Dataset):
    def __init__(self, values: np.ndarray, anchors: List[int], lookback: int, horizon: int):
        self.values = values.astype(np.float32)
        self.anchors = anchors
        self.lookback = int(lookback)
        self.horizon = int(horizon)

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, idx: int):
        anchor = self.anchors[idx]
        x = self.values[anchor - self.lookback : anchor].T
        y = self.values[anchor : anchor + self.horizon].T
        return torch.from_numpy(x), torch.from_numpy(y), int(anchor)


class StandardScaler:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)

    @classmethod
    def fit(cls, values: np.ndarray) -> "StandardScaler":
        return cls(values.mean(axis=0), values.std(axis=0))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean[None, :]) / self.std[None, :]

    def inverse_window(self, values: np.ndarray) -> np.ndarray:
        # Input shape: (N, C, H)
        return values * self.std[None, :, None] + self.mean[None, :, None]


def metrics_np(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-6)
    smape = float(np.mean(2.0 * np.abs(err) / denom) * 100.0)
    wape = float(np.sum(np.abs(err)) / max(np.sum(np.abs(y_true)), 1e-6) * 100.0)
    return {"mae": mae, "rmse": rmse, "wape": wape, "smape": smape}


def collect_actual(values_raw: np.ndarray, anchors: List[int], horizon: int) -> np.ndarray:
    return np.stack([values_raw[a : a + horizon].T for a in anchors], axis=0).astype(np.float32)


def naive_forecast(values_raw: np.ndarray, anchors: List[int], horizon: int, offset: int) -> np.ndarray:
    """Repeat the most recent daily/weekly seasonal pattern over the horizon.

    For H=192, using y[t+h-offset] directly would leak future observations when
    h >= offset. We therefore take the last observed seasonal pattern
    y[t-offset:t] and tile it until the forecast horizon is filled.
    """
    preds = []
    for anchor in anchors:
        start = anchor - offset
        if start < 0:
            block = values_raw[anchor - horizon : anchor]
            if len(block) < horizon:
                block = np.repeat(values_raw[anchor - 1 : anchor], horizon, axis=0)
        else:
            pattern = values_raw[start:anchor]
            repeat_count = int(math.ceil(horizon / max(len(pattern), 1)))
            block = np.tile(pattern, (repeat_count, 1))
        preds.append(block[:horizon].T)
    return np.stack(preds, axis=0).astype(np.float32)


class MovingAvg(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x_pad = torch.cat([front, x, end], dim=-1)
        return self.avg(x_pad)


class DLinear(nn.Module):
    def __init__(self, channels: int, lookback: int, horizon: int, kernel_size: int = 25):
        super().__init__()
        self.decomp = MovingAvg(kernel_size)
        self.trend = nn.Linear(lookback, horizon)
        self.seasonal = nn.Linear(lookback, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.decomp(x)
        seasonal = x - trend
        return self.trend(trend) + self.seasonal(seasonal)


class NLinear(nn.Module):
    def __init__(self, channels: int, lookback: int, horizon: int):
        super().__init__()
        self.linear = nn.Linear(lookback, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, :, -1:].detach()
        return self.linear(x - last) + last


class PatchTST(nn.Module):
    def __init__(self, channels: int, lookback: int, horizon: int, patch_len: int, patch_stride: int, hidden_dim: int, num_layers: int, num_heads: int, dropout: float):
        super().__init__()
        self.channels = channels
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        n_patches = 1 + max(0, (lookback - patch_len) // patch_stride)
        self.patch_embed = nn.Linear(patch_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(n_patches * hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        patches = patches.reshape(b * c, patches.shape[2], self.patch_len)
        z = self.patch_embed(patches)
        z = self.encoder(z)
        out = self.head(z.reshape(b * c, -1))
        return out.reshape(b, c, -1)


class ITransformer(nn.Module):
    def __init__(self, channels: int, lookback: int, horizon: int, hidden_dim: int, num_layers: int, num_heads: int, dropout: float):
        super().__init__()
        self.value_embed = nn.Linear(lookback, hidden_dim)
        self.channel_embed = nn.Parameter(torch.zeros(1, channels, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_dim, horizon)
        nn.init.normal_(self.channel_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.value_embed(x) + self.channel_embed[:, : x.shape[1], :]
        z = self.encoder(z)
        return self.head(z)


def build_model(name: str, channels: int, cfg: dict) -> nn.Module:
    lookback = int(cfg["lookback"])
    horizon = int(cfg["horizon"])
    if name == "dlinear":
        return DLinear(channels, lookback, horizon)
    if name == "nlinear":
        return NLinear(channels, lookback, horizon)
    if name == "patchtst":
        return PatchTST(channels, lookback, horizon, int(cfg["patch_len"]), int(cfg["patch_stride"]), int(cfg["hidden_dim"]), int(cfg["num_layers"]), int(cfg["num_heads"]), float(cfg["dropout"]))
    if name == "itransformer":
        return ITransformer(channels, lookback, horizon, int(cfg["hidden_dim"]), int(cfg["num_layers"]), int(cfg["num_heads"]), float(cfg["dropout"]))
    raise ValueError(f"Unsupported model: {name}")


def evaluate_model(model: nn.Module, loader: DataLoader, scaler: StandardScaler, device: torch.device) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds_scaled, actual_scaled, anchors = [], [], []
    with torch.no_grad():
        for x, y, a in loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            preds_scaled.append(pred)
            actual_scaled.append(y.numpy())
            anchors.extend([int(v) for v in a])
    preds_scaled_arr = np.concatenate(preds_scaled, axis=0)
    actual_scaled_arr = np.concatenate(actual_scaled, axis=0)
    pred = scaler.inverse_window(preds_scaled_arr)
    actual = scaler.inverse_window(actual_scaled_arr)
    return metrics_np(actual, pred), pred.astype(np.float32), actual.astype(np.float32), np.array(anchors, dtype=np.int64)


def train_neural(name: str, cfg: dict, train_values: np.ndarray, val_values: np.ndarray, test_values: np.ndarray, anchors: dict, scaler: StandardScaler, channels: List[str], run_dir: Path, device: torch.device, limit_train_batches: int | None = None) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model = build_model(name, len(channels), cfg).to(device)
    train_ds = WindowDataset(train_values, anchors["train"], cfg["lookback"], cfg["horizon"])
    val_ds = WindowDataset(val_values, anchors["val"], cfg["lookback"], cfg["horizon"])
    test_ds = WindowDataset(test_values, anchors["test"], cfg["lookback"], cfg["horizon"])
    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=int(cfg["batch_size"]), shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    loss_fn = nn.MSELoss()
    best_val = math.inf
    best_state = None
    bad_epochs = 0
    log_path = run_dir / "logs" / f"{name}.log"
    with open(log_path, "w", encoding="utf-8") as log:
        for epoch in range(1, int(cfg["max_epochs"]) + 1):
            model.train()
            losses = []
            for batch_idx, (x, y, _) in enumerate(train_loader):
                if limit_train_batches is not None and batch_idx >= limit_train_batches:
                    break
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.item()))
            val_metrics, _, _, _ = evaluate_model(model, val_loader, scaler, device)
            log.write(json.dumps({"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None, "val": val_metrics}) + "\n")
            log.flush()
            if val_metrics["wape"] < best_val:
                best_val = val_metrics["wape"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= int(cfg["patience"]):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(best_state or model.state_dict(), run_dir / "checkpoints" / f"{name}.pt")
    return evaluate_model(model, test_loader, scaler, device)


def write_npz(path: Path, pred: np.ndarray, actual: np.ndarray, anchors: np.ndarray, channels: List[str], timestamps: List[str]) -> None:
    np.savez_compressed(path, pred=pred, actual=actual, anchors=anchors, channels=np.array(channels), timestamps=np.array(timestamps))


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    set_seed(int(cfg["seed"]))

    df, channels = load_ridership(cfg["data_path"])
    raw_values = df[channels].to_numpy(dtype=np.float32)
    train_span, val_span, test_span = split_bounds(len(df), int(cfg["train_rows"]), int(cfg["val_rows"]))
    train_raw = raw_values[train_span[0] : train_span[1]]
    scaler = StandardScaler.fit(train_raw)
    scaled_values = scaler.transform(raw_values)

    anchors = {
        "train": anchors_for_split(train_span[0], train_span[1], int(cfg["lookback"]), int(cfg["horizon"]), int(cfg["stride"])),
        "val": anchors_for_split(val_span[0], val_span[1], int(cfg["lookback"]), int(cfg["horizon"]), int(cfg["stride"])),
        "test": anchors_for_split(test_span[0], test_span[1], int(cfg["lookback"]), int(cfg["horizon"]), int(cfg["stride"])),
    }
    if len(anchors["test"]) != 576:
        raise RuntimeError(f"Expected 576 test anchors; got {len(anchors['test'])}")

    requested = [m.strip() for m in args.models.split(",")] if args.models else list(cfg["models"])
    invalid = [m for m in requested if m not in MODEL_CHOICES]
    if invalid:
        raise ValueError(f"Unsupported models: {invalid}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root or "runs") / run_id
    for sub in ["metrics", "predictions", "reports", "logs", "checkpoints"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    actual_test = collect_actual(raw_values, anchors["test"], int(cfg["horizon"]))
    timestamp_blocks = [[str(v) for v in df["date"].iloc[a : a + int(cfg["horizon"])]] for a in anchors["test"]]
    flat_timestamps = ["|".join(block) for block in timestamp_blocks]

    summary = []
    for name in requested:
        if name == "seasonal_week":
            pred = naive_forecast(raw_values, anchors["test"], int(cfg["horizon"]), offset=168)
            metrics = metrics_np(actual_test, pred)
            anchor_arr = np.array(anchors["test"], dtype=np.int64)
        elif name == "last_day":
            pred = naive_forecast(raw_values, anchors["test"], int(cfg["horizon"]), offset=24)
            metrics = metrics_np(actual_test, pred)
            anchor_arr = np.array(anchors["test"], dtype=np.int64)
        else:
            metrics, pred, actual, anchor_arr = train_neural(name, cfg, scaled_values, scaled_values, scaled_values, anchors, scaler, channels, run_dir, device, args.limit_train_batches)
            actual_test_for_model = actual
            if not np.allclose(actual_test_for_model, actual_test[: len(actual_test_for_model)], atol=1e-4):
                actual_test = actual_test_for_model
        write_npz(run_dir / "predictions" / f"{name}.npz", pred, actual_test, anchor_arr, channels, flat_timestamps)
        row = {"model": name, **metrics}
        summary.append(row)
        print(json.dumps(row, indent=2))

    csv_path = run_dir / "metrics" / "baseline_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "mae", "rmse", "wape", "smape"])
        writer.writeheader()
        writer.writerows(summary)
    with open(run_dir / "metrics" / "baseline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    manifest = {
        "run_id": run_id,
        "data_path": cfg["data_path"],
        "data_sha256": sha256_file(cfg["data_path"]),
        "n_rows": int(len(df)),
        "n_channels": int(len(channels)),
        "date_start": str(df["date"].iloc[0]),
        "date_end": str(df["date"].iloc[-1]),
        "protocol": {k: cfg[k] for k in ["lookback", "horizon", "train_rows", "val_rows", "stride", "seed"]},
        "anchors": {k: len(v) for k, v in anchors.items()},
        "models": requested,
        "device": str(device),
        "notes": "Numerical baselines only; no event text, RAG, AutoSkill, LLM, or residual adapter correction.",
    }
    with open(run_dir / "reports" / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved run to {run_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--models", default=None, help="Comma-separated model list")
    p.add_argument("--output_root", default="runs")
    p.add_argument("--device", default=None)
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--limit_train_batches", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
