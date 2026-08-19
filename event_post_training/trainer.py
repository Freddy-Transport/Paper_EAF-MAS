"""Training utilities for event-aware post-training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from agents.event_adapter import EventResidualAdapter, adapter_config, build_adapter_from_config, save_adapter


def configure_trainable_parameters(moment_model: Optional[nn.Module], adapter: nn.Module, mode: str) -> List[str]:
    trainable: List[str] = []
    if moment_model is not None:
        for p in moment_model.parameters():
            p.requires_grad = False
        if mode == "peft_moment":
            if not hasattr(moment_model, "head"):
                raise ValueError("peft_moment requires moment_model.head")
            for name, p in moment_model.head.named_parameters():
                p.requires_grad = True
                trainable.append(f"moment.head.{name}")
        elif mode != "frozen_moment":
            raise ValueError(f"unsupported mode: {mode}")
    for name, p in adapter.named_parameters():
        p.requires_grad = True
        trainable.append(f"adapter.{name}")
    return trainable


class ResidualDataset(Dataset):
    def __init__(self, samples: List[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        row = self.samples[idx]
        return {
            "features": torch.tensor(row["features"], dtype=torch.float32),
            "target": torch.tensor(row["target"], dtype=torch.float32),
            "pred": torch.tensor(row["pred"], dtype=torch.float32),
        }


def _run_epoch(adapter: EventResidualAdapter, loader: DataLoader, optimizer=None, device: str = "cpu", l2: float = 0.001) -> float:
    training = optimizer is not None
    adapter.train(training)
    total = 0.0
    n = 0
    for batch in loader:
        features = batch["features"].to(device)
        target = batch["target"].to(device)
        x = features.reshape(-1, features.shape[-1])
        y = target.reshape(-1)
        pred = adapter(x)
        active = features[..., 0].reshape(-1) > 0
        weights = torch.where(active, torch.tensor(3.0, device=device), torch.tensor(1.0, device=device))
        loss = (torch.nn.functional.smooth_l1_loss(pred, y, reduction="none") * weights).mean() + float(l2) * (pred ** 2).mean()
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += float(loss.detach().cpu())
        n += 1
    return total / max(n, 1)


def _wape(actual: np.ndarray, pred: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    if mask is not None and np.asarray(mask).any():
        actual = actual[mask]
        pred = pred[mask]
    return float(np.sum(np.abs(actual - pred)) / max(float(np.sum(np.abs(actual))), 1.0) * 100.0)


def predict_adapter_correction(adapter: EventResidualAdapter, features: np.ndarray, device: str = "cpu") -> np.ndarray:
    with torch.no_grad():
        x = torch.tensor(features.reshape(-1, features.shape[-1]), dtype=torch.float32, device=device)
        correction = adapter.to(device)(x).detach().cpu().numpy().reshape(features.shape[:2])
    return np.clip(correction, -adapter.max_correction, adapter.max_correction).astype(np.float32)


def calibrate_validation_blend(
    adapter: EventResidualAdapter,
    val_samples: Sequence[dict],
    device: str = "cpu",
    candidates: Sequence[float] = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0),
) -> dict:
    """Choose correction strength using validation event cells only.

    Candidate 0.0 is included intentionally: if the adapter is not better on
    validation event windows, inference falls back to the numerical forecast.
    """
    if not val_samples:
        return {"blend_factor": 1.0, "baseline_event_wape": None, "best_event_wape": None, "candidates": []}
    scores = []
    baseline_values = []
    for sample in val_samples:
        actual = np.asarray(sample["actual"], dtype=np.float32)
        raw = np.asarray(sample["pred"], dtype=np.float32)
        features = np.asarray(sample["features"], dtype=np.float32)
        event_mask = features[..., 0] > 0
        if not event_mask.any():
            event_mask = None
        corr = predict_adapter_correction(adapter, features, device=device)
        baseline_values.append(_wape(actual, raw, event_mask))
        for blend in candidates:
            adjusted = raw * (1.0 + float(blend) * corr)
            scores.append({"blend_factor": float(blend), "event_wape": _wape(actual, adjusted, event_mask)})
    grouped = []
    for blend in candidates:
        vals = [row["event_wape"] for row in scores if row["blend_factor"] == float(blend)]
        grouped.append({"blend_factor": float(blend), "event_wape": float(np.mean(vals))})
    baseline = float(np.mean(baseline_values))
    best = min(grouped, key=lambda row: (row["event_wape"], abs(row["blend_factor"])))
    return {
        "blend_factor": float(best["blend_factor"]),
        "baseline_event_wape": baseline,
        "best_event_wape": float(best["event_wape"]),
        "improves_validation": bool(best["event_wape"] < baseline),
        "candidates": grouped,
    }


def train_adapter(
    samples_by_split: Dict[str, List[dict]],
    output_dir: str | Path,
    mode: str = "frozen_moment",
    epochs: int = 2,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    residual_bound: float = 0.10,
    device: str = "cpu",
    manifest: Optional[dict] = None,
) -> dict:
    train_samples = samples_by_split.get("train", [])
    val_samples = samples_by_split.get("val", [])
    if not train_samples:
        raise ValueError("no training samples generated")
    cfg = adapter_config(mode=mode, hidden_dim=hidden_dim, max_correction=residual_bound)
    adapter = build_adapter_from_config(cfg).to(device)
    configure_trainable_parameters(None, adapter, mode="frozen_moment")
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr)
    train_loader = DataLoader(ResidualDataset(train_samples), batch_size=1, shuffle=True)
    val_loader = DataLoader(ResidualDataset(val_samples), batch_size=1, shuffle=False) if val_samples else None
    history = []
    best = float("inf")
    best_state = None
    for epoch in range(int(epochs)):
        train_loss = _run_epoch(adapter, train_loader, opt, device=device)
        val_loss = _run_epoch(adapter, val_loader, None, device=device) if val_loader else train_loss
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss <= best:
            best = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
    if best_state is not None:
        adapter.load_state_dict(best_state)
    calibration = calibrate_validation_blend(adapter, val_samples, device=device)
    cfg["blend_factor"] = float(calibration["blend_factor"])
    adapter.blend_factor = float(calibration["blend_factor"])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    full_manifest = dict(manifest or {})
    full_manifest.update({
        "mode": mode,
        "epochs": epochs,
        "history": history,
        "best_val_loss": best,
        "validation_blend_calibration": calibration,
    })
    save_adapter(adapter, out, cfg, full_manifest)
    (out / "training_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return full_manifest
