from __future__ import annotations

import numpy as np


def summarize_history(history: np.ndarray) -> np.ndarray:
    arr = np.asarray(history, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("history must be (channels, seq_len)")
    mean = arr.mean(axis=1)
    std = arr.std(axis=1)
    last = arr[:, -1]
    prev = arr[:, -24] if arr.shape[1] >= 24 else arr[:, 0]
    trend = last - prev
    denom = np.maximum(mean, 1.0)
    feats = np.stack([mean / denom, std / denom, last / denom, trend / denom], axis=1)
    if feats.shape[1] < 16:
        feats = np.pad(feats, ((0, 0), (0, 16 - feats.shape[1])))
    return feats[:, :16].astype(np.float32)
