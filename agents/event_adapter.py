"""Event-aware residual adapter for Top128 station forecasts.

This module is intentionally independent from LLM/RAG code.  It converts event
records into dense per-channel/per-hour features and applies a small residual
network that corrects a raw MOMENT/LP forecast multiplicatively.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

try:
    from agents.schemas import EventInfo
except Exception:  # pragma: no cover - import fallback for script mode
    EventInfo = object

FEATURE_NAMES: List[str] = [
    "event_active",
    "expected_direction",
    "tier_score",
    "confidence_score",
    "duration_score",
    "distance_score",
    "rank_score",
    "station_match",
    "spillover_strength",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "log_raw_forecast",
    "bias",
]

TIER_SCORE = {"A": 1.0, "B": 0.65, "C": 0.25, "D": 0.0}
CONFIDENCE_SCORE = {"high": 1.0, "medium": 0.6, "low": 0.25}
EVENT_CATEGORY_DIRECTION = {
    "sports": 1.0,
    "concert": 1.0,
    "festival": 0.8,
    "parade": 0.8,
    "street_event": 0.5,
    "ceremony": 0.4,
    "disruption": -1.0,
    "construction": -0.7,
    "maintenance": -0.7,
    "admin": 0.0,
}


def _as_dict(event) -> dict:
    if hasattr(event, "model_dump"):
        return event.model_dump()
    if isinstance(event, dict):
        return dict(event)
    return {k: getattr(event, k) for k in dir(event) if not k.startswith("_") and not callable(getattr(event, k))}


def load_events_json(path: str | Path) -> List[EventInfo]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("events", payload) if isinstance(payload, dict) else payload
    out = []
    for row in rows or []:
        if isinstance(row, EventInfo):
            out.append(row)
        elif isinstance(row, dict):
            allowed = getattr(EventInfo, "model_fields", {})
            if allowed:
                out.append(EventInfo(**{k: v for k, v in row.items() if k in allowed}))
            else:
                out.append(row)
    return out


def load_channel_map(path: str | Path) -> List[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return list(payload.get("channels", []))
    return list(payload)


def channel_meta_by_name(channels: Sequence[dict]) -> Dict[str, dict]:
    return {str(row.get("channel_name")): row for row in channels if row.get("channel_name")}


def _text(event: dict) -> str:
    return " ".join(str(event.get(k, "") or "") for k in ["title", "content", "location", "event_type", "event_category"]).lower()


def _extract_kv(text: str, key: str) -> Optional[str]:
    match = re.search(rf"{re.escape(key)}=([^;|]+)", text)
    return match.group(1).strip() if match else None


def _event_station_id(event: dict) -> str:
    content = f"{event.get('content', '')}; {event.get('location', '')}"
    return str(event.get("station_complex_id") or _extract_kv(content, "station_complex_id") or "")


def _event_channel(event: dict) -> str:
    content = f"{event.get('content', '')}; {event.get('location', '')}"
    return str(event.get("channel_name") or _extract_kv(content, "channel_name") or "")


def _event_time(event: dict) -> Optional[pd.Timestamp]:
    raw = event.get("event_time") or event.get("start_time") or event.get("date")
    if not raw:
        return None
    try:
        return pd.Timestamp(raw)
    except Exception:
        return None


def _duration_hours(event: dict) -> float:
    for key in ["impact_duration_hours", "duration_hours", "duration"]:
        val = event.get(key)
        if val is not None and val != "":
            try:
                return max(1.0, min(float(val), 72.0))
            except Exception:
                pass
    return 4.0


def _distance_m(event: dict) -> Optional[float]:
    val = event.get("distance_m")
    if val is None or val == "":
        match = re.search(r"distance_m=([0-9.]+)", str(event.get("content", "")))
        val = match.group(1) if match else None
    try:
        return float(val) if val is not None else None
    except Exception:
        return None


def _direction(event: dict) -> float:
    explicit = str(event.get("expected_direction") or event.get("impact_direction") or "").lower()
    if explicit == "increase":
        return 1.0
    if explicit == "decrease":
        return -1.0
    blob = _text(event)
    if any(k in blob for k in ["closure", "closed", "maintenance", "service disruption", "construction"]):
        return -1.0
    category = str(event.get("event_category") or event.get("semantic_category") or event.get("event_type") or "").lower()
    for key, score in EVENT_CATEGORY_DIRECTION.items():
        if key in category or key in blob:
            return score
    return 0.0


def _channel_matches(channel_name: str, meta: dict, event: dict) -> Tuple[bool, float]:
    ev_ch = _event_channel(event).lower()
    ev_sid = _event_station_id(event)
    ch = channel_name.lower()
    if ev_ch and (ev_ch in ch or ch in ev_ch):
        return True, 1.0
    if ev_sid and str(meta.get("station_complex_id", "")).startswith(ev_sid):
        return True, 1.0
    if ev_sid and ev_sid in channel_name:
        return True, 1.0
    return False, 0.0


def _rank_score(meta: dict) -> float:
    try:
        rank = float(meta.get("rank", 128))
    except Exception:
        rank = 128.0
    return max(0.0, min(1.0, 1.0 - (rank - 1.0) / 128.0))


def build_event_feature_cube(
    raw_forecast: Sequence[Sequence[float]] | np.ndarray,
    timestamps: Sequence[str],
    channel_names: Sequence[str],
    events: Iterable[dict | EventInfo],
    channel_meta: Dict[str, dict],
) -> np.ndarray:
    """Return (n_channels, horizon, n_features) event/covariate tensor."""
    raw = np.asarray(raw_forecast, dtype=np.float32)
    n_channels, horizon = raw.shape
    cube = np.zeros((n_channels, horizon, len(FEATURE_NAMES)), dtype=np.float32)
    if not timestamps:
        timestamps = [str(i) for i in range(horizon)]
    parsed_ts = [pd.Timestamp(ts) if not isinstance(ts, pd.Timestamp) else ts for ts in timestamps]

    for t_idx, ts in enumerate(parsed_ts[:horizon]):
        hour = float(ts.hour)
        dow = float(ts.dayofweek)
        cube[:, t_idx, FEATURE_NAMES.index("hour_sin")] = math.sin(2 * math.pi * hour / 24.0)
        cube[:, t_idx, FEATURE_NAMES.index("hour_cos")] = math.cos(2 * math.pi * hour / 24.0)
        cube[:, t_idx, FEATURE_NAMES.index("dow_sin")] = math.sin(2 * math.pi * dow / 7.0)
        cube[:, t_idx, FEATURE_NAMES.index("dow_cos")] = math.cos(2 * math.pi * dow / 7.0)
        cube[:, t_idx, FEATURE_NAMES.index("is_weekend")] = 1.0 if ts.dayofweek >= 5 else 0.0
    cube[:, :, FEATURE_NAMES.index("log_raw_forecast")] = np.log1p(np.maximum(raw, 0.0)) / 10.0
    cube[:, :, FEATURE_NAMES.index("bias")] = 1.0

    rows = [_as_dict(e) for e in events or []]
    for event in rows:
        ev_time = _event_time(event)
        if ev_time is None:
            continue
        duration = _duration_hours(event)
        direction = _direction(event)
        tier = TIER_SCORE.get(str(event.get("impact_tier") or "C").upper(), 0.25)
        confidence = CONFIDENCE_SCORE.get(str(event.get("impact_confidence") or "low").lower(), 0.25)
        distance = _distance_m(event)
        distance_score = 0.0 if distance is None else max(0.0, min(1.0, 1.0 - distance / 1500.0))
        duration_score = max(0.0, min(duration / 24.0, 1.0))
        for c_idx, ch_name in enumerate(channel_names):
            meta = channel_meta.get(ch_name, {})
            matched, station_match = _channel_matches(ch_name, meta, event)
            if not matched:
                continue
            rank_score = _rank_score(meta)
            for t_idx, ts in enumerate(parsed_ts[:horizon]):
                delta_h = (ts - ev_time).total_seconds() / 3600.0
                if -2.0 <= delta_h <= duration:
                    proximity = 1.0 - min(abs(delta_h), max(duration, 1.0)) / max(duration, 1.0)
                    vals = {
                        "event_active": 1.0,
                        "expected_direction": direction,
                        "tier_score": tier,
                        "confidence_score": confidence,
                        "duration_score": duration_score,
                        "distance_score": distance_score,
                        "rank_score": rank_score,
                        "station_match": station_match,
                        "spillover_strength": max(0.0, proximity),
                    }
                    for name, val in vals.items():
                        i = FEATURE_NAMES.index(name)
                        cube[c_idx, t_idx, i] = max(cube[c_idx, t_idx, i], float(val)) if val >= 0 else min(cube[c_idx, t_idx, i], float(val))
    return cube


class EventResidualAdapter(nn.Module):
    """Small MLP that predicts multiplicative residual correction."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.05, max_correction: float = 0.10):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_correction = float(max_correction)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        out = self.net(features).squeeze(-1)
        correction = torch.clamp(torch.tanh(out) * self.max_correction, -self.max_correction, self.max_correction)
        if features.shape[-1] >= 1:
            event_gate = torch.clamp(features[..., 0], 0.0, 1.0)
            correction = correction * event_gate
        return correction


def adapter_config(mode: str = "frozen_moment", hidden_dim: int = 64, max_correction: float = 0.10) -> dict:
    return {
        "mode": mode,
        "input_dim": len(FEATURE_NAMES),
        "hidden_dim": int(hidden_dim),
        "dropout": 0.05,
        "max_correction": float(max_correction),
        "feature_names": FEATURE_NAMES,
    }


def build_adapter_from_config(config: dict) -> EventResidualAdapter:
    return EventResidualAdapter(
        input_dim=int(config.get("input_dim", len(FEATURE_NAMES))),
        hidden_dim=int(config.get("hidden_dim", 64)),
        dropout=float(config.get("dropout", 0.05)),
        max_correction=float(config.get("max_correction", 0.10)),
    )


def save_adapter(adapter: EventResidualAdapter, output_dir: str | Path, config: dict, manifest: Optional[dict] = None) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(adapter.state_dict(), output / "event_adapter.pt")
    (output / "adapter_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    if manifest is not None:
        (output / "training_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def load_adapter(adapter_dir: str | Path, device: str = "cpu") -> Tuple[EventResidualAdapter, dict]:
    adapter_dir = Path(adapter_dir)
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    manifest = {}
    manifest_path = adapter_dir / "training_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    adapter = build_adapter_from_config(config)
    state = torch.load(adapter_dir / "event_adapter.pt", map_location=device, weights_only=False)
    adapter.load_state_dict(state)
    blend = config.get("blend_factor")
    if blend is None:
        blend = (manifest.get("validation_blend_calibration") or {}).get("blend_factor", 1.0)
    adapter.blend_factor = float(blend)
    config["blend_factor"] = float(blend)
    config["training_manifest"] = manifest
    adapter.to(device).eval()
    return adapter, config


def apply_correction(
    adapter: EventResidualAdapter,
    raw_forecast: Sequence[Sequence[float]] | np.ndarray,
    features: np.ndarray,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw_forecast, dtype=np.float32)
    with torch.no_grad():
        x = torch.tensor(features.reshape(-1, features.shape[-1]), dtype=torch.float32, device=device)
        correction = adapter.to(device)(x).detach().cpu().numpy().reshape(raw.shape)
    blend = float(getattr(adapter, "blend_factor", 1.0))
    correction = np.clip(correction * blend, -adapter.max_correction, adapter.max_correction).astype(np.float32)
    return (raw * (1.0 + correction)).astype(np.float32), correction


def apply_adapter_to_forecast(
    raw_forecast: Sequence[Sequence[float]] | np.ndarray,
    timestamps: Sequence[str],
    channel_names: Sequence[str],
    events: Iterable[dict | EventInfo],
    channel_meta: Dict[str, dict],
    adapter_dir: str | Path,
    device: str = "cpu",
    channel_indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    adapter, _ = load_adapter(adapter_dir, device=device)
    raw = np.asarray(raw_forecast, dtype=np.float32)
    features = build_event_feature_cube(raw, timestamps, channel_names, events, channel_meta)
    adjusted, correction = apply_correction(adapter, raw, features, device=device)
    if channel_indices is not None:
        mask = np.zeros(raw.shape[0], dtype=bool)
        for idx in channel_indices:
            if 0 <= int(idx) < raw.shape[0]:
                mask[int(idx)] = True
        out = raw.copy()
        out[mask] = adjusted[mask]
        return out.astype(np.float32), correction.astype(np.float32)
    return adjusted.astype(np.float32), correction.astype(np.float32)
