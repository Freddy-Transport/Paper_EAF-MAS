"""Configuration helpers for event-aware post-training."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from agents import config as agent_cfg

PROJECT_ROOT = Path(agent_cfg.PROJECT_ROOT)
SEQ_LEN = agent_cfg.SEQ_LEN
FORECAST_HORIZON = agent_cfg.FORECAST_HORIZON
N_TRAIN_ROWS = 12 * 30 * 24
N_VAL_ROWS = 4 * 30 * 24
RESIDUAL_EPS = 1.0


def resolve_lp_model_path(path: str | Path, project_root: str | Path = PROJECT_ROOT) -> Tuple[Path, str]:
    requested = Path(path)
    if (requested / "lp_weights.pt").is_file():
        return requested, "requested"
    root = Path(project_root)
    fallback = root / "experiments" / "outputs" / "lp_fallback_nyc_top128"
    if (fallback / "lp_weights.pt").is_file():
        return fallback, "lp_fallback_nyc_top128"
    return requested, "missing"


@dataclass
class EventPostTrainingConfig:
    traffic_csv: Path = PROJECT_ROOT / "data" / "nyc_top128_station_hourly_flow.csv"
    events_json: Path = PROJECT_ROOT / "data" / "nyc_top128_station_events.json"
    channel_map_path: Path = PROJECT_ROOT / "data" / "nyc_top128_channel_map.json"
    fusion_channels_path: Path = PROJECT_ROOT / "data" / "venue37_fusion_channels.json"
    lp_model_path: Path = PROJECT_ROOT / "experiments" / "outputs" / "lp_top128"
    output_dir: Path = PROJECT_ROOT / "experiments" / "outputs" / "event_adapter_top128"
    moment_cache_dir: Path = PROJECT_ROOT / "experiments" / "outputs" / "event_adapter_top128" / "moment_cache"
    seq_len: int = SEQ_LEN
    horizon: int = FORECAST_HORIZON
    train_rows: int = N_TRAIN_ROWS
    val_rows: int = N_VAL_ROWS
    residual_bound: float = 0.10
    device: str = "cuda:0"
    channel_names: List[str] = field(default_factory=list)
    venue_channel_names: List[str] = field(default_factory=list)
    venue_top128_indices: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name in ["traffic_csv", "events_json", "channel_map_path", "fusion_channels_path", "lp_model_path", "output_dir", "moment_cache_dir"]:
            setattr(self, name, Path(getattr(self, name)))
        self._load_channels()

    def _load_channels(self) -> None:
        if self.channel_map_path.is_file():
            payload = json.loads(self.channel_map_path.read_text(encoding="utf-8"))
            channels = payload.get("channels", payload if isinstance(payload, list) else [])
            self.channel_names = [row["channel_name"] for row in channels if row.get("channel_name")]
        if self.fusion_channels_path.is_file():
            payload = json.loads(self.fusion_channels_path.read_text(encoding="utf-8"))
            self.venue_channel_names = list(payload.get("channel_names", []))
        name_to_idx = {name: i for i, name in enumerate(self.channel_names)}
        self.venue_top128_indices = [name_to_idx[name] for name in self.venue_channel_names if name in name_to_idx]
