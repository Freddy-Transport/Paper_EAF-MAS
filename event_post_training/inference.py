from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np

from agents.event_adapter import apply_adapter_to_forecast, channel_meta_by_name, load_channel_map


def apply_event_adapter(raw_forecast, timestamps, channel_names, events, channel_map_path, adapter_dir, device="cpu") -> Tuple[np.ndarray, np.ndarray]:
    meta = channel_meta_by_name(load_channel_map(channel_map_path))
    return apply_adapter_to_forecast(raw_forecast, timestamps, channel_names, events, meta, adapter_dir, device=device)
