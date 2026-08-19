from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd


def build_calendar_matrix(timestamps: Sequence[str]) -> np.ndarray:
    rows = []
    for ts in timestamps:
        t = pd.Timestamp(ts)
        rows.append([
            math.sin(2 * math.pi * t.hour / 24.0),
            math.cos(2 * math.pi * t.hour / 24.0),
            math.sin(2 * math.pi * t.dayofweek / 7.0),
            math.cos(2 * math.pi * t.dayofweek / 7.0),
            1.0 if t.dayofweek >= 5 else 0.0,
            1.0,
        ])
    return np.asarray(rows, dtype=np.float32)
