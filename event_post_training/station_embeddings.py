from __future__ import annotations

import torch
from torch import nn


class StationEmbedding(nn.Module):
    def __init__(self, n_stations: int, dim: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(n_stations, dim)

    def forward(self, station_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(station_ids)
