"""One ordinary shared predictor; no protected weights or learned critic."""
from __future__ import annotations

import torch
from torch import nn


class Predictor(nn.Module):
    def __init__(self, features: int, classes: int, width: int, seed: int):
        super().__init__()
        # Initialization does not advance any environment or memory RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.net = nn.Sequential(nn.Linear(features, width), nn.ReLU(),
                                     nn.Linear(width, width), nn.ReLU(),
                                     nn.Linear(width, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
