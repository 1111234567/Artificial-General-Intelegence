"""Batched Torch transition and public observations for the pulse-game policy."""

from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import Tensor


OBS_SIZE = 20
ACTION_TICKS = 15
TICKS = 16


def independent_context_map(tape):
    """Change the B map distribution without altering the public trial schedule."""
    maps = tape.good_map.copy()
    for row, seed in enumerate(tape.seeds):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0x856CD99A]))
        maps[row, 1] = rng.integers(0, 2, size=maps.shape[-1])
    good = maps[np.arange(tape.n)[:, None], tape.ctx, tape.cue]
    return replace(tape, good_map=maps, good_id=good)


@dataclass
class BatchTape:
    ctx: Tensor
    cue: Tensor
    left_id: Tensor
    good_id: Tensor
    distances: Tensor
    wind: Tensor

    @classmethod
    def from_tape(cls, tape, device="cpu"):
        return cls(
            torch.as_tensor(tape.ctx, device=device, dtype=torch.long),
            torch.as_tensor(tape.cue, device=device, dtype=torch.long),
            torch.as_tensor(tape.left_id, device=device, dtype=torch.long),
            torch.as_tensor(tape.good_id, device=device, dtype=torch.long),
            torch.as_tensor(tape.distances, device=device, dtype=torch.float32),
            torch.as_tensor(tape.wind, device=device, dtype=torch.float32),
        )

    def trial(self, index: int) -> "Trial":
        return Trial(
            self.ctx[:, index], self.cue[:, index], self.left_id[:, index], self.good_id[:, index],
            self.distances[:, index, :], self.wind[:, index, :],
        )


@dataclass
class Trial:
    ctx: Tensor
    cue: Tensor
    left_id: Tensor
    good_id: Tensor
    distances: Tensor
    wind: Tensor

    def initial(self):
        left = self.left_id
        zero = torch.zeros_like(left)
        x = zero.to(self.distances.dtype)
        return Vehicle(x, x, torch.ones_like(left, dtype=torch.bool), zero - 1, zero - 1)

    def static_observation(self, device, dtype):
        binary = torch.eye(2, device=device, dtype=dtype)
        cues = torch.eye(4, device=device, dtype=dtype)
        return torch.cat((binary[self.ctx], cues[self.cue], binary[self.left_id], binary[1 - self.left_id]), dim=1)


@dataclass
class Vehicle:
    x: Tensor
    v: Tensor
    active: Tensor
    receipt_id: Tensor
    last_contact: Tensor

    def observe(self, trial: Trial, static: Tensor, tick: int) -> Tensor:
        valid = self.receipt_id >= 0
        win = self.receipt_id == trial.good_id
        dynamic = torch.stack((
            self.x, self.v, self.active & (tick < ACTION_TICKS), valid & win, valid & ~win,
            self.receipt_id == 0, self.receipt_id == 1, torch.full_like(self.active, tick == 0),
        ), dim=1)
        return torch.cat((static, dynamic.to(static.dtype), trial.distances.to(static.dtype)), dim=1)

    def step(self, trial: Trial, pulse: Tensor, tick: int) -> "Vehicle":
        v = 0.65 * self.v + 0.15 * pulse + trial.wind[:, tick]
        x = self.x + v
        left = self.active & (x <= -trial.distances[:, 0])
        right = self.active & (x >= trial.distances[:, 1])
        contact = left | right
        parked = ~self.active | contact
        receipt = torch.where(left, trial.left_id, torch.where(right, 1 - trial.left_id, -1))
        x = torch.where(left, -trial.distances[:, 0], torch.where(right, trial.distances[:, 1], x))
        return Vehicle(
            torch.where(self.active, x, self.x), torch.where(parked, 0.0, v),
            ~parked, receipt, torch.where(contact, receipt, self.last_contact),
        )
