"""Fixed directed supports with independently fitted weights on active arcs."""

import math

import numpy as np
import torch
from torch import nn

from .ring_logic_model import _birth, train_step


ARMS = ("archive", "degree_switch")


@torch.jit.script
def advance(candidate: torch.Tensor, keep: torch.Tensor, state: torch.Tensor,
            transpose: torch.Tensor, reset: torch.Tensor, retain: int):
    outputs = torch.jit.annotate(list[torch.Tensor], [])
    for tick in range(candidate.size(2)):
        if reset.numel() > 0:
            state = torch.where(reset[..., tick, None], 0., state)
        proposal = torch.baddbmm(candidate[:, :, tick, :], state, transpose)
        state = torch.lerp(torch.tanh(proposal), state, keep[:, :, tick, :])
        if tick >= retain:
            outputs.append(state)
    return outputs, state


class Bank(nn.Module):
    def __init__(self, seeds, supports):
        super().__init__()
        if len(seeds) != len(supports) or not seeds:
            raise ValueError("Each fitted model requires a seed and a support")
        self.width = supports[0].shape[0]
        count = int(supports[0].sum())
        positions = []
        values = []
        for seed, support in zip(seeds, supports, strict=True):
            if support.shape != (self.width, self.width) or int(support.sum()) != count:
                raise ValueError("All supports must share the node and edge counts")
            target, source = np.nonzero(support)
            positions.append(torch.from_numpy(target * self.width + source))
            drive, bias, _, readout, offset = _birth(seed, self.width)
            rng = torch.Generator().manual_seed(int(seed) ^ 0x2A691E12)
            indegrees = torch.from_numpy(support.sum(axis=1)[target]).float()
            edges = torch.randn(count, generator=rng) * .82 * indegrees.rsqrt()
            values.append((drive, bias, edges, readout, offset))
        for name, parts in zip(("drive", "bias", "edges", "readout", "offset"),
                               zip(*values, strict=True), strict=True):
            self.register_parameter(name, nn.Parameter(torch.stack(parts)))
        self.register_buffer("positions", torch.stack(positions))

    def matrix(self):
        """The recurrent candidate matrix, indexed by [model, target, source]."""
        flat = self.edges.new_zeros((len(self.drive), self.width * self.width))
        return flat.scatter(-1, self.positions, self.edges).view(-1, self.width, self.width)

    def start(self, batch_size):
        return self.drive.new_zeros((len(self.drive), batch_size, self.width))

    def forward(self, observations, state, *, retain=None, drop=False, reset=None, path=0):
        """Use the shared physical scorer's bank interface."""
        if path != 0:
            raise ValueError("Archived support is evaluated as one directed matrix")
        if retain is None:
            retain = observations.shape[-2]
        if reset is None:
            reset = observations.new_empty(0)
        batch, clock = observations.shape[1:3]
        projected = observations.flatten(1, 2) @ self.drive + self.bias[:, None]
        candidate, leak = projected.reshape(-1, batch, clock, 2 * self.width).chunk(2, dim=-1)
        matrix = self.matrix().transpose(-1, -2)
        if drop:
            matrix = torch.zeros_like(matrix)
        states, state = advance(candidate.contiguous(), leak.sigmoid().contiguous(),
                                state, matrix, reset, retain)
        if not states:
            return None, state
        prediction = (torch.stack(states, dim=2) @ self.readout[:, None, :, None]).squeeze(-1)
        return prediction + self.offset[:, None, None], state
