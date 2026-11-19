"""Leaky circuits with a trainable receiver gain and an editable directed mask."""

import numpy as np
import torch
from torch import nn

from .free_graph_model import graph_for
from .ring_logic_model import PARAMETERS, WIDTH, _birth


def strong(mask):
    """Check mutual reachability in [target, source] convention."""
    def visit(neighbors):
        seen = {0}
        pending = [0]
        while pending:
            for other in np.flatnonzero(neighbors[pending.pop()]):
                if other not in seen:
                    seen.add(other)
                    pending.append(other)
        return len(seen) == len(mask)

    return visit(mask) and visit(mask.T)


def birth_mask(seed):
    sources, targets = graph_for(seed, "fully_free").T
    mask = np.zeros((WIDTH, WIDTH), dtype=bool)
    mask[targets, sources] = True
    return mask


def normalize(mask):
    return mask / mask.sum(dim=-1, keepdim=True).clamp_min(1).sqrt()


@torch.jit.script
def advance(candidate: torch.Tensor, keep: torch.Tensor, state: torch.Tensor,
            transmission: torch.Tensor, reset: torch.Tensor, retain: int):
    outputs = torch.jit.annotate(list[torch.Tensor], [])
    for tick in range(candidate.size(2)):
        if reset.numel() > 0:
            state = torch.where(reset[..., tick, None], 0., state)
        step = candidate[:, :, tick, :] + torch.bmm(state, transmission)
        state = torch.lerp(torch.tanh(step), state, keep[:, :, tick, :])
        if tick >= retain:
            outputs.append(state)
    return outputs, state


def simulate(observations, state, normalized, parameters, retain=None, drop=False, reset=None):
    drive, bias, gain, readout, offset = parameters
    if retain is None:
        retain = observations.shape[-2]
    if reset is None:
        reset = observations.new_empty(0)
    batch, clock = observations.shape[1:3]
    projected = observations.flatten(1, 2) @ drive + bias[:, None]
    candidate, leak = projected.reshape(-1, batch, clock, 2 * gain.shape[-1]).chunk(2, dim=-1)
    transmission = (normalized * gain[..., None]).transpose(1, 2)
    if drop:
        transmission = transmission * 0
    values, state = advance(candidate.contiguous(), leak.sigmoid().contiguous(), state,
                            transmission, reset, retain)
    if values:
        prediction = (torch.stack(values, dim=2) @ readout[:, None, :, None]).squeeze(-1)
        return prediction + offset[:, None, None], state
    return None, state


class Bank(nn.Module):
    """Independently fit 449 continuous coefficients in each model row."""

    def __init__(self, seeds, masks, width=WIDTH):
        super().__init__()
        for name, parts in zip(PARAMETERS, zip(*(_birth(seed, width) for seed in seeds)), strict=True):
            self.register_parameter(name, nn.Parameter(torch.stack(parts)))
        self.register_buffer("support", normalize(torch.as_tensor(masks)), persistent=False)
        self.width = width
        expected = (len(seeds), width, width)
        if self.support.shape != expected:
            raise ValueError(f"Mask shape must be {expected}")

    def start(self, batch_size):
        return self.drive.new_zeros((len(self.drive), batch_size, self.width))

    @torch.no_grad()
    def edit(self, masks):
        # Keep the storage address stable for CUDA graph replay across topology changes.
        self.support.copy_(normalize(torch.as_tensor(masks, device=self.drive.device)))

    def forward(self, observations, state, *, retain=None, drop=False, reset=None, path=0):
        if path != 0:
            raise ValueError("The free mask has no named backbone pathway")
        return simulate(observations, state, self.support, tuple(self.parameters()), retain, drop, reset)
