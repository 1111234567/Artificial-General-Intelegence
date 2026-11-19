"""Batched equal-parameter leaky circuits with oriented, reciprocal, or self bonds."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .ring_logic_world import SENSES
from .world import MOVEMENT_TICKS


WIDTH = 32
ARMS = ("directed", "undirected", "self", "multiloop")
PARAMETERS = ("drive", "bias", "gain", "readout", "offset")


def _birth(seed, width):
    rng = torch.Generator().manual_seed(int(seed))
    drive = torch.randn(SENSES, 2 * width, generator=rng) * .35
    bias = torch.cat((torch.zeros(width), torch.full((width,), 1.8)))
    gain = .82 + .07 * torch.randn(width, generator=rng)
    readout = torch.randn(width, generator=rng) / math.sqrt(width)
    return drive, bias, gain, readout, torch.zeros(())


def chord_sources(seed, width=WIDTH):
    """Sample a second permutation of senders, distinct from predecessor and self."""
    rng = torch.Generator().manual_seed(int(seed) ^ 0x72F5CC08)
    index = torch.arange(width)
    predecessor = (index - 1) % width
    while True:
        sources = torch.randperm(width, generator=rng)
        if not ((sources == index) | (sources == predecessor)).any():
            return sources


@torch.jit.script
def advance(candidate: torch.Tensor, keep: torch.Tensor, state: torch.Tensor,
            left: torch.Tensor, right: torch.Tensor, diagonal: torch.Tensor,
            second_source: torch.Tensor, reset: torch.Tensor, retain: int, path: int):
    outputs = torch.jit.annotate(list[torch.Tensor], [])
    for tick in range(candidate.size(2)):
        if reset.numel() > 0:
            state = torch.where(reset[..., tick, None], 0., state)
        proposal = candidate[:, :, tick, :]
        if path == 0 or path == 1 or path == 2 or path == 4:
            proposal = proposal + state.roll(1, -1) * left
        if path == 0 or path == 4:
            proposal = proposal + state.gather(-1, second_source) * right
        if path == 2:
            proposal = proposal + state.roll(-1, -1) * right
        if path == 0 or path == 3:
            proposal = proposal + state * diagonal
        memory = keep[:, :, tick, :]
        state = torch.lerp(torch.tanh(proposal), state, memory)
        if tick >= retain:
            outputs.append(state)
    return outputs, state


class Bank(nn.Module):
    """Pack independently optimized circuits in the leading dimension."""

    def __init__(self, seeds, arms, *, width=WIDTH):
        super().__init__()
        if len(seeds) != len(arms) or any(arm not in ARMS for arm in arms):
            raise ValueError("One registered topology is required for every birth seed")
        for name, parts in zip(PARAMETERS, zip(*(_birth(seed, width) for seed in seeds)), strict=True):
            self.register_parameter(name, nn.Parameter(torch.stack(parts)))
        self.register_buffer("arm", torch.tensor([ARMS.index(a) for a in arms])[:, None],
                             persistent=False)
        successor = (torch.arange(width) + 1) % width
        secondary = [chord_sources(seed, width) if arm == "multiloop" else successor
                     for seed, arm in zip(seeds, arms, strict=True)]
        self.register_buffer("secondary", torch.stack(secondary), persistent=False)
        self.width = width

    def coefficients(self, *, drop=False):
        gain = self.gain
        if drop:
            gain = gain * 0
        reciprocal = (self.arm == 1).to(gain.dtype) / math.sqrt(2)
        branched = (self.arm == 3).to(gain.dtype) / math.sqrt(2)
        left = gain * ((self.arm == 0) + reciprocal + branched)
        right = gain.roll(-1, -1) * reciprocal + gain * branched
        diagonal = gain * (self.arm == 2)
        return left, right, diagonal

    def matrix(self):
        """Explicit recurrent candidate matrix in [target, source] convention."""
        left, right, diagonal = self.coefficients()
        index = torch.arange(self.width, device=left.device)
        matrix = torch.diag_embed(diagonal)
        matrix[:, index, (index - 1) % self.width] = left
        matrix.scatter_(-1, self.secondary[..., None], right[..., None])
        return matrix

    def start(self, batch_size):
        return self.drive.new_zeros((len(self.drive), batch_size, self.width))

    def forward(self, observations, state, *, retain=None, drop=False, reset=None, path=0):
        """Advance the true recurrent state while retaining only requested output ticks."""
        if retain is None:
            retain = observations.shape[-2]
        if reset is None:
            reset = observations.new_empty(0)
        batch, clock = observations.shape[1:3]
        projected = observations.flatten(1, 2) @ self.drive + self.bias[:, None]
        candidate, leak = projected.reshape(-1, batch, clock, 2 * self.width).chunk(2, dim=-1)
        left, right, diagonal = (arr[:, None, :] for arr in self.coefficients(drop=drop))
        second_source = self.secondary[:, None, :].expand(-1, batch, -1)
        values, state = advance(candidate.contiguous(), leak.sigmoid().contiguous(), state,
                                left, right, diagonal, second_source, reset, retain, path)
        if values:
            prediction = (torch.stack(values, dim=2) @ self.readout[:, None, :, None]).squeeze(-1)
            return prediction + self.offset[:, None, None], state
        return None, state


def train_step(bank, prefix, query, sampled_side, taste, target_rules, groups,
               sensory_ticks, repetitions, optimizer):
    count = taste.shape[-1]
    cells = torch.arange(count * repetitions, device=query.device) % count
    roots = torch.randint(len(prefix), (int(groups.shape[0]), len(cells)), device=query.device)
    # Groups are banked in the same order as births. The same block of indices is
    # used by all topologies and targets trained from that seed.
    roots = roots.repeat_interleave(len(bank.drive) // len(groups), dim=0)
    with torch.no_grad():
        _, state = bank(prefix[roots], bank.start(len(cells)))
    observations = query[roots, cells, :sensory_ticks + MOVEMENT_TICKS]
    prediction, _ = bank(observations, state, retain=sensory_ticks)
    contacted = sampled_side[roots, cells]
    receipt = taste[target_rules[:, None], roots, cells]
    logit = prediction * (2 * contacted.clamp_min(0) - 1)[..., None]
    target = (receipt > 0).to(logit.dtype)[..., None].expand_as(logit)
    valid = receipt != 0
    error = F.binary_cross_entropy_with_logits(logit, target, reduction="none").mean(dim=-1)
    # The objective is a sum over independent models. Adding a model to the bank
    # leaves the derivative for every other model unchanged.
    loss = ((error * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)).sum()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.detach()
