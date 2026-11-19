"""Closed directed graph families with matched leaky, 32-state circuits."""

from functools import cache
import hashlib
import math
import random

import numpy as np
import torch
from torch import nn

from .ring_logic_model import PARAMETERS, WIDTH, _birth, chord_sources


EDGE_COUNT = 2 * WIDTH
ARMS = ("regular", "free_chords", "fully_free")


def strongly_connected(n, arcs):
    """Two traversals from the same vertex characterize strong connectivity."""
    forward = [[] for _ in range(n)]
    backward = [[] for _ in range(n)]
    for source, target in arcs:
        forward[source].append(target)
        backward[target].append(source)
    for neighbors in (forward, backward):
        reached = {0}
        agenda = [0]
        while agenda:
            pending = set(neighbors[agenda.pop()]) - reached
            reached.update(pending)
            agenda.extend(pending)
        if len(reached) != n:
            return False
    return True


@cache
def _completion_counts(n, length):
    """Number of sequences completing a prefix with m vertices still absent."""
    counts = [[1] + [0] * n]
    for _ in range(length):
        prior = counts[-1]
        counts.append([n * prior[0]] + [(n - m) * prior[m] + m * prior[m - 1]
                                       for m in range(1, n + 1)])
    return counts


def covering_sequence(rng, n, length):
    counts = _completion_counts(n, length)
    seen = []
    missing = list(range(n))
    sequence = []
    for rest in range(length - 1, -1, -1):
        m = len(missing)
        old = (n - m) * counts[rest][m]
        draw = rng.randrange(counts[rest + 1][m])
        if draw < old:
            value = seen[draw // counts[rest][m]]
        else:
            value = missing.pop((draw - old) // counts[rest][m - 1])
            seen.append(value)
        sequence.append(value)
    return sequence


def sample_uniform_graph(rng, n, count):
    """Exact uniform simple directed support conditional on strong connectivity."""
    if n < 2 or not n <= count <= n * (n - 1):
        raise ValueError("No strongly connected non-self graph has these dimensions")
    while True:
        arcs = list(zip(covering_sequence(rng, n, count), covering_sequence(rng, n, count),
                        strict=True))
        if any(a == b for a, b in arcs) or len(set(arcs)) != count:
            continue
        if strongly_connected(n, arcs):
            return arcs


@cache
def _immutable_graph(seed, arm):
    nodes = range(WIDTH)
    backbone = [((target - 1) % WIDTH, target) for target in nodes]
    if arm == "regular":
        arcs = backbone + list(zip(chord_sources(seed).tolist(), nodes, strict=True))
    elif arm == "free_chords":
        rng = random.Random((seed << 16) ^ 0x5B1D264100)
        available = [(source, target) for target in nodes for source in nodes
                     if source != target and source != (target - 1) % WIDTH]
        arcs = backbone + rng.sample(available, WIDTH)
    elif arm == "fully_free":
        rng = random.Random((seed << 16) ^ 0x539E4BD62B)
        arcs = sample_uniform_graph(rng, WIDTH, EDGE_COUNT)
    else:
        raise ValueError(f"Unknown graph family {arm!r}")
    return tuple(sorted(arcs, key=lambda arc: (arc[1], arc[0])))


def graph_for(seed, arm):
    """Canonical [source, target] arc list sorted by receiver, then sender."""
    return np.asarray(_immutable_graph(int(seed), arm), dtype=np.int64)


def graph_record(seed, arm):
    arcs = graph_for(seed, arm)
    sources, receivers = arcs.T
    adjacency = np.zeros((WIDTH, WIDTH), dtype=np.uint8)
    adjacency[receivers, sources] = 1
    walks = adjacency.copy()
    shortest_return = np.zeros(WIDTH, dtype=np.int16)
    for length in range(1, WIDTH + 1):
        np.copyto(shortest_return, length, where=(walks.diagonal() != 0) & (shortest_return == 0))
        walks = ((walks @ adjacency) != 0).astype(np.uint8)
    return {
        "arcs_source_target": arcs.tolist(),
        "adjacency_sha256_uint8_target_source": hashlib.sha256(adjacency.tobytes()).hexdigest(),
        "in_degree": adjacency.sum(axis=1).tolist(),
        "out_degree": adjacency.sum(axis=0).tolist(),
        "strongly_connected": strongly_connected(WIDTH, arcs),
        "self_edges": int(adjacency.trace()),
        "arc_count": int(adjacency.sum()),
        "ordered_backbone_arcs_present": int(adjacency[np.arange(WIDTH),
                                                        (np.arange(WIDTH) - 1) % WIDTH].sum()),
        "reciprocal_arcs": int((adjacency & adjacency.T).sum()),
        "shortest_directed_return": shortest_return.tolist(),
    }


@torch.jit.script
def advance(candidate: torch.Tensor, keep: torch.Tensor, state: torch.Tensor,
            recurrent: torch.Tensor, reset: torch.Tensor, retain: int):
    outputs = torch.jit.annotate(list[torch.Tensor], [])
    for tick in range(candidate.size(2)):
        if reset.numel() > 0:
            state = torch.where(reset[..., tick, None], 0., state)
        proposal = candidate[:, :, tick, :] + torch.bmm(state, recurrent)
        state = torch.lerp(proposal.tanh(), state, keep[:, :, tick, :])
        if tick >= retain:
            outputs.append(state)
    return outputs, state


class Bank(nn.Module):
    """Use one gain per recipient or one gain per arc with a common dense kernel."""

    def __init__(self, seeds, arms, *, independent=False):
        super().__init__()
        graph = torch.from_numpy(np.stack([graph_for(seed, arm) for seed, arm in
                                          zip(seeds, arms, strict=True)]))
        source, receiver = graph.unbind(dim=-1)
        degree = torch.zeros(len(seeds), WIDTH, dtype=torch.int64)
        degree.scatter_add_(1, receiver, torch.ones_like(receiver))
        self.register_buffer("receiver", receiver, persistent=False)
        self.register_buffer("addresses", receiver * WIDTH + source, persistent=False)
        self.register_buffer("degree", degree, persistent=False)
        self.independent = independent
        for name, parts in zip(PARAMETERS, zip(*(_birth(seed, WIDTH) for seed in seeds),
                                               strict=True), strict=True):
            init = torch.stack(parts)
            if name == "gain" and independent:
                init = init.gather(1, receiver)
            self.register_parameter(name, nn.Parameter(init))

    def matrix(self, *, drop=False):
        """Return candidate matrix with target rows and source columns."""
        gain = self.gain if self.independent else self.gain.gather(1, self.receiver)
        scale = self.degree.gather(1, self.receiver).to(gain.dtype).rsqrt()
        values = gain * scale
        if drop:
            values = values * 0
        return values.new_zeros((len(values), WIDTH * WIDTH)).scatter(
            1, self.addresses, values).reshape(-1, WIDTH, WIDTH)

    def start(self, batch_size):
        return self.drive.new_zeros((len(self.drive), batch_size, WIDTH))

    def forward(self, observations, state, *, retain=None, drop=False, reset=None, path=0):
        if path != 0:
            raise ValueError("The free graph bank has one common recurrent kernel")
        batch, clock = observations.shape[1:3]
        if retain is None:
            retain = clock
        if reset is None:
            reset = observations.new_empty(0)
        projected = observations.flatten(1, 2) @ self.drive + self.bias[:, None]
        candidate, leak = projected.reshape(-1, batch, clock, 2 * WIDTH).chunk(2, dim=-1)
        values, state = advance(candidate.contiguous(), leak.sigmoid().contiguous(), state,
                                self.matrix(drop=drop).transpose(1, 2), reset, retain)
        if values:
            prediction = (torch.stack(values, dim=2) @ self.readout[:, None, :, None]).squeeze(-1)
            return prediction + self.offset[:, None, None], state
        return None, state
