"""Forward-only inherited local plasticity on a single reciprocal port graph."""
from __future__ import annotations

from functools import lru_cache
import math
import random

import torch

from .model import connected_regular_mask


NODES = 80
INPUTS = 56
JOBS = 2
ACTIONS = 11
MODES = ("full", "weights", "frozen", "shuffled")
RULE_FEATURES = (
    "bias", "weight", "coactivity", "eligibility", "endpoint_difference",
    "begin_accepted_eligibility", "finish_accepted_eligibility",
    "begin_rejected_eligibility", "finish_rejected_eligibility",
)
WEIGHT_LIMIT = .45
WEIGHT_STEP = .003
GATE_STEP = .18
GATE_THRESHOLD = .16
GATE_INTERVAL = 12
EDGE_COOLDOWN = 36
MAX_EVENTS = 4


@lru_cache(maxsize=None)
def _pairs(nodes):
    return tuple(zip(*torch.triu_indices(nodes, nodes, offset=1).tolist()))


def _edge_id(nodes, u, v):
    u, v = min(u, v), max(u, v)
    return u * (2 * nodes - u - 1) // 2 + v - u - 1


def make_genome(seed, nodes=NODES):
    """Draw a connected degree-16 birth graph and a two-output local rule."""
    if nodes < INPUTS + JOBS * ACTIONS:
        raise ValueError("graph must contain all fixed input and action ports")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    mask = connected_regular_mask(nodes, 16, seed, switches_per_edge=2)
    weights = torch.randn(mask.shape, generator=rng).masked_fill_(~mask, 0)
    weights.mul_(.65 * math.sqrt(nodes / 2) / float(weights.norm()))
    node_bias = torch.randn(nodes, generator=rng) * .025
    node_bias[INPUTS:INPUTS + JOBS * ACTIONS:ACTIONS] += .04
    rule = torch.randn(2, len(RULE_FEATURES), generator=rng) * .18
    rule[0, 0] *= .2
    rule[0, 1] -= .2
    return {"birth_w": weights.clamp_(-WEIGHT_LIMIT, WEIGHT_LIMIT), "birth_mask": mask,
            "node_bias": node_bias, "rule": rule}


def mutate_genome(parent, seed, wide=False):
    """Perturb inherited values and occasionally switch two birth edges."""
    rng = torch.Generator(device="cpu").manual_seed(seed)
    chooser = random.Random(seed)
    child = {key: value.clone() for key, value in parent.items()}
    weights, mask = child["birth_w"], child["birth_mask"]
    scale = 2.5 if wide else 1.
    live = mask.nonzero().flatten().tolist()
    rewires = 4 if wide else 1
    nodes = child["node_bias"].numel()
    endpoints = _pairs(nodes)
    for _ in range(rewires * 6):
        if rewires == 0:
            break
        i, j = chooser.sample(range(len(live)), 2)
        first, second = live[i], live[j]
        a, b = endpoints[first]
        c, d = endpoints[second]
        if len({a, b, c, d}) != 4:
            continue
        if chooser.getrandbits(1):
            c, d = d, c
        left, right = _edge_id(nodes, a, d), _edge_id(nodes, b, c)
        if mask[left] or mask[right]:
            continue
        weights[left], weights[right] = weights[first], weights[second]
        weights[first] = weights[second] = 0.
        mask[left] = mask[right] = True
        mask[first] = mask[second] = False
        live[i], live[j] = left, right
        rewires -= 1
    touched = torch.rand(weights.shape, generator=rng).lt(.16 if wide else .09) & mask
    weights.add_(torch.randn(weights.shape, generator=rng) * touched * (.037 * scale))
    weights.clamp_(-WEIGHT_LIMIT, WEIGHT_LIMIT)
    for key, step, limit in (("node_bias", .009, .35), ("rule", .065, 2.)):
        value = child[key]
        value.add_(torch.randn(value.shape, generator=rng) * (step * scale)).clamp_(-limit, limit)
    return child


class GraphBatch:
    """Each batched actor retains its own h, reciprocal weights, gates and traces."""

    @torch.no_grad()
    def __init__(self, genomes, actors, device, mode="full", seed=0):
        if not genomes or actors < 1 or mode not in MODES:
            raise ValueError("graph batch requires genomes, actors and a valid plasticity mode")
        self.mode, self.actors, self.population = mode, actors, len(genomes)
        self.nodes = genomes[0]["node_bias"].numel()
        self.device = torch.device(device)
        self.step = 0
        self.u, self.v = torch.triu_indices(self.nodes, self.nodes, offset=1, device=self.device)
        self.pair_count = self.u.numel()
        self.max_edges = min(self.pair_count, 11 * self.nodes)
        self.active_set_changes = mode in ("full", "shuffled")

        def inherited(key):
            return torch.stack([genome[key] for genome in genomes]).to(self.device)

        birth_weights, birth_mask = inherited("birth_w"), inherited("birth_mask")
        self.bias = inherited("node_bias").unsqueeze(1)
        self.rule = inherited("rule")
        self.initial_edges = birth_mask.sum(-1)
        if bool(self.initial_edges.gt(self.max_edges).any()):
            raise ValueError("birth graph exceeds the lifetime edge cap")
        self.mask = birth_mask.unsqueeze(1).expand(-1, actors, -1).clone()
        self.w = birth_weights.unsqueeze(1).expand(-1, actors, -1).clone().masked_fill_(~self.mask, 0.)
        self.h = self.w.new_zeros(self.population, actors, self.nodes)
        self.activity = self.h.clone()
        self.matrix = self.w.new_zeros(self.population, actors, self.nodes, self.nodes)
        self.input_shape = (self.population, actors, INPUTS)
        shape = (self.population, actors)
        self.births = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.deaths = torch.zeros_like(self.births)
        self.weight_updates = torch.zeros_like(self.births)
        self.weight_change = self.w.new_zeros(shape)
        self.g = self.mask.to(self.w.dtype).mul_(.12).sub_(.06)
        self.node_trace = self.h.clone()
        self.eligibility = torch.zeros_like(self.w)
        self.last_change = torch.zeros_like(self.mask, dtype=torch.int32)
        self._update_matrix()
        if mode != "frozen":
            self._index_live()
        if mode == "shuffled":
            self.receipt_history = self.w.new_zeros(16, *shape, 4)
            rng = random.Random(seed)
            self.receipt_lags = tuple(rng.randrange(2, 16) for _ in range(64))

    def _update_matrix(self):
        self.matrix[..., self.u, self.v] = self.w
        self.matrix[..., self.v, self.u] = self.w

    def _index_live(self):
        values, self.live = self.mask.to(torch.int8).topk(self.max_edges, dim=-1, sorted=False)
        self.live_mask = values.bool()
        self.live_u, self.live_v = self.u[self.live], self.v[self.live]

    def _receipt(self, obs):
        jobs = obs.reshape(self.population, self.actors, JOBS, 28)
        begin = jobs[..., 12:14].sum(-1)
        finish = jobs[..., 14:22].sum(-1)
        attempted = torch.stack((begin, finish), dim=-1)
        accepted = (attempted * jobs[..., 22:23]).mean(-2)
        rejected = (attempted * jobs[..., 23:24]).mean(-2)
        receipt = torch.cat((accepted, rejected), dim=-1)
        if self.mode == "shuffled":
            index = self.step % len(self.receipt_history)
            lag = self.receipt_lags[self.step % len(self.receipt_lags)]
            self.receipt_history[index].copy_(receipt)
            receipt = self.receipt_history[(index - lag) % len(self.receipt_history)]
        return receipt

    def _score(self, rule, receipt, weight, first, second, eligibility):
        coefficient = (receipt * rule[:, None, 5:]).sum(-1, keepdim=True)
        return (rule[:, 0, None, None] + rule[:, 1, None, None] * weight / WEIGHT_LIMIT
                + rule[:, 2, None, None] * first * second
                + (rule[:, 3, None, None] + coefficient) * eligibility
                + rule[:, 4, None, None] * (first - second).abs())

    def _change_weights(self, local_activity, receipt):
        first = local_activity.gather(-1, self.live_u)
        second = local_activity.gather(-1, self.live_v)
        old_trace = self.eligibility.gather(-1, self.live)
        old_weight = self.w.gather(-1, self.live)
        new_trace = (.9 * old_trace + .1 * first * second).masked_fill_(~self.live_mask, 0.)
        score = self._score(self.rule[:, 0], receipt, old_weight, first, second, new_trace)
        proposal = (old_weight + WEIGHT_STEP * score.tanh()).clamp_(-WEIGHT_LIMIT, WEIGHT_LIMIT)
        new_weight = proposal.masked_fill_(~self.live_mask, 0.)
        self.w.scatter_(-1, self.live, new_weight)
        self.eligibility.scatter_(-1, self.live, new_trace)
        self.weight_updates.add_(new_weight.ne(old_weight).sum(-1))
        self.weight_change.add_((new_weight - old_weight).abs().sum(-1))

    def _change_support(self, local_activity, receipt):
        first, second = local_activity[..., self.u], local_activity[..., self.v]
        trace = self.node_trace[..., self.u] * self.node_trace[..., self.v]
        drive = self._score(self.rule[:, 1], receipt, self.w, first, second, trace)
        self.g.add_(GATE_STEP * drive.tanh()).clamp_(-1., 1.)

        ready = self.last_change.le(self.step - EDGE_COOLDOWN)
        retire_score = (-self.g).masked_fill_(~(ready & self.mask), -torch.inf)
        scores, retiring = retire_score.topk(MAX_EVENTS, dim=-1)
        retire = scores.ge(GATE_THRESHOLD)
        self.mask.scatter_(-1, retiring, self.mask.gather(-1, retiring) & ~retire)
        self.w.scatter_(-1, retiring, self.w.gather(-1, retiring).masked_fill_(retire, 0.))
        self.eligibility.scatter_(-1, retiring, self.eligibility.gather(-1, retiring).masked_fill_(retire, 0.))
        last = torch.where(retire, self.step, self.last_change.gather(-1, retiring))
        self.last_change.scatter_(-1, retiring, last)
        self.deaths.add_(retire.sum(-1))

        room = self.max_edges - self.mask.sum(-1, keepdim=True)
        ready = self.last_change.le(self.step - EDGE_COOLDOWN)
        scores, budding = self.g.masked_fill(~(ready & ~self.mask), -torch.inf).topk(MAX_EVENTS, dim=-1)
        admit = scores.ge(GATE_THRESHOLD) & (torch.arange(MAX_EVENTS, device=self.device) < room)
        self.mask.scatter_(-1, budding, self.mask.gather(-1, budding) | admit)
        current = self.w.gather(-1, budding)
        birth_drive = self._score(self.rule[:, 0], receipt, current, first.gather(-1, budding),
                                  second.gather(-1, budding), trace.gather(-1, budding)).tanh()
        initial_weight = .08 * birth_drive + torch.where(birth_drive.ge(0), .01, -.01)
        self.w.scatter_(-1, budding, torch.where(admit, initial_weight, current))
        last = torch.where(admit, self.step, self.last_change.gather(-1, budding))
        self.last_change.scatter_(-1, budding, last)
        self.births.add_(admit.sum(-1))
        self._index_live()

    @torch.no_grad()
    def act(self, obs):
        """Consume only public observations and return one native action per job."""
        if tuple(obs.shape) != self.input_shape:
            raise ValueError(f"expected observation shape {self.input_shape}")
        obs = obs.to(device=self.device, dtype=self.w.dtype)
        # The receipt belongs to the previous action. Credit its state before
        # injecting the next command into the continuous graph.
        if self.mode != "frozen" and self.step:
            receipt = self._receipt(obs)
            local_activity = (4. * self.h).tanh()
            self.node_trace.mul_(.95).add_(local_activity, alpha=.05)
            self._change_weights(local_activity, receipt)
            if self.active_set_changes and self.step % GATE_INTERVAL == 0:
                self._change_support(local_activity, receipt)
            self._update_matrix()
        current = torch.bmm(self.matrix.flatten(0, 1), self.activity.flatten(0, 1).unsqueeze(-1))
        current = current.reshape_as(self.h)
        self.h.mul_(.75).add_(current + self.bias, alpha=.25)
        self.h[..., :INPUTS].add_(obs, alpha=.25)
        actions = self.h[..., INPUTS:INPUTS + JOBS * ACTIONS].reshape(
            self.population, self.actors, JOBS, ACTIONS).argmax(-1)
        self.activity = self.h.tanh()
        self.step += 1
        return actions

    @torch.no_grad()
    def summary(self):
        """Return candidate totals for changes and actor means for edge counts."""
        return {
            "weight_updates": self.weight_updates.sum(-1),
            "weight_change": self.weight_change.sum(-1),
            "births": self.births.sum(-1),
            "deaths": self.deaths.sum(-1),
            "initial_edges": self.initial_edges.clone(),
            "final_edges": self.mask.sum(-1).float().mean(-1),
        }
