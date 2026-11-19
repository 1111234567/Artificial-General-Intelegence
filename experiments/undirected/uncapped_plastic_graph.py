"""Forward-only reciprocal graph with local hysteresis over every node pair."""
from __future__ import annotations

import torch

from .evolved_plasticity import (
    EDGE_COOLDOWN, GATE_INTERVAL, INPUTS, WEIGHT_LIMIT, WEIGHT_STEP, GraphBatch,
)


GATE_RATE = .12
BIRTH_THRESHOLD = .45
DEATH_THRESHOLD = -.45


class UncappedGraphBatch(GraphBatch):
    """Full plasticity; any pair can change support independently of edge count."""

    @torch.no_grad()
    def __init__(self, genomes, actors, device, mode="full", seed=0):
        if not genomes or actors < 1 or mode != "full":
            raise ValueError("uncapped graph requires genomes, actors and full plasticity")
        self.mode, self.actors, self.population = mode, actors, len(genomes)
        self.active_set_changes = True
        self.nodes = genomes[0]["node_bias"].numel()
        self.device = torch.device(device)
        self.step = 0
        self.u, self.v = torch.triu_indices(self.nodes, self.nodes, offset=1, device=self.device)
        self.pair_count = self.u.numel()

        def inherited(key):
            return torch.stack([genome[key] for genome in genomes]).to(self.device)

        birth_weights, birth_mask = inherited("birth_w"), inherited("birth_mask")
        self.bias = inherited("node_bias").unsqueeze(1)
        self.rule = inherited("rule")
        self.initial_edges = birth_mask.sum(-1)
        shape = (self.population, actors)
        self.mask = birth_mask.unsqueeze(1).expand(-1, actors, -1).clone()
        self.w = birth_weights.unsqueeze(1).expand(-1, actors, -1).clone().masked_fill_(~self.mask, 0.)
        self.h = self.w.new_zeros(*shape, self.nodes)
        self.activity = torch.zeros_like(self.h)
        self.matrix = self.w.new_zeros(*shape, self.nodes, self.nodes)
        self.input_shape = (*shape, INPUTS)
        self.births = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.deaths = torch.zeros_like(self.births)
        self.weight_updates = torch.zeros_like(self.births)
        self.weight_change = self.w.new_zeros(shape)
        self.g = torch.zeros_like(self.w)
        self.node_trace = torch.zeros_like(self.h)
        self.eligibility = torch.zeros_like(self.w)
        self.last_change = torch.zeros_like(self.mask, dtype=torch.int32)
        self.edge_count = self.initial_edges.unsqueeze(1).expand(-1, actors).clone()
        self.edge_ticks = torch.zeros_like(self.edge_count)
        self.edge_span_start = 0
        self._update_matrix()

    @staticmethod
    def _pair_drive(rule, receipt, weight, coactivity, difference, eligibility):
        coefficient = (receipt * rule[:, None, 5:]).sum(-1, keepdim=True)
        drive = torch.addcmul(rule[:, 0, None, None], weight, rule[:, 1, None, None] / WEIGHT_LIMIT)
        drive.addcmul_(coactivity, rule[:, 2, None, None])
        drive.addcmul_(eligibility, rule[:, 3, None, None] + coefficient)
        return drive.addcmul_(difference, rule[:, 4, None, None])

    def _pair_activity(self, local_activity):
        first = local_activity.index_select(-1, self.u)
        second = local_activity.index_select(-1, self.v)
        return first * second, (first - second).abs_()

    def _change_weights(self, local_activity, receipt):
        coactivity, difference = self._pair_activity(local_activity)
        self.eligibility.lerp_(coactivity, .1).masked_fill_(~self.mask, 0.)
        drive = self._pair_drive(self.rule[:, 0], receipt, self.w, coactivity,
                                 difference, self.eligibility)
        proposal = torch.add(self.w, drive.tanh_(), alpha=WEIGHT_STEP)
        proposal.clamp_(-WEIGHT_LIMIT, WEIGHT_LIMIT).masked_fill_(~self.mask, 0.)
        self.weight_updates.add_(proposal.ne(self.w).sum(-1))
        self.weight_change.add_((proposal - self.w).abs_().sum(-1))
        self.w = proposal

    def _change_support(self, local_activity, receipt):
        self.edge_ticks.add_(self.edge_count, alpha=self.step - self.edge_span_start)
        self.edge_span_start = self.step

        coactivity, difference = self._pair_activity(local_activity)
        trace = self.node_trace.index_select(-1, self.u) * self.node_trace.index_select(-1, self.v)
        drive = self._pair_drive(self.rule[:, 1], receipt, self.w, coactivity, difference, trace)
        # Bounded drive passes through a slow local average. From neutral,
        # even saturating evidence needs five gate updates to change support.
        self.g.lerp_(drive.tanh_(), GATE_RATE)
        ready = self.last_change.le(self.step - EDGE_COOLDOWN)
        retire = ready & self.mask & self.g.le(DEATH_THRESHOLD)
        admit = ready & ~self.mask & self.g.ge(BIRTH_THRESHOLD)

        birth_drive = self._pair_drive(self.rule[:, 0], receipt, self.w, coactivity,
                                       difference, trace).tanh_()
        initial_weight = .08 * birth_drive + torch.where(birth_drive.ge(0), .01, -.01)
        self.w = torch.where(admit, initial_weight, self.w).masked_fill_(retire, 0.)
        self.eligibility.masked_fill_(retire, 0.)
        self.mask.logical_xor_(admit | retire)
        self.last_change.masked_fill_(admit | retire, self.step)

        admitted, retired = admit.sum(-1), retire.sum(-1)
        self.births.add_(admitted)
        self.deaths.add_(retired)
        self.edge_count.add_(admitted).sub_(retired)

    @torch.no_grad()
    def summary(self):
        """Include the exact mean active edge count over actors and action ticks."""
        result = super().summary()
        edge_ticks = self.edge_ticks + self.edge_count * (self.step - self.edge_span_start)
        result["mean_edges"] = (edge_ticks.float().mean(-1) / self.step if self.step
                                else self.initial_edges.float())
        return result
