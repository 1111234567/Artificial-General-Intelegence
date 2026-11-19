"""Optional directed training backend with row-major plastic state and BMM weights."""

from __future__ import annotations

from functools import partial

import torch

from . import uncapped_directed_reward as backend
from . import uncapped_directed_reward_run as run
from .evolved_plasticity import ACTIONS, EDGE_COOLDOWN, GATE_INTERVAL, INPUTS, JOBS, WEIGHT_LIMIT
from .evolved_plasticity_world import FEATURES_PER_SLOT, FIRST_FINISH, N_ACTIONS, NEW, READY
from .uncapped_directed_reward_fast_run import (
    FIRST_RECEIPT, REQUEST_TICKS, _SkipEmptyReceiptTicks, _checked_schedule,
)
from .uncapped_plastic_graph import BIRTH_THRESHOLD, DEATH_THRESHOLD, GATE_RATE
from .uncapped_reward_precision import _ACK, _REJ


class _DenseRewardPolicy(backend._DirectedRewardPolicy):
    """Dense state for the receipt cadence validated by the training evaluator."""

    def __init__(self, birth, actors, device, mode):
        self.mode, self.actors, self.population = mode, actors, birth["node_bias"].size(0)
        self.nodes = birth["node_bias"].size(-1)
        self.device = device
        u, v = torch.triu_indices(self.nodes, self.nodes, offset=1, device=device)
        self.pair_count = u.numel()
        self.arc_indices = torch.cat((u * self.nodes + v, v * self.nodes + u))
        batch = (self.population, actors)
        ports = (*batch, self.nodes)
        arcs = (*ports, self.nodes)
        self.input_shape = (*batch, INPUTS)

        half = dict(device=device, dtype=torch.float16)
        self.bias = torch.empty((self.population, 1, self.nodes), **half)
        self.rule = torch.empty_like(birth["rule"])
        self.reward_rate = torch.empty_like(birth["reward_rate"])
        self.w = self.matrix = torch.empty(arcs, **half)
        self.proposal = torch.empty_like(self.w)
        self.g = torch.empty_like(self.w)
        self.eligibility = torch.empty_like(self.w)
        self.h = torch.empty(ports, **half)
        self.activity = torch.empty_like(self.h)
        self.node_trace = torch.empty_like(self.h)
        self.reward = torch.empty(batch, **half)
        self.birth_positive = torch.full((), .01, **half)
        self.birth_negative = torch.full((), -.01, **half)
        self.pending = torch.empty(batch, device=device, dtype=torch.bool)
        self.tried = torch.empty_like(self.pending)
        self.mask = torch.empty(arcs, device=device, dtype=torch.bool)
        self.last_change = torch.empty(arcs, device=device, dtype=torch.int32)
        self.initial_edges = torch.empty(self.population, device=device, dtype=torch.long)
        self.edge_count = torch.empty(batch, device=device, dtype=torch.long)
        self.edge_ticks = torch.empty_like(self.edge_count)
        self.births = torch.empty_like(self.edge_count)
        self.deaths = torch.empty_like(self.edge_count)
        self.weight_updates = torch.empty_like(self.edge_count)
        self.reward_events = torch.empty_like(self.edge_count)
        self.attempted_writes = torch.empty_like(self.edge_count)
        self.weight_change = torch.empty(batch, device=device, dtype=torch.float32)
        self.reset(birth)

    def reset(self, birth):
        self.step = self.edge_span_start = 0
        self.bias.copy_(birth["node_bias"].unsqueeze(1))
        self.rule.copy_(birth["rule"])
        self.reward_rate.copy_(birth["reward_rate"])
        shape = (self.population, self.nodes * self.nodes)
        birth_mask = self.mask.new_zeros(shape).index_copy_(-1, self.arc_indices, birth["birth_mask"])
        birth_w = self.w.new_zeros(shape).index_copy_(-1, self.arc_indices, birth["birth_w"])
        birth_w.masked_fill_(~birth_mask, 0.)
        self.mask.copy_(birth_mask.view(self.population, 1, self.nodes, self.nodes))
        self.w.copy_(birth_w.view(self.population, 1, self.nodes, self.nodes))
        self.initial_edges.copy_(birth["birth_mask"].sum(-1))
        self.edge_count.copy_(self.initial_edges.unsqueeze(1))
        for name in (*backend._ZERO_STATE, "reward", "pending", "tried", "reward_events", "attempted_writes"):
            if name != "matrix":  # Aliases w; the copy above sets the diagonal to +0.
                getattr(self, name).zero_()
        # No diagonal entry can ever become ready to change support.
        self.last_change.diagonal(dim1=-2, dim2=-1).fill_(torch.iinfo(torch.int32).max)

    def act(self, obs):
        if tuple(obs.shape) != self.input_shape:
            raise ValueError(f"expected observation shape {self.input_shape}")
        obs = obs.to(device=self.device, dtype=self.h.dtype)
        port = obs[..., :FEATURES_PER_SLOT]
        if self.step:
            self.reward.copy_((port[..., _ACK] - port[..., _REJ]) * self.pending)
            event = self.reward.ne(0)
            self.reward_events.add_(event)
            support_step = self.mode != "frozen" and self.step % GATE_INTERVAL == 0
            if self.mode != "frozen":
                local_activity = (4. * self.h).tanh()
                self.node_trace.mul_(.95).add_(local_activity, alpha=.05)
                if self.mode == "full" or support_step:
                    row, column = local_activity.unsqueeze(-1), local_activity.unsqueeze(-2)
                    coactivity = row * column
                if self.mode == "full":
                    self._change_weights(coactivity, event)
                if support_step:
                    self._change_support(coactivity, (row - column).abs_(), self._receipt(obs))

        self.tried.logical_and_(~port[..., NEW].bool())
        current = torch.bmm(self.matrix.flatten(0, 1), self.activity.flatten(0, 1).unsqueeze(-1))
        self.h.mul_(.75).add_(current.reshape_as(self.h) + self.bias, alpha=.25)
        self.h[..., :INPUTS].add_(obs, alpha=.25)
        action_ports = self.h[..., INPUTS:INPUTS + JOBS * ACTIONS]
        actions = action_ports.reshape(self.population, self.actors, JOBS, ACTIONS).argmax(-1)
        primary = actions[..., 0]
        self.pending.copy_(port[..., READY].bool() & ~self.tried
                           & primary.ge(FIRST_FINISH) & primary.lt(N_ACTIONS))
        self.tried.logical_or_(self.pending)
        torch.tanh(self.h, out=self.activity)
        self.step += 1
        return actions

    def _change_weights(self, coactivity, event):
        self.eligibility.lerp_(coactivity, .1).masked_fill_(~self.mask, 0.)
        if self.step % REQUEST_TICKS < FIRST_RECEIPT:
            return
        event_edges = event[..., None, None]
        self.attempted_writes.add_((self.eligibility.ne(0) & event_edges).sum((-2, -1)))
        scale = (self.reward_rate[:, None] * self.reward)[..., None, None]
        proposal = torch.addcmul(self.w, self.eligibility, scale, out=self.proposal).clamp_(-WEIGHT_LIMIT, WEIGHT_LIMIT)
        torch.where(event_edges, proposal, self.w, out=proposal)
        self.weight_updates.add_(proposal.ne(self.w).sum((-2, -1)))
        # Match the reference reduction shape and input order: an FP32 sum
        # can round differently if the FP16 differences are visited row-major.
        change = (proposal - self.w).abs_().flatten(-2).index_select(-1, self.arc_indices)
        self.weight_change.add_(change.view(self.population, self.actors, 2, self.pair_count)
                                .sum((-2, -1), dtype=torch.float32))
        self.w.copy_(proposal)

    def _change_support(self, coactivity, difference, receipt):
        self.edge_ticks.add_(self.edge_count, alpha=self.step - self.edge_span_start)
        self.edge_span_start = self.step
        trace = self.node_trace.unsqueeze(-1) * self.node_trace.unsqueeze(-2)
        drive = self._arc_drive(self.rule[:, 1], receipt, coactivity, difference, trace)
        drive.diagonal(dim1=-2, dim2=-1).zero_()
        self.g.lerp_(drive.tanh_(), GATE_RATE)
        ready = self.last_change.le(self.step - EDGE_COOLDOWN)
        retire = ready & self.mask & self.g.le(DEATH_THRESHOLD)
        admit = ready & ~self.mask & self.g.ge(BIRTH_THRESHOLD)
        birth_drive = self._arc_drive(self.rule[:, 0], receipt, coactivity, difference, trace).tanh_()
        offset = torch.where(birth_drive.ge(0), self.birth_positive, self.birth_negative)
        initial_weight = .08 * birth_drive + offset
        self.w.copy_(torch.where(admit, initial_weight, self.w).masked_fill_(retire, 0.))
        self.eligibility.masked_fill_(retire, 0.)
        changed = admit | retire
        self.mask.logical_xor_(changed)
        self.last_change.masked_fill_(changed, self.step)
        admitted, retired = admit.sum((-2, -1)), retire.sum((-2, -1))
        self.births.add_(admitted)
        self.deaths.add_(retired)
        self.edge_count.add_(admitted).sub_(retired)

    def summary(self):
        edge_ticks = self.edge_ticks + self.edge_count * (self.step - self.edge_span_start)
        totals = self.edge_count.sum(-1)
        reciprocated = (self.mask & self.mask.transpose(-2, -1)).sum((-3, -2, -1))
        updates = self.weight_updates.sum(-1)
        return {
            "initial_edges": self.initial_edges.clone(),
            "final_edges": self.edge_count.float().mean(-1),
            "mean_edges": edge_ticks.float().mean(-1) / self.step if self.step else self.initial_edges.float(),
            "final_reciprocity": reciprocated / totals.clamp_min(1),
            "births": self.births.sum(-1),
            "deaths": self.deaths.sum(-1),
            "reward_events": self.reward_events.sum(-1),
            "attempted_writes": self.attempted_writes.sum(-1),
            "changed_writes": updates,
            "weight_updates": updates,
            "weight_change": self.weight_change.sum(-1),
        }


class _DenseTrainingEvaluation(backend.RewardEvaluation):
    def __init__(self, schedule):
        super().__init__(partial(_checked_schedule, schedule))

    def __call__(self, *args, device, **kwargs):
        # As in the existing fast runner, training evaluation runs on the main thread.
        # CPU FP16 lerp can choose different scalar/vector rounding at layout boundaries.
        original = backend._DirectedRewardPolicy
        backend._DirectedRewardPolicy = (
            _DenseRewardPolicy if torch.device(device).type == "cuda" else _SkipEmptyReceiptTicks
        )
        try:
            return super().__call__(*args, device=device, **kwargs)
        finally:
            backend._DirectedRewardPolicy = original


def _evaluation(schedule):
    if schedule is run.training_schedule:
        return _DenseTrainingEvaluation(schedule)
    return backend.RewardEvaluation(schedule)


def main():
    original = run.RewardEvaluation
    run.RewardEvaluation = _evaluation
    try:
        print("Training uses row-major plastic state and skips guaranteed-empty receipt writes.", flush=True)
        run.main()
    finally:
        run.RewardEvaluation = original


if __name__ == "__main__":
    main()
