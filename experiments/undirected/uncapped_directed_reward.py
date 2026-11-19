"""FP16 reward-write evaluator with independent arcs and reciprocal controls.

Both arms use the same kernels and layout. For the upper-triangle index pairs
``(u, v)`` with ``u < v``, the first genome half fills ``W[u, v]`` (v to u)
and the second fills ``W[v, u]`` (u to v) in the same order. Every arc has
its own weight, mask, gate, eligibility and cooldown. The endpoint features
are symmetric, so equal twins in the reciprocal arm remain exactly equal.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from .evolved_plasticity import ACTIONS, EDGE_COOLDOWN, GATE_INTERVAL, INPUTS, JOBS, RULE_FEATURES, WEIGHT_LIMIT
from .evolved_plasticity_world import FEATURES_PER_SLOT, FIRST_FINISH, N_ACTIONS, NEW, READY, _int, _receipt_table
from .uncapped_answer_feedback import _RecordPrimaryActions, _answer_metrics
from .uncapped_bf16_backend import _device_tape
from .uncapped_cuda_backend import CUDAGraphUnavailable, _Capture, _ZERO_STATE, _rollout_with_preloaded_receipts
from .uncapped_plastic_graph import BIRTH_THRESHOLD, DEATH_THRESHOLD, GATE_RATE, UncappedGraphBatch
from .uncapped_reward_precision import _ACK, _CACHE_LOCK, _REJ, _first_finish_metrics, _matmul_precision


MODES = ("full", "no_reward_write", "frozen")


def _stack_directed_genomes(genomes):
    if not genomes:
        raise ValueError("reward evaluation requires at least one genome")
    nodes = genomes[0]["node_bias"].numel()
    if nodes < INPUTS + JOBS * ACTIONS:
        raise ValueError("graph must contain all fixed input and action ports")
    pairs = nodes * (nodes - 1) // 2
    fields = {
        "birth_w": ((2 * pairs,), torch.float32),
        "birth_mask": ((2 * pairs,), torch.bool),
        "node_bias": ((nodes,), torch.float32),
        "rule": ((2, len(RULE_FEATURES)), torch.float32),
        "directed": ((), torch.bool),
        "reward_rate": ((), torch.float32),
    }
    birth = {}
    for key, (shape, dtype) in fields.items():
        values = [genome[key] for genome in genomes]
        if any(not isinstance(field, torch.Tensor) or field.device.type != "cpu"
               or field.shape != shape or field.dtype != dtype for field in values):
            raise ValueError(f"expected CPU {key} tensors of shape {shape} and dtype {dtype}")
        birth[key] = torch.stack(values)
    rates = birth["reward_rate"]
    if not (rates.isfinite() & rates.ge(0)).all():
        raise ValueError("reward_rate must be finite and nonnegative")
    reciprocal = ~birth.pop("directed")
    for field in ("birth_mask", "birth_w"):
        twins = birth[field][reciprocal].view(-1, 2, pairs)
        if not torch.equal(twins[:, 0], twins[:, 1]):
            raise ValueError(f"reciprocal genomes require equal arc twins in {field}")
    # Rates are FP32 opmath operands; recurrent and per-arc storage is FP16.
    for field in ("birth_w", "node_bias", "rule"):
        birth[field] = birth[field].half()
    return nodes, birth


class _DirectedRewardPolicy(UncappedGraphBatch):
    def __init__(self, birth, actors, device, mode):
        self.mode, self.actors, self.population = mode, actors, birth["node_bias"].size(0)
        self.nodes = birth["node_bias"].size(-1)
        self.device = device
        self.u, self.v = torch.triu_indices(self.nodes, self.nodes, offset=1, device=device)
        self.pair_count = self.u.numel()
        self.arc_indices = torch.cat((self.u * self.nodes + self.v, self.v * self.nodes + self.u))
        batch = (self.population, actors)
        arcs = (*batch, 2, self.pair_count)
        ports = (*batch, self.nodes)
        self.input_shape = (*batch, INPUTS)

        half = dict(device=device, dtype=torch.float16)
        self.bias = torch.empty((self.population, 1, self.nodes), **half)
        self.rule = torch.empty_like(birth["rule"])
        self.reward_rate = torch.empty_like(birth["reward_rate"])
        self.w = torch.empty(arcs, **half)
        self.proposal = torch.empty_like(self.w)
        self.g = torch.empty_like(self.w)
        self.eligibility = torch.empty_like(self.w)
        self.h = torch.empty(ports, **half)
        self.activity = torch.empty_like(self.h)
        self.node_trace = torch.empty_like(self.h)
        self.matrix = torch.empty((*ports, self.nodes), **half)
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
        self.mask.copy_(birth["birth_mask"].view(self.population, 1, 2, self.pair_count))
        self.w.copy_(birth["birth_w"].view(self.population, 1, 2, self.pair_count)).masked_fill_(~self.mask, 0.)
        self.initial_edges.copy_(birth["birth_mask"].sum(-1))
        self.edge_count.copy_(self.initial_edges.unsqueeze(1))
        for name in (*_ZERO_STATE, "reward", "pending", "tried", "reward_events", "attempted_writes"):
            getattr(self, name).zero_()
        self._update_matrix()

    def _update_matrix(self):
        # One indexed write updates both directions; the zero diagonal is
        # established by reset and never appears in arc_indices.
        self.matrix.flatten(-2).index_copy_(-1, self.arc_indices, self.w.flatten(-2))

    def act(self, obs):
        if tuple(obs.shape) != self.input_shape:
            raise ValueError(f"expected observation shape {self.input_shape}")
        obs = obs.to(device=self.device, dtype=self.h.dtype)
        port = obs[..., :FEATURES_PER_SLOT]
        if self.step:
            # The incoming receipt belongs to the previous action, even if
            # this observation is simultaneously the start of a new request.
            self.reward.copy_((port[..., _ACK] - port[..., _REJ]) * self.pending)
            event = self.reward.ne(0)
            self.reward_events.add_(event)
            support_step = self.mode != "frozen" and self.step % GATE_INTERVAL == 0
            if self.mode != "frozen":
                local_activity = (4. * self.h).tanh()
                self.node_trace.mul_(.95).add_(local_activity, alpha=.05)
                if self.mode == "full" or support_step:
                    first = local_activity.index_select(-1, self.u)
                    second = local_activity.index_select(-1, self.v)
                    coactivity = (first * second).unsqueeze(-2)
                if self.mode == "full":
                    self._change_weights(coactivity, event)
                if support_step:
                    difference = (first - second).abs_().unsqueeze(-2)
                    self._change_support(coactivity, difference, self._receipt(obs))
                if self.mode == "full" or support_step:
                    self._update_matrix()

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
        event_edges = event[..., None, None]
        self.attempted_writes.add_((self.eligibility.ne(0) & event_edges).sum((-2, -1)))
        # Use FP32 arithmetic for the genome-specific rate with one rounding
        # back to FP16 in the output operand. CUDA casts the results as it
        # writes them, avoiding a full FP32 proposal and a separate copy kernel.
        scale = (self.reward_rate[:, None] * self.reward)[..., None, None]
        proposal = torch.addcmul(self.w, self.eligibility, scale, out=self.proposal).clamp_(-WEIGHT_LIMIT, WEIGHT_LIMIT)
        torch.where(event_edges, proposal, self.w, out=proposal)
        self.weight_updates.add_(proposal.ne(self.w).sum((-2, -1)))
        self.weight_change.add_((proposal - self.w).abs_().sum((-2, -1), dtype=torch.float32))
        self.w.copy_(proposal)

    def _arc_drive(self, rule, receipt, coactivity, difference, trace):
        coefficient = (receipt * rule[:, None, 5:]).sum(-1)[..., None, None]
        # The same endpoint features broadcast over both directions; the
        # weight term and the support state still belong to the individual arc.
        bias, weight, coactive, eligibility, endpoint = (
            value[:, None, None, None] for value in rule[:, :5].unbind(-1)
        )
        drive = torch.addcmul(bias, self.w, weight / WEIGHT_LIMIT)
        drive.addcmul_(coactivity, coactive)
        drive.addcmul_(trace, eligibility + coefficient)
        return drive.addcmul_(difference, endpoint)

    def _change_support(self, coactivity, difference, receipt):
        self.edge_ticks.add_(self.edge_count, alpha=self.step - self.edge_span_start)
        self.edge_span_start = self.step
        trace = (self.node_trace.index_select(-1, self.u) * self.node_trace.index_select(-1, self.v)).unsqueeze(-2)
        drive = self._arc_drive(self.rule[:, 1], receipt, coactivity, difference, trace)
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
        reciprocated = (self.mask[..., 0, :] & self.mask[..., 1, :]).sum((-2, -1)) * 2
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


def _rollout(policy, tape, table):
    observer = _RecordPrimaryActions(policy, tape, policy.population)
    world = _rollout_with_preloaded_receipts(table)(observer, tape, policy.population)
    answer = _answer_metrics(observer.actions, tape)
    return {**policy.summary(), **world, **answer, **_first_finish_metrics(observer.actions, tape)}


class _RewardCapture(_Capture):
    def __init__(self, birth, tape, actors, device, mode):
        self.stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(self.stream):
            self.birth = {key: tensor.to(device) for key, tensor in birth.items()}
            self.tape = _device_tape(tape, device)
            self.receipt_table = _receipt_table(device).half()
            self.policy = _DirectedRewardPolicy(self.birth, actors, device, mode)
            _rollout(self.policy, self.tape, self.receipt_table)
        self.graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(self.graph, stream=self.stream):
                self.policy.reset(self.birth)
                self.results = _rollout(self.policy, self.tape, self.receipt_table)
        except RuntimeError as error:
            raise CUDAGraphUnavailable(f"directed reward CUDA Graph capture failed: {error}") from error


class RewardEvaluation:
    """Evaluate fresh per-actor state with FP16 storage and FP32 CUDA GEMM accumulation."""

    def __init__(self, schedule):
        self.schedule = schedule
        self._cache = {}

    def clear_cache(self):
        """Synchronize and release this evaluator's captured rollout shapes."""
        with _CACHE_LOCK:
            for _, captured in self._cache.values():
                captured.stream.synchronize()
            self._cache.clear()

    @torch.no_grad()
    def __call__(self, genomes, *, seed, actors, device, visits=4, mode="full", capture=True):
        device = torch.device(device)
        actors = _int("actors", actors, 1)
        if mode not in MODES:
            raise ValueError(f"expected one of {MODES}, got {mode!r}")
        if device.type not in ("cpu", "cuda") or torch.is_autocast_enabled(device.type):
            raise ValueError("directed reward evaluation requires CPU or CUDA with autocast disabled")
        if capture and device.type != "cuda":
            raise CUDAGraphUnavailable("directed reward CUDA Graph replay requires CUDA")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise CUDAGraphUnavailable("directed reward evaluation requires CUDA availability")

        nodes, birth = _stack_directed_genomes(genomes)
        tape = self.schedule(seed=seed, actors=actors, device="cpu", visits=visits)
        tape = replace(tape, public=tape.public.half())
        if device.type == "cpu":
            return _rollout(_DirectedRewardPolicy(birth, actors, device, mode), tape, _receipt_table(device).half())

        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        with _CACHE_LOCK, torch.cuda.device(device), _matmul_precision(torch.float16, False):
            if not capture:
                policy = _DirectedRewardPolicy({key: tensor.to(device) for key, tensor in birth.items()},
                                               actors, device, mode)
                return _rollout(policy, _device_tape(tape, device), _receipt_table(device).half())

            if torch.cuda.is_current_stream_capturing():
                raise CUDAGraphUnavailable("directed reward CUDA Graph replay cannot nest in an active capture")
            caller_stream = torch.cuda.current_stream(device)
            key = (len(genomes), actors, nodes, mode, tuple(tape.public.shape))
            cached = self._cache.get(device.index)
            if cached is None or cached[0] != key:
                if cached is not None:
                    cached[1].stream.synchronize()
                    del self._cache[device.index], cached
                captured = _RewardCapture(birth, tape, actors, device, mode)
                self._cache[device.index] = (key, captured)
            else:
                _, captured = cached
            return captured.evaluate(birth, tape, caller_stream)
