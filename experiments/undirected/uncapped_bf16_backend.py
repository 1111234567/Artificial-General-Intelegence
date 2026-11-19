"""CUDA Graph evaluator for the uncapped policy with BF16 forward computation.

Recurrent state, reciprocal matrices, observations, rule coefficients and local
traces use BF16. Master weights, online weight addition, support gate averages,
and floating-point summary accumulators use FP32. Masks and event counts retain
their original boolean and integer types.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from threading import Lock

import torch

from .evolved_plasticity import ACTIONS, EDGE_COOLDOWN, GATE_INTERVAL, INPUTS, JOBS, WEIGHT_LIMIT, WEIGHT_STEP
from .evolved_plasticity_world import _int, _receipt_table
from .plasticity_continuous_world import VISITS, make_continuous_schedule
from .uncapped_cuda_backend import (
    CUDAGraphUnavailable, _Capture, _SHAPES_PER_DEVICE, _TAPE, _ZERO_STATE,
    _rollout_with_preloaded_receipts, _stack_genomes,
)
from .uncapped_plastic_graph import BIRTH_THRESHOLD, DEATH_THRESHOLD, GATE_RATE, UncappedGraphBatch


_CACHE = {}
_CACHE_LOCK = Lock()


class _BF16Policy(UncappedGraphBatch):
    def __init__(self, birth, actors, device):
        self.mode, self.actors, self.population = "full", actors, birth["node_bias"].size(0)
        self.active_set_changes = True
        self.nodes = birth["node_bias"].size(-1)
        self.device = device
        self.u, self.v = torch.triu_indices(self.nodes, self.nodes, offset=1, device=device)
        self.pair_count = self.u.numel()
        batch = (self.population, actors)
        pair_shape = (*batch, self.pair_count)
        port_shape = (*batch, self.nodes)
        self.input_shape = (*batch, INPUTS)

        bf16 = dict(device=device, dtype=torch.bfloat16)
        fp32 = dict(device=device, dtype=torch.float32)
        self.bias = torch.empty((self.population, 1, self.nodes), **bf16)
        self.rule = torch.empty(birth["rule"].shape, **bf16)
        self.h = torch.empty(port_shape, **bf16)
        self.activity = torch.empty_like(self.h)
        self.matrix = torch.empty((*port_shape, self.nodes), **bf16)
        self.node_trace = torch.empty_like(self.h)
        self.eligibility = torch.empty(pair_shape, **bf16)
        self.forward_w = torch.empty_like(self.eligibility)
        self.w = torch.empty(pair_shape, **fp32)
        self.g = torch.empty_like(self.w)
        self.weight_change = torch.empty(batch, **fp32)
        self.mask = torch.empty(pair_shape, device=device, dtype=torch.bool)
        self.last_change = torch.empty_like(self.mask, dtype=torch.int32)
        self.initial_edges = torch.empty((self.population,), device=device, dtype=torch.long)
        self.edge_count = torch.empty(batch, device=device, dtype=torch.long)
        self.edge_ticks = torch.empty_like(self.edge_count)
        self.births = torch.empty_like(self.edge_count)
        self.deaths = torch.empty_like(self.edge_count)
        self.weight_updates = torch.empty_like(self.edge_count)
        self.reset(birth)

    def reset(self, birth):
        self.step = self.edge_span_start = 0
        self.bias.copy_(birth["node_bias"].unsqueeze(1))
        self.rule.copy_(birth["rule"])
        self.mask.copy_(birth["birth_mask"].unsqueeze(1))
        self.w.copy_(birth["birth_w"].unsqueeze(1)).masked_fill_(~self.mask, 0.)
        self.initial_edges.copy_(birth["birth_mask"].sum(-1))
        self.edge_count.copy_(self.initial_edges.unsqueeze(1))
        for name in _ZERO_STATE:
            getattr(self, name).zero_()
        self._update_matrix()

    def _update_matrix(self):
        self.forward_w.copy_(self.w)
        self.matrix[..., self.u, self.v] = self.forward_w
        self.matrix[..., self.v, self.u] = self.forward_w

    def _change_weights(self, local_activity, receipt):
        coactivity, difference = self._pair_activity(local_activity)
        self.eligibility.lerp_(coactivity, .1).masked_fill_(~self.mask, 0.)
        drive = self._pair_drive(self.rule[:, 0], receipt, self.forward_w, coactivity,
                                 difference, self.eligibility)
        proposal = torch.add(self.w, drive.tanh_(), alpha=WEIGHT_STEP)
        proposal.clamp_(-WEIGHT_LIMIT, WEIGHT_LIMIT).masked_fill_(~self.mask, 0.)
        self.weight_updates.add_(proposal.ne(self.w).sum(-1))
        self.weight_change.add_((proposal - self.w).abs_().sum(-1))
        self.w = proposal

    def _change_support(self, local_activity, receipt):
        self.edge_ticks.add_(self.edge_count, alpha=self.step - self.edge_span_start)
        self.edge_span_start = self.step

        # The support rule sees the master proposal from this tick's weight change.
        self.forward_w.copy_(self.w)
        coactivity, difference = self._pair_activity(local_activity)
        trace = self.node_trace.index_select(-1, self.u) * self.node_trace.index_select(-1, self.v)
        drive = self._pair_drive(self.rule[:, 1], receipt, self.forward_w, coactivity, difference, trace)
        self.g.lerp_(drive.tanh_().float(), GATE_RATE)
        ready = self.last_change.le(self.step - EDGE_COOLDOWN)
        retire = ready & self.mask & self.g.le(DEATH_THRESHOLD)
        admit = ready & ~self.mask & self.g.ge(BIRTH_THRESHOLD)

        birth_drive = self._pair_drive(self.rule[:, 0], receipt, self.forward_w, coactivity,
                                       difference, trace).tanh_().float()
        initial_weight = .08 * birth_drive + torch.where(birth_drive.ge(0), .01, -.01)
        self.w = torch.where(admit, initial_weight, self.w).masked_fill_(retire, 0.)
        self.eligibility.masked_fill_(retire, 0.)
        changed = admit | retire
        self.mask.logical_xor_(changed)
        self.last_change.masked_fill_(changed, self.step)

        admitted, retired = admit.sum(-1), retire.sum(-1)
        self.births.add_(admitted)
        self.deaths.add_(retired)
        self.edge_count.add_(admitted).sub_(retired)

    def act(self, obs):
        if tuple(obs.shape) != self.input_shape:
            raise ValueError(f"expected observation shape {self.input_shape}")
        obs = obs.to(device=self.device, dtype=self.h.dtype)
        if self.step:
            receipt = self._receipt(obs)
            local_activity = (4. * self.h).tanh()
            self.node_trace.mul_(.95).add_(local_activity, alpha=.05)
            self._change_weights(local_activity, receipt)
            if self.step % GATE_INTERVAL == 0:
                self._change_support(local_activity, receipt)
            self._update_matrix()
        current = torch.bmm(self.matrix.flatten(0, 1), self.activity.flatten(0, 1).unsqueeze(-1))
        self.h.mul_(.75).add_(current.reshape_as(self.h) + self.bias, alpha=.25)
        self.h[..., :INPUTS].add_(obs, alpha=.25)
        action_ports = self.h[..., INPUTS:INPUTS + JOBS * ACTIONS]
        actions = action_ports.reshape(self.population, self.actors, JOBS, ACTIONS).argmax(-1)
        self.activity = self.h.tanh()
        self.step += 1
        return actions


def _device_tape(tape, device):
    return replace(tape, **{key: getattr(tape, key).to(device) for key in _TAPE})


def _rollout(policy, tape, receipt_table):
    rollout = _rollout_with_preloaded_receipts(receipt_table)
    metrics = rollout(policy, tape, policy.population)
    return {**policy.summary(), **metrics}


class _BF16Capture(_Capture):
    def __init__(self, birth, tape, actors, device):
        self.stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(self.stream):
            self.birth = {key: tensor.to(device) for key, tensor in birth.items()}
            self.tape = _device_tape(tape, device)
            self.receipt_table = _receipt_table(device).bfloat16()
            self.policy = _BF16Policy(self.birth, actors, device)
            _rollout(self.policy, self.tape, self.receipt_table)

        self.graph = torch.cuda.CUDAGraph()
        # Both tensors get rebound by act during tracing. Retain the entry
        # allocations whose contents the captured reset restores each replay.
        self.entry_w = self.policy.w
        self.entry_activity = self.policy.activity
        try:
            with torch.cuda.graph(self.graph, stream=self.stream):
                self.policy.reset(self.birth)
                self.results = _rollout(self.policy, self.tape, self.receipt_table)
        except RuntimeError as error:
            raise CUDAGraphUnavailable(f"uncapped BF16 CUDA Graph capture failed: {error}") from error


def _inputs(genomes, seed, actors, device, visits, mode):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise CUDAGraphUnavailable("uncapped BF16 evaluation requires native CUDA BF16 support")
    if mode != "full":
        raise ValueError("uncapped graph requires full plasticity")
    if torch.is_autocast_enabled("cuda"):
        raise CUDAGraphUnavailable("uncapped BF16 sets its own per-tensor dtypes")
    device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise CUDAGraphUnavailable(f"{device} lacks native CUDA BF16 support")
    actors = _int("actors", actors, 1)
    nodes, birth = _stack_genomes(genomes)
    tape = make_continuous_schedule(seed=seed, actors=actors, device="cpu", visits=visits)
    tape = replace(tape, public=tape.public.bfloat16())
    return actors, device, nodes, birth, tape


@torch.no_grad()
def evaluate_continuous_eager(genomes, *, seed, actors, device, visits=VISITS, mode="full"):
    """Evaluate the same BF16 policy directly, with one fresh graph per actor."""
    actors, device, _, birth, tape = _inputs(genomes, seed, actors, device, visits, mode)
    with torch.cuda.device(device):
        policy = _BF16Policy({key: value.to(device) for key, value in birth.items()}, actors, device)
        return _rollout(policy, _device_tape(tape, device), _receipt_table(device).bfloat16())


@torch.no_grad()
def evaluate_continuous(genomes, *, seed, actors, device, visits=VISITS, mode="full"):
    """Replay a shape-specific CUDA Graph with fresh birth and schedule inputs."""
    actors, device, nodes, birth, tape = _inputs(genomes, seed, actors, device, visits, mode)
    reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    key = (len(genomes), actors, nodes, tape.visits, reduction)
    with _CACHE_LOCK, torch.cuda.device(device):
        caller_stream = torch.cuda.current_stream(device)
        if torch.cuda.is_current_stream_capturing():
            raise CUDAGraphUnavailable("uncapped BF16 CUDA Graph replay cannot nest in an active capture")
        cache = _CACHE.setdefault(device.index, OrderedDict())
        captured = cache.get(key)
        if captured is None:
            if len(cache) >= _SHAPES_PER_DEVICE:
                _, retired = cache.popitem(last=False)
                retired.stream.synchronize()
                del retired
            captured = _BF16Capture(birth, tape, actors, device)
            cache[key] = captured
        cache.move_to_end(key)
        return captured.evaluate(birth, tape, caller_stream)
