"""Matched FP16, BF16 and FP32 probe with event-gated local weight updates."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import replace
from threading import Lock

import torch

from .evolved_plasticity import EDGE_COOLDOWN, INPUTS, WEIGHT_LIMIT
from .evolved_plasticity_world import (
    FEATURES_PER_SLOT, FIRST_FINISH, N_ACTIONS, N_SYMBOLS, NEW, READY, RECEIPT, _int, _receipt_table,
)
from .uncapped_answer_feedback import _RecordPrimaryActions
from .uncapped_bf16_backend import _BF16Policy, _device_tape
from .uncapped_cuda_backend import (
    CUDAGraphUnavailable, _Capture, _SHAPES_PER_DEVICE, _rollout_with_preloaded_receipts, _stack_genomes,
)
from .uncapped_plastic_graph import BIRTH_THRESHOLD, DEATH_THRESHOLD, GATE_RATE, UncappedGraphBatch


REWARD_RATE = .02
_ACK, _REJ = RECEIPT.stop - 2, RECEIPT.stop - 1
_CACHE = {}
_CACHE_LOCK = Lock()


class _PrecisionPolicy(_BF16Policy):
    # The inherited forward pass calls the weight and support steps before
    # injecting the new observation. There is no separate forward copy of W.
    _update_matrix = UncappedGraphBatch._update_matrix

    def __init__(self, birth, actors, device):
        self.mode, self.actors, self.population = "full", actors, birth["node_bias"].size(0)
        self.active_set_changes = True
        self.nodes = birth["node_bias"].size(-1)
        self.device = device
        self.u, self.v = torch.triu_indices(self.nodes, self.nodes, offset=1, device=device)
        self.pair_count = self.u.numel()
        batch = (self.population, actors)
        pairs = (*batch, self.pair_count)
        ports = (*batch, self.nodes)
        self.input_shape = (*batch, INPUTS)

        precision = dict(device=device, dtype=birth["birth_w"].dtype)
        self.bias = torch.empty((self.population, 1, self.nodes), **precision)
        self.rule = torch.empty_like(birth["rule"])
        self.w = torch.empty(pairs, **precision)
        self.g = torch.empty_like(self.w)
        self.eligibility = torch.empty_like(self.w)
        self.h = torch.empty(ports, **precision)
        self.activity = torch.empty_like(self.h)
        self.node_trace = torch.empty_like(self.h)
        self.matrix = torch.empty((*ports, self.nodes), **precision)
        self.reward = torch.empty(batch, **precision)
        self.birth_positive = torch.full((), .01, **precision)
        self.birth_negative = torch.full((), -.01, **precision)
        self.pending = torch.empty(batch, device=device, dtype=torch.bool)
        self.tried = torch.empty_like(self.pending)
        self.mask = torch.empty(pairs, device=device, dtype=torch.bool)
        self.last_change = torch.empty(pairs, device=device, dtype=torch.int32)
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
        super().reset(birth)
        for name in ("reward", "pending", "tried", "reward_events", "attempted_writes"):
            getattr(self, name).zero_()

    def act(self, obs):
        port = obs[..., :FEATURES_PER_SLOT]
        if self.step:
            # pending belongs to the previous observation. A NEW in this
            # observation rearms the next action but never discards its receipt.
            self.reward.copy_((port[..., _ACK] - port[..., _REJ]) * self.pending)
        self.tried.logical_and_(~port[..., NEW].bool())
        action = super().act(obs)
        primary = action[..., 0]
        self.pending.copy_(port[..., READY].bool() & ~self.tried
                           & primary.ge(FIRST_FINISH) & primary.lt(N_ACTIONS))
        self.tried.logical_or_(self.pending)
        return action

    def _change_weights(self, local_activity, receipt):
        coactivity = local_activity.index_select(-1, self.u) * local_activity.index_select(-1, self.v)
        self.eligibility.lerp_(coactivity, .1).masked_fill_(~self.mask, 0.)
        event = self.reward.ne(0)
        event_edges = event.unsqueeze(-1)
        self.reward_events.add_(event)
        # Count eligible deltas before any multiplication/addition rounding.
        self.attempted_writes.add_((self.eligibility.ne(0) & event_edges).sum(-1))
        proposal = torch.addcmul(self.w, self.eligibility, self.reward.unsqueeze(-1), value=REWARD_RATE)
        proposal.clamp_(-WEIGHT_LIMIT, WEIGHT_LIMIT)
        proposal = torch.where(event_edges, proposal, self.w)
        self.weight_updates.add_(proposal.ne(self.w).sum(-1))
        self.weight_change.add_((proposal - self.w).abs_().sum(-1, dtype=torch.float32))
        self.w.copy_(proposal)

    def _change_support(self, local_activity, receipt):
        self.edge_ticks.add_(self.edge_count, alpha=self.step - self.edge_span_start)
        self.edge_span_start = self.step

        coactivity, difference = self._pair_activity(local_activity)
        trace = self.node_trace.index_select(-1, self.u) * self.node_trace.index_select(-1, self.v)
        drive = self._pair_drive(self.rule[:, 1], receipt, self.w, coactivity, difference, trace)
        self.g.lerp_(drive.tanh_(), GATE_RATE)
        ready = self.last_change.le(self.step - EDGE_COOLDOWN)
        retire = ready & self.mask & self.g.le(DEATH_THRESHOLD)
        admit = ready & ~self.mask & self.g.ge(BIRTH_THRESHOLD)

        birth_drive = self._pair_drive(self.rule[:, 0], receipt, self.w, coactivity, difference, trace).tanh_()
        offset = torch.where(birth_drive.ge(0), self.birth_positive, self.birth_negative)
        initial_weight = .08 * birth_drive + offset
        self.w.copy_(torch.where(admit, initial_weight, self.w).masked_fill_(retire, 0.))
        self.eligibility.masked_fill_(retire, 0.)
        changed = admit | retire
        self.mask.logical_xor_(changed)
        self.last_change.masked_fill_(changed, self.step)
        admitted, retired = admit.sum(-1), retire.sum(-1)
        self.births.add_(admitted)
        self.deaths.add_(retired)
        self.edge_count.add_(admitted).sub_(retired)

    def summary(self):
        metrics = super().summary()
        return {**metrics, "reward_events": self.reward_events.sum(-1),
                "attempted_writes": self.attempted_writes.sum(-1),
                "changed_writes": metrics["weight_updates"]}


def _first_finish_metrics(actions, tape):
    ready = tape.public[..., 0, READY].bool() & tape.stage.eq(2)
    legal = actions.ge(FIRST_FINISH) & actions.lt(N_ACTIONS) & ready.unsqueeze(2)
    attempted, first_tick = legal.max(dim=1)
    first_action = actions.gather(1, first_tick.unsqueeze(1)).squeeze(1)
    correct = attempted & first_action.eq(tape.target.unsqueeze(1))
    # Each visit has one request per symbol, and each actor alternates maps.
    first_return = slice(2 * N_SYMBOLS, 4 * N_SYMBOLS)
    context = tape.context[first_return]
    metrics = {
        "first_finish_correct": correct.float().mean((0, 2)),
        "first_finish_correct_first_return": correct[first_return].float().mean((0, 2)),
        "ready_coverage_first_return": attempted[first_return].float().mean((0, 2)),
    }
    for index, suffix in enumerate(("a", "b")):
        mask = context.eq(index).unsqueeze(1)
        denominator = mask.sum().clamp_min(1)
        metrics[f"first_finish_correct_{suffix}_first_return"] = (correct[first_return] & mask).sum((0, 2)) / denominator
        metrics[f"ready_coverage_{suffix}_first_return"] = (attempted[first_return] & mask).sum((0, 2)) / denominator
    return metrics


def _rollout(policy, tape, table):
    observer = _RecordPrimaryActions(policy, tape, policy.population)
    metrics = _rollout_with_preloaded_receipts(table)(observer, tape, policy.population)
    return {**policy.summary(), **metrics, **_first_finish_metrics(observer.actions, tape)}


class _PrecisionCapture(_Capture):
    def __init__(self, birth, tape, actors, device):
        self.stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(self.stream):
            self.birth = {key: tensor.to(device) for key, tensor in birth.items()}
            self.tape = _device_tape(tape, device)
            self.receipt_table = _receipt_table(device).to(tape.public.dtype)
            self.policy = _PrecisionPolicy(self.birth, actors, device)
            _rollout(self.policy, self.tape, self.receipt_table)

        self.graph = torch.cuda.CUDAGraph()
        # The reset writes the entry allocation; act rebinds activity in tracing.
        self.entry_activity = self.policy.activity
        try:
            with torch.cuda.graph(self.graph, stream=self.stream):
                self.policy.reset(self.birth)
                self.results = _rollout(self.policy, self.tape, self.receipt_table)
        except RuntimeError as error:
            raise CUDAGraphUnavailable(f"reward precision CUDA Graph capture failed: {error}") from error


@contextmanager
def _matmul_precision(dtype, fp16_accumulation):
    matmul = torch.backends.cuda.matmul
    if dtype == torch.float32:
        previous = torch.get_float32_matmul_precision()
        specific = getattr(matmul, "fp32_precision", None)
        torch.set_float32_matmul_precision("highest")
        try:
            yield
        finally:
            torch.set_float32_matmul_precision(previous)
            if specific is not None and matmul.fp32_precision != specific:
                matmul.fp32_precision = specific
        return
    name = ("allow_fp16_reduced_precision_reduction" if dtype == torch.float16
            else "allow_bf16_reduced_precision_reduction")
    reduction = getattr(matmul, name)
    split_k = getattr(matmul, name + "_split_k", None)
    accumulation = getattr(matmul, "allow_fp16_accumulation", None) if dtype == torch.float16 else None
    try:
        setattr(matmul, name, False if split_k is None else (False, True))
        if accumulation is not None:
            matmul.allow_fp16_accumulation = fp16_accumulation
        yield
    finally:
        setattr(matmul, name, reduction if split_k is None else (reduction, split_k))
        if accumulation is not None:
            matmul.allow_fp16_accumulation = accumulation


class PrecisionEvaluation:
    """Run a fixed schedule with new actor state and one metric value per genome.

    Enable fp16_accumulation to use full FP16 GEMM accumulation on supported CUDA GPUs.
    """

    def __init__(self, schedule, *, fp16_accumulation=False):
        if type(fp16_accumulation) is not bool:
            raise TypeError("fp16_accumulation must be a bool")
        self.schedule = schedule
        self.fp16_accumulation = fp16_accumulation

    @torch.no_grad()
    def __call__(self, genomes, *, seed, actors, device, visits=4, dtype, capture=True):
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("reward precision probe requires torch.float16, torch.bfloat16 or torch.float32")
        device = torch.device(device)
        actors = _int("actors", actors, 1)
        if device.type not in ("cpu", "cuda") or torch.is_autocast_enabled(device.type):
            raise ValueError("reward precision probe requires CPU or CUDA with autocast disabled")
        if self.fp16_accumulation and (device.type != "cuda" or dtype != torch.float16):
            raise ValueError("fp16_accumulation requires CUDA with torch.float16")
        if capture and device.type != "cuda":
            raise CUDAGraphUnavailable("reward precision CUDA Graph replay requires CUDA")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise CUDAGraphUnavailable("reward precision probe requires CUDA availability")
        if self.fp16_accumulation and (
            torch.version.cuda is None
            or not hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation")
            or torch.cuda.get_device_capability(device) < (7, 0)
        ):
            raise CUDAGraphUnavailable(f"{device} lacks full FP16 CUDA accumulation support")

        nodes, birth = _stack_genomes(genomes)
        birth = {key: value.to(dtype) if value.is_floating_point() else value for key, value in birth.items()}
        tape = self.schedule(seed=seed, actors=actors, device="cpu", visits=visits)
        tape = replace(tape, public=tape.public.to(dtype))
        if device.type == "cpu":
            with _CACHE_LOCK:
                matmul = torch.backends.mkldnn.matmul
                previous = matmul.fp32_precision if dtype == torch.float32 else None
                if previous is not None:
                    matmul.fp32_precision = "ieee"
                try:
                    return _rollout(_PrecisionPolicy(birth, actors, device), tape, _receipt_table(device).to(dtype))
                finally:
                    if previous is not None:
                        matmul.fp32_precision = previous

        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        with _CACHE_LOCK, torch.cuda.device(device), _matmul_precision(dtype, self.fp16_accumulation):
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported(including_emulation=False):
                raise CUDAGraphUnavailable(f"{device} lacks native CUDA BF16 support")
            if not capture:
                policy = _PrecisionPolicy({key: tensor.to(device) for key, tensor in birth.items()}, actors, device)
                return _rollout(policy, _device_tape(tape, device), _receipt_table(device).to(dtype))

            if torch.cuda.is_current_stream_capturing():
                raise CUDAGraphUnavailable("reward precision CUDA Graph replay cannot nest in an active capture")
            caller_stream = torch.cuda.current_stream(device)
            cache = _CACHE.setdefault((device.index, dtype, self.fp16_accumulation), OrderedDict())
            key = (len(genomes), nodes, tape.public.shape)
            captured = cache.get(key)
            if captured is None:
                if len(cache) >= _SHAPES_PER_DEVICE:
                    _, retired = cache.popitem(last=False)
                    retired.stream.synchronize()
                    del retired
                captured = _PrecisionCapture(birth, tape, actors, device)
                cache[key] = captured
            cache.move_to_end(key)
            return captured.evaluate(birth, tape, caller_stream)
