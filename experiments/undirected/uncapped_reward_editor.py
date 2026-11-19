"""Pilot with a signed legal-FINISH reward and two addressed weight editors."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from threading import Lock

import torch

from .evolved_plasticity import ACTIONS, INPUTS, JOBS, WEIGHT_LIMIT, WEIGHT_STEP
from .evolved_plasticity_world import FIRST_FINISH, N_ACTIONS, N_SYMBOLS, READY, RECEIPT, _int, _receipt_table
from .plasticity_continuous_world import VISITS
from .uncapped_answer_feedback import _RecordPrimaryActions, _answer_metrics
from .uncapped_bf16_backend import _BF16Policy, _device_tape
from .uncapped_cuda_backend import (
    CUDAGraphUnavailable, _Capture, _SHAPES_PER_DEVICE, _rollout_with_preloaded_receipts, _stack_genomes,
)
from .uncapped_island_crossover_run import TIMING
from .uncapped_ready_signal_run import schedule


REWARD_CHANNEL = 9
ACK, REJ = RECEIPT.stop - 2, RECEIPT.stop - 1
EDITOR_START = INPUTS + JOBS * ACTIONS  # h78, h79
EDIT_STEP = .02
_CACHE = {}
_CACHE_LOCK = Lock()


class _EditorPolicy(_BF16Policy):
    def __init__(self, birth, actors, device):
        shape = (birth["node_bias"].size(0), actors)
        self.edit_idx = birth["edit_edges"].unsqueeze(1).expand(*shape, 2)
        self.edit_enabled, self.reward_enabled = birth["edit_enabled"], birth["reward_enabled"]
        self.previous_legal = torch.empty(shape, dtype=torch.bool, device=device)
        self.reward_waiting = torch.empty_like(self.previous_legal)
        self.write_due = torch.empty_like(self.previous_legal)
        self.edit_count = torch.empty(shape, dtype=torch.long, device=device)
        self.edit_abs_delta = torch.empty(shape, dtype=torch.float32, device=device)
        self.edit_signed_delta = torch.empty_like(self.edit_abs_delta)
        super().__init__(birth, actors, device)

    def reset(self, birth):
        super().reset(birth)
        for buffer in (self.previous_legal, self.reward_waiting, self.write_due, self.edit_count,
                       self.edit_abs_delta, self.edit_signed_delta):
            buffer.zero_()

    def act(self, obs):
        # The sealed world supplies receipts and a READY flag that is masked off
        # after acceptance. Its tape leaves the old wait hint at zero.
        reward = (obs[..., ACK] - obs[..., REJ]) * self.previous_legal
        obs[..., REWARD_CHANNEL].copy_(reward * self.reward_enabled)
        action = super().act(obs)
        primary = action[..., 0]
        self.previous_legal.copy_(obs[..., READY].bool() & primary.ge(FIRST_FINISH) & primary.lt(N_ACTIONS))
        # At t+1 the legal action at t supplies reward to h9. At t+2 one
        # recurrent update can carry it to h78/79. At the start of t+3 those
        # outputs may change the weights used for that tick. The event timing
        # stays the same in reward-off; only the observed scalar is removed.
        self.write_due.copy_(self.reward_waiting)
        self.reward_waiting.copy_(reward.ne(0))
        return action

    def _change_weights(self, local_activity, receipt):
        coactivity, difference = self._pair_activity(local_activity)
        self.eligibility.lerp_(coactivity, .1).masked_fill_(~self.mask, 0.)
        drive = self._pair_drive(self.rule[:, 0], receipt, self.forward_w, coactivity,
                                 difference, self.eligibility)
        proposal = torch.add(self.w, drive.tanh_(), alpha=WEIGHT_STEP)
        proposal.clamp_(-WEIGHT_LIMIT, WEIGHT_LIMIT).masked_fill_(~self.mask, 0.)
        # Addressed edge weights are reserved for the editor in every arm.
        # The usual support rule still controls whether each pair is active.
        proposal.scatter_(-1, self.edit_idx, self.w.gather(-1, self.edit_idx))
        self.weight_updates.add_(proposal.ne(self.w).sum(-1))
        self.weight_change.add_((proposal - self.w).abs_().sum(-1))
        self.w = proposal

    def _update_matrix(self):
        if self.step >= 3:
            # Use the active set after the scheduled support update. Skip
            # inactive targets and count only changes that survived clamping.
            current = self.w.gather(-1, self.edit_idx)
            enabled = self.mask.gather(-1, self.edit_idx) & (self.write_due & self.edit_enabled).unsqueeze(-1)
            command = self.activity[..., EDITOR_START:EDITOR_START + 2].float()
            proposal = torch.add(current, command, alpha=EDIT_STEP).clamp_(-WEIGHT_LIMIT, WEIGHT_LIMIT)
            updated = torch.where(enabled, proposal, current)
            self.w.scatter_(-1, self.edit_idx, updated)
            delta = updated - current
            changed, magnitude = delta.ne(0).sum(-1), delta.abs().sum(-1)
            self.edit_count.add_(changed)
            self.edit_abs_delta.add_(magnitude)
            self.edit_signed_delta.add_(delta.sum(-1))
            self.weight_updates.add_(changed)
            self.weight_change.add_(magnitude)
        super()._update_matrix()

    def summary(self):
        return {
            **super().summary(),
            "edit_count": self.edit_count.float().mean(-1),
            "edit_abs_delta": self.edit_abs_delta.mean(-1),
            "edit_signed_delta": self.edit_signed_delta.mean(-1),
        }


def _return_metrics(actions, tape):
    # Exactly one request per repeat visit. Legal attempts after retirement
    # cannot be first, since retirement itself requires a legal FINISH.
    requests = slice(2 * N_SYMBOLS, None, N_SYMBOLS)
    actions = actions[requests]
    ready = tape.stage[requests].eq(2) & tape.public[requests, ..., 0, READY].bool()
    legal = actions.ge(FIRST_FINISH) & actions.lt(N_ACTIONS) & ready.unsqueeze(2)
    attempted, first_tick = legal.max(dim=1)
    first_action = actions.gather(1, first_tick.unsqueeze(1)).squeeze(1)
    correct = attempted & first_action.eq(tape.target[requests].unsqueeze(1))
    result = {"return_first_correct": correct.float().mean((0, 2))}
    for context, key in enumerate(("return_A_first_correct", "return_B_first_correct")):
        mask = tape.context[requests].eq(context)
        result[key] = (correct & mask.unsqueeze(1)).sum((0, 2)) / mask.sum().clamp_min(1)
    return result


def _rollout(policy, tape, table):
    observer = _RecordPrimaryActions(policy, tape, policy.population)
    metrics = _rollout_with_preloaded_receipts(table)(observer, tape, policy.population)
    return {**policy.summary(), **metrics, **_answer_metrics(observer.actions, tape),
            **_return_metrics(observer.actions, tape)}


class _EditorCapture(_Capture):
    def __init__(self, birth, tape, actors, device):
        self.stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(self.stream):
            self.birth = {key: tensor.to(device) for key, tensor in birth.items()}
            self.tape = _device_tape(tape, device)
            self.receipt_table = _receipt_table(device).bfloat16()
            self.policy = _EditorPolicy(self.birth, actors, device)
            _rollout(self.policy, self.tape, self.receipt_table)
        self.graph = torch.cuda.CUDAGraph()
        # The Python trace rebinds both attributes. Keep the original buffers
        # alive so the captured reset restores their contents on each replay.
        self.entry_w, self.entry_activity = self.policy.w, self.policy.activity
        try:
            with torch.cuda.graph(self.graph, stream=self.stream):
                self.policy.reset(self.birth)
                self.results = _rollout(self.policy, self.tape, self.receipt_table)
        except RuntimeError as error:
            raise CUDAGraphUnavailable(f"reward editor CUDA Graph capture failed: {error}") from error


def _birth_tensors(genomes, edit_enabled, reward_enabled):
    nodes, birth = _stack_genomes(genomes)
    if nodes < EDITOR_START + 2:
        raise ValueError(f"reward editor requires {EDITOR_START + 2} nodes")
    if not isinstance(edit_enabled, bool) or not isinstance(reward_enabled, bool):
        raise ValueError("edit_enabled and reward_enabled must be bools")
    indices = [genome["edit_edges"] for genome in genomes]
    if any(not isinstance(field, torch.Tensor) or field.device.type != "cpu" or field.shape != (2,)
           or field.dtype != torch.long for field in indices):
        raise ValueError("edit_edges must be CPU torch.long tensors of shape (2,)")
    indices = torch.stack(indices)
    if (indices[:, 0].eq(indices[:, 1]).any() or indices.lt(0).any()
            or indices.ge(birth["birth_mask"].shape[-1]).any()):
        raise ValueError("each genome must address two distinct pair indices in range")
    birth.update(edit_edges=indices, edit_enabled=torch.tensor(edit_enabled),
                 reward_enabled=torch.tensor(reward_enabled))
    return nodes, birth


@torch.no_grad()
def evaluate(genomes, *, seed, actors=16, device="cuda", visits=VISITS,
             edit_enabled=True, reward_enabled=True, capture=None):
    """Return per-genome device metrics on matched tapes and fresh actor state.

    Edit counts and deltas are per-actor lifetime means. Reward-off suppresses
    the scalar input while preserving the receipts and legal-event timing.
    The CUDA Graph for each shape accepts both switches as replay inputs.
    """
    actors = _int("actors", actors, 1)
    device = torch.device(device)
    if capture is None:
        capture = device.type == "cuda"
    if device.type == "cuda":
        if not torch.cuda.is_available() or torch.is_autocast_enabled("cuda"):
            raise CUDAGraphUnavailable("reward editor requires CUDA with autocast disabled")
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise CUDAGraphUnavailable(f"{device} lacks native CUDA BF16 support")
    elif device.type != "cpu" or capture or torch.is_autocast_enabled("cpu"):
        raise CUDAGraphUnavailable("reward editor requires native CUDA BF16 or eager CPU with autocast disabled")

    nodes, birth = _birth_tensors(genomes, edit_enabled, reward_enabled)
    tape = schedule(seed=seed, actors=actors, device="cpu", visits=visits, timing=TIMING)
    tape = replace(tape, public=tape.public.bfloat16())
    if not capture:
        policy = _EditorPolicy({key: tensor.to(device) for key, tensor in birth.items()}, actors, device)
        return _rollout(policy, _device_tape(tape, device), _receipt_table(device).bfloat16())

    reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    key = (len(genomes), nodes, tape.public.shape, reduction)
    with _CACHE_LOCK, torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise CUDAGraphUnavailable("reward editor CUDA Graph replay cannot nest in an active capture")
        caller_stream = torch.cuda.current_stream(device)
        cache = _CACHE.setdefault(device.index, OrderedDict())
        captured = cache.get(key)
        if captured is None:
            if len(cache) >= _SHAPES_PER_DEVICE:
                _, retired = cache.popitem(last=False)
                retired.stream.synchronize()
                del retired
            captured = _EditorCapture(birth, tape, actors, device)
            cache[key] = captured
        cache.move_to_end(key)
        return captured.evaluate(birth, tape, caller_stream)
