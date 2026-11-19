"""Opt-in reward editor with activity-selected addresses and ordinary plasticity."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from threading import Lock

import torch

from .evolved_plasticity import WEIGHT_LIMIT
from .evolved_plasticity_world import _int, _receipt_table
from .plasticity_continuous_world import VISITS
from .uncapped_bf16_backend import _BF16Policy, _device_tape
from .uncapped_cuda_backend import CUDAGraphUnavailable, _Capture, _SHAPES_PER_DEVICE, _stack_genomes
from .uncapped_reward_editor import EDITOR_START, EDIT_STEP, TIMING, _EditorPolicy, _rollout, schedule


SELECTOR_NODES = 8
PAIRS_PER_CONTROLLER = 4
_WRITE_DELAY = 3
_CACHE = {}
_CACHE_LOCK = Lock()


class _DynamicEditorPolicy(_EditorPolicy):
    # Keep the ordinary local rule for every edge, including potential targets.
    _change_weights = _BF16Policy._change_weights

    def __init__(self, birth, actors, device):
        shape = (birth["node_bias"].size(0), actors)
        nodes = birth["node_bias"].size(-1)
        self.edit_selector = birth["edit_selector"].unsqueeze(1)
        self.edit_enabled, self.reward_enabled = birth["edit_enabled"], birth["reward_enabled"]
        self.previous_legal = torch.empty(shape, dtype=torch.bool, device=device)
        self.reward_waiting = torch.empty_like(self.previous_legal)
        self.write_due = torch.empty_like(self.previous_legal)
        self.edit_count = torch.empty(shape, dtype=torch.long, device=device)
        self.edit_abs_delta = torch.empty(shape, dtype=torch.float32, device=device)
        self.edit_signed_delta = torch.empty_like(self.edit_abs_delta)
        self.edit_seen = torch.empty((*shape, nodes * (nodes - 1) // 2), dtype=torch.bool, device=device)

        history_shape = (_WRITE_DELAY, *shape, 2, SELECTOR_NODES)
        self.finish_nodes = torch.empty(history_shape, dtype=torch.long, device=device)
        self.finish_scores = torch.empty(history_shape, dtype=torch.float32, device=device)
        self.left, self.right = torch.triu_indices(SELECTOR_NODES, SELECTOR_NODES, offset=1, device=device)
        u, v = torch.triu_indices(nodes, nodes, offset=1, device=device)
        self.pair_index = torch.zeros((nodes, nodes), dtype=torch.long, device=device)
        ids = torch.arange(u.numel(), device=device)
        self.pair_index[u, v] = ids
        self.pair_index[v, u] = ids
        # Initialize the common BF16 state without the fixed editor's addresses.
        _BF16Policy.__init__(self, birth, actors, device)

    def reset(self, birth):
        super().reset(birth)
        self.edit_seen.zero_()
        # The first possible write occurs after all ring slots have been filled.

    def act(self, obs):
        action = super().act(obs)
        score = self.activity.float().unsqueeze(-2) * self.edit_selector
        scores, nodes = score.topk(SELECTOR_NODES, dim=-1, sorted=False)
        slot = (self.step - 1) % _WRITE_DELAY
        self.finish_nodes[slot].copy_(nodes)
        self.finish_scores[slot].copy_(scores)
        return action

    def _select_pairs(self):
        # The rewarded FINISH supplied the salience snapshot. The current
        # support mask reflects any births/retirements through the write tick.
        slot = self.step % _WRITE_DELAY
        nodes, scores = self.finish_nodes[slot], self.finish_scores[slot] * .5
        candidates = self.pair_index[nodes.index_select(-1, self.left), nodes.index_select(-1, self.right)]
        active = self.mask.unsqueeze(-2).expand(-1, -1, 2, -1).gather(-1, candidates)
        priority = scores.index_select(-1, self.left) + scores.index_select(-1, self.right)
        chosen = priority.masked_fill_(~active, -torch.inf).topk(PAIRS_PER_CONTROLLER, dim=-1, sorted=False).indices
        idx = candidates.gather(-1, chosen)
        enabled = active.gather(-1, chosen) & (self.write_due & self.edit_enabled)[..., None, None]
        return idx.unbind(-2), enabled.unbind(-2)

    def _write(self, indices, enabled, command):
        current = self.w.gather(-1, indices)
        proposal = torch.add(current, command, alpha=EDIT_STEP).clamp_(-WEIGHT_LIMIT, WEIGHT_LIMIT)
        updated = torch.where(enabled, proposal, current)
        self.w.scatter_(-1, indices, updated)
        delta = updated - current
        changed = delta.ne(0)
        self.edit_seen.scatter_(-1, indices, self.edit_seen.gather(-1, indices) | changed)
        return changed.sum(-1), delta.abs().sum(-1), delta.sum(-1)

    def _update_matrix(self):
        if self.step >= _WRITE_DELAY:
            (first, second), (enable_first, enable_second) = self._select_pairs()
            first_command, second_command = self.activity[..., EDITOR_START:EDITOR_START + 2].float().unbind(-1)
            # Two controllers selecting one pair issue a single combined
            # change before clamping. Track that as one changed edge.
            shared = (first.unsqueeze(-1).eq(second.unsqueeze(-2))
                      & enable_first.unsqueeze(-1) & enable_second.unsqueeze(-2))
            combined = first_command.unsqueeze(-1) + torch.where(
                shared.any(-1), second_command.unsqueeze(-1), 0.
            )
            first_n, first_magnitude, first_delta = self._write(first, enable_first, combined)
            second_n, second_magnitude, second_delta = self._write(
                second, enable_second & ~shared.any(-2), second_command.unsqueeze(-1)
            )
            changed, magnitude = first_n + second_n, first_magnitude + second_magnitude
            self.edit_count.add_(changed)
            self.edit_abs_delta.add_(magnitude)
            self.edit_signed_delta.add_(first_delta + second_delta)
            self.weight_updates.add_(changed)
            self.weight_change.add_(magnitude)
        _BF16Policy._update_matrix(self)

    def summary(self):
        return {**super().summary(), "edit_unique_edges": self.edit_seen.sum(-1).float().mean(-1)}


class _DynamicEditorCapture(_Capture):
    def __init__(self, birth, tape, actors, device):
        self.stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(self.stream):
            self.birth = {key: tensor.to(device) for key, tensor in birth.items()}
            self.tape = _device_tape(tape, device)
            self.receipt_table = _receipt_table(device).bfloat16()
            self.policy = _DynamicEditorPolicy(self.birth, actors, device)
            _rollout(self.policy, self.tape, self.receipt_table)

        self.graph = torch.cuda.CUDAGraph()
        self.entry_w, self.entry_activity = self.policy.w, self.policy.activity
        try:
            with torch.cuda.graph(self.graph, stream=self.stream):
                self.policy.reset(self.birth)
                self.results = _rollout(self.policy, self.tape, self.receipt_table)
        except RuntimeError as error:
            raise CUDAGraphUnavailable(f"dynamic reward editor CUDA Graph capture failed: {error}") from error


def _birth_tensors(genomes, edit_enabled, reward_enabled):
    nodes, birth = _stack_genomes(genomes)
    if nodes < EDITOR_START + 2:
        raise ValueError(f"dynamic reward editor requires at least {EDITOR_START + 2} nodes")
    if not isinstance(edit_enabled, bool) or not isinstance(reward_enabled, bool):
        raise ValueError("edit_enabled and reward_enabled must be bools")
    selectors = [genome.get("edit_selector") for genome in genomes]
    if any(not isinstance(field, torch.Tensor) or field.device.type != "cpu" or field.shape != (2, nodes)
           or field.dtype != torch.float32 or not field.isfinite().all() for field in selectors):
        raise ValueError(f"edit_selector must be a finite CPU torch.float32 tensor of shape (2, {nodes})")
    birth.update(edit_selector=torch.stack(selectors), edit_enabled=torch.tensor(edit_enabled),
                 reward_enabled=torch.tensor(reward_enabled))
    return nodes, birth


@torch.no_grad()
def evaluate(genomes, *, seed, actors=16, device="cuda", visits=VISITS,
             edit_enabled=True, reward_enabled=True, capture=None):
    """Return matched per-genome metrics, including unique edited edges per life.

    Disabling reward removes the scalar input and preserves legal-event timing.
    The graph for each shape takes both switches as replay inputs.
    """
    actors = _int("actors", actors, 1)
    device = torch.device(device)
    if capture is None:
        capture = device.type == "cuda"
    if device.type == "cuda":
        if not torch.cuda.is_available() or torch.is_autocast_enabled("cuda"):
            raise CUDAGraphUnavailable("dynamic reward editor requires CUDA with autocast disabled")
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise CUDAGraphUnavailable(f"{device} lacks native CUDA BF16 support")
    elif device.type != "cpu" or capture or torch.is_autocast_enabled("cpu"):
        raise CUDAGraphUnavailable("dynamic reward editor requires native CUDA BF16 or eager CPU with autocast disabled")

    nodes, birth = _birth_tensors(genomes, edit_enabled, reward_enabled)
    tape = schedule(seed=seed, actors=actors, device="cpu", visits=visits, timing=TIMING)
    tape = replace(tape, public=tape.public.bfloat16())
    if not capture:
        policy = _DynamicEditorPolicy({key: tensor.to(device) for key, tensor in birth.items()}, actors, device)
        return _rollout(policy, _device_tape(tape, device), _receipt_table(device).bfloat16())

    reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    key = (len(genomes), nodes, tape.public.shape, reduction)
    with _CACHE_LOCK, torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise CUDAGraphUnavailable("dynamic reward editor CUDA Graph replay cannot nest in an active capture")
        caller_stream = torch.cuda.current_stream(device)
        cache = _CACHE.setdefault(device.index, OrderedDict())
        captured = cache.get(key)
        if captured is None:
            if len(cache) >= _SHAPES_PER_DEVICE:
                _, retired = cache.popitem(last=False)
                retired.stream.synchronize()
                del retired
            captured = _DynamicEditorCapture(birth, tape, actors, device)
            cache[key] = captured
        cache.move_to_end(key)
        return captured.evaluate(birth, tape, caller_stream)
