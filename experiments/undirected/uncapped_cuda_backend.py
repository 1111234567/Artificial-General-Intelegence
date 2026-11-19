"""Opt-in CUDA Graph evaluator for the sealed uncapped plasticity rollout.

The Python rollout builds one CUDA Graph per batch shape. Seeds and inherited
genomes change the values in its fixed-address input tensors, not its kernels.
The eager evaluator remains the reference and the default runner is unchanged.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from threading import Lock
from types import FunctionType

import torch

from .evolved_plasticity import ACTIONS, INPUTS, JOBS, RULE_FEATURES
from .evolved_plasticity_world import _int, _receipt_table
from .plasticity_continuous_world import VISITS, make_continuous_schedule, rollout_continuous
from .uncapped_plastic_graph import UncappedGraphBatch


class CUDAGraphUnavailable(RuntimeError):
    """The caller may deliberately choose the eager backend instead."""


_TAPE = ("public", "stage", "target", "context")
_ZERO_STATE = (
    "h", "activity", "matrix", "births", "deaths", "weight_updates",
    "weight_change", "g", "node_trace", "eligibility", "last_change", "edge_ticks",
)
_CACHE = {}
_CACHE_LOCK = Lock()
_SHAPES_PER_DEVICE = 2


def _supported_device(device, mode):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise CUDAGraphUnavailable("uncapped CUDA Graph replay requires a CUDA device")
    if mode != "full":
        raise ValueError("uncapped graph requires full plasticity")
    if (torch.get_float32_matmul_precision() != "highest"
            or torch.backends.cuda.matmul.allow_tf32 or torch.is_autocast_enabled("cuda")):
        raise CUDAGraphUnavailable("uncapped CUDA Graph replay requires full float32 precision")
    return torch.device(device.type, torch.cuda.current_device() if device.index is None else device.index)


def _stack_genomes(genomes):
    if not genomes:
        raise ValueError("uncapped graph requires at least one genome")
    nodes = genomes[0]["node_bias"].numel()
    if nodes < INPUTS + JOBS * ACTIONS:
        raise ValueError("graph must contain all fixed input and action ports")
    pairs = nodes * (nodes - 1) // 2
    keys = {
        "birth_w": ((pairs,), torch.float32),
        "birth_mask": ((pairs,), torch.bool),
        "node_bias": ((nodes,), torch.float32),
        "rule": ((2, len(RULE_FEATURES)), torch.float32),
    }
    stacked = {}
    for key, (shape, dtype) in keys.items():
        fields = [genome[key] for genome in genomes]
        if any(field.device.type != "cpu" or field.shape != shape or field.dtype != dtype
               for field in fields):
            raise CUDAGraphUnavailable(
                f"uncapped CUDA Graph replay requires CPU {key} tensors of shape {shape} and dtype {dtype}"
            )
        stacked[key] = torch.stack(fields)
    return nodes, stacked


def _rollout_with_preloaded_receipts(table):
    # The sealed rollout's only CPU->GPU transfer creates a constant receipt
    # table with torch.as_tensor. Supply the same table outside capture and run
    # the same bytecode, without changing its module globals or copying its math.
    rollout = rollout_continuous.__wrapped__
    return FunctionType(
        rollout.__code__, {**rollout.__globals__, "_receipt_table": lambda device: table},
        rollout.__name__, rollout.__defaults__, rollout.__closure__,
    )


class _Capture:
    def __init__(self, birth, tape, actors, device):
        self.stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(self.stream):
            self.birth = {key: tensor.to(device) for key, tensor in birth.items()}
            self.tape = replace(tape, **{key: getattr(tape, key).to(device) for key in _TAPE})
            self.receipt_table = _receipt_table(device)
            rollout = _rollout_with_preloaded_receipts(self.receipt_table)
            initial = [
                dict(zip(self.birth, row, strict=True))
                for row in zip(*(value.unbind() for value in self.birth.values()), strict=True)
            ]
            warm = UncappedGraphBatch(initial, actors=actors, device=device)
            rollout(warm, self.tape, len(initial))
            del warm
            self.policy = UncappedGraphBatch(initial, actors=actors, device=device)

        self.graph = torch.cuda.CUDAGraph()
        policy = self.policy
        # These buffers were allocated before capture and outlive the Python
        # reassignments of policy.w and policy.activity during tracing.
        self.entry_w, self.entry_activity = policy.w, policy.activity
        try:
            with torch.cuda.graph(self.graph, stream=self.stream):
                # These are the original w/activity allocations: act rebinds
                # both while Python is tracing. Replay runs the captured writes
                # against the original addresses and uses the same step/edge
                # span constants; Python does not advance steps during replay.
                policy.step = policy.edge_span_start = 0
                policy.bias.copy_(self.birth["node_bias"].unsqueeze(1))
                policy.rule.copy_(self.birth["rule"])
                policy.mask.copy_(self.birth["birth_mask"].unsqueeze(1))
                policy.w.copy_(self.birth["birth_w"].unsqueeze(1)).masked_fill_(~policy.mask, 0.)
                policy.initial_edges.copy_(self.birth["birth_mask"].sum(-1))
                policy.edge_count.copy_(policy.initial_edges.unsqueeze(1))
                for name in _ZERO_STATE:
                    getattr(policy, name).zero_()
                policy._update_matrix()
                metrics = rollout(policy, self.tape, len(initial))
                self.results = {**policy.summary(), **metrics}
        except RuntimeError as error:
            raise CUDAGraphUnavailable(f"uncapped CUDA Graph capture failed: {error}") from error

    def evaluate(self, birth, tape, caller_stream):
        with torch.cuda.stream(self.stream):
            for key, tensor in birth.items():
                self.birth[key].copy_(tensor)
            for key in _TAPE:
                getattr(self.tape, key).copy_(getattr(tape, key))
            self.graph.replay()
            # Return caller-owned snapshots. The next replay writes the same
            # captured output addresses even if earlier metrics are still live.
            results = {key: value.clone() for key, value in self.results.items()}
        caller_stream.wait_stream(self.stream)
        for value in results.values():
            value.record_stream(caller_stream)
        return results


@torch.no_grad()
def evaluate_continuous(genomes, *, seed, actors, device, visits=VISITS, mode="full"):
    """Return device tensors with one summary value per candidate.

    Supports the runner's full FP32 CUDA mode with CPU genomes. Unsupported
    configurations raise CUDAGraphUnavailable without an implicit eager run.
    """
    device = _supported_device(device, mode)
    actors = _int("actors", actors, 1)
    nodes, birth = _stack_genomes(genomes)
    tape = make_continuous_schedule(seed=seed, actors=actors, device="cpu", visits=visits)
    key = (len(genomes), actors, nodes, tape.visits)
    with _CACHE_LOCK, torch.cuda.device(device):
        caller_stream = torch.cuda.current_stream(device)
        if torch.cuda.is_current_stream_capturing():
            raise CUDAGraphUnavailable("uncapped CUDA Graph replay cannot nest in an active capture")
        cache = _CACHE.setdefault(device.index, OrderedDict())
        captured = cache.get(key)
        if captured is None:
            if len(cache) >= _SHAPES_PER_DEVICE:
                _, retired = cache.popitem(last=False)
                retired.stream.synchronize()
                del retired
            captured = _Capture(birth, tape, actors, device)
            cache[key] = captured
        cache.move_to_end(key)
        return captured.evaluate(birth, tape, caller_stream)
