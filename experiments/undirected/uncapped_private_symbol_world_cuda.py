"""Record raw private-symbol actions in the CUDA world transition."""

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

import torch

from . import uncapped_directed_world_fused as fused
from . import uncapped_private_symbol_world as private
from .evolved_plasticity_world import N_ACTIONS
from .uncapped_answer_feedback import _answer_metrics
from .uncapped_private_symbol_delta import DELTA_FIELD, _mean_surviving_delta
from .uncapped_reward_precision import _first_finish_metrics


def _kernel_source():
    source = private._kernel_source()
    source = private._replace_once(
        source, "int action_stride0, int action_stride1, int action_stride2\n",
        "int action_stride0, int action_stride1, int action_stride2, long long* primary_trace\n",
    )
    source = private._replace_once(
        source, "const long long act0 = actions[action_base];",
        "const long long act0 = actions[action_base];\n"
        "    if (primary_trace != nullptr) primary_trace[item] = act0;",
    )
    return source + "\n" + Path(__file__).with_suffix(".cu").read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _kernels(device):
    from ._cuda_nvrtc import compile_kernels

    names = ("directed_world_observation", "directed_world_action", "private_packed_observation")
    return compile_kernels(_kernel_source(), names, device=device)


def _rollout_cuda(policy, tape, table, *, trace, packed_observation):
    if table.dtype != torch.float16 or not table.is_contiguous():
        raise ValueError("private fused world requires a contiguous CUDA FP16 receipt table")
    if tuple(table.shape) != (2 * N_ACTIONS + 1, 13):
        raise ValueError("private fused world receipt table must have shape [23,13]")

    kernels = _kernels(table.device)
    action = kernels["directed_world_action"]
    if trace is not None:
        frames = iter(trace.flatten(0, 1).unbind())

        def record(grid, block, *args):
            action(grid, block, *args, next(frames))
    else:
        null_trace = ctypes.c_void_p()

        def record(grid, block, *args):
            action(grid, block, *args, null_trace)

    observe = kernels["directed_world_observation"]
    if packed_observation and tape.public.data_ptr() % 4 == 0:
        packed = kernels["private_packed_observation"]
        threads = fused._THREADS
        grid = ((policy.population * tape.public.shape[2] * 32 + threads - 1) // threads,)

        def observe(_grid, block, *args):
            packed(grid, block, *args)

    dispatch = {"directed_world_observation": observe, "directed_world_action": record}
    return fused._rollout_cuda(policy, tape, policy.population, table, dispatch)


@torch.no_grad()
def observed_rollout(policy, tape, table, *, packed_observation=False):
    """Return sealed metric tensors and the original int64 primary-action trace."""
    if table.device.type != "cuda":
        return private.observed_rollout(policy, tape, table)
    requests, ticks, actors = tape.stage.shape
    raw = torch.empty((requests, ticks, policy.population, actors), dtype=torch.long,
                      device=tape.public.device)
    world = _rollout_cuda(policy, tape, table, trace=raw, packed_observation=packed_observation)
    normalized = private.normalize_actions(raw)
    metrics = {**policy.summary(), **world, **_answer_metrics(normalized, tape),
               **_first_finish_metrics(normalized, tape), **private._symbol_metrics(normalized, tape)}
    return metrics, raw


def task_rollout(policy, tape, table):
    return observed_rollout(policy, tape, table)[0]


def packed_task_rollout(policy, tape, table):
    return observed_rollout(policy, tape, table, packed_observation=True)[0]


@torch.no_grad()
def packed_delta_rollout(policy, tape, table):
    born_weight = policy.w[:, 0].to(dtype=torch.float32, copy=True)
    born_mask = policy.mask[:, 0].clone()
    if table.device.type == "cuda":
        _rollout_cuda(policy, tape, table, trace=None, packed_observation=True)
    else:
        private.rollout_with_table(table)(policy, tape, policy.population)
    return {DELTA_FIELD: _mean_surviving_delta(policy, born_weight, born_mask)}
