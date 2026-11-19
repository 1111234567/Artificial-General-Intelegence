"""Exact half-precision continuous-world CUDA fusion for directed evaluation.

Only the environment observation and action bookkeeping are fused. Policy
execution and the sealed world's final Torch reductions retain their ordering.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from .evolved_plasticity_world import FEATURES_PER_SLOT, N_ACTIONS, N_SLOTS, N_SYMBOLS, _int
from .plasticity_continuous_world import _ratio
from .uncapped_cuda_backend import _rollout_with_preloaded_receipts


_THREADS = 128


@lru_cache(maxsize=None)
def _kernels(device):
    from ._cuda_nvrtc import compile_kernels

    source = Path(__file__).with_suffix(".cu").read_text(encoding="utf-8")
    return compile_kernels(source, ("directed_world_observation", "directed_world_action"),
                           device=device)


def rollout_with_table(receipt_table):
    """Build a ``(policy, schedule, population)`` rollout before graph capture.

    The fused path requires contiguous CUDA FP16 public frames and receipt
    table, with the schedule's original uint8 stages and int64 targets. CPU
    callers retain the sealed reference implementation for offline checks.
    Calling this factory in warm-up compiles both kernels before capture.
    """
    if receipt_table.device.type != "cuda":
        return _rollout_with_preloaded_receipts(receipt_table)
    if receipt_table.dtype != torch.float16 or not receipt_table.is_contiguous():
        raise ValueError("fused world requires a contiguous CUDA FP16 receipt table")
    if tuple(receipt_table.shape) != (2 * N_ACTIONS + 1, 13):
        raise ValueError("fused world receipt table must have shape [23,13]")
    kernels = _kernels(receipt_table.device)

    @torch.no_grad()
    def rollout(policy, schedule, population):
        return _rollout_cuda(policy, schedule, population, receipt_table, kernels)

    return rollout


def _rollout_cuda(policy, schedule, population, receipt_table, kernels):
    population = _int("population", population, 1)
    requests, ticks, actors, slots, features = schedule.public.shape
    device = schedule.public.device
    if (device != receipt_table.device or schedule.public.dtype != torch.float16
            or not schedule.public.is_contiguous()
            or (slots, features) != (N_SLOTS, FEATURES_PER_SLOT)):
        raise ValueError("fused world requires contiguous FP16 public [request,tick,actor,2,28]")
    if (schedule.stage.device != device or schedule.stage.dtype != torch.uint8
            or not schedule.stage.is_contiguous()
            or tuple(schedule.stage.shape) != (requests, ticks, actors)):
        raise ValueError("fused world requires contiguous uint8 stages [request,tick,actor]")
    if (schedule.target.device != device or schedule.target.dtype != torch.long
            or not schedule.target.is_contiguous()
            or tuple(schedule.target.shape) != (requests, actors)):
        raise ValueError("fused world requires contiguous int64 targets [request,actor]")
    shape = (population, actors)
    batch = population * actors
    counts = torch.zeros((8, *shape), dtype=torch.int32, device=device)
    receipt_idx = torch.zeros((*shape, N_SLOTS), dtype=torch.long, device=device)
    # Each request's first post-action kernel resets all six flags. Its first
    # pre-observation kernel treats complete as false without reading the old
    # flag, while receipts intentionally carry over from the previous request.
    flags = torch.empty((6, *shape), dtype=torch.bool, device=device)
    outcomes = torch.empty((6, requests, *shape), dtype=torch.bool, device=device)
    pre_grid = ((batch * N_SLOTS * FEATURES_PER_SLOT + _THREADS - 1) // _THREADS,)
    post_grid = ((batch + _THREADS - 1) // _THREADS,)
    block = (_THREADS,)
    for request, (frames, stages, target) in enumerate(zip(
            schedule.public.unbind(), schedule.stage.unbind(), schedule.target.unbind(), strict=True)):
        for tick, (frame, state) in enumerate(zip(frames.unbind(), stages.unbind(), strict=True)):
            obs = torch.empty((*shape, N_SLOTS, FEATURES_PER_SLOT), dtype=torch.float16, device=device)
            kernels["directed_world_observation"](
                pre_grid, block, frame, receipt_table, receipt_idx, flags, obs,
                actors, batch, int(tick == 0),
            )
            action = policy.act(obs.flatten(-2))
            if tuple(action.shape) != (*shape, N_SLOTS) or action.dtype != torch.long:
                raise ValueError("policy.act must return torch.long [population,actor,2]")
            if action.device != device:
                raise ValueError("policy.act must return actions on the observation device")
            kernels["directed_world_action"](
                post_grid, block, action, state, target, counts, receipt_idx, flags, outcomes,
                batch, actors, requests, request, int(tick == 0), int(tick == ticks - 1),
                *action.stride(),
            )

    complete, strict, covered, timing_clean, version_right, symbol_right = (
        outcomes.float().unbind(0)
    )
    (idle, idle_opportunities, rejected, attempted, off_ready, ready_opportunities,
     spare, legal_finish) = counts.sum(dim=-1).unbind(0)
    port_ticks = requests * ticks * actors
    covered_count = covered.sum((0, 2))
    metrics = {
        "completion": complete.mean((0, 2)),
        "strict": strict.mean((0, 2)),
        "idle": _ratio(idle, idle_opportunities),
        "error": rejected / (N_SLOTS * port_ticks),
        "error_on_attempt": _ratio(rejected, attempted),
        "off_ready_rate": _ratio(off_ready, port_ticks - ready_opportunities),
        "spare_rate": spare / port_ticks,
        "legal_attempt": _ratio(legal_finish, ready_opportunities),
        "ready_coverage": covered.mean((0, 2)),
        "timing_clean": timing_clean.mean((0, 2)),
        "version_accuracy": _ratio(version_right.sum((0, 2)), covered_count),
        "symbol_accuracy": _ratio(symbol_right.sum((0, 2)), covered_count),
        "version_coverage": version_right.mean((0, 2)),
        "symbol_coverage": symbol_right.mean((0, 2)),
    }
    first_end = 2 * N_SYMBOLS  # first visit to each map, regardless of actor's order

    def split(value, is_context):
        return _ratio((value * is_context.unsqueeze(1)).sum((0, 2)), is_context.sum())

    for context, name in enumerate(("a", "b")):
        mask = schedule.context == context
        metrics[f"completion_{name}"] = split(complete, mask)
        metrics[f"completion_{name}_return"] = split(complete[first_end:], mask[first_end:])
        metrics[f"strict_{name}"] = split(strict, mask)
        metrics[f"strict_{name}_first"] = split(strict[:first_end], mask[:first_end])
        metrics[f"strict_{name}_return"] = split(strict[first_end:], mask[first_end:])
        metrics[f"strict_gain_{name}"] = metrics[f"strict_{name}_return"] - metrics[f"strict_{name}_first"]

    for kind in ("completion", "strict"):
        metrics[f"{kind}_both"] = torch.minimum(metrics[f"{kind}_a"], metrics[f"{kind}_b"])
        metrics[f"{kind}_both_return"] = torch.minimum(
            metrics[f"{kind}_a_return"], metrics[f"{kind}_b_return"],
        )

    in_a = (schedule.context == 0).unsqueeze(1)
    returned = strict[first_end:].bool()
    in_a_returned = in_a[first_end:]
    successful_in_both = ((returned & in_a_returned).any(0) & (returned & ~in_a_returned).any(0))
    metrics["actors_strict_both_return"] = successful_in_both.float().mean(-1)

    for label, portion in (
        ("first", slice(first_end)),
        ("return", slice(first_end, None)),
        ("first_return", slice(first_end, 2 * first_end)),
        ("last", slice(-first_end, None)),
    ):
        metrics[f"completion_{label}"] = complete[portion].mean((0, 2))
        metrics[f"strict_{label}"] = strict[portion].mean((0, 2))
        metrics[f"symbol_{label}_coverage"] = symbol_right[portion].mean((0, 2))
    metrics["strict_gain"] = metrics["strict_return"] - metrics["strict_first"]
    metrics["completion_gain"] = metrics["completion_return"] - metrics["completion_first"]
    metrics["fitness"] = (.05 * metrics["ready_coverage"] + .10 * metrics["version_coverage"]
                          + .30 * metrics["completion"] + .25 * metrics["strict"]
                          + .30 * metrics["strict_return"]
                          ) * (1 - metrics["off_ready_rate"]) * (1 - metrics["spare_rate"])
    return metrics
