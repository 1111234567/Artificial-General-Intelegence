"""Private A/B symbol maps with READY timing and no public version input.

The two FINISH version halves name the same four symbol answers. Legality,
receipts, early actions and spare-port behavior retain the sealed world's
rules; only primary FINISH acceptance ignores the emitted version half.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import inspect
from pathlib import Path
from types import FunctionType

import torch

from . import uncapped_directed_world_fused as fused_world
from .evolved_plasticity_world import FIRST_FINISH, N_ACTIONS, N_SYMBOLS, READY, VERSION
from .plasticity_continuous_world import _ratio, rollout_continuous
from .uncapped_answer_feedback import _RecordPrimaryActions, _answer_metrics
from .uncapped_directed_reward_run import TIMING, WORLD_BLOCKS
from .uncapped_ready_signal_run import schedule as _versioned_schedule
from .uncapped_reward_precision import _first_finish_metrics


def schedule(*, seed, actors, device, visits, timing=TIMING, pad_to=None):
    """Keep fresh conflicted A/B maps, raw symbols and context; erase VERSION.

    Regular timing is READY at age 8 through 12, a four-tick FINISH window,
    and 18 native ticks per request. Optional padding follows the prior
    READY schedule's expired, silent public-frame convention.
    """
    tape = _versioned_schedule(seed=seed, actors=actors, device=device, visits=visits,
                               timing=timing, pad_to=pad_to)
    tape.public[..., VERSION].zero_()
    return replace(tape, target=FIRST_FINISH + (tape.target - FIRST_FINISH) % N_SYMBOLS)


def training_schedule(*, seed, actors, device, visits):
    """Use the previous four independent, context-balanced world blocks."""
    per_block, rest = divmod(actors, WORLD_BLOCKS)
    if rest or per_block < 2 or per_block % 2:
        raise ValueError("training needs four equally sized, context-balanced actor blocks")
    tapes = [schedule(seed=seed * WORLD_BLOCKS + block, actors=per_block, device=device,
                      visits=visits) for block in range(WORLD_BLOCKS)]
    return replace(
        tapes[0],
        **{field: torch.cat([getattr(tape, field) for tape in tapes], dim=2)
           for field in ("public", "stage")},
        **{field: torch.cat([getattr(tape, field) for tape in tapes], dim=1)
           for field in ("target", "context", "raw")},
        maps=torch.cat([tape.maps for tape in tapes], dim=0),
    )


def _replace_once(source, old, new):
    """Fail on sealed-source drift instead of silently retaining old semantics."""
    if source.count(old) != 1:
        raise RuntimeError(f"private symbol world expected one sealed source fragment: {old!r}")
    return source.replace(old, new, 1)


def _kernel_source():
    """Derive the private action kernel from the current fused implementation.

    Raw actions still determine legality and receipt rows. Canonicalization
    happens after legality and only changes acceptance and the now-trivial
    version outcome; no per-tick Torch operation is added to the CUDA path.
    """
    source = Path(fused_world.__file__).with_suffix(".cu").read_text(encoding="utf-8")
    source = _replace_once(
        source,
        "const bool accepted = ready && act0 == targets[actor];",
        "const bool accepted = legal && (act0 - 3) % 4 == (targets[actor] - 3) % 4;",
    )
    return _replace_once(
        source,
        "version_right = version_right || code / 4 == target_code / 4;",
        "version_right = true;  // FINISH version is irrelevant in the symbol-only task.",
    )


@lru_cache(maxsize=None)
def _kernels(device):
    from ._cuda_nvrtc import compile_kernels

    return compile_kernels(_kernel_source(),
                           ("directed_world_observation", "directed_world_action"), device=device)


@lru_cache(maxsize=1)
def _reference_rollout():
    """Compile the same acceptance change into the sealed CPU reference."""
    original = rollout_continuous.__wrapped__
    source = inspect.getsource(original)
    source = _replace_once(source, "@torch.no_grad()\n", "")
    source = _replace_once(source, "accepted = ready & (act0 == target)",
                           "accepted = legal & ((act0 - FIRST_FINISH) % N_SYMBOLS == target_symbol)")
    source = _replace_once(
        source,
        "version_right |= first_finish & ((action_code // N_SYMBOLS) == target_version)",
        "version_right |= first_finish",
    )
    namespace = dict(original.__globals__)
    exec(compile(source, str(Path(__file__).resolve()) + ":reference", "exec"), namespace)
    return namespace[original.__name__]


def rollout_with_table(table):
    """Create the private world rollout and compile CUDA kernels before capture."""
    if table.device.type != "cuda":
        original = _reference_rollout()
        return FunctionType(original.__code__,
                            {**original.__globals__, "_receipt_table": lambda device: table},
                            original.__name__, original.__defaults__, original.__closure__)
    if table.dtype != torch.float16 or not table.is_contiguous():
        raise ValueError("private fused world requires a contiguous CUDA FP16 receipt table")
    if tuple(table.shape) != (2 * N_ACTIONS + 1, 13):
        raise ValueError("private fused world receipt table must have shape [23,13]")
    kernels = _kernels(table.device)

    def rollout(policy, tape, population):
        return fused_world._rollout_cuda(policy, tape, population, table, kernels)

    return rollout


def normalize_actions(actions):
    """Fold legal FINISH identities once, retaining WAIT/reserved/invalid codes."""
    finish = actions.ge(FIRST_FINISH) & actions.lt(N_ACTIONS)
    return torch.where(finish, FIRST_FINISH + (actions - FIRST_FINISH) % N_SYMBOLS, actions)


def _symbol_metrics(actions, tape):
    ready = tape.public[..., 0, READY].bool() & tape.stage.eq(2)
    legal = actions.ge(FIRST_FINISH) & actions.lt(N_ACTIONS) & ready.unsqueeze(2)
    attempted, first_tick = legal.max(dim=1)
    first_action = actions.gather(1, first_tick.unsqueeze(1)).squeeze(1)
    correct = attempted & first_action.eq(tape.target.unsqueeze(1))
    metrics = {}
    for label, portion in (
        ("first", slice(0, 2 * N_SYMBOLS)),
        ("first_return", slice(2 * N_SYMBOLS, 4 * N_SYMBOLS)),
        ("late", slice(4 * N_SYMBOLS, 8 * N_SYMBOLS)),
    ):
        for index, name in enumerate(("a", "b")):
            mask = tape.context[portion].eq(index).unsqueeze(1)
            metrics[f"symbol_{label}_{name}"] = _ratio(
                (correct[portion] & mask).sum((0, 2)), mask.sum(),
            )
        metrics[f"symbol_{label}_mean"] = (
            metrics[f"symbol_{label}_a"] + metrics[f"symbol_{label}_b"]
        ) / 2
    return metrics


@torch.no_grad()
def observed_rollout(policy, tape, table):
    """Return metrics and raw primary actions [request,tick,population,actor].

    The raw action recording preserves emitted version identities for final
    diagnostics. Its batched normalized copy supplies every answer metric.
    """
    observer = _RecordPrimaryActions(policy, tape, policy.population)
    world = rollout_with_table(table)(observer, tape, policy.population)
    normalized = normalize_actions(observer.actions)
    metrics = {**policy.summary(), **world, **_answer_metrics(normalized, tape),
               **_first_finish_metrics(normalized, tape), **_symbol_metrics(normalized, tape)}
    return metrics, observer.actions


@torch.no_grad()
def task_rollout(policy, tape, table):
    """Backend ``_rollout`` replacement for the enabled private symbol task."""
    return observed_rollout(policy, tape, table)[0]
