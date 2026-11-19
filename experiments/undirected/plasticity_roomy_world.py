"""Extend the clock of the continuous A/B contract task."""

from __future__ import annotations

import numpy as np
import torch

from .evolved_plasticity_world import FEATURES_PER_SLOT, NEW, N_SLOTS, N_SYMBOLS, READY, VERSION, _int
from .plasticity_continuous_world import (
    FINISH_WINDOW as LEGACY_FINISH_WINDOW,
    MAX_WAIT as LEGACY_MAX_WAIT,
    MIN_WAIT as LEGACY_MIN_WAIT,
    VISITS,
    ContinuousSchedule,
    _FixedGuess,
    make_continuous_schedule,
    rollout_continuous,
)


MIN_WAIT = 8
MAX_WAIT = 16
FINISH_WINDOW = 8


def make_roomy_schedule(
    *, seed, actors, device, visits=VISITS,
    min_wait=MIN_WAIT, max_wait=MAX_WAIT, finish_window=FINISH_WINDOW,
):
    """Retain the original contracts and replace the wait and FINISH window.

    The wait hint on NEW linearly maps the configured delay range to [1, 3].
    A constant delay uses hint 2. The environment counts actual ticks.
    """
    min_wait = _int("min_wait", min_wait, 1)
    max_wait = _int("max_wait", max_wait, min_wait)
    finish_window = _int("finish_window", finish_window, 1)
    legacy_waits = (min_wait, max_wait) == (LEGACY_MIN_WAIT, LEGACY_MAX_WAIT)
    if legacy_waits and finish_window == LEGACY_FINISH_WINDOW:
        return make_continuous_schedule(seed=seed, actors=actors, device=device, visits=visits)

    original = make_continuous_schedule(seed=seed, actors=actors, device="cpu", visits=visits)
    source = original.public.numpy()[..., 0, :]
    requests, _, actors, _ = source.shape
    request_index = np.arange(requests)[:, None]
    actor_index = np.arange(actors)[None, :]
    issue = source[..., NEW].argmax(axis=1)
    new_features = source[request_index, issue, actor_index]

    if legacy_waits:
        wait = new_features[..., 9].astype(np.int64)
    else:
        seed = int(seed) % (1 << 64)
        low, high = seed & 0xffffffff, seed >> 32
        wait = np.empty_like(issue)
        for actor in range(actors):
            rng = np.random.default_rng(np.random.SeedSequence([low, high, actor, 2]))
            wait[:, actor] = rng.integers(min_wait, max_wait + 1, size=requests)

    ticks = max_wait + finish_window + 2
    age = np.arange(ticks)[None, :, None] - issue[:, None, :]
    finish_age = age - wait[:, None, :]
    ready = (finish_age >= 0) & (finish_age < finish_window)
    stage = np.where(age < 0, 0, np.where(finish_age < 0, 1, np.where(ready, 2, 3)))
    public = np.zeros((requests, ticks, actors, N_SLOTS, FEATURES_PER_SLOT), dtype=np.float32)
    port = public[..., 0, :]
    port[request_index, issue, actor_index] = new_features
    wait_hint = 2.0 if min_wait == max_wait else 1 + 2 * (wait - min_wait) / (max_wait - min_wait)
    port[request_index, issue, actor_index, 9] = wait_hint
    port[..., VERSION] += new_features[:, None, :, VERSION] * ready[..., None]
    port[..., READY] = ready

    return ContinuousSchedule(
        torch.from_numpy(public).to(device=device),
        torch.from_numpy(stage.astype(np.uint8)).to(device=device),
        original.target.to(device=device),
        original.context.to(device=device),
        original.raw.to(device=device),
        original.maps.to(device=device),
        original.visits,
    )


@torch.no_grad()
def evaluate_baseline(
    *, seed, actors, device, visits=VISITS, fixed_symbol=0,
    min_wait=MIN_WAIT, max_wait=MAX_WAIT, finish_window=FINISH_WINDOW,
):
    """Guess one fixed symbol per request using the public READY and version cues."""
    fixed_symbol = _int("fixed_symbol", fixed_symbol, 0)
    if fixed_symbol >= N_SYMBOLS:
        raise ValueError(f"fixed_symbol must be below {N_SYMBOLS}")
    schedule = make_roomy_schedule(
        seed=seed, actors=actors, device=device, visits=visits,
        min_wait=min_wait, max_wait=max_wait, finish_window=finish_window,
    )
    return rollout_continuous(_FixedGuess(fixed_symbol), schedule, 1)
