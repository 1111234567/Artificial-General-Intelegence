"""A single mixed contract task with alternating visits to private A/B maps.

Each actor gets two conflicting permutations, retained for its whole life.
Every visit presents all four raw symbols in random order, with the same
version, wait and feedback rules. The first context is counterbalanced across
actors. Only public cues and the previous action's receipt reach the graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import torch

from .evolved_plasticity_world import (
    CONTEXT, FEATURES_PER_SLOT, FIRST_FINISH, N_ACTIONS, N_SLOTS, N_SYMBOLS,
    N_VERSIONS, NEW, RAW, READY, RECEIPT, VERSION, WAIT, _DERANGEMENTS,
    _PERMUTATIONS, _int, _receipt_table,
)


VISITS = 8
MIN_WAIT = 1
MAX_WAIT = 3
FINISH_WINDOW = 2
TICKS_PER_REQUEST = MAX_WAIT + FINISH_WINDOW + 2


@dataclass(frozen=True)
class ContinuousSchedule:
    public: torch.Tensor  # [request, tick, actor, port, feature]
    stage: torch.Tensor  # [request, tick, actor]; 0 idle, 1 waiting, 2 ready, 3 expired
    target: torch.Tensor  # [request, actor]; private action code
    context: torch.Tensor  # [request, actor]; context label for metrics
    raw: torch.Tensor  # [request, actor]; diagnostic labels
    maps: torch.Tensor  # [actor, context, raw]; private permutations
    visits: int


def make_continuous_schedule(*, seed, actors, device, visits=VISITS):
    """Pair all policies on the same tape; adding actors preserves existing ones."""
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    actors = _int("actors", actors, 1)
    visits = _int("visits", visits, 4)
    if visits % 2:
        raise ValueError("visits must be even")
    requests = visits * N_SYMBOLS
    seed = int(seed) % (1 << 64)
    low, high = seed & 0xffffffff, seed >> 32
    flip = np.random.default_rng(np.random.SeedSequence([low, high])).integers(2)
    by_visit = (np.arange(actors)[None, :] + np.arange(visits)[:, None] + flip) % 2
    context = by_visit.repeat(N_SYMBOLS, axis=0)
    maps = np.empty((actors, 2, N_SYMBOLS), dtype=np.int64)
    raw = np.empty((requests, actors), dtype=np.int64)
    version = np.empty_like(raw)
    issue = np.empty_like(raw)
    wait = np.empty_like(raw)
    for actor in range(actors):
        rng = np.random.default_rng(np.random.SeedSequence([low, high, actor, 1]))
        a = _PERMUTATIONS[rng.integers(len(_PERMUTATIONS))]
        maps[actor, 0] = a
        maps[actor, 1] = a[_DERANGEMENTS[rng.integers(len(_DERANGEMENTS))]]
        for start in range(0, requests, N_SYMBOLS):
            raw[start:start + N_SYMBOLS, actor] = rng.permutation(N_SYMBOLS)
        version[:, actor] = rng.integers(N_VERSIONS, size=requests)
        issue[:, actor] = rng.integers(2, size=requests)
        wait[:, actor] = rng.integers(MIN_WAIT, MAX_WAIT + 1, size=requests)

    target = FIRST_FINISH + N_SYMBOLS * version + maps[np.arange(actors), context, raw]
    age = np.arange(TICKS_PER_REQUEST)[None, :, None] - issue[:, None, :]
    finish_age = age - wait[:, None, :]
    new = age == 0
    ready = (finish_age >= 0) & (finish_age < FINISH_WINDOW)
    stage = np.where(age < 0, 0, np.where(finish_age < 0, 1, np.where(ready, 2, 3)))
    public = np.zeros((requests, TICKS_PER_REQUEST, actors, N_SLOTS, FEATURES_PER_SLOT),
                      dtype=np.float32)
    port = public[..., 0, :]
    port[..., NEW] = new
    port[..., VERSION] = np.eye(N_VERSIONS, dtype=np.float32)[version][:, None] * (new | ready)[..., None]
    port[..., RAW] = np.eye(N_SYMBOLS, dtype=np.float32)[raw][:, None] * new[..., None]
    port[..., 9] = new * wait[:, None, :]
    port[..., READY] = ready
    port[..., CONTEXT] = np.eye(4, dtype=np.float32)[context][:, None] * new[..., None]

    def tensor(value):
        return torch.from_numpy(value).to(device=device)

    return ContinuousSchedule(tensor(public), tensor(stage.astype(np.uint8)), tensor(target),
                              tensor(context), tensor(raw), tensor(maps), visits)


def _ratio(numerator, denominator):
    return numerator / denominator.clamp_min(1)


@torch.no_grad()
def rollout_continuous(policy, schedule, population):
    """Advance one actor state throughout the entire tape, including idle ticks.

    Strict means acceptance without an earlier rejected attempt while that
    request was live. Report idle/spare behavior separately. Return is scored
    on the first request of each symbol in every revisit (exactly one request
    per symbol per visit), including the initial returns as a separate slice.
    Accuracy uses the first legal FINISH; coverage divides by all contracts.
    """
    population = _int("population", population, 1)
    requests, ticks, actors, _, _ = schedule.public.shape
    device = schedule.public.device
    shape = (population, actors)
    counts = torch.zeros((8, *shape), dtype=torch.int32, device=device)
    (idle, idle_opportunities, rejected, attempted,
     off_ready, ready_opportunities, spare, legal_finish) = counts.unbind(0)
    receipt_idx = torch.zeros((*shape, N_SLOTS), dtype=torch.long, device=device)
    receipt_table = _receipt_table(device)
    outcomes = []

    for frames, stages, target in zip(schedule.public.unbind(), schedule.stage.unbind(),
                                      schedule.target.unbind(), strict=True):
        complete = torch.zeros(shape, dtype=torch.bool, device=device)
        had_error = torch.zeros_like(complete)
        covered = torch.zeros_like(complete)
        version_right = torch.zeros_like(complete)
        symbol_right = torch.zeros_like(complete)
        timing_violation = torch.zeros_like(complete)
        target_symbol = (target - FIRST_FINISH) % N_SYMBOLS
        target_version = (target - FIRST_FINISH) // N_SYMBOLS
        for frame, state in zip(frames.unbind(), stages.unbind(), strict=True):
            incomplete = ~complete
            obs = frame.expand(population, *frame.shape).clone()
            obs[..., 0, VERSION] *= incomplete.unsqueeze(-1)
            obs[..., 0, READY] *= incomplete
            obs[..., RECEIPT] = receipt_table[receipt_idx]
            action = policy.act(obs.flatten(-2))
            if tuple(action.shape) != (*shape, N_SLOTS) or action.dtype != torch.long:
                raise ValueError("policy.act must return torch.long [population,actor,2]")
            act0, act1 = action.unbind(-1)
            attempt0, attempt1 = act0 != WAIT, act1 != WAIT
            valid = (action >= 0) & (action < N_ACTIONS)
            active = ((state == 1) | (state == 2)) & incomplete
            ready = (state == 2) & incomplete
            legal = ready & (act0 >= FIRST_FINISH) & valid[..., 0]
            first_finish = legal & ~covered
            accepted = ready & (act0 == target)
            rejected0 = attempt0 & ~accepted
            mistimed = ~ready & attempt0
            action_code = act0 - FIRST_FINISH
            version_right |= first_finish & ((action_code // N_SYMBOLS) == target_version)
            symbol_right |= first_finish & ((action_code % N_SYMBOLS) == target_symbol)
            complete |= accepted
            had_error |= active & rejected0
            covered |= legal
            timing_violation |= mistimed | attempt1

            idle += (~active & attempt0).int() + attempt1.int()
            idle_opportunities += (~active).int() + 1
            rejected += rejected0.int() + attempt1.int()
            attempted += attempt0.int() + attempt1.int()
            off_ready += mistimed
            ready_opportunities += ready
            spare += attempt1
            legal_finish += legal
            receipt_idx = torch.where(valid, action, 2 * N_ACTIONS)
            receipt_idx[..., 0] += accepted.long() * N_ACTIONS
        outcomes.append(torch.stack((complete, complete & ~had_error, covered,
                                     covered & ~timing_violation, version_right, symbol_right)))

    complete, strict, covered, timing_clean, version_right, symbol_right = (
        torch.stack(outcomes, dim=1).float().unbind(0)
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


@torch.no_grad()
def evaluate_continuous(genomes, *, seed, actors, device, visits=VISITS, mode="full"):
    """Return device tensors with one summary value per candidate."""
    from .uncapped_plastic_graph import UncappedGraphBatch

    schedule = make_continuous_schedule(seed=seed, actors=actors, device=device, visits=visits)
    graph = UncappedGraphBatch(genomes, actors=actors, device=device, mode=mode, seed=seed)
    metrics = rollout_continuous(graph, schedule, len(genomes))
    return {**graph.summary(), **metrics}


class _FixedGuess:
    def __init__(self, symbol):
        self.symbol = symbol
        self.tried = None

    def act(self, obs):
        port = obs[..., :FEATURES_PER_SLOT]
        ready = port[..., READY].bool()
        if self.tried is None:
            self.tried = torch.zeros_like(ready)
        self.tried &= ~port[..., NEW].bool()
        guess = ready & ~self.tried
        self.tried |= guess
        finish = FIRST_FINISH + N_SYMBOLS * port[..., VERSION].argmax(-1) + self.symbol
        return torch.stack((torch.where(guess, finish, WAIT), torch.zeros_like(finish)), dim=-1)


@torch.no_grad()
def evaluate_baseline(*, seed, actors, device, visits=VISITS, fixed_symbol=0):
    """Perfect timing and version, one fixed symbol guess per request, no map."""
    fixed_symbol = _int("fixed_symbol", fixed_symbol, 0)
    if fixed_symbol >= N_SYMBOLS:
        raise ValueError(f"fixed_symbol must be below {N_SYMBOLS}")
    schedule = make_continuous_schedule(seed=seed, actors=actors, device=device, visits=visits)
    return rollout_continuous(_FixedGuess(fixed_symbol), schedule, 1)
