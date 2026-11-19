"""Small, paired closed-loop world for evolving graph plasticity.

Each actor receives A, then B, then the original A, without resetting its
controller or receipts. A/B are private independently drawn actor-level
permutations, conditioned to disagree on every raw symbol. The public NEW
pulse is the raw symbol, context, version and announced wait. Only the legal
window and version are visible later; successful FINISH retires the request.
Nothing observes a target or receives it through a teacher action.

The two 28-feature ports retain the old NEW/version/raw/action/receipt/context
positions. Feature 9 is the announced duration (ticks); feature 10 is a READY
bit in this curriculum. Only port zero issues contracts. BEGIN is reserved
for a later curriculum and rejected here. Replies to WAIT have no receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from numbers import Integral

import numpy as np
import torch


N_SYMBOLS = 4
N_VERSIONS = 2
N_ACTIONS = 3 + N_SYMBOLS * N_VERSIONS
WAIT = 0
FIRST_FINISH = 3
N_SLOTS = 2
FEATURES_PER_SLOT = 28
INPUT_SIZE = N_SLOTS * FEATURES_PER_SLOT
NEW, READY = 0, 10
VERSION = slice(3, 5)
RAW = slice(5, 9)
RECEIPT = slice(11, 24)
CONTEXT = slice(24, 28)
PHASES = ("a", "b", "return")

_PERMUTATIONS = np.asarray(tuple(permutations(range(N_SYMBOLS))), dtype=np.int64)
_DERANGEMENTS = _PERMUTATIONS[(_PERMUTATIONS != np.arange(N_SYMBOLS)).all(axis=1)]


def _int(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class WorldSchedule:
    """Candidate-independent event tape; only public frames enter the graph."""

    public: torch.Tensor  # [request, tick, actor, port, feature]
    stage: torch.Tensor  # [request, tick, actor], 0 before NEW, 1 waiting, 2 ready, 3 expired
    target: torch.Tensor  # [request, actor], private full FINISH code
    raw: torch.Tensor  # [request, actor], for diagnostic comparison only
    maps: torch.Tensor  # [actor, context, symbol], never fed to the graph
    requests_per_phase: int
    finish_window: int


def make_schedule(*, seed, actors, device, requests_per_phase=6, max_wait=3,
                  min_wait=1, finish_window=2):
    """Generate the same events for each genotype and for every paired arm.

    A given actor's stream is also unchanged by the number of other actors.
    The first four requests in each phase expose each raw symbol once. Later
    requests continue in independently shuffled blocks of four. Return scores
    can therefore isolate the first probe of each old symbol before feedback
    on that same symbol in the return phase (earlier probes can still inform
    the mapping by elimination). Each request has at least one idle tick
    after its finish window so idle actions and the final receipt are seen.
    """
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    actors = _int("actors", actors, 1)
    requests_per_phase = _int("requests_per_phase", requests_per_phase, 1)
    min_wait = _int("min_wait", min_wait, 1)
    max_wait = _int("max_wait", max_wait, min_wait)
    finish_window = _int("finish_window", finish_window, 1)
    requests = len(PHASES) * requests_per_phase
    steps = max_wait + finish_window + 2  # one tick of issue jitter, at least one trailing tick
    seed = int(seed) % (1 << 64)
    maps = np.empty((actors, 2, N_SYMBOLS), dtype=np.int64)
    raw = np.empty((requests, actors), dtype=np.int64)
    version = np.empty_like(raw)
    issue = np.empty_like(raw)
    wait = np.empty_like(raw)
    for actor in range(actors):
        rng = np.random.default_rng(np.random.SeedSequence([seed & 0xffffffff, seed >> 32, actor]))
        a = _PERMUTATIONS[rng.integers(len(_PERMUTATIONS))]
        maps[actor, 0] = a
        maps[actor, 1] = a[_DERANGEMENTS[rng.integers(len(_DERANGEMENTS))]]
        for phase in range(len(PHASES)):
            start = phase * requests_per_phase
            for block in range(0, requests_per_phase, N_SYMBOLS):
                count = min(N_SYMBOLS, requests_per_phase - block)
                raw[start + block:start + block + count, actor] = rng.permutation(N_SYMBOLS)[:count]
        version[:, actor] = rng.integers(N_VERSIONS, size=requests)
        issue[:, actor] = rng.integers(2, size=requests)
        wait[:, actor] = rng.integers(min_wait, max_wait + 1, size=requests)

    context = np.repeat(np.array([0, 1, 0]), requests_per_phase)
    target = FIRST_FINISH + N_SYMBOLS * version + maps[np.arange(actors), context[:, None], raw]
    age = np.arange(steps)[None, :, None] - issue[:, None, :]
    finish_age = age - wait[:, None, :]
    new = age == 0
    ready = (finish_age >= 0) & (finish_age < finish_window)
    stage = np.where(age < 0, 0, np.where(finish_age < 0, 1, np.where(ready, 2, 3)))
    public = np.zeros((requests, steps, actors, N_SLOTS, FEATURES_PER_SLOT), dtype=np.float32)
    port = public[..., 0, :]
    port[..., NEW] = new
    port[..., VERSION] = np.eye(N_VERSIONS, dtype=np.float32)[version][:, None] * (new | ready)[..., None]
    port[..., RAW] = np.eye(N_SYMBOLS, dtype=np.float32)[raw][:, None] * new[..., None]
    port[..., 9] = new * wait[:, None, :]
    port[..., READY] = ready
    port[..., CONTEXT] = np.eye(4, dtype=np.float32)[context][:, None, None] * new[..., None]

    def tensor(value):
        return torch.from_numpy(value).to(device=device)

    return WorldSchedule(tensor(public), tensor(stage.astype(np.uint8)), tensor(target), tensor(raw),
                         tensor(maps), requests_per_phase, finish_window)


def _receipt_table(device):
    # Rows 0..10: WAIT / rejected action; 11..21: accepted action; 22: invalid code.
    table = np.zeros((2 * N_ACTIONS + 1, 13), dtype=np.float32)
    attempted = np.arange(1, N_ACTIONS)
    table[attempted, attempted] = 1
    table[attempted, 12] = 1  # REJ at feature 23
    table[attempted + N_ACTIONS, attempted] = 1
    table[attempted + N_ACTIONS, 11] = 1  # ACK at feature 22
    table[-1, 12] = 1
    return torch.as_tensor(table, device=device)


@torch.no_grad()
def rollout(policy, schedule: WorldSchedule, population: int):
    """Run one uninterrupted life, returning one metric value per genotype.

    Completion and strict rates use eligible requests as the denominator.
    Strict completion requires actual acceptance and no wrong/premature action
    on that contract. Strict return is scored on each raw symbol's first probe;
    strict_return_all also includes subsequent exposures. Idle and error are
    per idle port-tick and per total port-tick respectively; early/late are
    FINISH counts per pre-window/tail opportunity on the active port. The
    error_on_attempt metric is the rejection fraction among non-WAIT actions.
    A blind legal first guess is correct with probability 1/4; a policy that
    guesses different symbols in each legal tick reaches up to min(window,4)/4
    completion in a single unlearned request. Strict fitness distinguishes those
    outcomes and uses the old-context probe in addition to normal completion.
    """
    population = _int("population", population, 1)
    requests, steps, actors, _, _ = schedule.public.shape
    device = schedule.public.device
    shape = (population, actors)
    stat = torch.zeros((10, *shape), dtype=torch.int32, device=device)
    (idle, idle_opportunities, rejected, attempted, early, late,
     legal_finish, ready_opportunities, wrong_ready, stale_ready) = stat.unbind(0)
    requests_complete, requests_strict = [], []
    receipt_idx = torch.zeros((*shape, N_SLOTS), dtype=torch.long, device=device)
    receipt_table = _receipt_table(device)
    for request_frames, request_stages, target in zip(
            schedule.public.unbind(0), schedule.stage.unbind(0), schedule.target.unbind(0)):
        complete = torch.zeros(shape, dtype=torch.bool, device=device)
        had_error = torch.zeros_like(complete)
        for frame, stage in zip(request_frames.unbind(0), request_stages.unbind(0)):
            obs = frame.expand(population, *frame.shape).clone()
            obs[..., 0, VERSION] *= (~complete).unsqueeze(-1)
            obs[..., 0, READY] *= ~complete
            obs[..., RECEIPT] = receipt_table[receipt_idx]
            action = policy.act(obs.flatten(-2))
            if tuple(action.shape) != (*shape, N_SLOTS) or action.dtype != torch.long:
                raise ValueError("policy.act must return torch.long [population,actor,2]")
            finish = (action[..., 0] >= FIRST_FINISH) & (action[..., 0] < N_ACTIONS)
            attempt0 = action[..., 0] != WAIT
            attempt1 = action[..., 1] != WAIT
            active = ((stage == 1) | (stage == 2)) & ~complete
            ready = (stage == 2) & ~complete
            accepted = ready & (action[..., 0] == target)
            rejected0 = attempt0 & ~accepted
            had_error |= active & rejected0
            complete |= accepted

            idle += (~active & attempt0).int() + attempt1.int()
            idle_opportunities += (~active).int() + 1  # the second port is always idle
            rejected += rejected0.int() + attempt1.int()
            attempted += attempt0.int() + attempt1.int()
            early += finish & (stage == 1)
            late += finish & (stage == 3) & ~complete
            legal_finish += finish & ready
            ready_opportunities += ready
            same_version = (action[..., 0] - FIRST_FINISH).bitwise_xor(target - FIRST_FINISH) < N_SYMBOLS
            wrong_ready += finish & ready & same_version & ~accepted
            stale_ready += finish & ready & ~same_version
            receipt_idx = torch.where((action >= 0) & (action < N_ACTIONS), action, 2 * N_ACTIONS)
            receipt_idx[..., 0] += accepted.long() * N_ACTIONS
        requests_complete.append(complete)
        requests_strict.append(complete & ~had_error)

    phase_len = schedule.requests_per_phase
    first_len = min(N_SYMBOLS, phase_len)
    complete = torch.stack(requests_complete).reshape(len(PHASES), phase_len, *shape).float()
    strict = torch.stack(requests_strict).reshape_as(complete).float()
    main = {
        "completion": complete.mean((0, 1, 3)),
        "strict": strict.mean((0, 1, 3)),
        "idle": idle.sum(1) / idle_opportunities.sum(1),
        "error": rejected.sum(1) / (N_SLOTS * requests * steps * actors),
        "error_on_attempt": rejected.sum(1) / attempted.sum(1).clamp_min(1),
        "early": early.sum(1) / (schedule.stage == 1).sum().clamp_min(1),
        "late": late.sum(1) / (schedule.stage == 3).sum().clamp_min(1),
        "legal_attempt": legal_finish.sum(1) / ready_opportunities.sum(1).clamp_min(1),
        "wrong_ready": wrong_ready.sum(1) / legal_finish.sum(1).clamp_min(1),
        "stale_ready": stale_ready.sum(1) / legal_finish.sum(1).clamp_min(1),
    }
    for phase, name in enumerate(PHASES):
        main[f"completion_{name}"] = complete[phase].mean((0, 2))
        main[f"strict_{name}"] = strict[phase].mean((0, 2))
        main[f"strict_{name}_first"] = strict[phase, :first_len].mean((0, 2))
        if phase_len > first_len:
            main[f"strict_{name}_later"] = strict[phase, first_len:].mean((0, 2))
    main["strict_return_all"] = main["strict_return"]
    main["strict_return"] = main["strict_return_first"]
    main["return_gain_over_first_a"] = main["strict_return"] - main["strict_a_first"]
    main["fitness"] = (.55 * main["completion"] + .30 * main["strict"]
                       + .15 * main["strict_return"] - .30 * main["idle"] - .25 * main["error"])
    return main


@torch.no_grad()
def evaluate(genomes, *, seed, actors, device, mode="full", requests_per_phase=6,
             max_wait=3, min_wait=1, finish_window=2):
    """Evaluate one population on paired worlds; metrics stay on the device."""
    from experiments.undirected.evolved_plasticity import GraphBatch

    schedule = make_schedule(seed=seed, actors=actors, device=device,
                             requests_per_phase=requests_per_phase, max_wait=max_wait,
                             min_wait=min_wait, finish_window=finish_window)
    graph = GraphBatch(genomes, actors=actors, device=device, mode=mode, seed=seed)
    metrics = rollout(graph, schedule, len(genomes))
    return {**graph.summary(), **metrics}
