"""Train timing, version selection, and private symbol binding before A/B/A.

The graph always receives the native observation ports and its own ACK/REJ
receipts. Genotypes and plasticity modes share the same exogenous actor tape;
their actions can change only when a request retires and which receipt returns.
"""

from __future__ import annotations

import torch
from torch.nn.functional import one_hot

from .evolved_plasticity import GraphBatch
from .evolved_plasticity_world import (
    FIRST_FINISH, N_ACTIONS, N_SLOTS, N_SYMBOLS, N_VERSIONS, NEW, PHASES, RAW,
    READY, RECEIPT, VERSION, WAIT, WorldSchedule, _int, _receipt_table, make_schedule,
)


STAGES = ("timing", "version", "symbol", "conflict")
REQUESTS = {"timing": 6, "version": 6, "symbol": 16, "conflict": 6}


def make_phase_schedule(*, stage, seed, actors, device):
    """Return a paired tape, with one context until the original A/B/A world.

    Timing and version both ask for symbol zero in an identity mapping.
    Only version is variable in the version stage. Symbol repeats the raw and
    version observations throughout each active request: it gives the graph
    time to use the cue, never the intended target. Four independent shuffled
    blocks of raw symbols use the same private uniform permutation per actor.
    Conflict removes those repeated cues and uses the original schedule.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown curriculum stage {stage!r}")
    count = REQUESTS[stage]
    if stage == "conflict":
        return make_schedule(seed=seed, actors=actors, device=device, requests_per_phase=count)

    base = make_schedule(seed=seed, actors=actors, device="cpu", requests_per_phase=count)
    public = base.public[:count].clone()
    stages = base.stage[:count]
    target = base.target[:count]
    raw = base.raw[:count]
    maps = base.maps[:, :1]
    port = public[..., 0, :]
    version = (target - FIRST_FINISH) // N_SYMBOLS
    if stage == "symbol":
        active = ((stages == 1) | (stages == 2)).unsqueeze(-1)
        port[..., RAW] = one_hot(raw, N_SYMBOLS).unsqueeze(1) * active
        port[..., VERSION] = one_hot(version, N_VERSIONS).unsqueeze(1) * active
    else:
        raw = torch.zeros_like(raw)
        maps = torch.arange(N_SYMBOLS).expand(actors, 1, -1).clone()
        if stage == "timing":
            version = torch.zeros_like(version)
        target = FIRST_FINISH + N_SYMBOLS * version
        new = port[..., NEW].bool()
        port[..., RAW] = one_hot(raw, N_SYMBOLS).unsqueeze(1) * new.unsqueeze(-1)
        port[..., VERSION] = one_hot(version, N_VERSIONS).unsqueeze(1) * (new | (stages == 2)).unsqueeze(-1)
    return WorldSchedule(
        *(value.to(device=device) for value in (public, stages, target, raw, maps)),
        count, base.finish_window,
    )


def _ratio(numerator, denominator):
    return numerator / denominator.clamp_min(1)


@torch.no_grad()
def rollout_phase(policy, schedule, population, *, stage):
    """Run one life in parallel over candidates and actors.

    A legal FINISH lies in the live READY window and need not be correct.
    Coverage counts each contract once. Version and symbol accuracies use
    the first legal FINISH, comparing its two fields independently. Report
    both conditional accuracy and correct-per-contract coverage, since high
    conditional accuracy on rare attempts is not general capability. The
    cold probe is each actor's first contract; the first symbol block can
    already benefit from information gained on other symbols. Late symbol
    scores use the last block, which repeats exactly the same four raw symbols.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown curriculum stage {stage!r}")
    population = _int("population", population, 1)
    requests, steps, actors, _, _ = schedule.public.shape
    device = schedule.public.device
    shape = (population, actors)
    counts = torch.zeros((12, *shape), dtype=torch.int32, device=device)
    (idle, idle_opportunities, rejected, attempted, early, late, legal_finish,
     ready_opportunities, wrong_ready, stale_ready, off_ready, spare) = counts.unbind(0)
    receipt_idx = torch.zeros((*shape, N_SLOTS), dtype=torch.long, device=device)
    receipt_table = _receipt_table(device)

    outcomes = []
    for frames, stages, target in zip(schedule.public.unbind(0), schedule.stage.unbind(0),
                                      schedule.target.unbind(0), strict=True):
        complete = torch.zeros(shape, dtype=torch.bool, device=device)
        had_error = torch.zeros_like(complete)
        had_finish = torch.zeros_like(complete)
        version_right = torch.zeros_like(complete)
        symbol_right = torch.zeros_like(complete)
        timing_violation = torch.zeros_like(complete)
        target_code = target - FIRST_FINISH
        target_symbol = target_code % N_SYMBOLS
        target_version = target_code // N_SYMBOLS
        for frame, state in zip(frames.unbind(0), stages.unbind(0), strict=True):
            obs = frame.expand(population, *frame.shape).clone()
            obs[..., 0, VERSION] *= (~complete).unsqueeze(-1)
            obs[..., 0, READY] *= ~complete
            if stage == "symbol":
                obs[..., 0, RAW] *= (~complete).unsqueeze(-1)
            obs[..., RECEIPT] = receipt_table[receipt_idx]
            action = policy.act(obs.flatten(-2))
            if tuple(action.shape) != (*shape, N_SLOTS) or action.dtype != torch.long:
                raise ValueError("policy.act must return torch.long [population,actor,2]")
            act0, act1 = action.unbind(-1)
            attempt0, attempt1 = act0 != WAIT, act1 != WAIT
            valid_action = (action >= 0) & (action < N_ACTIONS)
            finish = (act0 >= FIRST_FINISH) & valid_action[..., 0]
            active = ((state == 1) | (state == 2)) & ~complete
            ready = (state == 2) & ~complete
            accepted = ready & (act0 == target)
            rejected0 = attempt0 & ~accepted
            legal = finish & ready
            first_finish = legal & ~had_finish
            act_code = act0 - FIRST_FINISH
            same_version = (act_code // N_SYMBOLS) == target_version
            version_right |= first_finish & same_version
            symbol_right |= first_finish & ((act_code % N_SYMBOLS) == target_symbol)
            had_finish |= legal

            premature_or_idle = ~ready & attempt0
            timing_violation |= premature_or_idle | attempt1
            had_error |= active & rejected0
            complete |= accepted
            idle += (~active & attempt0).int() + attempt1.int()
            idle_opportunities += (~active).int() + 1
            rejected += rejected0.int() + attempt1.int()
            attempted += attempt0.int() + attempt1.int()
            early += finish & (state == 1)
            late += finish & (state == 3) & ~complete
            legal_finish += legal
            ready_opportunities += ready
            wrong_ready += legal & same_version & ~accepted
            stale_ready += legal & ~same_version
            off_ready += premature_or_idle
            spare += attempt1
            receipt_idx = torch.where(valid_action, action, 2 * N_ACTIONS)
            receipt_idx[..., 0] += accepted.long() * N_ACTIONS
        outcomes.append(torch.stack((complete, complete & ~had_error, had_finish,
                                     had_finish & ~timing_violation, version_right, symbol_right)))

    complete, strict, covered, timing_clean, version_right, symbol_right = (
        torch.stack(outcomes, dim=1).float().unbind(0)
    )
    totals = counts.sum(dim=-1)
    (idle, idle_opportunities, rejected, attempted, early, late, legal_finish,
     ready_opportunities, wrong_ready, stale_ready, off_ready, spare) = totals.unbind(0)
    port_ticks = requests * steps * actors
    off_ready_rate = _ratio(off_ready, port_ticks - ready_opportunities)
    spare_rate = spare / port_ticks
    zero = torch.zeros_like(complete[0, :, 0])
    metric = {
        "completion": complete.mean((0, 2)),
        "strict": strict.mean((0, 2)),
        "idle": _ratio(idle, idle_opportunities),
        "error": rejected / (N_SLOTS * port_ticks),
        "error_on_attempt": _ratio(rejected, attempted),
        "early": _ratio(early, (schedule.stage == 1).sum()),
        "late": _ratio(late, (schedule.stage == 3).sum()),
        "legal_attempt": _ratio(legal_finish, ready_opportunities),
        "wrong_ready": _ratio(wrong_ready, legal_finish),
        "stale_ready": _ratio(stale_ready, legal_finish),
        "off_ready_rate": off_ready_rate,
        "spare_rate": spare_rate,
        "ready_coverage": covered.mean((0, 2)),
        "timing_clean": timing_clean.mean((0, 2)),
        "version_accuracy": _ratio(version_right.sum((0, 2)), covered.sum((0, 2))),
        "symbol_accuracy": _ratio(symbol_right.sum((0, 2)), covered.sum((0, 2))),
        "version_coverage": version_right.mean((0, 2)),
        "symbol_coverage": symbol_right.mean((0, 2)),
    }
    phase_len = schedule.requests_per_phase
    for i, name in enumerate(PHASES):
        start, end = i * phase_len, (i + 1) * phase_len
        if start >= requests:
            metric[f"completion_{name}"] = zero
            metric[f"strict_{name}"] = zero
            metric[f"strict_{name}_first"] = zero
            metric[f"strict_{name}_later"] = zero
        else:
            first_end = min(start + N_SYMBOLS, end)
            metric[f"completion_{name}"] = complete[start:end].mean((0, 2))
            metric[f"strict_{name}"] = strict[start:end].mean((0, 2))
            metric[f"strict_{name}_first"] = strict[start:first_end].mean((0, 2))
            metric[f"strict_{name}_later"] = strict[first_end:end].mean((0, 2)) if first_end < end else zero
    metric["strict_return_all"] = metric["strict_return"]
    metric["strict_return"] = metric["strict_return_first"]
    metric["return_gain_over_first_a"] = metric["strict_return"] - metric["strict_a_first"] if stage == "conflict" else zero

    for name, sample in (
        ("cold", slice(0, 1)),
        ("first", slice(0, N_SYMBOLS)),
        ("late", slice(-N_SYMBOLS, None)),
    ):
        if stage == "symbol":
            selected, hits = covered[sample], symbol_right[sample]
            metric[f"symbol_{name}"] = _ratio(hits.sum((0, 2)), selected.sum((0, 2)))
            metric[f"symbol_{name}_coverage"] = hits.mean((0, 2))
            metric[f"ready_coverage_{name}"] = selected.mean((0, 2))
        else:
            metric[f"symbol_{name}"] = metric[f"symbol_{name}_coverage"] = metric[f"ready_coverage_{name}"] = zero
    metric["symbol_gain"] = metric["symbol_late"] - metric["symbol_first"]

    # The two factors penalize acting when WAIT is required on either port.
    # Always waiting earns no reward; firing every tick on either port also
    # earns zero. A partially correct actor with some quiet ticks can improve
    # the score before it has learned the current stage's full task.
    coverage = metric["ready_coverage"]
    clean = metric["timing_clean"]
    version = metric["version_coverage"]
    completion = metric["completion"]
    if stage == "timing":
        reward = .60 * coverage + .25 * clean + .15 * completion
    elif stage == "version":
        reward = .25 * coverage + .10 * clean + .45 * version + .20 * completion
    elif stage == "symbol":
        reward = (.15 * coverage + .10 * version + .15 * metric["symbol_coverage"]
                  + .30 * metric["symbol_late_coverage"] + .15 * metric["strict"] + .15 * completion)
    else:
        reward = (.05 * coverage + .10 * version + .30 * completion + .25 * metric["strict"]
                  + .30 * metric["strict_return"])
    metric["fitness"] = reward * (1 - off_ready_rate) * (1 - spare_rate)
    return metric


@torch.no_grad()
def evaluate_phase(genomes, *, stage, seed, actors, device, mode="full"):
    """Evaluate a candidate batch in one curriculum stage on the chosen device."""
    schedule = make_phase_schedule(stage=stage, seed=seed, actors=actors, device=device)
    graph = GraphBatch(genomes, actors=actors, device=device, mode=mode, seed=seed)
    metrics = rollout_phase(graph, schedule, len(genomes), stage=stage)
    return {**graph.summary(), **metrics}
