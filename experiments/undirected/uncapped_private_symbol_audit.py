"""Paired held-out diagnostics for the private symbol experiment."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path
from statistics import fmean

import numpy as np
import torch

from experiments.seed_evolution.large_runner import _write
from .evolved_plasticity_run import read, unpack_scores
from .evolved_plasticity_world import CONTEXT, FIRST_FINISH, N_ACTIONS, N_SYMBOLS, NEW, RAW, READY
from .uncapped_directed_audit import _save_champions
from .uncapped_directed_reward import _DirectedRewardPolicy
from .uncapped_island_crossover_run import DELAYED, block_ticks
from . import uncapped_private_symbol_world as world


SCHEMA = 1
PERIODIC_SEED_START = 96_600_000
FINAL_SEED_START = 96_700_000
FORK_REQUEST = 24
_STATE_FIELDS = (
    "h", "activity", "w", "g", "mask", "eligibility", "node_trace", "last_change",
    "pending", "tried", "reward", "edge_count", "edge_ticks", "births", "deaths",
    "weight_updates", "reward_events", "attempted_writes", "weight_change",
)


class _ShuffledRewardPolicy(_DirectedRewardPolicy):
    """Permute real nonzero reward signs; leave event positions/receipts intact.

    Each candidate's simultaneous actual ACK/REJ events are sorted by actor
    and cyclically shifted by one event. Homogeneous signs and single events
    cannot change; diagnostics report the number of signs actually changed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sign_control_events = torch.zeros_like(self.reward_events)
        self.sign_control_changed_events = torch.zeros_like(self.reward_events)

    def reset(self, birth):
        super().reset(birth)
        if hasattr(self, "sign_control_events"):
            self.sign_control_events.zero_()
            self.sign_control_changed_events.zero_()

    def _change_weights(self, coactivity, event):
        actual = self.reward.clone()
        index = torch.arange(self.actors, device=self.device).expand_as(event)
        order = torch.where(event, index, index + self.actors).argsort(dim=-1)
        rank = event.long().cumsum(-1) - 1
        source_rank = (rank + 1) % event.sum(-1, keepdim=True).clamp_min(1)
        source = order.gather(-1, source_rank)
        shuffled = actual.gather(-1, source) * event
        self.sign_control_events.add_(event)
        self.sign_control_changed_events.add_(event & shuffled.ne(actual))
        self.reward.copy_(shuffled)
        try:
            super()._change_weights(coactivity, event)
        finally:
            self.reward.copy_(actual)

    def summary(self):
        return {**super().summary(), "sign_control_events": self.sign_control_events.sum(-1),
                "sign_control_changed_events": self.sign_control_changed_events.sum(-1)}


def fork_schedule(*, seed, actors, device, visits, question_context):
    """Six identical visits followed by one A/B question on the same fixed raw."""
    if visits < 8 or question_context not in (0, 1):
        raise ValueError("fork diagnosis requires at least eight visits and context zero or one")
    tape = world.schedule(seed=seed, actors=actors, device=device, visits=visits)
    fields = {field: getattr(tape, field)[:FORK_REQUEST + 1].clone()
              for field in ("public", "stage", "target", "context", "raw")}
    raw = torch.arange(actors, device=device) % N_SYMBOLS
    port = fields["public"][FORK_REQUEST, ..., 0, :]
    issued = port[..., NEW].bool()
    port[..., RAW].zero_()
    port[..., CONTEXT].zero_()
    port[..., RAW] = torch.nn.functional.one_hot(raw, N_SYMBOLS).to(port.dtype)[None] * issued[..., None]
    port[..., CONTEXT.start + question_context] = issued
    fields["raw"][FORK_REQUEST] = raw
    fields["context"][FORK_REQUEST].fill_(question_context)
    fields["target"][FORK_REQUEST] = FIRST_FINISH + tape.maps[
        torch.arange(actors, device=device), question_context, raw,
    ]
    return replace(tape, **fields)


class _ForkStateObserver:
    def __init__(self, policy, boundary):
        self.policy, self.boundary, self.tick = policy, boundary, 0
        self.prefix_state = None

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def act(self, observation):
        if self.tick == self.boundary:
            self.prefix_state = {name: getattr(self.policy, name).clone()
                                 for name in _STATE_FIELDS if hasattr(self.policy, name)}
            # The world carries the previous action's receipt into the next
            # request. Include both ports' incoming receipts in the equality.
            self.prefix_state["incoming_receipt"] = observation.reshape(
                *observation.shape[:-1], 2, 28,
            )[..., 11:24].clone()
        self.tick += 1
        return self.policy.act(observation)


class _ForkRollout:
    def __init__(self):
        self.prefix_state = None

    def __call__(self, policy, tape, table):
        observer = _ForkStateObserver(policy, FORK_REQUEST * tape.public.shape[1])
        metrics, actions = world.observed_rollout(observer, tape, table)
        self.prefix_state = observer.prefix_state
        query = world.normalize_actions(actions[FORK_REQUEST])
        ready = tape.public[FORK_REQUEST, ..., 0, READY].bool() & tape.stage[FORK_REQUEST].eq(2)
        legal = query.ge(FIRST_FINISH) & query.lt(N_ACTIONS) & ready.unsqueeze(1)
        attempted, first_tick = legal.max(0)
        first = query.gather(0, first_tick.unsqueeze(0)).squeeze(0)
        for actor in range(query.shape[-1]):
            metrics[f"fork_answer_{actor}"] = torch.where(attempted[:, actor], first[:, actor], -1)
            metrics[f"fork_covered_{actor}"] = attempted[:, actor].float()
            metrics[f"fork_correct_{actor}"] = (
                attempted[:, actor] & first[:, actor].eq(tape.target[FORK_REQUEST, actor])
            ).float()
        return metrics


def _means(rows):
    return {key: fmean(row[key] for row in rows) for key in rows[0] if key != "seed"}


def _aggregate(islands):
    """Average islands on each shared world seed; islands are not replicates."""
    columns = list(islands.values())
    rows = [{"seed": aligned[0]["seed"], **_means(list(aligned))}
            for aligned in zip(*columns, strict=True)]
    return {"means": _means(rows), "paired_scores": rows}


def _delta(left, right, draws):
    if [row["seed"] for row in left] != [row["seed"] for row in right]:
        raise ValueError("paired private-symbol audit uses different held-out seeds")
    result = {}
    for key in left[0].keys() & right[0].keys() - {"seed"}:
        if key.startswith("fork_"):
            continue
        differences = np.asarray([a[key] - b[key] for a, b in zip(left, right, strict=True)])
        result[key] = {"mean": float(differences.mean()), "paired_seed_ci95": (
            None if draws is None else np.quantile(differences[draws].mean(1), [.025, .975]).tolist()
        )}
    return result


def _evaluate(evaluator, genomes, seeds, mode, actors, device, *, rollout=None):
    rows = {str(index): [] for index in range(len(genomes))}
    states = []
    try:
        for seed in seeds:
            fields = evaluator(genomes, seed=seed, actors=actors, device=device,
                               visits=8, mode=mode, capture=torch.device(device).type == "cuda")
            for island, row in zip(rows, unpack_scores(fields, len(genomes)), strict=True):
                if mode != "full" and row["changed_writes"] != 0:
                    raise ValueError(f"{mode} changed reward-driven weights")
                if mode == "frozen" and (row["births"] != 0 or row["deaths"] != 0):
                    raise ValueError("frozen mode changed support")
                rows[island].append({"seed": seed, **row})
            if rollout is not None:
                if not rollout.prefix_state:
                    raise ValueError("fork did not record its pre-question state")
                states.append({key: tensor.detach().cpu().clone()
                               for key, tensor in rollout.prefix_state.items()})
    finally:
        clear = getattr(evaluator, "clear_cache", None)
        if clear is not None:
            clear()
    return rows, states


def _fork_diagnosis(factory, genomes, seeds, actors, device):
    scores, snapshots = {}, {}
    for context, label in enumerate(("a", "b")):
        rollout = _ForkRollout()
        evaluator = factory(partial(fork_schedule, question_context=context), rollout=rollout)
        scores[label], snapshots[label] = _evaluate(
            evaluator, genomes, seeds, "full", actors, device, rollout=rollout,
        )
    equal = []
    for seed, left, right in zip(seeds, snapshots["a"], snapshots["b"], strict=True):
        same = left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)
        if not same:
            raise ValueError(f"A/B fork prefix states differ for held-out seed {seed}")
        equal.append({"seed": seed, "exact_state_equal": True, "state_fields": sorted(left)})
    reports = {}
    for island in scores["a"]:
        rows = []
        for a, b in zip(scores["a"][island], scores["b"][island], strict=True):
            both = [a[f"fork_covered_{i}"] and b[f"fork_covered_{i}"] for i in range(actors)]
            switched = [both[i] and a[f"fork_answer_{i}"] != b[f"fork_answer_{i}"] for i in range(actors)]
            dual = [a[f"fork_correct_{i}"] and b[f"fork_correct_{i}"] for i in range(actors)]
            rows.append({"seed": a["seed"], "a_correct": fmean(a[f"fork_correct_{i}"] for i in range(actors)),
                         "b_correct": fmean(b[f"fork_correct_{i}"] for i in range(actors)),
                         "both_covered": fmean(both), "answer_switch": fmean(switched),
                         "dual_correct": fmean(dual)})
        reports[island] = {"means": _means(rows), "paired_scores": rows}
    return {"prefix_visits": 6, "question_request_zero_based": FORK_REQUEST,
            "raw_assignment": "actor index modulo four, identical in the A and B fork",
            "answer": "first legal FINISH before question feedback; missing answer is incorrect",
            "exact_prefix_checks": equal, "islands": reports}


def audit(out, generation, cfg, checkpoint, evaluator_factory, *, rank, publish=None):
    """Select original-rank island winners and measure shared fresh holdout worlds."""
    out = Path(out)
    final = generation == cfg["generations"]
    modes = ("full", "no_reward_write", "frozen") if final else ("full", "no_reward_write")
    count = int(cfg["audit_seeds"])
    if not 0 < count < FINAL_SEED_START - PERIODIC_SEED_START:
        raise ValueError("audit_seeds must be positive and keep final/periodic worlds disjoint")
    first = FINAL_SEED_START if final else PERIODIC_SEED_START
    seeds = list(range(first, first + count))
    winners = [max(pool, key=rank) for pool in checkpoint["islands"]]
    genomes = [checkpoint["genomes"][row["id"]] for row in winners]
    best = max(winners, key=rank)
    actors, device = cfg["audit_actors"], cfg["device"]
    identity = {"schema": SCHEMA, "generation": generation, "final": final, "seeds": seeds,
                "actors": actors, "visits": 8, "modes_enabled": list(modes),
                "candidate": best["id"], "champions": {str(i): row["id"] for i, row in enumerate(winners)},
                "training_seed": cfg["seed"], "allow_downhill": cfg.get("allow_downhill"),
                "experiment_kind": cfg.get("experiment_kind")}
    path = out / "audits" / f"{generation:04d}.json"
    if path.is_file():
        result = read(path)
        if any(result.get(key) != value for key, value in identity.items()):
            raise ValueError(f"private symbol audit identity differs: {path}")
        if final:
            _save_champions(out, winners, best, checkpoint)
        return result
    draws = None if count < 2 else np.random.default_rng(0).integers(count, size=(4096, count))
    conditions = {"regular_native": (world.schedule, world.TIMING, block_ticks(world.TIMING))}
    if final:
        conditions["regular_padded"] = (partial(world.schedule, pad_to=block_ticks(DELAYED)),
                                         world.TIMING, block_ticks(DELAYED))
        conditions["long"] = (partial(world.schedule, timing=DELAYED), DELAYED, block_ticks(DELAYED))
    reports = {}
    for condition, (make_tape, timing, ticks) in conditions.items():
        scores = {}
        for mode in modes:
            if publish:
                publish(phase="holdout", audit_condition=condition, holdout_mode=mode)
            scores[mode], _ = _evaluate(evaluator_factory(make_tape), genomes, seeds, mode, actors, device)
        reports[condition] = {
            "timing": {**timing, "ticks_per_request": ticks},
            "modes": {mode: {"aggregate": _aggregate(scores[mode]),
                              "islands": {island: {"means": _means(rows), "paired_scores": rows}
                                          for island, rows in scores[mode].items()}} for mode in modes},
            "full_minus_ablation": {mode: {island: _delta(scores["full"][island], scores[mode][island], draws)
                                            for island in scores["full"]} for mode in modes[1:]},
        }
    result = {**identity, "conditions": reports,
              "receipts": "True action-dependent public ACK/REJ receipts in every condition and control.",
              "intervals": "Paired percentile bootstrap over held-out world seeds for fixed champions.",
              "champion_rows": winners}
    if final:
        reference, _ = _evaluate(evaluator_factory(world.schedule, policy_type=_DirectedRewardPolicy),
                                 genomes, seeds, "full", actors, device)
        shuffled, _ = _evaluate(evaluator_factory(world.schedule, policy_type=_ShuffledRewardPolicy),
                                genomes, seeds, "full", actors, device)
        result["sign_control"] = {
            "method": _ShuffledRewardPolicy.__doc__.strip(),
            "reference_true_reward": {
                "aggregate": _aggregate(reference),
                "islands": {island: {"means": _means(rows), "paired_scores": rows}
                            for island, rows in reference.items()},
            },
            "aggregate": _aggregate(shuffled),
            "islands": {island: {"means": _means(rows), "paired_scores": rows}
                        for island, rows in shuffled.items()},
            "full_minus_shuffled": {island: _delta(
                reference[island], rows, draws,
            ) for island, rows in shuffled.items()},
            "optimized_full_minus_reference": {island: _delta(
                reports["regular_native"]["modes"]["full"]["islands"][island]["paired_scores"],
                rows, draws,
            ) for island, rows in reference.items()},
        }
        result["same_state_context_fork"] = _fork_diagnosis(evaluator_factory, genomes, seeds, actors, device)
        result["timing_deltas"] = {
            f"{left}_minus_{right}": {mode: {island: _delta(
                reports[left]["modes"][mode]["islands"][island]["paired_scores"],
                reports[right]["modes"][mode]["islands"][island]["paired_scores"], draws,
            ) for island in shuffled} for mode in modes}
            for left, right in (("long", "regular_padded"), ("regular_padded", "regular_native"))
        }
        _save_champions(out, winners, best, checkpoint)
    _write(path, result)
    return result
