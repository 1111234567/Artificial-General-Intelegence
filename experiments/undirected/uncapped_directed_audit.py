"""Paired holdout measurements for the directed and reciprocal islands."""
from __future__ import annotations

from statistics import fmean

import numpy as np
import torch

from experiments.seed_evolution.large_runner import _save, _write
from .evolved_plasticity_run import read, unpack_scores
from .plasticity_continuous_world import _FixedGuess, rollout_continuous
from .uncapped_answer_feedback import _RecordPrimaryActions, _answer_metrics
from .uncapped_island_crossover_run import DELAYED, TIMING, block_ticks
from .uncapped_ready_signal_run import schedule
from .uncapped_reward_precision import _first_finish_metrics


MODES = ("full", "no_reward_write", "frozen")
CHANCE = .25
CONTRAST_FIELDS = (
    "first_finish_correct", "first_finish_correct_first_return",
    "first_finish_correct_a_first_return", "first_finish_correct_b_first_return",
    "strict", "strict_both_return", "completion", "ready_coverage",
    "ready_coverage_a_first_return", "ready_coverage_b_first_return",
    "premature_request_rate", "wrong_answer_rate", "mean_edges", "final_edges",
    "final_reciprocity", "births", "deaths", "reward_events", "changed_writes",
)


def _means(rows):
    keys = tuple(key for key in rows[0] if key != "seed")
    return {key: fmean(row[key] for row in rows) for key in keys}


def _paired_delta(left, right, draws):
    """Summarize paired world-seed differences for fixed final genomes."""
    if [row["seed"] for row in left] != [row["seed"] for row in right]:
        raise ValueError("paired audit rows use different world seeds")
    effects = {}
    for key in CONTRAST_FIELDS:
        if key in left[0] and key in right[0]:
            differences = np.array([a[key] - b[key] for a, b in zip(left, right, strict=True)])
            interval = None if draws is None else np.quantile(
                differences[draws].mean(axis=1), [.025, .975],
            ).tolist()
            effects[key] = {"mean": fmean(differences), "paired_seed_ci95": interval}
    return effects


def _fixed_scores(conditions, seeds, actors, visits):
    scores = {name: [] for name in conditions}
    long_ticks = block_ticks(DELAYED)
    for seed in seeds:
        tapes = {
            name: schedule(seed=seed, actors=actors, device="cpu", visits=visits,
                           timing=timing, pad_to=ticks)
            for name, (_, timing, ticks) in conditions.items()
        }
        if "long" in tapes:
            native, long = tapes["regular_native"], tapes["long"]
            contracts = ("maps", "raw", "context", "target")
            if any(not torch.equal(getattr(native, key), getattr(long, key)) for key in contracts):
                raise ValueError("audit timing conditions must use the same contracts")
            if "regular_padded" in tapes:
                padded = tapes["regular_padded"]
                ticks = native.public.shape[1]
                shift = DELAYED["min_wait"] - TIMING["min_wait"]
                if (padded.public.shape[1] != long_ticks or long.public.shape[1] != long_ticks
                        or not torch.equal(padded.public[:, :ticks], native.public)
                        or not torch.equal(padded.stage[:, :ticks], native.stage)
                        or not padded.stage[:, ticks:].eq(3).all()
                        or not torch.equal(padded.stage[:, :-shift].eq(2), long.stage[:, shift:].eq(2))):
                    raise ValueError("audit padding or READY shift differs from the specified clock")
        for name, tape in tapes.items():
            observer = _RecordPrimaryActions(_FixedGuess(0), tape, 1)
            values = {
                **rollout_continuous(observer, tape, 1),
                **_answer_metrics(observer.actions, tape),
                **_first_finish_metrics(observer.actions, tape),
            }
            row = {"seed": seed, **{key: value.item() for key, value in values.items()}}
            for key in ("completion", "strict", "first_finish_correct", "first_finish_correct_first_return",
                        "first_finish_correct_a_first_return", "first_finish_correct_b_first_return"):
                if row[key] != CHANCE:
                    raise ValueError(f"fixed guess {key} must solve one of four requests")
            scores[name].append(row)
    return scores


def _measure(evaluate, genomes, islands, seeds, mode, actors, visits, device):
    paired = {island: [] for island in islands}
    for seed in seeds:
        fields = evaluate(genomes, seed=seed, actors=actors, device=device, visits=visits,
                          mode=mode, capture=torch.device(device).type == "cuda")
        for island, row in zip(islands, unpack_scores(fields, len(genomes)), strict=True):
            if mode != "full" and row["changed_writes"] != 0:
                raise ValueError(f"audit mode {mode} changed a weight by reward")
            if mode == "frozen" and (row["births"] != 0 or row["deaths"] != 0):
                raise ValueError("frozen audit mode changed graph support")
            paired[island].append({"seed": seed, **row, "average_edges": row["mean_edges"]})
    return paired


def _save_champions(out, winners, best, checkpoint):
    genomes = {row["id"]: checkpoint["genomes"][row["id"]] for row in winners}
    _save(out / "champion.pt", {"row": best, "genome": genomes[best["id"]]})
    _save(out / "island_champions.pt", {"islands": [[row] for row in winners], "genomes": genomes})


def audit(config, out, generation, checkpoint, publish, *, evaluate, delayed_evaluate, rank,
          matched_regular_evaluate=None):
    """Evaluate each island's development winner on shared holdout worlds."""
    path = out / "audits" / f"{generation:04d}.json"
    final = generation == config["generations"]
    if path.is_file():
        result = read(path)
        if final and not all((out / name).is_file() for name in ("champion.pt", "island_champions.pt")):
            winners = [max(pool, key=rank) for pool in checkpoint["islands"]]
            _save_champions(out, winners, max(winners, key=rank), checkpoint)
        return result

    winners = [max(pool, key=rank) for pool in checkpoint["islands"]]
    best = max(winners, key=rank)
    genomes = [checkpoint["genomes"][row["id"]] for row in winners]
    islands = [str(index) for index in range(len(winners))]
    pairs = list(zip(islands[::2], islands[1::2], strict=True))
    best_island = islands[winners.index(best)]
    actors = config["audit_actors"]
    visits = 8 if final else 4
    first = config["seed"] + (40_000_003 if final else 30_000_019)
    seeds = list(range(first, first + config["audit_seeds"]))
    draws = None if len(seeds) < 2 else np.random.default_rng(0).integers(
        len(seeds), size=(4096, len(seeds)),
    )

    conditions = {"regular_native": (evaluate, TIMING, None)}
    if final:
        if matched_regular_evaluate is not None:
            conditions["regular_padded"] = (matched_regular_evaluate, TIMING, block_ticks(DELAYED))
        conditions["long"] = (delayed_evaluate, DELAYED, block_ticks(DELAYED))
    modes = MODES if final else MODES[:1]
    publish(phase="holdout", holdout_mode="fixed_guess", holdout_modes_completed=[])
    fixed = _fixed_scores(conditions, seeds, actors, visits)

    reports = {}
    for name, (evaluator, timing, pad_to) in conditions.items():
        mode_reports, scores = {}, {}
        try:
            for mode in modes:
                publish(phase="holdout", holdout_mode=mode, holdout_modes_completed=["fixed_guess"],
                        audit_condition=name)
                scores[mode] = _measure(evaluator, genomes, islands, seeds, mode, actors, visits,
                                        config["device"])
                island_reports = {island: {"means": _means(rows), "paired_scores": rows}
                                  for island, rows in scores[mode].items()}
                mode_reports[mode] = {
                    "islands": island_reports,
                    "directed_minus_reciprocal": {
                        f"{directed}_minus_{reciprocal}": _paired_delta(
                            scores[mode][directed], scores[mode][reciprocal], draws,
                        )
                        for directed, reciprocal in pairs
                    },
                }
                first_returns = ",".join(
                    f"{report['means']['first_finish_correct_first_return']:.4f}"
                    for report in island_reports.values()
                )
                print(f"audit generation={generation} condition={name} mode={mode} "
                      f"first_return={first_returns}", flush=True)
        finally:
            clear = getattr(evaluator, "clear_cache", None)
            if clear is not None:
                clear()
        reports[name] = {
            "timing": {**timing, "ticks_per_request": pad_to or block_ticks(timing)},
            "modes": mode_reports,
            "fixed_guess": {"means": _means(fixed[name]), "paired_scores": fixed[name]},
            "full_minus_fixed_guess": {
                island: _paired_delta(scores["full"][island], fixed[name], draws) for island in islands
            },
            "full_minus_ablation": {
                mode: {island: _paired_delta(scores["full"][island], scores[mode][island], draws)
                       for island in islands}
                for mode in modes[1:]
            },
        }

    timing_pairs = []
    if "regular_padded" in reports:
        timing_pairs = [("long", "regular_padded"), ("regular_padded", "regular_native")]
    elif "long" in reports:
        timing_pairs = [("long", "regular_native")]
    timing_deltas = {
        f"{left}_minus_{right}": {
            mode: {island: _paired_delta(reports[left]["modes"][mode]["islands"][island]["paired_scores"],
                                         reports[right]["modes"][mode]["islands"][island]["paired_scores"], draws)
                   for island in islands}
            for mode in modes
        }
        for left, right in timing_pairs
    }
    regular = reports["regular_native"]
    display = {"full": regular["modes"]["full"]["islands"][best_island], "fixed_guess": regular["fixed_guess"]}
    result = {
        "schema": 1, "experiment_kind": config["experiment_kind"], "generation": generation,
        "final": final, "candidate": best["id"], "actors": actors, "seeds": seeds, "visits": visits,
        "modes": {mode: report["means"] for mode, report in display.items()},
        "paired_scores": {mode: report["paired_scores"] for mode, report in display.items()},
        "champions": {
            island: {"id": row["id"], "topology": "directed" if int(island) % 2 == 0 else "reciprocal",
                     "pair": int(island) // 2}
            for island, row in zip(islands, winners, strict=True)
        },
        "fixed_guess_reference": CHANCE,
        "intervals": "Percentile bootstrap across paired holdout seeds for fixed trained champions. "
                     "Intervals require at least two seeds and describe world sampling variability.",
        "conditions": reports, "timing_deltas": timing_deltas,
    }
    if final:
        _save_champions(out, winners, best, checkpoint)
    _write(path, result)
    publish(holdout_mode=None, holdout_modes_completed=["full", "fixed_guess"], audit_condition=None)
    return result
