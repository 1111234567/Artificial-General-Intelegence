"""Measure the four directed islands on the same held-out contracts."""
from __future__ import annotations

import numpy as np
import torch

from experiments.seed_evolution.large_runner import _write
from .evolved_plasticity_run import read
from .uncapped_directed_audit import _means, _measure, _paired_delta, _save_champions
from .uncapped_directed_reward import MODES
from .uncapped_island_crossover_run import DELAYED, TIMING, block_ticks
from .uncapped_ready_signal_run import schedule


SCHEMA = 2
ISLANDS = ("0", "1", "2", "3")
# The mixed-topology run used +30m/+40m; the continuation has fresh worlds.
INTERMEDIATE_SEED_OFFSET = 50_000_019
FINAL_SEED_OFFSET = 60_000_003


def _check_tapes(conditions, seeds, actors, visits):
    """Check clock matches using tapes only; no agent inhabits these worlds."""
    native_ticks = block_ticks(TIMING)
    shift = DELAYED["min_wait"] - TIMING["min_wait"]
    for seed in seeds:
        tapes = {}
        for name, (evaluate, timing, ticks) in conditions.items():
            make_tape = getattr(evaluate, "schedule", None)
            settings = dict(seed=seed, actors=actors, visits=visits, device="cpu")
            tape = (schedule(**settings, timing=timing, pad_to=ticks) if make_tape is None
                    else make_tape(**settings))
            if tape.public.shape[1] != ticks:
                raise ValueError(f"audit condition {name} does not use its specified clock")
            tapes[name] = tape
        native = tapes["regular_native"]
        for name, tape in tapes.items():
            if name != "regular_native" and any(
                not torch.equal(getattr(native, key), getattr(tape, key))
                for key in ("maps", "raw", "context", "target")
            ):
                raise ValueError(f"audit condition {name} does not share the regular contracts")
        if "regular_padded" in tapes:
            padded = tapes["regular_padded"]
            if (not torch.equal(padded.public[:, :native_ticks], native.public)
                    or not torch.equal(padded.stage[:, :native_ticks], native.stage)
                    or not padded.stage[:, native_ticks:].eq(3).all()
                    or padded.public[:, native_ticks:].any()):
                raise ValueError("regular padding must add only idle time to the native clock")
        if "long" in tapes:
            long = tapes["long"]
            if (long.stage[:, :shift].eq(2).any()
                    or not torch.equal(native.stage.eq(2), long.stage[:, shift:].eq(2))):
                raise ValueError("long audit must shift READY without changing the contract or window length")


def audit(config, out, generation, checkpoint, publish, *, evaluate, delayed_evaluate, rank,
          matched_regular_evaluate=None):
    """Audit one champion selected on development results per directed island."""
    pools = checkpoint["islands"]
    if len(pools) != len(ISLANDS) or any(not pool for pool in pools):
        raise ValueError("directed continuation audit requires four nonempty islands")
    winners = [max(pool, key=rank) for pool in pools]
    best = max(winners, key=rank)
    genomes = [checkpoint["genomes"][row["id"]] for row in winners]
    if len({row["id"] for row in winners}) != len(ISLANDS) or not all(bool(g["directed"]) for g in genomes):
        raise ValueError("directed continuation audit requires one distinct directed champion per island")
    champions = {island: {"id": row["id"], "topology": "directed"}
                 for island, row in zip(ISLANDS, winners, strict=True)}
    display_island = ISLANDS[winners.index(best)]

    path = out / "audits" / f"{generation:04d}.json"
    final = generation == config["generations"]
    actors = config["audit_actors"]
    visits = 8 if final else 4
    first = config["seed"] + (FINAL_SEED_OFFSET if final else INTERMEDIATE_SEED_OFFSET)
    count = config["audit_seeds"]
    if not 0 < count <= FINAL_SEED_OFFSET - INTERMEDIATE_SEED_OFFSET:
        raise ValueError("audit seed count must keep periodic and final holdout worlds disjoint")
    seeds = list(range(first, first + count))
    modes = MODES if final else MODES[:1]
    conditions = {"regular_native": (evaluate, TIMING, block_ticks(TIMING))}
    if final:
        long_ticks = block_ticks(DELAYED)
        if matched_regular_evaluate is not None:
            conditions["regular_padded"] = (matched_regular_evaluate, TIMING, long_ticks)
        conditions["long"] = (delayed_evaluate, DELAYED, long_ticks)

    identity = {
        "schema": SCHEMA, "experiment_kind": config["experiment_kind"], "generation": generation,
        "final": final, "candidate": best["id"], "display_island": display_island,
        "actors": actors, "seeds": seeds, "visits": visits, "champions": champions,
    }
    labels = [f"{name}/{mode}" for name in conditions for mode in modes]

    def finished():
        publish(holdout_mode=None, audit_condition=None, holdout_modes_completed=list(modes),
                holdout_pairs_completed=labels)

    if path.is_file():
        result = read(path)
        if (any(result.get(key) != value for key, value in identity.items())
                or set(result["conditions"]) != set(conditions)
                or any(set(report["modes"]) != set(modes) for report in result["conditions"].values())):
            raise ValueError(f"audit does not match the directed continuation checkpoint: {path}")
        if final and not all((out / name).is_file() for name in ("champion.pt", "island_champions.pt")):
            _save_champions(out, winners, best, checkpoint)
        finished()
        return result

    _check_tapes(conditions, seeds, actors, visits)
    draws = None if count < 2 else np.random.default_rng(0).integers(count, size=(4096, count))
    reports, completed = {}, []
    for name, (evaluator, timing, ticks) in conditions.items():
        mode_reports, scores = {}, {}
        try:
            for mode in modes:
                publish(phase="holdout", audit_condition=name, holdout_mode=mode,
                        holdout_modes_completed=[m for m in modes if all(
                            f"{condition}/{m}" in completed for condition in conditions
                        )], holdout_pairs_completed=list(completed))
                scores[mode] = _measure(evaluator, genomes, ISLANDS, seeds, mode, actors, visits, config["device"])
                island_reports = {island: {"means": _means(rows), "paired_scores": rows}
                                  for island, rows in scores[mode].items()}
                mode_reports[mode] = {"islands": island_reports}
                completed.append(f"{name}/{mode}")
                first_returns = ",".join(f"{island}:{report['means']['first_finish_correct_first_return']:.4f}"
                                         for island, report in island_reports.items())
                print(f"audit generation={generation} condition={name} mode={mode} "
                      f"first_return={first_returns}", flush=True)
        finally:
            clear = getattr(evaluator, "clear_cache", None)
            if clear is not None:
                clear()

        def mode_delta(left, right):
            return {island: _paired_delta(scores[left][island], scores[right][island], draws) for island in ISLANDS}

        reports[name] = {
            "timing": {**timing, "ticks_per_request": ticks},
            "modes": mode_reports,
            "full_minus_ablation": {mode: mode_delta("full", mode) for mode in modes[1:]},
        }
        if final:
            reports[name]["no_reward_write_minus_frozen"] = mode_delta("no_reward_write", "frozen")

    if "regular_padded" in reports:
        timing_pairs = (("long", "regular_padded"), ("regular_padded", "regular_native"))
    elif final:
        timing_pairs = (("long", "regular_native"),)
    else:
        timing_pairs = ()

    def rows(condition, mode, island):
        return reports[condition]["modes"][mode]["islands"][island]["paired_scores"]

    timing_deltas = {
        f"{left}_minus_{right}": {
            mode: {island: _paired_delta(rows(left, mode, island), rows(right, mode, island), draws)
                   for island in ISLANDS}
            for mode in modes
        }
        for left, right in timing_pairs
    }
    regular = reports["regular_native"]["modes"]
    display = {mode: regular[mode]["islands"][display_island] for mode in modes}
    result = {
        **identity,
        "modes": {mode: report["means"] for mode, report in display.items()},
        "paired_scores": {mode: report["paired_scores"] for mode, report in display.items()},
        "receipts": "Real, action-dependent ACK/REJ receipts in every mode.",
        "intervals": "Percentile bootstrap across paired holdout seeds for fixed trained champions. "
                     "Intervals require at least two seeds and describe world sampling variability.",
        "conditions": reports, "timing_deltas": timing_deltas,
    }
    if final:
        _save_champions(out, winners, best, checkpoint)
    _write(path, result)
    finished()
    return result
