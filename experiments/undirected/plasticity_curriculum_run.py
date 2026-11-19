"""Run paired evolution over a timing, version, symbol and conflict curriculum."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch

from experiments.seed_evolution.large_runner import _atomic, _lock, _save, _utc, _write
from experiments.undirected import evolved_plasticity_run as base
from experiments.undirected.evolved_plasticity_world import FIRST_FINISH, NEW, N_SYMBOLS, READY, VERSION
from experiments.undirected.plasticity_curriculum import STAGES, evaluate_phase, make_phase_schedule, rollout_phase


OUT = base.ROOT / "data/undirected/2026-11-16-plasticity-curriculum-v1"
ARMS = ("full", "weights")
COUNTS = (64, 192, 384, 384)
AUDIT_MODES = {
    "full": ("full", "full"), "weights": ("weights", "weights"),
    "full_weights": ("full", "weights"), "full_frozen": ("full", "frozen"),
    "weights_frozen": ("weights", "frozen"),
}


class FixedGuess:
    """Use public READY and version, then try symbol zero once per request."""

    def __init__(self, actors):
        self.tried = torch.zeros(1, actors, dtype=torch.bool)

    def act(self, obs):
        self.tried.logical_and_(obs[..., NEW] == 0)
        choose = (obs[..., READY] != 0) & ~self.tried
        self.tried.logical_or_(choose)
        version = obs[..., VERSION].argmax(dim=-1)
        action = torch.zeros(*choose.shape, 2, dtype=torch.long)
        action[..., 0] = choose * (FIRST_FINISH + N_SYMBOLS * version)
        return action


def protocol(args):
    root = base.ROOT
    sources = (
        "experiments/undirected/plasticity_curriculum_run.py",
        "experiments/undirected/plasticity_curriculum.py",
        "experiments/undirected/evolved_plasticity_run.py",
        "experiments/undirected/evolved_plasticity.py",
        "experiments/undirected/evolved_plasticity_world.py",
        "experiments/undirected/model.py",
        "experiments/seed_evolution/large_runner.py",
    )
    bounds = {}
    start = 1
    for stage, count in zip(STAGES, args.stage_counts, strict=True):
        bounds[stage] = [start, start + count - 1]
        start += count
    return {
        "schema": 1, "arms": list(ARMS), "generations": start - 1, "stage_bounds": bounds,
        "population": args.population, "islands": args.islands, "nodes": args.nodes,
        "actors": args.actors, "batch_size": args.batch_size, "device": args.device,
        "seed": args.seed, "holdout_seeds": args.holdout_seeds, "holdout_actors": args.holdout_actors,
        "selection": "independent arms, shared development tapes, reevaluated elite birth genomes",
        "world_seeds": "development and paired stage holdout streams are disjoint",
        "precision": "float32 highest", "torch_version": torch.__version__,
        "sources": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
    }


def candidate(entry, previous, nodes):
    if entry["kind"] == "elite":
        return previous["genomes"][entry["parent"]]
    return base.make_candidate(entry, previous, nodes)


def plan(config, generation, previous):
    entries = base.plan_generation(config, generation - 1, previous)
    if previous:
        for lane in range(config["population"] // config["islands"]):
            if lane % 16 == 0:
                for island, pool in enumerate(previous["islands"]):
                    parent = pool[(lane // 16) % len(pool)]
                    index = lane * config["islands"] + island
                    entries[index].update(kind="elite", parent=parent["id"], lineage=parent["lineage"])
    return entries


def elites(config, rows, previous):
    keep = max(1, config["population"] // config["islands"] // 8)
    pools, genomes = [], {}
    for island in range(config["islands"]):
        ranked = sorted((row for row in rows if row["island"] == island), key=base.score_key, reverse=True)
        selected, deferred, counts = [], [], {}
        for row in ranked:
            lineage = row["lineage"]
            if counts.get(lineage, 0) < max(1, keep // 4):
                selected.append(row)
                counts[lineage] = counts.get(lineage, 0) + 1
            else:
                deferred.append(row)
            if len(selected) == keep:
                break
        selected.extend(deferred[:keep - len(selected)])
        pools.append(selected)
        for row in selected:
            genomes[row["id"]] = candidate(row, previous, config["nodes"])
    return {"islands": pools, "genomes": genomes}


def audit(out, stage, config, generation, previous, publish):
    path = out / "audits" / (stage + ".json")
    if path.exists():
        return base.read(path)
    nominees = {}
    for arm in ARMS:
        pool = previous[arm]
        winner = max((row for island in pool["islands"] for row in island), key=base.score_key)
        nominees[arm] = (winner["id"], pool["genomes"][winner["id"]])
    first_seed = config["seed"] + 20_000_033 + STAGES.index(stage) * 100_000
    seeds = list(range(first_seed, first_seed + config["holdout_seeds"]))
    means, paired = {}, {}
    for label, (arm, mode) in AUDIT_MODES.items():
        publish(phase="holdout", holdout_mode=label, holdout_modes_completed=list(means))
        rows = [
            base.unpack_scores(evaluate_phase(
                [nominees[arm][1]], stage=stage, seed=seed,
                actors=config["holdout_actors"], device=config["device"], mode=mode,
            ), 1)[0]
            for seed in seeds
        ]
        means[label] = display_fields({key: statistics.fmean(row[key] for row in rows) for key in rows[0]}, stage)
        paired[label] = [{"seed": seed, **row} for seed, row in zip(seeds, rows, strict=True)]
    reference = {}
    for seed in seeds:
        tape = make_phase_schedule(stage=stage, seed=seed, actors=config["holdout_actors"], device="cpu")
        scores = rollout_phase(FixedGuess(config["holdout_actors"]), tape, 1, stage=stage)
        reference[seed] = {key: float(value[0]) for key, value in scores.items()}
    means["fixed_guess"] = display_fields({
        key: statistics.fmean(row[key] for row in reference.values()) for key in next(iter(reference.values()))
    }, stage)
    paired["fixed_guess"] = [{"seed": seed, **row} for seed, row in reference.items()]
    result = {
        "stage": stage, "generation": generation, "candidates": {arm: item[0] for arm, item in nominees.items()},
        "actors": config["holdout_actors"], "seeds": seeds, "modes": means, "paired_scores": paired,
    }
    if stage == STAGES[-1]:
        _save(out / "champions.pt", {arm: {"id": item[0], "genome": item[1]} for arm, item in nominees.items()})
    _write(path, result)
    print(f"audit stage={stage} completion { {key: round(row['completion'], 5) for key, row in means.items()} }",
          flush=True)
    return result


def record_batch(path, rows):
    encoded = json.dumps(rows, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    _atomic(path, lambda temp: temp.write_bytes(gzip.compress(encoded, compresslevel=1, mtime=0)))


def read_batch(path, entries):
    rows = json.loads(gzip.decompress(path.read_bytes()))
    if [row["id"] for row in rows] != [entry["id"] for entry in entries]:
        raise ValueError(f"Candidate order differs: {path}")
    return rows


def prepare(entries, arms, previous, nodes):
    if not previous and len(arms) == 2:
        genomes = [candidate(entry, {}, nodes) for entry in entries[arms[0]]]
        return dict.fromkeys(arms, genomes)
    return {
        arm: [candidate(entry, previous.get(arm, {}), nodes) for entry in entries[arm]]
        for arm in arms
    }


def pending_groups(folder, config, plans):
    width = config["population"]
    for start in range(0, width, config["batch_size"]):
        end = start + config["batch_size"]
        entries = {arm: plans[arm][start:end] for arm in ARMS}
        paths = {arm: folder / "batches" / arm / f"{start:04d}.json.gz" for arm in ARMS}
        missing = [arm for arm in ARMS if not paths[arm].exists()]
        yield entries, paths, missing


def prepared_groups(groups, previous, config, executor, depth):
    if not executor:
        for entries, paths, missing in groups:
            yield entries, paths, prepare(entries, missing, previous, config["nodes"])
        return
    queue = deque()
    for entries, paths, missing in groups:
        queue.append((entries, paths, executor.submit(prepare, entries, missing, previous, config["nodes"])))
        if len(queue) >= depth:
            first_entries, first_paths, future = queue.popleft()
            yield first_entries, first_paths, future.result()
    for entries, paths, future in queue:
        yield entries, paths, future.result()


def stage_for(config, generation):
    return next(stage for stage, (start, end) in config["stage_bounds"].items() if start <= generation <= end)


def arm_snapshot(rows, stage):
    if not rows:
        return None
    return display_fields(base.score_snapshot(rows), stage)


def display_fields(snapshot, stage):
    if stage == "timing":
        snapshot.pop("version_accuracy", None)
        snapshot.pop("version_coverage", None)
    if stage in ("timing", "version"):
        snapshot.pop("symbol_accuracy", None)
        snapshot.pop("symbol_coverage", None)
    if stage != "symbol":
        for key in ("symbol_first", "symbol_late"):
            snapshot.pop(key, None)
    if stage != "conflict":
        snapshot.pop("strict_b", None)
        snapshot.pop("strict_return", None)
    return snapshot


def recover(out, config):
    history = []
    for generation in range(1, config["generations"] + 1):
        folder = out / "generations" / f"{generation:04d}"
        if not (folder / "summary.json").exists():
            break
        history.append(base.read(folder / "summary.json"))
    previous = {}
    if history:
        checkpoint = out / "generations" / f"{len(history):04d}" / "elites.pt"
        previous = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return history, previous


def run(args, config, executor):
    out = args.out
    width = config["population"]
    session_start = time.perf_counter()
    started = base.read(out / "progress.json")["started_utc"] if (out / "progress.json").exists() else _utc()
    history, previous = recover(out, config)
    new_candidates = new_batches = 0
    status = {
        "state": "running", "phase": "search", "started_utc": started, "device": config["device"],
        "population": width, "generations": config["generations"], "stage_bounds": config["stage_bounds"],
        "completed_candidates": 2 * len(history) * width,
        "total_candidates": 2 * config["generations"] * width, "generation_total": 2 * width,
    }
    if history:
        status.update(history[-1], checkpoint=f"generations/{len(history):04d}/elites.pt")
        status.update(generation_completed=2 * width, arm_counts=dict.fromkeys(ARMS, width))

    def publish(**updates):
        status.update(updates, updated_utc=_utc())
        if new_candidates:
            status["candidates_per_minute"] = 60 * new_candidates / (time.perf_counter() - session_start)
        _write(out / "progress.json", status)

    def publish_rows(rows, stage, generation, checkpoint):
        arm_counts = {arm: len(rows[arm]) for arm in ARMS}
        done = sum(arm_counts.values())
        summaries = {arm: arm_snapshot(rows[arm], stage) for arm in ARMS}
        # These legacy top cards track the full-plasticity leader only.
        for key in ("strict_b", "strict_return", "version_accuracy", "symbol_accuracy"):
            status.pop(key, None)
        publish(
            **(summaries["full"] or {}), arms=summaries, arm_counts=arm_counts,
            completed_candidates=2 * (generation - 1) * width + done, generation_completed=done,
            generation=generation, stage=stage, checkpoint=checkpoint,
        )
        return summaries

    def start_generation(stage, generation):
        leader = status.get("arms", {}).get("full")
        for key in leader or {}:
            status.pop(key, None)
        publish(phase="search", generation=generation, stage=stage,
                arms=dict.fromkeys(ARMS), arm_counts=dict.fromkeys(ARMS, 0),
                generation_completed=0, holdout_mode=None, holdout_modes_completed=[])

    _write(out / "process.json", {"pid": os.getpid(), "started_utc": _utc(), "run_dir": str(out),
                                  "device": config["device"], "prefetch": args.prefetch})
    print(f"run={out} arms={ARMS} population={width} generations={config['generations']} "
          f"batch={config['batch_size']} prefetch={args.prefetch}", flush=True)
    try:
        # A process may have stopped after committing a boundary generation.
        if history:
            last = history[-1]
            _write(out / "history.json", history)
            if last["generation"] == config["stage_bounds"][last["stage"]][1]:
                result = audit(out, last["stage"], config, last["generation"], previous, publish)
                _write(out / "holdout.json", result)
        for generation in range(len(history) + 1, config["generations"] + 1):
            stage = stage_for(config, generation)
            folder = out / "generations" / f"{generation:04d}"
            rows = {arm: [] for arm in ARMS}
            plans = {arm: plan(config, generation, previous.get(arm, {})) for arm in ARMS}
            base.load_or_write(folder / "plans.json", plans)
            seed = config["seed"] + 10_000_019 + generation - 1
            start_generation(stage, generation)
            groups = pending_groups(folder, config, plans)
            for entries, paths, genomes in prepared_groups(groups, previous, config, executor, args.prefetch):
                for arm in ARMS:
                    path = paths[arm]
                    if arm in genomes:
                        started_batch = time.perf_counter()
                        measured = evaluate_phase(
                            genomes[arm], stage=stage, seed=seed, actors=config["actors"],
                            device=config["device"], mode=arm,
                        )
                        scores = base.unpack_scores(measured, len(entries[arm]))
                        batch = [{**entry, **score} for entry, score in zip(entries[arm], scores, strict=True)]
                        record_batch(path, batch)
                        new_candidates += len(batch)
                        new_batches += 1
                        print(f"generation={generation} stage={stage} arm={arm} "
                              f"committed={len(rows[arm])+len(batch)}/{width} "
                              f"eval_commit_s={time.perf_counter()-started_batch:.3f}", flush=True)
                    else:
                        batch = read_batch(path, entries[arm])
                    rows[arm].extend(batch)
                    summaries = publish_rows(rows, stage, generation, path.relative_to(out).as_posix())
                    if args.batch_limit and new_batches >= args.batch_limit:
                        publish(state="paused")
                        return

            previous = {arm: elites(config, rows[arm], previous.get(arm, {})) for arm in ARMS}
            summary = {
                **summaries["full"], "generation": generation, "stage": stage, "arms": summaries,
                "candidates": 2 * width, "development_seed": seed,
            }
            _save(folder / "elites.pt", previous)
            _write(folder / "summary.json", summary)
            history.append(summary)
            _write(out / "history.json", history)
            if generation == config["stage_bounds"][stage][1]:
                result = audit(out, stage, config, generation, previous, publish)
                _write(out / "holdout.json", result)
        publish(state="complete", phase="complete", holdout_mode=None,
                holdout_modes_completed=list(AUDIT_MODES), checkpoint="champions.pt")
    except (Exception, KeyboardInterrupt) as error:
        publish(state="paused" if isinstance(error, KeyboardInterrupt) else "failed",
                failure=f"{type(error).__name__}: {error}")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--population", type=int, default=1024)
    parser.add_argument("--islands", type=int, default=8)
    parser.add_argument("--nodes", type=int, default=80)
    parser.add_argument("--actors", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--prefetch", type=int, default=2, help="Number of CPU batch groups to prepare ahead.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--holdout-seeds", type=int, default=8)
    parser.add_argument("--holdout-actors", type=int, default=16)
    parser.add_argument("--stage-counts", type=int, nargs=4, default=COUNTS)
    parser.add_argument("--batch-limit", type=int, default=0, help="Pause after this many new arm batches.")
    args = parser.parse_args(argv)
    args.out = args.out.resolve()
    if (min(args.population, args.islands, args.actors, args.batch_size, args.holdout_seeds,
            args.holdout_actors, *args.stage_counts) < 1 or args.population % args.islands
            or args.batch_size > args.population or args.nodes < 78 or args.seed < 0
            or args.batch_limit < 0 or args.prefetch < 0):
        parser.error("Positive budgets required, with population divisible by islands and batch at most population.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    with _lock(args.out / ".lock"):
        config = base.load_or_write(args.out / "protocol.json", protocol(args))
        if (args.out / "progress.json").exists() and base.read(args.out / "progress.json")["state"] == "complete":
            print(f"already complete: {args.out}")
        elif args.prefetch:
            with ThreadPoolExecutor(max_workers=1) as executor:
                run(args, config, executor)
        else:
            run(args, config, None)


if __name__ == "__main__":
    main()
