"""Evolve birth graphs and local update rules in a closed-loop world."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

import torch

from experiments.seed_evolution.large_runner import _load, _lock, _save, _utc, _write
from experiments.undirected.evolved_plasticity import make_genome, mutate_genome
from experiments.undirected.evolved_plasticity_world import evaluate


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "data/undirected/2026-11-16-evolved-plasticity-v1"
MODES = ("full", "weights", "frozen", "shuffled")
COUNT_KEYS = ("weight_updates", "births", "deaths", "active_edges")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_or_write(path: Path, value):
    if path.exists():
        existing = read(path)
        if existing != value:
            raise ValueError(f"Checkpoint configuration differs: {path}")
        return existing
    _write(path, value)
    return value


def protocol(args) -> dict:
    source = Path(__file__).resolve().parent
    return {
        "schema": 1,
        "experiment": "evolved local plasticity on a reciprocal graph",
        "world": "single active slot, visible readiness, private A/B maps, continuous A/B/A",
        "training": "evolution, no autograd, birth genome only inherited",
        "selection": "rank by fitness with a small completion reserve per island",
        "population": args.population,
        "islands": args.islands,
        "generations": args.generations,
        "nodes": args.nodes,
        "actors": args.actors,
        "batch_size": args.batch_size,
        "requests_per_phase": args.requests_per_phase,
        "min_wait": args.min_wait,
        "max_wait": args.max_wait,
        "seed": args.seed,
        "dev_world_seed": args.seed + 10_000_019,
        "holdout_world_seeds": [args.seed + 20_000_033 + i for i in range(args.holdout_seeds)],
        "holdout_actors": args.holdout_actors,
        "device": args.device,
        "torch_version": torch.__version__,
        "sources": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in (
                "evolved_plasticity.py", "evolved_plasticity_world.py",
                "evolved_plasticity_run.py",
            )
        },
    }


def plan_generation(config: dict, generation: int, previous: dict) -> list[dict]:
    rng = random.Random(config["seed"] * 1_000_003 + generation)
    islands = config["islands"]
    width = config["population"] // islands
    plan = []
    for lane in range(width):
        for island in range(islands):
            identifier = f"g{generation:03d}-i{island:02d}-c{lane:04d}"
            draw = (lane + 3 * island) % 8
            parent = None
            kind = "founder"
            if previous and draw < 7:
                # Six eighths local, one eighth wide with access to the adjacent island.
                parent_island = island
                kind = "local"
                if draw == 6:
                    kind = "wide"
                    if rng.random() < .5:
                        parent_island = (island + 1) % islands
                pool = previous["islands"][parent_island]
                parent = rng.choice(pool)
            plan.append({
                "id": identifier, "island": island, "kind": kind,
                "seed": rng.randrange(1, 2**31),
                "parent": parent["id"] if parent else None,
                "lineage": parent["lineage"] if parent else identifier,
            })
    return plan


def make_candidate(entry: dict, previous: dict, nodes: int) -> dict:
    if entry["parent"] is None:
        return make_genome(entry["seed"], nodes)
    parent = previous["genomes"][entry["parent"]]
    return mutate_genome(parent, entry["seed"], wide=entry["kind"] == "wide")


def score_key(row: dict) -> tuple:
    return (
        row["fitness"], row["completion"], row["strict_return"],
        -row.get("idle", 0.0),
    )


def score_batch(config: dict, entries: list[dict], previous: dict) -> list[dict]:
    genomes = [make_candidate(entry, previous, config["nodes"]) for entry in entries]
    fields = evaluate(
        genomes, seed=config["dev_world_seed"], actors=config["actors"],
        device=config["device"], requests_per_phase=config["requests_per_phase"],
        min_wait=config["min_wait"], max_wait=config["max_wait"],
    )
    return [
        {**entry, **metrics}
        for entry, metrics in zip(entries, unpack_scores(fields, len(entries)), strict=True)
    ]


def unpack_scores(fields: dict, size: int) -> list[dict]:
    values = {
        key: (value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value)
        for key, value in fields.items()
    }
    if "active_edges" not in values and "final_edges" in values:
        values["active_edges"] = values["final_edges"]
    required = {"fitness", "completion", "strict_a", "strict_b", "strict_return", *COUNT_KEYS}
    missing = required.difference(values)
    if missing:
        raise ValueError(f"Evaluator fields missing: {sorted(missing)}")
    for key, column in values.items():
        if not isinstance(column, (tuple, list)) or len(column) != size:
            raise ValueError(f"Expected one value per candidate in {key}")
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in column):
            raise ValueError(f"Invalid scalar score in {key}")
    return [dict(zip(values, row, strict=True)) for row in zip(*values.values(), strict=True)]


def build_elites(config: dict, rows: list[dict], previous: dict) -> dict:
    width = config["population"] // config["islands"]
    survivor_count = min(width, max(1, width // 8))
    pools = []
    genomes = {}
    for island in range(config["islands"]):
        ranked = sorted(
            (row for row in rows if row["island"] == island), key=score_key, reverse=True,
        )
        reserve = survivor_count // 4
        primary_count = survivor_count - reserve
        completion_order = sorted(
            ranked[primary_count:],
            key=lambda row: (row["completion"], row["strict_return"], row["fitness"]),
            reverse=True,
        )
        completion_choices = completion_order[:reserve]
        chosen = {row["id"] for row in completion_choices}
        ranked = ranked[:primary_count] + completion_choices + [
            row for row in ranked[primary_count:] if row["id"] not in chosen
        ]
        selected = []
        deferred = []
        lineage_count = {}
        for row in ranked:
            lineage = row["lineage"]
            if lineage_count.get(lineage, 0) >= max(1, survivor_count // 4):
                deferred.append(row)
            else:
                selected.append(row)
                lineage_count[lineage] = lineage_count.get(lineage, 0) + 1
            if len(selected) == survivor_count:
                break
        selected.extend(deferred[:survivor_count - len(selected)])
        pools.append(selected)
        for row in selected:
            genomes[row["id"]] = make_candidate(row, previous, config["nodes"])
    return {"islands": pools, "genomes": genomes}


def score_snapshot(rows: list[dict]) -> dict:
    best = max(rows, key=score_key)
    metrics = {
        key: value for key, value in best.items()
        if isinstance(value, (int, float)) and key not in {"fitness", "seed", "island"}
    }
    return {
        **metrics, "best_candidate": best["id"], "best_fitness": best["fitness"],
        "mean_fitness": statistics.fmean(row["fitness"] for row in rows),
    }


def publish_holdout(args, config, champion: dict, progress) -> None:
    path = args.out / "holdout.json"
    if path.exists():
        if read(path)["candidate"] != champion["row"]["id"]:
            raise ValueError("Holdout candidate differs from committed champion")
        return
    modes = {}
    paired = {}
    for mode in MODES:
        progress(phase="holdout", holdout_modes_completed=list(modes), holdout_mode=mode)
        scores = []
        for seed in config["holdout_world_seeds"]:
            fields = evaluate(
                [champion["genome"]], seed=seed, actors=config["holdout_actors"],
                device=config["device"], mode=mode, requests_per_phase=config["requests_per_phase"],
                min_wait=config["min_wait"], max_wait=config["max_wait"],
            )
            scores.extend(unpack_scores(fields, 1))
        modes[mode] = {
            key: statistics.fmean(row[key] for row in scores) for key in scores[0]
        }
        paired[mode] = [
            {"seed": seed, **row}
            for seed, row in zip(config["holdout_world_seeds"], scores, strict=True)
        ]
        print(f"holdout mode={mode} completion={modes[mode]['completion']:.4f}", flush=True)
    seeds = config["holdout_world_seeds"]
    _write(path, {
        "candidate": champion["row"]["id"], "actors": config["holdout_actors"],
        "seeds": seeds, "modes": modes, "paired_scores": paired,
    })


def train(args, config: dict) -> None:
    out = args.out
    pop = config["population"]
    generations = config["generations"]
    started_utc = _utc()
    if (out / "progress.json").exists():
        started_utc = read(out / "progress.json")["started_utc"]
    _write(out / "process.json", {
        "pid": os.getpid(), "started_utc": _utc(), "run_dir": str(out), "device": config["device"],
    })
    status = {
        "state": "running", "phase": "search", "started_utc": started_utc, "generation": 1,
        "generations": generations, "population": pop, "device": config["device"],
        "completed_candidates": 0, "total_candidates": pop * generations,
        "generation_completed": 0, "generation_total": pop,
    }
    session_start = time.perf_counter()
    new_candidates = 0
    new_batches = 0

    def progress(**updates):
        status.update(updates)
        status["updated_utc"] = _utc()
        if new_candidates:
            status["candidates_per_minute"] = 60 * new_candidates / (time.perf_counter() - session_start)
        _write(out / "progress.json", status)

    previous = {}
    history = []
    champion = None
    try:
        progress()
        print(f"run={out} population={pop} generations={generations} device={config['device']}", flush=True)
        for generation in range(generations):
            folder = out / "generations" / f"generation_{generation:03d}"
            summary_path = folder / "summary.json"
            elites_path = folder / "elites.pt"
            if summary_path.exists():
                summary = read(summary_path)
                previous = _load(elites_path)
                history.append(summary)
                winner = max(
                    (row for pool in previous["islands"] for row in pool), key=score_key,
                )
                if champion is None or score_key(winner) > score_key(champion["row"]):
                    champion = {"row": winner, "genome": previous["genomes"][winner["id"]]}
                continue

            entries = load_or_write(folder / "plan.json", plan_generation(config, generation, previous))
            rows = []
            for start in range(0, pop, config["batch_size"]):
                end = min(pop, start + config["batch_size"])
                batch_path = folder / "batches" / f"batch_{start:04d}.json"
                if batch_path.exists():
                    batch = read(batch_path)
                    expected_ids = [entry["id"] for entry in entries[start:end]]
                    if [row["id"] for row in batch["rows"]] != expected_ids:
                        raise ValueError(f"Candidate order differs: {batch_path}")
                else:
                    clock = time.perf_counter()
                    batch = {"rows": score_batch(config, entries[start:end], previous)}
                    batch["eval_seconds"] = time.perf_counter() - clock
                    _write(batch_path, batch)
                    new_candidates += end - start
                    new_batches += 1
                    print(
                        f"generation={generation + 1} committed={end}/{pop} "
                        f"eval_s={batch['eval_seconds']:.2f}",
                        flush=True,
                    )
                rows.extend(batch["rows"])
                progress(
                    **score_snapshot(rows), generation=generation + 1,
                    completed_candidates=generation * pop + len(rows), generation_completed=len(rows),
                    checkpoint=str(batch_path.relative_to(out)),
                )
                if args.batch_limit and new_batches >= args.batch_limit:
                    progress(state="paused")
                    return

            previous = build_elites(config, rows, previous)
            winner = max(rows, key=score_key)
            if champion is None or score_key(winner) > score_key(champion["row"]):
                champion = {"row": winner, "genome": previous["genomes"][winner["id"]]}
            summary = {**score_snapshot(rows), "generation": generation + 1, "candidates": len(rows)}
            _save(elites_path, previous)
            _write(summary_path, summary)
            history.append(summary)
            _write(out / "history.json", history)
            print(
                f"generation={generation + 1} winner={winner['id']} "
                f"fitness={winner['fitness']:.5f} completion={winner['completion']:.4f}",
                flush=True,
            )

        _write(out / "history.json", history)
        _save(out / "champion.pt", champion)
        final_snapshot = {**score_snapshot([champion["row"]]), "mean_fitness": history[-1]["mean_fitness"]}
        progress(
            **final_snapshot,
            generation=generations, generation_completed=pop, completed_candidates=generations * pop,
            checkpoint="champion.pt",
        )
        publish_holdout(args, config, champion, progress)
        progress(state="complete", phase="complete", holdout_mode=None, holdout_modes_completed=list(MODES))
    except Exception as error:
        progress(state="failed", phase="failed", failure=f"{type(error).__name__}: {error}")
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--population", type=int, default=1024)
    parser.add_argument("--islands", type=int, default=8)
    parser.add_argument("--generations", type=int, default=32)
    parser.add_argument("--nodes", type=int, default=80)
    parser.add_argument("--actors", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--requests-per-phase", type=int, default=6)
    parser.add_argument("--min-wait", type=int, default=1)
    parser.add_argument("--max-wait", type=int, default=3)
    parser.add_argument("--holdout-seeds", type=int, default=8)
    parser.add_argument("--holdout-actors", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-limit", type=int, default=0, help="Pause after this many new batches.")
    args = parser.parse_args(argv)
    args.out = args.out.resolve()
    if (min(args.population, args.islands, args.generations, args.actors, args.batch_size,
            args.holdout_seeds, args.holdout_actors) < 1 or args.population % args.islands
            or args.nodes < 78 or args.requests_per_phase < 4 or not 1 <= args.min_wait <= args.max_wait
            or args.batch_limit < 0 or args.seed < 0):
        parser.error("Budgets must be positive, population divisible by islands, and wait bounds ordered.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was selected but is unavailable")
    torch.set_num_threads(1)
    with _lock(args.out / ".lock"):
        config = load_or_write(args.out / "protocol.json", protocol(args))
        if (args.out / "progress.json").exists() and read(args.out / "progress.json")["state"] == "complete":
            print(f"already complete: {args.out}")
        else:
            train(args, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
