"""Evolve full graph plasticity on a single, recurring A/B task."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import statistics
import time

import torch

from experiments.seed_evolution.large_runner import _lock, _save, _utc, _write
from experiments.undirected import evolved_plasticity_run as base
from experiments.undirected.plasticity_continuous_world import evaluate_baseline, evaluate_continuous
from experiments.undirected.plasticity_curriculum_run import candidate, plan, read_batch, record_batch


OUT = base.ROOT / "data/undirected/2026-11-16-plasticity-uncapped-v1"
KIND = "uncapped_continuous"


def protocol(args):
    files = (
        "uncapped_plasticity_run.py", "uncapped_plastic_graph.py", "plasticity_continuous_world.py",
        "evolved_plasticity_run.py", "evolved_plasticity_world.py", "evolved_plasticity.py",
        "plasticity_curriculum_run.py", "plasticity_curriculum.py", "model.py",
        "../seed_evolution/large_runner.py",
    )
    source = Path(__file__).resolve().parent
    return {
        "schema": 1, "experiment_kind": KIND, "training_modes": ["full"],
        "population": args.population, "generations": args.generations, "islands": args.islands,
        "nodes": 80, "edge_pairs": 3160, "birth_edges": 640,
        "actors": args.actors, "batch_size": args.batch_size, "visits": args.visits,
        "seed": args.seed, "device": args.device, "audit_every": args.audit_every,
        "audit_seeds": args.audit_seeds, "audit_actors": args.audit_actors,
        "inheritance": "birth genome and rule; reinitialize each actor's lifetime state",
        "ordering": ["strict", "strict_both_return", "strict_return", "completion", "reward for fewer average edges",
                     "idle", "ready_coverage"],
        "reward_gate": "at least one strict completion in both recurring contexts",
        "precision": "float32 highest", "torch_version": torch.__version__,
        "sources": {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in files},
    }


def score_fields(row, max_pairs):
    result = {**row, "fitness": row["strict"], "average_edges": row["mean_edges"]}
    if row["strict_a"] > 0 and row["strict_b"] > 0:
        result["sparsity_bonus"] = 1 - row["mean_edges"] / max_pairs
    return result


def rank(row):
    return (
        row["strict"], row["strict_both_return"], row["strict_return"], row["completion"],
        row.get("sparsity_bonus", -1.0), -row["idle"], row["ready_coverage"],
    )


def elites(config, rows, previous):
    keep = max(1, config["population"] // config["islands"] // 8)
    pools, genomes = [], {}
    for island in range(config["islands"]):
        ranked = sorted((row for row in rows if row["island"] == island), key=rank, reverse=True)
        primary, deferred, counts = [], [], {}
        for row in ranked:
            lineage = row["lineage"]
            if counts.get(lineage, 0) < max(1, keep // 4):
                primary.append(row)
                counts[lineage] = counts.get(lineage, 0) + 1
            else:
                deferred.append(row)
            if len(primary) == keep:
                break
        selected = primary + deferred[:keep - len(primary)]
        pools.append(selected)
        genomes.update({row["id"]: candidate(row, previous, config["nodes"]) for row in selected})
    return {"islands": pools, "genomes": genomes}


def snapshot(rows):
    winner = max(rows, key=rank)
    summary = {
        key: value for key, value in winner.items()
        if isinstance(value, (float, int)) and key not in {"fitness", "seed", "island"}
    }
    return {
        **summary, "best_candidate": winner["id"], "best_fitness": winner["fitness"],
        "mean_fitness": statistics.fmean(row["fitness"] for row in rows),
    }


def audit(config, out, generation, checkpoint, publish):
    path = out / "audits" / f"{generation:04d}.json"
    if path.exists():
        return base.read(path)
    winner = max((row for pool in checkpoint["islands"] for row in pool), key=rank)
    genome = checkpoint["genomes"][winner["id"]]
    final = generation == config["generations"]
    first = config["seed"] + (40_000_003 if final else 30_000_019)
    seeds = list(range(first, first + config["audit_seeds"]))
    pair = {}
    for mode in ("full", "fixed_guess"):
        publish(phase="holdout", holdout_mode=mode, holdout_modes_completed=list(pair))
        rows = []
        for seed in seeds:
            world = dict(seed=seed, actors=config["audit_actors"], device=config["device"], visits=config["visits"])
            if mode == "full":
                fields = evaluate_continuous([genome], **world)
                result = score_fields(base.unpack_scores(fields, 1)[0], config["edge_pairs"])
            else:
                fields = evaluate_baseline(**world)
                result = {}
                for key, value in fields.items():
                    result[key] = float(torch.as_tensor(value).reshape(-1)[0].item())
            rows.append({"seed": seed, **result})
        pair[mode] = rows
    means = {}
    for mode, rows in pair.items():
        keys = set.intersection(*(set(row) for row in rows)) - {"seed"}
        means[mode] = {key: statistics.fmean(row[key] for row in rows) for key in sorted(keys)}
    result = {
        "experiment_kind": KIND, "generation": generation, "candidate": winner["id"],
        "actors": config["audit_actors"], "seeds": seeds, "modes": means, "paired_scores": pair,
    }
    if final:
        _save(out / "champion.pt", {"row": winner, "genome": genome})
    _write(path, result)
    print(f"audit generation={generation} completion={means['full']['completion']:.4f} "
          f"strict={means['full']['strict']:.4f} fixed_strict={means['fixed_guess']['strict']:.4f}", flush=True)
    return result


def batches(folder, entries, config, previous, pool, depth):
    def make(batch):
        return [candidate(row, previous, config["nodes"]) for row in batch]

    queue = deque()
    for start in range(0, len(entries), config["batch_size"]):
        path = folder / "batches" / f"{start:04d}.json.gz"
        group = entries[start:start + config["batch_size"]]
        needed = not path.is_file()
        if pool is None:
            yield path, group, make(group) if needed else None
        else:
            queue.append((path, group, pool.submit(make, group) if needed else None))
            if len(queue) >= depth:
                batch_path, batch_group, future = queue.popleft()
                yield batch_path, batch_group, future.result() if future else None
    for path, group, future in queue:
        yield path, group, future.result() if future else None


def recover(out, config):
    history = []
    for gen in range(1, config["generations"] + 1):
        path = out / "generations" / f"{gen:04d}" / "summary.json"
        if not path.is_file():
            break
        history.append(base.read(path))
    if history:
        elites_path = out / "generations" / f"{len(history):04d}" / "elites.pt"
        checkpoint = torch.load(elites_path, map_location="cpu", weights_only=True)
    else:
        checkpoint = {}
    return history, checkpoint


def audit_due(config, generation):
    return generation in (1, config["generations"]) or generation % config["audit_every"] == 0


def run(args, config, pool):
    out = args.out
    pop = config["population"]
    generations = config["generations"]
    history, previous = recover(out, config)
    progress_path = out / "progress.json"
    started = base.read(progress_path)["started_utc"] if progress_path.is_file() else _utc()
    status = {
        "state": "running", "phase": "search", "experiment_kind": KIND,
        "device": config["device"], "started_utc": started, "population": pop,
        "generations": generations, "generation": max(1, len(history)),
        "generation_completed": pop if history else 0, "generation_total": pop,
        "total_candidates": generations * pop, "completed_candidates": len(history) * pop,
    }
    if history:
        status.update(history[-1], checkpoint=f"generations/{len(history):04d}/elites.pt")
    clock = time.perf_counter()
    new_candidates = new_batches = 0

    def publish(**updates):
        status.update(updates, updated_utc=_utc())
        if new_candidates:
            status["candidates_per_minute"] = 60 * new_candidates / (time.perf_counter() - clock)
        _write(progress_path, status)

    _write(out / "process.json", {"pid": os.getpid(), "started_utc": _utc(), "run_dir": str(out),
                                  "device": config["device"], "prefetch": args.prefetch})
    print(f"run={out} generations={generations} population={pop} batch={config['batch_size']} "
          f"visits={config['visits']} prefetch={args.prefetch}", flush=True)
    try:
        publish()
        if history:
            _write(out / "history.json", history)
            if audit_due(config, len(history)):
                _write(out / "holdout.json", audit(config, out, len(history), previous, publish))
        for generation in range(len(history) + 1, generations + 1):
            folder = out / "generations" / f"{generation:04d}"
            entries = base.load_or_write(folder / "plan.json", plan(config, generation, previous))
            seed = config["seed"] + 10_000_019 + generation - 1
            rows = []
            publish(phase="search", generation=generation, generation_completed=0,
                    holdout_mode=None, holdout_modes_completed=[])
            for path, group, genomes in batches(folder, entries, config, previous, pool, args.prefetch):
                if genomes is None:
                    batch = read_batch(path, group)
                else:
                    before = time.perf_counter()
                    measured = evaluate_continuous(
                        genomes, seed=seed, actors=config["actors"], device=config["device"], visits=config["visits"],
                    )
                    fields = base.unpack_scores(measured, len(group))
                    batch = [{**entry, **score_fields(row, config["edge_pairs"])}
                             for entry, row in zip(group, fields, strict=True)]
                    record_batch(path, batch)
                    new_candidates += len(batch)
                    new_batches += 1
                    print(f"generation={generation} committed={len(rows)+len(batch)}/{pop} "
                          f"eval_commit_s={time.perf_counter()-before:.3f}", flush=True)
                rows.extend(batch)
                summary = snapshot(rows)
                status.pop("sparsity_bonus", None)
                publish(**summary, completed_candidates=(generation - 1) * pop + len(rows),
                        generation_completed=len(rows), checkpoint=path.relative_to(out).as_posix())
                if args.batch_limit and new_batches >= args.batch_limit:
                    publish(state="paused")
                    return

            previous = elites(config, rows, previous)
            summary = {
                **summary, "generation": generation, "candidates": pop, "development_seed": seed,
            }
            _save(folder / "elites.pt", previous)
            _write(folder / "summary.json", summary)
            history.append(summary)
            _write(out / "history.json", history)
            print(f"generation={generation} winner={summary['best_candidate']} "
                  f"strict={summary['strict']:.4f} mean_edges={summary['average_edges']:.2f}", flush=True)
            if audit_due(config, generation):
                _write(out / "holdout.json", audit(config, out, generation, previous, publish))

        publish(state="complete", phase="complete", holdout_mode=None,
                holdout_modes_completed=["full", "fixed_guess"], checkpoint="champion.pt")
    except (Exception, KeyboardInterrupt) as error:
        state = "paused" if isinstance(error, KeyboardInterrupt) else "failed"
        publish(state=state, phase=state, failure=f"{type(error).__name__}: {error}")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--population", type=int, default=1024)
    parser.add_argument("--generations", type=int, default=1024)
    parser.add_argument("--islands", type=int, default=8)
    parser.add_argument("--actors", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--visits", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--audit-every", type=int, default=128)
    parser.add_argument("--audit-seeds", type=int, default=8)
    parser.add_argument("--audit-actors", type=int, default=16)
    parser.add_argument("--batch-limit", type=int, default=0, help="Pause after this many new batches.")
    args = parser.parse_args(argv)
    args.out = args.out.resolve()
    if (min(args.population, args.generations, args.islands, args.actors, args.batch_size,
            args.audit_every, args.audit_seeds, args.audit_actors) < 1
            or args.population % args.islands or args.batch_size > args.population
            or args.visits < 4 or args.visits % 2 or min(args.prefetch, args.batch_limit, args.seed) < 0):
        parser.error("Use positive budgets, evenly sized islands and an even number of at least four visits.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    with _lock(args.out / ".lock"):
        config = base.load_or_write(args.out / "protocol.json", protocol(args))
        path = args.out / "progress.json"
        if path.is_file() and base.read(path)["state"] == "complete":
            print(f"already complete: {args.out}")
        elif args.prefetch:
            with ThreadPoolExecutor(max_workers=1) as pool:
                run(args, config, pool)
        else:
            run(args, config, None)


if __name__ == "__main__":
    main()
