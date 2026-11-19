"""Run the matched original, local, and modulated birth-rule searches on one device."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
import os
from pathlib import Path
from statistics import fmean
import time

import torch

from experiments.seed_evolution.large_runner import _lock, _save, _utc, _write
from . import local_rule_genetics as genetics
from . import uncapped_directed_breeding as directed
from . import uncapped_directed_exploration as exploration
from .evolved_plasticity_run import load_or_write, read
from .plasticity_curriculum_run import read_batch, record_batch
from .uncapped_directed_reward_run import rank


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/undirected/2026-11-19-local-rule-v1"
ARCHIVE = OUT.with_name("2026-11-18-private-symbol-iid-v1")
ARMS = ("control", "local", "modulated")
ACTORS, VISITS = 8, 16
SCHEMA = 1
SCORE_FIELDS = ("selection_score", "strict", "first_return_worst_context", "first_return_mean")


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _sources():
    source = Path(__file__).resolve().parent
    names = ("local_rule_run.py", "local_rule_genetics.py", "local_rule_evaluator.py", "local_rule_world.py",
             "local_rule_receipt.py", "local_rule_policy.py", "local_rule_policy.cu",
             "uncapped_directed_breeding.py", "uncapped_directed_exploration.py", "uncapped_directed_reward_run.py",
             "uncapped_directed_reward.py", "uncapped_directed_matvec_tensor.cu", "uncapped_private_symbol_breeding.py",
             "uncapped_private_symbol_span.py", "uncapped_private_symbol_span.cu", "uncapped_private_symbol_span_run.py",
             "uncapped_private_symbol_run.py", "uncapped_private_symbol_world.py", "uncapped_private_symbol_iid_world.py",
             "uncapped_private_symbol_world_cuda.py", "uncapped_private_symbol_world_cuda.cu",
             "uncapped_private_symbol_post_answer.py", "uncapped_private_symbol_post_answer.cu",
             "uncapped_island_recombination.py", "variable_birth_genome.py", "evolved_plasticity.py",
             "evolved_plasticity_run.py", "plasticity_curriculum_run.py", "../seed_evolution/large_runner.py", "_cuda_nvrtc.py")
    return {name: _sha(source / name) for name in sorted(names)}


def _protocol(args, warm_start, warm_hash):
    return {
        "schema": SCHEMA, "experiment_kind": "local_rule_comparison", "arms": list(ARMS),
        "arm_schedule": "generation first, with control/local/modulated run serially on one device",
        "population": args.population, "generations": args.generations, "islands": directed.ISLANDS,
        "nodes": directed.NODES, "actors": ACTORS, "visits": VISITS,
        "world": {"requests_per_life": 64, "ticks_per_request": 18, "ready_delay": [8, 12],
                  "answer_window": 4, "raw": "independent, uniform with replacement",
                  "context": "independent, equal probability A and B",
                  "maps": "two private permutations per actor, conflicting for every raw symbol"},
        "score": "(strict_count + 0.1*recovered_count - 0.1*post_answer_actions + both_strict_pairs)/512",
        "revisit_pairing": "per actor and raw symbol, after BOTH contexts have been presented, pair the kth "
                          "subsequent A with the kth subsequent B; never expose pairs to the policy",
        "plasticity": {"control": "archived directed first-legal receipt rule and support schedule",
                       "local": "evolved local per-tick weights, biases, gate evidence and gradual support",
                       "modulated": "same local equations with signed receiving-node activity/trace gain",
                       "precision": "FP16 effective recurrent weights and dynamics, FP32 accumulation; "
                                    "local and modulated also accumulate slow parameter changes in FP32"},
        "device": args.device, "seed": args.seed, "torch_version": str(torch.__version__),
        "batch_size": args.batch_size, "eval_batch_size": args.eval_batch_size,
        "audit_every": args.audit_every, "audit_seeds": args.audit_seeds, "final_audit_seeds": args.final_audit_seeds,
        "receipt_generations": sorted({1, args.generations}), "receipt_seed_counts": {"first": 8, "final": 32},
        "development_seed_offset": 100_000_019, "audit_seed_offset": 200_000_033,
        "final_audit_seed_offset": 300_000_047,
        "warm_start": {**warm_start, "bank_sha256": warm_hash},
        "cold_founders": args.population - warm_start["count"],
        "founder_control": "one initial plan and identical birth tensors across the three arms",
        "inheritance": "only birth graph, rule coefficients, and scalar parameters; reset lifetime state per actor",
        "breeding": exploration.protocol(args.population, directed.ISLANDS),
        "ranking": ["selection_score", "first_return_worst_context", "first_return_mean", "strict", "sparsity_bonus"],
        "perturbation_pairs_per_island": args.perturbation_pairs,
        "perturbation": {"budget": "replace two ordinary local lanes per pair; both eligible for selection",
                         "directions": "seeded symmetric coefficient directions, independent of development scores",
                         "mutation_step": genetics.PLASTIC_MUTATION_STEP, "radius": genetics.PLASTIC_PERTURBATION,
                         "fields": list(genetics.PLASTIC_FIELDS)},
        "holdout": "evaluate every island's development winner on paired unseen worlds; no feedback to breeding",
        "sources": _sources(),
    }


def _warm_bank(args):
    path = args.out / "warm_start.pt"
    if path.exists():
        digest = _sha(path)
        if (args.out / "protocol.json").is_file():
            expected = read(args.out / "protocol.json")["warm_start"]["bank_sha256"]
            if digest != expected:
                raise ValueError(f"Warm birth bank hash differs from the sealed protocol in {path}")
        warm = torch.load(path, map_location="cpu", weights_only=True)
        if (warm["provenance"]["count"] != args.warm or warm["provenance"]["source_run"] != str(args.archive)
                or len(warm["descriptors"]) != args.warm or len(warm["genomes"]) != args.warm):
            raise ValueError(f"Warm birth bank differs from the requested archive or count in {path}")
    elif (args.out / "protocol.json").is_file():
        raise ValueError(f"The sealed run requires its warm birth artifact at {path}")
    else:
        descriptors, bank, provenance = genetics.reconstruct_warm(args.archive, args.warm)
        warm = {"descriptors": descriptors, "genomes": bank, "provenance": provenance}
        _save(path, warm)
        digest = _sha(path)
    return warm, digest


def _directory(out, arm, generation):
    return out / "arms" / arm / "generations" / f"{generation:04d}"


def _checkpoint(folder, arm, generation):
    path = folder / "elites.pt"
    value = torch.load(path, map_location="cpu", weights_only=True)
    ids = [row["id"] for pool in value["islands"] for row in pool]
    if ((value["arm"], value["generation"]) != (arm, generation) or len(ids) != len(set(ids))
            or set(ids) != value["genomes"].keys() or len(value["champions"]) != directed.ISLANDS):
        raise ValueError(f"Malformed checkpoint at {path}")
    return value


def _recover(out, config):
    history, latest = [], {arm: {} for arm in ARMS}
    found_gap = False
    for generation in range(1, config["generations"] + 1):
        for arm in ARMS:
            path = _directory(out, arm, generation) / "summary.json"
            if not path.is_file():
                found_gap = True
                continue
            row = read(path)
            expected = (arm, generation, config["population"])
            if found_gap or (row["arm"], row["generation"], row["candidates"]) != expected:
                raise ValueError(f"Generation commits are out of order at {path}")
            if not path.with_name("elites.pt").is_file():
                raise ValueError(f"Missing candidate checkpoint for sealed summary at {path}")
            history.append(row)
            latest[arm] = row
    previous = {
        arm: _checkpoint(_directory(out, arm, row["generation"]), arm, row["generation"]) if row else {}
        for arm, row in latest.items()
    }
    return history, latest, previous


def _batches(folder, entries, size, previous, warm_genomes, pool):
    def prepare(group):
        return [genetics.candidate(row, previous, directed.NODES, warm_genomes) for row in group]

    queue = deque()
    for start in range(0, len(entries), size):
        path = folder / "batches" / f"{start:04d}.json.gz"
        group = entries[start:start + size]
        queue.append((path, group, None if path.exists() else pool.submit(prepare, group)))
        if len(queue) == 2:
            path, group, future = queue.popleft()
            yield path, group, future.result() if future else None
    for path, group, future in queue:
        yield path, group, future.result() if future else None


def _score_rows(fields, count):
    if not isinstance(fields, list) or len(fields) != count:
        raise ValueError(f"Expected {count} score rows from the evaluator")
    for row in fields:
        if not isinstance(row, dict) or any(key not in row for key in SCORE_FIELDS):
            raise ValueError(f"Expected scalar rank fields {SCORE_FIELDS}")
        for key, value in row.items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"Nonfinite candidate metric {key}")
        if any(type(row[key]) not in (float, int) for key in SCORE_FIELDS):
            raise ValueError(f"Nonnumeric candidate ranking field in {row}")
    return fields


def _mean(rows):
    numeric = [{key for key, value in row.items() if type(value) in (int, float)} for row in rows]
    return {key: fmean(row[key] for row in rows) for key in sorted(set.intersection(*numeric))}


def _summary(rows, islands):
    def summarize(group):
        champion = max(group, key=rank)
        return {"candidates": len(group), "champion": champion, "mean": _mean([row["metrics"] for row in group])}

    groups = [[] for _ in range(islands)]
    for row in rows:
        groups[row["island"]].append(row)
    return {**summarize(rows), "islands": {str(island): summarize(group)
                                          for island, group in enumerate(groups) if group}}


def _training_rows(scored, entries):
    for metrics, entry in zip(scored, entries, strict=True):
        if set(metrics).intersection(entry) or "metrics" in metrics:
            raise ValueError(f"Evaluator fields overlap birth plan fields for {entry['id']}")
    return [{**entry, **metrics, "metrics": metrics} for entry, metrics in zip(entries, scored, strict=True)]


def _audit(out, config, generation, summaries, evaluator_for, progress):
    final = generation == config["generations"]
    if generation != 1 and not final and generation % config["audit_every"]:
        return

    def publish(phase, arm, **fields):
        progress(phase=phase, generation=generation, generation_completed=config["population"],
                 audit_generation=generation, current_arm=arm, latest=summaries[arm], **fields)

    def without_seed(rows):
        return [{key: value for key, value in row.items() if key != "seed"} for row in rows]

    root = out / "audits"
    path = root / f"{generation:04d}.json"
    first = config["seed"] + (config["final_audit_seed_offset"] if final else config["audit_seed_offset"] + 1024 * generation)
    count = config["final_audit_seeds"] if final else config["audit_seeds"]
    seeds = list(range(first, first + count))
    champions = {arm: {island: summary["champion"]["id"] for island, summary in summaries[arm]["islands"].items()}
                 for arm in ARMS}
    header = {"schema": SCHEMA, "generation": generation, "final": final, "actors": config["actors"],
              "visits": config["visits"], "seeds": seeds, "champions": champions}
    if path.exists():
        result = read(path)
        if any(result.get(key) != value for key, value in header.items()) or set(result["arms"]) != set(ARMS):
            raise ValueError(f"Holdout does not match the sealed development champions in {path}")
    else:
        result = {**header, "arms": {}}
        evaluator = evaluator_for("audit")
        for arm in ARMS:
            checkpoint = _checkpoint(_directory(out, arm, generation), arm, generation)
            winners = checkpoint["champions"]
            ids = [champions[arm][str(island)] for island in range(config["islands"])]
            if [item["row"]["id"] for item in winners] != ids:
                raise ValueError(f"Holdout checkpoint champions differ for {arm}, generation {generation}")
            per_island = {str(island): [] for island in range(config["islands"])}
            for index, seed in enumerate(seeds):
                publish("holdout", arm, audit_seeds_completed=index, audit_seeds_total=count)
                journal = root / f"{generation:04d}" / arm / f"{seed}.json"
                request = {"ids": ids, "seed": seed, "arm": arm, "generation": generation}
                if journal.exists():
                    entry = read(journal)
                    if any(entry.get(key) != value for key, value in request.items()):
                        raise ValueError(f"Holdout evaluation does not match its request at {journal}")
                    scores = _score_rows(entry["scores"], len(ids))
                else:
                    scores = _score_rows(evaluator([item["genome"] for item in winners], arm=arm, seed=seed,
                                                   actors=config["actors"], visits=config["visits"]), len(ids))
                    _write(journal, {**request, "scores": scores})
                for island, score in zip(per_island, scores, strict=True):
                    per_island[island].append({"seed": seed, **score})
            result["arms"][arm] = {"mean": _mean(without_seed([score for rows in per_island.values() for score in rows])),
                                   "islands": {island: {"candidate": champions[arm][island],
                                                        "mean": _mean(without_seed(rows)), "paired_scores": rows}
                                               for island, rows in per_island.items()}}
        _write(path, result)

    latest_path = out / "holdout.json"
    if not latest_path.exists() or read(latest_path)["generation"] <= generation:
        _write(latest_path, result)
    if generation in config["receipt_generations"]:
        from .local_rule_receipt import audit_receipts

        for arm in ARMS:
            publish("receipt_audit", arm, audit_seeds_completed=count, audit_seeds_total=count)
            checkpoint = _checkpoint(_directory(out, arm, generation), arm, generation)
            by_id = {item["row"]["id"]: item["genome"] for item in checkpoint["champions"]}
            audit_receipts(out, generation, arm, by_id, seed_count=config["receipt_seed_counts"]["final" if final else "first"],
                           device=config["device"])


def train(out, config, warm, *, batch_limit=0):
    from .local_rule_evaluator import Evaluation

    history, latest, previous = _recover(out, config)
    _write(out / "history.json", history)
    population = config["population"]
    generations = config["generations"]
    started = _utc()
    prior_progress = out / "progress.json"
    if prior_progress.exists():
        started = read(prior_progress)["started_utc"]
    _write(out / "process.json", {"pid": os.getpid(), "started_utc": _utc(), "device": config["device"],
                                  "run_dir": str(out)})
    status = {"schema": SCHEMA, "state": "running", "phase": "search", "generation": 1,
              "generations": generations, "population": population, "generation_total": population,
              "generation_completed": 0, "current_arm": ARMS[0], "device": config["device"],
              "started_utc": started, "total_candidates": len(ARMS) * population * generations,
              "latest": None, "arms_progress": {}}
    before = time.perf_counter()
    counters = {arm: population * row.get("generation", 0) for arm, row in latest.items()}
    new_batches = new_candidates = 0
    evaluators = {}

    def evaluator_for(kind):
        if kind not in evaluators:
            evaluators[kind] = Evaluation(config["device"], kind)
        return evaluators[kind]

    def progress(**fields):
        status.update(fields, updated_utc=_utc(), completed_candidates=sum(counters.values()),
                      candidates_per_minute=60 * new_candidates / max(.001, time.perf_counter() - before),
                      arms_progress={arm: {"sealed_generations": latest[arm].get("generation", 0),
                                           "candidates": counters[arm], "latest": latest[arm] or None}
                                     for arm in ARMS})
        _write(out / "progress.json", status)

    sealed = {(row["generation"], row["arm"]): row for row in history}
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="birth-prefetch") as pool:
        try:
            for generation in range(1, generations + 1):
                for arm in ARMS:
                    if (generation, arm) in sealed:
                        continue
                    folder = _directory(out, arm, generation)
                    entries = genetics.plan(config, generation, previous[arm], warm["descriptors"])
                    load_or_write(folder / "plan.json", entries)
                    if generation == 1 and arm != ARMS[0]:
                        first_plan = _directory(out, ARMS[0], 1) / "plan.json"
                        if read(first_plan) != entries:
                            raise ValueError(f"Shared founder plan differs in {arm}")
                    seed = config["seed"] + config["development_seed_offset"] + generation
                    rows = []
                    progress(phase="search", current_arm=arm, generation=generation, generation_completed=0,
                             latest=latest[arm] or None, development_seed=seed, audit_generation=None,
                             audit_seeds_completed=None, audit_seeds_total=None)
                    next_progress = time.perf_counter()
                    for path, group, genomes in _batches(folder, entries, config["batch_size"], previous[arm],
                                                        warm["genomes"], pool):
                        if genomes is None:
                            batch = read_batch(path, group)
                            scores = _score_rows([row["metrics"] for row in batch], len(group))
                        else:
                            evaluate = evaluator_for("training")
                            scores = []
                            width = config["eval_batch_size"]
                            for offset in range(0, len(genomes), width):
                                chunk = genomes[offset:offset + width]
                                values = evaluate(chunk, arm=arm, seed=seed, actors=config["actors"], visits=config["visits"])
                                scores.extend(_score_rows(values, len(chunk)))
                            record_batch(path, [{"id": row["id"], "metrics": metrics}
                                                for row, metrics in zip(group, scores, strict=True)])
                            new_batches += 1
                            new_candidates += len(group)
                        rows.extend(_training_rows(scores, group))
                        counters[arm] = (generation - 1) * population + len(rows)
                        now = time.perf_counter()
                        pause = batch_limit and new_batches >= batch_limit
                        if now >= next_progress or len(rows) == population or pause:
                            progress(generation_completed=len(rows), latest=_summary(rows, config["islands"]))
                            next_progress = now + .75
                        if pause:
                            progress(state="paused", phase="paused")
                            return

                    checkpoint = genetics.select(config, generation, rows, previous[arm], warm["genomes"])
                    _save(folder / "elites.pt", {"arm": arm, "generation": generation, **checkpoint})
                    summary = {"arm": arm, "generation": generation, "development_seed": seed,
                               **_summary(rows, config["islands"]), "perturbation": checkpoint["perturbation_summary"]}
                    _write(folder / "summary.json", summary)
                    sealed[generation, arm] = latest[arm] = summary
                    previous[arm] = checkpoint
                    history.append(summary)
                    _write(out / "history.json", history)
                    progress(latest=summary)
                    print(f"generation={generation} arm={arm} candidates={counters[arm]} "
                          f"best={summary['champion']['selection_score']:.5f} "
                          f"mean={summary['mean']['selection_score']:.5f}", flush=True)

                _audit(out, config, generation, {arm: sealed[generation, arm] for arm in ARMS}, evaluator_for, progress)

            champions = {arm: previous[arm]["champions"] for arm in ARMS}
            _save(out / "champions.pt", {"generation": generations, "arms": champions})
            from .local_final_comparison import compare_final

            compare_final(out, config, evaluator_for, progress)
            progress(state="complete", phase="complete", generation=generations, generation_completed=population,
                     current_arm=None, audit_generation=generations, audit_seeds_completed=config["final_audit_seeds"],
                     audit_seeds_total=config["final_audit_seeds"], latest=None)
        except (Exception, KeyboardInterrupt) as error:
            state = "paused" if isinstance(error, KeyboardInterrupt) else "failed"
            progress(state=state, phase=state, failure=f"{type(error).__name__}: {error}")
            raise
        finally:
            for evaluator in evaluators.values():
                if hasattr(evaluator, "clear_cache"):
                    evaluator.clear_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--population", type=int, default=4096)
    parser.add_argument("--warm", type=int)
    parser.add_argument("--generations", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--audit-every", type=int, default=32)
    parser.add_argument("--audit-seeds", type=int, default=8)
    parser.add_argument("--final-audit-seeds", type=int, default=32)
    parser.add_argument("--perturbation-pairs", type=int)
    parser.add_argument("--seed", type=int, default=20_261_119)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-limit", type=int, default=0, help="Pause after this many newly recorded training batches.")
    parser.add_argument("--prepare-only", action="store_true", help="Validate and seal the shared birth bank and run protocol.")
    args = parser.parse_args(argv)
    args.out, args.archive = args.out.resolve(), args.archive.resolve()
    if args.warm is None:
        args.warm = args.population // 2
    if (args.population < 16 * directed.ISLANDS or args.population % directed.ISLANDS
            or not 0 <= args.warm <= min(2048, args.population) or args.warm % directed.ISLANDS
            or not 1 <= args.generations <= 128 or args.seed < 0 or args.batch_limit < 0
            or min(args.batch_size, args.eval_batch_size, args.audit_every, args.audit_seeds, args.final_audit_seeds) < 1
            or args.eval_batch_size > args.batch_size):
        parser.error("Use four equally sized islands, a valid warm quota, budgets above zero, and 1–128 generations")
    available = exploration.protocol(args.population, directed.ISLANDS)["ordinary_offspring_per_island"]["local"] // 2
    if args.perturbation_pairs is None:
        args.perturbation_pairs = min(4, available)
    if not 0 <= args.perturbation_pairs <= available:
        parser.error("Perturbation pairs must fit inside the ordinary local slots in each island")
    if not args.prepare_only and args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    torch.set_num_threads(1)
    with _lock(args.out / ".lock"):
        warm, warm_hash = _warm_bank(args)
        config = load_or_write(args.out / "protocol.json", _protocol(args, warm["provenance"], warm_hash))
        if args.prepare_only:
            print(f"sealed {args.warm} distinct warm founders and {config['cold_founders']} cold slots at {args.out}", flush=True)
        else:
            train(args.out, config, warm, batch_limit=args.batch_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
