"""Continue private-symbol evolution with independent A/B requests in one arm."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import os
from pathlib import Path
import time

import torch

from experiments.seed_evolution.large_runner import _lock, _save, _utc, _write
from . import uncapped_private_symbol_breeding as breeding
from . import uncapped_private_symbol_iid_founders as founders
from . import uncapped_private_symbol_iid_world as world
from . import uncapped_private_symbol_run as previous_run
from . import uncapped_private_symbol_selection as selection
from .evolved_plasticity_world import N_SYMBOLS
from .uncapped_directed_audit import _save_champions
from .uncapped_directed_exploration import TEMPERATURE
from .uncapped_private_symbol_audit import _aggregate, _means
from .uncapped_private_symbol_fast_run import unpack_scores_lists
from .uncapped_private_symbol_pipeline_run import _PendingCommit, _record_batch
from .uncapped_private_symbol_post_answer import post_answer_actions
from .uncapped_private_symbol_span_run import MainSpanEvaluation
from .uncapped_island_crossover_run import block_ticks


OUT = previous_run.OUT.with_name("2026-11-18-private-symbol-iid-v1")
ACTORS, VISITS, SEED = 8, 16, 61_118_501
REQUESTS = ACTORS * VISITS * N_SYMBOLS
RULE = selection.POST_ANSWER_RULE
SOURCE_NAMES = (*previous_run.SOURCE_NAMES,
                "uncapped_private_symbol_iid_run.py", "uncapped_private_symbol_iid_world.py",
                "uncapped_private_symbol_iid_founders.py", "uncapped_private_symbol_selection.py",
                "uncapped_private_symbol_span.py", "uncapped_private_symbol_span.cu",
                "uncapped_private_symbol_post_answer.py", "uncapped_private_symbol_post_answer.cu",
                "uncapped_private_symbol_span_run.py", "uncapped_private_symbol_pipeline_run.py",
                "uncapped_private_symbol_fast_run.py", "uncapped_directed_support_rebuild.py")


def _rollout(policy, tape, table, *, training):
    metrics, actions = world.observed_rollout(policy, tape, table, training=training)
    metrics["post_answer_actions"] = post_answer_actions(actions, tape)
    return metrics


def _training_evaluator():
    # The sealed evaluator selects its CUDA span path by the training-schedule
    # identity at construction. The returned evaluator caches the IID tape.
    original = previous_run.world.training_schedule
    previous_run.world.training_schedule = world.training_schedule
    try:
        return MainSpanEvaluation(world.training_schedule, rollout=partial(_rollout, training=True))
    finally:
        previous_run.world.training_schedule = original


def _audit(job, generation, config, publish):
    path = job.out / "audits" / f"{generation:04d}.json"
    if generation not in (1, config["generations"]) and generation % config["audit_every"]:
        return
    if path.is_file():
        return
    final = generation == config["generations"]
    first = 96_900_000 if final else 96_800_000
    count = config["final_audit_seeds"] if final else config["audit_seeds"]
    seeds = list(range(first, first + count))
    winners = [max(pool, key=previous_run.rank) for pool in job.previous["islands"]]
    best = max(winners, key=previous_run.rank)
    genomes = [job.previous["genomes"][row["id"]] for row in winners]
    publish(generation, config["population"], phase="holdout", latest=job.history[-1])
    evaluator = previous_run.PrivateEvaluation(world.schedule, rollout=partial(_rollout, training=False))
    rows = {str(i): [] for i in range(len(genomes))}
    try:
        for seed in seeds:
            metrics = evaluator(genomes, seed=seed, actors=ACTORS, visits=VISITS, device=config["device"],
                                mode="full", capture=config["device"] == "cuda")
            for island, row in zip(rows, unpack_scores_lists(metrics, len(genomes)), strict=True):
                rows[island].append({"seed": seed, **row})
    finally:
        evaluator.clear_cache()
    report = {
        "generation": generation, "final": final, "seeds": seeds, "actors": ACTORS, "visits": VISITS,
        "candidate": best["id"], "champions": {str(i): row["id"] for i, row in enumerate(winners)},
        "conditions": {"regular_native": {"timing": {**world.TIMING, "ticks_per_request": block_ticks(world.TIMING)},
                       "modes": {"full": {"aggregate": _aggregate(rows), "islands": {
                           island: {"means": _means(scores), "paired_scores": scores}
                           for island, scores in rows.items()}}}}},
    }
    if final:
        _save_champions(job.out, winners, best, job.previous)
    _write(path, report)
    _write(job.out.parents[1] / "holdout.json", report)


def _generation(job, generation, warm, evaluator, pool, journal, publish):
    config = job.cfg
    population = config["population"]
    width = config["eval_batch_size"]
    folder = job.out / "generations" / f"{generation:04d}"
    runner = previous_run.runner
    entries = runner.base.load_or_write(folder / "plan.json", founders.plan(config, generation, job.previous,
                                                                            warm_founders=warm))
    seed = SEED + 100_000_019 + generation
    rows = []
    next_progress = time.perf_counter()
    for path, group, genomes in runner.batches(folder, entries, config, job.previous, pool, 2):
        if path.is_file():
            batch = runner.read_batch(path, group)
            selection.validate_rule(batch, RULE, path)
        else:
            scores = []
            for start in range(0, len(genomes), width):
                chunk = genomes[start:start + width]
                fields = evaluator(chunk, seed=seed, actors=ACTORS, visits=VISITS,
                                   device=config["device"], capture=config["device"] == "cuda")
                scores.extend(unpack_scores_lists(fields, len(chunk)))
            batch = [{**entry, **selection.score_fields(previous_run.score_fields(row, config["edge_pairs"]),
                                                        REQUESTS, RULE)}
                     for entry, row in zip(group, scores, strict=True)]
            journal.submit(job, _record_batch, path, batch)
        rows.extend(batch)
        now = time.perf_counter()
        if now >= next_progress or len(rows) == population:
            journal.drain()
            publish(generation, len(rows), phase="search", latest=selection.summarize(previous_run.summarize, rows))
            next_progress = now + .75
    journal.drain()
    checkpoint = breeding.elites(config, rows, job.previous, warm_genomes=job.warm_bank)
    summary = {**selection.summarize(previous_run.summarize, rows), "generation": generation,
               "candidates": population, "development_seed": seed, "exploration_summary": checkpoint["exploration_summary"]}
    _save(folder / "elites.pt", checkpoint)
    _write(folder / "summary.json", summary)
    job.previous = checkpoint
    job.history.append(summary)
    _write(job.out / "history.json", job.history)
    print(f"generation={generation} total={job.completed} score={summary['selection_score']:.5f} "
          f"strict={summary['strict']:.4f} late={summary['symbol_late_mean']:.4f} "
          f"post_answer_actions={summary['post_answer_actions']}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--archive", type=Path, default=previous_run.OUT)
    parser.add_argument("--population", type=int, default=4096)
    parser.add_argument("--warm", type=int, default=2048)
    parser.add_argument("--generations", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if (args.population < 64 or args.population % 64 or args.warm != args.population // 2
            or args.generations < 1 or args.eval_batch_size < 1 or 64 % args.eval_batch_size):
        parser.error("Use a population divisible by 64, half archived founders, and an evaluation width dividing 64")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    torch.set_num_threads(1)
    out = args.out.resolve()
    with _lock(out / ".lock"):
        warm, bank, provenance = founders.load_warm(args.archive, args.warm, SEED)
        config = {
            "schema": 1, "experiment_kind": "uncapped_continuous", "task": "private_symbol",
            "population": args.population, "generations": args.generations, "pairs": 1,
            "arms": ["downhill"], "islands": previous_run.ISLANDS,
            "nodes": previous_run.NODES, "edge_pairs": previous_run.NODES * (previous_run.NODES - 1),
            "actors": ACTORS, "visits": VISITS, "requests_per_visit": N_SYMBOLS, "context_sampling": "iid_per_request",
            "device": args.device, "batch_size": 64, "eval_batch_size": args.eval_batch_size,
            "audit_every": 64, "audit_seeds": 8, "final_audit_seeds": 32, "inheritance_pairs_per_island": 0,
            "reply_window": world.TIMING, "precision": "FP16 with FP32 matrix accumulation",
            "warm_start": provenance, "downhill_arms": {"downhill": TEMPERATURE},
            "selection_score": {"rule": RULE, "strict_points_per_request": 1, "recovered_points_per_request": .1,
                                "post_answer_penalty_per_action": .1, "requests_per_candidate": REQUESTS},
            "sources": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in SOURCE_NAMES},
        }
        previous_run.runner.base.load_or_write(out / "protocol.json", config)
        job = previous_run.RunArm(out / "pair-01" / "downhill", config, "downhill", SEED, 1, bank)
        if job.history:
            selection.validate_rule(job.history[-1:], RULE, job.out)
        evaluator = _training_evaluator()
        started, before, clock = _utc(), job.completed, time.perf_counter()
        status = {"state": "running", "experiment_kind": config["experiment_kind"], "device": args.device,
                  "total_candidates": args.population * args.generations, "generations": args.generations,
                  "population": args.population, "generation_total": args.population, "started_utc": started,
                  "execution_precision": "fp16 / fp32 matmul accumulation", "selection_rule": RULE}

        def publish(generation, committed, *, phase, latest=None, **fields):
            status.update(latest or {})
            status.update(fields, phase=phase, generation=generation, generation_completed=committed,
                          current_pair=1, current_arm="downhill", completed_candidates=job.completed,
                          arms_progress={"01/downhill": {"sealed_generations": len(job.history), "candidates": job.completed}},
                          candidates_per_minute=60 * (job.completed - before) / max(.001, time.perf_counter() - clock),
                          updated_utc=_utc())
            _write(out / "progress.json", status)

        _write(out / "process.json", {"pid": os.getpid(), "run_dir": str(out), "device": args.device,
                                      "started_utc": started, "prefetch": 2})
        original = previous_run.runner.candidate
        previous_run.runner.candidate = partial(breeding.candidate, warm_genomes=bank)
        with (ThreadPoolExecutor(max_workers=1, thread_name_prefix="birth-prefetch") as pool,
              ThreadPoolExecutor(max_workers=1, thread_name_prefix="candidate-journal") as writer):
            journal = _PendingCommit(writer)
            try:
                if job.history:
                    _audit(job, len(job.history), config, publish)
                for generation in range(len(job.history) + 1, args.generations + 1):
                    publish(generation, 0, phase="search")
                    _generation(job, generation, warm, evaluator, pool, journal, publish)
                    _audit(job, generation, config, publish)
                publish(args.generations, args.population, phase="complete", state="complete", latest=job.history[-1])
            except (Exception, KeyboardInterrupt) as error:
                try:
                    journal.drain()
                except Exception as commit_error:
                    error.add_note(f"Pending journal commit failed: {commit_error}")
                state = "paused" if isinstance(error, KeyboardInterrupt) else "failed"
                status.update(state=state, phase=state, failure=f"{type(error).__name__}: {error}", updated_utc=_utc())
                _write(out / "progress.json", status)
                raise
            finally:
                evaluator.clear_cache()
                previous_run.runner.candidate = original


if __name__ == "__main__":
    main()
