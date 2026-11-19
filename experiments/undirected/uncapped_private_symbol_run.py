"""Run paired searches on the version-free private A/B symbol task."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import os
from pathlib import Path
from statistics import fmean
import time

import torch

from experiments.seed_evolution.large_runner import _lock, _save, _utc, _write
from . import uncapped_directed_reward as backend
from . import uncapped_plasticity_run as runner
from . import uncapped_private_symbol_breeding as breeding
from . import uncapped_private_symbol_world as world
from .uncapped_directed_compact_execution_run import CompactEvaluation
from .uncapped_directed_reward_run import NODES, SCORING, rank, score_fields as prior_score_fields
from .uncapped_private_symbol_audit import audit
from .uncapped_private_symbol_delta import delta_rollout, sample_deltas


ARCHIVE = runner.OUT.with_name("2026-11-17-plasticity-directed-reward-v1")
OUT = ARCHIVE.with_name("2026-11-18-private-symbol-paired-v1")
ARMS = ("downhill", "nondecreasing")
ISLANDS, ACTORS, VISITS = 4, 16, 8
INHERITANCE_BLEND = .25
SOURCE_NAMES = (
    "uncapped_private_symbol_run.py", "uncapped_private_symbol_world.py", "uncapped_private_symbol_breeding.py",
    "uncapped_private_symbol_delta.py", "uncapped_private_symbol_audit.py",
    "uncapped_directed_exploration.py", "uncapped_directed_breeding.py", "uncapped_directed_reward.py",
    "uncapped_directed_compact_execution_run.py", "uncapped_directed_reward_cuda_compact.cu",
    "uncapped_directed_node_fused.cu", "uncapped_directed_support_fused.cu", "uncapped_directed_world_fused.cu",
    "uncapped_directed_packed.py", "plasticity_continuous_world.py", "plasticity_roomy_world.py",
    "uncapped_directed_reward_cuda_compact.py", "uncapped_directed_world_fused.py",
    "uncapped_directed_deep_execution_run.py", "uncapped_directed_prepared_run.py", "_cuda_nvrtc.py",
)


class PrivateEvaluation(CompactEvaluation):
    """Use the compact FP16 policy with private-symbol world rewards and stats."""

    def __init__(self, schedule, *, rollout=world.task_rollout, policy_type=None):
        super().__init__(schedule, fp16_accumulation=False)
        self.reference = backend.RewardEvaluation(schedule) if schedule is not world.training_schedule or policy_type else None
        self.reference_policy = policy_type
        self.rollout = rollout

    def __call__(self, *args, **kwargs):
        original = backend._rollout
        original_policy = backend._DirectedRewardPolicy
        backend._rollout = self.rollout
        try:
            if self.reference is not None:
                if self.reference_policy is not None:
                    backend._DirectedRewardPolicy = self.reference_policy
                return self.reference(*args, **kwargs)
            return super().__call__(*args, **kwargs)
        finally:
            backend._rollout = original
            backend._DirectedRewardPolicy = original_policy

    def clear_cache(self):
        super().clear_cache()
        if self.reference is not None:
            self.reference.clear_cache()


def score_fields(row, pairs):
    # Keep the public selector interface. These aliases refer to visits 5–8.
    selection_row = {
        **row, "first_finish_correct_a_first_return": row["symbol_late_a"],
        "first_finish_correct_b_first_return": row["symbol_late_b"],
    }
    result = prior_score_fields(selection_row, pairs)
    return {**result, "first_finish_correct_a_first_return": row["first_finish_correct_a_first_return"],
            "first_finish_correct_b_first_return": row["first_finish_correct_b_first_return"]}


def summarize(rows):
    best = max(rows, key=rank)
    return {**{key: best[key] for key in ("strict", "completion", "strict_return", "selection_score", "symbol_late_mean", "symbol_late_a",
                                          "symbol_late_b", "symbol_first_return_mean", "ready_coverage",
                                          "premature_request_rate", "idle")},
            "best_candidate": best["id"], "best_fitness": best["strict"],
            "mean_fitness": fmean(row["strict"] for row in rows),
            "mean_symbol_late": fmean(row["symbol_late_mean"] for row in rows)}


def inheritance_summary(rows, checkpoint):
    pairs = {}
    for row in rows:
        trial = row.get("inheritance_trial")
        if trial:
            pairs.setdefault(trial["pair"], []).append(row)
    contrasts = []
    for pair, group in pairs.items():
        if len(group) != 2:
            raise ValueError(f"Incomplete inheritance pair {pair}")
        plain, inherited = sorted(group, key=lambda row: row["inheritance_trial"]["fraction"])
        if (plain["seed"] != inherited["seed"] or plain["parent"] != inherited["parent"]
                or plain["island"] != inherited["island"]):
            raise ValueError(f"Inheritance pair {pair} was not matched")
        contrasts.append({"control": plain["id"], "inherited": inherited["id"],
                          "delta_late": inherited["symbol_late_mean"] - plain["symbol_late_mean"],
                          "delta_strict": inherited["strict"] - plain["strict"],
                          "delta_score": inherited["selection_score"] - plain["selection_score"],
                          "selected_control": plain["id"] in checkpoint["genomes"],
                          "selected_inherited": inherited["id"] in checkpoint["genomes"]})
    return {"pairs": len(contrasts), "mean_delta_late": fmean(x["delta_late"] for x in contrasts) if contrasts else None,
            "mean_delta_score": fmean(x["delta_score"] for x in contrasts) if contrasts else None,
            "retained_control": sum(x["selected_control"] for x in contrasts),
            "retained_inherited": sum(x["selected_inherited"] for x in contrasts), "paired": contrasts}


class RunArm:
    def __init__(self, out, config, arm, seed, pair, warm_bank):
        self.out, self.arm, self.pair = out, arm, pair
        self.cfg = {**config, "allow_downhill": arm == "downhill", "seed": seed}
        self.warm_bank = warm_bank
        runner.base.load_or_write(out / "protocol.json", self.cfg)
        self.history, self.previous = runner.recover(out, self.cfg)
        path = out / "generations" / f"{len(self.history) + 1:04d}" / "batches"
        partial = len(list(path.glob("*.json.gz"))) * config["batch_size"]
        self.completed = len(self.history) * config["population"] + partial


def parent_state(job, generation, entries, directory, delta_evaluator):
    parent_ids = sorted({entry["inheritance_trial"]["learned_parent"] for entry in entries
                         if entry.get("inheritance_trial")})
    if not parent_ids:
        return job.previous
    path = directory / "parent-deltas.pt"
    seed = job.cfg["seed"] + 200_000_003 + generation
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["seed"] != seed or sorted(saved["deltas"]) != parent_ids:
            raise ValueError(f"Parent delta cache differs: {path}")
        deltas = saved["deltas"]
    else:
        deltas = sample_deltas(delta_evaluator, {ident: job.previous["genomes"][ident] for ident in parent_ids},
                               seed=seed, actors=ACTORS, visits=VISITS, batch_size=min(16, job.cfg["eval_batch_size"]),
                               device=job.cfg["device"])
        _save(path, {"seed": seed, "deltas": deltas})
    return {**job.previous, "learned_deltas": deltas}


def run_generation(job, generation, warm_descriptors, pool, training, delta_evaluator, publish, *, mirror=None):
    cfg = job.cfg
    population = cfg["population"]
    folder = job.out / "generations" / f"{generation:04d}"
    entries = runner.base.load_or_write(folder / "plan.json", breeding.plan(cfg, generation, job.previous,
                                                                           warm_founders=warm_descriptors))
    if mirror is not None and runner.base.read(mirror / "plan.json") != entries:
        raise ValueError("Paired founder plans differ")
    previous = parent_state(job, generation, entries, folder, delta_evaluator)
    seed = cfg["seed"] + 100_000_019 + generation
    rows = []
    next_progress = time.perf_counter()
    execution = None if mirror is not None else runner.batches(folder, entries, cfg, previous, pool, 2)
    if execution is None:
        width = cfg["batch_size"]
        execution = ((folder / "batches" / f"{i:04d}.json.gz", entries[i:i+width], None)
                     for i in range(0, population, width))
    for path, group, genomes in execution:
        if path.is_file():
            batch = runner.read_batch(path, group)
        else:
            if mirror is not None:
                batch = runner.read_batch(mirror / "batches" / path.name, group)
            else:
                evaluated = []
                size = cfg["eval_batch_size"]
                for start in range(0, len(genomes), size):
                    chunk = genomes[start:start+size]
                    metrics = training(chunk, seed=seed, actors=ACTORS, visits=VISITS,
                                       device=cfg["device"], capture=cfg["device"] == "cuda")
                    evaluated.extend(runner.base.unpack_scores(metrics, len(chunk)))
                batch = [{**entry, **score_fields(row, cfg["edge_pairs"])}
                         for entry, row in zip(group, evaluated, strict=True)]
            runner.record_batch(path, batch)
            job.completed += len(batch)
        rows.extend(batch)
        now = time.perf_counter()
        if now >= next_progress or len(rows) == population:
            publish(job, generation, len(rows), phase="search", latest=summarize(rows))
            next_progress = now + .75
    checkpoint = breeding.elites(cfg, rows, previous, warm_genomes=job.warm_bank)
    matched = inheritance_summary(rows, checkpoint)
    _write(folder / "inheritance.json", matched)
    compact_matched = {key: val for key, val in matched.items() if key != "paired"}
    summary = {**summarize(rows), "generation": generation, "candidates": population, "development_seed": seed,
               "inheritance_summary": compact_matched, "exploration_summary": checkpoint["exploration_summary"]}
    _save(folder / "elites.pt", checkpoint)
    _write(folder / "summary.json", summary)
    job.previous = checkpoint
    job.history.append(summary)
    _write(job.out / "history.json", job.history)
    print(f"pair={job.pair:02d} arm={job.arm} generation={generation} total={job.completed} "
          f"late={summary['symbol_late_mean']:.4f} strict={summary['strict']:.4f} "
          f"inheritance_pairs={matched['pairs']} delta={matched['mean_delta_late']}", flush=True)


def summarize_paired_audits(out, jobs, generation):
    paths = [job.out / "audits" / f"{generation:04d}.json" for job in jobs]
    if not all(path.is_file() for path in paths):
        return
    metrics = ("symbol_late_mean", "symbol_late_a", "symbol_late_b", "symbol_first_return_mean", "strict")
    pairs = []
    for index in range(0, len(jobs), len(ARMS)):
        left, right = (runner.base.read(path) for path in paths[index:index+len(ARMS)])
        if left["seeds"] != right["seeds"]:
            raise ValueError("Search arms used different holdout worlds")
        rows = {arm: result["conditions"]["regular_native"]["modes"]["full"]["aggregate"]["means"]
                for arm, result in zip(ARMS, (left, right), strict=True)}
        pairs.append({"pair": jobs[index].pair, "training_seed": jobs[index].cfg["seed"],
                      "arm_means": {arm: {name: row[name] for name in metrics} for arm, row in rows.items()},
                      "downhill_minus_nondecreasing": {name: rows[ARMS[0]][name] - rows[ARMS[1]][name]
                                                       for name in metrics}})
    _write(out / "paired-audits" / f"{generation:04d}.json", {
        "generation": generation, "search_pairs": pairs,
        "mean_paired_delta": {name: fmean(pair["downhill_minus_nondecreasing"][name] for pair in pairs)
                              for name in metrics},
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--population", type=int, default=4096)
    parser.add_argument("--generations", type=int, default=1024)
    parser.add_argument("--warm", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--journal-size", type=int, default=64)
    parser.add_argument("--inheritance-pairs-per-island", type=int, default=8)
    parser.add_argument("--audit-every", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--audit-seeds", type=int, default=8)
    parser.add_argument("--final-audit-seeds", type=int, default=32)
    args = parser.parse_args(argv)
    if (min(args.pairs, args.generations, args.population, args.eval_batch_size, args.journal_size,
            args.audit_every, args.audit_seeds, args.final_audit_seeds) < 1 or args.population < 64 or args.population % ISLANDS
            or args.population % args.journal_size or args.journal_size % args.eval_batch_size
            or not 0 <= args.warm <= args.population or args.warm % ISLANDS):
        parser.error("Expected four equal islands and compatible evaluation, journal and warm sizes")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    torch.set_num_threads(1)
    out = args.out.resolve()
    with _lock(out / ".lock"):
        warm_descriptors, warm_bank, provenance = breeding.load_warm(args.archive.resolve(), args.warm, 20261118)
        config = {
            "schema": 1, "experiment_kind": "uncapped_continuous", "task": "private_symbol",
            "population": args.population, "generations": args.generations, "pairs": args.pairs, "islands": ISLANDS,
            "nodes": NODES, "edge_pairs": NODES*(NODES-1), "actors": ACTORS, "visits": VISITS,
            "batch_size": args.journal_size, "eval_batch_size": args.eval_batch_size, "device": args.device,
            "audit_every": args.audit_every, "audit_seeds": args.audit_seeds, "audit_actors": ACTORS,
            "final_audit_seeds": args.final_audit_seeds,
            "inheritance_pairs_per_island": args.inheritance_pairs_per_island,
            "reply_window": world.TIMING, "precision": "FP16 with FP32 matrix accumulation",
            "warm_start": provenance, "selection_score": {**SCORING, "return_range": "visits 5 through 8"},
            "downhill_arms": {"downhill": .01, "nondecreasing": 0.0},
            "inheritance": {"ordinary_local_pairs_per_island": args.inheritance_pairs_per_island, "blend": INHERITANCE_BLEND,
                            "state": "mean final-minus-birth weight on original surviving arcs over 16 separate maps"},
            "sources": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in SOURCE_NAMES},
        }
        runner.base.load_or_write(out / "protocol.json", config)
        jobs = [RunArm(out / f"pair-{pair:02d}" / arm, config, arm, 20261180 + 1_000_000*pair, pair, warm_bank)
                for pair in range(1, args.pairs + 1) for arm in ARMS]
        training = PrivateEvaluation(world.training_schedule)
        delta_evaluator = PrivateEvaluation(world.training_schedule, rollout=delta_rollout)
        original_candidate, original_unpack = runner.candidate, runner.base.unpack_scores
        from .uncapped_directed_packed import unpack_scores_packed
        runner.candidate = partial(breeding.candidate, warm_genomes=warm_bank)
        runner.base.unpack_scores = partial(unpack_scores_packed, unpack_scores=original_unpack)
        total = args.population * args.generations * len(jobs)
        at_start = sum(job.completed for job in jobs)
        started = _utc()
        clock = time.perf_counter()
        status = {"state": "running", "experiment_kind": "uncapped_continuous", "device": args.device,
                  "total_candidates": total, "generations": args.generations, "started_utc": started,
                  "population": args.population, "generation_total": args.population,
                  "execution_precision": "fp16 / fp32 matmul accumulation"}

        def publish(job, gen, committed, *, phase, **fields):
            done = sum(arm.completed for arm in jobs)
            elapsed = time.perf_counter() - clock
            counts = {f"{arm.pair:02d}/{arm.arm}": {"sealed_generations": len(arm.history), "candidates": arm.completed}
                      for arm in jobs}
            latest = fields.pop("latest", {})
            status.update(**fields, **latest, phase=phase, generation=gen, generation_completed=committed,
                          completed_candidates=done, updated_utc=_utc(), current_pair=job.pair, current_arm=job.arm,
                          arms_progress=counts, candidates_per_minute=60*(done-at_start)/max(elapsed, .001))
            _write(out / "progress.json", status)

        def audit_if_due(job, generation):
            if generation not in (1, args.generations) and generation % args.audit_every:
                return
            audit_path = job.out / "audits" / f"{generation:04d}.json"
            if audit_path.is_file():
                return
            publish(job, generation, args.population, phase="holdout")
            audit_config = {**job.cfg, "audit_seeds": args.final_audit_seeds} if generation == args.generations else job.cfg
            result = audit(job.out, generation, audit_config, job.previous, PrivateEvaluation, rank=rank)
            _write(out / "holdout.json", result)

        _write(out / "process.json", {"pid": os.getpid(), "run_dir": str(out), "device": args.device,
                                      "started_utc": started, "prefetch": 2})
        try:
            for job in jobs:
                if job.history:
                    audit_if_due(job, len(job.history))
            with ThreadPoolExecutor(max_workers=1) as pool:
                first = min(len(job.history) for job in jobs) + 1
                for generation in range(first, args.generations + 1):
                    for index, job in enumerate(jobs):
                        if len(job.history) >= generation:
                            continue
                        if len(job.history) != generation - 1:
                            raise ValueError("Checkpoints are not contiguous")
                        mirror = jobs[index-1].out / "generations" / "0001" if generation == 1 and index % 2 else None
                        publish(job, generation, 0, phase="search")
                        run_generation(job, generation, warm_descriptors, pool, training, delta_evaluator, publish,
                                       mirror=mirror)
                        audit_if_due(job, generation)
                    if generation in (1, args.generations) or generation % args.audit_every == 0:
                        summarize_paired_audits(out, jobs, generation)
            publish(jobs[-1], args.generations, args.population, phase="complete", state="complete")
        except (Exception, KeyboardInterrupt) as error:
            state = "paused" if isinstance(error, KeyboardInterrupt) else "failed"
            status.update(state=state, phase=state, failure=f"{type(error).__name__}: {error}", updated_utc=_utc())
            _write(out / "progress.json", status)
            raise
        finally:
            runner.candidate, runner.base.unpack_scores = original_candidate, original_unpack


if __name__ == "__main__":
    main()
