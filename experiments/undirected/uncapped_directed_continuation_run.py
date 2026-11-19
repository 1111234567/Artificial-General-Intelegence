"""Continue a sealed checkpoint with four directed islands and protected exploration."""

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
from . import uncapped_directed_exploration as exploration
from . import uncapped_directed_reward_run as original
from . import uncapped_plasticity_run as runner
from .uncapped_directed_continuation_audit import audit
from .uncapped_directed_fused_accumulation_probe import FusedAccumulationEvaluation, _LongAccumulationEvaluation


def migrate(checkpoint):
    pools = [[{**row, "directed": True} for row in pool] for pool in checkpoint["islands"]]
    genomes = {name: {**genome, "directed": torch.ones_like(genome["directed"])}
               for name, genome in checkpoint["genomes"].items()}
    return {**checkpoint, "islands": pools, "genomes": genomes}


def snapshot(rows):
    result = original.runner_snapshot(rows)
    islands = {}
    for index in range(original.ISLANDS):
        group = [row for row in rows if row["island"] == index]
        if group:
            best = max(group, key=original.rank)
            islands[str(index)] = {"candidates": len(group), "best_candidate": best["id"],
                                   "first_return_mean": best["first_return_mean"], "strict": best["strict"],
                                   "mean_first_return": fmean(row["first_return_mean"] for row in group),
                                   "selection_score": best["selection_score"]}
    return {**result, "directed_islands": islands, "mean_first_return": fmean(r["first_return_mean"] for r in rows)}


def stage_protocol(out, config, cutoff, fp16):
    source = Path(__file__).resolve().parent
    names = (Path(__file__).name, "uncapped_directed_exploration.py", "uncapped_directed_continuation_audit.py",
             "uncapped_directed_fused_accumulation_probe.py", "uncapped_directed_reward_cuda_fused.py",
             "uncapped_directed_reward_cuda_fused.cu", "uncapped_directed_reward_fused_run.py", "_cuda_nvrtc.py")
    checkpoint = out / "generations" / f"{cutoff:04d}" / "elites.pt"
    return {"from_generation": cutoff, "first_generation": cutoff + 1, "source_checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "island_topologies": ["directed"] * config["islands"], "precision": "fp16",
            "matmul_accumulation": "fp16" if fp16 else "fp32", "audit_modes": ["full", "no_reward_write", "frozen"],
            "selection_score": original.SCORING, "exploration": exploration.protocol(config["population"], config["islands"]),
            "sources": {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names}}


def continuation_protocol(out, config, cutoff, fp16):
    """Preserve the historical seal while allowing this runner's budget support."""
    path = out / "continuation-protocol.json"
    current = stage_protocol(out, config, cutoff, fp16)
    if not path.is_file():
        _write(path, current)
        return current
    sealed = runner.base.read(path)
    runner_name = Path(__file__).name
    sealed_hash = sealed.get("sources", {}).get(runner_name)
    if (not isinstance(sealed_hash, str) or len(sealed_hash) != 64
            or any(char not in "0123456789abcdef" for char in sealed_hash)):
        raise ValueError(f"Invalid historical runner source hash: {path}")
    expected = {**current, "sources": {**current["sources"],
                                     runner_name: sealed_hash}}
    if sealed != expected:
        raise ValueError(f"Checkpoint configuration differs: {path}")
    return sealed


def training_budget(out, config, generations=None):
    """Validate a separate total budget; the caller persists it under .lock."""
    def positive(value, name):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return value

    original_target = positive(config["generations"], "protocol generations")
    committed = sorted(int(path.parent.name) for path in (out / "generations").glob("*/summary.json")
                       if path.parent.name.isdecimal())
    if any(generation != index for index, generation in enumerate(committed, 1)):
        raise ValueError("Committed generation history is not contiguous")
    path = out / "training-budget.json"
    saved = runner.base.read(path) if path.is_file() else None
    current_target = original_target
    if saved is not None:
        if (not isinstance(saved, dict) or type(saved.get("schema")) is not int or saved["schema"] != 1
                or type(saved.get("original_generations")) is not int
                or saved["original_generations"] != original_target):
            raise ValueError(f"Training budget does not match the sealed protocol: {path}")
        current_target = positive(saved.get("generations"), "persisted generations")
        if current_target < original_target or current_target < len(committed):
            raise ValueError("Persisted training budget is below the sealed plan or committed history")
    target = current_target if generations is None else positive(generations, "generations")
    if target < max(original_target, current_target, len(committed)):
        raise ValueError("Training budget cannot decrease or fall below committed history")
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if saved is not None and target == current_target and saved.get("runner_sha256") == source_hash:
        return saved
    return {"schema": 1, "original_generations": original_target, "generations": target,
            "runner_sha256": source_hash, "updated_utc": _utc()}


def evolve(out, config, stage, prefetch):
    population = config["population"]
    history, previous = runner.recover(out, config)
    cutoff = stage["from_generation"]
    if not cutoff <= len(history) <= config["generations"]:
        raise ValueError("Continuation source generation is not committed")
    if len(history) == cutoff:
        previous = migrate(previous)
    elif not previous.get("exploration"):
        raise ValueError("Checkpoint lacks the exploration state for this continuation")
    progress_path = out / "progress.json"
    saved = runner.base.read(progress_path)
    if saved.get("state") == "complete" and len(history) == config["generations"]:
        print("Experiment already complete", flush=True)
        return
    accum = stage["matmul_accumulation"]
    training = FusedAccumulationEvaluation(original.training_schedule, fp16_accumulation=accum == "fp16")
    eval_options = {"fp16_accumulation": accum == "fp16"}
    regular = FusedAccumulationEvaluation(partial(original.schedule, timing=original.TIMING), **eval_options)
    delayed = _LongAccumulationEvaluation(partial(original.schedule, timing=original.DELAYED), **eval_options)
    padded = _LongAccumulationEvaluation(partial(original.schedule, timing=original.TIMING,
                                                  pad_to=original.block_ticks(original.DELAYED)), **eval_options)
    status = {**history[-1], "state": "running", "phase": "search", "experiment_kind": config["experiment_kind"],
              "device": config["device"], "started_utc": saved["started_utc"], "population": population,
              "generations": config["generations"], "generation": len(history), "generation_completed": population,
              "generation_total": population, "total_candidates": population * config["generations"],
              "completed_candidates": population * len(history), "continuation_start_generation": cutoff + 1,
              "checkpoint": f"generations/{len(history):04d}/elites.pt",
              "execution_precision": f"fp16 / {accum} matmul accumulation"}
    status.pop("topology_groups", None)
    clock = time.perf_counter()
    evaluated = 0

    def publish(**fields):
        status.update(fields, updated_utc=_utc())
        if evaluated:
            status["candidates_per_minute"] = 60 * evaluated / (time.perf_counter() - clock)
        _write(progress_path, status)

    def run_audit(generation):
        path = out / "audits" / f"{generation:04d}.json"
        if generation < config["generations"] and path.is_file() and runner.base.read(path).get("final"):
            # An earlier completed budget retains its historical final holdout.
            return
        training.clear_cache()
        result = audit(config, out, generation, previous, publish, evaluate=regular, delayed_evaluate=delayed,
                       matched_regular_evaluate=padded, rank=original.rank)
        _write(out / "holdout.json", result)

    def needs_audit(generation):
        return generation in (cutoff + 1, config["generations"]) or generation % config["audit_every"] == 0

    _write(out / "process.json", {"pid": os.getpid(), "started_utc": _utc(), "run_dir": str(out),
                                  "device": config["device"], "prefetch": prefetch})
    print(f"Continuing after generation {cutoff}; all islands directed; accumulation {accum}", flush=True)
    try:
        publish()
        _write(out / "history.json", history)
        if len(history) > cutoff and needs_audit(len(history)):
            run_audit(len(history))
        with ThreadPoolExecutor(max_workers=1) as pool:
            for generation in range(len(history) + 1, config["generations"] + 1):
                folder = out / "generations" / f"{generation:04d}"
                entries = runner.base.load_or_write(folder / "plan.json", exploration.plan(config, generation, previous))
                seed = config["seed"] + 10_000_019 + generation - 1
                publish(phase="search", generation=generation, generation_completed=0, holdout_mode=None,
                        holdout_modes_completed=[], holdout_pairs_completed=[])
                rows = []
                next_progress = time.perf_counter()
                for path, group, genomes in runner.batches(folder, entries, config, previous, pool, prefetch):
                    if genomes is None:
                        batch = runner.read_batch(path, group)
                    else:
                        before = time.perf_counter()
                        metrics = training(genomes, seed=seed, actors=config["actors"], device=config["device"],
                                           visits=config["visits"], capture=config["device"] == "cuda")
                        fields = runner.base.unpack_scores(metrics, len(group))
                        batch = [{**entry, **original.score_fields(row, config["edge_pairs"])}
                                 for entry, row in zip(group, fields, strict=True)]
                        runner.record_batch(path, batch)
                        evaluated += len(batch)
                        print(f"generation={generation} committed={len(rows)+len(batch)}/{population} "
                              f"eval_commit_s={time.perf_counter()-before:.3f}", flush=True)
                    rows.extend(batch)
                    status.update(completed_candidates=(generation - 1) * population + len(rows),
                                  generation_completed=len(rows), checkpoint=path.relative_to(out).as_posix())
                    now = time.perf_counter()
                    if now >= next_progress or len(rows) == population:
                        status.pop("sparsity_bonus", None)
                        publish(**snapshot(rows))
                        next_progress = now + .5
                previous = exploration.elites(config, rows, previous)
                summary = {**snapshot(rows), "generation": generation, "candidates": population,
                           "development_seed": seed, "exploration_summary": previous["exploration_summary"]}
                _save(folder / "elites.pt", previous)
                _write(folder / "summary.json", summary)
                history.append(summary)
                _write(out / "history.json", history)
                publish(exploration_summary=previous["exploration_summary"], checkpoint=f"generations/{generation:04d}/elites.pt")
                print(f"generation={generation} exploration={previous['exploration_summary']}", flush=True)
                if needs_audit(generation):
                    run_audit(generation)
        publish(state="complete", phase="complete", holdout_mode=None,
                holdout_modes_completed=stage["audit_modes"], checkpoint="champion.pt")
    except (Exception, KeyboardInterrupt) as error:
        state = "paused" if isinstance(error, KeyboardInterrupt) else "failed"
        publish(state=state, phase=state, failure=f"{type(error).__name__}: {error}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=original.OUT)
    parser.add_argument("--from-generation", type=int, required=True)
    parser.add_argument("--accumulation", choices=("fp16", "fp32"), default="fp32")
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--generations", type=int, help="Extend the persisted total training budget.")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.prefetch < 1:
        parser.error("prefetch must be positive")
    torch.set_num_threads(1)
    with _lock(out / ".lock"):
        sealed_config = runner.base.read(out / "protocol.json")
        try:
            budget = training_budget(out, sealed_config, args.generations)
        except ValueError as error:
            parser.error(str(error))
        config = {**sealed_config, "generations": budget["generations"]}
        if not 1 <= args.from_generation < config["generations"]:
            parser.error("Select a sealed checkpoint before the last generation")
        stage = continuation_protocol(out, config, args.from_generation, args.accumulation == "fp16")
        _write(out / "training-budget.json", budget)
        candidate = runner.candidate
        runner.candidate = exploration.candidate
        try:
            evolve(out, config, stage, args.prefetch)
        finally:
            runner.candidate = candidate


if __name__ == "__main__":
    main()
