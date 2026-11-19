"""Measure candidate throughput for single-process GPU batches."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
from pathlib import Path
from time import perf_counter

import torch

from experiments.seed_evolution import core, device_benchmark, gpu_batch_train, hybrid_backend, large_runner
from experiments.seed_evolution.lazy_training_streams import LazyTrainingStreams


def _candidate_initializations(source, count):
    genomes, archived_fitness = [], []
    for index in range(count):
        folder = large_runner.OUT / "search" / f"rep_{source}" / f"shared_g00_i{index:02d}"
        record = large_runner.base.read(folder / "candidate.json")
        path = folder / "initial.pt"
        large_runner._verify_hash(path, record["artifacts"]["initial.pt"])
        genomes.append(large_runner._load(path))
        archived_fitness.append(record["fitness"])
    return genomes, archived_fitness


def _measure(name, count, steps, device, operation, *, release_cache=True):
    torch.cuda.synchronize(device)
    if release_cache:
        torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    free_before, _ = torch.cuda.mem_get_info(device)
    started = perf_counter()
    result = operation()
    torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    gib = 1 << 30
    stats = dict(mode=name, candidates=count, steps=steps, wall_seconds=elapsed,
                 candidate_updates_per_second=count * steps / elapsed,
                 peak_allocated_gib=torch.cuda.max_memory_allocated(device) / gib,
                 peak_reserved_gib=torch.cuda.max_memory_reserved(device) / gib,
                 free_at_start_gib=free_before / gib, timing=result.get("timing"))
    if "metadata" in result:
        stats["trainer_metadata"] = result["metadata"]
    return stats, result


def _train_eager(source, genome, steps, device):
    with LazyTrainingStreams().install():
        result = core.inner_train(source, genome, steps, (steps,), device)
    checkpoint = result["checkpoints"][steps]
    return dict(states=[checkpoint["model"]], checkpoints=[checkpoint])


def _train_hybrid(source, genome, steps):
    with LazyTrainingStreams().install():
        result = hybrid_backend.train(source, genome, steps, (steps,), backend="hybrid", cache=True)
    checkpoint = result["checkpoints"][steps]
    return dict(states=[checkpoint["model"]], checkpoints=[checkpoint])


def _compare(reference, checkpoint):
    old_model, new_model = (item["model"] for item in (reference, checkpoint))
    changes = device_benchmark.tensor_errors(old_model, new_model)
    moments = device_benchmark.tensor_errors(
        {f"{index}.{key}": value for index, state in reference["optimizer"]["state"].items()
         for key, value in state.items()},
        {f"{index}.{key}": value for index, state in checkpoint["optimizer"]["state"].items()
         for key, value in state.items()},
    )
    return dict(model_max_abs=changes["max_abs"], model_different_scalars=changes["nonzero"],
                optimizer_max_abs=moments["max_abs"])


def _score(source, model):
    started = perf_counter()
    return core.score_model(source, model)["fitness"], perf_counter() - started


def run(source, widths, steps, output, *, dtype, eager, hybrid, scores, executor_chunk=None, score_workers=1):
    device = torch.device("cuda:0")
    device_benchmark.strict_fp32()
    torch.backends.cudnn.benchmark = False
    properties = torch.cuda.get_device_properties(device)
    genomes, archived_fitness = _candidate_initializations(source, max(widths))
    report = dict(source_seed=source, device=properties.name, dtype=str(dtype), torch=str(torch.__version__),
                  cuda=torch.version.cuda, steps=steps, executor_chunk=executor_chunk,
                  gpu_total_gib=properties.total_memory / (1 << 30),
                  trials=[])
    reference = None
    reference_mode = None

    def publish(row):
        report["trials"].append(row)
        large_runner._write(output, report)
        print(row, flush=True)

    if eager:
        stats, reference = _measure("eager", 1, steps, device,
                                    lambda: _train_eager(source, genomes[0], steps, device))
        reference_mode = "eager"
        publish(stats)
    if hybrid:
        stats, hybrid_result = _measure("hybrid", 1, steps, device,
                                       lambda: _train_hybrid(source, genomes[0], steps))
        if reference is None:
            reference = hybrid_result
            reference_mode = "hybrid"
        publish(stats)
    for width in widths:
        options = dict(device=device, dtype=dtype)
        if executor_chunk is not None:
            options["executor_chunk"] = executor_chunk
        try:
            stats, result = _measure("batch", width, steps, device, lambda: gpu_batch_train.train_batch(
                source, genomes[:width], steps, **options,
            ))
        except torch.OutOfMemoryError as error:
            publish(dict(mode="batch", candidates=width, status="out_of_memory", error=str(error)))
            break
        if reference is not None:
            stats["reference_mode"] = reference_mode
            stats["first_candidate_diff"] = _compare(reference["checkpoints"][0], result["checkpoints"][0])
        publish(stats)
        if scores and width == widths[-1]:
            torch.cuda.empty_cache()
            scored = result["states"][:scores]
            workers = min(score_workers, len(scored))
            score = partial(_score, source)
            started = perf_counter()
            if workers == 1:
                rows = list(map(score, scored))
            else:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    rows = list(pool.map(score, scored))
            score_wall = perf_counter() - started
            values, timings = map(list, zip(*rows))
            evaluation = dict(seconds=timings, fitness=values, wall_seconds=score_wall, workers=workers)
            stats["cpu_development_evaluation"] = evaluation
            if steps == large_runner.CONFIG["inner_steps"]:
                evaluation["archived_cpu_fitness"] = archived_fitness[:len(values)]
            if len(scored) == width:
                stats["candidates_per_minute_including_cpu_development"] = 60 * width / (
                    stats["wall_seconds"] + score_wall)
            if reference is not None:
                reference_fitness, _ = score(reference["states"][0])
                evaluation["reference_first_fitness"] = reference_fitness
                evaluation["first_fitness_diff"] = values[0] - reference_fitness
            large_runner._write(output, report)
            print(stats["cpu_development_evaluation"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--widths", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--scores", type=int, default=0)
    parser.add_argument("--score-workers", type=int, default=min(61, os.process_cpu_count() or 1),
                        help="Number of CPU scorers (default: available logical processors)")
    parser.add_argument("--executor-chunk", type=int)
    args = parser.parse_args()
    if (min(args.widths) < 1 or max(args.widths) > 64 or not 1 <= args.steps <= 512 or args.scores < 0
            or not 1 <= args.score_workers <= 61
            or (args.executor_chunk is not None and args.executor_chunk < 1)):
        parser.error("candidate count or training steps are out of range")
    if args.output.exists():
        parser.error("choose a new benchmark output path")
    run(args.source, args.widths, args.steps, args.output, dtype=getattr(torch, args.dtype),
        eager=args.eager, hybrid=args.hybrid, scores=args.scores, executor_chunk=args.executor_chunk,
        score_workers=args.score_workers)


if __name__ == "__main__":
    main()
