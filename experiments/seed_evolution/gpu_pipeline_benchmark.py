"""Measure a sustained GPU training and CPU scoring pipeline."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import torch

from experiments.seed_evolution import device_benchmark, gpu_batch_benchmark, gpu_batch_train, large_runner


def _timed_score(source, model):
    started = perf_counter()
    fitness, _ = gpu_batch_benchmark._score(source, model)
    return fitness, started, perf_counter()


def run(output, candidates, steps, chunk, workers, sources):
    device = torch.device("cuda:0")
    device_benchmark.strict_fp32()
    torch.backends.cudnn.benchmark = False
    properties = torch.cuda.get_device_properties(device)
    inputs = [gpu_batch_benchmark._candidate_initializations(source, candidates)[0] for source in sources]
    report = dict(state="running", device=properties.name, torch=str(torch.__version__),
                  candidates_per_batch=candidates, steps=steps, cpu_workers=workers,
                  executor_dtype="bfloat16", executor_chunk=chunk, batches=[])
    started = perf_counter()

    def elapsed():
        return perf_counter() - started

    def publish(event, row):
        large_runner._write(output, report)
        print(f"{elapsed():.2f}s {event}: {row}", flush=True)

    def collect(pending):
        batch, futures = pending
        scored = elapsed()
        results = [future.result() for future in futures]
        begin = min(result[1] for result in results) - started
        finish = max(result[2] for result in results) - started
        batch.update(score_start_seconds=begin, score_finish_seconds=finish,
                     score_active_seconds=finish - begin,
                     score_submit_to_last_finish_seconds=finish - batch["score_submit_seconds"],
                     score_wait_after_training_seconds=elapsed() - scored,
                     fitness=[result[0] for result in results])
        publish("scored", {key: batch[key] for key in ("source", "score_active_seconds",
                                                       "score_submit_to_last_finish_seconds")})

    pending = None
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for source, genomes in zip(sources, inputs, strict=True):
                batch = dict(source=source, train_start_seconds=elapsed())
                report["batches"].append(batch)
                publish("starting GPU batch", source)
                stats, result = gpu_batch_benchmark._measure(
                    "batch", candidates, steps, device, lambda: gpu_batch_train.train_batch(
                        source, genomes, steps, device=device, dtype=torch.bfloat16, executor_chunk=chunk,
                    ), release_cache=False,
                )
                batch.update(train_finish_seconds=elapsed(), training=stats)
                publish("trained", dict(source=source, wall_seconds=stats["wall_seconds"],
                                        peak_reserved_gib=stats["peak_reserved_gib"]))
                if pending is not None:
                    previous, _ = pending
                    collect(pending)
                    overlap = min(batch["train_finish_seconds"], previous["score_finish_seconds"]) - max(
                        batch["train_start_seconds"], previous["score_start_seconds"])
                    batch["overlap_with_previous_score_seconds"] = max(0, overlap)
                batch["score_submit_seconds"] = elapsed()
                pending = batch, [pool.submit(_timed_score, source, state) for state in result["states"]]
                publish("submitted CPU scores", source)
            collect(pending)
        first, *_, last = report["batches"]
        total = candidates * len(sources)
        steady_seconds = last["score_submit_seconds"] - first["score_submit_seconds"]
        steady_count = total - candidates
        report.update(state="complete", total_candidates=total, wall_seconds=elapsed(),
                      candidates_per_minute=60 * total / elapsed(),
                      sustained_window_seconds=steady_seconds, sustained_candidates=steady_count,
                      sustained_candidates_per_minute=60 * steady_count / steady_seconds)
        publish("complete", {key: report[key] for key in ("total_candidates", "wall_seconds",
                                                          "candidates_per_minute", "sustained_candidates_per_minute")})
    except Exception as error:
        report.update(state="failed", error=repr(error), wall_seconds=elapsed())
        publish("failed", report["error"])
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=40)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--sources", type=int, nargs="+", default=(0, 1, 2), choices=(0, 1, 2))
    args = parser.parse_args()
    if (not 1 <= args.candidates <= 64 or not 1 <= args.steps <= 512 or not 1 <= args.workers <= 61
            or not 1 <= args.chunk <= args.candidates or len(args.sources) < 2):
        parser.error("invalid batch configuration")
    if args.output.exists():
        parser.error("choose a new benchmark output path")
    run(args.output, args.candidates, args.steps, args.chunk, args.workers, args.sources)


if __name__ == "__main__":
    main()
