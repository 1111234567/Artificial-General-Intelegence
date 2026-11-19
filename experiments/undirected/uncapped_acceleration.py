"""Benchmark and optionally replay the sealed uncapped evaluator as a CUDA graph."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import random
import statistics
import sys
import time

import torch

from experiments.seed_evolution.large_runner import _utc, _write
from experiments.undirected import uncapped_plasticity_run as runner
from experiments.undirected.evolved_plasticity import mutate_genome
from experiments.undirected.plasticity_continuous_world import evaluate_continuous as eager
from experiments.undirected import uncapped_cuda_backend


def replay(genomes, *, seed, actors, device, visits=8, mode="full"):
    # Leave tiny audits on the reference path; the production batch reuses one graph.
    evaluate = uncapped_cuda_backend.evaluate_continuous if device == "cuda" and len(genomes) >= 32 else eager
    return evaluate(genomes, seed=seed, actors=actors, device=device, visits=visits, mode=mode)


def provenance():
    sources = (Path(__file__), Path(uncapped_cuda_backend.__file__))
    return {
        "backend": "cuda_graph", "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "sources": {source.name: hashlib.sha256(source.read_bytes()).hexdigest() for source in sources},
    }


def inputs(path, generation, width):
    elites = torch.load(path / "generations" / f"{generation:04d}" / "elites.pt",
                        map_location="cpu", weights_only=True)
    pool = list(elites["genomes"].values())
    parent_batch = [pool[i % len(pool)] for i in range(width)]
    mutant_batch = [mutate_genome(parent, 780012 + i, wide=i % 8 == 0)
                    for i, parent in enumerate(parent_batch)]
    dense_batch = mutant_batch.copy()
    for index, bias in enumerate((2.0, -2.0)):
        dense_batch[index] = {**dense_batch[index], "rule": dense_batch[index]["rule"].clone()}
        dense_batch[index]["rule"][1, :] = 0
        dense_batch[index]["rule"][1, 0] = bias
    return parent_batch, mutant_batch, dense_batch


def same_result(left, right):
    if left.keys() != right.keys():
        raise ValueError("Evaluator metric columns differ")
    differences = [key for key in left if not torch.equal(left[key], right[key])]
    if differences:
        raise ValueError(f"Evaluator values differ: {differences}")


def measure(evaluate, genomes, world):
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    columns = evaluate(genomes, **world)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    memory = torch.cuda.max_memory_allocated()
    return {key: value.cpu() for key, value in columns.items()}, elapsed, memory


def benchmark(path, out, trials):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    config = runner.base.read(path / "protocol.json")
    history = runner.base.read(path / "history.json")
    generation = history[-1]["generation"]
    cases = inputs(path, generation, config["batch_size"])
    shape = dict(actors=config["actors"], device="cuda", visits=config["visits"])
    world = dict(seed=58180023, **shape)
    started = time.perf_counter()
    initial = replay(cases[0], **world)
    torch.cuda.synchronize()
    capture_s = time.perf_counter() - started
    same_result({key: value.cpu() for key, value in initial.items()}, measure(eager, cases[0], world)[0])
    print(f"capture_s={capture_s:.3f}", flush=True)

    samples = []
    for index in range(trials):
        names = ["eager", "cuda_graph"]
        if random.Random(index // 2 + 8313).getrandbits(1) == index % 2:
            names.reverse()
        records = {}
        case_index = index % len(cases)
        genomes = cases[case_index]
        world = dict(seed=58180201 + index, **shape)
        for name in names:
            records[name] = measure(eager if name == "eager" else replay, genomes, world)
        same_result(records["eager"][0], records["cuda_graph"][0])
        sample = {
            "trial": index, "seed": world["seed"], "case": case_index, "order": names,
            "eager_s": records["eager"][1], "cuda_graph_s": records["cuda_graph"][1],
            "eager_peak_mib": records["eager"][2] / 2**20,
            "cuda_graph_peak_mib": records["cuda_graph"][2] / 2**20,
        }
        samples.append(sample)
        print(f"pair={index + 1} eager={sample['eager_s']:.4f}s "
              f"replay={sample['cuda_graph_s']:.4f}s exact=true", flush=True)

    eager_time = statistics.median(row["eager_s"] for row in samples)
    replay_time = statistics.median(row["cuda_graph_s"] for row in samples)
    result = {
        **provenance(), "finished_utc": _utc(), "generation": generation,
        "batch_size": config["batch_size"], "actors": config["actors"], "visits": config["visits"],
        "all_metrics_exact": True, "capture_s": capture_s, "samples": samples,
        "eager_median_s": eager_time, "cuda_graph_median_s": replay_time,
        "median_paired_speedup": statistics.median(row["eager_s"] / row["cuda_graph_s"] for row in samples),
    }
    _write(out, result)
    print(f"median speedup={result['median_paired_speedup']:.4f}", flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        parser = argparse.ArgumentParser(description="Pair eager and CUDA-graph evaluation on archived genotypes.")
        parser.add_argument("--run-dir", type=Path, default=runner.OUT)
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument("--trials", type=int, default=8)
        args = parser.parse_args(sys.argv[2:])
        if args.trials < 4:
            parser.error("Use at least four paired trials")
        benchmark(args.run_dir, args.out, args.trials)
    else:
        print(f"Evaluation accelerator: {provenance()}", flush=True)
        runner.evaluate_continuous = replay
        runner.main()


if __name__ == "__main__":
    main()
