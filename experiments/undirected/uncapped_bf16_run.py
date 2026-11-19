"""Resume the uncapped experiment with BF16 graph evaluation."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import statistics
import sys

import torch

from experiments.seed_evolution.large_runner import _utc, _write
from experiments.undirected import uncapped_acceleration as fp32
from experiments.undirected import uncapped_bf16_backend as backend
from experiments.undirected import uncapped_plasticity_run as runner


PRECISION = "bf16"


def configure():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Native CUDA BF16 support is required")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def provenance():
    sources = (Path(__file__), Path(backend.__file__))
    return {
        "backend": "cuda_graph", "precision": PRECISION,
        "accumulation": "FP32 master weights, structural gates and dense inner products",
        "device": torch.cuda.get_device_name(), "torch_version": torch.__version__,
        "sources": {source.name: hashlib.sha256(source.read_bytes()).hexdigest() for source in sources},
    }


def evaluate(genomes, *, seed, actors, device, visits=8, mode="full"):
    # Keep holdouts in the same precision as search without a second persistent capture.
    evaluator = backend.evaluate_continuous if len(genomes) >= 32 else backend.evaluate_continuous_eager
    return evaluator(genomes, seed=seed, actors=actors, device=device, visits=visits, mode=mode)


def benchmark(path, out, trials):
    config = runner.base.read(path / "protocol.json")
    history = runner.base.read(path / "history.json")
    generation = history[-1]["generation"]
    cases = fp32.inputs(path, generation, config["batch_size"])
    shape = dict(actors=config["actors"], device=config["device"], visits=config["visits"])
    measures = {"fp32": fp32.replay, "bf16": evaluate}

    verified = []
    for index, case in enumerate(cases):
        world = dict(seed=91300171 + index, **shape)
        exact = fp32.measure(backend.evaluate_continuous_eager, case, world)[0]
        actual = fp32.measure(evaluate, case, world)[0]
        fp32.same_result(exact, actual)
        if not all(torch.isfinite(column).all() for column in actual.values()):
            raise ValueError("BF16 evaluator returned a nonfinite metric")
        edge_counts = (actual["final_edges"] - actual["initial_edges"]) * config["actors"]
        if not torch.equal(edge_counts, actual["births"] - actual["deaths"]):
            raise ValueError("BF16 support changes violate edge count conservation")
        update_limit = config["edge_pairs"] * config["actors"] * config["visits"] * 28
        if bool(actual["weight_updates"].gt(update_limit).any()):
            raise ValueError("BF16 weight-update count exceeded the rollout capacity")
        if not bool(actual["weight_updates"].gt(0).any()):
            raise ValueError("Plastic weights did not change")
        if index == 2 and (actual["births"][0].item() <= 0 or actual["deaths"][1].item() <= 0):
            raise ValueError("Forced BF16 growth or deletion did not execute")
        verified.append({"case": index, "metrics": len(actual),
                         "births": int(actual["births"].sum()), "deaths": int(actual["deaths"].sum()),
                         "weight_updates": int(actual["weight_updates"].sum())})
    print(f"BF16 reference matched all {verified[0]['metrics']} fields across {len(verified)} batches", flush=True)
    fp32.measure(fp32.replay, cases[0], dict(seed=91300173, **shape))

    samples = []
    for index in range(trials):
        order = ("fp32", "bf16") if index % 2 == 0 else ("bf16", "fp32")
        world = dict(seed=91301000 + index, **shape)
        genomes = cases[index % 2]
        results = {mode: fp32.measure(measures[mode], genomes, world) for mode in order}
        sample = {
            "trial": index + 1, "seed": world["seed"], "order": order,
            **{f"{mode}_s": results[mode][1] for mode in order},
            "strict_change": (results["bf16"][0]["strict"] - results["fp32"][0]["strict"]).mean().item(),
            "completion_change": (results["bf16"][0]["completion"] - results["fp32"][0]["completion"]).mean().item(),
        }
        samples.append(sample)
        print(f"pair={sample['trial']} fp32={sample['fp32_s']:.4f}s bf16={sample['bf16_s']:.4f}s", flush=True)
    result = {
        **provenance(), "finished_utc": _utc(), "generation": generation, "batch_shape": [config["batch_size"],
            config["actors"], config["visits"]],
        "verification": verified, "samples": samples,
        "fp32_median_s": statistics.median(row["fp32_s"] for row in samples),
        "bf16_median_s": statistics.median(row["bf16_s"] for row in samples),
        "median_paired_speedup": statistics.median(row["fp32_s"] / row["bf16_s"] for row in samples),
        "mean_strict_change": statistics.fmean(row["strict_change"] for row in samples),
        "mean_completion_change": statistics.fmean(row["completion_change"] for row in samples),
    }
    _write(out, result)
    print(f"median speedup={result['median_paired_speedup']:.4f}", flush=True)


def train(args, config, pool):
    # The original dispatcher holds the experiment lock while invoking this hook.
    path = args.out / "precision-segments.json"
    current = provenance()
    if path.is_file():
        segments = runner.base.read(path)
        if segments["backend"] != current or not 1 <= segments["bf16_from_generation"] <= config["generations"]:
            raise ValueError("Recorded BF16 backend or generation differs")
    else:
        history, _ = runner.recover(args.out, config)
        next_generation = len(history) + 1
        folder = args.out / "generations" / f"{next_generation:04d}" / "batches"
        partial_batches = sorted(item.name for item in folder.glob("*.json.gz"))
        start = next_generation + bool(partial_batches)
        if start > config["generations"]:
            raise ValueError("No whole generation remains for a precision change")
        segments = {
            "previous_precision": "fp32", "next_precision": PRECISION,
            "bf16_from_generation": start, "fp32_through_generation": start - 1,
            "prior_committed_batches": partial_batches, "recorded_utc": _utc(), "backend": current,
        }
        _write(path, segments)
    first_bf16 = segments["bf16_from_generation"]
    seed_offset = config["seed"] + 10_000_019
    print(f"execution precision: FP32 through generation {first_bf16 - 1}; "
          f"BF16 starts at generation {first_bf16}", flush=True)

    def for_generation(generation):
        return evaluate if generation >= first_bf16 else fp32.replay

    def evaluate_search(genomes, *, seed, **world):
        return for_generation(seed - seed_offset + 1)(genomes, seed=seed, **world)

    def audit_segment(config, out, generation, checkpoint, publish):
        runner.evaluate_continuous = for_generation(generation)
        try:
            return reference_audit(config, out, generation, checkpoint, publish)
        finally:
            runner.evaluate_continuous = evaluate_search

    def write_metadata(path, value):
        if path.name == "process.json":
            value = {**value, "bf16_from_generation": first_bf16}
        elif (path.name in {"progress.json", "summary.json", "holdout.json"}
              or path.parent.name == "audits") and isinstance(value, dict) and "generation" in value:
            precision = PRECISION if value["generation"] >= first_bf16 else "fp32"
            value = {**value, "execution_precision": precision, "bf16_from_generation": first_bf16}
        _write(path, value)

    reference_audit = runner.audit
    runner.evaluate_continuous, runner.audit, runner._write = evaluate_search, audit_segment, write_metadata
    reference_run(args, config, pool)


reference_run = runner.run


def main():
    configure()
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        parser = argparse.ArgumentParser(description="Compare sealed FP32 and BF16 graph replay on matched inputs.")
        parser.add_argument("--run-dir", type=Path, default=runner.OUT)
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument("--trials", type=int, default=6)
        args = parser.parse_args(sys.argv[2:])
        if args.trials < 2 or args.trials % 2:
            parser.error("Use a positive, even trial count")
        benchmark(args.run_dir, args.out, args.trials)
        return

    print(f"Evaluation accelerator: {provenance()}", flush=True)
    runner.run = train
    runner.main()


if __name__ == "__main__":
    main()
