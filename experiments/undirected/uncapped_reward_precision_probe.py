"""Compare three precisions on the same 96-node birth genomes and worlds."""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
from itertools import combinations, permutations
import json
from pathlib import Path
import statistics
import time

import torch

from experiments.seed_evolution.large_runner import _save, _utc, _write
from .evolved_plasticity_world import NEW, N_SYMBOLS, READY
from .uncapped_island_crossover_run import DELAYED, OUT as ISLAND_OUT, TIMING, block_ticks
from .uncapped_ready_signal_run import schedule
from .uncapped_reward_precision import PrecisionEvaluation
from .variable_birth_genome import make_genome


OUT = ISLAND_OUT.with_name("2026-11-17-reward-precision-probe-v2")
SOURCE = ISLAND_OUT / "generations" / "0972" / "elites.pt"
NODES = 96
ACTORS = 16
VISITS = 4
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
PAIRINGS = tuple(combinations(DTYPES, 2))
ORDERS = tuple(permutations(DTYPES))
QUALITY_SEEDS = range(84_290_100, 84_290_112)
SPEED_SEEDS = range(84_290_200, 84_290_212)
FIELDS = (
    "first_finish_correct_a_first_return", "first_finish_correct_b_first_return",
    "ready_coverage_a_first_return", "ready_coverage_b_first_return", "ready_coverage", "strict",
    "reward_events", "attempted_writes", "changed_writes", "births", "deaths",
)


def expand_genome(source, seed):
    """Embed the old graph by endpoint and initialize connections to added nodes."""
    old_nodes = source["node_bias"].numel()
    left, right = torch.triu_indices(old_nodes, old_nodes, offset=1)
    pair = left * (2 * NODES - left - 1) // 2 + right - left - 1
    genome = make_genome(seed, nodes=NODES)
    for key in ("birth_mask", "birth_w"):
        genome[key][pair] = source[key]
    genome["node_bias"][:old_nodes] = source["node_bias"]
    genome["rule"].copy_(source["rule"])
    return genome


def load_genomes(source):
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    warm_names = [entry["id"] for island in checkpoint["islands"] for entry in island[:4]]
    warm = [expand_genome(checkpoint["genomes"][name], 84_290_300 + i) for i, name in enumerate(warm_names)]
    fresh = [make_genome(84_290_400 + i, nodes=NODES) for i in range(64 - len(warm))]
    return warm + fresh, warm_names


def verify_tapes():
    """Check that the padding holds request cadence fixed while READY moves."""
    ticks = block_ticks(DELAYED)
    shift = DELAYED["min_wait"] - TIMING["min_wait"]
    for seed in QUALITY_SEEDS:
        settings = {"seed": seed, "actors": ACTORS, "device": "cpu", "visits": VISITS, "pad_to": ticks}
        regular, long = (schedule(timing=timing, **settings) for timing in (TIMING, DELAYED))
        fields = ("maps", "raw", "context", "target")
        if not all(torch.equal(getattr(regular, key), getattr(long, key)) for key in fields):
            raise RuntimeError(f"request tapes differ at seed {seed}")
        if not torch.equal(regular.public[..., 0, NEW], long.public[..., 0, NEW]):
            raise RuntimeError(f"issue timing differs at seed {seed}")
        if not torch.equal(regular.public[:, :-shift, :, 0, READY], long.public[:, shift:, :, 0, READY]):
            raise RuntimeError(f"unexpected READY shift at seed {seed}")
        for tape in (regular, long):
            if not tape.stage.eq(2).sum(dim=1).eq(TIMING["finish_window"]).all():
                raise RuntimeError(f"unexpected FINISH window at seed {seed}")


def interval(samples, critical):
    center = statistics.fmean(samples)
    radius = critical * statistics.stdev(samples) / len(samples) ** .5
    return {"mean": center, "interval95": [center - radius, center + radius]}


def summarize(row, start, stop):
    fields = {key: statistics.fmean(row[key][start:stop]) for key in FIELDS}
    fields["first_return_correct"] = (
        fields["first_finish_correct_a_first_return"] + fields["first_finish_correct_b_first_return"]
    ) / 2
    fields["writes_per_feedback"] = fields["changed_writes"] / (fields["reward_events"] or 1.)
    fields["fraction_of_proposals_written"] = fields["changed_writes"] / (fields["attempted_writes"] or 1.)
    return fields


def run_quality(evaluators, genomes, warm_count, device):
    groups = {"all": (0, len(genomes)), "warm": (0, warm_count), "random": (warm_count, len(genomes))}
    tests = {}
    for condition, evaluator in evaluators.items():
        paired = []
        for i, seed in enumerate(QUALITY_SEEDS):
            order = ORDERS[i % len(ORDERS)]
            row = {"seed": seed}
            for precision in order:
                results = evaluator(genomes, seed=seed, actors=ACTORS, visits=VISITS, device=device,
                                    dtype=DTYPES[precision])
                row[precision] = {field: results[field].cpu().tolist() for field in FIELDS}
                events = row[precision]["reward_events"]
                coverage = row[precision]["ready_coverage"]
                if any(count != rate * (ACTORS * N_SYMBOLS * VISITS)
                       for count, rate in zip(events, coverage, strict=True)):
                    raise RuntimeError(f"legal first finishes and rewarded events differ on seed {seed}")
                if any(changed > offered for changed, offered in zip(
                    row[precision]["changed_writes"], row[precision]["attempted_writes"], strict=True,
                )):
                    raise RuntimeError(f"changed a weight with no eligible proposal on seed {seed}")
            paired.append(row)
            print(f"paired quality {condition} {i + 1}/{len(QUALITY_SEEDS)}", flush=True)

        summaries = {}
        for group, (start, stop) in groups.items():
            by_precision = {
                name: [summarize(row[name], start, stop) for row in paired] for name in DTYPES
            }
            metrics = tuple(by_precision["fp16"][0])
            summaries[group] = {
                "means": {name: {field: statistics.fmean(row[field] for row in seed_rows) for field in metrics}
                          for name, seed_rows in by_precision.items()},
                "paired_deltas": {
                    f"{first}_minus_{second}": {field: interval(
                        [a[field] - b[field] for a, b in zip(by_precision[first], by_precision[second], strict=True)],
                        2.200985,
                    ) for field in metrics}
                    for first, second in PAIRINGS
                },
            }
        tests[condition] = {"groups": summaries, "paired": paired}
    return tests


def run_speed(evaluator, genomes, device):
    # Capture both graphs and run a second replay before taking any timing.
    first_seed = SPEED_SEEDS.start
    for _ in range(2):
        for dtype in DTYPES.values():
            evaluator(genomes, seed=first_seed, actors=ACTORS, visits=VISITS, device=device, dtype=dtype)
    torch.cuda.synchronize(device)

    blocks = []
    for i, seed in enumerate(SPEED_SEEDS):
        first_half = ORDERS[i % len(ORDERS)]
        order = (*first_half, *reversed(first_half))
        block = {"seed": seed, "order": order, **{name: [] for name in DTYPES}}
        for precision in order:
            started = time.perf_counter()
            evaluator(genomes, seed=seed, actors=ACTORS, visits=VISITS, device=device, dtype=DTYPES[precision])
            torch.cuda.synchronize(device)
            block[precision].append(time.perf_counter() - started)
        blocks.append(block)
        elapsed = " ".join(f"{name}={statistics.fmean(block[name]):.4f}s" for name in DTYPES)
        print(f"speed block {i + 1}/{len(SPEED_SEEDS)} {elapsed}", flush=True)

    mean_times = {name: statistics.fmean(t for block in blocks for t in block[name]) for name in DTYPES}
    return {
        "mean_batch_seconds": mean_times,
        "candidates_per_second": {name: len(genomes) / seconds for name, seconds in mean_times.items()},
        "paired_speed_ratios": {
            f"{first}_over_{second}": interval([
                statistics.fmean(block[second]) / statistics.fmean(block[first]) for block in blocks
            ], 2.200985)
            for first, second in PAIRINGS
        },
        "blocks": blocks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    protocol_path = args.out / "protocol.json"
    if protocol_path.exists():
        parser.error("Choose a fresh output directory.")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        parser.error("Native CUDA BF16 and FP16 support is required.")
    torch.set_num_threads(1)
    device = torch.device("cuda", torch.cuda.current_device())
    verify_tapes()
    genomes, warm_names = load_genomes(args.source)
    birth_path = args.out / "births.pt"
    _save(birth_path, {"genomes": genomes, "warm_sources": warm_names})
    old_run_status = json.loads((ISLAND_OUT / "progress.json").read_text(encoding="utf-8"))
    _write(protocol_path, {
        "started_utc": _utc(), "device": str(device), "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__, "nodes": NODES, "actors": ACTORS, "visits": VISITS,
        "formats": list(DTYPES), "matrix_accumulation": "FP32, TF32 disabled",
        "candidates": len(genomes), "warm_count": len(warm_names), "warm_sources": warm_names,
        "warm_checkpoint": str(args.source.resolve()), "births_sha256": hashlib.sha256(birth_path.read_bytes()).hexdigest(),
        "quality_seeds": list(QUALITY_SEEDS), "speed_seeds": list(SPEED_SEEDS),
        "training_progress_at_start": {key: old_run_status[key] for key in ("state", "phase", "generation")},
        "timing": {"regular": TIMING, "long": DELAYED, "quality_ticks_per_request": block_ticks(DELAYED),
                   "native_quality_and_speed_ticks_per_request": block_ticks(TIMING)},
        "perf": "synchronized wall time of warmed CUDA Graph calls, including host input preparation and upload",
        "quality": "paired across tapes; per-seed aggregates over fixed genomes and 16 independent lives",
        "sources": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (Path(__file__), Path(__file__).with_name("uncapped_reward_precision.py"))},
    })

    ticks = block_ticks(DELAYED)
    speed_evaluator = PrecisionEvaluation(partial(schedule, timing=TIMING))
    evaluators = {
        "regular_native": speed_evaluator,
        "regular_padded": PrecisionEvaluation(partial(schedule, timing=TIMING, pad_to=ticks)),
        "long": PrecisionEvaluation(partial(schedule, timing=DELAYED, pad_to=ticks)),
    }
    quality = run_quality(evaluators, genomes, len(warm_names), device)
    result_path = args.out / "results.json"
    _write(result_path, {"state": "measuring_speed", "quality": quality})
    speed_status = json.loads((ISLAND_OUT / "progress.json").read_text(encoding="utf-8"))
    speed = run_speed(speed_evaluator, genomes, device)
    speed["training_progress_at_start"] = {key: speed_status[key] for key in ("state", "phase", "generation")}
    speed_status = json.loads((ISLAND_OUT / "progress.json").read_text(encoding="utf-8"))
    speed["training_progress_at_end"] = {key: speed_status[key] for key in ("state", "phase", "generation")}
    report = {"state": "complete", "completed_utc": _utc(), "quality": quality, "speed": speed}
    _write(result_path, report)
    print(json.dumps({"quality": {name: row["groups"]["all"]["means"] for name, row in quality.items()},
                      "speed": speed["candidates_per_second"]}), flush=True)


if __name__ == "__main__":
    main()
