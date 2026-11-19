"""Compare FP16 and FP32 CUDA matrix accumulation in the same FP16 policy."""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from experiments.seed_evolution.large_runner import _utc, _write
from .evolved_plasticity_world import N_SYMBOLS
from .uncapped_island_crossover_run import DELAYED, OUT as ISLAND_OUT, TIMING
from .uncapped_ready_signal_run import schedule
from .uncapped_reward_precision import PrecisionEvaluation, _matmul_precision
from .uncapped_reward_precision_probe import (
    ACTORS, FIELDS, NODES, OUT as PRECISION_OUT, QUALITY_SEEDS, SPEED_SEEDS, VISITS, interval, summarize,
)


OUT = ISLAND_OUT.with_name("2026-11-17-reward-accumulation-probe-v1")
FORMATS = {"fp16_accum": True, "fp32_accum": False}
CRITICAL = 2.200985  # 12 paired seed blocks


def prior_progress():
    state = json.loads((ISLAND_OUT / "progress.json").read_text(encoding="utf-8"))
    return {key: state[key] for key in ("state", "phase", "generation")}


def wait_before_speed(limit):
    state = prior_progress()
    if state["state"] == "running" and limit:
        print(f"Waiting up to {limit}s for the previous run to leave the GPU before timing.", flush=True)
        deadline = time.monotonic() + limit
        while state["state"] == "running" and time.monotonic() < deadline:
            time.sleep(min(5., max(0., deadline - time.monotonic())))
            state = prior_progress()
    return state


@torch.no_grad()
def check_matmul_path(genomes, device):
    # Use the same reciprocal matrices and batch shape as the evaluator, with
    # a fixed nonzero activity snapshot to make rounding differences visible.
    rows, cols = torch.triu_indices(NODES, NODES, offset=1, device=device)
    birth = torch.stack([g["birth_w"] for g in genomes]).to(device=device, dtype=torch.float16)
    weights = birth.repeat_interleave(ACTORS, dim=0)
    matrix = torch.zeros((len(weights), NODES, NODES), dtype=torch.float16, device=device)
    matrix[:, rows, cols] = weights
    matrix[:, cols, rows] = weights
    rng = torch.Generator(device=device).manual_seed(84_290_450)
    activity = torch.randn((len(weights), NODES, 1), device=device, generator=rng).tanh_().half()
    values = {}
    for name, flag in FORMATS.items():
        with _matmul_precision(torch.float16, flag):
            values[name] = torch.bmm(matrix, activity)
    first, second = values.values()
    different = int(first.ne(second).sum().item())
    if not different or not all(torch.isfinite(value).all().item() for value in values.values()):
        raise RuntimeError("Same-shape bmm did not verify a finite, distinct accumulation path.")
    gap = (first.float() - second.float()).abs()
    return {"shape": list(matrix.shape), "values": first.numel(), "changed_values": different,
            "mean_absolute_difference": gap.mean().item(), "max_absolute_difference": gap.max().item()}


def make_evaluators(timing):
    tape = partial(schedule, timing=timing)
    return {name: PrecisionEvaluation(tape, fp16_accumulation=flag) for name, flag in FORMATS.items()}


def call(evaluators, genomes, device, seed, name):
    return evaluators[name](genomes, seed=seed, actors=ACTORS, visits=VISITS, device=device, dtype=torch.float16)


def quality(evaluators, genomes, device, condition):
    pairs = []
    for i, seed in enumerate(QUALITY_SEEDS):
        order = tuple(FORMATS) if i % 2 == 0 else tuple(reversed(FORMATS))
        pair = {"seed": seed}
        for name in order:
            result = call(evaluators, genomes, device, seed, name)
            row = {field: result[field].cpu().tolist() for field in FIELDS}
            for count, ready, changed, proposed in zip(
                row["reward_events"], row["ready_coverage"], row["changed_writes"],
                row["attempted_writes"], strict=True,
            ):
                if count != ready * (ACTORS * N_SYMBOLS * VISITS) or changed > proposed:
                    raise RuntimeError(f"reward telemetry mismatch at seed {seed} in {name}")
            pair[name] = row
        pairs.append(pair)
        print(f"{condition} quality {i + 1}/{len(QUALITY_SEEDS)}", flush=True)
    by_format = {
        name: [summarize(pair[name], 0, len(genomes)) for pair in pairs] for name in FORMATS
    }
    keys = tuple(by_format["fp16_accum"][0])
    return {
        "formats": {name: {field: statistics.fmean(row[field] for row in rows) for field in keys}
                    for name, rows in by_format.items()},
        "fp16_minus_fp32_accum": {field: interval([
            a[field] - b[field] for a, b in zip(by_format["fp16_accum"], by_format["fp32_accum"], strict=True)
        ], CRITICAL) for field in keys},
        "paired": pairs,
    }


def speed(evaluators, genomes, device):
    for _ in range(2):
        for name in FORMATS:
            call(evaluators, genomes, device, SPEED_SEEDS.start, name)
    torch.cuda.synchronize(device)

    blocks = []
    for i, seed in enumerate(SPEED_SEEDS):
        first, second = tuple(FORMATS) if i % 2 == 0 else tuple(reversed(FORMATS))
        order = (first, second, second, first)
        block = {"seed": seed, "order": order, **{name: [] for name in FORMATS}}
        for name in order:
            started = time.perf_counter()
            call(evaluators, genomes, device, seed, name)
            torch.cuda.synchronize(device)
            block[name].append(time.perf_counter() - started)
        blocks.append(block)
        elapsed = " ".join(f"{name}={statistics.fmean(block[name]):.4f}s" for name in FORMATS)
        print(f"speed block {i + 1}/{len(SPEED_SEEDS)} {elapsed}", flush=True)
    means = {name: statistics.fmean(t for block in blocks for t in block[name]) for name in FORMATS}
    return {"mean_batch_seconds": means,
            "candidates_per_second": {name: len(genomes) / elapsed for name, elapsed in means.items()},
            "paired_fp16_accum_speedup": interval([
                statistics.fmean(block["fp32_accum"]) / statistics.fmean(block["fp16_accum"]) for block in blocks
            ], CRITICAL), "blocks": blocks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--idle-wait", type=int, default=180,
                        help="Seconds to wait for the prior training run to finish before measuring speed.")
    args = parser.parse_args()
    if args.idle_wait < 0 or (args.out / "protocol.json").exists():
        parser.error("Use a nonnegative wait and a fresh output directory.")
    if not torch.cuda.is_available():
        parser.error("CUDA is required.")
    torch.set_num_threads(1)
    device = torch.device("cuda", torch.cuda.current_device())
    source = PRECISION_OUT / "births.pt"
    prior_hash = json.loads((PRECISION_OUT / "protocol.json").read_text(encoding="utf-8"))["births_sha256"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != prior_hash:
        raise RuntimeError("Birth genome archive differs from the three-format experiment.")
    genomes = torch.load(source, map_location="cpu", weights_only=True)["genomes"]
    proof = check_matmul_path(genomes, device)
    print(f"Accumulation witness changed {proof['changed_values']}/{proof['values']} bmm outputs.", flush=True)
    _write(args.out / "protocol.json", {
        "started_utc": _utc(), "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "formats": FORMATS, "model_dtype": "torch.float16", "reduced_precision_reduction": False,
        "births": str(source), "births_sha256": prior_hash, "genomes": len(genomes), "actors": ACTORS,
        "nodes": NODES, "visits": VISITS, "quality_seeds": list(QUALITY_SEEDS), "speed_seeds": list(SPEED_SEEDS),
        "timing": {"regular": TIMING, "long": DELAYED}, "accumulation_witness": proof,
        "quality": "paired at each timing; identical seed tapes per accumulation arm",
        "speed": "synchronized wall time, native regular timing, warmed CUDA Graph, alternating ABBA and BAAB",
        "sources": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (Path(__file__), Path(__file__).with_name("uncapped_reward_precision.py"))},
    })
    regular = make_evaluators(TIMING)
    long = make_evaluators(DELAYED)
    measured = {"regular": quality(regular, genomes, device, "regular"),
                "long": quality(long, genomes, device, "long")}
    result_path = args.out / "results.json"
    _write(result_path, {"state": "awaiting_speed", "quality": measured})
    training_before = wait_before_speed(args.idle_wait)
    throughput = speed(regular, genomes, device)
    throughput.update(training_before=training_before, training_after=prior_progress())
    _write(result_path, {"state": "complete", "completed_utc": _utc(), "quality": measured, "speed": throughput})
    print(json.dumps({"speed": throughput["candidates_per_second"],
                      "paired_fp16_accum_speedup": throughput["paired_fp16_accum_speedup"]}), flush=True)


if __name__ == "__main__":
    main()
