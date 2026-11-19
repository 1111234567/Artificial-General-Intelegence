"""Verify an execution policy on sealed batches of the running private-symbol search."""

from __future__ import annotations

import argparse
import hashlib
from importlib import import_module
import json
from pathlib import Path
from statistics import fmean, stdev
from time import perf_counter

import torch

from . import uncapped_directed_compact_execution_run as compact
from . import uncapped_private_symbol_fast_run as fast
from . import uncapped_private_symbol_run as run
from .uncapped_directed_packed import PackedResults
from .uncapped_private_symbol_delta import delta_rollout


GENERATION = 55
JOURNAL_OFFSETS = (640, 896, 1344, 1856)


def load_classes(specs):
    classes = []
    for spec in specs:
        module, sep, name = spec.partition(":")
        if not sep:
            raise ValueError(f"Expected module:Class, got {spec!r}")
        cls = getattr(import_module(module), name)
        if not isinstance(cls, type):
            raise TypeError(f"Expected a class in {spec!r}")
        classes.append(cls)
    return tuple(classes)


class CandidateEvaluation(fast.TunedEvaluation):
    def __init__(self, policy, capture, **options):
        super().__init__(run.world.training_schedule, **options)
        self.policy_type = policy
        self.capture_type = capture

    def __call__(self, *args, **kwargs):
        original = compact._CachedCapture
        compact._CachedCapture = self.capture_type
        try:
            return super().__call__(*args, **kwargs)
        finally:
            compact._CachedCapture = original


def verify_tensors(expected, actual, label):
    if expected.keys() != actual.keys():
        raise ValueError(f"{label}: tensor keys differ")
    for name, value in expected.items():
        other = actual[name]
        if value.dtype != other.dtype or value.shape != other.shape:
            raise ValueError(f"{label}: tensor metadata differs for {name}")
        left = value.contiguous().reshape(-1).view(torch.uint8)
        right = other.contiguous().reshape(-1).view(torch.uint8)
        if not torch.equal(left, right):
            raise ValueError(f"{label}: tensor bits differ for {name}")


def score_rows(fields, entries, pairs):
    rows = fast.unpack_scores_lists(fields, len(entries))
    return [{**entry, **run.score_fields(row, pairs)} for entry, row in zip(entries, rows, strict=True)]


def read_main_batches(arm):
    read = run.runner.base.read
    config = read(arm / "protocol.json")
    folder = arm / "generations" / f"{GENERATION:04d}"
    parent_checkpoint = folder.with_name(f"{GENERATION - 1:04d}") / "elites.pt"
    saved_parents = folder / "parent-deltas.pt"
    plan = read(folder / "plan.json")
    previous = torch.load(parent_checkpoint, map_location="cpu", weights_only=True)
    saved = torch.load(saved_parents, map_location="cpu", weights_only=True)
    parents = {ident: previous["genomes"][ident] for ident in sorted(saved["deltas"])}
    previous["learned_deltas"] = saved["deltas"]
    samples = []
    for offset in JOURNAL_OFFSETS:
        group = plan[offset:offset + config["batch_size"]]
        entries = group[:config["eval_batch_size"]]
        journal = run.runner.read_batch(folder / "batches" / f"{offset:04d}.json.gz", group)[:len(entries)]
        genomes = [run.breeding.candidate(entry, previous, config["nodes"]) for entry in entries]
        samples.append((entries, genomes, journal))
    seeds = [read(arm / "generations" / f"{generation:04d}" / "summary.json")["development_seed"]
             for generation in range(GENERATION - 7, GENERATION + 1)]
    with parent_checkpoint.open("rb") as stream:
        checkpoint_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return config, samples, seeds, parents, saved, checkpoint_digest


def verify_scores(training, samples, settings, config):
    metric_count = 0
    for offset, (entries, genomes, journal) in zip(JOURNAL_OFFSETS, samples, strict=True):
        control, candidate = (evaluator(genomes, **settings) for evaluator in training)
        if not isinstance(control, PackedResults) or not isinstance(candidate, PackedResults):
            raise ValueError("Expected the captured packed CUDA score path")
        verify_tensors(control.cpu_fields(), candidate.cpu_fields(), f"journal {offset:04d}")
        for label, result in (("baseline", control), ("candidate", candidate)):
            for actual, expected in zip(score_rows(result, entries, config["edge_pairs"]), journal, strict=True):
                if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
                    different = sorted(key for key in actual.keys() | expected.keys()
                                       if json.dumps(actual.get(key), sort_keys=True)
                                       != json.dumps(expected.get(key), sort_keys=True))
                    raise ValueError(f"{label} differs from journal for {expected['id']}: {different}")
        metric_count = len(control)
    return metric_count


def paired_timings(training, samples, seeds, world_settings):
    def sweep(evaluator, seed):
        for entries, genomes, _ in samples:
            fields = evaluator(genomes, seed=seed, capture=True, **world_settings)
            fast.unpack_scores_lists(fields, len(entries))

    rows = []
    sync = torch.cuda.synchronize
    for index, seed in enumerate(seeds):
        outer = index % len(training)
        inner = 1 - outer
        for evaluator in (training[outer], training[inner]):
            sweep(evaluator, seed)
        seconds = ([], [])
        for variant in (outer, inner, inner, outer):
            sync()
            start = perf_counter()
            sweep(training[variant], seed)
            sync()
            seconds[variant].append(perf_counter() - start)
        baseline, candidate = seconds
        rows.append({"seed": seed, "baseline_seconds": baseline, "candidate_seconds": candidate,
                     "speedup": fmean(baseline) / fmean(candidate)})
    ratios = [row["speedup"] for row in rows]
    center = fmean(ratios)
    # Two-sided Student t interval for the eight tape-seed pairing units.
    radius = 2.364624251 * stdev(ratios) / len(ratios) ** .5
    count = sum(len(entries) for entries, _, _ in samples)
    rates = {label: 60 * count / fmean(t for row in rows for t in row[f"{label}_seconds"])
             for label in ("baseline", "candidate")}
    return {"paired": rows, "speedup": center, "interval95": [center - radius, center + radius],
            "candidates_per_minute": rates}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=run.OUT)
    parser.add_argument("--policy", action="append", required=True,
                        help="candidate policy mixin module:Class, repeat in MRO order")
    parser.add_argument("--capture-mixin", action="append", default=[],
                        help="candidate capture mixin module:Class, repeat in MRO order")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    policy = type("VerificationPolicy", (*load_classes(args.policy), fast._TunedPolicy), {})
    capture = type("VerificationCapture", (*load_classes(args.capture_mixin), compact._CachedCapture), {})
    arm = args.run_dir.resolve() / "pair-01" / "downhill"
    config, samples, seeds, parents, saved, checkpoint_digest = read_main_batches(arm)
    settings = {key: config[key] for key in ("actors", "device", "visits")}

    def evaluators(**options):
        return (fast.TunedEvaluation(run.world.training_schedule, **options),
                CandidateEvaluation(policy, capture, **options))

    training = evaluators()
    score_settings = {**settings, "seed": seeds[-1], "capture": True}
    metric_count = verify_scores(training, samples, score_settings, config)
    timings = paired_timings(training, samples, seeds, settings)
    for evaluator in training:
        evaluator.clear_cache()
    for label, evaluator in zip(("baseline", "candidate"), evaluators(rollout=delta_rollout), strict=True):
        deltas = fast.padded_sample_deltas(evaluator, parents, seed=saved["seed"], **settings)
        verify_tensors(saved["deltas"], deltas, f"{label} parent-deltas.pt")
        evaluator.clear_cache()
    report = {
        "run_arm": str(arm), "generation": GENERATION, "parent_checkpoint_sha256": checkpoint_digest,
        "development_seed": seeds[-1], "journal_offsets": JOURNAL_OFFSETS,
        "actors": settings["actors"], "visits": settings["visits"], "eval_batch_size": config["eval_batch_size"],
        "exact_metric_fields": metric_count, "exact_journal_rows": sum(len(entries) for entries, _, _ in samples),
        "exact_parent_deltas": len(parents), "parent_delta_seed": saved["seed"],
        "torch_version": str(torch.__version__), "device": torch.cuda.get_device_name(),
        "policy": args.policy, "capture_mixin": args.capture_mixin, "timing": timings,
    }
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
