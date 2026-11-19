"""Replicate the editor transplant across all seven remaining birth graphs."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from . import self_write_transfer as transfer


REPLICATION = transfer.PLAN.parent / "replication"


def replicate_birth(birth, original, checkpoint_dir, output, view):
    seed = birth["bootstrap"]
    plan = {
        **original,
        "bootstrap_checkpoint": f"own-view bootstrap root {seed}",
        "donor_root": birth["donor_root"],
        "recipient_root": birth["recipient_root"],
    }
    directory = output / str(seed)
    directory.mkdir(parents=True, exist_ok=True)
    per_birth_plan = directory / "preregistered.json"
    per_birth_plan.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    checkpoint = checkpoint_dir / f"bootstrap-own-{seed}.npz"
    transfer.run_assay(checkpoint, per_birth_plan, directory, view=view)


def hierarchical_draws(root_count, donor_count, pair_count, seed, samples):
    rng = np.random.default_rng(seed)
    root_draw = rng.integers(root_count, size=(samples, root_count, 1, 1))
    donor_draw = rng.integers(donor_count, size=(samples, root_count, donor_count, 1))
    pair_draw = rng.integers(pair_count, size=(samples, root_count, donor_count, pair_count))
    return root_draw, donor_draw, pair_draw


def aggregate(traces, original, protocol, connected_by_regime=None):
    root_count = len(traces)
    donors = original["donors_per_regime"]
    pairs = original["fresh_world_pairs_per_donor"]
    root_draw, donor_draw, pair_draw = hierarchical_draws(
        root_count, donors, pairs, protocol["hierarchical_bootstrap_seed"], protocol["bootstrap_draws"],
    )
    results = {}
    for regime in original["donor_regimes"]:
        results[regime] = {}
        if connected_by_regime is not None:
            connected = connected_by_regime[regime]
            eligible = np.array([np.flatnonzero(row).tolist() + [-1] * (~row).sum() for row in connected])
            counts = connected.sum(axis=1)[root_draw]
            rng = np.random.default_rng(protocol["hierarchical_bootstrap_seed"] + 1)
            index = np.floor(rng.random(donor_draw.shape) * counts).astype(np.intp)
            conditional_draw = eligible[root_draw, index]
            draw_mask = np.arange(donors)[None, None, :, None] < counts
        for noise in original["primary_noise"]:
            cohort = f"{regime}|{noise}|{original['primary_condition']}"
            contrasts = {}
            results[regime][str(noise)] = contrasts
            for comparison in original["primary_contrasts"]:
                main, control = comparison.split(" - ")
                score_delta = np.stack([
                    raw[f"{cohort}|{main}|query_score"] - raw[f"{cohort}|{control}|query_score"]
                    for raw in traces
                ])
                gain = score_delta[:, :, :, 1:13].mean(axis=3) - score_delta[:, :, :, 0]
                complements = gain.reshape(root_count, donors, pairs, 2, -1).mean(axis=3)
                statistics = {}
                contrasts[comparison] = statistics
                for phase in original["primary_phases"]:
                    values = complements[:, :, :, phase]
                    if connected_by_regime is None:
                        boot = values[root_draw, donor_draw, pair_draw].mean(axis=(1, 2, 3))
                        birth_means = values.mean(axis=(1, 2))
                    else:
                        selected = values[root_draw, conditional_draw, pair_draw]
                        per_root = (selected * draw_mask).sum(axis=(2, 3)) / (pairs * counts[:, :, 0, 0])
                        boot = per_root.mean(axis=1)
                        birth_means = np.array([values[i, row].mean() for i, row in enumerate(connected)])
                    statistics[str(phase)] = {
                        "mean": float(birth_means.mean()),
                        "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
                        "birth_means": birth_means.tolist(),
                    }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=REPLICATION / "protocol.json")
    parser.add_argument("--output", type=Path, default=REPLICATION)
    parser.add_argument("--checkpoints", type=Path, default=REPLICATION.parents[1] / "core")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    original_bytes = transfer.PLAN.read_bytes()
    original = json.loads(original_bytes)
    if not args.aggregate_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = [
                pool.submit(replicate_birth, birth, original, args.checkpoints, args.output, protocol["view"])
                for birth in protocol["births"]
            ]
            for job in jobs:
                job.result()
    seeds = [birth["bootstrap"] for birth in protocol["births"]]
    raw_paths = [args.output / str(seed) / "transfer_traces.npz" for seed in seeds]
    per_birth = [
        json.loads((args.output / str(seed) / "transfer_results.json").read_bytes()) for seed in seeds
    ]
    evidence = []
    for seed, raw, birth in zip(seeds, raw_paths, per_birth, strict=True):
        checkpoint = args.checkpoints / f"bootstrap-own-{seed}.npz"
        checkpoint_sha256 = sha256(checkpoint.read_bytes()).hexdigest()
        raw_sha256 = sha256(raw.read_bytes()).hexdigest()
        if checkpoint_sha256 != birth["checkpoint_sha256"] or raw_sha256 != birth["traces_sha256"]:
            raise ValueError(f"Checkpoint or trace digest mismatched for root {seed}")
        evidence.append({"seed": seed, "checkpoint_sha256": checkpoint_sha256, "traces_sha256": raw_sha256})
    connected = {
        regime: np.array([birth["results"][regime]["donors"]["intact_connected"] for birth in per_birth])
        for regime in original["donor_regimes"]
    }
    traces = [np.load(raw, allow_pickle=False) for raw in raw_paths]
    try:
        results = aggregate(traces, original, protocol)
        connected_only = aggregate(traces, original, protocol, connected)
    finally:
        for raw in traces:
            raw.close()
    report = {
        "protocol_sha256": sha256(protocol_bytes).hexdigest(),
        "original_protocol_sha256": sha256(original_bytes).hexdigest(),
        "birth_evidence": evidence,
        "primary_results": results,
        "posthoc_connected_only": {
            "reason": "Eight grafts across seven births lost strong connectivity under writer-only transplant.",
            "connected_donors_by_birth": {regime: masks.sum(axis=1).tolist() for regime, masks in connected.items()},
            "results": connected_only,
        },
    }
    (args.output / "replication_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
