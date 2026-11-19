"""Probe hidden-node alignment and directional perturbations of the sealed IID champions."""
from __future__ import annotations

import argparse
from functools import lru_cache, partial
import hashlib
import json
from pathlib import Path

import torch

from experiments.seed_evolution.large_runner import _utc, _write
from . import uncapped_directed_function_geometry_audit as geometry
from . import uncapped_private_symbol_iid_run as run
from . import uncapped_private_symbol_iid_world as world
from .uncapped_private_symbol_selection import selected_score


BRANCH = run.OUT / "pair-01" / "downhill"
GENERATION = 1024
DEFAULT_OUTPUT = BRANCH / "audits" / "1024-iid-function-geometry-2026-11-19.json"
TRACE_SEEDS = tuple(range(97_419_300, 97_419_304))
DISCOVERY_SEEDS = tuple(range(97_419_400, 97_419_412))
HOLDOUT_SEEDS = tuple(range(97_419_500, 97_419_524))
PRIMARY = "selection_score"
NATIVE_FIELDS = ("strict", "completion", "strict_both_return", "actors_strict_both_return", "symbol_late_mean",
                 "post_answer_actions", "ready_coverage", "premature_request_rate", "idle")
FIELDS = (PRIMARY, *NATIVE_FIELDS)


def private_evaluator(*, cache_size=32):
    return run.previous_run.PrivateEvaluation(lru_cache(maxsize=cache_size)(world.schedule),
                                              rollout=partial(run._rollout, training=False))


class IIDEvaluator(geometry.Evaluator):
    def __init__(self, device, batch_size):
        super().__init__(device, batch_size, actors=run.ACTORS, visits=run.VISITS,
                         evaluator=private_evaluator(), fields=NATIVE_FIELDS)

    def score_rows(self, values, count):
        rows = super().score_rows(values, count)
        for row in rows:
            row["post_answer_actions"] = int(row["post_answer_actions"])
            row[PRIMARY], _ = selected_score(row, run.REQUESTS, run.RULE)
        return rows


def verify_archive(evaluator, parents, meta, archive):
    labels = tuple(f"champion_{i}" for i in range(4))
    originals = archive["conditions"]["regular_native"]["modes"]["full"]["islands"]
    seeds = (archive["seeds"][0], archive["seeds"][-1])
    rows = evaluator.measure({key: parents[key] for key in labels}, seeds, "sealed-baseline replay")
    for i, name in enumerate(labels):
        if meta[name]["id"] != archive["champions"][str(i)]:
            raise ValueError("Checkpoint rank does not reproduce the sealed champion IDs")
        for seed in seeds:
            prior = next(row for row in originals[str(i)]["paired_scores"] if row["seed"] == seed)
            for key in NATIVE_FIELDS:
                if prior[key] != rows[seed][name][key]:
                    raise ValueError(f"Sealed IID audit replay differs on {name}, {seed}, {key}")
    return {"seeds": seeds, "champion_ids": [meta[name]["id"] for name in labels],
            "exact_native_fields": NATIVE_FIELDS, "paired_scores": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.batch_size < 1 or args.output.exists():
        parser.error("Require a positive batch size and a fresh output path")
    torch.set_num_threads(4)
    parents, meta, hashes = geometry.load_parents(BRANCH, GENERATION, run.previous_run.rank)
    path = BRANCH / "audits" / f"{GENERATION:04d}.json"
    raw_archive = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(raw_archive).hexdigest()
    archive = json.loads(raw_archive)
    evaluator = IIDEvaluator(args.device, args.batch_size)
    verification = verify_archive(evaluator, parents, meta, archive)
    orders = [geometry.random_relabel(871_000 + i) for i in range(4)]
    relabelled = {f"self_relabel_{i}": geometry.relabel(parents[f"champion_{i}"], order)
                  for i, order in enumerate(orders)}
    traces, witness = geometry.record_traces(parents, relabelled, orders, args.device,
                                              evaluator=private_evaluator(cache_size=4), seeds=TRACE_SEEDS,
                                              actors=run.ACTORS, visits=run.VISITS)
    print("Recorded IID hidden trajectories and relabeling witness", flush=True)
    crosses, cross_specs, alignments, self_checks = geometry.crossover_cases(parents, relabelled, orders, traces)
    perturbations, perturb_specs, discovery, discovery_norms = geometry.perturbation_cases(
        parents, evaluator, discovery_seeds=DISCOVERY_SEEDS, endpoint=PRIMARY)
    crossover_rows = evaluator.measure(crosses, HOLDOUT_SEEDS, "IID crossover holdout")
    perturbation_rows = evaluator.measure(perturbations, HOLDOUT_SEEDS, "IID perturbation holdout")
    cross_results = geometry.summarize_cross(crossover_rows, cross_specs, FIELDS)
    perturbation_results = geometry.summarize_perturbations(perturbation_rows, FIELDS)

    def baseline(world_rows):
        return {name: {field: sum(row[name][field] for row in world_rows.values()) / len(world_rows)
                       for field in FIELDS} for name in parents if name in next(iter(world_rows.values()))}

    source_files = (Path(__file__), Path(geometry.__file__))
    _write(args.output, {
        "schema": 1, "completed_utc": _utc(), "generation": GENERATION, "branch": str(BRANCH),
        "archive_sha256": hashes, "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "device": args.device, "device_name": torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
        "torch_version": torch.__version__, "batch_size": args.batch_size, "lives_per_world": run.ACTORS,
        "visits": run.VISITS, "requests_per_life": run.VISITS * run.N_SYMBOLS,
        "schedule": "IID per-request A/B, native deadlines and real action-dependent receipts",
        "precision": "sealed IID holdout evaluator, FP16 state and FP32 matmul accumulation",
        "timing": world.TIMING, "parents": meta, "primary_endpoint": PRIMARY, "selection_rule": run.RULE,
        "strict_both_return": "Minimum of the two context-specific strict return rates in the sealed metric.",
        "intervals": "95% percentile bootstrap over paired held-out world seeds. Parents, donor pairs, crossover draws "
                     "and perturbation directions are fixed. Intervals are nominal and not adjusted for multiplicity.",
        "sealed_reference_verification": verification,
        "trace_discovery": {"world_seeds": TRACE_SEEDS, "ticks_in_request": geometry.TRACE_TICKS,
                            "hidden_nodes": list(range(geometry.PORTS, geometry.NODES)), "permutation_witness": witness},
        "crossover": {"world_seeds": HOLDOUT_SEEDS, "base_parents": baseline(crossover_rows), "alignments": alignments,
                      "self_checks": self_checks, "candidate_specs": cross_specs, "paired_scores": crossover_rows,
                      "paired_contrasts": cross_results},
        "perturbation": {"discovery_world_seeds": DISCOVERY_SEEDS, "holdout_world_seeds": HOLDOUT_SEEDS,
                         "birth_weight_l2": geometry.RADIUS, "uniform_scope_headroom": geometry.HEADROOM,
                         "discovery_direction_count": geometry.DISCOVERY_DIRECTIONS,
                         "random_control_count": geometry.RANDOM_DIRECTIONS, "base_parents": baseline(perturbation_rows),
                         "parent_specs": perturb_specs, "discovery_norms": discovery_norms,
                         "discovery_paired_scores": discovery, "holdout_paired_scores": perturbation_rows,
                         "holdout_paired_contrasts": perturbation_results},
    })
    for group, keys in ((cross_results, ("self/relabelled_parent_minus_primary", "self/raw_minus_primary",
                        "self/aligned_minus_primary", "inter/raw_minus_primary", "intra/raw_minus_primary",
                        "inter/aligned_minus_raw", "intra/aligned_minus_raw", "inter_minus_intra/raw")),
                        (perturbation_results, ("plus_minus_parent", "minus_minus_parent", "plus_minus_minus",
                        "plus_minus_random_plus", "directional_advantage_vs_random"))):
        for key in keys:
            print(key, {metric: group[key][metric] for metric in (PRIMARY, "strict", "strict_both_return")}, flush=True)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
