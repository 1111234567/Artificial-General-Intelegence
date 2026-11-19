"""Compare reciprocal rule swaps and degree-preserving birth-graph crossover."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import random
import statistics

import torch

from experiments.seed_evolution.large_runner import _utc, _write
from experiments.undirected.evolved_plasticity import ACTIONS, INPUTS, _edge_id, _pairs
from experiments.undirected.model import _connected
from experiments.undirected.uncapped_bf16_backend import evaluate_continuous
from experiments.undirected.uncapped_bf16_run import configure
from experiments.undirected.uncapped_plasticity_run import OUT, base


OUTPUT = OUT.with_name("2026-11-16-uncapped-recombination") / "paired.json"
SEEDS = tuple(range(98_722_161, 98_722_173))
PATCH_SEED = 4_251_316
METRICS = (
    "strict", "completion", "strict_return", "strict_both_return", "actors_strict_both_return",
    "ready_coverage", "version_accuracy", "symbol_accuracy", "off_ready_rate",
)
DEVELOPMENT_ORDER = ("strict", "strict_both_return", "strict_return", "completion")


def select_parents(run):
    parents, records = {}, {}
    for generation in (768, 896, 1024):
        name = "alternate_896" if generation == 896 else f"champion_{generation}"
        checkpoint_path = run / "generations" / f"{generation:04d}" / "elites.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        rows = [row for island in checkpoint["islands"] for row in island]
        official_id = base.read(run / "audits" / f"{generation:04d}.json")["candidate"]
        official = next(row for row in rows if row["id"] == official_id)
        if generation == 896:
            viable = [row for row in rows if row["lineage"] != official["lineage"]
                      and row["strict_both_return"] > 0]
            record = max(viable, key=lambda row: tuple(row[key] for key in DEVELOPMENT_ORDER))
        else:
            record = official
        records[name] = {key: record[key] for key in ("id", "lineage", "strict", "strict_both_return")}
        records[name].update(generation=generation, checkpoint=str(checkpoint_path))
        parents[name] = checkpoint["genomes"][record["id"]]
    return parents, records


def switch_candidates(recipient, donor, rng):
    """Find ABAB 4-cycles and ordinary switches sharing their removed edges."""
    nodes = recipient["node_bias"].numel()
    endpoints = _pairs(nodes)
    present = recipient["birth_mask"].tolist()
    other = donor["birth_mask"].tolist()
    inactive_ports = range(INPUTS + ACTIONS, INPUTS + 2 * ACTIONS)
    removed = [index for index, (a, b) in enumerate(endpoints)
               if present[index] and not other[index] and a not in inactive_ports and b not in inactive_ports]
    possible = list(combinations(removed, 2))
    rng.shuffle(possible)
    patches, used = [], set()
    for first, second in possible:
        a, b = endpoints[first]
        c, d = endpoints[second]
        if len({a, b, c, d}) != 4 or used.intersection((a, b, c, d)):
            continue
        options = (
            (_edge_id(nodes, a, c), _edge_id(nodes, b, d)),
            (_edge_id(nodes, a, d), _edge_id(nodes, b, c)),
        )
        if any(present[edge] for option in options for edge in option):
            continue
        shared = tuple(sum(other[edge] for edge in option) for option in options)
        if sorted(shared) != [0, 2]:
            continue
        patches.append(((first, second), options[shared.index(2)], options[shared.index(0)]))
        used.update((a, b, c, d))
        if len(patches) == 4:
            return patches
    raise ValueError("Too few disjoint alternating cycles with matched controls")


def switched(recipient, donor, patches, operation):
    child = {key: value.clone() for key, value in recipient.items()}
    for removed, graft, control in patches:
        inserted = control if operation == "single_parent" else graft
        if operation == "donor_weights":
            values = donor["birth_w"][list(inserted)]
        else:
            values = child["birth_w"][list(removed)]
        child["birth_w"][list(inserted)] = values
        child["birth_w"][list(removed)] = 0
        child["birth_mask"][list(inserted)] = True
        child["birth_mask"][list(removed)] = False

    neighbors = [set() for _ in range(child["node_bias"].numel())]
    for (a, b), present in zip(_pairs(len(neighbors)), child["birth_mask"].tolist(), strict=True):
        if present:
            neighbors[a].add(b)
            neighbors[b].add(a)
    if not all(len(row) == 16 for row in neighbors) or not _connected(neighbors):
        raise ValueError("Crossover must preserve birth degree and connectivity")
    removed_count = int((recipient["birth_mask"] & ~child["birth_mask"]).sum())
    added_count = int((~recipient["birth_mask"] & child["birth_mask"]).sum())
    if removed_count != added_count or added_count != 2 * len(patches):
        raise ValueError("Crossover changed an unexpected number of birth edges")
    return child


def make_variants(parents):
    genomes = dict(parents)
    metadata, comparisons = {}, {}
    rng = random.Random(PATCH_SEED)
    for mate in ("champion_768", "alternate_896"):
        for recipient, donor in (("champion_1024", mate), (mate, "champion_1024")):
            prefix = f"{recipient}_from_{donor}"
            name = f"{prefix}_rule"
            genomes[name] = {**parents[recipient], "rule": parents[donor]["rule"]}
            metadata[name] = {"recipient": recipient, "donor": donor, "operation": "whole_rule"}
            comparisons[name] = (name, recipient)
            cycles = switch_candidates(parents[recipient], parents[donor], rng)
            for count in (1, 4):
                patch = cycles[:count]
                label = f"{prefix}_{count}cycle"
                for operation in ("donor_weights", "carry_weights", "single_parent"):
                    name = f"{label}_{operation}"
                    genomes[name] = switched(parents[recipient], parents[donor], patch, operation)
                    metadata[name] = dict(recipient=recipient, donor=donor, operation=operation,
                                          cycle_count=count, patch_pair_indices=patch)
                    comparisons[name] = (name, recipient)
                comparisons[f"{label}_donor_vs_control"] = (f"{label}_donor_weights", f"{label}_single_parent")
                comparisons[f"{label}_topology_vs_control"] = (f"{label}_carry_weights", f"{label}_single_parent")
                comparisons[f"{label}_weights_vs_carry"] = (f"{label}_donor_weights", f"{label}_carry_weights")
    return genomes, metadata, comparisons


def summarize(paired, comparisons):
    n = len(paired)
    t_value = 2.201  # 95% t interval for 12 paired schedule seeds
    if n != 12:
        raise ValueError("Update paired interval critical value for a different number of seeds")
    result = {}
    for name, (child, reference) in comparisons.items():
        metrics = {}
        for key in METRICS:
            deltas = [row[child][key] - row[reference][key] for row in paired]
            mean = statistics.fmean(deltas)
            margin = t_value * statistics.stdev(deltas) / n ** .5
            metrics[key] = {"delta": mean, "ci95": [mean - margin, mean + margin],
                            "positive_seeds": sum(value > 0 for value in deltas)}
        result[name] = {"child": child, "reference": reference, "metrics": metrics}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=OUT)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    configure()
    parents, sources = select_parents(args.run_dir)
    genomes, metadata, comparisons = make_variants(parents)
    paired = []
    for seed in SEEDS:
        evaluation = evaluate_continuous(list(genomes.values()), seed=seed, actors=16, device="cuda")
        columns = {key: evaluation[key].cpu().tolist() for key in METRICS}
        paired.append({"seed": seed, **{name: {key: values[index] for key, values in columns.items()}
                                      for index, name in enumerate(genomes)}})
    means = {name: {key: statistics.fmean(row[name][key] for row in paired) for key in METRICS}
             for name in genomes}
    _write(args.out, {"created_utc": _utc(), "sources": sources, "precision": "bf16", "actors": 16,
                     "visits": 8, "patch_seed": PATCH_SEED, "variants": metadata, "paired": paired,
                     "means": means, "contrasts": summarize(paired, comparisons)})
    for name, values in means.items():
        print(name, *(f"{key}={100 * values[key]:.2f}%" for key in METRICS[:5]))


if __name__ == "__main__":
    main()
