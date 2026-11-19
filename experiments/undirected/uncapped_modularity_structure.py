"""Compare the completed uncapped run's inherited graphs with matched structural nulls."""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from statistics import fmean, median

import networkx as nx
import numpy as np
import torch

from .model import connected_regular_mask
from .uncapped_plasticity_run import OUT, rank


TARGET = OUT.with_name("2026-11-16-uncapped-modularity") / "structure.json"
NODE_COUNT = 80
LEFT, RIGHT = np.triu_indices(NODE_COUNT, 1)
PORTS = {
    "port_zero": [*range(28), *range(56, 67)],
    "port_one": [*range(28, 56), *range(67, 78)],
    "undriven": [78, 79],
}
ROLES = {
    "port_zero_timing_version": [0, 3, 4, 9, 10],
    "port_zero_raw_symbol": list(range(5, 9)),
    "port_zero_context": list(range(24, 26)),
    "port_zero_receipt": list(range(11, 24)),
    "port_zero_reserved_observation": [1, 2, 26, 27],
    "port_zero_wait_begin": list(range(56, 59)),
    "port_zero_finish_v0": list(range(59, 63)),
    "port_zero_finish_v1": list(range(63, 67)),
    "port_one_receipt": list(range(39, 52)),
    "port_one_reserved_observation": [*range(28, 39), *range(52, 56)],
    "port_one_wait_begin": list(range(67, 70)),
    "port_one_finish_v0": list(range(70, 74)),
    "port_one_finish_v1": list(range(74, 78)),
    "undriven": [78, 79],
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def groups_by_label(groups):
    labels = np.full(NODE_COUNT, -1, dtype=np.int64)
    for i, group in enumerate(groups):
        nodes = list(group)
        if (labels[nodes] >= 0).any():
            raise ValueError("groups overlap")
        labels[nodes] = i
    if (labels < 0).any():
        raise ValueError("groups do not cover every node")
    return labels


def modularity(u, v, w, labels):
    total = float(w.sum())
    if total == 0:
        return 0.0
    count = int(labels.max()) + 1
    endpoint_mass = np.bincount(labels[u], weights=w, minlength=count)
    endpoint_mass += np.bincount(labels[v], weights=w, minlength=count)
    within = float(w[labels[u] == labels[v]].sum())
    return float(within / total - ((endpoint_mass / (2 * total)) ** 2).sum())


def scores_for_groups(u, v, w, labels):
    positive = np.maximum(w, 0)
    negative = np.maximum(-w, 0)
    same = labels[u] == labels[v]
    q_pos = modularity(u, v, positive, labels)
    q_neg = modularity(u, v, negative, labels)
    return {
        "unweighted_q": modularity(u, v, np.ones_like(w), labels),
        "abs_weight_q": modularity(u, v, np.abs(w), labels),
        "positive_q": q_pos,
        "negative_q": q_neg,
        "signed_q": float((positive.sum() * q_pos - negative.sum() * q_neg) / np.abs(w).sum()),
        "within_edge_fraction": float(same.mean()),
        "within_abs_mass_fraction": float(np.abs(w[same]).sum() / np.abs(w).sum()),
        "within_positive_mass_fraction": float(positive[same].sum() / positive.sum()),
        "within_negative_mass_fraction": float(negative[same].sum() / negative.sum()),
    }


def adjusted_rand(a, b):
    columns = int(b.max()) + 1
    counts = np.bincount(a * columns + b, minlength=(int(a.max()) + 1) * columns).reshape(-1, columns)

    def pairs(values):
        return float((values * (values - 1)).sum() / 2)

    row, col = pairs(counts.sum(1)), pairs(counts.sum(0))
    chance = row * col / pairs(np.array([len(a)]))
    upper = (row + col) / 2
    return (pairs(counts) - chance) / (upper - chance) if upper > chance else 1.0


def detect(edges, abs_weights, restarts):
    graph = nx.Graph()
    graph.add_nodes_from(range(NODE_COUNT))
    if abs_weights is None:
        graph.add_edges_from(edges)
    else:
        graph.add_weighted_edges_from((*edge, float(weight)) for edge, weight in zip(edges, abs_weights, strict=True))
    weight_key = None if abs_weights is None else "weight"
    outcomes = []
    for i in range(restarts):
        groups = nx.community.louvain_communities(graph, weight=weight_key, seed=20261116 + i)
        score = nx.community.modularity(graph, groups, weight=weight_key)
        outcomes.append((float(score), groups))
    best_score, best_groups = max(outcomes, key=lambda outcome: outcome[0])
    return best_score, sorted((sorted(group) for group in best_groups), key=lambda group: (-len(group), group)), outcomes


def pair_arrays(edges):
    return np.asarray(edges, dtype=np.int64).T


def archive_samples(run):
    final = torch.load(run / "champion.pt", map_location="cpu", weights_only=True)
    quiet = "g895-i04-c0112"
    labels = {1: "first_audited_winner", 640: "g640_audited_winner", 768: "g768_audited_winner", 896: "g896_audited_winner"}
    result = [("final_champion", 1024, "champion.pt", final["row"], final["genome"])]
    for generation in (*labels, 1024):
        path = Path("generations") / f"{generation:04d}" / "elites.pt"
        checkpoint = torch.load(run / path, map_location="cpu", weights_only=True)
        rows = [row for island in checkpoint["islands"] for row in island]
        if generation == 1024:
            same = checkpoint["genomes"][final["row"]["id"]]
            if not all(torch.equal(value, same[key]) for key, value in final["genome"].items()):
                raise ValueError("champion disagrees with final elites checkpoint")
            other = max((row for row in rows if row["lineage"] != final["row"]["lineage"]), key=rank)
            chosen = [("final_other_lineage_best_elite", other)]
        else:
            audited = read(run / "audits" / f"{generation:04d}.json")["candidate"]
            chosen = [(labels[generation], next(row for row in rows if row["id"] == audited))]
            if generation == 896:
                chosen.append(("g896_pruning_elite", next(row for row in rows if row["id"] == quiet)))
        result.extend((label, generation, path.as_posix(), row, checkpoint["genomes"][row["id"]]) for label, row in chosen)
    return result


def summary(values, observed):
    n = len(values)
    return {
        "n": n,
        "mean": fmean(values),
        "median": median(values),
        "central_95pct": np.quantile(values, [.025, .975]).tolist(),
        "maximum": max(values),
        "p_upper_monte_carlo": (1 + sum(value >= observed - 1e-12 for value in values)) / (n + 1),
        "values": values,
    }


def annotate_groups(groups, u, v, w):
    for group in groups:
        inside = np.isin(u, group) & np.isin(v, group)
        cross = np.isin(u, group) ^ np.isin(v, group)
        yield {
            "nodes": group,
            "role_counts": {role: len(set(group) & set(nodes)) for role, nodes in ROLES.items() if set(group) & set(nodes)},
            "within_edges": int(inside.sum()),
            "crossing_edges": int(cross.sum()),
            "within_abs_mass": float(np.abs(w[inside]).sum()),
            "crossing_abs_mass": float(np.abs(w[cross]).sum()),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=OUT)
    parser.add_argument("--output", type=Path, default=TARGET)
    parser.add_argument("--replicates", type=int, default=64)
    parser.add_argument("--restarts", type=int, default=8)
    args = parser.parse_args()
    if args.replicates < 1 or args.restarts < 2:
        parser.error("Replicates must be positive and at least two restarts are required.")
    torch.set_num_threads(1)
    rng = np.random.default_rng(2026111601)
    protocol = read(args.run / "protocol.json")
    progress = read(args.run / "progress.json")
    if (protocol["nodes"], protocol["birth_edges"], progress["state"]) != (80, 640, "complete"):
        raise ValueError("expected the complete 80-node, 640-birth-edge run")
    nulls = {"degree16_regular": [], "degree16_initializer": []}
    for _ in range(args.replicates):
        regular = nx.random_regular_graph(16, NODE_COUNT, seed=int(rng.integers(2**31)))
        if not nx.is_connected(regular):
            raise ValueError("degree-matched null disconnected")
        nulls["degree16_regular"].append(sorted((min(u, v), max(u, v)) for u, v in regular.edges))
        mask = connected_regular_mask(NODE_COUNT, 16, seed=int(rng.integers(2**31)), switches_per_edge=2).numpy()
        nulls["degree16_initializer"].append(list(zip(LEFT[mask], RIGHT[mask], strict=True)))
    raw_nulls = {key: [detect(edges, None, args.restarts)[0] for edges in group] for key, group in nulls.items()}
    fixed = {"ports": groups_by_label(PORTS.values()), "roles": groups_by_label(ROLES.values())}
    matrices, candidates, source_files = {}, {}, {}
    for label, generation, source, row, genome in archive_samples(args.run):
        source_files.setdefault(source, sha256((args.run / source).read_bytes()).hexdigest())
        print(f"{label}: {row['id']}", flush=True)
        mask = genome["birth_mask"].numpy()
        u, v = LEFT[mask], RIGHT[mask]
        raw_weights = genome["birth_w"].numpy()
        w = raw_weights[mask].astype(np.float64)
        if np.any(raw_weights[~mask]):
            raise ValueError("inactive weights must be zero")
        edges = list(zip(u, v, strict=True))
        if len(edges) != 640 or (np.bincount(np.concatenate((u, v)), minlength=NODE_COUNT) != 16).any():
            raise ValueError("expected a degree-16, 640-edge birth graph")
        observed = {}
        for measure in ("unweighted", "abs_weight"):
            score, groups, outcomes = detect(edges, None if measure == "unweighted" else np.abs(w), args.restarts)
            labels = [groups_by_label(g) for _, g in outcomes]
            metrics = scores_for_groups(u, v, w, groups_by_label(groups))
            if not np.isclose(score, metrics[f"{measure}_q"], rtol=0, atol=1e-12):
                raise ValueError("detector and independent modularity calculation disagree")
            observed[measure] = {
                "max_q": score,
                "min_q": min(q for q, _ in outcomes),
                "detector_restart_pairwise_ari_median": median(adjusted_rand(a, b) for a, b in combinations(labels, 2)),
                "communities": list(annotate_groups(groups, u, v, w)),
                **metrics,
            }
            matrices.setdefault(label, {})[measure] = groups_by_label(groups)
        fixed_observed = {key: scores_for_groups(u, v, w, labels) for key, labels in fixed.items()}
        comparisons = {}
        for null_name, null_edges in {**nulls, "observed_topology_shuffled_weights": [edges] * args.replicates}.items():
            abs_q = []
            fixed_null = {key: {metric: [] for metric in ("unweighted_q", "abs_weight_q", "signed_q")} for key in fixed}
            for edge_set in null_edges:
                shuffled = rng.permutation(w)
                abs_q.append(detect(edge_set, np.abs(shuffled), args.restarts)[0])
                nu, nv = pair_arrays(edge_set)
                for key, labels in fixed.items():
                    metrics = scores_for_groups(nu, nv, shuffled, labels)
                    for metric in fixed_null[key]:
                        fixed_null[key][metric].append(metrics[metric])
            comparisons[null_name] = {
                "abs_weight": summary(abs_q, observed["abs_weight"]["max_q"]),
                "fixed": {key: {metric: summary(values, fixed_observed[key][metric]) for metric, values in results.items()} for key, results in fixed_null.items()},
            }
            if null_name in raw_nulls:
                comparisons[null_name]["unweighted"] = summary(raw_nulls[null_name], observed["unweighted"]["max_q"])
        audit_file = args.run / "audits" / f"{generation:04d}.json"
        audit = read(audit_file)
        candidates[label] = {
            "archive_generation": generation,
            "source": source,
            "development_row": row,
            "published_holdout": audit["modes"] if audit["candidate"] == row["id"] else None,
            "birth": {"edges": len(edges), "positive": int((w > 0).sum()), "negative": int((w < 0).sum()), "zero_weight_live": int((w == 0).sum())},
            "detected": observed,
            "fixed_partitions": fixed_observed,
            "nulls": comparisons,
        }
        print(f"  unweighted Q {observed['unweighted']['max_q']:.4f}, absolute Q {observed['abs_weight']['max_q']:.4f}", flush=True)
    result = {
        "run": str(args.run.resolve()),
        "birth_checkpoint_sha256": source_files,
        "analysis": {
            "networkx": nx.__version__,
            "replicates": args.replicates,
            "louvain_restarts": args.restarts,
            "louvain_resolution": 1,
            "louvain_seed_start": 20261116,
            "null_rng_seed": 2026111601,
            "degree16_regular": "NetworkX random_regular_graph, n=80 and degree=16. For weighted comparisons permute the measured candidate's signed weights onto null edges.",
            "degree16_initializer": "The exact founder-mask sampler, two attempted switches per edge. For weighted comparisons permute the measured candidate's signed weights onto null edges.",
            "observed_topology_shuffled_weights": "Keep the entire observed support fixed and independently permute the exact signed weights of the candidate.",
            "signed_q": "Positive-layer Q and negative-magnitude-layer Q use each layer's own expected mass from vertex strengths. Signed Q is their difference after normalizing by total absolute mass.",
            "scope": "Archived birth genotypes, one run, selected and related candidates. Nominal upper-tail probabilities are exploratory and unadjusted. Communities are fit on graph alone. A/B contexts share the same fixed port-zero indices.",
        },
        "fixed_ports": PORTS,
        "fixed_roles": ROLES,
        "partition_agreement": {
            left: {right: {measure: adjusted_rand(groups[measure], other[measure]) for measure in groups} for right, other in matrices.items()}
            for left, groups in matrices.items()
        },
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
