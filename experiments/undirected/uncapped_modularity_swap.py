"""Test whether an unused output port's behavior transfers between saved genotypes."""
from __future__ import annotations

import argparse
from pathlib import Path
import statistics

import torch

from experiments.seed_evolution.large_runner import _utc, _write
from experiments.undirected.evolved_plasticity import ACTIONS, INPUTS
from experiments.undirected.evolved_plasticity_world import FEATURES_PER_SLOT, RECEIPT, _receipt_table
from experiments.undirected.plasticity_curriculum_run import candidate
from experiments.undirected import uncapped_bf16_backend as backend
from experiments.undirected.uncapped_plasticity_run import OUT, base


OUTPUT = OUT.with_name("2026-11-16-uncapped-modularity") / "port_transfer.json"
KEYS = ("strict", "completion", "strict_return", "version_accuracy", "version_coverage",
        "symbol_accuracy", "idle", "spare_rate", "off_ready_rate", "timing_clean")


def archived(path, generation, identifier):
    folder = path / "generations" / f"{generation:04d}"
    checkpoint = torch.load(folder / "elites.pt", map_location="cpu", weights_only=True)
    if identifier in checkpoint["genomes"]:
        return checkpoint["genomes"][identifier]
    previous = torch.load(folder.parent / f"{generation - 1:04d}" / "elites.pt",
                          map_location="cpu", weights_only=True)
    row = next(entry for entry in base.read(folder / "plan.json") if entry["id"] == identifier)
    return candidate(row, previous, checkpoint["genomes"][next(iter(checkpoint["genomes"]))]["node_bias"].numel())


def graft(recipient, donor, nodes, *, biases, edges):
    graph = {key: value.clone() for key, value in recipient.items()}
    if biases:
        graph["node_bias"][nodes] = donor["node_bias"][nodes]
    if edges:
        endpoints = torch.triu_indices(len(graph["node_bias"]), len(graph["node_bias"]), offset=1)
        selected = torch.isin(endpoints, nodes).any(dim=0)
        graph["birth_mask"][selected] = donor["birth_mask"][selected]
        graph["birth_w"][selected] = donor["birth_w"][selected]
    return graph


def variants(champion, quiet, final):
    port = torch.arange(INPUTS + ACTIONS, INPUTS + 2 * ACTIONS)
    models = {"champion_896": champion, "quiet_896": quiet, "champion_1024": final}
    for name, genome in (("champion_896", champion), ("champion_1024", final)):
        for operation, bias, edges in (("bias", True, False), ("edges", False, True), ("port", True, True)):
            models[f"{name}_with_quiet_{operation}"] = graft(genome, quiet, port, biases=bias, edges=edges)
    models["quiet_896_with_champion_port"] = graft(quiet, champion, port, biases=True, edges=True)

    # Control for edits of the same size without specifically targeting port two.
    # Use each recipient's actual numbers of changes to its edge and bias tensors.
    rng = torch.Generator().manual_seed(1_937_632)
    all_nodes = torch.arange(len(champion["node_bias"]))
    all_edges = torch.triu_indices(len(all_nodes), len(all_nodes), offset=1)
    outside_port = ~torch.isin(all_edges, port).any(dim=0)
    outside_nodes = all_nodes[~torch.isin(all_nodes, port)]
    target = models["champion_896_with_quiet_port"]
    kinds = (
        (~champion["birth_mask"]) & quiet["birth_mask"],
        champion["birth_mask"] & (~quiet["birth_mask"]),
        champion["birth_mask"] & quiet["birth_mask"] & (champion["birth_w"] != quiet["birth_w"]),
    )
    for index in range(6):
        graph = {key: value.clone() for key, value in champion.items()}
        bias_count = int(target["node_bias"].ne(champion["node_bias"]).sum())
        different_biases = outside_nodes[champion["node_bias"][outside_nodes] != quiet["node_bias"][outside_nodes]]
        controls = different_biases[torch.randperm(len(different_biases), generator=rng)[:bias_count]]
        graph["node_bias"][controls] = quiet["node_bias"][controls]
        changed = target["birth_mask"].ne(champion["birth_mask"]) | target["birth_w"].ne(champion["birth_w"])
        for kind in kinds:
            count = int((changed & kind).sum())
            available = (kind & outside_port).nonzero().flatten()
            if len(available) < count:
                raise ValueError("Too few eligible control edges")
            sampled = available[torch.randperm(len(available), generator=rng)[:count]]
            graph["birth_mask"][sampled] = quiet["birth_mask"][sampled]
            graph["birth_w"][sampled] = quiet["birth_w"][sampled]
        models[f"champion_896_control_{index+1}"] = graph
    return models


def summarize(models, rows):
    comparisons = {}
    for name in models:
        baseline = "_".join(name.split("_")[:2])
        if name == baseline:
            continue
        fields = {}
        for key in KEYS:
            values = [row[name][key] - row[baseline][key] for row in rows]
            fields[key] = {"mean": statistics.fmean(values),
                           "sem": statistics.stdev(values) / len(rows) ** .5,
                           "positive_seeds": sum(value > 0 for value in values)}
        comparisons[name] = fields
    return comparisons


class SpareFeedbackProbe:
    variants = ("original", "silent", "echo", "ports_only", "rule_only")

    def __init__(self, graph, table):
        self.graph = graph
        self.population = graph.population
        self.table = table
        variant = torch.arange(graph.population, device=graph.device) % len(self.variants)
        self.silent = variant[:, None].ne(0)
        variant = variant[:, None, None]
        self.port_echo = variant.eq(2) | variant.eq(3)
        self.rule_echo = variant.eq(2) | variant.eq(4)
        self.previous = torch.zeros(graph.population, graph.actors, dtype=torch.long, device=graph.device)
        self.native_receipt = graph._receipt
        graph._receipt = self.receipt

    def receipt(self, _obs):
        return self.native_receipt(self.rule_observation)

    def act(self, obs):
        start = FEATURES_PER_SLOT + RECEIPT.start
        stop = FEATURES_PER_SLOT + RECEIPT.stop
        echo = self.table[self.previous]
        self.rule_observation = obs.clone()
        self.rule_observation[..., start:stop] = torch.where(self.rule_echo, echo, obs[..., start:stop])
        obs[..., start:stop] = torch.where(self.port_echo, echo, obs[..., start:stop])
        action = self.graph.act(obs)
        self.previous.copy_(action[..., 1])
        action[..., 1].masked_fill_(self.silent, 0)
        return action


@torch.no_grad()
def feedback_probe(models, seeds):
    """Suppress the spare actions, with a control preserving their original feedback."""
    names = [f"{key}_{label}" for key in models for label in SpareFeedbackProbe.variants]
    genomes = [genome for genome in models.values() for _ in SpareFeedbackProbe.variants]
    rows = []
    for seed in seeds:
        actors, device, _, birth, tape = backend._inputs(genomes, seed, 16, "cuda", 8, "full")
        graph = backend._BF16Policy({key: value.to(device) for key, value in birth.items()}, actors, device)
        table = _receipt_table(device).bfloat16()
        rollout = backend._rollout_with_preloaded_receipts(table)
        metrics = rollout(SpareFeedbackProbe(graph, table), backend._device_tape(tape, device), len(genomes))
        columns = {key: metrics[key].cpu().tolist() for key in KEYS}
        rows.append({"seed": seed, **{name: {key: column[i] for key, column in columns.items()}
                                     for i, name in enumerate(names)}})
    means = {name: {key: statistics.fmean(row[name][key] for row in rows) for key in KEYS} for name in names}
    for key in models:
        for field in ("strict", "completion", "strict_return", "version_accuracy"):
            assert all(row[f"{key}_original"][field] == row[f"{key}_echo"][field] for row in rows)
    return {"means": means, "paired": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=OUT)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    primary_id, quiet_id = "g895-i06-c0042", "g895-i04-c0112"
    primary = archived(args.run_dir, 896, primary_id)
    quiet = archived(args.run_dir, 896, quiet_id)
    final = torch.load(args.run_dir / "champion.pt", map_location="cpu", weights_only=True)
    models = variants(primary, quiet, final["genome"])
    rows = []
    seeds = range(89_170_021, 89_170_033)
    for seed in seeds:
        result = backend.evaluate_continuous(list(models.values()), seed=seed, actors=16, device="cuda")
        columns = {key: result[key].cpu().tolist() for key in KEYS}
        rows.append({"seed": seed, **{name: {key: column[i] for key, column in columns.items()}
                                   for i, name in enumerate(models)}})
    means = {name: {key: statistics.fmean(row[name][key] for row in rows) for key in KEYS} for name in models}
    result = {"created_utc": _utc(), "precision": "bf16", "actors": 16,
              "source_candidates": [primary_id, quiet_id, final["row"]["id"]],
              "birth_edges": {name: int(genome["birth_mask"].sum()) for name, genome in models.items()},
              "means": means, "paired": rows, "contrasts": summarize(models, rows)}
    result["spare_feedback"] = feedback_probe({key: models[key] for key in
                                                ("champion_896", "quiet_896", "champion_1024")}, seeds)
    _write(args.out, result)
    for name, metrics in means.items():
        print(name, " ".join(f"{key}={100 * metrics[key]:.2f}%" for key in
                             ("strict", "completion", "version_accuracy", "spare_rate", "off_ready_rate")), flush=True)
    for name, metrics in result["spare_feedback"]["means"].items():
        print(name, *(f"{key}={100 * metrics[key]:.2f}%" for key in ("strict", "completion", "spare_rate")),
              flush=True)


if __name__ == "__main__":
    main()
