"""Test predictions about output behavior made from the final birth weights."""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import statistics

import torch

from experiments.seed_evolution.large_runner import _utc, _write
from experiments.undirected import uncapped_bf16_backend as backend
from experiments.undirected.evolved_plasticity_world import _receipt_table
from experiments.undirected.uncapped_plasticity_run import OUT


SAVE = OUT.with_name("2026-11-16-uncapped-recombination") / "weight_predictions.json"
VERSION_RAILS = ((3, 63), (4, 62))
WAIT_RAILS = ((3, 56), (4, 56))
METRICS = ("strict", "completion", "strict_return", "ready_coverage",
           "version_accuracy", "version_coverage", "symbol_accuracy")


class FrozenBirth(backend._BF16Policy):
    def _change_weights(self, local_activity, receipt):
        pass

    def _change_support(self, local_activity, receipt):
        pass


def weight_matrix(genome):
    nodes = genome["node_bias"].numel()
    u, v = torch.triu_indices(nodes, nodes, 1)
    matrix = torch.zeros(nodes, nodes)
    matrix[u, v] = matrix[v, u] = genome["birth_w"]
    return matrix


def change_transmission(graph, edits):
    pairs = [(lane, u, v, factor) for lane, group in enumerate(edits)
             for (u, v), factor in group.items()]
    if not pairs:
        return
    lanes, u, v, scale = zip(*pairs, strict=True)
    lanes = torch.tensor(lanes, device=graph.device)
    u = torch.tensor(u, device=graph.device)
    v = torch.tensor(v, device=graph.device)
    scale = torch.tensor(scale, dtype=torch.bfloat16, device=graph.device).unsqueeze(-1)
    original = graph._update_matrix

    def update():
        original()
        values = graph.matrix[lanes, :, u, v] * scale
        graph.matrix[lanes, :, u, v] = values
        graph.matrix[lanes, :, v, u] = values

    graph._update_matrix = update
    update()


def make_policy(genome, policy, edits, actors, device):
    _, birth = backend._stack_genomes([genome] * len(edits))
    graph = policy({key: value.to(device) for key, value in birth.items()}, actors, device)
    change_transmission(graph, edits)
    return graph


def pulse(genome, device, inputs, outputs, edits, delay):
    graph = make_policy(genome, FrozenBirth, [{}, edits], 2, device)
    cue = torch.zeros((2, 2, 56), device=device, dtype=torch.bfloat16)
    cue[:, 1, list(inputs)] = 1
    graph.act(cue)
    blank = torch.zeros_like(cue)
    for _ in range(delay):
        graph.act(blank)
    selected = graph.h[..., list(outputs)].float()
    return (selected[:, 1] - selected[:, 0]).cpu().tolist()


def compensating_controls(matrix):
    pairs = [(u, v) for u in range(28) if u not in (3, 4)
             for v in range(59, 67) if matrix[u, v] < 0]
    order = sorted(pairs, key=lambda pair: abs(abs(matrix[pair].item()) - .428))
    rng = random.Random(10280203)
    choices = [tuple(rng.sample(order[:min(14, len(order))], 2)) for _ in range(8)]
    return {f"other_negative_{i + 1}": dict.fromkeys(group, 0.) for i, group in enumerate(choices)}


def describe(diffs):
    radius = 2.13145 * statistics.stdev(diffs) / len(diffs) ** .5
    mean = statistics.fmean(diffs)
    return {"mean": mean, "interval95": [mean - radius, mean + radius],
            "up": sum(x > 0 for x in diffs), "down": sum(x < 0 for x in diffs)}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=OUT)
    parser.add_argument("--out", type=Path, default=SAVE)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    device = torch.device("cuda")
    record = torch.load(args.run_dir / "champion.pt", map_location="cpu", weights_only=True)
    genome = record["genome"]
    matrix = weight_matrix(genome)
    conditions = {
        "original": {}, "remove_opponent_suppression": dict.fromkeys(VERSION_RAILS, 0.),
        "reverse_opponent_suppression": dict.fromkeys(VERSION_RAILS, -1.),
        "remove_version_to_wait": dict.fromkeys(WAIT_RAILS, 0.),
        **compensating_controls(matrix),
    }
    predictions = {
        "version": "The two negative edges suppress one prominent opposing-version FINISH each. "
                   "Silencing them reduces version accuracy; reversing their signs hurts more.",
        "wait": "The visible version inhibits WAIT through two negative edges. "
                "Silencing these edges reduces in-window response coverage.",
        "two_hop": "READY drives WAIT through positive 10–38 and 38–56, with no direct edge. "
                   "Silencing 38–56 reduces the delayed READY effect on WAIT.",
        "raw_7": "A raw-7 cue reaches action node 65 through two negative edges, via node 31. "
                 "Silencing 7–31 reduces the delayed positive contribution to 65.",
        "main_rejection": "The main port's rejection input lifts WAIT and suppresses FINISH 66. "
                          "Silencing 23–56 and 23–66 reduces the margin.",
    }
    static_edges = {
        f"{u}-{v}": matrix[u, v].item() for u, v in
        (*VERSION_RAILS, *WAIT_RAILS, (10, 38), (38, 56), (10, 56), (7, 31), (31, 65), (23, 56), (23, 66))
    }
    _write(args.out.with_name("weight_predictions_before_replay.json"), {
        "recorded_utc": _utc(), "candidate": record["row"]["id"], "predictions": predictions,
        "weights": static_edges, "rule": genome["rule"].tolist(), "conditions": {
            name: [[*edge, value] for edge, value in edits.items()] for name, edits in conditions.items()},
    })
    print("Static predictions written", flush=True)
    pulses = {
        "version_0_on_wrong_finish": pulse(genome, device, (3,), (63,), {(3, 63): 0.}, 1),
        "version_1_on_wrong_finish": pulse(genome, device, (4,), (62,), {(4, 62): 0.}, 1),
        "ready_on_wait": pulse(genome, device, (10,), (56,), {(38, 56): 0.}, 2),
        "raw_7_on_finish": pulse(genome, device, (7,), (65,), {(7, 31): 0.}, 2),
        "rejection_on_wait_and_finish": pulse(genome, device, (23,), (56, 66),
                                              {(23, 56): 0., (23, 66): 0.}, 1),
    }
    device_table = _receipt_table(device).bfloat16()
    rows = []
    names = list(conditions)
    for seed in range(98_919_330, 98_919_346):
        _, _, _, _, tape = backend._inputs([genome], seed, 16, device, 8, "full")
        graph = make_policy(genome, backend._BF16Policy, list(conditions.values()), 16, device)
        result = backend._rollout(graph, backend._device_tape(tape, device), device_table)
        columns = {key: result[key].cpu().tolist() for key in METRICS}
        rows.append({"seed": seed, **{name: {key: values[index] for key, values in columns.items()}
                                     for index, name in enumerate(names)}})
    means = {name: {key: statistics.fmean(row[name][key] for row in rows) for key in METRICS}
             for name in names}
    differences = {name: {key: describe([row[name][key] - row["original"][key] for row in rows])
                          for key in METRICS} for name in names if name != "original"}
    _write(args.out, {"candidate": record["row"]["id"], "pulse_responses_original_and_intervened": pulses,
                     "means": means, "comparisons": differences, "paired": rows})
    print(pulses)
    for name, metrics in means.items():
        print(name, {key: round(metrics[key] * 100, 2) for key in METRICS})


if __name__ == "__main__":
    main()
