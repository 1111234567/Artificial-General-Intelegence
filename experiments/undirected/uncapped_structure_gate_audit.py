"""Replay archived champions and measure crossings of their lifetime edge gates."""
from __future__ import annotations

import statistics
from pathlib import Path

import torch

from experiments.seed_evolution.large_runner import _utc, _write
from experiments.undirected import uncapped_bf16_backend as backend
from experiments.undirected.evolved_plasticity_world import _receipt_table
from experiments.undirected.uncapped_modularity_swap import archived
from experiments.undirected.uncapped_plasticity_run import OUT, base


SAVE = OUT.with_name("2026-11-16-uncapped-structure-check") / "gate_replay.json"
SEEDS = range(96_712_930, 96_712_938)
THRESHOLDS = (.45, .35, .25)
FIELDS = ("strict", "completion", "ready_coverage", "version_accuracy", "initial_edges",
          "final_edges", "mean_edges", "births", "deaths")


class RecordGate(backend._BF16Policy):
    def __init__(self, birth, actors, device):
        super().__init__(birth, actors, device)
        self.changes = []

    def _change_support(self, local_activity, receipt):
        mask = self.mask.clone()
        super()._change_support(local_activity, receipt)
        self.changes.append((self.step, torch.stack((
            (~mask & self.mask).sum(-1), (mask & ~self.mask).sum(-1),
            self.g.masked_fill(mask, -torch.inf).amax(-1),
            (-self.g).masked_fill(~mask, -torch.inf).amax(-1),
        )).float()))


def finite_numbers(values):
    return [float(value) for value in values if float("-inf") < float(value) < float("inf")]


def extrema_by_actor(gates, bucket):
    return finite_numbers(gates[:, bucket].amax(0).tolist())


def describe(values):
    if not values:
        return None
    return {"minimum": min(values), "median": statistics.median(values), "maximum": max(values)}


@torch.no_grad()
def evaluate(genomes, seed, threshold):
    actors, device, _, birth, tape = backend._inputs(genomes, seed, 16, "cuda", 8, "full")
    original = backend.BIRTH_THRESHOLD, backend.DEATH_THRESHOLD
    try:
        backend.BIRTH_THRESHOLD, backend.DEATH_THRESHOLD = threshold, -threshold
        policy = RecordGate({key: value.to(device) for key, value in birth.items()}, actors, device)
        measures = backend._rollout(policy, backend._device_tape(tape, device), _receipt_table(device).bfloat16())
    finally:
        backend.BIRTH_THRESHOLD, backend.DEATH_THRESHOLD = original
    ticks = [tick for tick, _ in policy.changes]
    gates = torch.stack([data for _, data in policy.changes]).cpu()
    columns = {key: measures[key].cpu().tolist() for key in FIELDS}
    if not torch.equal(gates[:, :2].sum((0, 3)), torch.stack((measures["births"], measures["deaths"])).cpu()):
        raise AssertionError("Recorded per-tick support counts differ from sealed totals")
    rows = []
    for index in range(len(genomes)):
        values = gates[:, :, index]
        birth_events = values[:, 0].sum(-1)
        death_events = values[:, 1].sum(-1)
        row = {key: column[index] for key, column in columns.items()}
        row.update({
            "ticks": ticks, "birth_counts": birth_events.tolist(), "death_counts": death_events.tolist(),
            "dormant_maxima": extrema_by_actor(values, 2), "live_negative_maxima": extrema_by_actor(values, 3),
        })
        rows.append(row)
    return rows


def average(samples):
    means = {name: statistics.fmean(row[name] for row in samples) for name in FIELDS}
    for key in ("dormant_maxima", "live_negative_maxima"):
        means[key] = describe([v for row in samples for v in row[key]])
    for key in ("birth_counts", "death_counts"):
        totals = [sum(row[key][i] for row in samples) for i in range(len(samples[0]["ticks"]))]
        means[f"{key}_per_check"] = totals
        total = sum(totals)
        means[f"{key}_fraction_last_56_ticks"] = (
            sum(count for tick, count in zip(samples[0]["ticks"], totals, strict=True) if tick >= 168) / total
            if total else None
        )
        means[f"{key}_weighted_mean_tick"] = (
            sum(tick * count for tick, count in zip(samples[0]["ticks"], totals, strict=True)) / total
            if total else None
        )
    return means


def main():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    generations = (256, 384, 512, 640, 768, 896, 1024)
    labels = [str(g) for g in generations] + ["pruning_896"]
    identifiers = [base.read(OUT / "audits" / f"{generation:04d}.json")["candidate"]
                   for generation in generations] + ["g895-i04-c0112"]
    genomes = [archived(OUT, g, i) for g, i in zip((*generations, 896), identifiers, strict=True)]
    partitions = {}
    for threshold in THRESHOLDS:
        scores = [evaluate(genomes, seed, threshold) for seed in SEEDS]
        summary = {key: average([row[index] for row in scores]) for index, key in enumerate(labels)}
        partitions[str(threshold)] = {
            "means": summary, "paired": [{**dict(zip(labels, row, strict=True)), "seed": seed}
                                         for seed, row in zip(SEEDS, scores, strict=True)],
        }
        for key, row in summary.items():
            print(threshold, key, {
                k: (round(row[k], 4) if isinstance(row[k], float) else row[k]) for k in
                ("dormant_maxima", "live_negative_maxima", "births", "deaths", "final_edges", "mean_edges",
                 "birth_counts_weighted_mean_tick", "birth_counts_fraction_last_56_ticks", "strict")
            }, flush=True)
    _write(SAVE, {"run": str(OUT), "created_utc": _utc(), "seeds": list(SEEDS), "actors_per_seed": 16,
                  "candidates": dict(zip(labels, identifiers, strict=True)),
                  "gate_rules": {name: genome["rule"][1].tolist()
                                 for name, genome in zip(labels, genomes, strict=True)},
                  "threshold_probes": partitions})


if __name__ == "__main__":
    main()
