"""Summarize paired physical order outcomes and benchmark frozen inference."""

from argparse import ArgumentParser
from pathlib import Path
import time

import numpy as np
import torch

from . import order_model as model, order_study as study, order_world as env


BOOTSTRAPS = 8192


class CrossedBootstrap:
    def __init__(self, births, roots):
        rng = np.random.default_rng(2791368)
        self.births = births
        self.roots = roots
        self.birth_weights = rng.multinomial(births, np.full(births, 1 / births), size=BOOTSTRAPS) / births
        self.root_weights = rng.multinomial(roots, np.full(roots, 1 / roots), size=BOOTSTRAPS) / roots

    def __call__(self, values):
        if values.shape != (self.births, self.roots):
            raise ValueError(f"Expected one outcome per birth and world, got {values.shape}")
        samples = ((self.birth_weights @ values) * self.root_weights).sum(1)
        return {"mean": float(values.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist(),
                "per_birth": values.mean(axis=1).tolist()}


def accuracy(values):
    return values[..., 0].mean(axis=-1)


def summarize_profile(out, profile, interval):
    with np.load(out / f"{profile}-worlds.npz") as worlds:
        if not np.array_equal(worlds["roots"], study.TEST):
            raise ValueError("Mismatched world roots")
        diagnostics = {name: {"accuracy": float(worlds[name][..., 0].mean()),
                             "contact_rate": float((worlds[name][..., 1] >= 0).mean())}
                       for name in ("oracle", "explorer", "oracle_swap")}
        symbols = worlds["target_symbols"]
        goal = worlds["goal_side"]
        gaps = np.diff(worlds["target_ticks"].astype(int), axis=-1)[..., 0]
        data = {"ground_truth_pair_counts": {f"{i}->{j}": int(((symbols[..., 0] == i) & (symbols[..., 1] == j)).sum())
                                            for i in range(env.ODORS) for j in range(env.ODORS) if i != j},
                "gap_counts": {str(int(gap)): int((gaps == gap).sum()) for gap in np.unique(gaps)},
                "true_right": float(goal.mean()), "diagnostic": diagnostics}
    records = []
    for birth in study.BIRTHS:
        with np.load(out / "births" / str(birth) / f"{profile}.npz") as raw:
            records.append(dict(raw.items()))
    values = {key: np.stack([record[key] for record in records]) for key in records[0]}
    data["condition"] = {}
    measurements = {}
    for arm in study.TOPOLOGIES:
        intact = values[f"{arm}_intact"]
        measurements[arm] = accuracy(intact)
        coverage = (intact[..., 1] >= 0).mean(axis=-1)
        section = {"intact": interval(measurements[arm]), "contact_rate": interval(coverage),
                   "timeouts": int((intact[..., 1] == -1).sum()), "opportunities": intact[..., 1].size}
        for condition in ("drop_candidate_edge", "erase_history", "no_recurrence"):
            intervention = values[f"{arm}_{condition}"]
            changed = accuracy(intervention)
            section[condition] = {"accuracy": interval(changed), "intact_minus_changed": interval(measurements[arm] - changed),
                                  "contact_rate": interval((intervention[..., 1] >= 0).mean(axis=-1))}
        base = intact[:, :, -1]
        counterfactual = values[f"{arm}_swap"]
        both_contacts = (base[..., 1] >= 0) & (counterfactual[..., 1] >= 0)
        section["swap"] = {
            "base_accuracy": interval(base[..., 0]), "counterfactual_accuracy": interval(counterfactual[..., 0]),
            "both_correct": interval((base[..., 0] == 1) & (counterfactual[..., 0] == 1)),
            "opposite_physical_contacts": interval((base[..., 1] != counterfactual[..., 1]) & both_contacts),
        }
        for condition in ("first_only", "second_only", "timing_only"):
            original = values[f"{arm}_{condition}"]
            twin = values[f"{arm}_{condition}_flip"]
            exact_chance = (original[..., 0] + twin[..., 0]) / 2
            if not np.array_equal(exact_chance, .5 * (original[..., 1] >= 0)):
                raise ValueError(f"{arm}: {condition} failed exact chance check")
            section[condition] = {"paired_accuracy": interval(exact_chance), "physical_contact_rate":
                                  interval(original[..., 1] >= 0)}
        data["condition"][arm] = section
    data["graph_minus_self"] = interval(measurements["graph"] - measurements["self"])
    return data


def benchmark(out):
    """Balance run order and measure exact frozen CPU paths with presampled worlds."""
    tape = env.make(study.TEST, study.TEST_REQUESTS, "long")
    manifest = study.read(out / "frozen.json")
    nets = {arm: study.load_net(out, study.BIRTHS[0], arm, manifest) for arm in study.TOPOLOGIES}
    reference = {arm: study.outcomes(model.run(net, tape)) for arm, net in nets.items()}
    block = []
    opportunities = tape.body.trials * tape.body.n
    for repeat in range(12):
        sequence = study.TOPOLOGIES if repeat % 2 == 0 else study.TOPOLOGIES[::-1]
        record = {}
        for arm in sequence:
            clock = time.perf_counter()
            actual = model.run(nets[arm], tape)
            seconds = time.perf_counter() - clock
            if not np.array_equal(study.outcomes(actual), reference[arm]):
                raise RuntimeError(f"Benchmark replay mismatch for {arm}")
            record[arm] = seconds
        block.append(record)
    ratio = np.asarray([pair["graph"] / pair["self"] for pair in block])
    rng = np.random.default_rng(854789)
    weights = rng.integers(len(block), size=(4096, len(block)))
    return {"workload": "192 presampled worlds x 12 long requests, full sensory controller and exact physical scoring",
            "torch_threads": torch.get_num_threads(), "timing_excludes": "tape generation and warmup",
            "wall_seconds_by_block": block,
            "median_opportunities_per_second": {
                arm: float(opportunities / np.median([pair[arm] for pair in block])) for arm in study.TOPOLOGIES},
            "graph_over_self_wall_ratio": {"geometric_mean": float(np.exp(np.log(ratio).mean())),
                                           "ci95": np.exp(np.quantile(np.log(ratio[weights]).mean(axis=1), [.025, .975])).tolist()}}


def summarize(out, evidence):
    manifest = study.read(out / "frozen.json")
    interval = CrossedBootstrap(len(study.BIRTHS), len(study.TEST))
    results = {"births": list(study.BIRTHS), "root_count": len(study.TEST), "trials_per_root": study.TEST_REQUESTS,
               "registration_sha256": manifest["registered_sha256"], "recipe_sha256": manifest["recipe_sha256"],
               "freeze_sha256": study.digest(out / "frozen.json"),
               "primary": "long/graph_minus_self",
               "estimator": "Percentile 95% crossed bootstrap on matched birth pairs and independent world roots. Each world's requests are averaged first.",
               "bootstrap_replicates": BOOTSTRAPS,
               "profile": {profile: summarize_profile(out, profile, interval) for profile in env.TIMING},
               "benchmark": benchmark(out)}
    study.write(out / "summary.json", results)
    if evidence:
        study.write(evidence / "order-summary.json", results)
        for member in ("registered.json", "amendments.json", "recipe.json", "frozen.json"):
            study.write(evidence / f"order-{member}", study.read(out / member))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=study.DEFAULT)
    parser.add_argument("--evidence", type=Path)
    flags = parser.parse_args()
    torch.set_num_threads(1)
    summarize(flags.out, flags.evidence)
