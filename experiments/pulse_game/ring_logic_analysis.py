"""Matched world/birth inference and reproducibility checks for the ring logic trial."""

from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path
import platform
import time

import numpy as np
import torch

from . import ring_logic_model as model, ring_logic_study as study, ring_logic_world as env


def _observations(out, frozen):
    nested = {}
    for step in study.STEPS:
        received = []
        for artifact in sorted(frozen["files"], key=lambda x: x["first_group"]):
            if artifact["updates"] == step:
                group = artifact["first_group"]
                source = out / "scores" / f"birth-{group:02d}-step-{step}.npz"
                with np.load(source, allow_pickle=False) as data:
                    received.append((artifact, {key: value for key, value in data.items()}))
        nested[step] = received
    return nested


def _joined(observed, metric, all_restarts=False):
    pieces = []
    for artifact, data in observed:
        rows = data[metric]
        if all_restarts:
            rows = rows.reshape(-1, study.RESTARTS, len(env.RULES), len(model.ARMS), *rows.shape[1:])
        else:
            if metric in ("standard", "long"):
                rows = rows[study._selection(artifact)]
            rows = rows.reshape(-1, len(env.RULES), len(model.ARMS), *rows.shape[1:])
        pieces.append(rows)
    return np.concatenate(pieces)


def _estimate(left, right, weights, alpha):
    difference = left.mean(axis=-1) - right.mean(axis=-1)
    births, roots = weights
    draws = ((births @ difference) * roots).sum(axis=-1)
    return {"estimate": float(difference.mean()), "interval": np.quantile(draws, (alpha, 1 - alpha)).tolist()}


def _weights(n_births, n_roots):
    rng = np.random.default_rng(41527306)
    n_draws = study.BOOTSTRAP_DRAWS
    groups = rng.multinomial(n_births, np.full(n_births, 1 / n_births), size=n_draws)
    roots = rng.multinomial(n_roots, np.full(n_roots, 1 / n_roots), size=n_draws)
    births = groups.astype(np.float32) / n_births
    return births, roots.astype(np.float32) / n_roots


def _arm_metrics(chosen, restarts, rule):
    rewards = chosen[..., 2]
    correct = rewards > 0
    by_group = correct.mean(axis=(-1, -2))
    all_correct = restarts[..., 2] > 0
    full_restart = all_correct.mean(axis=(-1, -2))
    best_restart = full_restart.max(axis=1)
    per_cell = correct.mean(axis=-2)
    mask = env.truth()[rule] != env.truth()[rule][env.SWAP]
    swap_joint = correct & correct[..., env.SWAP]
    return {
        "accuracy": float(correct.mean()), "timeout_rate": float((rewards == 0).mean()),
        "selection_birth_accuracy": by_group.tolist(),
        "all_restart_birth_accuracy": full_restart.tolist(),
        "best_restart_accuracy_descriptive": best_restart.tolist(),
        "max_single_restart_accuracy": float(np.max(full_restart)),
        "all_restart_success_95": int(np.sum(full_restart >= .95)),
        "all_restart_total": int(full_restart.size),
        "selected_birth_success_95": int(np.sum(by_group >= .95)),
        "selected_birth_total": int(by_group.size),
        "strict_all_cells_95_by_birth": int(np.sum(per_cell.min(axis=-1) >= .95)),
        "full_20_cell_world_accuracy": float(correct.all(axis=-1).mean()),
        "median_min_cell_accuracy": float(np.median(per_cell.min(axis=-1))),
        "cell_accuracy": correct.mean(axis=(0, 1)).tolist(),
        "swapped_joint_accuracy": float(swap_joint.mean()),
        "swap_label_changes_joint_accuracy": float(swap_joint[..., mask].mean()) if mask.any() else None,
        "swap_label_same_joint_accuracy": float(swap_joint[..., ~mask].mean()) if (~mask).any() else None,
    }


def summarize(out):
    study.require_registered(out)
    frozen = study.freeze_guard(out)
    observed = _observations(out, frozen)
    weights = _weights(study.MAIN_BIRTHS, len(study.roots("confirmation")))
    alpha_primary = .05 / (2 * (3 * len(env.RULES)))
    comparisons = [(a, b) for a, b in combinations(study.PRIMARY_ARMS, 2)]
    comparisons += [("undirected", a) for a in study.PRIMARY_ARMS]
    summary = {"primary_interval": "Family-wise Bonferroni over 9 contrasts (three comparisons by three rules), 95%.",
               "other_intervals": "Individual 95% intervals are descriptive.",
               "contrast_order": "second-named accuracy minus first-named accuracy",
               "steps": {}, "controls": {}}
    for step, records in observed.items():
        profiles = {}
        for profile in ("standard", "long"):
            chosen = _joined(records, profile)
            restarts = _joined(records, profile, all_restarts=True)
            tables = {}
            for r, rule in enumerate(env.RULES):
                contrasts = {}
                metrics = {}
                learnability = {}
                for a, arm in enumerate(model.ARMS):
                    metrics[arm] = _arm_metrics(chosen[:, r, a], restarts[:, :, r, a], r)
                    if step == study.STEPS[-1] and profile == "standard":
                        short = _joined(observed[study.STEPS[0]], profile)
                        learnability[arm] = _estimate(chosen[:, r, a, ..., 2] > 0,
                                                      short[:, r, a, ..., 2] > 0, weights, .025)
                for a, b in comparisons:
                    is_primary = a in study.PRIMARY_ARMS and step == study.STEPS[-1] and profile == "standard"
                    left, right = (chosen[:, r, model.ARMS.index(x), ..., 2] > 0 for x in (a, b))
                    contrasts[f"{b} minus {a}"] = {
                        "co_primary": is_primary,
                        **_estimate(right, left, weights, alpha_primary if is_primary else .025),
                    }
                tables[rule] = {"arms": metrics, "contrasts": contrasts}
                if learnability:
                    tables[rule]["long_minus_short_budget"] = learnability
            profiles[profile] = tables
        summary["steps"][str(step)] = profiles
    high = observed[study.STEPS[-1]]
    intact = _joined(high, "standard")[..., 2] > 0
    for intervention in study.INTERVENTIONS:
        data = _joined(high, intervention)
        correct = data[..., 2] > 0
        contact = data[..., 2] != 0
        if intervention.endswith("_only"):
            if not np.array_equal(2 * correct.sum(axis=-1), contact.sum(axis=-1)):
                raise RuntimeError(f"Observed departure from exact missing-cue chance: {intervention}")
        summary["controls"][intervention] = {
            rule: {arm: {
                "accuracy": float(correct[:, r, a].mean()),
                "contact_rate": float(contact[:, r, a].mean()),
                "drop_from_intact": _estimate(intact[:, r, a], correct[:, r, a], weights, .025),
            } for a, arm in enumerate(model.ARMS)}
            for r, rule in enumerate(env.RULES)}
    with np.load(out / "oracles.npz", allow_pickle=False) as f:
        summary["public_oracle"] = {key: dict(zip(env.RULES, (rows[..., 2] > 0).mean(axis=(1, 2)).tolist()))
                                    for key, rows in f.items()}
    study.write(out / "summary.json", summary)
    return summary


def verify(out, device):
    study.require_registered(out)
    frozen = study.freeze_guard(out)
    outcomes = []
    reference = study.episode("confirmation", "standard")
    cases = ("standard", *study.INTERVENTIONS)
    for artifact in frozen["files"]:
        group, step = artifact["first_group"], artifact["updates"]
        picked = np.arange(len(env.RULES) * len(model.ARMS))
        dest = out / "scores" / f"birth-{group:02d}-step-{step}.npz"
        with np.load(dest, allow_pickle=False) as saved:
            bank, rows = study.load_checkpoint(out / artifact["path"], device, picked)
            different = int(np.count_nonzero(study.score(bank, rows, reference) != saved["standard"][picked]))
            results = {"standard": different}
            if step == study.STEPS[-1]:
                picked_restart = study._selection(artifact)[picked]
                selected, rows = study.load_checkpoint(out / artifact["path"], device, picked_restart)
                for case in cases[1:]:
                    replay = study.score(selected, rows, reference, case)
                    results[case] = int(np.count_nonzero(replay != saved[case][picked]))
        outcomes.append({"first_group": group, "updates": step, "different_values": results})
    data = {"device": str(device), "rows": outcomes,
            "all_values_replayed_exactly": all(v == 0 for row in outcomes for v in row["different_values"].values())}
    study.write(out / "verification.json", data)
    return data


def benchmark(out):
    artifact = study.freeze_guard(out)["files"][-1]
    pick = study._selection(artifact)[:len(model.ARMS)]
    reference = study.episode("confirmation", "standard")
    device = torch.device("cpu")
    # Score an identical first rule/first group and use sparse inference paths
    # specialized to the nonzero edges rather than measuring the mixed-arm bank.
    banks = [study.load_checkpoint(out / artifact["path"], device, [j]) for j in pick]
    for i, (bank, rows) in enumerate(banks):
        np.testing.assert_array_equal(study.score(bank, rows, reference, path=i + 1),
                                      study.score(bank, rows, reference))
    timings = []
    for repeat in range(12):
        indices = np.roll(np.arange(len(banks)), repeat % len(banks))
        for i in indices:
            bank, rows = banks[i]
            begin = time.perf_counter()
            study.score(bank, rows, reference, path=int(i) + 1)
            timings.append({"repeat": repeat, "topology": rows[0][4],
                            "seconds": time.perf_counter() - begin})
    study.write(out / "timing.json", {"platform": platform.platform(),
                                     "cpu_arch": platform.processor(), "threads": 1,
                                     "physical_requests_per_call": len(study.roots("confirmation")) * len(env.PAIRS),
                                     "seconds_per_batch": timings})


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("summarize", "verify", "benchmark"))
    parser.add_argument("--out", type=Path, default=study.DEFAULT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    if args.action == "summarize":
        summarize(args.out)
    elif args.action == "verify":
        verify(args.out, torch.device(args.device))
    else:
        benchmark(args.out)


if __name__ == "__main__":
    main()
