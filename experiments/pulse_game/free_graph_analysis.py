"""Report fixed free-degree graph results with crossed paired bootstrap intervals."""

from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from . import free_graph_study as study, ring_logic_study as original, ring_logic_world as env
from .ring_logic_analysis import _arm_metrics


def _weights(n_roots):
    rng = np.random.default_rng(study.BOOTSTRAP_SEED)
    sizes = (study.GROUPS, n_roots)
    return tuple(rng.multinomial(n, np.full(n, 1 / n), size=study.BOOTSTRAP_DRAWS)
                 .astype(np.float32) / n for n in sizes)


def _estimate(difference, weights, family):
    births, roots = weights
    draws = np.sum((births @ difference.astype(np.float32)) * roots, axis=1)
    tail = .05 / (2 * family)
    return {"estimate": float(difference.mean()), "interval": np.quantile(draws, (tail, 1 - tail)).tolist(),
            "family_size": family}


def _read_scores(out, frozen, step, profile, design):
    arms = study.DESIGNS[design]
    selected = []
    all_restarts = []
    for item in sorted(frozen["files"], key=lambda row: row["first_group"]):
        if item["updates"] != step or item["design"] != design:
            continue
        with np.load(study.score_path(out, item), allow_pickle=False) as saved:
            raw = saved[profile]
        selected.append(raw[study.selection_indices(item)].reshape(-1, len(env.RULES), len(arms), *raw.shape[1:]))
        all_restarts.append(raw.reshape(-1, study.RESTARTS, len(env.RULES), len(arms), *raw.shape[1:]))
    return np.concatenate(selected), np.concatenate(all_restarts)


def _outcomes(correct):
    return {"accuracy": correct.mean(axis=-1), "all_20": correct.all(axis=-1).astype(np.int8)}


def _graph_totals(out):
    graph_rows = original.read(out / "graphs.json")
    result = {}
    for arm in study.DESIGNS["tied"]:
        graphs = [row for row in graph_rows if row["arm"] == arm]
        summary = {"graphs": len(graphs), "strongly_connected": sum(g["strongly_connected"] for g in graphs),
                   "all_32_backbone_arcs": sum(g["ordered_backbone_arcs_present"] == 32 for g in graphs)}
        for key in ("in_degree", "out_degree", "shortest_directed_return"):
            values, counts = np.unique([value for row in graphs for value in row[key]], return_counts=True)
            summary[key + "_node_counts"] = dict(zip(values.astype(str).tolist(), counts.tolist(), strict=True))
        summary["backbone_arcs_per_graph"] = [row["ordered_backbone_arcs_present"] for row in graphs]
        result[arm] = summary
    return result


def summarize(out):
    frozen = study.freeze_guard(out)
    results = {
        "intervals": "95% percentile intervals from independently crossed resamples of paired birth groups and world roots. Separate Bonferroni families at the 1200 standard checkpoint for cell accuracy and all-20 root accuracy, within each parameter design and for the design interaction. Early and long-gap contrasts are individual intervals.",
        "contrast_order": "Second-named minus first-named; fractions of physical opportunities or fully correct roots.",
        "selection": "Calibration-selected outcomes carry inference. Holdout maxima over all restarts are descriptive existence witnesses.",
        "graphs": _graph_totals(out), "steps": {},
    }
    for budget in study.STEPS:
        step = {}
        results["steps"][str(budget)] = step
        for profile, stage in (("standard", "confirmation"), ("long", "long")):
            data = {}
            step[profile] = data
            confirmatory = budget == study.STEPS[-1] and profile == "standard"
            weights = _weights(len(study.roots(stage)))
            outcomes = {}
            for design, arms in study.DESIGNS.items():
                selected, restarts = _read_scores(out, frozen, budget, profile, design)
                correct = selected[..., 2] > 0
                outcomes[design] = _outcomes(correct)
                family = len(arms) * (len(arms) - 1) // 2 * len(env.RULES) if confirmatory else 1
                data[design] = {}
                for r, rule in enumerate(env.RULES):
                    metrics = {}
                    for i, arm in enumerate(arms):
                        stats = _arm_metrics(selected[:, r, i], restarts[:, :, r, i], r)
                        solved = (restarts[:, :, r, i, ..., 2] > 0).all(axis=-1)
                        stats["all_20_roots_by_restart"] = solved.sum(axis=-1).tolist()
                        stats["all_20_roots_by_selected_birth"] = correct[:, r, i].all(axis=-1).sum(axis=-1).tolist()
                        metrics[arm] = stats
                    contrasts = {f"{arms[j]} minus {arms[i]}": {
                        key: _estimate(score[:, r, j] - score[:, r, i], weights, family)
                        for key, score in outcomes[design].items()}
                        for i, j in combinations(range(len(arms)), 2)}
                    data[design][rule] = {"arms": metrics, "contrasts": contrasts}
            family = len(env.RULES) if confirmatory else 1
            data["independent_minus_tied_graph_effect"] = {
                rule: {endpoint: _estimate(
                    (outcomes["independent"][endpoint][:, r, -1] - outcomes["independent"][endpoint][:, r, 0])
                    - (outcomes["tied"][endpoint][:, r, -1] - outcomes["tied"][endpoint][:, r, 0]),
                    weights, family) for endpoint in outcomes["tied"]}
                for r, rule in enumerate(env.RULES)}
    with np.load(out / "oracles.npz", allow_pickle=False) as references:
        results["public_oracle"] = {profile: dict(zip(env.RULES, (refs[..., 2] > 0).mean(axis=(1, 2)).tolist(),
                                                     strict=True)) for profile, refs in references.items()}
    original.write(out / "summary.json", results)
    return results


def verify(out, device):
    frozen = study.freeze_guard(out)
    suites = {"standard": study.scoring_data(study.roots("confirmation"), "standard"),
              "long": study.scoring_data(study.roots("long"), "long")}
    rows = []
    for artifact in frozen["files"]:
        pick = np.arange(2 * len(env.RULES) * len(study.DESIGNS[artifact["design"]]))
        bank, specs = study.load_checkpoint(out / artifact["path"], device, pick)
        with np.load(study.score_path(out, artifact), allow_pickle=False) as scores:
            rows.append({"first_group": artifact["first_group"], "design": artifact["design"],
                         "updates": artifact["updates"], "replayed_rows": len(pick),
                         "different_values": {profile: int(np.count_nonzero(
                             original.score(bank, specs, episode, batch_roots=4) != scores[profile][pick]))
                                              for profile, episode in suites.items()}})
    passed = all(not any(row["different_values"].values()) for row in rows)
    original.write(out / "verification.json", {"device": str(device), "all_exact": passed, "rows": rows})
    if not passed:
        raise RuntimeError("At least one recorded contact did not replay exactly")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("summarize", "verify"))
    parser.add_argument("--out", type=Path, default=study.DEFAULT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    if args.action == "summarize":
        summarize(args.out)
    else:
        verify(args.out, torch.device(args.device))


if __name__ == "__main__":
    main()
