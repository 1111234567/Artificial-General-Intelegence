"""Describe ancestry signal on shared development tapes from scored batch files.

Run from any directory: python plasticity_lineage_audit.py RUN_DIR --out lineage.json
Uses only the standard library. It does not rerun evaluations or edit the run.
"""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
from statistics import fmean, median


KINDS = ("elite", "local", "wide", "founder")
BASE_METRICS = ("fitness", "completion", "strict", "off_ready_rate")
EXTRA_METRICS = {
    "timing": ("ready_coverage", "timing_clean"),
    "version": ("version_coverage",),
    "symbol": ("symbol_first_coverage", "symbol_late_coverage", "symbol_gain"),
    "conflict": ("strict_b", "strict_return", "return_gain_over_first_a"),
}
PAIR_METRICS = ("fitness", "strict")


def read_rows(run, generation, arm, population):
    directory = run / "generations" / f"{generation:04d}" / "batches" / arm
    rows = []
    for path in sorted(directory.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as file:
            rows.extend(json.load(file))
    if len(rows) != population or len({row["id"] for row in rows}) != population:
        raise ValueError(f"Expected {population} unique candidates in {directory}")
    return rows


def win_counts(value, sorted_comparison):
    wins = bisect_left(sorted_comparison, value)
    ties = bisect_right(sorted_comparison, value) - wins
    return wins, ties


def describe_generation(rows, metrics):
    groups = defaultdict(list)
    for row in rows:
        if row["kind"] not in KINDS or (row["kind"] == "founder") != (row["parent"] is None):
            raise ValueError(f"Unexpected candidate kind/parent: {row['id']}")
        groups[row["kind"]].append(row)
    copies = {row["parent"]: row for row in groups.get("elite", ())}
    founders = {key: sorted(row[key] for row in groups["founder"]) for key in PAIR_METRICS}
    if len(copies) != len(groups.get("elite", ())):
        raise ValueError("Exact-parent matching needs at most one elite copy of each parent")
    results = {}
    for kind, members in groups.items():
        result = {"n": len(members), "metrics": {}}
        for key in metrics:
            values = [row[key] for row in members]
            stats = {"mean": fmean(values), "median": median(values)}
            if key in PAIR_METRICS and kind != "founder":
                counts = [win_counts(value, founders[key]) for value in values]
                stats["vs_founder_win_probability"] = sum(w + .5 * t for w, t in counts) / (
                    len(values) * len(founders[key])
                )
            result["metrics"][key] = stats
        if kind in ("local", "wide"):
            matches = [(row, copies[row["parent"]]) for row in members if row["parent"] in copies]
            result["exact_parent"] = {"n": len(matches), "metrics": {}}
            for key in PAIR_METRICS:
                delta = [child[key] - parent[key] for child, parent in matches]
                if delta:
                    result["exact_parent"]["metrics"][key] = {
                        "mean_delta": fmean(delta),
                        "better": sum(x > 0 for x in delta),
                        "tied": sum(x == 0 for x in delta),
                    }
        results[kind] = result
    return results


def aggregate(generations, metrics):
    out = {}
    for kind in KINDS:
        generations_with_kind = [generation for generation in generations if kind in generation]
        group = [generation[kind] for generation in generations_with_kind]
        if not group:
            continue
        stats = {"n": sum(g["n"] for g in group), "generations": len(group), "metrics": {}}
        for key in metrics:
            per_generation = [g["metrics"][key] for g in group]
            result = {
                "mean_of_generation_means": fmean(m["mean"] for m in per_generation),
                "median_of_generation_medians": median(m["median"] for m in per_generation),
            }
            if kind != "founder":
                delta = [m["mean"] - g["founder"]["metrics"][key]["mean"]
                         for m, g in zip(per_generation, generations_with_kind, strict=True)]
                result["paired_mean_delta_vs_founder"] = fmean(delta)
                result["generations_mean_above_founder"] = sum(x > 0 for x in delta)
                if key in PAIR_METRICS:
                    result["mean_vs_founder_win_probability"] = fmean(
                        m["vs_founder_win_probability"] for m in per_generation
                    )
            stats["metrics"][key] = result
        if kind in ("local", "wide"):
            matches = [g["exact_parent"] for g in group if g["exact_parent"]["n"]]
            n = sum(m["n"] for m in matches)
            stats["exact_parent"] = {"n": n, "metrics": {}}
            for key in PAIR_METRICS:
                if n:
                    pairs = [(m["n"], m["metrics"][key]) for m in matches]
                    stats["exact_parent"]["metrics"][key] = {
                        "mean_delta": sum(count * metric["mean_delta"] for count, metric in pairs) / n,
                        "better_fraction": sum(metric["better"] for _, metric in pairs) / n,
                        "tied_fraction": sum(metric["tied"] for _, metric in pairs) / n,
                    }
        out[kind] = stats
    return out


def audit(run):
    config = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    stages, checkpoints = {}, {}
    birth_mix = {arm: Counter() for arm in config["arms"]}
    for stage, (start, stop) in config["stage_bounds"].items():
        metrics = BASE_METRICS + EXTRA_METRICS[stage]
        by_arm = {arm: [] for arm in config["arms"]}
        for generation in range(start, stop + 1):
            for arm in by_arm:
                rows = read_rows(run, generation, arm, config["population"])
                summary = describe_generation(rows, metrics)
                winner = max(rows, key=lambda row: (row["fitness"], row["completion"],
                                                    row["strict_return"], -row["idle"]))
                birth_mix[arm][winner["kind"]] += 1
                if generation in {start, start + 1, stop}:
                    checkpoints.setdefault(generation, {"stage": stage, "arms": {}})["arms"][arm] = summary
                if generation > 1:
                    by_arm[arm].append(summary)
        stages[stage] = {arm: aggregate(generations, metrics) for arm, generations in by_arm.items()}
    return {
        "run": str(run.resolve()), "protocol": config, "stage_results": stages,
        "checkpoints": checkpoints, "generation_winner_kind": birth_mix,
        "notes": [
            "Within-stage summaries exclude generation 1, which contains only random founders.",
            "Win probability compares two independently drawn rows of the same generation and arm,"
            " one from the named group and one random founder, counting ties as half.",
            "Exact-parent comparisons use only mutants whose parent was copied unmodified and evaluated"
            " on the same development tape. This subset favors selected parents and excludes parents"
            " without a copy lane; mutant/parent comparisons do not isolate the learned rule.",
            "Successive generations/relatives are dependent. These are descriptive stats of one run."
            " Fitness differs by stage, and development metrics are not holdout metrics.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = json.dumps(audit(args.run), indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result, encoding="utf-8")
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
