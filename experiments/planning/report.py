"""Build a portable evidence bundle from completed, frozen planning runs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(runs, key):
    groups = defaultdict(list)
    for run in runs:
        for row in run["summary"][key]:
            group = tuple(row[name] for name in ("method", "distribution", "family", "depth"))
            groups[group].append(row)
    output = []
    for group, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        item = dict(zip(("method", "distribution", "family", "depth"), group))
        item.update(seeds=len(rows), n_total=sum(row["n"] for row in rows))
        for metric in ("success_rate", "optimal_success_rate", "mean_core_calls",
                       "mean_transition_calls", "mean_latency_ms", "mean_expanded_scored_states"):
            values = [row[metric] for row in rows if row.get(metric) is not None]
            item[metric] = {"mean": mean(values), "sample_sd": stdev(values) if len(values) > 1 else 0.0} if values else None
        output.append(item)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"evidence destination must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    with gzip.open(out / "test_cases.jsonl.gz", "wt", encoding="utf-8") as cases, \
            gzip.open(out / "predictions.jsonl.gz", "wt", encoding="utf-8") as predictions, \
            gzip.open(out / "uniform_predictions.jsonl.gz", "wt", encoding="utf-8") as uniform_predictions:
        for directory in args.runs:
            config = read(directory / "config.json")
            manifest = read(directory / "manifest.json")
            if sha(directory / "test.jsonl") != manifest["datasets"]["test"]["sha256"]:
                raise ValueError("test data no longer matches its frozen manifest")
            for filename, expected in manifest["sources"].items():
                if sha(directory / "source" / filename) != expected:
                    raise ValueError("source snapshot no longer matches its manifest")
            for filename, stream in (("test.jsonl", cases), ("results_test.jsonl", predictions)):
                with (directory / filename).open(encoding="utf-8") as source:
                    for line in source:
                        stream.write(json.dumps(dict(run_seed=config["seed"], **json.loads(line)), separators=(",", ":")) + "\n")
            if (directory / "uniform_search_test.jsonl").exists():
                with (directory / "uniform_search_test.jsonl").open(encoding="utf-8") as source:
                    for line in source:
                        uniform_predictions.write(json.dumps(dict(run_seed=config["seed"], **json.loads(line)), separators=(",", ":")) + "\n")
            runs.append({"directory": directory.resolve().as_posix(), "config": config,
                         "manifest": manifest, "environment": read(directory / "environment.json"),
                         "training": read(directory / "training_summary.json"),
                         "summary": read(directory / "summary_test.json"),
                         "posthoc_search_control": read(directory / "summary_uniform_search_test.json")
                         if (directory / "summary_uniform_search_test.json").exists() else None,
                         "result_sha256": sha(directory / "results_test.jsonl")})
    overall = {"runs": runs, "all_cases": summarize(runs, "all_cases"),
               "paired_search_subset": summarize(runs, "paired_search_subset"),
               "reporter_sha256": sha(Path(__file__)),
               "notes": ["Means and sample standard deviations across training/data seeds.",
                         "Search comparisons use only paired_search_subset.",
                         "Core-layer calls are a proxy, not equal FLOPs or end-to-end latency.",
                         "Greedy batch-amortized and sequential search latencies are not comparable.",
                         "Compressed cases and predictions contain every held-out observation and plan."]}
    overall["posthoc_search_controls"] = summarize(
        [{"summary": run["posthoc_search_control"]} for run in runs if run["posthoc_search_control"]], "all_cases")
    # Preserve development decisions and generator falsification evidence too;
    # the failed/easier generator is not silently erased from the record.
    for filename in ("generator-audit-v1.json", "main-study-v1.json"):
        source = args.runs[0].parent / filename
        if source.exists():
            shutil.copyfile(source, out / filename)
    pilots = []
    for directory in (args.runs[0].parent / "pilot-seed101-v1", args.runs[0].parent / "pilot-seed102-v2"):
        if directory.exists():
            pilots.append({"directory": directory.resolve().as_posix(),
                           "config": read(directory / "config.json"),
                           "training": read(directory / "training_summary.json"),
                           "validation": [json.loads(line) for line in (directory / "validation.jsonl").read_text().splitlines()],
                           "test_evaluated": (directory / "evaluation_test.started.json").exists()})
    overall["validation_only_pilots"] = pilots
    methods = [("shallow_r1", "Shallow"), ("recurrent_r4", "Shared x4"),
               ("untied_r4", "Untied x4"), ("detach_r4", "Shared x4, detach"),
               ("heuristic_relaxed", "Relaxed heuristic"), ("random", "Random")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, family in zip(axes, ("chain", "fork", "cross")):
        for method, label in methods:
            rows = sorted((r for r in overall["all_cases"] if r["method"] == method and r["family"] == family), key=lambda r: r["depth"])
            if rows:
                ax.errorbar([r["depth"] for r in rows], [100*r["success_rate"]["mean"] for r in rows],
                            yerr=[100*r["success_rate"]["sample_sd"] for r in rows], marker="o", ms=3, capsize=2, label=label)
        ax.set(title=family, xlabel="Shortest plan length", ylim=(-3, 103))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Executable plan success (%)")
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle("Consumable resource planning — mean and sample SD across seeds")
    fig.tight_layout()
    fig.savefig(out / "plan_success.png", dpi=180)
    fig.savefig(out / "plan_success.svg")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for distribution, label in (("iid", "In distribution"), ("length_ood", "Longer plans"),
                                ("structural_ood", "Held-out structure, matched length"),
                                ("structural_length_ood", "Held-out structure + length")):
        rows = [r for r in overall["all_cases"] if r["family"] == "all" and r["distribution"] == distribution
                and r["method"].startswith("recurrent_r")]
        rows.sort(key=lambda r: int(r["method"].split("_r")[-1]))
        if rows:
            ax.errorbar([int(r["method"].split("_r")[-1]) for r in rows],
                        [100*r["success_rate"]["mean"] for r in rows],
                        yerr=[100*r["success_rate"]["sample_sd"] for r in rows], marker="o", capsize=3, label=label)
    ax.set(xlabel="Internal rounds at inference (trained at 4)", ylabel="Executable plan success (%)", xticks=[1, 2, 4, 8], ylim=(-3, 103))
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "inference_rounds.png", dpi=180)
    plt.close(fig)
    overall["evidence_hashes"] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    (out / "summary.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for table in ("all_cases", "paired_search_subset"):
        print(table)
        for row in overall[table]:
            if row["family"] == "all" and row["distribution"] != "all":
                metric = row["success_rate"]
                print(f"{row['method']:26s} {row['distribution']:24s} {100*metric['mean']:6.2f} +/- {100*metric['sample_sd']:5.2f} n={row['n_total']}")


if __name__ == "__main__":
    main()
