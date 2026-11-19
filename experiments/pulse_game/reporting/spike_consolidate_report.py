"""Report predeclared gates and summarize the consolidation experiments."""

import argparse
import json
from pathlib import Path

import numpy as np

from ..spike_analysis import estimates


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def gate_checks(confirm, replay):
    confirmation = confirm["paired_analysis"]["pairwise_candidate_minus"]
    fixed = replay["paired_analysis"]["pairwise_candidate_minus"]
    check = {
        "closed_loop_old_positive_interval": confirmation["ordinary"]["old_revisit"]["ci95"][0] > 0,
        "closed_loop_old_effect_at_least_15pp": confirmation["ordinary"]["old_revisit"]["mean"] >= .15,
        "closed_loop_new_noninferiority": confirmation["ordinary"]["new_revisit"]["ci95"][0] >= -.07,
        "closed_loop_new_vs_stopped": confirmation["candidate-stop-before-new"]["new_revisit"]["mean"] >= .10
        and confirmation["candidate-stop-before-new"]["new_revisit"]["ci95"][0] > 0,
        "closed_loop_old_new_vs_yoke": all(confirmation["candidate-yoked"][name]["ci95"][0] > 0
                                          for name in ("old_revisit", "new_revisit")),
        "replayed_old_vs_budget_controls": all(fixed[name]["old_revisit"]["mean"] >= .10
                                                and fixed[name]["old_revisit"]["ci95"][0] > 0
                                                for name in ("ordinary-budget-matched", "permuted-budget-matched")),
        "replayed_new_noninferiority": fixed["ordinary-budget-matched"]["new_revisit"]["ci95"][0] >= -.07,
    }
    statistics = {}
    for name in dict.fromkeys(row["name"] for row in replay["results"]):
        rows = [row for row in replay["results"] if row["name"] == name]
        statistics[name] = {
            "writes": float(np.mean([item["updates"] for row in rows for item in row["weights"]])),
            "error_max": max((row.get("write_budget_match_max_abs", 0.0) for row in rows)),
            "mean_l2_write_sum": float(np.mean([item["l2_write_sum"] for row in rows for item in row["weights"]])),
            "mean_coordinates_changed": float(np.mean([item["weight_coordinates_changed"] for row in rows
                                                      for item in row["weights"]])),
            "mean_lateral_updates": float(np.mean([item["covariance_updates"] for row in rows
                                                   for item in row["weights"]])),
        }
    retention = {}
    for method in ("candidate", "ordinary"):
        full, stopped = [np.mean([row["scores"]["old_revisit"] for row in confirm["results"]
                                 if row["name"] == arm], axis=0)
                         for arm in (method, method + "-stop-before-new")]
        retention[method] = full - stopped
    retention = {"old_after_new_minus_own_stop": {method: estimates(values) for method, values in retention.items()},
                 "difference_in_interference": estimates(retention["candidate"] - retention["ordinary"])}
    return {"preregistered_gates": check, "replay_budget_summary": statistics, "retention": retention}


def plot(discovery, confirm, replay, partitions, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    plt.rcParams.update({"font.size": 10})
    older, newer = "#256b65", "#8366ad"

    def quiet_scores(ax, artifact, names, labels):
        arms = artifact["paired_analysis"]["per_arm"]
        for i, (dimension, color, label) in enumerate((
            ("old_revisit", older, "Old cues"), ("new_revisit", newer, "New cues")
        )):
            for j, name in enumerate(names):
                score = arms[name][dimension]
                if name == "candidate-yoked":
                    worlds = np.mean([row["scores"][dimension] for row in artifact["results"]
                                      if row["name"] == name], axis=0)
                    score = estimates(worlds.reshape(-1, 2).mean(axis=1))
                mean = score["mean"]
                a, b = score["ci95"]
                ax.errorbar(mean, j + (.13 if i == 0 else -.13),
                            xerr=[[mean - a], [b - mean]], color=color,
                            marker="o" if i == 0 else "s", ms=4, lw=1.4, capsize=2,
                            label=label if j == 0 else None)
        ax.set_yticks(range(len(names)), labels)
        ax.set_ylim(len(names) - .5, -.5)
        ax.set_xlim(.39, 1)
        ax.axvline(.5, color="#666666", ls=":", lw=.9)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("Correct first contact on quiet trials")
        ax.grid(axis="x", color="#ddd", lw=.7)
        ax.legend(loc="lower right", frameon=True, framealpha=.96, edgecolor="none")

    ax = axes[0, 0]
    quiet_scores(ax, confirm, ("candidate", "ordinary", "candidate-stop-before-new", "candidate-yoked"),
                 ("Activity filter", "Ordinary learning", "Stop before new cues", "Yoked receipts"))
    ax.set_title("A   Independent closed-loop confirmation", loc="left", fontweight="bold")
    ax = axes[0, 1]
    quiet_scores(ax, replay, ("candidate-same-motion", "ordinary-same-motion", "ordinary-budget-matched",
                              "permuted-budget-matched", "diag-budget-matched"),
                 ("Activity filter", "Ordinary, original writes", "Ordinary, same write norm",
                  "Permuted, same write norm", "Diagonal, same write norm"))
    ax.set_xlim(.60, 1)
    ax.set_title("B   Identical training actions and receipts", loc="left", fontweight="bold")

    ax = axes[1, 0]
    designs = tuple(dict.fromkeys(row["name"] for row in discovery["results"]))
    for name in designs:
        points = [row["scores"] for row in discovery["results"] if row["name"] == name]
        old = np.mean([x["old_revisit"] for x in points])
        new = np.mean([x["new_revisit"] for x in points])
        known = {"ordinary": ("Ordinary", "#333333", "x"),
                 "decorrelate-length-0.055": ("Selected activity filter", older, "D"),
                 "protect-60": ("Slow local writes", "#aa5a35", "^")}.get(name)
        if known is None:
            ax.scatter(old, new, s=24, color="#a6acb7")
        else:
            ax.scatter(old, new, s=46, color=known[1], marker=known[2], label=known[0], zorder=4)
    for ref in (ax.xaxis, ax.yaxis):
        ref.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlim(.4, .95)
    ax.set_ylim(.55, .95)
    ax.set_xlabel("Old cues on quiet trials")
    ax.set_ylabel("New cues on quiet trials")
    ax.set_title("C   Discovery candidates and trade-offs", loc="left", fontweight="bold")
    ax.legend(loc="lower left", frameon=False)

    ax = axes[1, 1]
    groups = [key for key in partitions["summary"] if key != "all"]
    for i, group in enumerate(groups):
        for j, (metric, color, mark) in enumerate((("old_revisit", older, "o"), ("new_revisit", newer, "s"))):
            effect = partitions["summary"][group][metric]["difference"]
            mean = effect["mean"]
            a, b = effect["ci95"]
            ax.errorbar(mean, i + (.13 if j == 0 else -.13), xerr=[[mean - a], [b - mean]],
                        color=color, marker=mark, ms=4, lw=1.4, capsize=2)
    ax.set_yticks(range(len(groups)), ["Cue groups 0,1 / 2,3", "Cue groups 0,2 / 1,3",
                                     "Cue groups 0,3 / 1,2 (strong conflict)"])
    ax.set_ylim(len(groups) - .5, -.5)
    ax.axvline(0, color="#666666", ls=":", lw=.9)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{100*x:.0f}"))
    ax.set_xlabel("Activity filter minus ordinary, percentage points")
    ax.set_title("D   Held-out cue partitions and both orders", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#ddd", lw=.7)
    for panel in axes.ravel():
        panel.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    path = args.directory
    confirm = read(path / "spike-consolidate-confirm-v1.json")
    replay = read(path / "spike-consolidate-replay-v1.json")
    summary = gate_checks(confirm, replay)
    (path / "spike-consolidate-gates.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.plot:
        discovery = read(path / "spike-consolidate-discovery-v1.json")
        partitions = read(path / "spike-consolidate-partition-v1.json")
        plot(discovery, confirm, replay, partitions, path / "spike-consolidate.png")


if __name__ == "__main__":
    main()
