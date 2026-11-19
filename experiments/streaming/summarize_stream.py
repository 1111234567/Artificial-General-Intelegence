"""Export completed stream experiments; preserve seed-level evidence."""
import hashlib
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "streaming"
OUT = ROOT / "docs" / "evidence" / "streaming-2026-10-26"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def family(directory, policy):
    rows = [read(DATA / directory / f"{policy}-{seed}.json") for seed in range(3)]
    assert {row["seed"] for row in rows} == {0, 1, 2}
    return rows


def statistic(values):
    return dict(mean=statistics.mean(values), std=statistics.stdev(values), values=values)


def main():
    stages = {
        "common_reader": family("large-common-v1", "mru"),
        "calibrated_reader": family("large-stopping-v1", "mru"),
        "straight_through": family("large-st-writer-v1", "learned"),
        "reinforce": family("large-rl-writer-v1", "learned"),
    }
    long = {
        "mru": family("long-mru-cpu-v1", "mru"),
        "straight_through": family("long-st-cpu-v1", "learned"),
        "reinforce": family("long-rl-cpu-v1", "learned"),
    }
    summary = {}
    for name, rows in stages.items():
        keys = set.intersection(*(set(row["heldout"]) for row in rows))
        summary[name] = {case: {metric: statistic([r["heldout"][case]["metrics"][metric] for r in rows])
                              for metric in rows[0]["heldout"][case]["metrics"]} for case in sorted(keys)}
    long_summary = {name: {metric: statistic([r["heldout"]["metrics"][metric] for r in rows])
                          for metric in rows[0]["heldout"]["metrics"]} for name, rows in long.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = dict(summary=summary, long_summary=long_summary, runs=stages, long_runs=long,
                    local_learning=read(DATA / "local-v1" / "summary.json"),
                    local_manifest=read(DATA / "local-v1" / "manifest.json"),
                    limitations=[
                        "Only three seeds; sample SD is not a confidence interval.",
                        "MRU is an explicit two-demand-context rule, not a learned writer.",
                        "ST uses final-answer supervision; RL receives binary rewards. Training RNGs differ.",
                        "Short ST heldout seed differs; long CPU comparisons share worlds and device.",
                        "RL initial/final short holdout is paired with frozen reader and halt parameters.",
                        "Long evaluation freezes parameters; only cache and demand state continue.",
                    ])
    (OUT / "stream-results.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), layout="constrained")
    colors = {"mru": "#4D4D4D", "straight_through": "#D55E00", "reinforce": "#0072B2"}
    labels = {"mru": "Explicit memory rule", "straight_through": "ST writer", "reinforce": "Reward-trained writer"}
    for name, rows in long.items():
        values = np.array([[record["accuracy"] for record in row["heldout"]["per_cycle"]] for row in rows]) * 100
        x = np.arange(1, values.shape[1] + 1)
        mean, std = values.mean(0), values.std(0, ddof=1)
        axes[0].plot(x, mean, color=colors[name], label=labels[name], linewidth=1.8)
        axes[0].fill_between(x, mean - std, mean + std, color=colors[name], alpha=.12)
    axes[0].set(title="Continuing state over 64 task cycles", xlabel="Task cycle (no state reset)",
                ylabel="Exact answer (%)", ylim=(0, 103))
    axes[0].legend(loc="lower right", fontsize=9)
    phases = ["a_acquire", "b_acquire", "return", "revision"]
    x = np.arange(4)
    for offset, case, color, label in [(-.19, "initial_learned_stream", "#909090", "Before writer training"),
                                      (.19, "stream", "#0072B2", "After real-reward training")]:
        values = np.array([[row["heldout"][case]["metrics"][phase + "/accuracy"] for phase in phases]
                           for row in stages["reinforce"]]) * 100
        axes[1].bar(x + offset, values.mean(0), .38, yerr=values.std(0, ddof=1), color=color,
                    capsize=3, label=label)
    axes[1].set(title="Paired holdout: same frozen reader", ylabel="Exact answer (%)", ylim=(0, 105))
    axes[1].set_xticks(x, ["Acquire A", "Acquire B", "Return before\nre-exposure", "Revised facts"])
    axes[1].legend(loc="upper left", fontsize=9)
    for ax in axes:
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("Learned memory decisions: 3 seeds, mean +/- sample SD", fontsize=14)
    fig.savefig(OUT / "memory.png", dpi=180)
    fig.savefig(OUT / "memory.svg")
    plt.close(fig)
    for name, metrics in long_summary.items():
        print(name, {key: (round(metrics[key]["mean"], 6), round(metrics[key]["std"], 6))
                     for key in ("accuracy", "return/accuracy", "revision/accuracy", "reads")})
    for case in ("initial_learned_stream", "stream", "return_a", "noise32"):
        print("RL", case, {key: (round(summary["reinforce"][case][key]["mean"], 6),
                                round(summary["reinforce"][case][key]["std"], 6))
                           for key in ("accuracy", "return/accuracy", "revision/accuracy")})


if __name__ == "__main__":
    main()
