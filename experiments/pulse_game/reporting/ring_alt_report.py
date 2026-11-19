"""Gate checks and plots for real bites, sign intervention, and quiet probes."""

import argparse
import json
from pathlib import Path

import numpy as np


def read(dir_path, name):
    return json.loads((dir_path / name).read_text(encoding="utf-8"))


def gate_checks(confirm, probe):
    arms = confirm["summary"]["arms"]
    effects = confirm["summary"]["candidate_minus_comparator"]
    effects_probe = probe["summary"]["same"]["negative_minus_comparator"]["muted"]["repeat"]

    def arc_rate(name):
        rows = [row for row in confirm["rows"] if row["name"] == name]
        edges = sum(sum(row["network"]["effective_edge_writes_per_life"]) for row in rows)
        outcomes = sum(np.asarray(row["network"]["writes_per_life"]).sum() for row in rows)
        return float(edges / outcomes)

    dense, sparse = arc_rate("negative-live"), arc_rate("negative-sparse65-same-filter")
    checks = {
        "real_first_greater_than_70": arms["negative-live"]["late_first"]["mean"] > .70,
        "same_cue_both_backgrounds_greater_than_40": arms["negative-live"]["late_context_pairs_both_correct"]["mean"] > .40,
        "true_vs_mute_and_yoke_above_20": all(effects[name]["late_first"]["mean"] >= .20
                                             and effects[name]["late_first"]["ci95"][0] > 0
                                             for name in ("negative-no-feedback", "negative-yoked")),
        "filter_gain_at_least_10": effects["negative-no-filter"]["late_first"]["mean"] >= .10
        and effects["negative-no-filter"]["late_first"]["ci95"][0] > 0,
        "single_error_selectively_reduces_repeat_at_least_10": effects_probe["mean"] <= -.10
        and effects_probe["ci95"][1] < 0 and not any(x["untouched_groups_written"] for x in probe["rows"]),
        "sparse_noninferior_and_budget_at_most_45_percent": effects["negative-sparse65-same-filter"]["late_first"]["ci95"][1] <= .05
        and sparse / dense <= .45,
    }
    return {"checks": checks, "edge_writes_per_negative_receipt": {"dense": dense, "sparse65": sparse,
                                                                   "ratio": sparse / dense}}


def plot(confirm, retry, probe, switched, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    green, amber, purple, grey, red = "#1e6b63", "#9a7333", "#775ba7", "#697383", "#b04b49"

    ax = axes[0, 0]
    names = (("negative-live", "Negative only", green, "-"), ("signed", "Both signs", purple, "-"),
             ("positive-only", "Positive only", amber, "--"), ("negative-no-filter", "Negative, no filter", red, "-."),
             ("negative-yoked", "Negative, matched yoke", grey, ":"))
    for arm, label, color, style in names:
        values = np.array([row["first_correct"] for row in confirm["rows"] if row["name"] == arm], dtype=np.float64)
        rows = values.reshape(values.shape[0], values.shape[1], -1, 32).mean(axis=(0, 1, 3))
        ax.plot(np.arange(len(rows)) * 32 + 16.5, rows, color=color, ls=style, marker="o", ms=3, label=label)
    ax.axhline(.25, ls=":", lw=.8, color="#777777")
    ax.set_xlabel("Trial (32-trial bins)")
    ax.set_ylabel("Correct first bite")
    ax.set_ylim(.18, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend(loc="center right", bbox_to_anchor=(1., .29), frameon=True,
              framealpha=.96, edgecolor="none", fontsize=9)
    ax.set_title("A   Single-bite training on new maps", loc="left", fontweight="bold")

    ax = axes[0, 1]
    names = ("negative-live", "negative-blind", "negative-no-feedback", "positive-only", "signed")
    labels = ("Negative only", "Blind scan", "No feedback", "Positive only", "Both signs")
    data = retry["summary"]["arms"]
    for index, (metric, color, marker, label) in enumerate((("late_first", green, "o", "First bite"),
                                                            ("late_solved", purple, "s", "Solved by any bite"))):
        for row, arm in enumerate(names):
            val = data[arm][metric]
            mean, lo, hi = val["mean"], *val["ci95"]
            ax.errorbar(mean, row + (.12 if index == 0 else -.12), xerr=[[mean - lo], [hi - mean]],
                        color=color, marker=marker, lw=1.3, capsize=2, label=label if row == 0 else None)
    ax.set_yticks(range(len(names)), labels)
    ax.set_ylim(len(names) - .5, -.5)
    ax.set_xlim(.15, 1.04)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    for row, name in enumerate(names):
        ax.text(.18, row - .26, f"{data[name]['late_bites']['mean']:.2f} bites", va="center", color="#525252", fontsize=8)
    ax.legend(loc="center", bbox_to_anchor=(.5, .69), frameon=True, framealpha=.96, edgecolor="none", fontsize=8)
    ax.set_xlabel("Correct meal rate, 95% world bootstrap intervals")
    ax.set_title("B   Retry access saturates terminal success", loc="left", fontweight="bold")

    ax = axes[1, 0]
    summary = probe["summary"]
    frames = (("same", "repeat", "Repeat wrong fruit, same key"),
              ("other-context", "repeat", "Repeat fruit, other context"),
              ("other-context", "correct_when_previous_wrong_is_target", "Eat same fruit when it becomes correct"),
              ("other-cue", "repeat", "Repeat fruit, other cue"))
    for index, (relation, metric, _) in enumerate(frames):
        info = summary[relation]["negative_minus_comparator"]["muted"][metric]
        mid, lo, hi = info["mean"], *info["ci95"]
        color = green if index == 0 else red if index == 2 else grey
        ax.errorbar(mid, index, xerr=[[mid - lo], [hi - mid]], color=color, marker="o", lw=1.3, capsize=3)
    ax.set_yticks(range(len(frames)), [row[2] for row in frames])
    ax.set_ylim(len(frames) - .5, -.5)
    ax.set_xlim(-.26, .08)
    ax.axvline(0, ls=":", linewidth=.8, color="#777777")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{100*x:.0f}"))
    ax.set_xlabel("One authentic negative minus muted (percentage points)")
    ax.set_title("C   A single wrong bite suppresses that fruit", loc="left", fontweight="bold")

    ax = axes[1, 1]
    steps = [int(step) for step in switched["summary"]]
    for arm, color, label in (("negative", green, "Negative + filter"), ("negative-no-filter", red, "Negative, no filter")):
        for key, ls, mark in (("old", "-", "o"), ("new", "--", "s")):
            curve = [switched["summary"][str(trial)][arm][key]["mean"] for trial in steps]
            ax.plot(steps, curve, color=color, ls=ls, marker=mark, ms=4, label=f"{label}, {key}")
    ax.set_ylim(.08, .98)
    ax.set_xticks(steps)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend(loc="lower right", frameon=True, framealpha=.96, edgecolor="none", fontsize=8)
    ax.set_xlabel("Training trials before quiet fork")
    ax.set_ylabel("Correct first bite in probe")
    ax.set_title("D   Retention across repeated cue changes", loc="left", fontweight="bold")
    for panel in axes.ravel():
        panel.spines[["top", "right"]].set_visible(False)
        panel.grid(axis="y" if panel in (axes[0, 0], axes[1, 1]) else "x", color="#dddddd", lw=.65)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_dose(dose, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    names = ("silent", "one-wrong-fruit", "two-distinct-wrong-fruits",
             "three-distinct-wrong-fruits", "same-wrong-fruit-three-times")
    labels = ("Silent", "1 distinct", "2 distinct", "3 distinct", "Same fruit 3 times")
    means = [dose["summary"][arm]["correct"]["mean"] for arm in names]
    ci = np.asarray([dose["summary"][arm]["correct"]["ci95"] for arm in names])
    fig, ax = plt.subplots(figsize=(8.3, 4.2), layout="constrained")
    ax.spines[["top", "right"]].set_visible(False)
    ax.bar(range(len(names)), means, width=.62, color=("#697383", "#7eb7aa", "#43978a", "#1e6b63", "#b78e52"))
    ax.errorbar(range(len(names)), means, yerr=[means - ci[:, 0], ci[:, 1] - means],
                color="#2d3035", fmt="none", capsize=3, lw=1.1)
    ax.axhline(.25, color="#697383", ls=":", lw=.9)
    ax.set_xticks(range(len(names)), labels)
    ax.set_ylim(0, 1.03)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_ylabel("Correct first bite in silent probe")
    ax.set_xlabel("Authentic wrong-fruit receipts revealed on the same 12-bite training track")
    ax.set_title("Distinct negative receipts combine across candidates", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#dddddd", lw=.65)
    ax.set_axisbelow(True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    confirm = read(args.data, "ring-alt-confirm-v1.json")
    probe = read(args.data, "ring-alt-probe-v1.json")
    gates = gate_checks(confirm, probe)
    (args.data / "ring-alt-gates-v1.json").write_text(json.dumps(gates, indent=2) + "\n", encoding="utf-8")
    if args.plot:
        retry = read(args.data, "ring-alt-retry-v1.json")
        switched = read(args.data, "ring-alt-switch-v1.json")
        plot(confirm, retry, probe, switched, args.data / "ring-alt-results-v1.png")
        dose_path = args.data / "ring-alt-dose-v1.json"
        if dose_path.exists():
            plot_dose(read(args.data, dose_path.name), args.data / "ring-alt-dose-v1.png")


if __name__ == "__main__":
    main()
