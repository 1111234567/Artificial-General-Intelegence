"""Summarize paired life-level differences and plot the closed-loop controls."""

import argparse
import json
from pathlib import Path

import numpy as np


def estimates(values):
    """Resample worlds after averaging the two fixed topology replicas."""
    x = np.asarray(values, dtype=np.float64)
    draws = np.random.default_rng(1492).integers(len(x), size=(4096, len(x)))
    return {"mean": float(x.mean()), "ci95": np.quantile(x[draws].mean(axis=1), [.025, .975]).tolist()}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def by_name(rows, name, metric):
    found = [row[metric] for row in rows if row["name"] == name]
    if not found:
        raise KeyError(f"no trials for {name}")
    return np.asarray(found).mean(axis=0)


def summarize(controls, revisits):
    summary = {"control_lives": len(controls["life_seeds"]), "revisit_lives": len(revisits["life_seeds"])}
    control = controls["results"]
    stage = revisits["results"]
    summary["controls"] = {}
    for name in dict.fromkeys(row["name"] for row in control):
        early = by_name(control, name, "early")
        late = by_name(control, name, "late")
        info = {"early32": float(early.mean()), "late32": float(late.mean())}
        if name != "yoked":
            info["late32_minus_early32"] = estimates(late - early)
            info["late32_minus_live"] = estimates(late - by_name(control, "live", "late"))
        paired = by_name(control, name, "opposing_context_pair_correct_count")[:, 1]
        info["post_fourth_context_pairs_both_correct"] = float((paired[:, 0] / paired[:, 1]).mean())
        selected = [x for row in control if row["name"] == name for x in row["network_statistics"]]
        info["network_means"] = {key: float(np.mean([x[key] for x in selected])) for key in selected[0]}
        summary["controls"][name] = info
    summary["yoked_late32_difference_descriptive"] = float(
        (by_name(control, "live", "late") - by_name(control, "yoked", "late")).mean())

    stage_metrics = ("old_last16", "new_first16", "new_last16", "revisit_first16", "revisit_last16")
    summary["revisit"] = {
        name: {metric: estimates(by_name(stage, name, metric)) for metric in stage_metrics}
        for name in dict.fromkeys(row["name"] for row in stage)
    }
    summary["paired_differences"] = {}
    comparisons = (
        ("interference", "learn-old-new-freeze-at-revisit", "learn-old-freeze-before-new"),
        ("review_feedback", "live-review-freeze-at-revisit", "quiet-review-freeze-at-revisit"),
    )
    for label, live, matched in comparisons:
        summary["paired_differences"][label] = {
            metric: estimates(by_name(stage, live, metric) - by_name(stage, matched, metric))
            for metric in stage_metrics[2:]
        }
    return summary


def plot(controls, revisits, summary, target):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")

    def learning_curves(axis, rows, series, *, minimum=0.38):
        for name, label, color, style in series:
            score = by_name(rows, name, "rolling16").mean(axis=1)
            axis.plot(8.5 + np.arange(len(score)) * 16, score, marker="o", ms=3,
                      label=label, color=color, linestyle=style)
        axis.axhline(.5, color="#777777", lw=.8, ls=":")
        axis.set_ylim(minimum, 1)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.set_xlabel("Trial (16-trial blocks)")
        axis.set_ylabel("Correct first contact")
        axis.grid(axis="y", color="#dddddd", lw=.55)
        axis.legend(loc="lower right", frameon=True, framealpha=.96, edgecolor="none", fontsize=8)

    live = ("live", "True signed receipt", "#106a62", "-")
    learning_curves(axes[0, 0], controls["results"], (
        live,
        ("yoked", "Other life's signed receipt", "#95783d", "--"),
        ("contact-only", "Contact identity only", "#b03243", "-."),
        ("frozen-birth", "Birth weights frozen", "#5b6173", ":"),
    ), minimum=.27)
    axes[0, 0].set_title("A   Closed-loop learning and receipt controls", loc="left", fontweight="bold")
    learning_curves(axes[0, 1], controls["results"], (
        live,
        ("no-recurrence", "Reservoir recurrence severed", "#95783d", "--"),
        ("frozen-after-learning", "Receipt muted after trial 72", "#515bc0", "-"),
        ("frozen-after-learning-swap", "Muted and target sides exchanged", "#cf725e", ":"),
    ), minimum=.25)
    axes[0, 1].axvline(72.5, color="#555555", lw=.8, ls=":")
    axes[0, 1].set_title("B   Retention and visual routing", loc="left", fontweight="bold")
    learning_curves(axes[1, 0], revisits["results"], (
        ("learn-old-new-freeze-at-revisit", "Learn new, revisit without receipt", "#b03243", "-"),
        ("learn-old-freeze-before-new", "Mute before new cues", "#5b6173", "--"),
        ("live-review-freeze-at-revisit", "Old-cue review with receipt", "#106a62", "-"),
        ("quiet-review-freeze-at-revisit", "Same review with muted receipt", "#95783d", ":"),
    ), minimum=.10)
    for t in (48.5, 96.5):
        axes[1, 0].axvline(t, color="#555555", lw=.8, ls=":")
    axes[1, 0].text(25, .975, "Old cues", ha="center", va="top", fontsize=8)
    axes[1, 0].text(73, .975, "New cues", ha="center", va="top", fontsize=8)
    axes[1, 0].text(122, .975, "Old revisits", ha="center", va="top", fontsize=8)
    axes[1, 0].set_title("C   Interference and interleaved review", loc="left", fontweight="bold")

    axis = axes[1, 1]
    axis.axvline(0, color="#777777", lw=.8, ls=":")
    segments = ("revisit_first16", "revisit_last16")
    markers = (("interference", "#b03243", 0), ("review_feedback", "#106a62", 1))
    for key, color, group in markers:
        for i, phase in enumerate(segments):
            x = summary["paired_differences"][key][phase]
            a, b = x["ci95"]
            axis.errorbar(x["mean"], group + (.10 if i == 0 else -.1),
                          xerr=[[x["mean"] - a], [b - x["mean"]]], color=color,
                          marker="o" if i == 0 else "s", ms=4, capsize=3)
    axis.set_yticks([0, 1], ["New learning vs. muted", "Review receipt vs. muted"])
    axis.set_ylim(-.45, 1.45)
    axis.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{100 * x:.0f}"))
    axis.set_xlabel("Paired accuracy difference (pp), 95% world bootstrap interval")
    axis.set_title("D   First (●) and last (■) 16 old-cue revisits", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#dddddd", lw=.55)
    figure.savefig(target, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("controls", type=Path)
    parser.add_argument("revisits", type=Path)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    controls = load(args.controls)
    revisits = load(args.revisits)
    summary = summarize(controls, revisits)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.plot:
        plot(controls, revisits, summary, args.plot)


if __name__ == "__main__":
    main()
