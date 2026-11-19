"""Plot only the complete, strict native lifelong aggregate (never raw scores).

Run after ``experiments.lifelong.report``. All uncertainty is across the three
trained seeds. A family average is computed within each seed before aggregation.
The companion JSON preserves every plotted seed value and the aggregate SHA256.
Requires Matplotlib and ijson; irrelevant large diagnostic tables are streamed
past, so plotting does not materialize the full aggregate in memory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (0, 1, 2)
TICKS = (131072, 163840, 164864, 167936, 172032)
TICK_LABELS = ("Main end\n131,072", "Withdrawal end\n163,840", "Return +1,024\n164,864",
               "Return +4,096\n167,936", "Return +8,192\n172,032")
CONDITIONS = ("short", "long_wait", "long_idle", "long_both", "far")
CONDITION_LABELS = ("Short", "Long wait", "Long idle", "Long both", "Far\n(out of support)")
POLICIES = ("fixed_shrink", "grow_shrink", "open_shrink")
POLICY_LABELS = {"fixed_shrink": "Fixed", "grow_shrink": "Grow", "open_shrink": "Open"}
POLICY_COLORS = {"fixed_shrink": "#4477AA", "grow_shrink": "#228833", "open_shrink": "#CC6677"}
LEARNING_COLORS = {"recent_repeat": "#4477AA", "history_replay": "#CC6677"}
SEED_MARKERS = ("o", "s", "^")
SEED_OFFSETS = (-.045, 0., .045)
METRICS = {
    "strict_completion": (100., "percent", "No-error completion (%)",
                          "clean_completed / matured primary requests; right-censored requests excluded"),
    "completion": (100., "percent", "Ordinary completion (%)",
                   "completed / matured primary requests; right-censored requests excluded"),
    "inactive_per_1000_idle": (1., "actions per 1000 idle opportunities", "Idle actions / 1,000 idle opportunities",
                               "1000 * inactive actions / idle opportunities"),
    "active": (1., "recurrent coordinates", "Numerically nonzero recurrent coordinates", "count(W != 0)"),
    "allowed": (1., "recurrent coordinates", "Update-eligible recurrent coordinates", "count(allowed_mask)"),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_plot_tables(path):
    import ijson
    from ijson.common import ObjectBuilder

    wanted = {"format", "complete", "audit", "counts", "protocol_plan", "protocol_sha256",
              "supplement_protocol_sha256", "matched_memory_per_seed", "checkpoints",
              "combined_per_seed", "matched_paired_contrasts_per_seed"}
    report, seen, active, builder = {}, set(), None, None
    try:
        with Path(path).open("rb") as stream:
            for prefix, event, value in ijson.parse(stream, use_float=True):
                if prefix == "" and event == "map_key":
                    if builder is not None:
                        report[active] = builder.value
                    if value in seen:
                        raise ValueError(f"duplicate top-level aggregate key: {value}")
                    seen.add(value)
                    active = value if value in wanted else None
                    builder = ObjectBuilder() if active is not None else None
                elif prefix == "" and event == "end_map":
                    if builder is not None:
                        report[active] = builder.value
                    builder = None
                elif builder is not None:
                    builder.event(event, value)
    except ijson.JSONError as error:
        raise ValueError(f"invalid aggregate JSON: {error}") from error
    return report


def load_report(path):
    path = Path(path).resolve()
    digest = sha256(path)
    report = _read_plot_tables(path)
    if (report.get("format") != "lifelong-study-summary-v1" or report.get("complete") is not True
            or report.get("audit", {}).get("status") != "passed"
            or report.get("audit", {}).get("errors") != []):
        raise ValueError("plots require a complete, passed strict lifelong-study-summary-v1 report")
    expected = {"training_runs": 30, "trained_seeds": 3, "full_checkpoints": 390,
                "primary_evaluation_forks": 1380, "supplemental_evaluation_forks": 480,
                "evaluation_forks": 1860, "matched_memory_forks": 840}
    if any(report.get("counts", {}).get(key) != value for key, value in expected.items()):
        raise ValueError("the audited report does not contain the complete formal study")
    if (report.get("protocol_plan", {}).get("seeds") != list(SEEDS)
            or len(report.get("matched_memory_per_seed", [])) != 840
            or len(report.get("checkpoints", [])) != 390):
        raise ValueError("the three-seed matched-horizon/checkpoint grid is incomplete")
    return report, digest


def select(rows, **conditions):
    return [row for row in rows if all(row.get(key) == value for key, value in conditions.items())]


def source_identity(row):
    keys = ("id", "seed", "architecture", "schedule", "learning", "coverage", "tick", "family",
            "condition", "steps", "probe_origin", "contrast", "within_training_support")
    item = {key: row[key] for key in keys if key in row}
    if "primary" in row:
        item.update(primary_matured=row["primary"]["matured"], primary_censored=row["primary_censored"])
    return item


def seed_summary(rows, metric, *, family_average=False, structural=False, difference=False):
    """Enforce seed replication and keep family measurements inside each seed."""
    scale, unit, _, definition = METRICS[metric]
    if difference:
        if metric not in ("strict_completion", "completion"):
            raise ValueError("only fraction differences use percentage-point units")
        unit = "percentage points"
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["seed"]].append(row)
    if set(grouped) != set(SEEDS):
        raise ValueError(f"expected seeds {SEEDS}, got {sorted(grouped)}")
    points = []
    for seed in SEEDS:
        selected = sorted(grouped[seed], key=lambda row: row.get("family", -1))
        if family_average:
            if [row.get("family") for row in selected] != [0, 1, 2, 3]:
                raise ValueError("family mean requires exactly families 0, 1, 2, 3 within each seed")
        elif len(selected) != 1:
            raise ValueError(f"duplicate seed {seed} in a plotted statistical group")
        values = [(row["structure"] if structural else row["metrics"])[metric] for row in selected]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in values):
            raise ValueError(f"undefined/nonfinite {metric} for plotted seed {seed}")
        raw = statistics.mean(values)
        points.append(dict(seed=seed, raw_value=raw, value=scale * raw,
                           sources=[dict(source_identity(row), raw_metric_value=value)
                                    for row, value in zip(selected, values)]))
    displayed = [point["value"] for point in points]
    return dict(metric=metric, unit=unit, display_scale=scale, definition=definition,
                within_seed_aggregation="unweighted mean of four family-specific values" if family_average else "single measurement",
                n=3, mean=statistics.mean(displayed), sample_sd=statistics.stdev(displayed), per_seed=points)


def _style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                         "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.fontsize": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": False, "figure.facecolor": "white", "savefig.facecolor": "white",
                         "svg.fonttype": "none", "svg.hashsalt": "lifelong-2026-10-29"})


def tidy(ax):
    ax.grid(axis="y", color="#D9DDE2", linewidth=.65, alpha=.8)
    ax.set_axisbelow(True)


def _seed_legend():
    return [Line2D([], [], marker=marker, color="#777777", markerfacecolor="white", linestyle="none",
                   markersize=4, label=f"Seed {seed}") for seed, marker in zip(SEEDS, SEED_MARKERS)]


def draw_series(ax, x, summaries, *, color, label, linestyle="-", shift=0., points=True):
    x = [value + shift for value in x]
    ax.errorbar(x, [row["mean"] for row in summaries], yerr=[row["sample_sd"] for row in summaries],
                color=color, label=label, linestyle=linestyle, linewidth=1.7, marker="D", markersize=3.8,
                elinewidth=1., capsize=2.5, zorder=4)
    if points:
        for index, marker in enumerate(SEED_MARKERS):
            ax.scatter([value + SEED_OFFSETS[index] for value in x],
                       [row["per_seed"][index]["value"] for row in summaries], s=16, marker=marker,
                       facecolor="white", edgecolor=color, linewidth=.8, alpha=.85, zorder=5)


def rate_limits(axes, groups):
    lower = min([0.] + [row["mean"] - row["sample_sd"] for row in groups])
    upper = max([100.] + [row["mean"] + row["sample_sd"] for row in groups])
    pad = max(3., (upper - lower) * .035)
    for ax in axes:
        ax.set_ylim(lower - pad, upper + pad)
        ax.set_yticks([0, 25, 50, 75, 100])


def save_figure(fig, output, stem, description, panels):
    files = []
    for suffix in ("png", "svg"):
        path = output / f"{stem}.{suffix}"
        metadata = {"Software": "Matplotlib"} if suffix == "png" else {"Date": None, "Creator": "Matplotlib"}
        fig.savefig(path, dpi=200, bbox_inches="tight", metadata=metadata)
        files.append(dict(name=path.name, sha256=sha256(path)))
    plt.close(fig)
    return dict(stem=stem, description=description, files=files, panels=panels)


def retention_figure(report, output):
    rows = report["matched_memory_per_seed"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), sharex=True)
    panels, rates = [], []
    domains = (("narrow", "short", "Narrow training / short probe"),
               ("wide", "long_both", "Wide training / long-both probe"))
    series_specs = [(schedule, learning) for schedule in ("blocked", "mixed")
                    for learning in ("recent_repeat", "history_replay")]
    for col, (coverage, condition, title) in enumerate(domains):
        for line, metric in enumerate(("strict_completion", "completion")):
            ax = axes[line, col]
            panel = dict(title=title, metric=metric, selection=dict(architecture="fixed_shrink", family=0,
                         coverage=coverage, condition=condition, steps=2048), series=[])
            for index, (schedule, learning) in enumerate(series_specs):
                summaries = []
                for tick in TICKS:
                    selected = select(rows, architecture="fixed_shrink", family=0, coverage=coverage,
                                      condition=condition, schedule=schedule, learning=learning, tick=tick, steps=2048)
                    if any(row["within_training_support"] is not True for row in selected):
                        raise ValueError("headline retention panels must be within training support")
                    summaries.append(dict(x_tick=tick, **seed_summary(selected, metric)))
                label = f"{schedule.title()} / {'recent repeat' if learning == 'recent_repeat' else 'history replay'}"
                draw_series(ax, range(len(TICKS)), summaries, color=LEARNING_COLORS[learning], label=label,
                            linestyle="--" if schedule == "blocked" else "-", shift=(index - 1.5) * .028)
                panel["series"].append(dict(label=label, schedule=schedule, learning=learning, points=summaries))
                rates.extend(summaries)
            panels.append(panel)
            tidy(ax)
            ax.axvline(.5, color="#999999", linewidth=.7, linestyle=":")
            ax.axvline(1.5, color="#999999", linewidth=.7, linestyle=":")
            ax.set(xlim=(-.3, 4.3), ylabel=METRICS[metric][2], xticks=range(5), xticklabels=TICK_LABELS)
            if line == 0:
                ax.set_title(title, pad=10)
            if line == 1:
                ax.set_xlabel("Checkpoint (categorical; unequal tick intervals)")
    rate_limits(axes.flat, rates)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .925), ncol=4, frameon=False)
    fig.legend(handles=_seed_legend(), loc="lower center", bbox_to_anchor=(.5, .078), ncol=3, frameon=False)
    fig.suptitle("Family 0 retention and return", x=.5, y=.985, fontsize=17, weight="bold")
    fig.text(.5, .947, "Fixed sparse policy; the two probe conditions are within their respective training supports",
             ha="center", fontsize=11, color="#444444")
    fig.text(.5, .024, "Diamonds and bars: mean +/- sample SD across 3 trained seeds; small markers: individual seeds. All probes: 2,048 ticks.\n"
             "Rates use matured primary requests; right-censored requests are excluded. Source h/world states differ across checkpoints.",
             ha="center", fontsize=9, linespacing=1.5, color="#444444")
    fig.subplots_adjust(left=.075, right=.985, bottom=.205, top=.865, hspace=.19, wspace=.17)
    return save_figure(fig, output, "family0-retention-return", "Matched-horizon family0 function queries within each coverage's timing support.", panels)


def structural_figure(report, output):
    rows, checkpoints = report["combined_per_seed"], report["checkpoints"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.3))
    panels, rates = [], []
    for ax, metric, title in zip((axes[0, 0], axes[0, 1], axes[1, 0]),
                                 ("strict_completion", "completion", "inactive_per_1000_idle"),
                                 ("No-error completion", "Ordinary completion", "Idle actions (lower is better)")):
        panel = dict(title=title, metric=metric, selection=dict(schedule="mixed", learning="history_replay", coverage="wide",
                     tick=172032, families=[0, 1, 2, 3], steps=2048), series=[])
        for index, policy in enumerate(POLICIES):
            summaries = []
            for condition in CONDITIONS:
                selected = select(rows, architecture=policy, schedule="mixed", learning="history_replay", coverage="wide",
                                  tick=172032, condition=condition, steps=2048)
                summaries.append(dict(x_condition=condition, **seed_summary(selected, metric, family_average=True)))
            draw_series(ax, range(5), summaries, color=POLICY_COLORS[policy], label=POLICY_LABELS[policy], shift=(index - 1) * .10)
            panel["series"].append(dict(label=POLICY_LABELS[policy], architecture=policy, points=summaries))
            if metric != "inactive_per_1000_idle":
                rates.extend(summaries)
        panels.append(panel)
        tidy(ax)
        ax.axvspan(3.5, 4.45, color="#EEEEEE", zorder=0)
        ax.set(title=title, ylabel=METRICS[metric][2], xlim=(-.4, 4.45), xticks=range(5), xticklabels=CONDITION_LABELS)
        if metric == "inactive_per_1000_idle":
            lower, upper = ax.get_ylim()
            ax.set_ylim(min(0., lower), upper)
        ax.set_xlabel("Frozen evaluation condition")
    rate_limits((axes[0, 0], axes[0, 1]), rates)
    capacity = axes[1, 1]
    panel = dict(title="Recurrent numerical support and eligibility", selection=dict(schedule="mixed", learning="history_replay", coverage="wide"), series=[])
    all_ticks = sorted({row["tick"] for row in checkpoints})
    allocations = set()
    capacity_groups = []
    for policy in POLICIES:
        for metric in ("active", "allowed"):
            summaries = []
            for tick in all_ticks:
                selected = select(checkpoints, architecture=policy, schedule="mixed", learning="history_replay", coverage="wide", tick=tick)
                allocations.update(row["total_parameters"] for row in selected)
                summaries.append(dict(x_tick=tick, **seed_summary(selected, metric, structural=True)))
            capacity_groups.extend(summaries)
            x = [tick / 1000 for tick in all_ticks]
            means, sd = [row["mean"] for row in summaries], [row["sample_sd"] for row in summaries]
            capacity.plot(x, means, color=POLICY_COLORS[policy], linewidth=1.8 if metric == "active" else 1.1,
                          linestyle="-" if metric == "active" else "--", zorder=3)
            if metric == "active":
                capacity.fill_between(x, [m - s for m, s in zip(means, sd)], [m + s for m, s in zip(means, sd)],
                                      color=POLICY_COLORS[policy], alpha=.13, linewidth=0)
                for index, marker in enumerate(SEED_MARKERS):
                    capacity.scatter(x, [row["per_seed"][index]["value"] for row in summaries], marker=marker,
                                     s=9, facecolor="white", edgecolor=POLICY_COLORS[policy], linewidth=.55, alpha=.65, zorder=4)
            panel["series"].append(dict(label=POLICY_LABELS[policy], architecture=policy, metric=metric, points=summaries))
    if len(allocations) != 1:
        raise ValueError("unexpected unequal allocated tensor parameter counts")
    panels.append(panel)
    tidy(capacity)
    for tick in (131.072, 163.840):
        capacity.axvline(tick, color="#999999", linewidth=.7, linestyle=":")
    capacity_lower = min([0.] + [row["mean"] - row["sample_sd"] for row in capacity_groups])
    capacity_upper = max([16384.] + [row["mean"] + row["sample_sd"] for row in capacity_groups])
    capacity.set(title="Recurrent numerical support and eligibility", xlabel="Training ticks (thousands)", ylabel="Recurrent coordinates",
                 xlim=(0, 176), ylim=(capacity_lower - 400, capacity_upper + 916), xticks=(0, 65.536, 131.072, 172.032), xticklabels=("0", "65.5", "131.1", "172.0"),
                 yticks=(0, 4096, 8192, 12288, 16384))
    capacity.legend(handles=[Line2D([], [], color="#555555", label="Nonzero support; shade = SD"),
                             Line2D([], [], color="#555555", linestyle="--", label="Update eligibility")],
                    loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("Native structural policies: behavior and actual capacity", y=.985, fontsize=17, weight="bold")
    fig.text(.5, .947, "Mixed schedule / history replay / wide coverage; behavior at tick 172,032", ha="center", fontsize=11, color="#444444")
    handles = [Line2D([], [], color=POLICY_COLORS[policy], label=POLICY_LABELS[policy], linewidth=2) for policy in POLICIES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .928), ncol=3, frameon=False)
    fig.legend(handles=_seed_legend(), loc="lower center", bbox_to_anchor=(.5, .077), ncol=3, frameon=False)
    fig.text(.5, .022, "Behavior: unweighted mean over 4 families within each seed, then mean +/- sample SD across 3 seeds; all probes: 2,048 ticks.\n"
             f"All policies allocate {next(iter(allocations)):,} tensor parameters. Capacity and eligibility differ; these compare complete update policies.",
             ha="center", fontsize=9, linespacing=1.5, color="#444444")
    fig.subplots_adjust(left=.075, right=.985, bottom=.20, top=.865, hspace=.35, wspace=.20)
    return save_figure(fig, output, "native-structure-behavior-capacity", "Native structural policy comparisons with family aggregation inside each trained seed.", panels)


def contrasts_figure(report, output):
    rows = report["matched_paired_contrasts_per_seed"]
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 6.0), sharey=True)
    panels, all_summaries = [], []
    ticks = (131072, 163840, 172032)
    domains = (("narrow", "short", "#4477AA", "Narrow / short"),
               ("wide", "long_both", "#CC6677", "Wide / long both"))
    for ax, contrast, title in zip(axes, ("replay_minus_recent", "mixed_minus_blocked", "wide_minus_narrow"),
                                  ("History replay - recent repeat", "Mixed - blocked schedule", "Wide - narrow coverage")):
        panel = dict(title=title, metric="strict_completion", selection=dict(architecture="fixed_shrink", family=0, steps=2048), series=[])
        specs = []
        if contrast == "replay_minus_recent":
            for coverage, condition, color, domain in domains:
                for schedule in ("blocked", "mixed"):
                    specs.append((dict(coverage=coverage, condition=condition, schedule=schedule),
                                  f"{domain}; {schedule}", color, "--" if schedule == "blocked" else "-"))
        elif contrast == "mixed_minus_blocked":
            for coverage, condition, color, domain in domains:
                for learning in ("recent_repeat", "history_replay"):
                    specs.append((dict(coverage=coverage, condition=condition, learning=learning),
                                  f"{domain}; {'repeat' if learning == 'recent_repeat' else 'replay'}",
                                  color, "--" if learning == "recent_repeat" else "-"))
        else:
            for learning in ("recent_repeat", "history_replay"):
                for schedule in ("blocked", "mixed"):
                    specs.append((dict(condition="short", learning=learning, schedule=schedule),
                                  f"Short; {schedule}; {'repeat' if learning == 'recent_repeat' else 'replay'}",
                                  LEARNING_COLORS[learning], "--" if schedule == "blocked" else "-"))
        for index, (conditions, label, color, linestyle) in enumerate(specs):
            summaries = []
            for tick in ticks:
                selected = select(rows, architecture="fixed_shrink", family=0, contrast=contrast,
                                  steps=2048, tick=tick, **conditions)
                summaries.append(dict(x_tick=tick, **seed_summary(selected, "strict_completion", difference=True)))
            draw_series(ax, range(3), summaries, color=color, label=label, linestyle=linestyle, shift=(index - 1.5) * .032)
            panel["series"].append(dict(label=label, contrast=contrast, selection=conditions, points=summaries))
            all_summaries.extend(summaries)
        panels.append(panel)
        tidy(ax)
        ax.axhline(0, color="#555555", linewidth=1.)
        ax.set(title=title, xlim=(-.25, 2.25), xticks=range(3), xticklabels=("Main end\n131,072", "Withdrawal end\n163,840", "Return +8,192\n172,032"),
               xlabel="Checkpoint (categorical; unequal tick intervals)")
        ax.legend(loc="upper left", bbox_to_anchor=(0, -.23), fontsize=8.5, frameon=False, handlelength=2.6)
    seed_values = [point["value"] for row in all_summaries for point in row["per_seed"]]
    lower = min([0.] + seed_values + [row["mean"] - row["sample_sd"] for row in all_summaries])
    upper = max([0.] + seed_values + [row["mean"] + row["sample_sd"] for row in all_summaries])
    pad = max(3., (upper - lower) * .09)
    for ax in axes:
        ax.set_ylim(lower - pad, upper + pad)
        ax.yaxis.set_major_locator(MaxNLocator(6))
    axes[0].set_ylabel("Change in no-error completion (percentage points)")
    fig.suptitle("Family 0: paired factor contrasts", y=.985, fontsize=17, weight="bold")
    fig.text(.5, .925, "Within-seed differences; fixed sparse policy; matched 2,048-tick probes", ha="center", fontsize=11, color="#444444")
    fig.text(.5, .036, "Diamonds and bars: mean +/- sample SD of 3 paired seed differences; small circles/squares/triangles: seeds 0/1/2.\n"
             "Each series holds the other factors fixed. Coverage contrasts use the short condition shared by both training supports.\n"
             "Fixed training ticks do not ensure equal task exposures; wider timing coverage also changes event density.",
             ha="center", fontsize=9, linespacing=1.5, color="#444444")
    fig.subplots_adjust(left=.065, right=.99, top=.84, bottom=.40, wspace=.13)
    return save_figure(fig, output, "family0-paired-factor-contrasts", "Matched-probe paired differences, preserving the other factor levels and the seed as statistical unit.", panels)


def create_figures(report_path, output):
    report_path, output = Path(report_path).resolve(), Path(output).resolve()
    report, report_hash = load_report(report_path)
    if output == report_path.parent or report_path.parent in output.parents or output in report_path.parents:
        raise ValueError("figure outputs must be separate from the aggregate directory")
    output.mkdir(parents=True, exist_ok=True)
    _style()
    path = output / "figure-data.json"
    # A rendering/validation failure must leave any earlier evidence set intact.
    # The data manifest is published last, after all its hash-bound images exist.
    with tempfile.TemporaryDirectory(prefix=".plot-stage-", dir=output) as temporary:
        stage = Path(temporary).resolve()
        if stage.parent != output:
            raise ValueError("temporary figure directory escaped the requested output directory")
        figures = [retention_figure(report, stage), structural_figure(report, stage), contrasts_figure(report, stage)]
        if sha256(report_path) != report_hash:
            raise ValueError("aggregate report changed while plotting")
        evidence = dict(format="lifelong-figures-v1", input_report=str(report_path), input_report_sha256=report_hash,
                        protocol_sha256=report["protocol_sha256"], supplement_protocol_sha256=report["supplement_protocol_sha256"],
                        plot_source=str(Path(__file__).resolve()), plot_source_sha256=sha256(__file__),
                        matplotlib_version=matplotlib.__version__, statistical_unit="independently trained seed",
                        seeds=list(SEEDS), uncertainty="mean +/- sample standard deviation across 3 seeds; not confidence intervals",
                        denominators="Completion uses matured primary requests; censored requests excluded. Idle uses idle opportunities.",
                        horizon="All plotted behavior and behavior contrasts use 2048-tick frozen evaluations.",
                        causal_scope="Frozen probes inherit checkpoint h/world states; structural policies differ in capacity and optimizer behavior.",
                        figures=figures)
        (stage / path.name).write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        for figure in figures:
            for artifact in figure["files"]:
                (stage / artifact["name"]).replace(output / artifact["name"])
        (stage / path.name).replace(path)
    return dict(input_report_sha256=report_hash, figures=[figure["stem"] for figure in figures], data=str(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "docs/evidence/lifelong-2026-10-29/aggregate-v1/report.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/evidence/lifelong-2026-10-29/figures")
    args = parser.parse_args(argv)
    try:
        result = create_figures(args.report, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Lifelong plotting rejected: {error}\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
