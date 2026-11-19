"""Scientific figures from passed organization audit artifacts only.

This module deliberately imports neither torch nor any experimental execution
or report module. CSV rows and small/streamed report metadata are the only data
inputs. No raw trajectory, checkpoint, or pre-audit score is opened.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


SEEDS = (0, 1, 2)
TICKS = (131072, 163840, 172032)
SNAPSHOTS = (0, *TICKS)
CONDITIONS = ("short", "long_both")
METRICS = ("strict_completion", "completion", "inactive_per_1000_idle")
ACQUISITION = ("fixed_shrink__wide", "open_shrink__wide", "fixed_shrink__narrow")
PASSIVE = ("plain__passive", "parallel__passive", "coupled__passive")
FUNCTIONAL = ("plain__functional_fixed", "plain__functional_task", "plain__functional_magnitude",
              "plain__functional_random", "coupled__functional_fixed", "coupled__functional_task")
ALL_MAIN = (*PASSIVE, *FUNCTIONAL)
LABELS = {
    "fixed_shrink__wide": "Fixed / wide", "open_shrink__wide": "Open / wide",
    "fixed_shrink__narrow": "Fixed / narrow", "plain__passive": "Plain",
    "parallel__passive": "Parallel", "coupled__passive": "Coupled",
    "plain__functional_fixed": "Plain / fixed", "plain__functional_task": "Plain / task",
    "plain__functional_magnitude": "Plain / magnitude", "plain__functional_random": "Plain / random",
    "coupled__functional_fixed": "Coupled / fixed", "coupled__functional_task": "Coupled / task",
}
COLORS = {
    "fixed_shrink__wide": "#0072B2", "open_shrink__wide": "#D55E00", "fixed_shrink__narrow": "#CC79A7",
    "plain__passive": "#0072B2", "parallel__passive": "#E69F00", "coupled__passive": "#009E73",
    "plain__functional_fixed": "#777777", "plain__functional_task": "#009E73",
    "plain__functional_magnitude": "#E69F00", "plain__functional_random": "#CC79A7",
    "coupled__functional_fixed": "#0072B2", "coupled__functional_task": "#D55E00",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def _stream_item(path, prefix):
    # The report may contain large arrays; never json.load the full report.
    import ijson
    with Path(path).open("rb") as stream:
        iterator = ijson.items(stream, prefix)
        try:
            return next(iterator)
        except StopIteration as error:
            raise ValueError(f"missing report metadata: {prefix}") from error


def _audited(folder, kind, bindings):
    folder = Path(folder).resolve()
    manifest = folder / "artifacts.json"
    artifacts = _json(manifest)
    if not isinstance(artifacts, dict):
        raise ValueError("audit artifacts must bind filenames to SHA256 values")
    bindings[str(manifest)] = sha256(manifest)

    def checked(name):
        path = folder / name
        actual = sha256(path)
        if artifacts.get(name) != actual:
            raise ValueError(f"changed or unbound audited artifact: {path}")
        bindings[str(path)] = actual
        return path

    report = checked("report.json")
    meta = {key: _stream_item(report, key)
            for key in ("format", "complete", "audit", "counts", "protocol", "protocol_sha256")}
    expected_format = "organization-acquisition-summary-v1" if kind == "acquisition" else "organization-study-summary-v1"
    if (meta["format"] != expected_format or meta["complete"] is not True
            or meta["audit"].get("status") != "passed" or meta["audit"].get("errors") != []):
        raise ValueError("figures require a complete passed strict audit with no errors")
    expected_counts = (dict(new_training_runs=9, independent_training_seeds=3, new_full_checkpoints=36,
                           new_evaluation_forks=216, borrowed_old_evaluation_forks=216,
                           additional_baseline_training_runs=0)
                       if kind == "acquisition" else
                       dict(training_runs=27, trained_seeds=3, full_checkpoints=108,
                            evaluation_files=27, frozen_forks=648))
    if any(meta["counts"].get(key) != value for key, value in expected_counts.items()):
        raise ValueError("formal audited run/checkpoint/fork inventory differs")
    if kind == "main":
        header = _json(checked("header.json"))
        if any(header.get(key) != meta[key] for key in ("format", "complete", "audit", "counts", "protocol_sha256")):
            raise ValueError("main header differs from its bound passed report")
    protocol = Path(meta["protocol"]).resolve()
    protocol_hash = sha256(protocol)
    if protocol_hash != meta["protocol_sha256"]:
        raise ValueError("audited protocol bytes changed")
    bindings[str(protocol)] = protocol_hash
    plan = _json(protocol)
    device = "cuda" if kind == "acquisition" else "cpu"
    if (plan.get("target_ticks") != 172032 or plan.get("checkpoint_ticks") != list(SNAPSHOTS)
            or plan.get("training_device") != device or plan.get("evaluation_device") != "cpu"
            or any(spec.get("steps") != 2048 for spec in plan.get("evaluation_specs", []))
            or len(plan.get("evaluation_specs", [])) != 24
            or plan.get("native_template", {}).get("width") != 128
            or plan.get("native_template", {}).get("batch_size") != 32):
        raise ValueError("plotting is restricted to the full CPU-main / CUDA-acquisition protocols")
    if kind == "main" and plan.get("mechanical_test") is not False:
        raise ValueError("mechanical fixtures are not formal organization evidence")
    return dict(folder=folder, checked=checked, metadata={key: meta[key] for key in
                ("format", "complete", "audit", "counts", "protocol", "protocol_sha256")})


def _rows(path, columns):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if (not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames)
                or not set(columns).issubset(reader.fieldnames)):
            raise ValueError(f"missing or duplicate CSV columns: {path}")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"malformed CSV row: {path}")
            yield row


def _number(value):
    if value in (None, ""):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite CSV value")
    return result


def _integer(value):
    result = _number(value)
    if result is None or result < 0 or int(result) != result:
        raise ValueError("nonnegative integer CSV value required")
    return int(result)


def seed_statistics(values):
    """Require the trained seed inventory; omit a mean if any value is undefined."""
    if set(values) != set(SEEDS):
        raise ValueError("each plotted group requires trained seeds 0, 1 and 2")
    observed = [values[seed] for seed in SEEDS if values[seed] is not None]
    if any(not math.isfinite(value) for value in observed):
        raise ValueError("nonfinite seed value")
    complete = len(observed) == len(SEEDS)
    return dict(n_total=3, n_defined=len(observed), n_undefined=3 - len(observed),
        mean=statistics.mean(observed) if complete else None,
        sample_sd=statistics.stdev(observed) if complete else None,
        per_seed=[dict(seed=seed, value=values[seed]) for seed in SEEDS])


def _metric_values(row):
    result = {metric: _number(row[metric]) for metric in METRICS}
    for metric, value in result.items():
        if value is not None and (value < 0 or (metric != "inactive_per_1000_idle" and value > 1)):
            raise ValueError("behavior rate outside its audited metric range")
    return result


def _score_rows(paths, kind, family_view):
    grouped = defaultdict(dict)
    columns = ("seed", "tick", "condition", "steps", *METRICS)
    for path in paths:
        for row in _rows(path, columns):
            seed, tick = _integer(row["seed"]), _integer(row["tick"])
            if seed not in SEEDS or tick not in TICKS or row["condition"] not in CONDITIONS or _integer(row["steps"]) != 2048:
                raise ValueError("foreign seed, stage, condition or horizon in behavior CSV")
            if kind == "main":
                variant = row["field_mode"] + "__" + row["structural_rule"]
                if variant not in ALL_MAIN:
                    raise ValueError("undeclared main variant")
                shrink = None
                if family_view == "mean4":
                    if _integer(row["n_families"]) != 4:
                        raise ValueError("main family mean must contain exactly four families per seed")
                    family = -1
                else:
                    if _integer(row["family"]) != 0:
                        continue
                    family = 0
            else:
                variant = row["architecture"] + "__" + row["coverage"]
                shrink = _number(row["shrink_step"])
                if (variant not in ACQUISITION or row["schedule"] != "mixed" or row["learning"] != "history_replay"
                        or shrink not in (0., 2e-4)
                        or row["evidence"] != ("new_frozen_probe" if shrink == 0. else "borrowed_original_frozen_probe")):
                    raise ValueError("foreign acquisition pairing identity")
                family = _integer(row["family"])
                if family not in range(4):
                    raise ValueError("foreign family")
                if family_view == "family0" and family != 0:
                    continue
            key = variant, shrink, tick, row["condition"], seed
            if family in grouped[key]:
                raise ValueError("duplicate within-seed behavior row")
            grouped[key][family] = _metric_values(row)
    variants = ALL_MAIN if kind == "main" else ACQUISITION
    shrinks = (None,) if kind == "main" else (0., 2e-4)
    expected = {(variant, shrink, tick, condition, seed) for variant in variants for shrink in shrinks
                for tick in TICKS for condition in CONDITIONS for seed in SEEDS}
    if set(grouped) != expected:
        raise ValueError("incomplete behavior seed/variant/stage inventory")
    result = {}
    required_families = {-1} if kind == "main" and family_view == "mean4" else ({0} if family_view == "family0" else set(range(4)))
    for key, families in grouped.items():
        if set(families) != required_families:
            raise ValueError("family aggregation requires the complete selected family inventory")
        result[key] = {metric: statistics.mean([row[metric] for row in families.values()])
                      if all(row[metric] is not None for row in families.values()) else None for metric in METRICS}
    return result


def _score_data(scores):
    result = []
    identities = sorted({key[:4] for key in scores}, key=str)
    for variant, shrink, tick, condition in identities:
        for metric in METRICS:
            values = {seed: scores[variant, shrink, tick, condition, seed][metric] for seed in SEEDS}
            result.append(dict(variant=variant, shrink_step=shrink, tick=tick, condition=condition,
                metric=metric, unit="per_1000_idle" if metric == "inactive_per_1000_idle" else "fraction",
                **seed_statistics(values)))
    return result


def _paired_data(scores, kind):
    comparisons = ([(variant, variant, 0., 2e-4, variant + ": shrink0 minus old2e-4") for variant in ACQUISITION]
                   if kind == "acquisition" else
                   [(left, right, None, None, left + " minus " + right) for left, right in
                    ((PASSIVE[1], PASSIVE[0]), (PASSIVE[2], PASSIVE[0]), (PASSIVE[2], PASSIVE[1]))])
    result = []
    for left, right, left_shrink, right_shrink, contrast in comparisons:
        for tick in TICKS:
            for condition in CONDITIONS:
                for metric in METRICS:
                    values = {}
                    for seed in SEEDS:
                        a, b = scores[left, left_shrink, tick, condition, seed][metric], scores[right, right_shrink, tick, condition, seed][metric]
                        values[seed] = None if a is None or b is None else a - b
                    result.append(dict(contrast=contrast, tick=tick, condition=condition, metric=metric,
                                       **seed_statistics(values)))
    return result


def _structure_data(report):
    capacity, bridges, acceptance = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    capacity_path = report["checked"]("capacity_trajectory.csv")
    for row in _rows(capacity_path, ("seed", "field_mode", "structural_rule", "tick", "active", "allowed")):
        variant = row["field_mode"] + "__" + row["structural_rule"]
        if variant not in FUNCTIONAL:
            continue
        seed, tick = _integer(row["seed"]), _integer(row["tick"])
        if tick < 128 or tick > TICKS[-1] or tick % 128:
            raise ValueError("foreign capacity update tick")
        for metric in ("active", "allowed"):
            key = variant, tick, metric
            if seed in capacity[key]:
                raise ValueError("duplicate capacity seed row")
            capacity[key][seed] = _integer(row[metric])
    checkpoints = report["checked"]("checkpoints.csv")
    for row in _rows(checkpoints, ("seed", "field_mode", "structural_rule", "tick", "structure")):
        variant = row["field_mode"] + "__" + row["structural_rule"]
        if variant not in FUNCTIONAL:
            continue
        seed, tick, structure = _integer(row["seed"]), _integer(row["tick"]), json.loads(row["structure"])
        if tick not in SNAPSHOTS:
            raise ValueError("foreign structural checkpoint tick")
        for metric in ("active_cross_bank_bridges", "allowed_cross_bank_bridges"):
            key = variant, tick, metric
            if seed in bridges[key]:
                raise ValueError("duplicate cross-bank bridge seed row")
            bridges[key][seed] = _integer(structure[metric])
        if tick == 0:
            for metric in ("active", "allowed"):
                capacity[variant, 0, metric][seed] = _integer(structure[metric + "_edges"])
    expected_capacity = {(variant, tick, metric) for variant in FUNCTIONAL
                         for tick in range(0, TICKS[-1] + 1, 128) for metric in ("active", "allowed")}
    expected_bridges = {(variant, tick, metric) for variant in FUNCTIONAL for tick in SNAPSHOTS
                        for metric in ("active_cross_bank_bridges", "allowed_cross_bank_bridges")}
    if set(capacity) != expected_capacity or set(bridges) != expected_bridges:
        raise ValueError("incomplete functional capacity or bridge inventory")
    training_rows, denominators = [], []
    training = report["checked"]("training_per_seed.csv")
    for row in _rows(training, ("seed", "field_mode", "structural_rule", "probe_summary", "work")):
        variant = row["field_mode"] + "__" + row["structural_rule"]
        if variant not in FUNCTIONAL:
            continue
        seed, summary = _integer(row["seed"]), json.loads(row["probe_summary"])
        training_rows.append(dict(variant=variant, seed=seed, work=json.loads(row["work"]), probe_summary=summary,
            structural_forward_fraction_of_base=_number(row["structural_forward_fraction_of_base"])))
        trials = _integer(summary["trials"])
        for kind in ("birth", "exit"):
            accepted = _integer(summary[f"accepted_{kind}_batches"])
            harmed = _integer(summary[f"accepted_{kind}_history_worsening_batches"])
            if not 0 <= harmed <= accepted <= trials:
                raise ValueError("invalid functional acceptance/history-damage denominators")
            for metric, numerator, denominator in ((f"{kind}_acceptance", accepted, trials),
                                                    (f"{kind}_accepted_history_worsening", harmed, accepted)):
                key = variant, metric
                if seed in acceptance[key]:
                    raise ValueError("duplicate functional probe seed summary")
                acceptance[key][seed] = numerator / denominator if denominator else None
                denominators.append(dict(variant=variant, seed=seed, metric=metric,
                                         numerator=numerator, denominator=denominator))
    expected_acceptance = {(variant, kind + suffix) for variant in FUNCTIONAL for kind in ("birth", "exit")
                           for suffix in ("_acceptance", "_accepted_history_worsening")}
    if set(acceptance) != expected_acceptance or len(training_rows) != 18:
        raise ValueError("incomplete functional training/probe summaries")
    return dict(
        capacity=[dict(variant=key[0], tick=key[1], metric=key[2], **seed_statistics(values))
                  for key, values in sorted(capacity.items())],
        bridges=[dict(variant=key[0], tick=key[1], metric=key[2], **seed_statistics(values))
                 for key, values in sorted(bridges.items())],
        acceptance=[dict(variant=key[0], metric=key[1], **seed_statistics(values))
                    for key, values in sorted(acceptance.items())],
        acceptance_denominators=denominators, training_per_seed=training_rows)


def _line(ax, records, *, color, label, style="-", scale=1., time_axis=False, seed_ticks=None):
    records = sorted(records, key=lambda row: row["tick"])
    xs = [row["tick"] / 1024 if time_axis else TICKS.index(row["tick"]) for row in records]
    means = [math.nan if row["mean"] is None else row["mean"] * scale for row in records]
    sds = [math.nan if row["sample_sd"] is None else row["sample_sd"] * scale for row in records]
    if time_axis:
        ax.plot(xs, means, color=color, linestyle=style, label=label)
        ax.fill_between(xs, [a - b for a, b in zip(means, sds)], [a + b for a, b in zip(means, sds)],
                        color=color, alpha=.10, linewidth=0)
    else:
        ax.errorbar(xs, means, yerr=sds, color=color, linestyle=style, marker="o", markersize=3.7,
                    capsize=3, label=label)
    for x, row in zip(xs, records):
        if seed_ticks is not None and row["tick"] not in seed_ticks:
            continue
        for index, value in enumerate(row["per_seed"]):
            if value["value"] is not None:
                jitter = (index - 1) * (.95 if time_axis else .045)
                ax.scatter(x + jitter, value["value"] * scale, s=15, facecolors="none", edgecolors=color,
                           linewidths=.8, zorder=4)


def _stages(ax):
    ax.set_xticks(range(3), ("Main end\n131,072", "Withdrawal end\n163,840", "Return end\n172,032"))
    ax.set_xlim(-.18, 2.18)


def _score_figure(plt, records, variants, title, family_label, acquisition=False):
    fig, axes = plt.subplots(3, 2, figsize=(13.6, 10.5), sharex=True)
    metric_labels = ("Strict completion (%)", "Ordinary completion (%)", "Idle false actions / 1,000 idle opportunities")
    for row_index, (metric, ylabel) in enumerate(zip(METRICS, metric_labels)):
        for column, condition in enumerate(CONDITIONS):
            ax = axes[row_index, column]
            for variant in variants:
                shrinks = (0., 2e-4) if acquisition else (None,)
                for shrink in shrinks:
                    selected = [row for row in records if row["variant"] == variant and row["shrink_step"] == shrink
                                and row["metric"] == metric and row["condition"] == condition]
                    label = LABELS[variant] + (" / shrink 0" if shrink == 0. else " / old shrink 2e-4") if acquisition else LABELS[variant]
                    _line(ax, selected, color=COLORS[variant], label=label,
                          style="--" if shrink == 2e-4 else "-", scale=1. if metric == METRICS[2] else 100.)
            _stages(ax)
            ax.set_ylabel(ylabel)
            if row_index == 0:
                ax.set_title("Short" if condition == "short" else "Long wait + long idle")
            if metric != METRICS[2]:
                ax.set_ylim(0, 100)
            else:
                ax.set_ylim(bottom=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .045), ncol=3, frameon=False)
    note = ("Fixed/narrow long_both is outside training support. Old records are borrowed audited evidence."
            if acquisition else "All modes: 36,396 parameters and 16,384 recurrent MAC/field. Parallel/coupled also match tanh counts.")
    fig.suptitle(title + "\n" + family_label, fontsize=13, y=.99)
    fig.text(.5, .012, "Means ± sample SD across exactly 3 trained seeds; open points are seeds. " + note,
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .10, 1, .94))
    return fig


def _ratio_panel(ax, records, suffix, title):
    for index, variant in enumerate(FUNCTIONAL):
        for kind, offset, marker in (("birth", -.13, "o"), ("exit", .13, "^")):
            row = next(row for row in records if row["variant"] == variant and row["metric"] == kind + suffix)
            x = index + offset
            if row["mean"] is not None:
                ax.errorbar([x], [row["mean"] * 100], yerr=[row["sample_sd"] * 100],
                            color=COLORS[variant], marker=marker, markersize=5, capsize=3)
            else:
                ax.text(x, 3, f"{row['n_defined']}/3\ndefined", ha="center", va="bottom", fontsize=6, rotation=90)
            for seed_index, value in enumerate(row["per_seed"]):
                if value["value"] is not None:
                    ax.scatter(x + (seed_index - 1) * .025, value["value"] * 100,
                               color=COLORS[variant], s=8, alpha=.65, zorder=4)
    ax.set_xticks(range(6), ("Plain\nfixed", "Plain\ntask", "Plain\nmagnitude", "Plain\nrandom", "Coupled\nfixed", "Coupled\ntask"))
    ax.set_title(title)
    ax.set_ylabel("Within-seed fraction (%)")
    ax.set_ylim(0, 100)


def _functional_figure(plt, behavior, structure, family_label):
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(3, 2, figsize=(14.8, 14.5))
    for column, condition in enumerate(CONDITIONS):
        ax = axes[0, column]
        for variant in FUNCTIONAL:
            selected = [row for row in behavior if row["variant"] == variant and row["condition"] == condition
                        and row["metric"] == "strict_completion"]
            _line(ax, selected, color=COLORS[variant], label=LABELS[variant],
                  style="--" if variant.startswith("coupled") else "-", scale=100.)
        _stages(ax)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Strict completion (%)")
        ax.set_title("Short" if condition == "short" else "Long wait + long idle")
    for variant in FUNCTIONAL:
        style = "--" if variant.startswith("coupled") else "-"
        _line(axes[1, 0], [row for row in structure["capacity"] if row["variant"] == variant and row["metric"] == "allowed"],
              color=COLORS[variant], label=LABELS[variant], style=style, time_axis=True, seed_ticks=set(SNAPSHOTS))
        bridge = [row for row in structure["bridges"] if row["variant"] == variant and row["metric"] == "active_cross_bank_bridges"]
        _line(axes[1, 1], bridge, color=COLORS[variant], label=LABELS[variant],
              style=style, time_axis=True, seed_ticks=set(SNAPSHOTS))
    for ax in axes[1]:
        for tick in TICKS[:2]:
            ax.axvline(tick / 1024, color="#888888", linewidth=.7, linestyle=":", zorder=0)
        ax.set_xlabel("Live ticks (×1,024)")
        ax.set_xlim(-2, 171)
        ax.set_ylim(bottom=0)
    axes[1, 0].set_title("Eligible recurrent coordinates; numerical support is retained in figure data")
    axes[1, 0].set_ylabel("Allowed coordinates")
    axes[1, 1].set_title("Actual nonzero cross-bank bridges at four saved states")
    axes[1, 1].set_ylabel("Nonzero cross-bank coordinates")
    _ratio_panel(axes[2, 0], structure["acceptance"], "_acceptance", "Accepted candidate batches / executed trials")
    _ratio_panel(axes[2, 1], structure["acceptance"], "_accepted_history_worsening", "History loss increased / accepted batches")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [Line2D([], [], color="black", marker="o", linestyle="none", label="Birth batch"),
                Line2D([], [], color="black", marker="^", linestyle="none", label="Exit batch")]
    labels += ["Birth batch (bottom row)", "Exit batch (bottom row)"]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .035), ncol=4, frameon=False)
    fig.suptitle("Functional structure: behavior, capacity and accepted history damage\n" + family_label, fontsize=13, y=.99)
    fig.text(.5, .01, "Seeds are the independent units (n=3). SD is between seeds; trials/events do not add n. "
             "No accepted batch → undefined history fraction; no zero substitution. Mixed improvement can harm history.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, .95))
    return fig


def plot(acquisition, main, output, *, family_view="mean4"):
    if family_view not in ("mean4", "family0"):
        raise ValueError("family_view must be mean4 or family0")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("figures require a new output directory")
    bindings = {str(Path(__file__).resolve()): sha256(__file__)}
    acq_report, main_report = _audited(acquisition, "acquisition", bindings), _audited(main, "main", bindings)
    acq_scores = _score_rows([acq_report["checked"]("per_seed.csv"), acq_report["checked"]("baseline_per_seed.csv")], "acquisition", family_view)
    main_name = "family_average_per_seed.csv" if family_view == "mean4" else "per_seed.csv"
    main_scores = _score_rows([main_report["checked"](main_name)], "main", family_view)
    acq_data, main_data = _score_data(acq_scores), _score_data(main_scores)
    structure = _structure_data(main_report)
    data = dict(format="organization-figure-data-v1", family_view=family_view, trained_seeds=list(SEEDS),
        source_reports=dict(acquisition=acq_report["metadata"], main=main_report["metadata"]),
        inputs=bindings, acquisition=acq_data, passive_and_functional_behavior=main_data,
        acquisition_paired_differences=_paired_data(acq_scores, "acquisition"),
        passive_paired_differences=_paired_data(main_scores, "main"), functional=structure,
        definitions=dict(mean="Equal family-rate mean within each trained seed (mean4), or family 0 only. Then mean and sample SD across the three trained seeds.",
            missing="An undefined source denominator stays None. The three-seed mean/SD are omitted if any seed is undefined; available seed points remain visible.",
            idle="False actions per 1,000 actual idle opportunities; completion metrics use matured eligible primary requests.",
            capacity="Functional eligibility changes and numerical nonzero support are separate. Bridge counts use four actual saved states, not interpolation evidence.",
            history_damage="Within each seed, accepted batches with strictly increased history loss divided by accepted batches. Zero accepted batches means undefined.",
            points="Seed points receive a small horizontal jitter for visibility; their actual tick and seed are retained in this data file.",
            replication="Actors, families, requests, snapshots, candidate batches and probe events do not create extra independent seeds."))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": .18, "grid.linewidth": .5, "lines.linewidth": 1.8,
        "legend.fontsize": 8, "svg.fonttype": "none", "savefig.dpi": 220})
    family_label = "Equal four-family rate mean within each seed" if family_view == "mean4" else "Withheld family 0 only"
    figures = (
        ("acquisition-shrink", _score_figure(plt, acq_data, ACQUISITION, "Birth-time shrink: paired acquisition controls", family_label, acquisition=True)),
        ("passive-topology", _score_figure(plt, main_data, PASSIVE, "Matched nonlinear field organization (passive)", family_label)),
        ("functional-structure", _functional_figure(plt, main_data, structure, family_label)),
    )
    output.mkdir(parents=True)
    products = {}
    try:
        for name, figure in figures:
            for extension in ("png", "svg"):
                path = output / f"{name}.{extension}"
                metadata = {"Software": "organization.plot"} if extension == "png" else {"Date": None, "Creator": "organization.plot"}
                figure.savefig(path, bbox_inches="tight", metadata=metadata)
                products[path.name] = sha256(path)
        for path, expected in bindings.items():
            if sha256(path) != expected:
                raise ValueError(f"figure input changed while plotting: {path}")
        data["runtime"] = dict(python=sys.version, matplotlib=matplotlib.__version__)
        data_path = output / "figure-data-v1.json"
        data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        products[data_path.name] = sha256(data_path)
        manifest = dict(format="organization-figure-artifacts-v1", inputs=bindings, outputs=products,
                        family_view=family_view, independent_training_seeds=3)
        (output / "artifacts.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        for _, figure in figures:
            plt.close(figure)
    return products


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", type=Path, required=True, help="Passed acquisition aggregate directory")
    parser.add_argument("--main", type=Path, required=True, help="Passed main aggregate directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family-view", choices=("mean4", "family0"), default="mean4")
    args = parser.parse_args(argv)
    try:
        products = plot(args.acquisition, args.main, args.output, family_view=args.family_view)
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(1, f"Organization figures rejected: {error}\n")
    print(json.dumps(dict(output=str(args.output.resolve()), files=products), ensure_ascii=False))


if __name__ == "__main__":
    main()
