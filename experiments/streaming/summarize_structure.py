"""Export the frozen large structural experiment without training or model loads.

Run from any directory. Only the requested evidence JSON/PNG/SVG are written.
The committed evidence JSON contains all seed series needed to regenerate the
figures without the ignored raw experiment directory:
    python experiments/streaming/summarize_structure.py
    python experiments/streaming/summarize_structure.py --evidence docs/evidence/streaming-2026-10-26/structure-results.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PHASES = ("initial", "A1", "B", "A2")
POSITIONS = ("after_local_update", "after_structure", "stage")
PLOT_ARMS = ("gradual_free", "immediate_budget", "peak_static")
LABELS = {"gradual_free": "Gradual free", "immediate_budget": "Immediate budget", "peak_static": "Peak static"}
COLORS = {"gradual_free": "#0072B2", "immediate_budget": "#D55E00", "peak_static": "#595959"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pack_constant(values):
    return values[0] if values and all(value == values[0] for value in values) else values


def unpack(value, count):
    return np.full(count, value, dtype=float) if isinstance(value, (int, float)) else np.asarray(value, dtype=float)


def array_json(value, level=0):
    """Keep primitive numeric series on one line for compact, reviewable JSON."""
    indent = "  " * level
    if isinstance(value, dict):
        return "{\n" + ",\n".join("  " * (level + 1) + json.dumps(key, ensure_ascii=False) + ": " +
            array_json(item, level + 1) for key, item in value.items()) + "\n" + indent + "}"
    if isinstance(value, list) and any(isinstance(item, (dict, list)) for item in value):
        return "[\n" + ",\n".join("  " * (level + 1) + array_json(item, level + 1) for item in value) + "\n" + indent + "]"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def export_evidence(directory):
    manifest_path, summary_path = directory / "manifest.json", directory / "summary.json"
    manifest, summary = read_json(manifest_path), read_json(summary_path)
    config = manifest["configuration"]
    seeds, arms = config["seeds"], manifest["arms"]
    if summary["completed_runs"] != len(seeds) * len(arms):
        raise ValueError("The formal result is incomplete")
    if summary["experiment_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Summary and manifest fingerprints disagree")
    evidence = dict(schema="agi.streaming.structure.results.v1",
        provenance=dict(formal_directory=relative(directory), manifest_sha256=sha256(manifest_path),
            formal_fingerprint=manifest["fingerprint"], summary_sha256=sha256(summary_path),
            formal_source_sha256=manifest["sources"], raw_result_sha256={}),
        configuration=config, formal_environment=manifest["environment"],
        measurement=dict(seeds=seeds, arms=arms, stages=list(PHASES),
            validation_examples_per_context=config["n_val"], test_examples_per_context=config["n_test"],
            transition_probe_examples_per_context=min(config["probe_size"], config["n_val"]),
            series_encoding="Numeric arrays align with shared axes; a scalar broadcasts over the entire corresponding axis",
            uncertainty="Sample SD across three seed means; window observations are not independent seeds",
            connection_curve="Actual nonzero coefficients, not allocated capacity pools",
            counters="Cumulative events can revisit the same connection location",
            cut_only="Read-only pre-window weights with only the post-window retention mask; no newborn weights",
            capacity_control="Peak static is a larger pool derived from the free training trajectory, not test labels",
            timing="Formal GPU work ran concurrently with other reader jobs; times are not exclusive-GPU benchmarks"),
        formal_summary=summary["summary"], elapsed_seconds=summary["elapsed_seconds"], axes={}, runs=[])
    connection_grid = validation_grid = window_grid = recovery_grid = None
    paired_streams = {}
    for seed in seeds:
        for arm in arms:
            path = directory / f"{arm}-seed{seed}.json"
            raw = read_json(path)
            if raw["experiment_fingerprint"] != manifest["fingerprint"] or not raw["test_evaluated"]:
                raise ValueError(f"Wrong fingerprint or nonformal result: {path}")
            evidence["provenance"]["raw_result_sha256"][path.name] = sha256(path)
            if seed in paired_streams and raw["phase_streams"] != paired_streams[seed]:
                raise ValueError("Comparison arms do not have identical data streams")
            paired_streams[seed] = raw["phase_streams"]
            connections = raw["connection_curve"]
            c_grid = [(point["step"], PHASES.index(point["phase"]), POSITIONS.index(point["position"]))
                      for point in connections]
            val = [dict(step=0, phase="initial", validation=raw["stages"][0]["validation"])] + raw["validation_curve"]
            v_grid = [(point["step"], PHASES.index(point["phase"])) for point in val]
            events = raw["structural_events"]
            w_grid = [(event["step"], PHASES.index(event["phase"])) for event in events]
            recovery = [(event["step"], point) for event in events for point in event["recovery"]]
            r_grid = [(event_step, point["offset"]) for event_step, point in recovery]
            for title, expected, observed in (("connection", connection_grid, c_grid),
                ("validation", validation_grid, v_grid), ("window", window_grid, w_grid), ("recovery", recovery_grid, r_grid)):
                if expected is not None and observed != expected:
                    raise ValueError(f"Unaligned {title} grid in {path.name}")
            connection_grid, validation_grid, window_grid, recovery_grid = c_grid, v_grid, w_grid, r_grid
            stage_records = [dict(stage=stage["stage"], step=stage["step"], test=stage["test"],
                validation=stage["validation"], actual_nonzero=stage["actual_nonzero"],
                pool_connections=stage["pool_connections"],
                continuous_reference_survival=stage["continuous_reference_survival"]) for stage in raw["stages"]]
            window_metrics = {}
            for context in ("A", "B"):
                window_metrics[context] = {metric: pack_constant([event["probe"][context][metric] for event in events])
                    for metric in ("accuracy_change_pp", "prediction_flip_rate", "probability_total_variation",
                                   "cut_only_accuracy_change_pp", "cut_only_prediction_flip_rate",
                                   "cut_only_probability_total_variation")}
            run = dict(seed=seed, arm=arm, stages=stage_records,
                active_edges=pack_constant([point["actual_nonzero"] for point in connections]),
                pool_edges=pack_constant([point["pool_connections"] for point in connections]),
                validation={context: [point["validation"][context]["accuracy"] for point in val] for context in ("A", "B")},
                window_probe=window_metrics,
                window_added=pack_constant([event["pool_added_at_window"] for event in events]),
                window_removed=pack_constant([event["pool_removed_at_window"] for event in events]),
                window_nonzero_cuts=pack_constant([event["nonzero_removed_at_window"] for event in events]),
                zero_exits_between_windows=pack_constant([event["zero_exits_since_previous_window"] for event in events]),
                recovery_validation={context: [point["probe"][context]["accuracy"] for _, point in recovery] for context in ("A", "B")},
                final_counts=raw["final_counts"], joint_peak_gpu_bytes=raw["joint_peak_gpu_bytes"],
                seed_total_seconds=raw["seed_total_seconds"], joint_population_seconds=raw["joint_population_seconds"],
                joint_population_timings=raw["joint_population_timings"],
                checkpoint_sha256=raw["checkpoint_sha256"])
            evidence["runs"].append(run)
    evidence["axes"] = dict(phase_labels=list(PHASES), position_labels=list(POSITIONS),
        connection=dict(step=[row[0] for row in connection_grid], phase=[row[1] for row in connection_grid], position=[row[2] for row in connection_grid]),
        validation=dict(step=[row[0] for row in validation_grid], phase=[row[1] for row in validation_grid]),
        window=dict(step=[row[0] for row in window_grid], phase=[row[1] for row in window_grid]),
        recovery=dict(event_step=[row[0] for row in recovery_grid], offset=[row[1] for row in recovery_grid]))
    evidence["provenance"]["paired_phase_streams"] = {str(seed): value for seed, value in paired_streams.items()}
    # Verify portable series directly against published aggregate stage means.
    for arm in arms:
        rows = [run for run in evidence["runs"] if run["arm"] == arm]
        for index, stage in enumerate(PHASES):
            for context in ("A", "B"):
                values = [run["stages"][index]["test"][context]["accuracy"] for run in rows]
                stored = evidence["formal_summary"][f"{arm}/{stage}/{context}"]
                if stored["values"] != values or stored["mean"] != statistics.mean(values) or stored["std"] != statistics.stdev(values):
                    raise ValueError("Published statistics disagree with exported seed records")
    return evidence


def phase_background(ax, phase_steps, end):
    for left, right, text, color in ((0, phase_steps, "$A_1$", "#eef4f8"),
        (phase_steps, 2 * phase_steps, "$B$", "#fbf5e9"), (2 * phase_steps, end, "$A_2$", "#eef4f8")):
        ax.axvspan(left / 1000, right / 1000, color=color, zorder=0)
        ax.text((left + right) / 2000, 1.015, text, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=11)
    for step in (phase_steps, 2 * phase_steps):
        ax.axvline(step / 1000, color="#999999", lw=.7, ls=":", zorder=1)
    ax.set_xlim(0, end / 1000)
    ax.set_xticks(np.arange(4) * phase_steps / 1000)
    ax.set_xlabel("Training steps (thousands)")


def render(evidence, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.titlesize": 11.5, "axes.labelsize": 10.5, "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5, "svg.hashsalt": evidence["provenance"]["formal_fingerprint"]})
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.6))
    fig.subplots_adjust(left=.055, right=.99, top=.84, bottom=.23, wspace=.31)
    phase_steps = evidence["configuration"]["steps_per_phase"]
    end = 3 * phase_steps
    val_x = np.asarray(evidence["axes"]["validation"]["step"]) / 1000
    edge_x = np.asarray(evidence["axes"]["connection"]["step"]) / 1000
    rows = {arm: [run for run in evidence["runs"] if run["arm"] == arm] for arm in PLOT_ARMS}
    # Draw static first so its small cross markers remain distinguishable from
    # the nearly coincident free curves, without changing the meaning of SD.
    for arm in reversed(PLOT_ARMS):
        for context, style in (("A", "-"), ("B", "--")):
            values = np.asarray([run["validation"][context] for run in rows[arm]]) * 100
            mean, sd = values.mean(0), values.std(0, ddof=1)
            axes[0].fill_between(val_x, mean - sd, mean + sd, color=COLORS[arm], alpha=.10, lw=0)
            axes[0].plot(val_x, mean, ls=style, color=COLORS[arm], lw=1.8 if arm != "peak_static" else 1.0,
                         marker="x" if arm == "peak_static" else None, ms=3.5, zorder=3 if arm == "gradual_free" else 2)
        values = np.asarray([unpack(run["active_edges"], len(edge_x)) for run in rows[arm]])
        mean, sd = values.mean(0), values.std(0, ddof=1)
        axes[1].fill_between(edge_x, mean - sd, mean + sd, color=COLORS[arm], alpha=.15, lw=0)
        axes[1].plot(edge_x, mean, color=COLORS[arm], lw=1.8 if arm != "peak_static" else 1.2,
                     marker="x" if arm == "peak_static" else None, markevery=120, ms=3.5)
    phase_background(axes[0], phase_steps, end)
    phase_background(axes[1], phase_steps, end)
    axes[0].set_title("a  Validation accuracy", loc="left", pad=29, fontweight="bold")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_yticks([0, 25, 50, 75, 100])
    axes[1].set_title("b  Actual active connections", loc="left", pad=29, fontweight="bold")
    axes[1].set_ylabel("Nonzero weights")
    axes[1].set_ylim(0, 70000)
    axes[1].set_yticks([0, 20000, 40000, 60000])
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k" if value else "0"))

    window_phase = np.asarray(evidence["axes"]["window"]["phase"])
    in_b = window_phase == PHASES.index("B")
    bar_width = .32
    for index, arm in enumerate(PLOT_ARMS):
        for shift, metric, hatch in ((-.17, "accuracy_change_pp", None), (.17, "cut_only_accuracy_change_pp", "///")):
            seed_means = [unpack(run["window_probe"]["A"][metric], len(window_phase))[in_b].mean() for run in rows[arm]]
            mean, sd = np.mean(seed_means), np.std(seed_means, ddof=1)
            axes[2].bar(index + shift, mean, bar_width, yerr=sd, capsize=3,
                color=COLORS[arm] if hatch is None else "white", edgecolor=COLORS[arm], hatch=hatch,
                linewidth=1.1, error_kw={"elinewidth": 1.1, "ecolor": "#222222"}, zorder=3)
    axes[2].axhline(0, color="#333333", lw=.8)
    axes[2].set_xticks(range(3), ["Free", "Budget", "Static"])
    axes[2].set_ylabel("Accuracy change per window (pp)")
    axes[2].set_ylim(-.27, .035)
    axes[2].set_yticks([-.25, -.20, -.15, -.10, -.05, 0])
    axes[2].set_title("c  Old-A shock during B", loc="left", pad=29, fontweight="bold")
    axes[2].legend(handles=[Patch(facecolor="#777777", label="Net window"),
        Patch(facecolor="white", edgecolor="#777777", hatch="///", label="Cut only")],
        loc="lower left", frameon=False, fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#cccccc", lw=.5, alpha=.55, zorder=0)
    legend = [Line2D([0], [0], color=COLORS[arm], lw=2, label=LABELS[arm]) for arm in PLOT_ARMS]
    legend.extend([Line2D([0], [0], color="#333333", lw=1.5, ls="-", label="Task A"),
                   Line2D([0], [0], color="#333333", lw=1.5, ls="--", label="Task B")])
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .075), ncol=5, frameon=False, fontsize=10)
    fig.text(.055, .025, "Bands and error bars: sample SD across 3 seeds.  Shock: fixed 1,024-example validation probe; negative means accuracy loss.", fontsize=9, color="#444444")
    png, svg = output / "structure.png", output / "structure.svg"
    fig.savefig(png, dpi=180, facecolor="white", metadata={"Software": "agi/summarize_structure.py"})
    fig.savefig(svg, facecolor="white", metadata={"Creator": "agi/summarize_structure.py", "Date": None})
    plt.close(fig)
    return png, svg


def absolute(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/streaming/structure-free-v1")
    parser.add_argument("--output-dir", default="docs/evidence/streaming-2026-10-26")
    parser.add_argument("--evidence", help="Regenerate entirely from portable evidence JSON")
    args = parser.parse_args(argv)
    output = absolute(args.output_dir)
    evidence = read_json(absolute(args.evidence)) if args.evidence else export_evidence(absolute(args.input_dir))
    evidence.pop("rendering", None)
    data_hash = hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    png, svg = render(evidence, output)
    evidence["rendering"] = dict(script=relative(__file__), script_sha256=sha256(__file__),
        matplotlib=matplotlib.__version__, numerical_payload_sha256=data_hash,
        figures={relative(png): sha256(png), relative(svg): sha256(svg)})
    path = output / "structure-results.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(array_json(evidence) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(dict(evidence=relative(path), bytes=path.stat().st_size,
        sha256=sha256(path), numerical_payload_sha256=data_hash,
        figures=[relative(png), relative(svg)])), flush=True)


if __name__ == "__main__":
    main()
