"""Figures from SHA-bound audited tables; never load a training checkpoint.

Importing this module reads no study data and imports no training/audit code.
``validate`` checks both audit gates and the plotting contract without drawing.
``render`` writes PNG/SVG plus a data/source SHA manifest into a new folder.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CREDIT_CHECK = ROOT / "docs/evidence/undirected-2026-10-30/credit-path-check.json"
KEYS = ("interface", "mode", "credit", "structure", "course")
SEEDS = (0, 1, 2)
FAMILIES = (0, 1, 2, 3)
CONDITIONS = ("short", "long_both")
METRICS = ("strict_completion", "completion", "inactive_per_1000_idle")
LABELS = {"strict_completion": "Strict completion (%)", "completion": "Completion (%)",
          "inactive_per_1000_idle": "Inactive actions / 1,000 idle opportunities"}
FACTORS = {"strict_completion": 100., "completion": 100., "inactive_per_1000_idle": 1.}
BASE = dict(interface="ports", mode="undirected", credit="rtrl", structure="fixed", course="warmup")
COLORS = ("#24658a", "#be6d32", "#3d8769", "#855a91", "#747b83")


class PlotDataError(ValueError):
    pass


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _variant(**changes):
    return tuple(({**BASE, **changes})[name] for name in KEYS)


COMPARISONS = {
    "interfaces_512": [(f"{interface.title()}\n{mode}", _variant(interface=interface, mode=mode, credit="recorded512"))
                       for interface in ("core", "ports") for mode in ("undirected", "untied")],
    "temporal_credit": [(name, _variant(credit=method)) for name, method in
                        (("128", "tbptt128"), ("512", "recorded512"), ("RTRL", "rtrl"), ("RTRL reset", "rtrl_reset"))],
    "structure": [(name, _variant(structure=name)) for name in ("fixed_probe", "grow", "retire", "both")],
    "course": [(f"{name}\n{course}", _variant(credit=method, course=course))
               for name, method in (("128", "tbptt128"), ("RTRL", "rtrl")) for course in ("warmup", "wide")],
}


def _decode(value):
    if value is None or value == "":
        return None
    if value[:1] in ("{", "[") or value in ("null", "true", "false"):
        return json.loads(value)
    try:
        return int(value)
    except ValueError:
        try:
            result = float(value)
        except ValueError:
            return value
        if not math.isfinite(result):
            raise PlotDataError("audited CSV contains a nonfinite number")
        return result


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _same(actual, expected):
    return actual is None if expected is None else _number(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)


def _key(row):
    return tuple(row[name] for name in KEYS)


def _tick(row):
    return row.get("checkpoint_tick", row.get("tick"))


def _path(value, parent):
    result = Path(value)
    return result.resolve() if result.is_absolute() else (parent / result).resolve()


def _verified_entry(entry, parent, inputs, role):
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("sha256"), str):
        raise PlotDataError(f"incomplete audited artifact binding: {role}")
    path = _path(entry["path"], parent)
    digest = file_sha256(path)
    if digest != entry["sha256"]:
        raise PlotDataError(f"audited artifact bytes changed: {path}")
    inputs[str(path)] = dict(sha256=digest, role=role)
    return path


def _read_table(entry, parent, inputs, name):
    path = _verified_entry(entry, parent, inputs, name)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = [{key: _decode(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    if len(rows) != entry.get("rows"):
        raise PlotDataError(f"audited table row count differs: {name}")
    return rows


def _audit_gate(branches, parent, inputs):
    """Require independent reconstruction; only RTRL has full saved S to project."""
    counts = {"projected_full_S": 0, "primal_history_only": 0, "window_method_no_saved_S": 0,
              "no_successful_live_blocks": 0}
    for row in branches:
        audit = row.get("sensitivity_audit")
        if not isinstance(audit, dict) or audit.get("status") == "pending_independent_jvp":
            raise PlotDataError(f"independent sensitivity audit is not ready: {row['id']}")
        if row["last_tick"] == 0 and audit.get("status") == "not_applicable_no_successful_live_blocks":
            if audit.get("calibrated") is not False:
                raise PlotDataError("a zero-block branch cannot claim a saved-S calibration")
            if audit.get("path"):
                _verified_entry(audit, parent, inputs, f"independent_audit:{row['id']}")
            counts["no_successful_live_blocks"] += 1
            continue
        if row["credit"] not in ("rtrl", "rtrl_reset") and audit.get("status") == "not_required_method_has_no_full_S":
            if audit.get("calibrated") is not False:
                raise PlotDataError("window credit has no full saved S to claim calibrated")
            counts["window_method_no_saved_S"] += 1
            continue
        path = _verified_entry(audit, parent, inputs, f"independent_audit:{row['id']}")
        value = _json(path)
        if value.get("format") != "undirected-sensitivity-audit-v1" or value.get("status") != audit.get("status"):
            raise PlotDataError("independent audit format/status differs from the accounting audit")
        if value.get("method") != row["credit"]:
            raise PlotDataError("independent audit method differs from its branch")
        if row["last_tick"] == 0:
            if value.get("status") != "not_applicable_no_successful_live_blocks" or audit.get("calibrated") is not False:
                raise PlotDataError("zero-block branch must explicitly retain its not-applicable audit status")
            counts["no_successful_live_blocks"] += 1
        elif value.get("status") != "passed":
            raise PlotDataError("independent history reconstruction did not pass")
        elif row["credit"] in ("rtrl", "rtrl_reset"):
            if (audit.get("calibrated") is not True or value.get("sensitivity_calibrated") is not True
                    or value.get("directions") != 2 or value.get("validation_scope") != "primal_history_and_projected_saved_S"):
                raise PlotDataError("RTRL requires its two-direction saved-S projection calibration")
            counts["projected_full_S"] += 1
        else:
            if audit.get("calibrated") is not False:
                raise PlotDataError("window credit has no full saved S to claim calibrated")
            counts["primal_history_only"] += 1
    return counts


def family_values(branches, per_seed, ticks):
    """Four equal family rates within a seed, including absent failed branches."""
    lookup = {}
    for row in per_seed:
        key = row["id"], _tick(row), row["condition"], row["family"]
        if key in lookup:
            raise PlotDataError("duplicate audited family evaluation")
        lookup[key] = row
        denominator = row["primary_matured"]
        for metric, numerator in (("strict_completion", row["primary_clean_completed"]), ("completion", row["primary_completed"])):
            expected = numerator / denominator if denominator else None
            if not _same(row[metric], expected):
                raise PlotDataError("completion rate differs from its audited matured-request denominator")
    rows = []
    for branch in branches:
        for tick in ticks:
            for condition in CONDITIONS:
                families = [lookup.get((branch["id"], tick, condition, family)) for family in FAMILIES]
                if branch["status"] == "success" and any(row is None for row in families):
                    raise PlotDataError("successful seed is missing an audited family fork")
                for metric in METRICS:
                    values = [row[metric] for row in families if row is not None and row[metric] is not None]
                    value = statistics.mean(values) if branch["status"] == "success" and len(values) == 4 else None
                    rows.append(dict(id=branch["id"], seed=branch["seed"], **{name: branch[name] for name in KEYS},
                        tick=tick, condition=condition, metric=metric, value=value, status=branch["status"], n_families_defined=len(values)))
    return rows


def summarize(rows):
    if sorted(row["seed"] for row in rows) != list(SEEDS):
        raise PlotDataError("a displayed group must contain the three declared seeds exactly once")
    success = sum(row["status"] == "success" for row in rows)
    values = [row["value"] for row in rows if row["value"] is not None]
    if any(not _number(value) for value in values):
        raise PlotDataError("per-seed values must be finite or explicitly undefined")
    complete = success == len(values) == 3
    return dict(mean=statistics.mean(values) if complete else None,
                sample_sd=statistics.stdev(values) if complete else None, n_success=success, n_defined=len(values), rows=rows)


def _check_family_tables(rows, average_rows, summary_rows):
    averages = {(row["id"], _tick(row), row["condition"]): row for row in average_rows}
    summaries = {(_key(row), _tick(row), row["condition"], row["metric"]): row for row in summary_rows}
    if len(averages) != len(average_rows) or len(summaries) != len(summary_rows):
        raise PlotDataError("duplicate audited family-average or summary rows")
    groups = {}
    for row in rows:
        average = averages.get((row["id"], row["tick"], row["condition"]))
        if average is None or average["status"] != row["status"] or not _same(average[row["metric"]], row["value"]):
            raise PlotDataError("audited family-average table differs from four original family rates")
        counts = average.get("defined_family_counts")
        if not isinstance(counts, dict) or counts[row["metric"]] != row["n_families_defined"]:
            raise PlotDataError("audited family denominator-definition count differs")
        key = _key(row), row["tick"], row["condition"], row["metric"]
        groups.setdefault(key, []).append(row)
    for key, group in groups.items():
        actual, expected = summaries.get(key), summarize(group)
        if (actual is None or not _same(actual["mean"], expected["mean"])
                or not _same(actual["sample_sd"], expected["sample_sd"])
                or actual["n_total"] != 3 or actual["n_success"] != expected["n_success"] or actual["n_defined"] != expected["n_defined"]):
            raise PlotDataError("audited three-seed summary differs from declared-seed aggregation")


def _check_graphs(graphs, checkpoints):
    anchors = {(row["id"], _tick(row)): row for row in checkpoints}
    identities = set()
    for graph in graphs:
        key = graph["branch_id"], graph["role"]
        if key in identities or graph["role"] not in ("birth", "last_valid") or graph.get("downsampled") is not False:
            raise PlotDataError("graphs must be unique full-node birth/last-valid audited artifacts")
        identities.add(key)
        anchor = anchors.get((graph["branch_id"], graph["tick"]))
        if (anchor is None or anchor["matrix_hash"] != graph["matrix_hash"] or anchor["checkpoint_sha256"] != graph["checkpoint_sha256"]
                or graph["state_width"] != anchor["state_width"]):
            raise PlotDataError("audited graph differs from its checkpoint-table binding")
        _matrix(graph)


def _comparison_contract(header):
    """A failed empirical equality does not invalidate a passed count audit."""
    audit, comparison = header.get("audit", {}), header.get("fixed_probe_comparison", {})
    if audit.get("scope") != "provenance_and_accounting":
        raise PlotDataError("the passed audit must explicitly identify its provenance/accounting scope")
    rows = comparison.get("per_seed", [])
    if comparison.get("check_code") != "FIXED_PROBE_SAME_LEARNING_EXECUTION" or [r.get("seed") for r in rows] != list(SEEDS):
        raise PlotDataError("the original fixed/probe exact comparison and all three seeds must remain disclosed")
    for row in rows:
        for name in ("training_trace", "frozen_evaluation"):
            value = row.get(name, {})
            count, total = value.get("mismatch_count"), value.get("compared_records")
            if (type(count) is not int or type(total) is not int or not 0 <= count <= total
                    or value.get("status") != ("failed" if count else "passed")
                    or (value.get("first_mismatch_tick") is None) != (count == 0)):
                raise PlotDataError("fixed/probe equality status contradicts its exact difference inventory")
            for field in value.get("fields", {}).values():
                n = field.get("mismatch_count")
                if type(n) is not int or not 0 <= n <= count or (field.get("first_mismatch_tick") is None) != (n == 0):
                    raise PlotDataError("fixed/probe field differences are inconsistent")
    training_status = "failed" if any(r["training_trace"]["status"] == "failed" for r in rows) else "passed"
    evaluation_status = "different" if any(r["frozen_evaluation"]["status"] == "failed" for r in rows) else "equal"
    if (comparison.get("status") != training_status or comparison.get("frozen_evaluation_status") != evaluation_status
            or audit.get("scopes", {}).get("fixed_probe_exact_training_equality") != training_status
            or audit.get("scopes", {}).get("fixed_probe_frozen_evaluation_comparison") != evaluation_status):
        raise PlotDataError("fixed/probe differences cannot be silently relabeled as exact equivalence")
    return comparison


def read_audited_bundle(report):
    """Read audited analysis artifacts only; no path in inputs.csv is opened."""
    path = Path(report).resolve()
    header, inputs = _json(path), {str(path): dict(sha256=file_sha256(path), role="accounting_audit_header")}
    if header.get("format") != "undirected-analysis-v1" or header.get("audit", {}).get("status") != "passed":
        raise PlotDataError("a passed source-bound accounting report is required")
    _comparison_contract(header)
    if header.get("mechanical_test") is not False:
        raise PlotDataError("formal behavior figures require the formal audited protocol")
    counts = header.get("counts", {})
    if counts.get("declared_branches") != 42 or counts.get("accounted_branches") != 42:
        raise PlotDataError("all 42 declared branches must be accounted before plotting")
    required = ("branches", "checkpoints", "per_seed", "family_average_per_seed", "family_average_summary")
    if not all(name in header.get("tables", {}) for name in required):
        raise PlotDataError("audited family/branch/checkpoint tables are not ready")
    tables = {name: _read_table(header["tables"][name], path.parent, inputs, name) for name in required}
    branches = tables["branches"]
    if (len(branches) != 42 or len({row["id"] for row in branches}) != 42
            or any(row["status"] not in ("success", "numerical_failure") for row in branches)):
        raise PlotDataError("incomplete or nonterminal declared branch inventory")
    variants = {_key(row) for row in branches}
    if len(variants) != 14 or any(sorted(row["seed"] for row in branches if _key(row) == variant) != list(SEEDS) for variant in variants):
        raise PlotDataError("formal variants require exactly seeds 0/1/2")
    independent = _audit_gate(branches, path.parent, inputs)
    ticks = [tick for tick in header["checkpoint_ticks"] if tick]
    values = family_values(branches, tables["per_seed"], ticks)
    _check_family_tables(values, tables["family_average_per_seed"], tables["family_average_summary"])
    graph_path = _verified_entry(header.get("graph_artifact"), path.parent, inputs, "audited_full_node_graphs")
    graph_data = _json(graph_path)
    if (graph_data.get("format") != "undirected-audited-graphs-v1"
            or graph_data.get("protocol_sha256") != header["protocol_sha256"]
            or len(graph_data.get("graphs", [])) != header["graph_artifact"]["rows"]):
        raise PlotDataError("graph artifact format/protocol/inventory mismatch")
    _check_graphs(graph_data["graphs"], tables["checkpoints"])
    for source, expected in header.get("analysis_sources", {}).items():
        _verified_entry(dict(path=source, sha256=expected), path.parent, inputs, "accounting_analysis_source")
    # inputs.csv is itself bound, but this renderer deliberately never follows
    # its raw checkpoint/ledger/evaluation paths.
    _verified_entry(header["input_manifest"], path.parent, inputs, "audited_input_manifest")
    return dict(header=header, tables=tables, values=values, graphs=graph_data["graphs"], inputs=inputs,
                independent_audit_counts=independent)


def _matrix(graph):
    import numpy as np
    width = graph["state_width"]
    if not isinstance(width, int) or width < 1 or graph.get("width", width) != width:
        raise PlotDataError("invalid audited graph dimensions")
    matrix, seen = np.zeros((width, width), dtype=np.float64), set()
    eligible = {tuple(edge) for edge in graph["eligible_edges"]}
    if len(eligible) != len(graph["eligible_edges"]) or any(len(edge) != 2 or not 0 <= edge[0] < edge[1] < width for edge in eligible):
        raise PlotDataError("noncanonical audited eligibility pairs")
    for u, v, forward, reverse in graph["support_edges"]:
        if (not isinstance(u, int) or not isinstance(v, int) or not 0 <= u < v < width or (u, v) in seen
                or (u, v) not in eligible or not _number(forward) or not _number(reverse) or (forward == reverse == 0)):
            raise PlotDataError("invalid audited physical pair weights")
        seen.add((u, v))
        matrix[u, v], matrix[v, u] = forward, reverse
    if graph["mode"] == "undirected" and not np.array_equal(matrix, matrix.T):
        raise PlotDataError("shared audited graph is not exactly symmetric")
    return matrix


def _graphics():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    return plt


def _data_sha(inputs):
    payload = json.dumps(inputs, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _save(figure, folder, name, inputs, outputs):
    import matplotlib as mpl
    digest = _data_sha(inputs)
    figure.text(.01, .005, f"audited data SHA256 {digest}", fontsize=5.5, color="#626a73")
    with mpl.rc_context({"svg.hashsalt": digest}):
        result = dict(name=name, data_sha256=digest)
        for extension in ("png", "svg"):
            path = folder / f"{name}.{extension}"
            metadata = {"Creator": "experiments.undirected.plot", "Description": f"Audited input data SHA256 {digest}"}
            if extension == "svg":
                metadata["Date"] = None
            figure.savefig(path, dpi=220, bbox_inches="tight", metadata=metadata)
            result[extension] = dict(path=str(path.resolve()), sha256=file_sha256(path))
        outputs.append(result)
    _graphics().close(figure)


def _group(bundle, variant, tick, condition, metric):
    rows = [row for row in bundle["values"] if _key(row) == variant and row["tick"] == tick
            and row["condition"] == condition and row["metric"] == metric]
    return summarize(rows)


def _points(axis, group, position, color, factor=1.):
    for row, offset in zip(sorted(group["rows"], key=lambda row: row["seed"]), (-.12, 0., .12)):
        if row["value"] is not None:
            axis.scatter(position + offset, row["value"] * factor, color=color, s=22, zorder=3)
        else:
            marker, label = ("x", "NF") if row["status"] != "success" else ("D", "U")
            axis.scatter(position + offset, -.10, marker=marker, s=20, color="#ae3e36", transform=axis.get_xaxis_transform(), clip_on=False)
            axis.annotate(f"{label}{row['seed']}", (position + offset, -.10), xycoords=axis.get_xaxis_transform(),
                          xytext=(0, -9), textcoords="offset points", ha="center", va="top", fontsize=6, color="#ae3e36")
    if group["mean"] is not None:
        axis.errorbar(position, group["mean"] * factor, yerr=group["sample_sd"] * factor,
                      color="#202833", marker="_", markersize=18, capsize=4, linewidth=1.4, zorder=4)
    axis.grid(axis="y", color="#e4e8eb", linewidth=.6)


def outcome_figure(bundle, comparison, tick, folder, outputs):
    plt = _graphics()
    variants = COMPARISONS[comparison]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.8), squeeze=False)
    for row, condition in enumerate(CONDITIONS):
        for col, metric in enumerate(METRICS):
            axis = axes[row, col]
            for index, (_, variant) in enumerate(variants):
                _points(axis, _group(bundle, variant, tick, condition, metric), index, COLORS[index], FACTORS[metric])
            axis.set_xticks(range(len(variants)), [name for name, _ in variants], fontsize=8)
            axis.set_ylabel(LABELS[metric])
            axis.set_title(condition)
            axis.margins(x=.16, y=.16)
    fig.suptitle(f"{comparison.replace('_', ' ')} | checkpoint {tick:,}", fontsize=13)
    fig.text(.02, .025, "Points: independent seeds 0/1/2, each averaging four defined family rates. Black mean ± sample SD only at 3/3 success.\n"
             "NF: numerical failure; U: undefined family denominator. No failure is placed at rate zero. Completion denominator: matured post-fork requests.", fontsize=8)
    fig.tight_layout(rect=(0, .10, 1, .95), h_pad=3)
    _save(fig, folder, f"outcomes_{comparison}_tick_{tick:06d}", bundle["inputs"], outputs)


def cost_figure(bundle, tick, folder, outputs, comparison="temporal_credit"):
    plt = _graphics()
    variants = COMPARISONS[comparison]
    branches = {row["id"]: row for row in bundle["tables"]["branches"]}
    checkpoints = {(row["id"], _tick(row)): row for row in bundle["tables"]["checkpoints"]}
    definitions = (
        ("Recorded forward actor-ticks", lambda r: sum(r["work"].get(name, 0) for name in
            ("live_forward_actor_ticks", "aux_forward_actor_ticks", "credit_loss_forward_actor_ticks",
             "credit_jacobian_forward_actor_ticks", "structural_probe_forward_actor_ticks"))),
        ("Recorded J @ S MACs", lambda r: r["work"]["credit_sensitivity_matmul_mac"]),
        ("Saved full S (MiB)", lambda r: r["sensitivity_bytes"] / (1024 ** 2)),
        ("Dynamic potential parameters", lambda r: r["dynamic_parameters"]),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    for axis, (label, value_fn) in zip(axes.flat, definitions):
        for index, (_, variant) in enumerate(variants):
            rows = []
            for seed in SEEDS:
                branch = next(row for row in branches.values() if _key(row) == variant and row["seed"] == seed)
                checkpoint = checkpoints.get((branch["id"], tick))
                value = value_fn(checkpoint) if branch["status"] == "success" and checkpoint is not None else None
                rows.append(dict(seed=seed, status=branch["status"], value=value))
            _points(axis, summarize(rows), index, COLORS[index])
        axis.set_xticks(range(len(variants)), [name for name, _ in variants])
        axis.set_ylabel(label)
        axis.margins(x=.16, y=.16)
    fig.suptitle(f"{comparison.replace('_', ' ')} compute/storage | equal checkpoint {tick:,}", fontsize=13)
    fig.text(.02, .025, "Forward actor-ticks are recorded logical counters, not total FLOPs or wall-time measurements. J @ S excludes Jacobian construction.\n"
             "All potential coordinates count even when inactive. NF/U markers retain failed or unavailable equal-budget endpoints; no replacement seeds.", fontsize=8)
    fig.tight_layout(rect=(0, .10, 1, .95), h_pad=3)
    _save(fig, folder, f"cost_{comparison}_tick_{tick:06d}", bundle["inputs"], outputs)


def _find_graph(bundle, variant, role, seed=0):
    rows = [row for row in bundle["graphs"] if _key(row) == variant and row["role"] == role and row["seed"] == seed]
    if len(rows) != 1:
        raise PlotDataError("missing prespecified seed-zero audited graph; do not select a replacement based on outcomes")
    return rows[0]


def geometry_figures(bundle, folder, outputs, layout_seed=731):
    import numpy as np
    from matplotlib.collections import LineCollection
    plt = _graphics()
    graph = _find_graph(bundle, _variant(credit="recorded512"), "birth")
    matrix, width = _matrix(graph), graph["state_width"]
    labels = np.random.default_rng(layout_seed).permutation(width)
    angles = np.arange(width) * (2 * np.pi / width)
    positions = np.empty((width, 2))
    positions[labels] = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.2))
    axis = axes[0, 0]
    segments = [[positions[u], positions[v]] for u, v, _, _ in graph["support_edges"]]
    axis.add_collection(LineCollection(segments, colors="#6f7b87", linewidths=.45, alpha=.22))
    inputs, outports = set(graph["input_ports"]), set(graph["output_ports"])
    for nodes, color, name in ((sorted(inputs), COLORS[0], "External input ports"), (sorted(outports), COLORS[1], "External output ports"),
                               ([node for node in range(width) if node not in inputs | outports], "#69727b", "Other state nodes")):
        if nodes:
            axis.scatter(positions[nodes, 0], positions[nodes, 1], color=color, s=23, label=name, zorder=3)
    axis.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), aspect="equal")
    axis.axis("off")
    axis.legend(loc="lower center", fontsize=7, frameon=False)
    axis.set_title(f"Audited birth graph: all {width} nodes\nFixed random circular layout, seed {layout_seed}")
    vmax = max(float(np.abs(matrix).max()), 1e-12)
    for axis, data, title in ((axes[0, 1], matrix, "Actual shared birth W"), (axes[1, 0], matrix - matrix.T, "Actual W − Wᵀ")):
        image = axis.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        fig.colorbar(image, ax=axis, shrink=.8)
        axis.set(title=title, xlabel="Source node j", ylabel="Receiving node i")
    variants = COMPARISONS["interfaces_512"]
    counts = []
    for _, variant in variants:
        values = [row["total_parameters"] for row in bundle["tables"]["checkpoints"] if _key(row) == variant and _tick(row) == 0]
        if len(values) != 3 or len(set(values)) != 1:
            raise PlotDataError("audited birth parameter count must agree across all declared seeds")
        counts.append(values[0])
    axis = axes[1, 1]
    axis.bar(range(4), counts, color=COLORS[:4])
    axis.set_xticks(range(4), [name for name, _ in variants], fontsize=8)
    axis.set_ylabel("Actual independent parameters")
    for index, count in enumerate(counts):
        axis.text(index, count, f"{count:,}", ha="center", va="bottom", fontsize=8)
    axis.margins(y=.16)
    fig.suptitle("Same complete newborn function; different interface and direction freedoms", fontsize=12)
    fig.text(.02, .015, "Circle positions show connectivity only, not learned clusters or layers. Port colors identify declared external interfaces only.\n"
             "The zero diagonal is an experiment setting. The transpose difference uses the same color scale as W; geometry is not a behavioral score.", fontsize=8)
    fig.tight_layout(rect=(0, .07, 1, .95))
    _save(fig, folder, "audited_birth_graph_and_parameter_budget", bundle["inputs"], outputs)
    graphs = [_find_graph(bundle, _variant(credit="recorded512", mode=mode), "last_valid") for mode in ("undirected", "untied")]
    matrices = [_matrix(graph) for graph in graphs]
    scales = [max(float(np.abs(m).max()) for m in matrices), max(float(np.abs(m - m.T).max()) for m in matrices)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
    for row, (graph, matrix) in enumerate(zip(graphs, matrices)):
        for col, data in enumerate((matrix, matrix - matrix.T)):
            scale = max(scales[col], 1e-12)
            image = axes[row, col].imshow(data, cmap="RdBu_r", vmin=-scale, vmax=scale, interpolation="nearest")
            fig.colorbar(image, ax=axes[row, col], shrink=.8)
            axes[row, col].set(title=f"{graph['mode']} | {'W' if col == 0 else 'W − Wᵀ'} | tick {graph['tick']:,}",
                               xlabel="Source node j", ylabel="Receiving node i")
    fig.suptitle("Prespecified seed 0 | Ports / recorded512 / fixed / warmup", fontsize=12)
    fig.text(.02, .02, "Audited last-valid states. A failed branch may end at an earlier budget; these matrices describe physical coupling, not equal-endpoint performance.\n"
             "||W − Wᵀ||F equals twice the skew-part norm. No checkpoint was loaded by the renderer.", fontsize=8)
    fig.tight_layout(rect=(0, .08, 1, .95))
    _save(fig, folder, "audited_last_valid_direction_weights", bundle["inputs"], outputs)


def _credit_check(path, inputs):
    path = Path(path).resolve()
    value = _json(path)
    if (value.get("format") != "undirected-credit-path-check-v1" or value.get("total_ticks") != 1152
            or value.get("cue_tick") != 0 or value.get("supervised_suffix") != [1024, 1152]):
        raise PlotDataError("the mechanical illustration requires the source-bound 1152-tick cue check")
    inputs[str(path)] = dict(sha256=file_sha256(path), role="mechanical_credit_check_no_behavior")
    for source, digest in value["sources"].items():
        _verified_entry(dict(path=source, sha256=digest), path.parent, inputs, "mechanical_credit_check_source")
    return value


def credit_check_figure(path, folder, inputs, outputs):
    plt = _graphics()
    value = _credit_check(path, inputs)
    methods = ("tbptt128", "recorded512", "recorded1024", "rtrl", "rtrl_reset")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": (1.5, 1)})
    for index, method in enumerate(methods):
        ticks = value["credit_path_ticks"][method]
        magnitude = value["encoder_gradient_norm"][method]
        if not _number(magnitude) or magnitude < 0:
            raise PlotDataError("invalid mechanical gradient norm")
        axes[0].plot([1152 - ticks, 1152], [index, index], color=COLORS[index], linewidth=5)
        axes[1].barh(index, magnitude, color=COLORS[index])
        axes[1].text(magnitude, index, f" {magnitude:.4g}", va="center", fontsize=8)
    axes[0].axvspan(1024, 1152, color="#bbc4cc", alpha=.35)
    axes[0].axvline(0, color="#27323d", linestyle=":", linewidth=1)
    axes[0].set(yticks=range(5), yticklabels=methods, xlabel="Tick along recorded changing-parameter path", xlim=(-40, 1192))
    axes[0].set_title("Cue only at tick 0; supervision [1024,1152)")
    axes[1].set(yticks=range(5), yticklabels=methods, xlabel="Cue-encoder gradient L2")
    axes[1].set_title("Mechanical derivative, not a learning score")
    for axis in axes:
        axis.invert_yaxis()
    fig.suptitle("1152-tick numerical credit validation | eight fixed parameter changes", fontsize=12)
    fig.text(.02, .02, f"Full recorded BPTT vs RTRL max abs error: {value['rtrl_full_bptt_max_abs_error']:.3g}; primal error: {value['primal_max_abs_error']:.3g}.\n"
             "This manually arranged slow-mode example checks derivatives and temporal reach. It does not demonstrate learned task performance.", fontsize=8)
    fig.tight_layout(rect=(0, .10, 1, .92))
    _save(fig, folder, "mechanical_credit_1152_not_behavior", inputs, outputs)


def render(report, output, *, ticks=None, credit_check=DEFAULT_CREDIT_CHECK, layout_seed=731):
    bundle = read_audited_bundle(report)
    # Bind all inputs before drawing so each figure and the final manifest
    # share a reproducible data SHA, including the separate mechanical panel.
    _credit_check(credit_check, bundle["inputs"])
    folder = Path(output).resolve()
    if folder.exists():
        raise FileExistsError("figures require a new output directory")
    selected = [bundle["header"]["primary_tick"], max(bundle["header"]["checkpoint_ticks"])] if ticks is None else list(ticks)
    selected = sorted(set(selected))
    if any(tick not in bundle["header"]["checkpoint_ticks"] or tick == 0 for tick in selected):
        raise PlotDataError("behavior figures require declared nonzero checkpoint ticks")
    folder.mkdir(parents=True)
    outputs = []
    geometry_figures(bundle, folder, outputs, layout_seed)
    for tick in selected:
        for comparison in COMPARISONS:
            outcome_figure(bundle, comparison, tick, folder, outputs)
        for comparison in ("interfaces_512", "temporal_credit", "structure"):
            cost_figure(bundle, tick, folder, outputs, comparison)
    credit_check_figure(credit_check, folder, bundle["inputs"], outputs)
    # Every input is checked again after rendering. No raw path in the audited
    # input inventory is opened, and no missing seed is silently dropped.
    for path, entry in bundle["inputs"].items():
        if file_sha256(path) != entry["sha256"]:
            raise PlotDataError(f"input changed during plotting: {path}")
    import matplotlib
    import numpy
    manifest = dict(format="undirected-figures-v1", protocol_sha256=bundle["header"]["protocol_sha256"],
        accounting_audit="passed", accounting_audit_scope=bundle["header"]["audit"]["scope"],
        scientific_status=bundle["header"]["scientific_status"],
        fixed_probe_comparison=bundle["header"]["fixed_probe_comparison"],
        independent_audit_counts=bundle["independent_audit_counts"],
        inputs=bundle["inputs"], data_sha256=_data_sha(bundle["inputs"]), figures=outputs,
        analysis_sources={str(Path(__file__).resolve()): file_sha256(__file__)}, matplotlib=matplotlib.__version__,
        numpy=numpy.__version__, circular_layout_seed=layout_seed, prespecified_graph_seed=0,
        statistics="Equal four-family averages within each seed; three-seed mean/sample SD only at 3/3 success and defined denominators.",
        failures="NF and U markers retain failure/undefined outcomes; no zero imputation or replacement seeds.",
        geometry="Full audited nodes/edges; fixed random circle is a display layout, not learned functional clustering.",
        credit_check="Mechanical 1152-tick derivative illustration; not behavioral learning evidence.")
    (folder / "figure-manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="verify both audits and table semantics without drawing")
    validate.add_argument("--report", required=True)
    draw = sub.add_parser("render", help="render audited figures only after both audits are ready")
    draw.add_argument("--report", required=True)
    draw.add_argument("--output", required=True)
    draw.add_argument("--tick", type=int, action="append", dest="ticks")
    draw.add_argument("--credit-check", default=str(DEFAULT_CREDIT_CHECK))
    draw.add_argument("--layout-seed", type=int, default=731)
    args = parser.parse_args(argv)
    if args.command == "validate":
        bundle = read_audited_bundle(args.report)
        print(json.dumps(dict(status="ready", branches=len(bundle["tables"]["branches"]),
                              independent_audits=bundle["independent_audit_counts"], data_sha256=_data_sha(bundle["inputs"]))))
    else:
        result = render(args.report, args.output, ticks=args.ticks, credit_check=args.credit_check, layout_seed=args.layout_seed)
        print(json.dumps(dict(figures=len(result["figures"]), data_sha256=result["data_sha256"])))


if __name__ == "__main__":
    main()
