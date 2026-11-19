"""Complete-study aggregation of frozen observational probe diagnostics.

Read one source JSON at a time. Never refit a probe, load a controller or infer
unrecorded intersections of phase and public-token slices. Statistical repeats
are training seeds, not actors, requests, ticks or duplicate native readouts.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.continuity.run import atomic_json
from experiments.credit.probe import CONTENT_TARGETS, PHASES, TRANSFER_TRAIN

FORMAT_VERSION = "credit-probe-report-v1"
COMPACT_FORMAT_VERSION = "credit-probe-compact-evidence-v1"
AXES = ("arm", "scenario", "feature", "target", "control", "phase", "slice")
COMPACT_ARMS = ("frozen", "bp64", "replay64", "bp128", "coact_mix")
COMPACT_SCENARIOS = ("within/longer", "within/long_sparse", "transfer_1to8_to_8to16")
COMPACT_FEATURES = ("raw_h", "tanh_h", "obs96")
COMPACT_TARGETS = ("joint", "payload", "active")
COMPACT_SLICES = (("cue", "all"), ("waiting", "all"), ("legal", "all"),
                  ("all", "no_public_payload_token"))
COMPACT_SEEDS = (0, 1, 2)
COMPACT_METRICS = ("accuracy", "balanced_accuracy", "majority_accuracy",
                   "native_accuracy", "native_balanced_accuracy", "n")
PROTOCOL_ALIASES = {
    "exact_model_tanh_h": "tanh_h", "actual_obs96": "obs96",
    "request_consistent_random_joint_labels": "random_joint_labels",
    "within_condition_job_split_feature_shuffle": "shuffled_features",
}


class ProbeAuditError(ValueError):
    pass


def _check(value, message):
    if not value:
        raise ProbeAuditError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary_stat(values):
    """None is undefined, never converted to zero or silently counted as valid."""
    values = sorted(values, key=lambda item: item[0])
    finite = [float(value) for _, value in values if value is not None]
    _check(all(math.isfinite(value) for value in finite), "nonfinite reported metric")
    return dict(n=len(finite), n_total=len(values), n_undefined=len(values) - len(finite),
                mean=float(np.mean(finite)) if finite else None,
                sample_sd=float(np.std(finite, ddof=1)) if len(finite) >= 2 else None,
                per_seed=[dict(seed=seed, value=value) for seed, value in values])


def _signature(metric):
    # Frozen probe source guarantees the same scored-row selector. These
    # independent checks guard accidental denominator/phase/case mismatches.
    return json.dumps({key: metric[key] for key in ("n", "class_support", "exposure")}, sort_keys=True, separators=(",", ":"))


def _validate_metric(metric, classes, test_actors):
    n, correct = metric["n"], metric["correct"]
    matrix = np.asarray(metric["confusion"])
    support = np.asarray(metric["class_support"])
    _check(isinstance(n, int) and 0 <= correct <= n, "invalid metric counts")
    _check(matrix.shape == (classes, classes) and support.shape == (classes,), "wrong confusion dimensions")
    _check(np.all(matrix >= 0) and np.all(matrix == np.floor(matrix)), "invalid confusion counts")
    _check(np.array_equal(matrix.sum(1), support) and int(matrix.sum()) == n and int(np.trace(matrix)) == correct,
           "confusion/count mismatch")
    if n:
        _check(metric["accuracy"] is not None and abs(metric["accuracy"] - correct / n) < 1e-12, "accuracy denominator mismatch")
    else:
        _check(metric["accuracy"] is None, "empty accuracy must be undefined")
    if np.all(support > 0):
        wanted = float((np.diag(matrix) / support).mean())
        _check(metric["balanced_accuracy"] is not None and abs(metric["balanced_accuracy"] - wanted) < 1e-12,
               "balanced accuracy mismatch")
    else:
        _check(metric["balanced_accuracy"] is None, "missing-class balanced accuracy must be undefined")
    exposure = metric["exposure"]
    _check(exposure["samples"] == n and sum(exposure["per_actor"].values()) == n, "exposure denominator mismatch")
    _check(set(map(int, exposure["per_actor"])) <= test_actors, "non-test actor entered heldout score")


def _slices(groups):
    yield "all", "all", groups["overall"]
    for phase in PHASES:
        yield phase, "all", groups["by_phase"][phase]
    yield "all", "no_public_payload_token", groups["by_public_payload_token"]["0"]
    yield "all", "public_payload_token", groups["by_public_payload_token"]["1"]


def extract_document(document, protocol, *, source="memory"):
    _check(document.get("format") == "credit-frozen-probe-v1" and document.get("kind") == "observational_ridge_probe",
           "unsupported probe result")
    collection = document["collection"]
    _check(collection["config"] == protocol["config"] and document["config"] == protocol["config"], "probe config differs from protocol")
    seed, arm = int(collection["training_seed"]), collection["training_arm"]
    _check(seed in protocol["seeds"] and arm in protocol["modes"], "unexpected training seed/arm")
    partition = {int(actor): int(split) for actor, split in collection["actor_partition"].items()}
    _check([sum(value == index for value in partition.values()) for index in range(3)] == protocol["config"]["actor_split"],
           "actor split count mismatch")
    for actor, split in partition.items():
        _check(document["actor_partition"][str(actor)] == ("train", "validation", "test")[split], "actor partition identity mismatch")
    train_actors = {actor for actor, split in partition.items() if split == 0}
    val_actors = {actor for actor, split in partition.items() if split == 1}
    test_actors = {actor for actor, split in partition.items() if split == 2}
    conditions = protocol["conditions"]
    _check(len(collection["records"]) == len(conditions) and {r["condition"] for r in collection["records"]} == set(conditions),
           "missing/duplicate collection condition")
    for record in collection["records"]:
        _check(record["parameter_updates"] == record["world_resets"] == record["hidden_resets"] == record["teacher_actions"] == 0,
               "collection changed model/control continuity")
    expected_hash = next((value for key, value in protocol.get("file_hashes", {}).items() if Path(key).name == "probe.py"), None)
    if expected_hash:
        _check(collection["collector_source_sha256"] == expected_hash, "collector source hash differs from protocol")
    scenarios = {f"within/{name}": ((name,), (name,)) for name in conditions}
    if "longer" in conditions and set(TRANSFER_TRAIN) <= set(conditions):
        scenarios["transfer_1to8_to_8to16"] = (TRANSFER_TRAIN, ("longer",))
    targets = protocol["targets"]
    if isinstance(targets, list):
        sizes = dict(payload=4, version=2, joint=8, active=2, raw_started=2, active_started=2)
        targets = {name: sizes[name] for name in targets}
    views = [PROTOCOL_ALIASES.get(value, value) for value in protocol["features"]]
    controls = [PROTOCOL_ALIASES.get(value, value) for value in protocol["controls"]]
    expected = {(scenario, view, target, control) for scenario in scenarios for view in views
                for control in controls for target in targets
                if control == "none" or target in CONTENT_TARGETS}
    seen, rows, job_identity = set(), [], None
    for record_index, record in enumerate(document["records"]):
        key = tuple(record[name] for name in ("scenario", "feature", "target", "control"))
        _check(key in expected and key not in seen, "unexpected or duplicate probe record")
        seen.add(key)
        training_conditions, testing_conditions = scenarios[record["scenario"]]
        _check(record["training_conditions"] == list(training_conditions) and record["testing_conditions"] == list(testing_conditions),
               "probe train/test domain changed")
        jobs = {model["job"] for model in record["models"]}
        if job_identity is None:
            job_identity = jobs
        _check(jobs == job_identity and len(jobs) == len(record["models"]), "job head identity mismatch")
        for model in record["models"]:
            _check(set(map(int, model["train"]["per_actor"])) <= train_actors, "training probe used heldout actor")
            _check(set(map(int, model["validation"]["per_actor"])) <= val_actors, "validation probe used wrong actor")
            if model["status"] == "fitted":
                _check(model["alpha"] in protocol["config"]["ridge_alphas"], "unregistered alpha")
        classes = targets[record["target"]]
        full = record["test"]["overall"]
        phase_groups = record["test"]["by_phase"]
        token_groups = record["test"]["by_public_payload_token"]
        _check(sum(group["n"] for group in phase_groups.values()) == full["n"], "phase counts fail to partition heldout samples")
        _check(sum(group["correct"] for group in phase_groups.values()) == full["correct"], "phase correct counts fail to partition")
        _check(sum(group["n"] for group in token_groups.values()) == full["n"]
               and sum(group["correct"] for group in token_groups.values()) == full["correct"], "public-token slices fail to partition")
        native_groups = dict(((phase, slice_name), metric) for phase, slice_name, metric in _slices(record["native_finish_head"])) if "native_finish_head" in record else {}
        _check(bool(native_groups) == (record["control"] == "none" and record["target"] in CONTENT_TARGETS), "native namespace mismatch")
        for phase, slice_name, metric in _slices(record["test"]):
            _validate_metric(metric, classes, test_actors)
            native = native_groups.get((phase, slice_name))
            if native is not None:
                _validate_metric(native, classes, test_actors)
                _check(_signature(metric) == _signature(native), "native/readout selected population differs")
            rows.append(dict(seed=seed, arm=arm, scenario=record["scenario"], feature=record["feature"],
                target=record["target"], control=record["control"], phase=phase, slice=slice_name,
                classification=record["classification"], input_dimension=record["input_dimension"],
                train_samples=sum(model["train"]["samples"] for model in record["models"]),
                validation_samples=sum(model["validation"]["samples"] for model in record["models"]),
                n=metric["n"], correct=metric["correct"], accuracy=metric["accuracy"], balanced_accuracy=metric["balanced_accuracy"],
                majority_accuracy=metric["train_majority_accuracy"], class_support=metric["class_support"],
                test_actors=metric["exposure"]["actors"], test_request_groups=metric["exposure"]["request_groups"],
                excluded_test_samples=record["excluded_test_samples"],
                native_accuracy=native["accuracy"] if native is not None else None,
                native_balanced_accuracy=native["balanced_accuracy"] if native is not None else None,
                native_correct=native["correct"] if native is not None else None,
                native_confusion=native["confusion"] if native is not None else None,
                selection_signature=_signature(metric), source=source, source_record=record_index))
    _check(seen == expected, "missing probe records")
    return (seed, arm), rows


def _difference(left, right):
    return None if left is None or right is None else left - right


def aggregate_rows(rows, seeds):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in AXES)].append(row)
    summary = []
    for key, values in sorted(groups.items()):
        _check({row["seed"] for row in values} == set(seeds) and len(values) == len(seeds), "missing/duplicate per-seed slice")
        summary.append(dict(zip(AXES, key)) | dict(metrics={
            name: summary_stat([(row["seed"], row[name]) for row in values])
            for name in ("accuracy", "balanced_accuracy", "majority_accuracy", "native_accuracy", "native_balanced_accuracy", "n")
        }))
    index = {(row["seed"],) + tuple(row[key] for key in AXES): row for row in rows}
    differences = defaultdict(list)
    native_canonical = {}
    for row in rows:
        base = {key: row[key] for key in AXES}
        def add(name, other=None, right_native=False):
            if other is not None:
                _check(row["selection_signature"] == other["selection_signature"], f"comparison population differs: {name}")
            identity = tuple(base[key] for key in AXES) + (name,)
            for metric in ("accuracy", "balanced_accuracy"):
                right = row[f"native_{metric}"] if right_native else other[metric]
                differences[(identity, metric)].append((row["seed"], _difference(row[metric], right)))
        if row["control"] == "none" and row["target"] in CONTENT_TARGETS:
            native_key = (row["seed"], row["arm"], row["scenario"], row["target"], row["phase"], row["slice"])
            native_value = row["selection_signature"], row["native_confusion"]
            _check(native_key not in native_canonical or native_canonical[native_key] == native_value,
                   "native readout changed across feature views")
            native_canonical[native_key] = native_value
            add("probe_minus_native", right_native=True)
        for wanted, name in (("tanh_h", "raw_minus_tanh"), ("obs96", "raw_minus_obs")):
            if row["feature"] == "raw_h":
                other_key = (row["seed"],) + tuple(wanted if key == "feature" else row[key] for key in AXES)
                add(name, index[other_key])
        if row["feature"] == "tanh_h":
            other_key = (row["seed"],) + tuple("obs96" if key == "feature" else row[key] for key in AXES)
            add("tanh_minus_obs", index[other_key])
        if row["scenario"] == "within/longer":
            other_key = (row["seed"],) + tuple("transfer_1to8_to_8to16" if key == "scenario" else row[key] for key in AXES)
            if other_key in index:
                add("within_longer_minus_transfer", index[other_key])
    comparisons = []
    for (identity, metric), values in sorted(differences.items()):
        _check({seed for seed, _ in values} == set(seeds), "comparison missing seed")
        comparisons.append(dict(zip(AXES + ("comparison",), identity)) | dict(metric=metric, statistics=summary_stat(values)))
    return summary, comparisons


def aggregate_files(paths, protocol):
    expected = {(int(seed), mode) for seed in protocol["seeds"] for mode in protocol["modes"]}
    rows, inputs, cases = [], [], set()
    canonical_partition = None
    jobs = {job["id"]: job for job in protocol.get("jobs", [])}
    for path in paths:
        path = Path(path).resolve()
        document = json.loads(path.read_text(encoding="utf-8"))
        case, extracted = extract_document(document, protocol, source=str(path))
        _check(case not in cases, "duplicate checkpoint probe input")
        partition = document["collection"]["actor_partition"]
        if canonical_partition is None:
            canonical_partition = partition
        _check(partition == canonical_partition, "actor partition changed across source checkpoints")
        job = jobs.get(f"seed_{case[0]:03d}_{case[1]}")
        if job is not None:
            command = job["command"]
            expected_checkpoint = Path(command[command.index("--checkpoint") + 1]).resolve()
            actual_checkpoint = Path(document["collection"]["checkpoint"]).resolve()
            _check(actual_checkpoint == expected_checkpoint, "source checkpoint differs from protocol job")
            expected_checkpoint_hash = protocol.get("file_hashes", {}).get(str(expected_checkpoint))
            if expected_checkpoint_hash:
                _check(document["collection"]["checkpoint_sha256"] == expected_checkpoint_hash, "checkpoint hash differs from protocol")
        cases.add(case)
        rows.extend(extracted)
        inputs.append(dict(path=str(path), sha256=file_sha256(path), seed=case[0], arm=case[1],
                           checkpoint=document["collection"]["checkpoint"],
                           checkpoint_sha256=document["collection"]["checkpoint_sha256"],
                           collector_source_sha256=document["collection"]["collector_source_sha256"]))
        del document
    _check(cases == expected, f"incomplete probe study; missing {sorted(expected - cases)}, unexpected {sorted(cases - expected)}")
    summary, comparisons = aggregate_rows(rows, protocol["seeds"])
    return dict(format=FORMAT_VERSION, complete=True, audit=dict(status="passed", error_count=0, cases=len(cases)),
                inputs=inputs, per_seed=rows, summary=summary, comparisons=comparisons,
                definitions={
                    "replicate": "One trained seed. Actor/request/tick counts are exposure, not independent training repeats.",
                    "statistics": "Unweighted mean and sample SD across seeds; undefined values remain None with explicit n_undefined.",
                    "selection": "Native/state/observation comparisons require identical recorded heldout count, class support and per-actor/request exposure; frozen probe source uses the same scored-row selector.",
                    "slices": "Phase marginals and all-phase public-payload-token marginals are separate. Their joint intersections are not in the frozen source output and are not fabricated here.",
                    "native": "Existing conditional eight-FINISH argmax, with payload/version derived from that choice. Repeated native scores across feature views are not independent evidence.",
                    "meaning": "Independent supervised readout accessibility is not autonomous task control, information-theoretic retention, or a pure encoding/decay attribution.",
                    "protocol_aliases": PROTOCOL_ALIASES,
                })


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def write_report(result, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    per_seed = [{key: value for key, value in row.items() if key not in ("selection_signature", "native_confusion")} for row in result["per_seed"]]
    _write_csv(output / "per_seed.csv", per_seed)
    _write_csv(output / "summary.csv", [dict((key, row[key]) for key in AXES) | dict(metric=metric, **stats)
               for row in result["summary"] for metric, stats in row["metrics"].items()])
    _write_csv(output / "comparisons.csv", [dict((key, value) for key, value in row.items() if key != "statistics") | row["statistics"] for row in result["comparisons"]])
    compact = {key: value for key, value in result.items() if key != "per_seed"}
    compact["tables"] = [dict(path=name, sha256=file_sha256(output / name)) for name in ("per_seed.csv", "summary.csv", "comparisons.csv")]
    atomic_json(output / "summary.json", compact)
    atomic_json(output / "audit.json", result["audit"])


def _compact_keys():
    return {(arm, scenario, feature, target, "none", phase, slice_name)
            for arm, scenario, feature, target, (phase, slice_name) in product(
                COMPACT_ARMS, COMPACT_SCENARIOS, COMPACT_FEATURES, COMPACT_TARGETS, COMPACT_SLICES)}


def _row_key(row):
    return tuple(row[name] for name in AXES)


def _directory_manifest(directory):
    files = [dict(path=path.relative_to(directory).as_posix(), bytes=path.stat().st_size,
                  sha256=file_sha256(path))
             for path in sorted(directory.rglob("*")) if path.is_file()]
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return dict(files=files, total_bytes=sum(row["bytes"] for row in files),
                manifest_sha256=hashlib.sha256(encoded).hexdigest())


def _control_ranges(summary):
    """Show the entire aggregate's controls, including domains outside the display."""
    controls = sorted({row["control"] for row in summary if row["control"] != "none"})
    _check(set(controls) == {"random_joint_labels", "shuffled_features"}, "missing negative-control kind")
    result = []
    for scope in ("overall_all_domains", "all_recorded_marginals"):
        for control, target, metric in product(controls, ("joint", "payload", "version"),
                                                ("accuracy", "balanced_accuracy")):
            rows = [row for row in summary if row["control"] == control and row["target"] == target
                    and (scope == "all_recorded_marginals" or (row["phase"], row["slice"]) == ("all", "all"))]
            _check(rows, "missing negative-control target/scope")
            finite = [row for row in rows if row["metrics"][metric]["mean"] is not None]
            low = min(finite, key=lambda row: row["metrics"][metric]["mean"]) if finite else None
            high = max(finite, key=lambda row: row["metrics"][metric]["mean"]) if finite else None
            by_seed = []
            for seed in COMPACT_SEEDS:
                values = [next(item["value"] for item in row["metrics"][metric]["per_seed"] if item["seed"] == seed)
                          for row in rows]
                defined = [value for value in values if value is not None]
                _check(all(math.isfinite(value) for value in defined), "nonfinite negative-control score")
                by_seed.append(dict(seed=seed, defined_cells=len(defined), undefined_cells=len(values) - len(defined),
                                    minimum=min(defined) if defined else None, maximum=max(defined) if defined else None))
            result.append(dict(scope=scope, control=control, target=target, metric=metric,
                summary_cells=len(rows), defined_mean_cells=len(finite), undefined_mean_cells=len(rows) - len(finite),
                minimum_seed_mean=low["metrics"][metric]["mean"] if low else None,
                maximum_seed_mean=high["metrics"][metric]["mean"] if high else None,
                minimum_cell={name: low[name] for name in AXES} if low else None,
                maximum_cell={name: high[name] for name in AXES} if high else None,
                per_seed=by_seed))
    return result


def _selected_seed_rows(path, selected):
    integer_fields = ("seed", "input_dimension", "train_samples", "validation_samples", "n", "correct",
                      "test_actors", "test_request_groups", "excluded_test_samples", "native_correct", "source_record")
    float_fields = tuple(name for name in COMPACT_METRICS if name != "n")
    keep = ("seed",) + AXES + ("classification", "input_dimension", "train_samples", "validation_samples",
        "n", "correct", "accuracy", "balanced_accuracy", "majority_accuracy", "class_support",
        "test_actors", "test_request_groups", "excluded_test_samples", "native_accuracy",
        "native_balanced_accuracy", "native_correct", "source_record")
    rows, seen = [], set()
    with path.open(newline="", encoding="utf-8") as stream:
        for source in csv.DictReader(stream):
            if _row_key(source) not in selected:
                continue
            row = {name: source[name] for name in keep}
            for name in integer_fields:
                row[name] = int(row[name]) if row[name] else None
            for name in float_fields:
                row[name] = float(row[name]) if row[name] else None
            row["class_support"] = json.loads(row["class_support"])
            identity = row["seed"], _row_key(row)
            _check(identity not in seen and row["seed"] in COMPACT_SEEDS, "duplicate/unexpected compact seed row")
            seen.add(identity)
            _check(row["n"] >= 0 and sum(row["class_support"]) == row["n"], "compact class-support denominator mismatch")
            _check(0 <= row["correct"] <= row["n"], "compact correct count mismatch")
            _check(row["accuracy"] == (row["correct"] / row["n"] if row["n"] else None), "compact accuracy denominator mismatch")
            if row["native_correct"] is not None:
                _check(row["native_accuracy"] == (row["native_correct"] / row["n"] if row["n"] else None),
                       "compact native denominator mismatch")
            rows.append(row)
    _check(seen == {(seed, key) for seed, key in product(COMPACT_SEEDS, selected)}, "missing compact seed row")
    index = {(row["seed"], _row_key(row)): row for row in rows}
    for key, summary in selected.items():
        for metric in COMPACT_METRICS:
            wanted = summary_stat([(seed, index[seed, key][metric]) for seed in COMPACT_SEEDS])
            _check(summary["metrics"][metric] == wanted, "compact summary/seed values differ")
    return sorted(rows, key=lambda row: (_row_key(row), row["seed"]))


def write_compact_evidence(full_aggregate, output):
    """Package fixed display axes without changing or refitting any full result.

    This is an explicitly selected repository view of an already completed
    aggregate, not an analysis preregistration or a replacement for its cells.
    The complete source audit is reused, with original file and input hashes.
    """
    source = Path(full_aggregate).resolve()
    directory = source if source.is_dir() else source.parent
    summary_path = directory / "summary.json"
    _check(source.is_dir() or source == summary_path, "compact source must be an aggregate directory or summary.json")
    output = Path(output).resolve()
    _check(not output.is_relative_to(directory), "compact output would mutate the full aggregate")
    if output.exists():
        raise FileExistsError(output)
    manifest = _directory_manifest(directory)
    files = {row["path"]: row for row in manifest["files"]}
    full = json.loads(summary_path.read_text(encoding="utf-8"))
    _check(full.get("format") == FORMAT_VERSION and full.get("complete") is True, "incomplete/unsupported full aggregate")
    _check(full["audit"] == dict(status="passed", error_count=0, cases=15), "full source audit is not complete for 15 cases")
    _check(json.loads((directory / "audit.json").read_text(encoding="utf-8")) == full["audit"], "full audit file mismatch")
    cases = {(row["seed"], row["arm"]) for row in full["inputs"]}
    _check(cases == set(product(COMPACT_SEEDS, COMPACT_ARMS)) and len(full["inputs"]) == 15,
           "compact input provenance lacks all 15 sources")
    for table in full["tables"]:
        _check(table["path"] in files and files[table["path"]]["sha256"] == table["sha256"], "full aggregate table hash mismatch")
    expected = _compact_keys()
    selected = {}
    for row in full["summary"]:
        key = _row_key(row)
        if key in expected:
            _check(key not in selected, "duplicate compact summary cell")
            selected[key] = row
    _check(set(selected) == expected, "missing compact summary cell")
    seed_rows = _selected_seed_rows(directory / "per_seed.csv", selected)
    ranges = _control_ranges(full["summary"])
    selection = dict(
        basis="Display axes requested after study completion; values do not affect inclusion. This is not a preregistration.",
        arms=list(COMPACT_ARMS), scenarios=list(COMPACT_SCENARIOS), features=list(COMPACT_FEATURES),
        targets=list(COMPACT_TARGETS), control="none", slices=[dict(phase=phase, slice=slice_name) for phase, slice_name in COMPACT_SLICES],
        seeds=list(COMPACT_SEEDS), metrics=list(COMPACT_METRICS),
        excluded="Other scenarios/targets/phases, individual control cells, and all paired-comparison tables remain in the full aggregate.")
    audit = dict(status="passed", error_count=0, full_source_cases=15, source_audit_reused=True,
                 full_table_hashes_verified=True, selected_cells=len(selected), selected_seed_rows=len(seed_rows),
                 selected_seed_metrics_match_full_summary=True)
    result = dict(format=COMPACT_FORMAT_VERSION, selected_view=True, complete_source_audit=True,
        selection=selection, audit=audit, protocol=full.get("protocol"), inputs=full["inputs"],
        full_aggregate=dict(directory=str(directory), summary=str(summary_path),
            summary_sha256=files["summary.json"]["sha256"], audit=full["audit"],
            summary_cells=len(full["summary"]), paired_comparison_cells=len(full["comparisons"]), **manifest),
        packager=dict(path=str(Path(__file__).resolve()), sha256=file_sha256(__file__)),
        summary=[selected[key] for key in sorted(selected)], negative_control_ranges=ranges,
        definitions=full["definitions"] | dict(
            selected_view="Exactly the listed 540 summary cells / 1620 seed rows. This file does not contain all full-aggregate cells.",
            control_ranges="Global ranges use every full-aggregate domain and feature, separately for overall cells and all recorded marginals. The latter includes small/empty phase populations and is not a chance-level encoding test.",
            seed_csv="Per-seed n, class support, accuracy, balanced accuracy, native counts/scores, exposures, and exclusions for selected cells. Source-record indices refer to the matching seed/arm input listed above.",
            completeness="The complete frozen 15-source audit is preserved, not re-run on raw probe data. Packaging additionally verifies full table hashes, selection completeness, and per-seed agreement."))
    output.mkdir(parents=True)
    _write_csv(output / "per_seed.csv", seed_rows)
    _write_csv(output / "negative_control_ranges.csv", ranges)
    result["tables"] = [dict(path=name, sha256=file_sha256(output / name))
                        for name in ("per_seed.csv", "negative_control_ranges.csv")]
    atomic_json(output / "summary.json", result)
    atomic_json(output / "audit.json", audit)
    (output / "overview.md").write_text(
        "# Selected probe evidence\n\n"
        f"This is a selected display view: {len(selected)} summary cells and {len(seed_rows)} seed rows. "
        "All three seed values and undefined denominators are retained. Selection uses named axes, never scores; "
        "it was requested after study completion and is not a preregistration.\n\n"
        "The complete 15-source audit passed. Global negative-control ranges include domains omitted from the display. "
        "Other cells and all paired-comparison tables remain in the full aggregate.\n\n"
        f"Full aggregate: `{directory}`\n\nFull summary SHA256: `{files['summary.json']['sha256']}`\n\n"
        f"Full file-manifest SHA256: `{manifest['manifest_sha256']}`\n\n"
        "Reproduce with `python -m experiments.credit.probe_report --compact-evidence FULL_DIRECTORY --output NEW_DIRECTORY`.\n",
        encoding="utf-8")
    _check(_directory_manifest(directory) == manifest, "full aggregate changed during packaging")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--probe", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compact-evidence", type=Path,
                        help="Package a fixed selected view from a complete aggregate directory or its summary.json; no refitting")
    args = parser.parse_args(argv)
    if args.compact_evidence:
        if args.protocol or args.root or args.probe:
            parser.error("--compact-evidence uses an existing aggregate; omit --protocol, --root, and --probe")
        result = write_compact_evidence(args.compact_evidence, args.output)
        print(json.dumps(dict(selected_view=True, cases=result["audit"]["full_source_cases"],
            selected_cells=len(result["summary"]), summary=str((args.output / "summary.json").resolve()))))
        return
    if args.protocol is None:
        parser.error("--protocol required when aggregating probe source files")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    paths = args.probe or sorted(args.root.rglob("probe.json")) if args.root else args.probe
    if not paths:
        parser.error("--root or --probe with complete result files required")
    result = aggregate_files(paths, protocol)
    result["protocol"] = dict(path=str(args.protocol.resolve()), sha256=file_sha256(args.protocol))
    write_report(result, args.output)
    print(json.dumps(dict(complete=True, cases=len(result["inputs"]), summary=str((args.output / "summary.json").resolve()))))


if __name__ == "__main__":
    main()
