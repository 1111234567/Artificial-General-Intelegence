"""Probe summaries must preserve populations, undefined values and seed units."""
from copy import deepcopy
import csv
from itertools import product
import json

import pytest

from experiments.credit.probe import ProbeConfig, actor_partition, fit_probe_suite
from experiments.credit.test_probe import _encoded_dataset
from experiments.credit.probe_report import (
    ProbeAuditError, aggregate_files, aggregate_rows, extract_document, file_sha256, main,
    summary_stat, write_compact_evidence, write_report,
)


@pytest.fixture(scope="module")
def fixture():
    config = ProbeConfig(actor_split=(2, 2, 2))
    partition = actor_partition(range(6), config)
    suite = fit_probe_suite({"iid": _encoded_dataset("iid")}, partition, config=config)
    suite["collection"] = dict(config=suite["config"], training_seed=0, training_arm="bp64",
        actor_partition={str(actor): split for actor, split in partition.items()},
        collector_source_sha256="fixture", checkpoint="fixture.pt", checkpoint_sha256="fixture",
        records=[dict(condition="iid", parameter_updates=0, world_resets=0, hidden_resets=0, teacher_actions=0)])
    suite = json.loads(json.dumps(suite))
    protocol = dict(config=suite["config"], seeds=[0, 1, 2], modes=["bp64"], conditions=["iid"],
        features=["raw_h", "tanh_h", "obs96"],
        targets=dict(payload=4, version=2, joint=8, active=2, raw_started=2, active_started=2),
        controls=["none", "random_joint_labels", "shuffled_features"])
    return suite, protocol


def test_seed_statistics_use_sample_sd_and_preserve_undefined_denominators():
    value = summary_stat([(0, 0.), (1, .5), (2, 1.)])
    assert value["mean"] == .5 and value["sample_sd"] == .5
    undefined = summary_stat([(0, None), (1, None), (2, None)])
    assert undefined["n"] == 0 and undefined["n_undefined"] == 3
    assert undefined["mean"] is undefined["sample_sd"] is None
    partial = summary_stat([(0, .5), (1, None), (2, 1.)])
    assert partial["n_total"] == 3 and partial["n"] == 2 and partial["mean"] == .75


def test_extracts_phase_and_public_token_marginals_without_fabricating_intersection(fixture):
    document, protocol = fixture
    case, rows = extract_document(document, protocol)
    assert case == (0, "bp64")
    assert len(rows) == 36 * 11
    assert all(row["phase"] == "all" for row in rows if row["slice"] != "all")
    assert any(row["slice"] == "no_public_payload_token" and row["n"] > 0 for row in rows)
    assert all(row["n"] == sum(row["class_support"]) for row in rows)
    assert all(row["native_accuracy"] is not None for row in rows
               if row["control"] == "none" and row["target"] == "joint" and row["n"])


def test_three_seed_complete_study_aggregates_and_emits_csv(tmp_path, fixture):
    document, protocol = fixture
    paths = []
    for seed in range(3):
        copied = deepcopy(document)
        copied["collection"]["training_seed"] = seed
        path = tmp_path / f"seed{seed}.json"
        path.write_text(json.dumps(copied), encoding="utf-8")
        paths.append(path)
    result = aggregate_files(paths, protocol)
    assert result["complete"] and result["audit"]["cases"] == 3
    assert all(row["metrics"]["accuracy"]["n_total"] == 3 for row in result["summary"])
    row = next(row for row in result["summary"] if row["target"] == "active" and row["phase"] == "cue")
    assert row["metrics"]["balanced_accuracy"]["n_undefined"] == 3
    comparisons = [row for row in result["comparisons"] if row["comparison"] == "raw_minus_tanh" and row["metric"] == "accuracy"]
    assert comparisons
    assert all(row["statistics"]["mean"] in (0, None) for row in comparisons if row["control"] == "none")
    output = tmp_path / "report"
    write_report(result, output)
    assert (output / "per_seed.csv").is_file() and (output / "summary.json").is_file()
    saved = json.loads((output / "summary.json").read_text())
    assert "per_seed" not in saved and saved["complete"]
    with pytest.raises(ProbeAuditError, match="incomplete"):
        aggregate_files(paths[:2], protocol)
    with pytest.raises(ProbeAuditError, match="duplicate"):
        aggregate_files(paths + paths[:1], protocol)


@pytest.mark.parametrize("fault,expected", (
    ("native_population", "selected population"), ("missing_record", "missing probe"),
    ("confusion", "confusion/count"), ("actor_leak", "heldout actor"),
    ("undefined", "missing-class"), ("alpha", "unregistered alpha"),
))
def test_deliberately_broken_provenance_denominators_and_selection_are_rejected(fixture, fault, expected):
    original, protocol = fixture
    document = deepcopy(original)
    record = document["records"][0]
    if fault == "native_population":
        record["native_finish_head"]["overall"]["exposure"]["request_groups"] += 1
    elif fault == "missing_record":
        document["records"].pop()
    elif fault == "confusion":
        record["test"]["overall"]["confusion"][0][0] += 1
    elif fault == "actor_leak":
        test_actor = next(actor for actor, split in document["actor_partition"].items() if split == "test")
        record["models"][0]["train"]["per_actor"][test_actor] = 1
    elif fault == "undefined":
        active = next(row for row in document["records"] if row["target"] == "active")
        active["test"]["by_phase"]["cue"]["balanced_accuracy"] = 1.
    elif fault == "alpha":
        record["models"][0]["alpha"] = 17
    with pytest.raises(ProbeAuditError, match=expected):
        extract_document(document, protocol)


def test_cross_view_population_mismatch_is_rejected_before_paired_comparison(fixture):
    document, protocol = fixture
    _, rows = extract_document(document, protocol)
    all_rows = []
    for seed in range(3):
        copied = deepcopy(rows)
        for row in copied:
            row["seed"] = seed
        all_rows.extend(copied)
    row = next(row for row in all_rows if row["feature"] == "obs96")
    row["selection_signature"] += "tampered"
    with pytest.raises(ProbeAuditError, match="population differs"):
        aggregate_rows(all_rows, protocol["seeds"])


def test_descriptive_protocol_labels_map_to_exact_frozen_result_keys(fixture):
    document, protocol = fixture
    protocol = deepcopy(protocol)
    protocol["features"] = ["raw_h", "exact_model_tanh_h", "actual_obs96"]
    protocol["controls"] = ["none", "request_consistent_random_joint_labels", "within_condition_job_split_feature_shuffle"]
    _, rows = extract_document(document, protocol)
    assert {row["feature"] for row in rows} == {"raw_h", "tanh_h", "obs96"}
    assert {row["control"] for row in rows} == {"none", "random_joint_labels", "shuffled_features"}


@pytest.fixture
def compact_source(tmp_path):
    """Synthetic saved aggregate, with all display axes and outside-view controls."""
    arms = ("frozen", "bp64", "replay64", "bp128", "coact_mix")
    scenarios = ("within/longer", "within/long_sparse", "transfer_1to8_to_8to16")
    features, targets = ("raw_h", "tanh_h", "obs96"), ("joint", "payload", "active")
    slices = (("cue", "all"), ("waiting", "all"), ("legal", "all"), ("all", "no_public_payload_token"))
    axes = ("arm", "scenario", "feature", "target", "control", "phase", "slice")
    rows, summary = [], []
    def add(arm, scenario, feature, target, control, phase, slice_name):
        key = dict(zip(axes, (arm, scenario, feature, target, control, phase, slice_name)))
        members = []
        classes = dict(joint=8, payload=4, active=2, version=2)[target]
        for seed in range(3):
            empty = (arm, scenario, target, phase, seed) == ("coact_mix", "within/longer", "joint", "legal", 2)
            n = 0 if empty else 80
            correct = 0 if empty else (70 if control != "none" and scenario == "within/iid" else 40 + seed*8)
            support = [n//classes]*classes
            single_class = target == "active" and phase in ("cue", "waiting", "legal")
            if single_class:
                support = [0, n]
            native_correct = (0 if empty else 16 + seed*8) if control == "none" and target != "active" else None
            row = key | dict(seed=seed, classification="fixture", input_dimension=96 if feature == "obs96" else 128,
                train_samples=100, validation_samples=20, n=n, correct=correct, accuracy=correct/n if n else None,
                balanced_accuracy=correct/n if n and not single_class else None, majority_accuracy=1/classes if n else None,
                class_support=support, test_actors=12 if n else 0, test_request_groups=n//2, excluded_test_samples=7,
                native_correct=native_correct, native_accuracy=native_correct/n if n and native_correct is not None else None,
                native_balanced_accuracy=native_correct/n if n and native_correct is not None else None,
                source_record=17)
            members.append(row)
        rows.extend(members)
        summary.append(key | dict(metrics={name: summary_stat([(row["seed"], row[name]) for row in members])
            for name in ("accuracy", "balanced_accuracy", "majority_accuracy", "native_accuracy", "native_balanced_accuracy", "n")}))
    for arm, scenario, feature, target, (phase, slice_name) in product(arms, scenarios, features, targets, slices):
        add(arm, scenario, feature, target, "none", phase, slice_name)
    # Control ranges must include iid even though compact primary cells exclude it.
    for control, target, scenario in product(("random_joint_labels", "shuffled_features"),
                                             ("joint", "payload", "version"), ("within/longer", "within/iid")):
        add("frozen", scenario, "tanh_h", target, control, "all", "all")
    result = dict(format="credit-probe-report-v1", complete=True, audit=dict(status="passed", error_count=0, cases=15),
        per_seed=rows, summary=summary,
        comparisons=[dict(comparison="fixture", metric="accuracy", statistics=summary_stat([(0, 0.), (1, 0.), (2, 0.)]))],
        definitions=dict(replicate="Training seed", slices="Separate marginals"),
        inputs=[dict(seed=seed, arm=arm, path=f"seed_{seed}_{arm}/probe.json", sha256="source-fixture",
            checkpoint=f"seed_{seed}_{arm}/checkpoint.pt", checkpoint_sha256="checkpoint-fixture",
            collector_source_sha256="frozen-collector-fixture") for seed, arm in product(range(3), arms)],
        protocol=dict(path="frozen-protocol.json", sha256="protocol-fixture"))
    directory = tmp_path / "full"
    write_report(result, directory)
    return directory


def test_compact_cli_retains_all_seed_values_denominators_and_full_provenance_without_mutation(compact_source, tmp_path, capsys):
    before = {path.name: path.read_bytes() for path in compact_source.iterdir()}
    output = tmp_path / "compact"
    main(["--compact-evidence", str(compact_source), "--output", str(output)])
    assert json.loads(capsys.readouterr().out)["selected_cells"] == 540
    saved = json.loads((output / "summary.json").read_text())
    assert saved["selected_view"] and saved["complete_source_audit"]
    assert saved["audit"]["full_source_cases"] == len(saved["inputs"]) == 15
    assert len(saved["summary"]) == 540 and saved["audit"]["selected_seed_rows"] == 1620
    assert {row["scenario"] for row in saved["summary"]} == {"within/longer", "within/long_sparse", "transfer_1to8_to_8to16"}
    assert all(row["control"] == "none" for row in saved["summary"])
    assert all(row["phase"] == "all" for row in saved["summary"] if row["slice"] == "no_public_payload_token")
    with (output / "per_seed.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1620 and {int(row["seed"]) for row in rows} == {0, 1, 2}
    assert all(int(row["n"]) == sum(json.loads(row["class_support"])) for row in rows)
    empty = next(row for row in saved["summary"] if row["arm"] == "coact_mix" and row["scenario"] == "within/longer"
                 and row["feature"] == "raw_h" and row["target"] == "joint" and row["phase"] == "legal")
    assert empty["metrics"]["accuracy"]["n"] == 2 and empty["metrics"]["accuracy"]["per_seed"][2]["value"] is None
    assert empty["metrics"]["n"]["per_seed"][2]["value"] == 0
    single = next(row for row in saved["summary"] if row["target"] == "active" and row["phase"] == "cue")
    assert single["metrics"]["balanced_accuracy"]["n_undefined"] == 3
    ranges = saved["negative_control_ranges"]
    assert len(ranges) == 24 and all(row["maximum_seed_mean"] == .875 for row in ranges)
    assert all(row["maximum_cell"]["scenario"] == "within/iid" for row in ranges)
    assert saved["full_aggregate"]["summary_sha256"] == file_sha256(compact_source / "summary.json")
    assert saved["full_aggregate"]["summary_cells"] > len(saved["summary"])
    assert {path.name: path.read_bytes() for path in compact_source.iterdir()} == before
    assert (output / "overview.md").is_file()


def test_compact_rejects_missing_display_cell_even_with_complete_source_audit(compact_source, tmp_path):
    path = compact_source / "summary.json"
    document = json.loads(path.read_text())
    document["summary"].pop(0)
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ProbeAuditError, match="missing compact summary cell"):
        write_compact_evidence(compact_source, tmp_path / "compact")


@pytest.mark.parametrize("fault", ("missing_seed", "changed_score"))
def test_compact_rejects_changed_selected_seed_population_or_values(compact_source, tmp_path, fault):
    path = compact_source / "per_seed.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    if fault == "missing_seed":
        rows.pop(0)
    else:
        rows[0]["correct"] = "0"
        rows[0]["accuracy"] = "0.0"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    # Bypass only the byte-integrity gate to exercise independent seed/count auditing.
    summary_path = compact_source / "summary.json"
    document = json.loads(summary_path.read_text())
    next(row for row in document["tables"] if row["path"] == "per_seed.csv")["sha256"] = file_sha256(path)
    summary_path.write_text(json.dumps(document), encoding="utf-8")
    message = "missing compact seed row" if fault == "missing_seed" else "summary/seed values differ"
    with pytest.raises(ProbeAuditError, match=message):
        write_compact_evidence(compact_source, tmp_path / "compact")


def test_compact_rejects_table_tampering_and_output_inside_full_source(compact_source, tmp_path):
    with pytest.raises(ProbeAuditError, match="would mutate"):
        write_compact_evidence(compact_source, compact_source / "compact")
    path = compact_source / "summary.csv"
    with path.open("a", encoding="utf-8") as stream:
        stream.write("tampered\n")
    with pytest.raises(ProbeAuditError, match="table hash mismatch"):
        write_compact_evidence(compact_source, tmp_path / "compact")
