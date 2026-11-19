"""Synthetic complete-dataset and corruption tests; no historical run/training."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import statistics

import pytest
import torch

from experiments.loops.interventions import build_interventions
from experiments.loops.report import (
    AuditError, CONDITIONS, INTERVENTIONS, MASK_SEEDS, MODES, SEEDS,
    build_report, file_sha256, main, write_artifacts,
)
from experiments.loops.structure import analyze_states, analyze_weight


def digest(*values):
    return hashlib.sha256(repr(values).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def spread(total, batch=16):
    return [total // batch + int(index < total % batch) for index in range(batch)]


def cohort(n, completed, clean, fingerprint):
    return dict(matured=n, completed=completed, clean_completed=clean,
                completion_rate=completed/n if n else None,
                clean_completion_rate=clean/n if n else None,
                request_fingerprint=fingerprint, per_actor={
                    "matured": spread(n), "completed": spread(completed), "clean_completed": spread(clean)})


def details(n, completed, clean):
    errors = completed-clean
    return dict(requests=n, completed=completed, clean_completed=clean,
        with_recorded_or_inherited_error=errors,
        first_error_patterns={"wrong_payload": errors} if errors else {},
        first_error_causes={"wrong_payload": errors} if errors else {},
        uncompleted_no_begin=n-completed, uncompleted_began_but_no_legal_opportunity=0,
        uncompleted_despite_legal_opportunity=0, first_observed_legal_count=completed,
        first_observed_legal_content_correct=completed, cue_observed_count=0,
        cue_content_correct_count=0, cue_legal_pair_count=0, cue_legal_both_correct=0,
        cue_legal_lost=0, cue_legal_gained=0, cue_legal_both_wrong=0, cue_legal_cue_correct=0,
        cue_legal_legal_correct=0, cue_without_legal_count=0, legal_without_cue_count=completed,
        neither_cue_nor_legal_count=n-completed, cue_content_accuracy=None,
        cue_legal_cue_accuracy=None, cue_legal_legal_accuracy=None,
        cue_legal_accuracy_given_cue_correct=None)


def record(source, mask, condition, name, n=160):
    identifier, seed = source["id"], source["seed"]
    effect = {"none": 0, "diagonal_only": -20, "dag_cut": -10, "random_cut": -6,
              "dag_norm_matched": -8, "random_norm_matched": -4, "dag_quarter": -2,
              "random_quarter": -1}[name]
    clean = 80 + seed*8 + MASK_SEEDS.index(mask)*4 + effect*(seed+1) if n else 0
    completed = clean+8 if n else 0
    total, active = 1024*16*2, 1000
    idle, actions = total-active, 20+seed*4+MASK_SEEDS.index(mask)+INTERVENTIONS.index(name)
    action_hash, observation_hash = digest(identifier, mask, condition, name, "actions"), digest(identifier, mask, condition, name, "observations")
    parameter_hash = digest(identifier, mask, name, "parameters")
    duration = {"iid": [1., 4.], "long_sparse": [4., 8.], "longer": [8., 16.]}[condition]
    row = dict(format="integration-own-trace-v2", condition=condition, intervention="none", action_policy="model",
        graph_intervention=name, mask_seed=mask, graph_parameter_hash=parameter_hash,
        graph_action_policy="unmodified_model" if name == "none" else "parameter_intervention_counterfactual",
        world_resets=0, hidden_resets=0, parameter_updates=0, steps=1024, dt=.25,
        batch_size=16, n_jobs=2, active_jobs=2, actor_ids=list(range(16)),
        case_ids=[f"{identifier}/{index}" for index in range(16)], fork_tick=100, fork_time=25.,
        end_tick=1124, end_time=281., evaluation_regime_id=3,
        regime=dict(duration_range=duration, gap_range=[4., 12.] if condition == "long_sparse" else [.5, 2.],
                    window_range=[1.5, 2.5], cancel=.2, replace=.3),
        primary_eligible_issued=n+4, carryover_contract_eligible_issued=0, initial_eligible_pending=0,
        censored={"primary": dict(count=4, completed=1, clean_completed=1),
                  "inherited": dict(count=0, completed=0, clean_completed=0),
                  "carryover_contract": dict(count=0, completed=0, clean_completed=0)},
        request_diagnosis={"primary": details(n, completed, clean), "inherited": details(0, 0, 0),
                           "carryover_contract": details(0, 0, 0)},
        content_by_cohort_phase_age=[dict(cohort="primary", phase="legal_finish", opportunities=active,
            finish_choice_correct=active, conditional_finish_accuracy=1., first_legal_opportunities=completed,
            first_legal_choice_correct=completed, first_legal_conditional_accuracy=1. if completed else None)],
        opportunities=dict(opportunities=total, active_opportunities=active, inactive_opportunities=idle,
            inactive_actions=actions, inactive_begin_actions=5, inactive_finish_actions=actions-5,
            disabled_slot_opportunities=0, disabled_slot_actions=0, model_action_changes=0,
            observation_ablation_opportunities=0, active_finish_attempts=100,
            legally_timed_finish_attempts=80, illegally_timed_finish_attempts=20),
        inactive_actions_per_idle_opportunity=actions/idle,
        inactive_actions_per_1000_idle_opportunities=1000*actions/idle,
        inactive_actions_per_1000_total_opportunities=1000*actions/total,
        inactive_history=[dict(cause="completion", time_since_terminal_bin="[4,8)", opportunities=idle,
            actions=actions, action_rate_per_idle_opportunity=actions/idle)],
        post_fork_counters=dict(false_action=actions, completed_all=completed, finish_attempted=1000),
        post_fork_finish_precision=completed/1000,
        start_world_state_hash=digest(identifier, "world"), end_world_state_hash=digest(identifier, mask, condition, name, "end_world"),
        start_hidden_hash=digest(identifier, "hidden"), end_hidden_hash=digest(identifier, mask, condition, name, "end_hidden"),
        start_scheduler_hash=digest(identifier, "scheduler"), end_scheduler_hash=digest(identifier, condition, "end_scheduler"),
        start_event_hash=digest(identifier, "events"), end_event_hash=digest(identifier, condition, "end_events"),
        parameter_hash=parameter_hash, source_continuation_hash=digest(identifier, mask, name, "continuation"),
        first_observation_hash=digest(identifier, condition, "first_observation"),
        first_model_observation_hash=digest(identifier, condition, "first_observation"),
        action_hash=action_hash, model_action_hash=action_hash,
        observation_hash=observation_hash, raw_observation_hash=observation_hash, model=dict(kind="learned", width=4))
    for name, count in (("primary", n), ("inherited", 0), ("carryover_contract", 0)):
        row[name] = cohort(count, completed if count else 0, clean if count else 0, digest(identifier, condition, name))
    return row


@lru_cache(maxsize=1)
def synthetic_graphs():
    weight = torch.arange(1, 17, dtype=torch.float64).reshape(4, 4) / 20
    metadata = {seed: build_interventions(weight, seed)[1] for seed in MASK_SEEDS}
    structure = dict(weight=analyze_weight(weight),
                     state=analyze_states(weight, torch.zeros(16, 4, dtype=torch.float64)),
                     sampling="Synthetic stored-state samples; no trajectory or historical checkpoint was run.")
    return metadata, structure


@pytest.fixture
def dataset(tmp_path):
    raw = tmp_path / "raw"
    source_file = tmp_path / "frozen_source.py"
    source_file.write_text("# synthetic frozen source\n", encoding="utf-8")
    sources = []
    for seed in SEEDS:
        for mode in MODES:
            identifier = f"seed_{seed:03d}_{mode}"
            path = tmp_path / "checkpoints" / identifier / "checkpoint.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"synthetic bytes only: {identifier}".encode())
            sources.append(dict(id=identifier, seed=seed, mode=mode, path=str(path), sha256=file_sha256(path)))
    plan = dict(format="loops-existing-audit-protocol-v1", checkpoints=sources,
        sources={str(source_file): file_sha256(source_file)}, mask_seeds=list(MASK_SEEDS),
        conditions=list(CONDITIONS), steps=1024, dt=.25, state_samples=16, device="cpu", parameter_updates=0,
        interpretation={"statistics": "Masks within seed, then three trained seeds."})
    protocol = tmp_path / "protocol.json"
    save(protocol, plan)
    protocol_hash = file_sha256(protocol)
    metadata, structure = synthetic_graphs()
    for source in sources:
        for mask in MASK_SEEDS:
            names = INTERVENTIONS if mask == MASK_SEEDS[0] else INTERVENTIONS[2:]
            item = dict(format="loops-existing-audit-v1", protocol=str(protocol), protocol_sha256=protocol_hash,
                source=source, mask_seed=mask, source_semantic_hash_before=digest(source["id"], "semantic"),
                source_semantic_hash_after=digest(source["id"], "semantic"), checkpoint_bytes_unchanged=True,
                source_files=plan["sources"], intervention_metadata=metadata[mask],
                structure=structure if mask == MASK_SEEDS[0] else None,
                records=[record(source, mask, condition, name) for condition in CONDITIONS for name in names])
            save(raw / f"{source['id']}_mask_{mask}.json", item)
    return protocol, raw, tmp_path / "output"


def mutate(dataset, change, *, filename="seed_000_A_mask_11.json"):
    path = dataset[1] / filename
    item = json.loads(path.read_text(encoding="utf-8"))
    change(item)
    save(path, item)


def select(report, name, *, key="tables"):
    field = "intervention" if key == "tables" else "contrast"
    return next(row for row in report[key] if row["mode"] == "A" and row["condition"] == "iid" and row[field] == name)


def test_complete_nested_seed_statistics_provenance_and_artifacts(dataset):
    protocol, raw, output = dataset
    before = {str(path): file_sha256(path) for path in raw.glob("*.json")}
    report = build_report(protocol, raw)
    assert report["complete"] is True and report["audit"]["status"] == "passed"
    assert report["audit"]["checks"] > 30000 and report["audit"]["errors"] == []
    assert report["counts"] == dict(sources=9, jobs=27, forks=540, structural_summaries=9)
    assert len(report["tables"]) == 72 and len(report["paired_differences"]) == 81
    result = select(report, "random_norm_matched")["metrics"]["strict_clean_completion"]
    assert result["n"] == result["n_total"] == 3
    assert result["mean"] == pytest.approx(84/160)
    assert result["sample_sd"] == pytest.approx(4/160)
    assert result["sample_sd"] == pytest.approx(statistics.stdev(row["value"] for row in result["per_seed"]))
    assert all(row["mask_count"] == 3 and row["mask_seeds"] == list(MASK_SEEDS) for row in result["per_seed"])
    baseline = select(report, "none")["metrics"]["strict_clean_completion"]
    assert all(row["mask_count"] == 1 for row in baseline["per_seed"])
    paired = select(report, "dag_norm_matched-minus-random_norm_matched", key="paired_differences")["metrics"]["strict_clean_completion"]
    assert paired["mean"] == pytest.approx(-8/160) and paired["sample_sd"] == pytest.approx(4/160)
    delta = select(report, "dag_norm_matched-minus-none", key="paired_differences")["metrics"]["strict_clean_completion"]
    assert delta["mean"] == pytest.approx(-12/160) and delta["sample_sd"] == pytest.approx(8/160)
    assert {row["path"]: row["sha256"] for row in report["raw_results"]} == before
    assert len(report["analysis_source_files"]) == 2
    assert all(file_sha256(path) == value for path, value in report["analysis_source_files"].items())
    assert len(report["structural_summaries"]) == 9
    assert all("samples" not in row["state"] for row in report["structural_summaries"])
    write_artifacts(report, output)
    assert {path.name for path in output.iterdir()} == {"report.json", "report.md", "table.csv", "paired_differences.csv"}
    assert json.loads((output / "report.json").read_text(encoding="utf-8")) == report
    assert "file://" not in (output / "report.md").read_text(encoding="utf-8")
    assert before == {str(path): file_sha256(path) for path in raw.glob("*.json")}


def test_missing_and_extra_job_are_rejected(dataset):
    path = dataset[1] / "seed_000_A_mask_11.json"
    value = path.read_bytes()
    path.unlink()
    with pytest.raises(AuditError, match="JOB_COUNT"):
        build_report(*dataset[:2])
    path.write_bytes(value)
    (dataset[1] / "duplicate.json").write_bytes(value)
    with pytest.raises(AuditError, match="JOB_COUNT"):
        build_report(*dataset[:2])


def test_duplicate_job_even_identical_bytes_is_rejected(dataset):
    source = dataset[1] / "seed_000_A_mask_11.json"
    target = dataset[1] / "seed_000_A_mask_29.json"
    target.write_bytes(source.read_bytes())
    with pytest.raises(AuditError, match="DUPLICATE_JOB"):
        build_report(*dataset[:2])


def test_root_execution_status_is_validated_and_does_not_count_as_a_result(dataset):
    protocol, raw, _ = dataset
    plan = json.loads(protocol.read_text(encoding="utf-8"))
    manifest_path = protocol.parent / "audit-commands-v1.json"
    jobs = [{"id": path.stem, "command": ["synthetic", path.stem], "outputs": [str(path)]}
            for path in sorted(raw.glob("*.json"))]
    manifest = dict(file_hashes={**plan["sources"], str(protocol): file_sha256(protocol)}, jobs=jobs)
    save(manifest_path, manifest)
    canonical = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    status = dict(plan=str(manifest_path), plan_digest=canonical(manifest), complete=True,
                  jobs={job["id"]: dict(command_digest=canonical(job["command"]), returncode=0,
                        missing_outputs=[]) for job in jobs})
    save(raw / "execution.json", status)
    report = build_report(protocol, raw)
    assert report["counts"]["jobs"] == 27 and len(report["raw_results"]) == 27
    assert report["execution_status"]["sha256"] == file_sha256(raw / "execution.json")
    (raw / "unknown-result.json").write_bytes((raw / "seed_000_A_mask_11.json").read_bytes())
    with pytest.raises(AuditError, match="JOB_COUNT"):
        build_report(protocol, raw)


def test_a_result_disguised_as_execution_status_is_rejected(dataset):
    raw = dataset[1]
    (raw / "execution.json").write_bytes((raw / "seed_000_A_mask_11.json").read_bytes())
    with pytest.raises(AuditError, match="EXECUTION_STATUS_SCHEMA"):
        build_report(*dataset[:2])


@pytest.mark.parametrize("change,code", [
    (lambda item: item["records"].pop(), "RECORD_COVERAGE"),
    (lambda item: item["records"].append(deepcopy(item["records"][0])), "DUPLICATE_RECORD"),
    (lambda item: item.update(protocol_sha256="0"*64), "PROTOCOL_SHA256"),
    (lambda item: item.update(protocol="elsewhere.json"), "PROTOCOL_LINK"),
    (lambda item: item["source"].update(path="elsewhere.pt"), "RAW_SOURCE_LINK"),
    (lambda item: item.update(source_semantic_hash_after=digest("changed")), "SOURCE_STATE_HASH"),
    (lambda item: item.update(checkpoint_bytes_unchanged=False), "CHECKPOINT_UNCHANGED"),
    (lambda item: item.update(source_files={}), "RAW_FROZEN_SOURCES"),
    (lambda item: item.update(structure=None), "STRUCTURE_SCHEMA"),
    (lambda item: item["records"][0].update(hidden_resets=1), "FROZEN_FORK"),
    (lambda item: item["records"][0].update(parameter_updates=1), "FROZEN_FORK"),
    (lambda item: item["records"][0]["primary"].update(clean_completion_rate=.123), "RATE_DENOMINATOR"),
    (lambda item: item["records"][0]["primary"]["per_actor"]["matured"].__setitem__(0, 999), "ACTOR_COUNTS"),
    (lambda item: item["records"][0]["censored"]["primary"].update(count=5), "CENSOR_ACCOUNTING"),
    (lambda item: item["records"][0].update(inactive_actions_per_idle_opportunity=.9), "INACTIVE_DENOMINATOR"),
    (lambda item: item["records"][0]["opportunities"].update(inactive_opportunities=5), "OPPORTUNITY_PARTITION"),
    (lambda item: item["records"][0].update(model_action_hash=digest("oracle")), "UNMODIFIED_ACTIONS"),
    (lambda item: item["records"][0]["opportunities"].update(model_action_changes=1), "NO_ACTION_ORACLE"),
    (lambda item: item["records"][0].update(graph_parameter_hash=digest("other weights")), "OWN_ACTION_ONLY"),
    (lambda item: item["intervention_metadata"]["interventions"]["dag_cut"]["retained_offdiagonal_per_target"].__setitem__(0, 999), "MASK_COUNTS"),
    (lambda item: item["intervention_metadata"]["interventions"]["dag_norm_matched"]["offdiagonal_row_l2"].__setitem__(0, 999.), "MATCHED_NORMS"),
])
def test_corrupted_provenance_denominators_and_controls_are_rejected(dataset, change, code):
    mutate(dataset, change)
    with pytest.raises(AuditError, match=code):
        build_report(*dataset[:2])


@pytest.mark.parametrize("field", ["start_world_state_hash", "start_hidden_hash", "start_scheduler_hash",
                                      "end_scheduler_hash", "end_event_hash", "first_observation_hash"])
def test_external_requests_and_start_state_are_paired_across_mask_files(dataset, field):
    def change(item):
        for row in item["records"]:
            row[field] = digest("different external history", field)
            if field == "first_observation_hash":
                row["first_model_observation_hash"] = row[field]
    mutate(dataset, change, filename="seed_000_A_mask_29.json")
    with pytest.raises(AuditError, match="EXOGENOUS_PAIRING_ACROSS_MASKS"):
        build_report(*dataset[:2])


def test_request_fingerprint_is_paired_across_mask_files(dataset):
    mutate(dataset, lambda item: [row["primary"].update(request_fingerprint=digest("unpaired"))
        for row in item["records"]], filename="seed_000_A_mask_29.json")
    with pytest.raises(AuditError, match="EXOGENOUS_PAIRING_ACROSS_MASKS"):
        build_report(*dataset[:2])


@pytest.mark.parametrize("target", ["source", "checkpoint"])
def test_actual_frozen_file_bytes_are_verified(dataset, target):
    plan = json.loads(dataset[0].read_text(encoding="utf-8"))
    path = Path(next(iter(plan["sources"]))) if target == "source" else Path(plan["checkpoints"][0]["path"])
    path.write_bytes(b"changed after protocol freeze")
    with pytest.raises(AuditError, match="FROZEN_SOURCE_SHA256|CHECKPOINT_SHA256"):
        build_report(*dataset[:2])


def test_zero_denominators_are_explicit_and_not_silently_dropped(dataset):
    for mask in MASK_SEEDS:
        def change(item):
            item["records"] = [record(item["source"], mask, row["condition"], row["graph_intervention"],
                n=0 if row["condition"] == "iid" else 160) for row in item["records"]]
        mutate(dataset, change, filename=f"seed_000_A_mask_{mask}.json")
    report = build_report(*dataset[:2])
    result = select(report, "random_norm_matched")["metrics"]["strict_clean_completion"]
    assert result["n"] == 2 and result["n_total"] == 3 and result["n_undefined"] == 1
    assert result["per_seed"][0]["value"] is None and result["per_seed"][0]["mask_values"] == [None]*3
    assert select(report, "random_norm_matched")["metrics"]["idle_action_rate"]["n"] == 3


def test_cli_rejects_before_writing_complete_artifacts(dataset):
    mutate(dataset, lambda item: item.update(protocol_sha256="0"*64))
    with pytest.raises(SystemExit) as error:
        main(["--protocol", str(dataset[0]), "--root", str(dataset[1]), "--output", str(dataset[2])])
    assert error.value.code == 1 and not dataset[2].exists()


def test_rejects_duplicate_json_keys_and_nonfinite_json(dataset):
    path = dataset[1] / "seed_000_A_mask_11.json"
    saved = path.read_text(encoding="utf-8")
    path.write_text('{"format": "x", "format": "y"}', encoding="utf-8")
    with pytest.raises(AuditError, match="duplicate JSON key"):
        build_report(*dataset[:2])
    path.write_text(saved[:-1] + ', "bad": NaN}', encoding="utf-8")
    with pytest.raises(AuditError, match="nonfinite JSON number"):
        build_report(*dataset[:2])
