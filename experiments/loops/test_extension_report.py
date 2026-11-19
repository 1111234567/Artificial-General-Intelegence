"""Small real CPU continuations and independent corruption fixtures.

The toy helpers exercise the current public checkpoint loader, Adam/world/
hidden restoration and native observer. They do not run the formal experiment.
The separate protocol tests enforce the formal finite budget and source set.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import TrainConfig, Trainer, Variant, rng_state, state_digest
from experiments.loops import extension_report as report
from experiments.loops.extension_study import evaluate
from experiments.loops.train import LoopTrainer, write_artifacts


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


@pytest.fixture(scope="module")
def toy_study(tmp_path_factory):
    root = tmp_path_factory.mktemp("extension-report")
    raw = root / "raw"
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with _preserve_process_state():
        base = Trainer(Variant("extension-report-toy", width=128, policy="online"),
            TrainConfig(seed=0, batch_size=2, n_jobs=4, tbptt=4, updates=1, device="cpu",
                train_dts=(.25,), replay_capacity=2, replay_block_ticks=8, replay_burnin=4, log_every=2))
        base.run_updates(1)
        source = root / "A.pt"
        base.save_checkpoint(source)
        frozen = root / "frozen.py"
        frozen.write_text("# synthetic finite helper protocol\n", encoding="utf-8")
        prior = root / "prior.json"
        save(prior, dict(complete=True, audit=dict(status="passed", errors=[]),
            counts=dict(sources=9, jobs=27, forks=540, structural_summaries=9)))
        branches = [dict(id=f"seed_000_{arm}", seed=0, arm=arm, source=str(source),
            source_sha256=report.file_sha256(source), output=str(raw / "training" / "seed_000" / arm)) for arm in report.ARMS]
        plan = dict(format="loops-extension-protocol-v1", sources={str(frozen): report.file_sha256(frozen)},
            completed_existing_audit=str(prior), completed_existing_audit_sha256=report.file_sha256(prior),
            branches=branches, training_device="cpu", evaluation_device="cpu", target_ticks=64, chunk_ticks=64,
            conditions=list(report.CONDITIONS), diagnostic_conditions=list(report.DIAGNOSTIC_CONDITIONS),
            evaluation_steps=48, solver_refinement_max_step=.125, dt=.25,
            parameters=dict(state_scalars=128, baseline=34348, source_square=50732, interaction=50732))
        protocol = root / "protocol.json"
        save(protocol, plan)
        protocol_hash = report.file_sha256(protocol)
        for item in branches:
            base = Trainer.from_checkpoint(source, device="cpu")
            trainer = LoopTrainer(base, item["arm"], chunk_ticks=64,
                origin=dict(checkpoint=str(source), checkpoint_sha256=report.file_sha256(source)))
            trainer.train_update()
            folder = Path(item["output"])
            write_artifacts(trainer, folder)
            checkpoint = folder / "checkpoint.pt"
            save(folder.with_name(folder.name + ".launch-start.json"),
                 dict(format="loops-launch-start-v1", protocol_sha256=protocol_hash, branch=item))
            save(folder / "launch-complete.json", dict(format="loops-launch-complete-v1", protocol_sha256=protocol_hash,
                branch=item, stage_ticks=64, target=64, checkpoint_sha256=report.file_sha256(checkpoint),
                semantic_hash=state_digest(trainer.semantic_state()), hidden_resets=0))
            evaluate(protocol, item["id"], checkpoint, raw / "evaluation" / f"{item['id']}.json")
    try:
        yield dict(root=root, raw=raw, source=source, protocol=protocol, protocol_hash=protocol_hash, plan=plan)
    finally:
        torch.set_num_threads(threads)


@pytest.fixture
def case(toy_study, request):
    value = deepcopy(toy_study)
    arm = getattr(request, "param", "interaction")
    value["item"] = next(item for item in value["plan"]["branches"] if item["arm"] == arm)
    folder = Path(value["item"]["output"])
    value.update(folder=folder, checkpoint=folder / "checkpoint.pt", training_report=folder / "report.json",
                 marker=folder / "launch-complete.json", start=folder.with_name(folder.name + ".launch-start.json"),
                 evaluation=value["raw"] / "evaluation" / f"{value['item']['id']}.json")
    saved = {path: path.read_bytes() for path in value["root"].rglob("*") if path.is_file()}
    checks, inputs = report._Checks(), {}
    value["origin"] = report._source_summary(value["source"], "cpu", checks, inputs)
    try:
        yield value
    finally:
        for path, data in saved.items():
            if not path.exists() or path.read_bytes() != data:
                path.write_bytes(data)


def branch(case):
    audit, inputs = report._Checks(), {}
    value = report._audit_completed_branch(case["item"], case["origin"], case["plan"], case["protocol_hash"], audit, inputs)
    return value, audit, inputs


def change_json(path, function):
    value = json.loads(path.read_text(encoding="utf-8"))
    function(value)
    save(path, value)


def change_checkpoint(case, function, *, refresh_semantic=False):
    value = torch.load(case["checkpoint"], map_location="cpu", weights_only=False)
    function(value)
    torch.save(value, case["checkpoint"])
    change_json(case["marker"], lambda marker: marker.update(checkpoint_sha256=report.file_sha256(case["checkpoint"])))
    if refresh_semantic:
        with _preserve_process_state():
            trainer = LoopTrainer.from_checkpoint(case["checkpoint"], device="cpu")
            semantic = state_digest(trainer.semantic_state())
        change_json(case["marker"], lambda marker: marker.update(semantic_hash=semantic))


@pytest.mark.parametrize("case", report.ARMS, indirect=True)
def test_actual_toy_checkpoint_training_and_observer_are_audited(case, monkeypatch):
    original = LoopTrainer.from_checkpoint
    calls = []
    def load(*args, **kwargs):
        calls.append((args[0], kwargs))
        return original(*args, **kwargs)
    monkeypatch.setattr(report.LoopTrainer, "from_checkpoint", load)
    before = state_digest(rng_state())
    value, audit, inputs = branch(case)
    rows = report._evaluation(case["evaluation"], case["item"], value, case["plan"], case["protocol_hash"], audit, inputs)
    assert len(calls) == 1 and calls[0][1] == dict(device="cpu", strict_sources=True)
    assert state_digest(rng_state()) == before
    assert audit.result()["errors"] == [] and audit.result()["checks"] > 1000
    assert len(rows) == (7 if case["item"]["arm"] == "baseline" else 9)
    assert value["adam"]["recurrent"]["step"] == case["origin"]["adam"]["recurrent"]["step"] + 1
    assert value["model"]["total_parameters"] == case["plan"]["parameters"][case["item"]["arm"]]
    if case["item"]["arm"] != "baseline":
        assert value["adam"]["branch"]["step"] == 1
        assert value["branch_metrics"]["branch_weight_l2"] > 0
    assert all(report.file_sha256(path) == digest for path, digest in inputs.items())
    # No full tensor, Adam moment array or training trace is copied into output.
    json.dumps(value, allow_nan=False)
    assert "trace" not in value and "optimizer" not in value


@pytest.mark.parametrize("target,change,code", [
    ("start", lambda value: value.update(protocol_sha256="0"*64), "START_BINDING"),
    ("start", lambda value: value["branch"].update(seed=1), "START_BINDING"),
    ("marker", lambda value: value.update(protocol_sha256="0"*64), "COMPLETE_BINDING"),
    ("marker", lambda value: value.update(stage_ticks=0), "COMPLETE_BINDING"),
    ("marker", lambda value: value.update(hidden_resets=1), "COMPLETE_BINDING"),
    ("marker", lambda value: value.update(checkpoint_sha256="0"*64), "COMPLETE_BINDING"),
    ("marker", lambda value: value.update(semantic_hash="0"*64), "ACTUAL_CHECKPOINT_SEMANTIC"),
    ("training_report", lambda value: value.update(stage_ticks=0), "TRAINING_BUDGET"),
    ("training_report", lambda value: value.update(hidden_hash="0"*64), "ACTUAL_VALUE_HASH"),
    ("training_report", lambda value: value["work"].update(old_samples=1), "CHECKPOINT_REPORT"),
    ("training_report", lambda value: value["trace"].pop(), "TRACE_ACTUAL_CHECKPOINT"),
    ("training_report", lambda value: value["runtime"].update(device="cuda"), "RUNTIME_LAST_TRAINING"),
    ("training_report", lambda value: value["branch_metrics"].update(branch_weight_l2=999.), "SCALAR_VALUE"),
])
def test_completion_start_training_and_runtime_bindings_reject_corruption(case, target, change, code):
    change_json(case[target], change)
    with pytest.raises(report.AuditError, match=code):
        branch(case)


def test_actual_parameter_edit_is_not_accepted_when_only_byte_marker_is_refreshed(case):
    change_checkpoint(case, lambda value: value["credit"]["continuation"]["model"]["recurrent"].add_(.01))
    with pytest.raises(report.AuditError, match="ACTUAL_CHECKPOINT_SEMANTIC"):
        branch(case)


def test_actual_adam_step_edit_is_rejected_even_with_refreshed_semantic_marker(case):
    change_checkpoint(case, lambda value: value["credit"]["continuation"]["optimizer"]["state"][4]["step"].add_(1),
                      refresh_semantic=True)
    with pytest.raises(report.AuditError, match="ADAM_CONTINUOUS_STEP"):
        branch(case)


def test_public_loader_rejects_malformed_adam_moments(case):
    change_checkpoint(case, lambda value: value["credit"]["continuation"]["optimizer"]["state"][4].update(exp_avg=torch.zeros(1)))
    with pytest.raises(ValueError, match="Adam checkpoint moment shape"):
        branch(case)


def test_embedded_continuity_source_text_is_rehashed_even_when_semantic_marker_is_unchanged(case):
    change_checkpoint(case, lambda value: value["credit"]["continuation"]["sources"].update(
        {"model.py": value["credit"]["continuation"]["sources"]["model.py"] + "\n# corrupted embedded snapshot\n"}))
    with pytest.raises(report.AuditError, match="EMBEDDED_SOURCE_HASH"):
        branch(case)


def test_checkpoint_from_other_source_is_rejected_with_refreshed_marker(case):
    change_checkpoint(case, lambda value: value["origin"].update(checkpoint_sha256="0"*64), refresh_semantic=True)
    change_json(case["training_report"], lambda value: value["loop_origin"].update(checkpoint_sha256="0"*64))
    with pytest.raises(report.AuditError, match="ORIGIN_BYTES"):
        branch(case)


def test_declared_trained_seed_is_checked_against_actual_source_and_checkpoint(case):
    case["item"]["seed"] = 1
    change_json(case["start"], lambda value: value["branch"].update(seed=1))
    change_json(case["marker"], lambda value: value["branch"].update(seed=1))
    with pytest.raises(report.AuditError, match="ACTUAL_TRAINED_SEED"):
        branch(case)


@pytest.mark.parametrize("change,code", [
    (lambda value: value.update(checkpoint_state_hash="0"*64), "EVALUATION_PROVENANCE"),
    (lambda value: value.update(training_device_normalized_state_hash="0"*64), "EVALUATION_PROVENANCE"),
    (lambda value: value["records"].pop(), "FORK_COVERAGE"),
    (lambda value: value["records"].append(deepcopy(value["records"][0])), "DUPLICATE_FORK"),
    (lambda value: value["records"][0]["primary"].update(clean_completion_rate=.917), "RATE_DENOMINATOR"),
    (lambda value: value["records"][0]["censored"]["primary"].update(count=999), "CENSOR_ACCOUNTING"),
    (lambda value: value["records"][0].update(inactive_actions_per_idle_opportunity=.9), "INACTIVE_DENOMINATOR"),
    (lambda value: value["records"][0].update(start_hidden_hash="0"*64), "ACTUAL_EVALUATION_BOUNDARY"),
    (lambda value: value["records"][0].update(parameter_hash="0"*64), "ACTUAL_EVALUATION_BOUNDARY"),
    (lambda value: value["records"][0].update(parameter_updates=1), "FROZEN_FORK"),
    (lambda value: value["records"][0]["opportunities"].update(model_action_changes=1), "NO_ORACLE"),
    (lambda value: value["records"][0].update(model_action_hash="0"*64), "UNMODIFIED_ACTIONS"),
    (lambda value: value["records"][0].update(observation_hash="0"*64), "UNMODIFIED_OBSERVATIONS"),
])
def test_evaluation_provenance_denominators_and_complete_set_are_checked(case, change, code):
    checkpoint, audit, inputs = branch(case)
    change_json(case["evaluation"], change)
    with pytest.raises(report.AuditError, match=code):
        report._evaluation(case["evaluation"], case["item"], checkpoint, case["plan"], case["protocol_hash"], audit, inputs)


def test_solver_fork_cannot_substitute_a_different_exogenous_cohort(case):
    checkpoint, audit, inputs = branch(case)
    def change(value):
        row = next(row for row in value["records"] if row["extension_intervention"] == "solver_half")
        row["primary"]["request_fingerprint"] = "0"*64
    change_json(case["evaluation"], change)
    with pytest.raises(report.AuditError, match="DIAGNOSTIC_EXOGENOUS_PAIRING"):
        report._evaluation(case["evaluation"], case["item"], checkpoint, case["plan"], case["protocol_hash"], audit, inputs)


def test_device_normalization_is_exactly_one_config_field_and_restores_value(case):
    with _preserve_process_state():
        trainer = LoopTrainer.from_checkpoint(case["checkpoint"], device="cpu")
        value = trainer.semantic_state()
        actual = state_digest(value)
        expected = deepcopy(value)
        expected["credit"]["continuation"]["config"]["device"] = "cuda"
        normalized = report._normalized_semantic_hash(value, "cuda")
        assert normalized == state_digest(expected) and normalized != actual
        assert state_digest(value) == actual
        expected["credit"]["continuation"]["hidden"][0, 0] += 1
        assert report._normalized_semantic_hash(expected, "cuda") != normalized


def test_evaluation_binds_actual_cpu_and_cuda_normalized_hashes_separately(case):
    checkpoint, audit, inputs = branch(case)
    checkpoint["training_device_normalized_state_hash"] = hashlib.sha256(b"different CUDA metadata digest").hexdigest()
    change_json(case["evaluation"], lambda value: value.update(
        training_device_normalized_state_hash=checkpoint["training_device_normalized_state_hash"]))
    report._evaluation(case["evaluation"], case["item"], checkpoint, case["plan"], case["protocol_hash"], audit, inputs)
    change_json(case["evaluation"], lambda value: value.update(checkpoint_state_hash=checkpoint["training_device_normalized_state_hash"]))
    with pytest.raises(report.AuditError, match="EVALUATION_PROVENANCE"):
        report._evaluation(case["evaluation"], case["item"], checkpoint, case["plan"], case["protocol_hash"], audit, inputs)


@pytest.fixture
def formal_plan(tmp_path):
    sources = {}
    groups = {"loops": ("__init__.py", "model.py", "train.py", "test_model.py", "test_train.py", "extension_study.py", "audit_run.py", "structure.py", "interventions.py"),
        "continuity": ("model.py", "world.py", "run.py", "evaluate.py"), "integration": ("run.py", "diagnose.py"),
        "credit": ("run.py", "local.py", "batch.py", "windows_runner.py", "evaluate_ready.py")}
    for group, names in groups.items():
        for name in names:
            source = report.ROOT / "experiments" / group / name
            copied = tmp_path / "frozen" / "experiments" / group / name
            copied.parent.mkdir(parents=True, exist_ok=True)
            copied.write_bytes(source.read_bytes())
            sources[str(copied)] = report.file_sha256(copied)
    prior = tmp_path / "prior.json"
    save(prior, dict(complete=True, audit=dict(status="passed", errors=[]),
         counts=dict(sources=9, jobs=27, forks=540, structural_summaries=9)))
    raw = tmp_path / "raw"
    branches, history = [], []
    for seed in report.SEEDS:
        source = tmp_path / f"A_{seed}.pt"
        source.write_bytes(f"Synthetic source-binding bytes {seed}".encode())
        old = tmp_path / f"old_{seed}.pt"
        old.write_bytes(f"Synthetic historical-binding bytes {seed}".encode())
        history.append(dict(seed=seed, path=str(old), sha256=report.file_sha256(old)))
        for arm in report.ARMS:
            branches.append(dict(id=f"seed_{seed:03d}_{arm}", seed=seed, arm=arm, source=str(source),
                source_sha256=report.file_sha256(source), output=str(raw / "training" / f"seed_{seed:03d}" / arm)))
    plan = dict(format="loops-extension-protocol-v1", sources=sources, completed_existing_audit=str(prior),
        completed_existing_audit_sha256=report.file_sha256(prior), branches=branches, historical_bp64_checkpoints=history,
        training_device="cuda", evaluation_device="cpu", target_ticks=65536, chunk_ticks=64,
        conditions=list(report.CONDITIONS), evaluation_steps=2048, diagnostic_conditions=list(report.DIAGNOSTIC_CONDITIONS),
        solver_refinement_max_step=.125, dt=.25,
        parameters=dict(state_scalars=128, baseline=34348, source_square=50732, interaction=50732))
    protocol = tmp_path / "protocol.json"
    save(protocol, plan)
    return protocol, raw, plan


def test_formal_protocol_declares_exact_nine_branches_and_historical_bindings(formal_plan):
    protocol, raw, _ = formal_plan
    audit, inputs = report._Checks(), {}
    plan, branches = report._plan(protocol, raw, audit, inputs)
    assert len(branches) == 9 and len(plan["historical_bp64_checkpoints"]) == 3
    assert audit.result()["status"] == "passed"
    assert all(report.file_sha256(path) == digest for path, digest in inputs.items())


@pytest.mark.parametrize("change,code", [
    (lambda value: value.update(target_ticks=64), "PROTOCOL"),
    (lambda value: value.update(training_device="cpu"), "PROTOCOL"),
    (lambda value: value["branches"].pop(), "BRANCH_COUNT"),
    (lambda value: value["branches"].__setitem__(1, deepcopy(value["branches"][0])), "BRANCH_ID"),
    (lambda value: value["branches"][0].update(source_sha256="0"*64), "BOUND_SOURCE_SHA256"),
    (lambda value: value["branches"][0].update(output="elsewhere"), "BRANCH_OUTPUT"),
    (lambda value: value["historical_bp64_checkpoints"].pop(), "HISTORICAL_COUNT"),
    (lambda value: value["historical_bp64_checkpoints"][0].update(sha256="0"*64), "HISTORICAL_SHA256"),
])
def test_formal_protocol_does_not_accept_partial_or_rebound_work(formal_plan, change, code):
    protocol, raw, _ = formal_plan
    change_json(protocol, change)
    with pytest.raises(report.AuditError, match=code):
        report._plan(protocol, raw, report._Checks(), {})


def test_frozen_runtime_source_coverage_and_actual_bytes_are_checked(formal_plan):
    protocol, raw, plan = formal_plan
    original = protocol.read_bytes()
    key = next(key for key in plan["sources"] if key.replace("\\", "/").endswith("loops/model.py"))
    change_json(protocol, lambda value: value["sources"].pop(key))
    with pytest.raises(report.AuditError, match="FROZEN_RUNTIME_SOURCE_COVERAGE"):
        report._plan(protocol, raw, report._Checks(), {})
    protocol.write_bytes(original)
    Path(key).write_bytes(b"Changed frozen runtime code")
    with pytest.raises(report.AuditError, match="FROZEN_SOURCE_SHA256"):
        report._plan(protocol, raw, report._Checks(), {})


def test_statistics_use_three_trained_seeds_with_explicit_undefined_values():
    rows = [dict(seed=seed, arm="interaction", condition="iid", metric="strict_completion", value=value)
            for seed, value in enumerate((.2, .5, .8))]
    result = report._summaries(rows, ("arm", "condition", "metric"))[0]
    assert result["mean"] == pytest.approx(.5) and result["sample_sd"] == pytest.approx(.3)
    assert result["n"] == result["n_total"] == 3 and result["per_seed"] == [{"seed": i, "value": v} for i, v in enumerate((.2, .5, .8))]
    rows[1]["value"] = None
    result = report._summaries(rows, ("arm", "condition", "metric"))[0]
    assert result["n"] == 2 and result["n_undefined"] == 1 and result["n_total"] == 3
    assert result["mean"] == pytest.approx(.5) and result["sample_sd"] == pytest.approx(.6 / 2**.5)
    with pytest.raises(report.AuditError, match="SUMMARY_SEED_COVERAGE"):
        report._summaries(rows + [rows[0]], ("arm", "condition", "metric"))


def test_world_normalization_removes_only_explicit_regime_names(case):
    with _preserve_process_state():
        trainer = LoopTrainer.from_checkpoint(case["checkpoint"], device="cpu")
        original = trainer.world.state_dict()
    renamed = deepcopy(original)
    renamed["regime"]["name"] = "credit"
    for row in renamed["regime_history"]:
        row["name"] = "legacy metadata name"
    assert state_digest(original) != state_digest(renamed)
    assert state_digest(report._world_without_regime_names(original)) == state_digest(report._world_without_regime_names(renamed))
    renamed["regime"]["gap_range"] = (1., 2.)
    assert state_digest(report._world_without_regime_names(original)) != state_digest(report._world_without_regime_names(renamed))


def test_public_report_rejects_toy_budget_and_cli_writes_no_complete_result(case, tmp_path):
    with pytest.raises(report.AuditError, match="PROTOCOL"):
        report.build_report(case["protocol"], case["raw"])
    output = tmp_path / "rejected"
    with pytest.raises(SystemExit) as failure:
        report.main(["--protocol", str(case["protocol"]), "--root", str(case["raw"]), "--output", str(output)])
    assert failure.value.code == 1 and not output.exists()
