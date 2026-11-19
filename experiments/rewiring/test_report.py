"""Real small CPU continuations and independently corrupted audit inputs."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path

import pytest
import torch

from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import TrainConfig, Trainer, Variant, rng_state, state_digest
from experiments.credit.run import CreditConfig, CreditTrainer
from experiments.rewiring import report
from experiments.rewiring.plasticity import RewireConfig
from experiments.rewiring.run import RewireTrainer, run_to_target
from experiments.rewiring.study import evaluate


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def change_json(path, change):
    value = json.loads(path.read_text(encoding="utf-8"))
    change(value)
    save(path, value)


@pytest.fixture(scope="module")
def toy_study(tmp_path_factory):
    root = tmp_path_factory.mktemp("rewiring-report")
    raw, old_threads = root / "raw", torch.get_num_threads()
    torch.set_num_threads(1)
    with _preserve_process_state(), redirect_stdout(io.StringIO()):
        base = Trainer(Variant("rewiring-report-toy", width=128, policy="online"),
            TrainConfig(seed=0, batch_size=2, n_jobs=4, tbptt=4, updates=2, device="cpu",
                train_dts=(.25,), replay_capacity=2, replay_block_ticks=8, replay_burnin=4, log_every=2))
        base.run_updates(2)
        source = root / "A.pt"
        base.save_checkpoint(source)
        frozen = root / "frozen.py"
        frozen.write_text("# Small helper fixture, never a formal study\n", encoding="utf-8")
        branches = [dict(id=f"seed_000_{arm}", seed=0, condition_id=arm, arm=arm,
            shrink_step=0. if arm == "dense" else 1e-4, structural_seed=91000,
            source=str(source), source_sha256=report.file_sha256(source),
            output=str(raw / "training" / "seed_000" / arm)) for arm in ("dense", "retire", "learned", "random")]
        history = root / "old.pt"
        old = CreditTrainer.from_base_checkpoint(source, CreditConfig(mode="bp64", chunk_ticks=4), device="cpu")
        old.run_updates(2)
        old.save_checkpoint(history)
        plan = dict(format="rewiring-study-protocol-v1", sources={str(frozen): report.file_sha256(frozen)},
            branches=branches, arms=list(report.ARMS), seeds=[0], phase_ticks=8, target_ticks=16,
            window_updates=2, chunk_ticks=4, training_device="cpu", evaluation_device="cpu", evaluation_steps=48,
            conditions=list(report.CONDITIONS), phases=list(report.PHASES), dt=.25,
            historical_bp64_checkpoints=[dict(seed=0, path=str(history), sha256=report.file_sha256(history))],
            regimes={"B": {"duration": [4., 8.], "gap": [4., 12.]}, "A2": {"duration": [1., 4.], "gap": [.5, 2.]}})
        protocol = root / "protocol.json"
        save(protocol, plan)
        protocol_hash = report.file_sha256(protocol)
        for item in branches:
            base = Trainer.from_checkpoint(source, device="cpu")
            trainer = RewireTrainer(base, RewireConfig(arm=item["arm"], shrink_step=item["shrink_step"],
                window_updates=2, structural_seed=91000), chunk_ticks=4, phase_ticks=8,
                origin=dict(checkpoint=str(source), checkpoint_sha256=report.file_sha256(source)))
            folder = Path(item["output"])
            run_to_target(trainer, 16, folder, checkpoint_every=0)
            save(folder.with_name(folder.name + ".launch-start.json"),
                 dict(format="rewiring-launch-start-v1", protocol_sha256=protocol_hash, branch=item))
            checkpoints = {}
            for phase, tick in (("B", 8), ("A2", 16)):
                path = folder / f"checkpoint_{phase}.pt"
                cp = RewireTrainer.from_checkpoint(path, device="cpu")
                checkpoints[phase] = dict(path=str(path), sha256=report.file_sha256(path), stage_ticks=tick,
                                         semantic_hash=state_digest(cp.semantic_state()))
            save(folder / "launch-complete.json", dict(format="rewiring-launch-complete-v1", protocol_sha256=protocol_hash,
                branch=item, checkpoints=checkpoints, hidden_resets=0))
            evaluate(protocol, item["id"], folder / "checkpoint.pt", raw / "evaluation" / f"{item['id']}.json")
    try:
        yield dict(root=root, raw=raw, source=source, protocol=protocol, protocol_hash=protocol_hash, plan=plan)
    finally:
        torch.set_num_threads(old_threads)


@pytest.fixture
def case(toy_study, request):
    value = deepcopy(toy_study)
    arm = getattr(request, "param", "learned")
    value["item"] = next(row for row in value["plan"]["branches"] if row["arm"] == arm)
    folder = Path(value["item"]["output"])
    value.update(folder=folder, checkpoint=folder / "checkpoint_A2.pt", training_report=folder / "report.json",
                 marker=folder / "launch-complete.json", start=folder.with_name(folder.name + ".launch-start.json"),
                 evaluation=value["raw"] / "evaluation" / f"{value['item']['id']}.json", trace=folder / "training.jsonl",
                 windows=folder / "structural_windows.jsonl")
    saved = {path: path.read_bytes() for path in value["root"].rglob("*") if path.is_file()}
    value["origin"] = report._source(value["source"], "cpu", report._Checks(), {})
    try:
        yield value
    finally:
        for path, data in saved.items():
            if not path.exists() or path.read_bytes() != data:
                path.write_bytes(data)


def branch(case):
    audit, inputs = report._Checks(), {}
    checkpoints = report._audit_completed_branch(case["item"], case["origin"], case["plan"], case["protocol_hash"], audit, inputs)
    return checkpoints, audit, inputs


def change_checkpoint(case, change, *, semantic=False):
    path = case["checkpoint"]
    value = torch.load(path, map_location="cpu", weights_only=False)
    change(value)
    torch.save(value, path)
    change_json(case["marker"], lambda marker: marker["checkpoints"]["A2"].update(sha256=report.file_sha256(path)))
    if semantic:
        with _preserve_process_state():
            trainer = RewireTrainer.from_checkpoint(path, device="cpu", strict_sources=True)
            digest = state_digest(trainer.semantic_state())
        change_json(case["marker"], lambda marker: marker["checkpoints"]["A2"].update(semantic_hash=digest))


@pytest.mark.parametrize("case", ("dense", "retire", "learned", "random"), indirect=True)
def test_real_checkpoints_and_frozen_forks_pass_without_rng_mutation(case, monkeypatch):
    loader, calls = RewireTrainer.from_checkpoint, []
    def load(*args, **kwargs):
        calls.append((args[0], kwargs))
        return loader(*args, **kwargs)
    monkeypatch.setattr(report.RewireTrainer, "from_checkpoint", load)
    before = state_digest(rng_state())
    checkpoints, audit, inputs = branch(case)
    rows = report._evaluation(case["evaluation"], case["item"], checkpoints, case["plan"], case["protocol_hash"], audit, inputs)
    assert state_digest(rng_state()) == before
    assert len(calls) == 3 and all(kwargs == dict(device="cpu", strict_sources=True) for _, kwargs in calls)
    assert len(rows) == 10 and set(checkpoints) == {"B", "A2"}
    assert audit.result()["status"] == "passed" and audit.result()["checks"] > 1000
    for phase, cp in checkpoints.items():
        assert cp["adam"]["recurrent"]["step"] == case["origin"]["adam"]["recurrent"]["step"] + (2 if phase == "B" else 4)
        assert cp["structure"]["active"] == 128**2 + cp["structure"]["cumulative_births"] - cp["structure"]["cumulative_deaths"]
        assert cp["topology"]["device"] == "cpu"
        assert len(cp["capacity_trajectory"]) == (3 if phase == "B" else 5)
        assert cp["capacity_trajectory"][0]["stage_ticks"] == 0
        assert cp["capacity_trajectory"][0]["active"] == 128**2
        assert cp["cost"]["structural_tensor_bytes"] > 0
        assert cp["cost"]["shadow_float_scalars"] == 3 * 128**2
        assert len(cp["structural_windows"]) == (1 if phase == "B" else 2)
    if case["item"]["arm"] == "retire":
        assert checkpoints["A2"]["structure"]["cumulative_births"] == 0
        assert checkpoints["A2"]["structure"]["cumulative_deaths"] > 0
    assert all(report.file_sha256(path) == digest for path, digest in inputs.items())
    json.dumps(checkpoints, allow_nan=False)


@pytest.mark.parametrize("target,change,code", [
    ("start", lambda v: v.update(protocol_sha256="0"*64), "START_BINDING"),
    ("start", lambda v: v["branch"].update(seed=1), "START_BINDING"),
    ("marker", lambda v: v.update(protocol_sha256="0"*64), "COMPLETE_BINDING"),
    ("marker", lambda v: v.update(hidden_resets=1), "COMPLETE_BINDING"),
    ("marker", lambda v: v["checkpoints"]["B"].update(stage_ticks=4), "PHASE_BINDING"),
    ("marker", lambda v: v["checkpoints"]["B"].update(sha256="0"*64), "PHASE_BINDING"),
    ("marker", lambda v: v["checkpoints"]["B"].update(semantic_hash="0"*64), "ACTUAL_CHECKPOINT_SEMANTIC"),
    ("training_report", lambda v: v.update(stage_ticks=0), "CHECKPOINT_REPORT"),
    ("training_report", lambda v: v["structure"].update(active=1), "CHECKPOINT_REPORT"),
    ("training_report", lambda v: v["structure"].update(cumulative_births=12345), "CHECKPOINT_REPORT"),
    ("training_report", lambda v: v["rewire_config"].update(shrink_step=.001), "CHECKPOINT_REPORT"),
    ("training_report", lambda v: v["work"].update(old_samples=1), "CHECKPOINT_REPORT"),
    ("training_report", lambda v: v["runtime"].update(device="cuda"), "RUNTIME_LAST_TRAINING"),
])
def test_branch_bindings_and_report_are_checked(case, target, change, code):
    change_json(case[target], change)
    with pytest.raises(report.AuditError, match=code):
        branch(case)


def test_edited_actual_parameter_requires_semantic_binding(case):
    change_checkpoint(case, lambda value: value["credit"]["continuation"]["model"]["encoder.weight"].add_(.01))
    with pytest.raises(report.AuditError, match="ACTUAL_CHECKPOINT_SEMANTIC"):
        branch(case)


def test_edited_actual_adam_step_rejected_even_with_new_semantic_marker(case):
    change_checkpoint(case, lambda v: v["credit"]["continuation"]["optimizer"]["state"][1]["step"].add_(1), semantic=True)
    with pytest.raises(report.AuditError, match="ADAM_CONTINUOUS_STEP"):
        branch(case)


def test_actual_operator_count_rejected_by_public_loader(case):
    change_checkpoint(case, lambda v: v["rewiring"].update(cumulative_deaths=v["rewiring"]["cumulative_deaths"] + 1))
    with pytest.raises(ValueError, match="cumulative structural counts"):
        branch(case)


def test_event_accounting_rejected_even_with_new_semantic_marker(case):
    change_checkpoint(case, lambda v: v["credit"]["continuation"]["trace"][-1]["parameter_update"]["structure"].update(births=999), semantic=True)
    with pytest.raises(report.AuditError, match="BIRTH_EVENT_TOTAL"):
        branch(case)


@pytest.mark.parametrize("field,bad", [("survival_sha256", "0"*64), ("proposal_observations", 999)])
def test_final_operator_summary_is_checked_against_event_and_window(case, field, bad):
    def change(value):
        value["credit"]["continuation"]["trace"][-1]["parameter_update"]["structure"][field] = bad
        value["structural_windows"][-1][field] = bad
    change_checkpoint(case, change, semantic=True)
    with pytest.raises(report.AuditError, match="FINAL_OPERATOR_TRACE"):
        branch(case)


def test_rewire_source_text_is_rehashed_independently(case):
    change_checkpoint(case, lambda v: v["sources"].update({"run.py": v["sources"]["run.py"] + "\n# corruption\n"}))
    with pytest.raises((ValueError, report.AuditError), match="source|SOURCE"):
        branch(case)


def test_continuity_source_text_is_rehashed_even_without_semantic_change(case):
    change_checkpoint(case, lambda v: v["credit"]["continuation"]["sources"].update({"model.py":
        v["credit"]["continuation"]["sources"]["model.py"] + "\n# corruption\n"}))
    with pytest.raises(report.AuditError, match="EMBEDDED_SOURCE_HASH"):
        branch(case)


def test_origin_binding_is_not_replaceable_with_new_semantic_marker(case):
    change_checkpoint(case, lambda v: v["origin"].update(checkpoint_sha256="0"*64), semantic=True)
    with pytest.raises(report.AuditError, match="ORIGIN_BYTES"):
        branch(case)


def test_training_jsonl_and_checkpoint_trace_must_match(case):
    rows = [json.loads(line) for line in case["trace"].read_text().splitlines()]
    rows[0]["parameter_update"]["structure"]["active"] += 1
    case["trace"].write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(report.AuditError, match="TRAINING_JSONL"):
        branch(case)


@pytest.mark.parametrize("change,code", [
    (lambda v: v.update(source_unchanged=False), "EVALUATION_PROVENANCE"),
    (lambda v: v["states"]["B"].update(state_hash="0"*64), "EVALUATION_STATE"),
    (lambda v: v["states"]["A2"].update(training_device_normalized_state_hash="0"*64), "EVALUATION_STATE"),
    (lambda v: v["records"].pop(), "FORK_COVERAGE"),
    (lambda v: v["records"].append(deepcopy(v["records"][0])), "DUPLICATE_FORK"),
    (lambda v: v["records"][0].update(learning_phase="A"), "EXPECTED_FORK"),
    (lambda v: v["records"][0]["primary"].update(clean_completion_rate=.917), "RATE_DENOMINATOR"),
    (lambda v: v["records"][0]["censored"]["primary"].update(count=999), "CENSOR_ACCOUNTING"),
    (lambda v: v["records"][0].update(inactive_actions_per_idle_opportunity=.9), "INACTIVE_DENOMINATOR"),
    (lambda v: v["records"][0].update(start_hidden_hash="0"*64), "ACTUAL_EVALUATION_BOUNDARY"),
    (lambda v: v["records"][0].update(parameter_hash="0"*64), "ACTUAL_EVALUATION_BOUNDARY"),
    (lambda v: v["records"][0].update(source_continuation_hash="0"*64), "ACTUAL_EVALUATION_BOUNDARY"),
    (lambda v: v["records"][0].update(parameter_updates=1), "FROZEN_FORK"),
    (lambda v: v["records"][0]["opportunities"].update(model_action_changes=1), "NO_ORACLE"),
    (lambda v: v["records"][0].update(model_action_hash="0"*64), "UNMODIFIED_ACTIONS"),
])
def test_evaluation_metadata_boundaries_rates_and_complete_set(case, change, code):
    checkpoints, audit, inputs = branch(case)
    change_json(case["evaluation"], change)
    with pytest.raises(report.AuditError, match=code):
        report._evaluation(case["evaluation"], case["item"], checkpoints, case["plan"], case["protocol_hash"], audit, inputs)


def test_device_normalization_changes_exactly_one_config_field_and_restores_it(case):
    with _preserve_process_state():
        trainer = RewireTrainer.from_checkpoint(case["checkpoint"], device="cpu")
        value = trainer.semantic_state()
        original, expected = state_digest(value), deepcopy(value)
        expected["credit"]["continuation"]["config"]["device"] = "cuda"
        digest = report._normalized_semantic_hash(value, "cuda")
        assert digest == state_digest(expected) and digest != original
        assert state_digest(value) == original
        expected["rewiring"]["mask"][0, 0] = ~expected["rewiring"]["mask"][0, 0]
        assert report._normalized_semantic_hash(expected, "cuda") != digest


def test_dense_B_exactly_reproduces_historical_bp64_with_only_world_names_normalized(case):
    item = next(row for row in case["plan"]["branches"] if row["arm"] == "dense")
    audit, inputs = report._Checks(), {}
    checkpoints = report._audit_completed_branch(item, case["origin"], case["plan"], case["protocol_hash"], audit, inputs)
    rows = report._historical(case["plan"], {(0, "dense", "B"): checkpoints["B"]}, {0: case["origin"]}, audit, inputs)
    assert rows[0]["core_reproduction_equal"] is True
    assert rows[0]["equal"]["world"] is False
    assert rows[0]["equal"]["world_without_regime_names"] is True
    checkpoints["B"]["action_hash"] = "0"*64
    with pytest.raises(report.AuditError, match="DENSE_B_HISTORICAL_REPRODUCTION"):
        report._historical(case["plan"], {(0, "dense", "B"): checkpoints["B"]}, {0: case["origin"]}, report._Checks(), {})


@pytest.fixture
def formal_plan(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    sources = {}
    for group, names in report.SOURCE_GROUPS.items():
        for name in names:
            copied = tmp_path / "frozen" / "experiments" / group / name
            copied.parent.mkdir(parents=True, exist_ok=True)
            copied.write_bytes((repository / "experiments" / group / name).read_bytes())
            sources[str(copied)] = report.file_sha256(copied)
    raw, branches, historical = tmp_path / "raw", [], []
    for seed in report.SEEDS:
        source, old = tmp_path / f"A_{seed}.pt", tmp_path / f"old_{seed}.pt"
        source.write_bytes(f"Synthetic source-binding bytes {seed}".encode())
        old.write_bytes(f"Synthetic historical-binding bytes {seed}".encode())
        historical.append(dict(seed=seed, path=str(old), sha256=report.file_sha256(old)))
        for arm in report.ARMS:
            branches.append(dict(id=f"seed_{seed:03d}_{arm['id']}", seed=seed, condition_id=arm["id"], arm=arm["arm"],
                shrink_step=arm["shrink_step"], structural_seed=91000+seed, source=str(source), source_sha256=report.file_sha256(source),
                output=str(raw / "training" / f"seed_{seed:03d}" / arm["id"])))
    plan = dict(format="rewiring-study-protocol-v1", sources=sources, branches=branches, arms=list(report.ARMS), seeds=list(report.SEEDS),
        phase_ticks=65536, target_ticks=131072, chunk_ticks=64, window_updates=8, training_device="cuda", evaluation_device="cpu",
        evaluation_steps=2048, conditions=list(report.CONDITIONS), phases=list(report.PHASES), dt=.25,
        historical_bp64_checkpoints=historical, regimes={"B": {"duration": [4., 8.], "gap": [4., 12.]}, "A2": {"duration": [1., 4.], "gap": [.5, 2.]}})
    protocol = tmp_path / "protocol.json"
    save(protocol, plan)
    return protocol, raw, plan


def test_formal_protocol_exact_branch_dose_phase_and_history_declaration(formal_plan):
    protocol, raw, _ = formal_plan
    audit, inputs = report._Checks(), {}
    plan, branches = report._plan(protocol, raw, audit, inputs)
    assert len(branches) == 21 and plan["target_ticks"] == 2 * plan["phase_ticks"]
    assert len(plan["historical_bp64_checkpoints"]) == 3 and audit.result()["status"] == "passed"
    assert all(report.file_sha256(path) == digest for path, digest in inputs.items())


@pytest.mark.parametrize("change,code", [
    (lambda v: v.update(target_ticks=65536), "PROTOCOL"),
    (lambda v: v.update(phase_ticks=8), "PROTOCOL"),
    (lambda v: v.update(evaluation_steps=48), "PROTOCOL"),
    (lambda v: v.update(training_device="cpu"), "PROTOCOL"),
    (lambda v: v.update(window_updates=4), "PROTOCOL"),
    (lambda v: v["branches"].pop(), "BRANCH_COUNT"),
    (lambda v: v["branches"].__setitem__(1, deepcopy(v["branches"][0])), "BRANCH_ID"),
    (lambda v: v["branches"][1].update(shrink_step=.001), "BRANCH_ARM"),
    (lambda v: v["branches"][0].update(structural_seed=1), "STRUCTURAL_SEED"),
    (lambda v: v["branches"][0].update(source_sha256="0"*64), "BOUND_SOURCE_SHA256"),
    (lambda v: v["branches"][0].update(output="elsewhere"), "BRANCH_OUTPUT"),
    (lambda v: v["historical_bp64_checkpoints"].pop(), "HISTORICAL_COUNT"),
    (lambda v: v["historical_bp64_checkpoints"][0].update(sha256="0"*64), "HISTORICAL_SHA256"),
])
def test_formal_protocol_rejects_partial_or_rebound_work(formal_plan, change, code):
    protocol, raw, _ = formal_plan
    change_json(protocol, change)
    with pytest.raises(report.AuditError, match=code):
        report._plan(protocol, raw, report._Checks(), {})


def test_frozen_runtime_source_coverage_and_bytes_are_checked(formal_plan):
    protocol, raw, plan = formal_plan
    original = protocol.read_bytes()
    key = next(key for key in plan["sources"] if key.replace("\\", "/").endswith("rewiring/plasticity.py"))
    change_json(protocol, lambda v: v["sources"].pop(key))
    with pytest.raises(report.AuditError, match="FROZEN_RUNTIME_SOURCE_COVERAGE"):
        report._plan(protocol, raw, report._Checks(), {})
    protocol.write_bytes(original)
    Path(key).write_bytes(b"Changed frozen runtime code")
    with pytest.raises(report.AuditError, match="FROZEN_SOURCE_SHA256"):
        report._plan(protocol, raw, report._Checks(), {})


def test_three_seed_statistics_preserve_undefined_values_and_sample_sd():
    rows = [dict(seed=seed, condition_id="learned_low", learning_phase="B", condition="iid", metric="strict_completion", value=value)
            for seed, value in enumerate((.2, .5, .8))]
    dimensions = ("condition_id", "learning_phase", "condition", "metric")
    summary = report._summaries(rows, dimensions)[0]
    assert summary["mean"] == pytest.approx(.5) and summary["sample_sd"] == pytest.approx(.3)
    assert summary["n"] == summary["n_total"] == 3
    rows[1]["value"] = None
    summary = report._summaries(rows, dimensions)[0]
    assert summary["n"] == 2 and summary["n_undefined"] == 1 and summary["n_total"] == 3
    assert summary["sample_sd"] == pytest.approx(.6 / 2**.5)
    with pytest.raises(report.AuditError, match="SUMMARY_SEED_COVERAGE"):
        report._summaries(rows + [rows[0]], dimensions)


def test_public_report_and_cli_reject_toy_budget_and_publish_nothing(case, tmp_path):
    with pytest.raises(report.AuditError, match="PROTOCOL"):
        report.build_report(case["protocol"], case["raw"])
    output = tmp_path / "invalid-report"
    with pytest.raises(SystemExit) as error:
        report.main(["--protocol", str(case["protocol"]), "--root", str(case["raw"]), "--output", str(output)])
    assert error.value.code == 1 and not output.exists()
