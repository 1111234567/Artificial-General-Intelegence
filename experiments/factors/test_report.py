"""Exercise real short continuations and adversarial factor-report bindings."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path

import pytest
import torch

from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import Trainer, TrainConfig, Variant, rng_state, state_digest
from experiments.factors import report
from experiments.factors.run import FactorTrainer, run_to_target
from experiments.factors.structure import FactorConfig
from experiments.integration.diagnose import diagnose_fork
from experiments.rewiring.plasticity import RewireConfig
from experiments.rewiring.run import RewireTrainer


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def alter(path, change):
    value = json.loads(path.read_text(encoding="utf-8"))
    change(value)
    save(path, value)


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    root = tmp_path_factory.mktemp("factors-report")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with _preserve_process_state(), redirect_stdout(io.StringIO()):
        base = Trainer(Variant("factor-report-toy", width=128, policy="online"),
            TrainConfig(seed=0, batch_size=2, n_jobs=4, tbptt=4, updates=2,
                device="cpu", train_dts=(.25,), replay_capacity=2,
                replay_block_ticks=8, replay_burnin=4, log_every=2))
        base.run_updates(2)
        initial = root / "A.pt"
        base.save_checkpoint(initial)
        old = RewireTrainer(base, RewireConfig(arm="retire", shrink_step=.0002,
            window_updates=8, structural_seed=91000), chunk_ticks=4, phase_ticks=16,
            origin=dict(checkpoint=str(initial), checkpoint_sha256=report.file_sha256(initial)))
        old.run_updates(4)
        source = root / "B.pt"
        old.save_checkpoint(source)
        old.run_updates(4)
        historical = root / "A2.pt"
        old.save_checkpoint(historical)
        plan = dict(target_ticks=16, source_stage_ticks=16, chunk_ticks=4,
            training_device="cpu", evaluation_device="cpu", evaluation_steps=48,
            dt=.25, return_regime=dict(duration=[1., 4.], gap=[.5, 2.], name="rewiring-A2"))
        branches = []
        for mode in report.MODES:
            for label, strength in (("zero", 0.), ("small", .1)):
                item = dict(id=f"seed_000_{mode}_idle_{label}", seed=0, mode=mode,
                    idle_label=label, idle_lambda=strength, shrink_step=.0002,
                    source=str(source), source_sha256=report.file_sha256(source), origin_id="seed_000",
                    output=str(root / "raw" / "training" / "seed_000" / f"{mode}_idle_{label}"))
                trainer = FactorTrainer(RewireTrainer.from_checkpoint(source, device="cpu"),
                    FactorConfig(mode=mode, idle_lambda=strength), origin=dict(checkpoint=str(source), checkpoint_sha256=item["source_sha256"]))
                folder = Path(item["output"])
                run_to_target(trainer, 16, folder, checkpoint_every=0)
                save(folder.with_name(folder.name + ".launch-start.json"),
                    dict(format="factors-launch-start-v1", branch=item, protocol_sha256="a" * 64))
                save(folder / "launch-complete.json", dict(format="factors-launch-complete-v1", branch=item,
                    protocol_sha256="a" * 64, target=16, stage_ticks=16, hidden_resets=0,
                    checkpoint_sha256=report.file_sha256(folder / "checkpoint.pt"), semantic_hash=state_digest(trainer.semantic_state())))
                branches.append(item)
    try:
        yield dict(root=root, source=source, historical=historical, plan=plan, branches=branches)
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def case(toy, request):
    result = deepcopy(toy)
    mode, strength = getattr(request, "param", ("retire", 0.))
    result["item"] = next(item for item in toy["branches"] if item["mode"] == mode and item["idle_lambda"] == strength)
    folder = Path(result["item"]["output"])
    result.update(folder=folder, checkpoint=folder / "checkpoint.pt", marker=folder / "launch-complete.json",
        saved_report=folder / "report.json", trace=folder / "training.jsonl")
    saved = {path: path.read_bytes() for path in folder.rglob("*") if path.is_file()}
    result["origin"] = report._source(result["source"], "cpu", report._Checks(), {})
    try:
        yield result
    finally:
        for path, value in saved.items():
            if not path.exists() or path.read_bytes() != value:
                path.write_bytes(value)


def checkpoint(case):
    audit, inputs = report._Checks(), {}
    cp = report._checkpoint(case["item"], case["origin"], case["plan"], "a" * 64, audit, inputs)
    return cp, audit, inputs


@pytest.mark.parametrize("case", [(mode, strength) for mode in report.MODES for strength in (0., .1)], indirect=True)
def test_real_short_checkpoint_audit_and_rng_preservation(case):
    before = state_digest(rng_state())
    cp, audit, inputs = checkpoint(case)
    assert state_digest(rng_state()) == before
    assert audit.result()["status"] == "passed"
    assert cp["factor_ticks"] == 16 and cp["training_trace"]["count"] == 4
    assert cp["adam"]["recurrent"]["step"] == case["origin"]["adam"]["recurrent"]["step"] + 4
    assert cp["structure"]["active"] <= cp["structure"]["allowed"]
    assert cp["diagnostics"]["idle_count"] == cp["diagnostics"]["buckets"]["idle"]["count"]
    assert len(cp["capacity_trajectory"]) == 4
    assert all(report.file_sha256(path) == digest for path, digest in inputs.items())
    json.dumps(cp, allow_nan=False)


@pytest.mark.parametrize("target,change,code", [
    ("marker", lambda value: value.update(checkpoint_sha256="0" * 64), "COMPLETE_BINDING"),
    ("marker", lambda value: value.update(semantic_hash="0" * 64), "ACTUAL_CHECKPOINT_SEMANTIC"),
    ("saved_report", lambda value: value["runtime"].update(device="cuda"), "RUNTIME_LAST_TRAINING"),
    ("saved_report", lambda value: value["structure"].update(allowed=1), "CHECKPOINT_REPORT"),
    ("saved_report", lambda value: value["diagnostics"].update(idle_count=1), "CHECKPOINT_REPORT"),
])
def test_json_bindings_and_actual_report_are_checked(case, target, change, code):
    alter(case[target], change)
    with pytest.raises(report.AuditError, match=code):
        checkpoint(case)


def update_checkpoint(case, change):
    value = torch.load(case["checkpoint"], map_location="cpu", weights_only=False)
    change(value)
    torch.save(value, case["checkpoint"])
    trainer = FactorTrainer.from_checkpoint(case["checkpoint"], device="cpu", strict_sources=True)
    alter(case["marker"], lambda marker: marker.update(checkpoint_sha256=report.file_sha256(case["checkpoint"]),
        semantic_hash=state_digest(trainer.semantic_state())))


def test_origin_cannot_be_rebound_even_with_new_semantic_marker(case):
    update_checkpoint(case, lambda value: value["factor_origin"].update(inherited_semantic_hash="0" * 64))
    with pytest.raises(report.AuditError, match="FACTOR_ORIGIN"):
        checkpoint(case)


def test_inherited_adam_step_checked_independently_of_semantic_binding(case):
    update_checkpoint(case, lambda value: value["credit"]["continuation"]["optimizer"]["state"][0]["step"].add_(1))
    with pytest.raises(report.AuditError, match="ADAM_CONTINUOUS_STEP"):
        checkpoint(case)


def test_idle_before_action_denominator_checked_in_actual_trace(case):
    def change(value):
        value["credit"]["continuation"]["trace"][-1]["live"]["diagnostics"]["idle_count"] += 1
    update_checkpoint(case, change)
    with pytest.raises(report.AuditError, match="IDLE_BEFORE_ACTION_COUNT"):
        checkpoint(case)


def test_training_jsonl_is_bound_to_selected_actual_trace_rows(case):
    rows = report._jsonl(case["trace"])
    rows[0]["action_hash"] = "0" * 64
    case["trace"].write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(report.AuditError, match="TRAINING_JSONL"):
        checkpoint(case)


def test_retire_zero_reproduces_core_operator_and_every_historical_trace_field(case):
    cp, audit, inputs = checkpoint(case)
    reference = dict(seed=0, historical_A2=str(case["historical"]), historical_A2_sha256=report.file_sha256(case["historical"]))
    result = report._historical(reference, cp, case["origin"], audit, inputs)
    assert result["equal"] and all(result["core_equal"].values()) and all(result["operator_equal"].values())
    assert result["selected_trace_sha256"] == result["historical_trace_sha256"]


def test_real_frozen_record_and_idle_history_denominators(case):
    cp, audit, _ = checkpoint(case)
    trainer = FactorTrainer.from_checkpoint(case["checkpoint"], device="cpu", strict_sources=True)
    with _preserve_process_state():
        row = diagnose_fork(trainer.model, trainer.world, trainer.h, condition="sparse", steps=48, dt=.25, device="cpu")
    row = report.normalize(row)
    cells = report._record(row, cp, case["plan"], audit, "toy/sparse")
    assert len(cells) == len(report.IDLE_CAUSES) * (len(row["age_edges"]) + 1)
    assert sum(cell["opportunities"] for cell in cells) == row["opportunities"]["inactive_opportunities"]
    assert any(cell["metrics"]["idle_action_rate"] is None for cell in cells)
    broken = deepcopy(row)
    broken["inactive_history"][0]["finish_actions"] += 1
    with pytest.raises(report.AuditError, match="IDLE_HISTORY_SUBSETS"):
        report._idle_history(broken, report._Checks(), "corrupt")
    cell = {**dict.fromkeys(report.IDLE_COUNT_FIELDS, 0),
        "cause": "completion", "time_since_terminal_bin": "[0,1)",
        "opportunities": 10, "actions": 2, "finish_actions": 2,
        "known_retired_content_opportunities": 5, "finish_matches_retired_content": 1,
        "conditional_finish_matches_retired_content": 2, "model_wait_preferences": 8,
        "model_finish_preferences": 2, "finish_preferred_to_wait": 2,
        "retired_exact_finish_preferred_to_wait": 2,
        "finish_minus_wait_logit_sum": 0., "action_rate_per_idle_opportunity": .2,
        "actions_per_1000_total_opportunities": 100., "mean_finish_minus_wait_logit": 0.}
    synthetic = dict(age_edges=row["age_edges"], inactive_history=[cell], opportunities=dict(
        opportunities=20, inactive_opportunities=10, inactive_actions=2,
        inactive_begin_actions=0, inactive_finish_actions=2, inactive_finish_matches_retired_content=1))
    report._idle_history(synthetic, report._Checks(), "matched")
    cell["finish_matches_retired_content"] = 2
    with pytest.raises(report.AuditError, match="IDLE_HISTORY_TOTAL"):
        report._idle_history(synthetic, report._Checks(), "corrupt-retired-match")


def test_paired_statistics_use_seed_differences_and_keep_undefined():
    rows = [dict(seed=seed, condition="sparse", metric="example", value=value)
        for seed, value in enumerate((.1, .2, None))]
    value = report._summaries(rows, ("condition", "metric"))[0]
    assert value["n_total"] == 3 and value["n"] == 2 and value["n_undefined"] == 1
    assert value["mean"] == pytest.approx(.15)
    assert value["sample_sd"] == pytest.approx(2 ** -.5 / 10)
    assert report._difference(None, 1.) is None
    with pytest.raises(report.AuditError, match="SUMMARY_SEED_COVERAGE"):
        report._summaries(rows[:2], ("condition", "metric"))


def test_workers_are_bounded_and_single_worker_keeps_order():
    assert report._map_workers(abs, [-2, -1, 0], 1) == [2, 1, 0]
    for invalid in (0, 7, True, 1.5):
        with pytest.raises(ValueError, match="workers"):
            report._map_workers(abs, [], invalid)


def test_device_normalization_changes_only_config_device(case):
    value = FactorTrainer.from_checkpoint(case["checkpoint"], device="cpu").semantic_state()
    before = state_digest(value)
    actual = report._normalized_semantic_hash(value, "cuda")
    expected = deepcopy(value)
    expected["credit"]["continuation"]["config"]["device"] = "cuda"
    assert actual == state_digest(expected) and state_digest(value) == before
    assert report._loss_close(.000929814821574837, .0009298176959466499)
    assert not report._loss_close(.000929814821574837, .001)


def test_evaluation_complete_set_and_actual_boundaries(case, tmp_path):
    cp, audit, inputs = checkpoint(case)
    trainer = FactorTrainer.from_checkpoint(case["checkpoint"], device="cpu")
    with _preserve_process_state():
        records = report.normalize([diagnose_fork(trainer.model, trainer.world, trainer.h,
            condition=condition, steps=48, dt=.25, device="cpu") for condition in report.CONDITIONS])
    path = tmp_path / "evaluation.json"
    value = dict(format="factors-evaluation-v1", branch=case["item"], protocol_sha256="a" * 64,
        source_unchanged=True, **report._binding(cp), records=records)
    save(path, value)
    result, history = report._evaluation(path, case["item"], cp, case["plan"], "a" * 64, audit, inputs)
    assert set(result) == set(report.CONDITIONS) and set(history) == set(report.CONDITIONS)
    for change, code in ((lambda v: v.update(state_hash="0" * 64), "EVALUATION_PROVENANCE"),
            (lambda v: v["records"].pop(), "FORK_COVERAGE"),
            (lambda v: v["records"].append(v["records"][0]), "EXPECTED_FORK"),
            (lambda v: v["records"][0].update(start_hidden_hash="0" * 64), "ACTUAL_EVALUATION_BOUNDARY"),
            (lambda v: v["records"][0]["primary"].update(clean_completion_rate=.987654), "RATE_DENOMINATOR"),
            (lambda v: v["records"][0]["opportunities"].update(model_action_changes=1), "NO_INFERENCE_GATE")):
        broken = deepcopy(value)
        change(broken)
        save(path, broken)
        with pytest.raises(report.AuditError, match=code):
            report._evaluation(path, case["item"], cp, case["plan"], "a" * 64, report._Checks(), {})


@pytest.fixture
def formal(tmp_path):
    from experiments.factors import study
    # prepare creates only protocol/command metadata here, never a training run.
    evidence, raw = tmp_path / "evidence", tmp_path / "raw"
    study.prepare(evidence, raw)
    protocol = evidence / "study-protocol-v1.json"
    return protocol, raw, json.loads(protocol.read_text(encoding="utf-8"))


def test_formal_protocol_covers_exact_factor_cross_and_shared_origins(formal):
    protocol, raw, _ = formal
    plan, branches, origins = report._plan(protocol, raw, report._Checks(), {})
    assert len(branches) == 54 and len(origins) == 3
    assert set(branches) == {(seed, mode, strength) for seed in report.SEEDS for mode in report.MODES for strength in report.IDLE_LAMBDAS}
    assert plan["target_ticks"] == 65536


@pytest.mark.parametrize("change,code", [
    (lambda v: v.update(target_ticks=64), "PROTOCOL"),
    (lambda v: v.update(evaluation_steps=16), "PROTOCOL"),
    (lambda v: v["branches"].pop(), "BRANCH_COUNT"),
    (lambda v: v["origins"].pop(), "ORIGIN_COUNT"),
    (lambda v: v["branches"][0].update(source_sha256="0" * 64), "BRANCH_BINDING"),
    (lambda v: v["branches"][0].update(idle_label="chosen_after_results"), "BRANCH_BINDING"),
    (lambda v: v["branches"][0].update(mode="unknown"), "BRANCH_ID"),
    (lambda v: v["branches"][0].update(output="C:/foreign"), "BRANCH_OUTPUT"),
    (lambda v: v["sources"].pop(next(path for path in v["sources"] if path.endswith("structure.py") and "factors" in path)), "FROZEN_SOURCE_COVERAGE"),
])
def test_formal_protocol_rejects_rebound_or_partial_design(formal, change, code):
    protocol, raw, plan = formal
    change(plan)
    save(protocol, plan)
    with pytest.raises(report.AuditError, match=code):
        report._plan(protocol, raw, report._Checks(), {})


def test_paired_contrasts_use_one_shared_baseline_per_seed():
    records, history = {}, {}
    for seed in report.SEEDS:
        for condition in report.CONDITIONS:
            records[(seed, "baseline", None, condition)] = dict(score=seed * 100.)
            for index, mode in enumerate(report.MODES):
                for strength in report.IDLE_LAMBDAS:
                    records[(seed, mode, strength, condition)] = dict(score=seed * 100. + index + strength)
                    history[(seed, mode, strength, condition)] = {("completion", "[0,1)"): dict(margin=index + strength)}
    contrasts, penalties, changes, idle_contrasts, idle_penalties = report._contrasts(records, history)
    assert len(changes) == 3 * 5 * 6 * 3
    assert len(contrasts) == 3 * 5 * 3 * len(report.STRUCTURAL_CONTRASTS)
    assert all(row["value"] == pytest.approx(row["idle_lambda"]) for row in penalties)
    assert all(row["value"] == pytest.approx(row["idle_lambda"]) for row in idle_penalties)
    assert len(idle_contrasts) == len(contrasts)
    summaries = report._summaries(penalties, ("mode", "idle_lambda", "condition", "metric"))
    assert all(row["n_total"] == row["n"] == 3 and row["sample_sd"] == pytest.approx(0., abs=1e-12) for row in summaries)


def test_public_report_rejects_small_incomplete_protocol_without_output(toy, tmp_path):
    protocol, output = tmp_path / "toy.json", tmp_path / "published"
    save(protocol, dict(format="factors-study-protocol-v1", target_ticks=16))
    with pytest.raises(SystemExit) as error:
        report.main(["--protocol", str(protocol), "--root", str(toy["root"]), "--output", str(output), "--workers", "1"])
    assert error.value.code == 1 and not output.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_training_runtime_and_cpu_semantic_audit(toy, tmp_path):
    plan = {**toy["plan"], "training_device": "cuda"}
    item = {**toy["branches"][0], "output": str(tmp_path / "cuda-branch")}
    folder = Path(item["output"])
    with _preserve_process_state(), redirect_stdout(io.StringIO()):
        inherited = RewireTrainer.from_checkpoint(toy["source"], device="cuda", strict_sources=True)
        trainer = FactorTrainer(inherited, FactorConfig(mode=item["mode"], idle_lambda=0.),
            origin=dict(checkpoint=str(toy["source"]), checkpoint_sha256=report.file_sha256(toy["source"])))
        run_to_target(trainer, 16, folder, checkpoint_every=0)
        digest = state_digest(trainer.semantic_state())
    save(folder.with_name(folder.name + ".launch-start.json"),
        dict(format="factors-launch-start-v1", branch=item, protocol_sha256="a" * 64))
    save(folder / "launch-complete.json", dict(format="factors-launch-complete-v1", branch=item,
        protocol_sha256="a" * 64, target=16, stage_ticks=16, hidden_resets=0,
        checkpoint_sha256=report.file_sha256(folder / "checkpoint.pt"), semantic_hash=digest))
    origin = report._source(toy["source"], "cuda", report._Checks(), {})
    cp = report._checkpoint(item, origin, plan, "a" * 64, report._Checks(), {})
    assert cp["training_device_normalized_state_hash"] == digest
    assert cp["state_hash"] != digest
