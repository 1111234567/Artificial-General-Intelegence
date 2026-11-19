"""Synthetic study-launch and full-state evaluation checks; no formal runs."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import TrainConfig, Trainer, Variant, rng_state, state_digest
from experiments.credit.run import file_sha256
from experiments.integration.diagnose import diagnose_fork
from experiments.loops import extension_study
from experiments.loops.model import ARMS
from experiments.loops.train import LoopTrainer


def _json(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def extension_case(tmp_path, request):
    arm = getattr(request, "param", "interaction")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with _preserve_process_state():
        base = Trainer(Variant("extension-study-fixture", width=6, policy="online"),
                       TrainConfig(seed=173, batch_size=2, n_jobs=2, tbptt=4, updates=2,
                                   device="cpu", train_dts=(.25,), replay_capacity=2,
                                   replay_block_ticks=8, replay_burnin=4, log_every=2))
        base.run_updates(2)
        source = tmp_path / "A.pt"
        base.save_checkpoint(source)
        source_code = tmp_path / "frozen-source.py"
        source_code.write_text("# frozen synthetic source\n", encoding="utf-8")
        audit = tmp_path / "completed-audit.json"
        _json(audit, dict(complete=True, audit=dict(status="passed", errors=[]),
                          counts=dict(sources=9, jobs=27, forks=540, structural_summaries=9)))
        item = dict(id=f"seed_173_{arm}", seed=173, arm=arm, source=str(source),
                    source_sha256=file_sha256(source), output=str(tmp_path / "training"))
        plan = dict(format="loops-extension-protocol-v1",
                    sources={str(source_code): file_sha256(source_code)},
                    completed_existing_audit=str(audit), completed_existing_audit_sha256=file_sha256(audit),
                    branches=[item], training_device="cpu", evaluation_device="cpu", target_ticks=4,
                    chunk_ticks=4, conditions=["iid", "long_wait", "longer", "sparse", "long_sparse"],
                    evaluation_steps=48, diagnostic_conditions=["longer", "long_sparse"],
                    solver_refinement_max_step=.125, dt=.25)
        protocol = tmp_path / "protocol.json"
        _json(protocol, plan)
    try:
        yield dict(protocol=protocol, plan=plan, item=item, source=source,
                   source_code=source_code, audit=audit, tmp_path=tmp_path)
    finally:
        torch.set_num_threads(threads)


def _checkpoint(case, *, marker=True):
    with _preserve_process_state():
        item, plan = case["item"], case["plan"]
        base = Trainer.from_checkpoint(case["source"], device=plan["training_device"])
        trainer = LoopTrainer(base, item["arm"], chunk_ticks=plan["chunk_ticks"],
                              origin=dict(checkpoint=item["source"], checkpoint_sha256=item["source_sha256"]))
        trainer.run_updates(plan["target_ticks"] // plan["chunk_ticks"])
        output = Path(item["output"])
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / "checkpoint.pt"
        trainer.save_checkpoint(checkpoint)
        if marker:
            _json(output / "launch-complete.json", dict(format="loops-launch-complete-v1",
                  protocol_sha256=file_sha256(case["protocol"]), branch=item,
                  stage_ticks=trainer.stage_ticks, target=plan["target_ticks"],
                  checkpoint_sha256=file_sha256(checkpoint),
                  semantic_hash=state_digest(trainer.semantic_state()), hidden_resets=trainer.hidden_resets))
    return checkpoint


def _start_binding(case):
    output = Path(case["item"]["output"])
    path = output.with_name(output.name + ".launch-start.json")
    value = dict(format="loops-launch-start-v1", protocol_sha256=file_sha256(case["protocol"]),
                 branch=case["item"])
    _json(path, value)
    return path, value


@pytest.mark.parametrize("extension_case", ARMS, indirect=True)
def test_evaluate_full_state_treatments_and_exact_original_baseline(extension_case, monkeypatch):
    case = extension_case
    checkpoint = _checkpoint(case)
    protocol, plan, item = case["protocol"], case["plan"], case["item"]
    output = case["tmp_path"] / "evaluation.json"
    preserved_bytes = {path: path.read_bytes() for path in
                       (protocol, checkpoint, case["source"], case["source_code"], case["audit"])}
    original_load = LoopTrainer.from_checkpoint
    loaded = []

    def track_load(*args, **kwargs):
        trainer = original_load(*args, **kwargs)
        loaded.append((trainer, state_digest(trainer.state_dict())))
        return trainer

    monkeypatch.setattr(extension_study.LoopTrainer, "from_checkpoint", track_load)
    before_rng = state_digest(rng_state())
    extension_study.evaluate(protocol, item["id"], checkpoint, output)
    assert before_rng == state_digest(rng_state())
    assert len(loaded) == 1
    trainer, initial_state = loaded[0]
    with _preserve_process_state():
        # semantic_state includes global RNG; restore this checkpoint's RNG
        # before comparing its complete persisted continuation representation.
        reference = original_load(checkpoint, device="cpu")
        assert initial_state == state_digest(trainer.state_dict())
        assert state_digest(trainer.semantic_state()) == state_digest(reference.semantic_state())
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["source_unchanged"] and result["branch"] == item
    assert result["checkpoint_sha256"] == file_sha256(checkpoint)
    assert result["protocol_sha256"] == file_sha256(protocol)
    expected = dict(none=5, solver_half=2)
    if item["arm"] != "baseline":
        expected["branch_zero"] = 2
    assert Counter(row["extension_intervention"] for row in result["records"]) == Counter(expected)
    assert len(result["records"]) == (7 if item["arm"] == "baseline" else 9)
    for condition in plan["conditions"]:
        rows = [row for row in result["records"] if row["condition"] == condition]
        baseline = diagnose_fork(trainer.model, trainer.world, trainer.h, condition=condition,
                                 steps=plan["evaluation_steps"], dt=plan["dt"], device="cpu")
        plain = {key: value for key, value in rows[0].items() if key != "extension_intervention"}
        assert plain == json.loads(json.dumps(baseline, allow_nan=False))
        for row in rows:
            assert row["action_hash"] == row["model_action_hash"]
            assert row["world_resets"] == row["hidden_resets"] == row["parameter_updates"] == 0
            for key in ("start_world_state_hash", "start_hidden_hash", "start_scheduler_hash",
                        "end_scheduler_hash", "end_event_hash", "primary_eligible_issued"):
                assert row[key] == baseline[key]
            for cohort in ("primary", "inherited", "carryover_contract"):
                assert row[cohort]["request_fingerprint"] == baseline[cohort]["request_fingerprint"]
            if row["extension_intervention"] == "solver_half":
                expected_model = deepcopy(trainer.model)
                expected_model.max_step = .125
                assert row["model"] == expected_model.config()
            elif row["extension_intervention"] == "branch_zero":
                expected_model = deepcopy(trainer.model)
                with torch.no_grad():
                    expected_model.branch.zero_()
                assert row["parameter_hash"] == state_digest(expected_model.state_dict())
    assert all(path.read_bytes() == before for path, before in preserved_bytes.items())
    assert before_rng == state_digest(rng_state())


def test_train_launcher_completion_and_resume_are_bound_to_protocol(extension_case, monkeypatch):
    case = extension_case
    calls = []

    def training_command(argv):
        output = Path(case["item"]["output"])
        binding = output.with_name(output.name + ".launch-start.json")
        assert json.loads(binding.read_text(encoding="utf-8")) == dict(
            format="loops-launch-start-v1", protocol_sha256=file_sha256(case["protocol"]),
            branch=case["item"])
        calls.append(argv)
        _checkpoint(case, marker=False)

    monkeypatch.setattr(extension_study, "train_main", training_command)
    extension_study.train(case["protocol"], case["item"]["id"])
    assert len(calls) == 1
    command = calls[0]
    for flag, expected in (("--source", case["item"]["source"]), ("--arm", "interaction"),
                           ("--target-ticks", "4"), ("--device", "cpu")):
        assert command[command.index(flag) + 1] == expected
    assert "--resume" not in command
    marker_path = Path(case["item"]["output"]) / "launch-complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["branch"] == case["item"] and marker["stage_ticks"] == 4
    assert marker["checkpoint_sha256"] == file_sha256(marker_path.parent / "checkpoint.pt")
    extension_study.train(case["protocol"], case["item"]["id"])
    assert len(calls) == 1
    marker_path.unlink()
    extension_study.train(case["protocol"], case["item"]["id"])
    assert len(calls) == 2 and "--resume" in calls[-1]


@pytest.mark.parametrize("target", ["source_code", "audit", "source"])
def test_read_plan_rejects_modified_bound_evidence(extension_case, target):
    case = extension_case
    path = case[target]
    path.write_bytes(path.read_bytes() + b"\nchanged")
    with pytest.raises(ValueError, match="changed"):
        extension_study.read_plan(case["protocol"], case["item"]["id"])


@pytest.mark.parametrize("target", ["protocol", "source_code", "audit", "source", "checkpoint"])
def test_evaluation_rejects_midrun_mutation_without_publishing(extension_case, monkeypatch, target):
    case = extension_case
    case["plan"]["evaluation_steps"] = 1
    _json(case["protocol"], case["plan"])
    checkpoint = _checkpoint(case)
    original = extension_study.diagnose_fork
    changed = False

    def mutate(*args, **kwargs):
        nonlocal changed
        record = original(*args, **kwargs)
        if not changed:
            path = checkpoint if target == "checkpoint" else case[target]
            path.write_bytes(path.read_bytes() + b"\n")
            changed = True
        return record

    monkeypatch.setattr(extension_study, "diagnose_fork", mutate)
    output = case["tmp_path"] / "rejected.json"
    before_rng = state_digest(rng_state())
    with pytest.raises((AssertionError, ValueError), match="changed"):
        extension_study.evaluate(case["protocol"], case["item"]["id"], checkpoint, output)
    assert changed and not output.exists()
    assert before_rng == state_digest(rng_state())


@pytest.mark.parametrize("operation", ["train", "evaluate"])
def test_rejects_completion_marker_from_another_seed(extension_case, operation):
    case = extension_case
    checkpoint = _checkpoint(case)
    marker_path = checkpoint.parent / "launch-complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["branch"]["id"] = "seed_174_interaction"
    marker["branch"]["seed"] = 174
    _json(marker_path, marker)
    with pytest.raises(ValueError, match="provenance|completion|branch"):
        if operation == "train":
            extension_study.train(case["protocol"], case["item"]["id"])
        else:
            extension_study.evaluate(case["protocol"], case["item"]["id"], checkpoint,
                                     case["tmp_path"] / "rejected.json")


def test_rejects_checkpoint_origin_from_another_source(extension_case):
    case = extension_case
    checkpoint = _checkpoint(case)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state["origin"]["checkpoint_sha256"] = "0" * 64
    torch.save(state, checkpoint)
    marker_path = checkpoint.parent / "launch-complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["checkpoint_sha256"] = file_sha256(checkpoint)
    _json(marker_path, marker)
    with pytest.raises(ValueError, match="origin|provenance|source"):
        extension_study.evaluate(case["protocol"], case["item"]["id"], checkpoint,
                                 case["tmp_path"] / "rejected.json")


@pytest.mark.parametrize("target", ["origin_sha", "origin_path", "chunk_ticks", "seed", "arm",
                                    "duration_range", "gap_range", "live_regime", "hidden_resets", "past_target", "partial_chunk"])
def test_foreign_partial_checkpoint_is_rejected_before_training(extension_case, monkeypatch, target):
    case = extension_case
    checkpoint = _checkpoint(case, marker=False)
    binding_path, _ = _start_binding(case)
    binding_bytes = binding_path.read_bytes()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    continuation = state["credit"]["continuation"]
    if target == "origin_sha":
        state["origin"]["checkpoint_sha256"] = "0" * 64
    elif target == "origin_path":
        state["origin"]["checkpoint"] = str(case["tmp_path"] / "another-A.pt")
    elif target == "chunk_ticks":
        state["credit"]["credit_config"]["chunk_ticks"] = 8
    elif target == "seed":
        continuation["config"]["seed"] = 174
    elif target == "arm":
        state["arm"] = "source_square"
    elif target in {"duration_range", "gap_range"}:
        state["credit"]["credit_config"][target] = (1., 4.)
    elif target == "live_regime":
        with _preserve_process_state():
            changed = LoopTrainer.from_checkpoint(checkpoint, device="cpu")
            changed.change_regime((1., 4.), (.5, 2.), "off-protocol")
            assert changed.credit_config.duration_range == (4., 8.)
            assert changed.credit_config.gap_range == (4., 12.)
            state = changed.state_dict()
    elif target == "hidden_resets":
        continuation["hidden_resets"] = 1
    elif target == "past_target":
        continuation["ticks"] += 4
    elif target == "partial_chunk":
        continuation["ticks"] -= 1
    torch.save(state, checkpoint)
    calls = []
    monkeypatch.setattr(extension_study, "train_main", lambda command: calls.append(command))
    before_rng = state_digest(rng_state())
    with pytest.raises(ValueError, match="origin|source|provenance|chunk|protocol|configuration|continuous"):
        extension_study.train(case["protocol"], case["item"]["id"])
    assert calls == []
    assert not (checkpoint.parent / "launch-complete.json").exists()
    assert binding_path.read_bytes() == binding_bytes
    assert before_rng == state_digest(rng_state())


@pytest.mark.parametrize("target", ["missing", "protocol_sha256", "branch", "format"])
def test_missing_or_wrong_start_binding_refuses_partial_checkpoint_before_loading(
        extension_case, monkeypatch, target):
    case = extension_case
    checkpoint = _checkpoint(case, marker=False)
    if target != "missing":
        binding_path, binding = _start_binding(case)
        if target == "protocol_sha256":
            binding[target] = "0" * 64
        elif target == "branch":
            binding[target]["seed"] = 174
        else:
            binding[target] = "another-launch-format"
        _json(binding_path, binding)
    calls = []
    monkeypatch.setattr(extension_study, "train_main", lambda command: calls.append(command))

    def no_load(*args, **kwargs):
        pytest.fail("invalid launch binding must reject before loading/training")

    monkeypatch.setattr(extension_study.LoopTrainer, "from_checkpoint", no_load)
    before_bytes = checkpoint.read_bytes()
    with pytest.raises(ValueError, match="provenance|protocol|binding"):
        extension_study.train(case["protocol"], case["item"]["id"])
    assert calls == [] and checkpoint.read_bytes() == before_bytes
    assert not (checkpoint.parent / "launch-complete.json").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_checkpoint_evaluates_on_cpu_with_only_device_metadata_normalized(extension_case):
    case = extension_case
    case["plan"].update(training_device="cuda", conditions=["iid"], diagnostic_conditions=[],
                        evaluation_steps=8)
    _json(case["protocol"], case["plan"])
    caller_rng = state_digest(rng_state())
    checkpoint = _checkpoint(case)
    assert caller_rng == state_digest(rng_state())
    before_bytes = checkpoint.read_bytes()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    marker = json.loads((checkpoint.parent / "launch-complete.json").read_text(encoding="utf-8"))
    assert saved["credit"]["continuation"]["config"]["device"] == "cuda"
    assert len(saved["credit"]["continuation"]["rng"]["torch_cuda"]) > 0
    with _preserve_process_state():
        migrated = LoopTrainer.from_checkpoint(checkpoint, device="cpu")
        assert migrated.device.type == "cpu" and migrated.h.device.type == "cpu"
        assert migrated.model.recurrent.device.type == "cpu"
        assert state_digest(rng_state()) == state_digest(saved["credit"]["continuation"]["rng"])
        semantic = migrated.semantic_state()
        raw_hash = state_digest(semantic)
        assert raw_hash != marker["semantic_hash"]
        semantic["credit"]["continuation"]["config"]["device"] = "cuda"
        assert state_digest(semantic) == marker["semantic_hash"]
        assert extension_study.comparable_semantic_hash(migrated, "cuda") == marker["semantic_hash"]
    output = case["tmp_path"] / "cuda-to-cpu-evaluation.json"
    extension_study.evaluate(case["protocol"], case["item"]["id"], checkpoint, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["evaluation_device"] == "cpu" and len(result["records"]) == 1
    assert result["checkpoint_state_hash"] == raw_hash
    assert result["training_device_normalized_state_hash"] == marker["semantic_hash"]
    assert caller_rng == state_digest(rng_state())
    assert checkpoint.read_bytes() == before_bytes


@pytest.mark.parametrize("target", ["stage_ticks", "hidden_resets"])
def test_incomplete_or_reset_completion_marker_is_rejected(extension_case, target):
    case = extension_case
    checkpoint = _checkpoint(case)
    marker_path = checkpoint.parent / "launch-complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker[target] = 0 if target == "stage_ticks" else 1
    _json(marker_path, marker)
    with pytest.raises(ValueError, match="completion|provenance|continuous|reset|ticks"):
        extension_study.evaluate(case["protocol"], case["item"]["id"], checkpoint,
                                 case["tmp_path"] / "rejected.json")


@pytest.mark.parametrize("audit", [dict(),
    dict(complete=False, audit=dict(status="passed", errors=[]),
         counts=dict(sources=9, jobs=27, forks=540, structural_summaries=9)),
    dict(complete=True, audit=dict(status="failed", errors=["incomplete cohort"]),
         counts=dict(sources=9, jobs=27, forks=540, structural_summaries=9)),
    dict(complete=True, audit=dict(status="passed", errors=[]),
         counts=dict(sources=9, jobs=26, forks=522, structural_summaries=9)),
])
def test_prepare_requires_complete_prior_audit_before_freezing(tmp_path, audit):
    audit_path = tmp_path / "audit.json"
    _json(audit_path, audit)
    folder = tmp_path / "protocol"
    with pytest.raises(ValueError, match="audit"):
        extension_study.prepare(folder, tmp_path / "raw", audit_path)
    assert not (folder / "extension-protocol-v1.json").exists()
