"""Causal fork, cohort denominator, hidden reset and checkpoint read-only checks."""

import hashlib
import json

import pytest
import torch

from experiments.continuity.evaluate import (
    CONDITIONS, evaluate_checkpoint, evaluate_fork, main,
)
from experiments.continuity.model import AgentModel
from experiments.continuity.run import TrainConfig, Trainer, Variant, rng_state, state_digest
from experiments.continuity.world import NEW, OnlineWorld


def _wait_model(world, width=8):
    model = AgentModel(input_size=world.input_size, n_jobs=world.n_jobs, width=width)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


def _world(**kwargs):
    values = dict(seed=31, batch_size=3, n_jobs=2, dt=0.25, cancel=0, replace=0)
    values.update(kwargs)
    return OnlineWorld(**values)


def _first_started_world():
    world = _world(batch_size=1, active_jobs=1, duration_range=(1.0, 1.0),
                   gap_range=(20.0, 20.0), window_range=(2.0, 2.0))
    for _ in range(16):
        world.observe()
        first = bool((world.event_kind == NEW).any())
        world.act(world.oracle())
        if first:
            return world
    raise AssertionError("first command did not arrive")


def _tiny_checkpoint(tmp_path, *, adaptation=False):
    config = TrainConfig(batch_size=2, n_jobs=2, tbptt=4, updates=3,
                         train_dts=(0.25,), replay_block_ticks=8, replay_burnin=4,
                         replay_capacity=2, log_every=100, device="cpu")
    trainer = Trainer(Variant("tiny", width=8, policy="online"), config)
    trainer.run_updates(3)
    if adaptation:
        trainer.begin_adaptation("frozen")
        trainer.run_updates(1)
    path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(path)
    return trainer, path


def test_fork_preserves_inherited_pending_receipt_clock_and_source_state():
    world = _first_started_world()
    model = _wait_model(world)
    model.train()
    hidden = torch.ones((world.batch_size, model.width))
    before = state_digest({"world": world.state_dict(), "model": model.state_dict(), "h": hidden})
    expected = OnlineWorld.from_state_dict(world.state_dict())
    expected.set_regime(**CONDITIONS["iid"], name="evaluation/iid", dt=0.25)
    observation = expected.observe()
    assert observation.reshape(1, 2, 24)[0, 0, 22] == 1

    result = evaluate_fork(model, world, hidden, steps=32)

    assert result["first_observation_hash"] == state_digest({"observation": observation})
    assert result["fork_tick"] == world.tick
    assert result["fork_time"] == world.now
    assert result["end_tick"] == world.tick + 32
    assert result["end_time"] == world.now + 8
    assert result["world_resets"] == result["parameter_updates"] == 0
    assert result["inherited"]["initial_eligible_pending"] == 1
    assert result["inherited"]["matured"] == 1
    assert result["primary"]["matured"] == result["primary_eligible_issued"] == 0
    assert result["primary"]["clean_completion_rate"] is None
    assert model.training
    assert before == state_digest({"world": world.state_dict(), "model": model.state_dict(), "h": hidden})


def test_hidden_reset_has_same_exogenous_request_cohorts_even_without_begin():
    world = _world(batch_size=4)
    model = _wait_model(world)
    hidden = torch.ones((world.batch_size, model.width))
    baseline = evaluate_fork(model, world, hidden, steps=96, boundary_interval=8)
    reset = evaluate_fork(model, world, hidden, steps=96, boundary_interval=8, reset_hidden=True)

    for key in ("start_world_state_hash", "start_hidden_hash", "first_observation_hash",
                "end_event_hash", "observation_hash", "action_hash", "boundaries"):
        assert baseline[key] == reset[key]
    for name in ("primary", "inherited", "carryover_contract", "boundary_crossing",
                 "inherited_boundary_crossing", "carryover_boundary_crossing"):
        assert baseline[name] == reset[name]
    assert baseline["boundary_crossing"]["matured"] > 0
    assert baseline["boundary_crossing"]["completed"] == 0
    assert baseline["hidden_resets"] == 0
    assert reset["hidden_resets"] == 11
    assert reset["boundaries"][0]["tick"] == 8
    assert reset["state_rms"]["final"] == 0
    assert baseline["state_rms"]["final"] > 0
    assert baseline["end_hidden_hash"] != reset["end_hidden_hash"]


def test_completed_but_unmatured_windows_are_censored_not_counted_as_failures():
    world = _world(batch_size=16)
    model = _wait_model(world)
    hidden = model.initial_state(world.batch_size)
    result = evaluate_fork(model, world, hidden, steps=8, teacher_forced=True)

    assert result["primary_eligible_issued"] > 0
    assert result["primary"]["matured"] == 0
    assert result["primary"]["completion_rate"] is None
    assert result["censored"]["primary"]["count"] == result["primary_eligible_issued"]
    assert result["inherited"]["initial_eligible_pending"] == 0


@pytest.mark.parametrize("condition", tuple(CONDITIONS))
def test_teacher_continuation_completes_all_matured_postfork_requests(condition):
    world = _world(batch_size=4)
    model = _wait_model(world)
    result = evaluate_fork(model, world, model.initial_state(world.batch_size),
                           condition=condition, steps=160, teacher_forced=True)

    assert result["primary"]["matured"] > 0
    assert result["primary"]["clean_completion_rate"] == 1.0
    assert result["primary"]["completed"] == result["primary"]["matured"]
    assert result["post_fork_counters"]["rejected"] == 0
    assert result["primary_eligible_issued"] == (
        result["primary"]["matured"] + result["censored"]["primary"]["count"])
    assert result["content_probe"]["count"] > 0
    assert result["content_probe"]["accuracy"] <= 1.0
    assert result["action_policy"] == "teacher"
    assert result["regime"]["duration_range"] == CONDITIONS[condition]["duration_range"]


def test_existing_inflight_contracts_do_not_change_when_fork_regime_changes():
    world = _first_started_world()
    model = _wait_model(world)
    hidden = model.initial_state(world.batch_size)
    result = evaluate_fork(model, world, hidden, condition="longer", steps=32,
                           teacher_forced=True)
    assert result["regime"]["duration_range"] == (8.0, 16.0)
    assert result["inherited"]["matured"] == 1
    assert result["inherited"]["clean_completion_rate"] == 1.0
    assert result["primary"]["matured"] == 0


def test_scheduled_prior_regime_replacements_have_separate_carryover_denominator():
    world = _world(batch_size=8, duration_range=(4.0, 4.0),
                   gap_range=(0.5, 0.5), window_range=(2.0, 2.0), replace=1)
    for _ in range(16):
        world.observe()
        has_command = bool((world.event_kind == NEW).any())
        world.act(world.oracle())
        if has_command:
            break
    model = _wait_model(world)
    result = evaluate_fork(model, world, model.initial_state(world.batch_size),
                           steps=96, teacher_forced=True, boundary_interval=8)

    assert result["inherited"]["initial_eligible_pending"] == 0
    assert result["carryover_contract_eligible_issued"] > 0
    assert result["carryover_contract"]["matured"] > 0
    assert result["carryover_contract"]["clean_completion_rate"] == 1.0
    assert result["primary"]["clean_completion_rate"] == 1.0
    assert result["carryover_contract"]["matured"] + result["censored"]["carryover_contract"]["count"] == (
        result["carryover_contract_eligible_issued"])
    assert result["primary_eligible_issued"] + result["carryover_contract_eligible_issued"] == (
        result["post_fork_counters"]["eligible"])
    assert result["content_probe"]["count"] == (
        result["primary"]["completed"] + result["censored"]["primary"]["completed"])


def test_checkpoint_evaluation_preserves_file_rngs_weights_and_full_continuation(tmp_path):
    trainer, path = _tiny_checkpoint(tmp_path)
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    original_runtime = state_digest(trainer.semantic_state())
    rng_before = state_digest(rng_state())
    settings_before = (torch.are_deterministic_algorithms_enabled(),
                       torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32)

    result = evaluate_checkpoint(path, conditions=("iid", "long_wait"), steps=64,
                                 reset_interval=16, content_probe=True)

    assert result["checkpoint_sha256"] == file_hash == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["checkpoint_state_hash"] == original_runtime
    assert state_digest(trainer.semantic_state()) == original_runtime
    assert rng_before == state_digest(rng_state())
    assert settings_before == (torch.are_deterministic_algorithms_enabled(),
                               torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32)
    assert len(result["records"]) == 5
    baseline, reset, probe, long_wait, long_probe = result["records"]
    assert baseline["end_event_hash"] == reset["end_event_hash"] == probe["end_event_hash"]
    assert baseline["primary"]["request_fingerprint"] == reset["primary"]["request_fingerprint"]
    assert baseline["parameter_hash"] == long_wait["parameter_hash"] == long_probe["parameter_hash"]
    assert result["training_config"]["batch_size"] == 2
    assert result["model"]["state_scalars"] == 8
    assert result["source_hashes"]["evaluate.py"]


def test_stage_b_checkpoint_keeps_stage_identity_and_excludes_lifetime_counts(tmp_path):
    trainer, path = _tiny_checkpoint(tmp_path, adaptation=True)
    result = evaluate_checkpoint(path, conditions=("iid", "long_wait"), steps=96)
    assert result["training_stage"] == "B"
    assert result["adaptation_mode"] == "frozen"
    assert result["training_ticks"] == trainer.ticks
    assert result["records"][0]["fork_tick"] == trainer.world.tick
    for record in result["records"]:
        assert record["primary"]["matured"] + record["censored"]["primary"]["count"] == (
            record["primary_eligible_issued"])
        assert record["primary_eligible_issued"] + record["carryover_contract_eligible_issued"] == (
            record["post_fork_counters"]["eligible"])


def test_cli_saves_compact_json_from_full_checkpoint(tmp_path):
    _, checkpoint = _tiny_checkpoint(tmp_path)
    output = tmp_path / "evaluation.json"
    main(["--checkpoint", str(checkpoint), "--output", str(output), "--device", "cpu",
          "--conditions", "iid", "--steps", "16", "--content-probe"])
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert len(saved["evaluations"]) == 1
    assert len(saved["evaluations"][0]["records"]) == 3
    assert "world" not in saved["evaluations"][0]


def test_invalid_forks_and_conditions_are_rejected(tmp_path):
    world = _world()
    model = _wait_model(world)
    hidden = model.initial_state(world.batch_size)
    for bad in (dict(steps=0), dict(dt=0), dict(boundary_interval=0), dict(condition="unknown")):
        with pytest.raises(ValueError):
            evaluate_fork(model, world, hidden, **bad)
    world.observe()
    with pytest.raises(ValueError, match="completed action"):
        evaluate_fork(model, world, hidden)
    with pytest.raises(ValueError, match="distinct"):
        evaluate_checkpoint(tmp_path / "absent.pt", conditions=("iid", "iid"))
