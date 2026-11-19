"""Own-state causality, full-fork identity and honest cohort/idle denominators."""

import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn

from experiments.continuity.evaluate import CONDITIONS as FROZEN_CONDITIONS, evaluate_fork
from experiments.continuity.model import AgentModel
from experiments.continuity.run import TrainConfig, Trainer, Variant, rng_state, state_digest
from experiments.continuity.world import NEW, OnlineWorld
from experiments.integration.diagnose import (
    CONDITIONS, FIRST_ERROR_PRIORITY, INTERVENTIONS, diagnose_checkpoint, diagnose_fork, main,
)
import experiments.integration.diagnose as diagnosis


def _world(**kwargs):
    values = dict(seed=31, batch_size=3, n_jobs=2, dt=.25)
    values.update(kwargs)
    return OnlineWorld(**values)


def _wait_model(world, width=8):
    model = AgentModel(input_size=world.input_size, n_jobs=world.n_jobs, width=width)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


class PublicController(nn.Module):
    """Test controller using only public command/receipt features and its own h.

    Its retained old FINISH content remains readable while WAIT wins after a
    terminal receipt. Removing exactly that receipt produces repeated FINISH
    attempts, giving the observer a causal, fully inspectable test case.
    """

    def __init__(self, world, *, finish_fraction=1.0, wrong_payload=False, wrong_begin=False):
        super().__init__()
        self.input_size, self.n_jobs, self.n_actions = world.input_size, world.n_jobs, 11
        self.width = self.n_jobs * 5
        self.finish_fraction = finish_fraction
        self.wrong_payload, self.wrong_begin = wrong_payload, wrong_begin
        self.anchor = nn.Parameter(torch.zeros(()))

    def initial_state(self, batch_size):
        return torch.zeros(batch_size, self.width)

    def step(self, observation, state, dt):
        features = observation.reshape(-1, self.n_jobs, 24)
        memory = state.clone().reshape(-1, self.n_jobs, 5)
        # version, payload, duration, own elapsed time, internally active
        memory[..., 3] += float(dt) * memory[..., 4]
        command = (features[..., 0] == 1) | (features[..., 1] == 1)
        cancel = features[..., 2] == 1
        finish_receipt = (features[..., 14:22].sum(-1) > 0) & (features[..., 22] == 1)
        memory[..., 4][cancel | finish_receipt] = 0
        memory[..., 0][command] = features[..., 3:5].argmax(-1)[command].float()
        memory[..., 1][command] = features[..., 5:9].argmax(-1)[command].float()
        memory[..., 2][command] = features[..., 9][command]
        memory[..., 3][command] = 0
        memory[..., 4][command] = 1
        version = memory[..., 0].long()
        payload = (memory[..., 1].long() + int(self.wrong_payload)) % 4
        begin_version = (version + int(self.wrong_begin)) % 2
        finish_code = 3 + 4 * version + payload
        logits = features.new_full((*features.shape[:2], 11), -8)
        logits[..., 0] = 4
        logits.scatter_(-1, (1 + begin_version)[..., None], 0)
        logits.scatter_(-1, finish_code[..., None], 1)
        due = (memory[..., 4] == 1) & (memory[..., 3] >= self.finish_fraction * memory[..., 2] - 1e-6)
        action = torch.where(command, 1 + begin_version, torch.where(due, finish_code, 0))
        logits.scatter_(-1, action[..., None], 8)
        return logits, memory.flatten(1)

    def config(self):
        return dict(input_size=self.input_size, n_jobs=self.n_jobs, n_actions=11,
                    width=self.width, kind="public_test_controller")

    def parameter_report(self):
        return dict(total_parameters=1, state_scalars=self.width)


class TransitionController(PublicController):
    """Choose cue/readiness readability independently through public test state."""

    def __init__(self, world, *, cue_correct, legal_correct):
        super().__init__(world)
        self.cue_readable, self.legal_readable = cue_correct, legal_correct

    def step(self, observation, state, dt):
        logits, memory = super().step(observation, state, dt)
        features = observation.reshape(-1, self.n_jobs, 24)
        command = (features[..., 0] == 1) | (features[..., 1] == 1)
        original_finish = logits.argmax(-1) >= 3
        public_memory = memory.reshape(-1, self.n_jobs, 5)
        version, payload = public_memory[..., 0].long(), public_memory[..., 1].long()
        if self.cue_readable == "replacement_only":
            cue_readable = features[..., 1] == 1
        else:
            cue_readable = torch.full_like(command, bool(self.cue_readable))
        readable = torch.where(command, cue_readable, self.legal_readable)
        choice = 3 + 4 * version + (payload + (~readable).long()) % 4
        logits[..., 3:] = -8
        selected = torch.where(original_finish, 8, 1).to(logits.dtype)
        logits.scatter_(-1, choice[..., None], selected[..., None])
        return logits, memory


def _tiny_checkpoint(tmp_path):
    config = TrainConfig(batch_size=2, n_jobs=2, tbptt=4, updates=2,
                         train_dts=(.25,), replay_block_ticks=8, replay_burnin=4,
                         replay_capacity=2, log_every=100, device="cpu")
    trainer = Trainer(Variant("tiny", width=8, policy="online"), config)
    trainer.run_updates(2)
    path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(path)
    return trainer, path


@pytest.mark.parametrize("condition", ("iid", "long_wait", "longer", "sparse", "conflict"))
def test_no_intervention_is_identical_to_frozen_own_action_runtime(condition):
    world = _world()
    model = _wait_model(world)
    hidden = torch.ones(world.batch_size, model.width)
    before = state_digest({"world": world.state_dict(), "model": model.state_dict(), "h": hidden})
    model.train()

    ordinary = evaluate_fork(model, world, hidden, condition=condition, steps=96)
    diagnosed = diagnose_fork(model, world, hidden, condition=condition, steps=96)

    for key in ("start_world_state_hash", "end_world_state_hash", "start_hidden_hash", "end_hidden_hash",
                "first_observation_hash", "end_event_hash", "primary", "inherited", "carryover_contract",
                "censored", "post_fork_counters"):
        assert ordinary[key] == diagnosed[key], key
    assert diagnosed["action_hash"] == diagnosed["model_action_hash"]
    assert diagnosed["raw_observation_hash"] == diagnosed["observation_hash"]
    assert diagnosed["opportunities"]["model_action_changes"] == 0
    assert model.training
    assert before == state_digest({"world": world.state_dict(), "model": model.state_dict(), "h": hidden})


def test_inherited_receipt_and_pending_world_are_not_reinitialized():
    world = _world(batch_size=1, active_jobs=1, cancel=0, replace=0,
                   duration_range=(1, 1), gap_range=(20, 20), window_range=(2, 2))
    for _ in range(16):
        world.observe()
        is_new = bool((world.event_kind == NEW).any())
        world.act(world.oracle())
        if is_new:
            break
    model = _wait_model(world)
    hidden = torch.ones(1, model.width)
    expected = OnlineWorld.from_state_dict(world.state_dict())
    expected.set_regime(**CONDITIONS["iid"], name="evaluation/iid", dt=.25)
    observation = expected.observe()
    assert observation.reshape(1, 2, 24)[0, 0, 22] == 1

    result = diagnose_fork(model, world, hidden, steps=20)

    assert result["first_observation_hash"] == state_digest({"observation": observation})
    assert result["initial_eligible_pending"] == result["inherited"]["matured"] == 1
    assert result["primary"]["matured"] == 0
    assert result["world_resets"] == result["hidden_resets"] == result["parameter_updates"] == 0
    assert result["fork_time"] == world.now


def test_no_intervention_identity_includes_nontrivial_begin_finish_and_receipts():
    world = _world(batch_size=4)
    torch.manual_seed(1907)
    model = AgentModel(input_size=world.input_size, n_jobs=world.n_jobs, width=12)
    with torch.no_grad():
        model.head.bias[1] += 2
        model.head.bias[model.n_actions + 6] += 2
    hidden = torch.randn(world.batch_size, model.width)
    ordinary = evaluate_fork(model, world, hidden, steps=128)
    observer = diagnose_fork(model, world, hidden, steps=128)
    assert observer["post_fork_counters"]["attempted"] > 0
    assert observer["post_fork_counters"]["finish_attempted"] > 0
    assert observer["post_fork_counters"]["accepted"] > 0
    assert ordinary["end_world_state_hash"] == observer["end_world_state_hash"]
    assert ordinary["end_hidden_hash"] == observer["end_hidden_hash"]
    assert ordinary["post_fork_counters"] == observer["post_fork_counters"]


@pytest.mark.parametrize("cue_correct,legal_correct,quadrant", (
    (True, True, "cue_legal_both_correct"), (True, False, "cue_legal_lost"),
    (False, True, "cue_legal_gained"), (False, False, "cue_legal_both_wrong"),
))
def test_matched_cue_to_legal_readability_partitions_matured_requests(cue_correct, legal_correct, quadrant):
    world = _world(batch_size=3)
    model = TransitionController(world, cue_correct=cue_correct, legal_correct=legal_correct)
    result = diagnose_fork(model, world, model.initial_state(3), steps=96)
    request = result["request_diagnosis"]["primary"]
    matured = result["primary"]["matured"]
    assert matured > 0
    assert request["cue_observed_count"] == request["cue_legal_pair_count"] == matured
    assert request["cue_content_correct_count"] == (matured if cue_correct else 0)
    quadrants = ("cue_legal_both_correct", "cue_legal_lost", "cue_legal_gained", "cue_legal_both_wrong")
    assert request[quadrant] == matured
    assert sum(request[key] for key in quadrants) == request["cue_legal_pair_count"]
    assert request["cue_legal_cue_correct"] == request["cue_legal_both_correct"] + request["cue_legal_lost"]
    assert request["cue_legal_legal_correct"] == request["cue_legal_both_correct"] + request["cue_legal_gained"]
    assert request["cue_legal_cue_accuracy"] == float(cue_correct)
    assert request["cue_legal_legal_accuracy"] == float(legal_correct)
    assert request["cue_legal_accuracy_given_cue_correct"] == (float(legal_correct) if cue_correct else None)
    assert request["cue_without_legal_count"] == request["legal_without_cue_count"] == 0


def test_observer_pair_capture_does_not_change_actions_hidden_or_full_world(monkeypatch):
    world = _world()
    model = PublicController(world)
    h = model.initial_state(3)
    paired = diagnose_fork(model, world, h, steps=96)
    monkeypatch.setattr(diagnosis._Requests, "capture_cue", lambda self, **kwargs: None)
    control = diagnose_fork(model, world, h, steps=96)
    assert paired["request_diagnosis"]["primary"]["cue_legal_pair_count"] > 0
    assert control["request_diagnosis"]["primary"]["cue_legal_pair_count"] == 0
    for key in ("action_hash", "model_action_hash", "observation_hash", "end_hidden_hash", "end_world_state_hash",
                "end_scheduler_hash", "end_event_hash", "primary", "post_fork_counters"):
        assert paired[key] == control[key], key


def test_cue_pairs_exclude_cancellation_replaced_original_and_unmatured_windows(monkeypatch):
    common = dict(duration_range=(1, 1), gap_range=(1, 1), window_range=(1, 1))
    monkeypatch.setitem(CONDITIONS, "all_canceled", dict(**common, cancel=1, replace=0))
    world = _world(batch_size=4)
    model = TransitionController(world, cue_correct=True, legal_correct=True)
    canceled = diagnose_fork(model, world, model.initial_state(4), condition="all_canceled", steps=96)
    assert canceled["observed_terminal_events"]["cancellation"] > 0
    assert canceled["primary"]["matured"] == canceled["request_diagnosis"]["primary"]["cue_observed_count"] == 0

    monkeypatch.setitem(CONDITIONS, "all_replaced", dict(**common, cancel=0, replace=1))
    replacement_model = TransitionController(world, cue_correct="replacement_only", legal_correct=True)
    replaced = diagnose_fork(replacement_model, world, replacement_model.initial_state(4),
                              condition="all_replaced", steps=96)
    request = replaced["request_diagnosis"]["primary"]
    assert replaced["observed_terminal_events"]["replacement"] > 0
    assert request["cue_legal_pair_count"] == request["cue_legal_both_correct"] == replaced["primary"]["matured"] > 0
    assert request["cue_legal_gained"] == 0  # Ineligible original NEW cues were wrong and must not leak here.

    newborn = _world(batch_size=16)
    newborn_model = PublicController(newborn)
    censored = diagnose_fork(newborn_model, newborn, newborn_model.initial_state(16), steps=8)
    assert censored["censored"]["primary"]["count"] > 0
    assert censored["primary"]["matured"] == censored["request_diagnosis"]["primary"]["cue_observed_count"] == 0


def test_never_legal_requests_are_explicitly_excluded_from_cue_legal_pairs():
    world = _world(batch_size=4)
    model = _wait_model(world)
    result = diagnose_fork(model, world, model.initial_state(4), steps=96)
    request = result["request_diagnosis"]["primary"]
    assert request["cue_observed_count"] == result["primary"]["matured"] > 0
    assert request["cue_legal_pair_count"] == 0
    assert request["cue_without_legal_count"] == request["uncompleted_no_begin"] == result["primary"]["matured"]
    assert request["cue_legal_accuracy_given_cue_correct"] is None


def test_correct_own_state_content_does_not_hide_premature_attempts_or_first_error():
    world = _world(batch_size=4)
    model = PublicController(world, finish_fraction=.5)
    h = model.initial_state(world.batch_size)

    baseline = diagnose_fork(model, world, h, steps=160)
    content = diagnose_fork(model, world, h, steps=160, intervention="finish_content_oracle")
    timing = diagnose_fork(model, world, h, steps=160, intervention="timing_oracle")

    assert baseline["primary"]["matured"] > 0
    assert baseline["primary"]["completion_rate"] == 1
    assert baseline["primary"]["clean_completion_rate"] == 0
    assert baseline["opportunities"]["content_correct_but_illegally_timed_finish_attempts"] > 0
    assert baseline["request_diagnosis"]["primary"]["first_error_causes"] == {
        "early": baseline["primary"]["matured"]}
    assert content["action_hash"] == baseline["action_hash"]
    assert content["end_hidden_hash"] == baseline["end_hidden_hash"]
    assert timing["primary"]["clean_completion_rate"] == 1
    assert timing["opportunities"]["illegally_timed_finish_attempts"] == 0
    for row in baseline["content_by_cohort_phase_age"]:
        assert row["conditional_finish_accuracy"] == 1
    assert all(x["action_policy"] == "diagnostic_counterfactual" for x in (content, timing))


def test_content_correction_changes_only_active_finish_content_and_retains_timing():
    world = _world(batch_size=4)
    model = PublicController(world, wrong_payload=True)
    h = model.initial_state(world.batch_size)
    baseline = diagnose_fork(model, world, h, steps=160, keep_trace=True)
    fixed = diagnose_fork(model, world, h, steps=160, intervention="finish_content_oracle", keep_trace=True)
    timing = diagnose_fork(model, world, h, steps=160, intervention="timing_oracle")

    assert baseline["primary"]["completion_rate"] == 0
    assert timing["primary"]["completion_rate"] == 0
    assert fixed["primary"]["clean_completion_rate"] == 1
    assert fixed["request_diagnosis"]["primary"]["finish_attempts_with_wrong_content"] == 0
    assert fixed["opportunities"]["legally_timed_wrong_content_attempts"] == 0
    assert fixed["opportunities"]["model_action_changes"] > 0
    for frame in fixed["trace"]:
        model_action = np.asarray(frame["model_action"])
        action = np.asarray(frame["executed_action"])
        changed = model_action != action
        assert np.all(~changed | np.asarray(frame["active_before_action"]))
        assert np.all(~changed | ((model_action >= 3) & (action >= 3)))
    # Own-state content readout remains wrong; correcting the executed token is
    # not allowed to relabel the model's conditional FINISH head as accurate.
    assert all(row["finish_choice_correct"] == 0 for row in fixed["content_by_cohort_phase_age"])


def test_timing_oracle_does_not_supply_begin_version_content():
    world = _world(batch_size=4)
    model = PublicController(world, wrong_begin=True)
    result = diagnose_fork(model, world, model.initial_state(4), steps=160, intervention="timing_oracle")
    assert result["primary"]["matured"] > 0
    assert result["primary"]["completed"] == 0
    assert result["request_diagnosis"]["primary"]["first_error_causes"] == {
        "stale_version": result["primary"]["matured"]}
    assert result["opportunities"]["active_finish_attempts"] == 0


def test_first_error_priority_is_exclusive_but_preserves_joint_causes():
    world = _world(batch_size=4)
    model = PublicController(world, finish_fraction=.5, wrong_payload=True)
    result = diagnose_fork(model, world, model.initial_state(4), steps=160)
    details = result["request_diagnosis"]["primary"]
    assert details["first_error_patterns"] == {"early+wrong_payload": result["primary"]["matured"]}
    assert details["first_error_causes"] == {"early": result["primary"]["matured"]}
    assert sum(details["first_error_causes"].values()) == details["with_recorded_or_inherited_error"]
    assert result["definitions"]["first_error_priority"] == list(FIRST_ERROR_PRIORITY)


@pytest.mark.parametrize("intervention,terminal", (
    ("completion_receipt_ablation", "completion"), ("cancel_command_ablation", "cancellation"),
))
def test_feedback_ablation_has_no_effect_before_target_and_preserves_external_events(intervention, terminal):
    world = _world(batch_size=4)
    model = PublicController(world)
    h = model.initial_state(4)
    baseline = diagnose_fork(model, world, h, steps=160, keep_trace=True)
    ablation = diagnose_fork(model, world, h, steps=160, intervention=intervention, keep_trace=True)

    first = next(i for i, frame in enumerate(ablation["trace"]) if np.any(frame["ablated"]))
    assert first > 0
    for left, right in zip(baseline["trace"][:first], ablation["trace"][:first]):
        for field in ("executed_action", "model_observation_hash", "hidden_hash", "event_hash"):
            assert left[field] == right[field]
    assert baseline["trace"][first]["raw_observation_hash"] == ablation["trace"][first]["raw_observation_hash"]
    assert baseline["trace"][first]["model_observation_hash"] != ablation["trace"][first]["model_observation_hash"]
    assert ablation["opportunities"]["observation_ablation_opportunities"] > 0
    assert ablation["opportunities"]["inactive_actions"] > baseline["opportunities"]["inactive_actions"]
    assert ablation["end_event_hash"] == baseline["end_event_hash"]
    for cohort in ("primary", "inherited", "carryover_contract"):
        assert ablation[cohort]["request_fingerprint"] == baseline[cohort]["request_fingerprint"]
    assert ablation["observed_terminal_events"][terminal] > 0


def test_inactive_denominators_histories_and_retained_content_wait_are_distinct():
    world = _world(batch_size=4, active_jobs=1)
    model = PublicController(world)
    result = diagnose_fork(model, world, model.initial_state(4), condition="sparse", steps=192)
    total = result["opportunities"]
    assert total["opportunities"] == 192 * 4
    assert total["disabled_slot_opportunities"] == 192 * 4
    assert total["disabled_slot_actions"] == 0
    assert total["inactive_actions"] == 0
    assert total["inactive_opportunities"] > 0
    assert total["opportunities"] == total["active_opportunities"] + total["inactive_opportunities"]
    assert sum(row["opportunities"] for row in result["inactive_history"]) == total["inactive_opportunities"]
    completed_idle = [row for row in result["inactive_history"] if row["cause"] == "completion"]
    assert sum(row["retired_content_readout_with_wait"] for row in completed_idle) > 0
    assert all(row["actions"] == 0 for row in completed_idle)
    assert all(row["conditional_finish_matches_retired_content"] == row["opportunities"] for row in completed_idle)

    ablation = diagnose_fork(model, world, model.initial_state(4), condition="sparse", steps=192,
                             intervention="completion_receipt_ablation")
    counts = ablation["opportunities"]
    assert counts["inactive_actions"] > 0
    assert ablation["inactive_actions_per_idle_opportunity"] == counts["inactive_actions"] / counts["inactive_opportunities"]
    assert ablation["inactive_actions_per_1000_total_opportunities"] == 1000 * counts["inactive_actions"] / counts["opportunities"]
    assert ablation["inactive_actions_per_1000_idle_opportunities"] > ablation["inactive_actions_per_1000_total_opportunities"]
    assert sum(row["actions"] for row in ablation["inactive_history"]) == counts["inactive_actions"]
    assert any(row["cause"] == "completion" and row["finish_matches_retired_content"] > 0
               for row in ablation["inactive_history"])


def test_expiry_and_cancellation_are_recorded_without_calling_them_completions():
    world = _world(batch_size=4)
    model = _wait_model(world)
    result = diagnose_fork(model, world, model.initial_state(4), steps=192)
    assert result["observed_terminal_events"]["expiry"] > 0
    assert result["observed_terminal_events"]["cancellation"] > 0
    assert result["observed_terminal_events"]["completion"] == 0
    assert {row["cause"] for row in result["inactive_history"]} >= {"expiry", "cancellation", "no_request_yet"}
    assert result["request_diagnosis"]["primary"]["uncompleted_no_begin"] == result["primary"]["matured"]


def test_matured_and_right_censored_denominators_do_not_depend_on_actions():
    world = _world(batch_size=8, cancel=0, replace=1,
                   duration_range=(4, 4), gap_range=(.5, .5), window_range=(2, 2))
    for _ in range(16):
        world.observe()
        is_new = bool((world.event_kind == NEW).any())
        world.act(world.oracle())
        if is_new:
            break
    model = PublicController(world, wrong_payload=True)
    h = model.initial_state(8)
    baseline = diagnose_fork(model, world, h, steps=128)
    fixed = diagnose_fork(model, world, h, steps=128, intervention="finish_content_oracle")
    assert baseline["carryover_contract_eligible_issued"] > 0
    for record in (baseline, fixed):
        assert record["primary"]["matured"] + record["censored"]["primary"]["count"] == record["primary_eligible_issued"]
        assert record["carryover_contract"]["matured"] + record["censored"]["carryover_contract"]["count"] == record["carryover_contract_eligible_issued"]
        assert record["inherited"]["matured"] + record["censored"]["inherited"]["count"] == record["initial_eligible_pending"]
    assert fixed["end_event_hash"] == baseline["end_event_hash"]
    for cohort in ("primary", "inherited", "carryover_contract"):
        assert fixed[cohort]["request_fingerprint"] == baseline[cohort]["request_fingerprint"]


def test_long_sparse_condition_is_local_and_does_not_change_frozen_conditions():
    assert "long_sparse" not in FROZEN_CONDITIONS
    assert CONDITIONS["long_sparse"]["duration_range"] == (4, 8)
    assert CONDITIONS["long_sparse"]["gap_range"] == (4, 12)
    assert CONDITIONS["short_sparse"] == FROZEN_CONDITIONS["sparse"]
    world = _world()
    model = _wait_model(world)
    result = diagnose_fork(model, world, model.initial_state(3), condition="long_sparse", steps=8)
    assert result["primary"]["matured"] == 0
    assert result["primary_eligible_issued"] == result["censored"]["primary"]["count"]


def test_checkpoint_diagnosis_preserves_bytes_rng_optimizer_replay_and_live_state(tmp_path):
    trainer, checkpoint = _tiny_checkpoint(tmp_path)
    before = state_digest(trainer.semantic_state())
    file_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    process_rng = state_digest(rng_state())
    settings = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark,
                torch.backends.cuda.matmul.allow_tf32)

    result = diagnose_checkpoint(checkpoint, conditions=("iid", "long_sparse"),
                                 interventions=("finish_content_oracle", "completion_receipt_ablation"), steps=24)

    assert result["checkpoint_state_hash"] == before == state_digest(trainer.semantic_state())
    assert file_hash == result["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert process_rng == state_digest(rng_state())
    assert settings == (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark,
                        torch.backends.cuda.matmul.allow_tf32)
    assert len(result["records"]) == 6
    assert result["evaluation_args"]["interventions"][0] == "none"
    assert result["training_ticks"] == trainer.ticks
    for records in (result["records"][:3], result["records"][3:]):
        assert len({record["end_event_hash"] for record in records}) == 1
        assert len({record["start_world_state_hash"] for record in records}) == 1
        assert len({record["start_hidden_hash"] for record in records}) == 1
        assert len({record["end_scheduler_hash"] for record in records}) == 1


def test_cli_saves_observer_summary_and_invalid_forks_are_rejected(tmp_path):
    _, checkpoint = _tiny_checkpoint(tmp_path)
    output = tmp_path / "diagnosis.json"
    main(["--checkpoint", str(checkpoint), "--output", str(output), "--device", "cpu",
          "--conditions", "iid", "--steps", "8"])
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert len(saved["evaluations"][0]["records"]) == 1
    assert saved["evaluations"][0]["records"][0]["action_policy"] == "model"
    world = _world()
    model = _wait_model(world)
    h = model.initial_state(3)
    for kwargs in (dict(steps=0), dict(steps=True), dict(dt=0), dict(condition="unknown"),
                   dict(intervention="unknown"), dict(age_edges=(0, 2, 1))):
        with pytest.raises(ValueError):
            diagnose_fork(model, world, h, **kwargs)
    world.observe()
    with pytest.raises(ValueError, match="completed action"):
        diagnose_fork(model, world, h)
    with pytest.raises(ValueError, match="distinct"):
        diagnose_checkpoint(tmp_path / "absent.pt", conditions=("iid", "iid"))
    with pytest.raises(ValueError, match="distinct"):
        diagnose_checkpoint(tmp_path / "absent.pt", interventions=(INTERVENTIONS[0], INTERVENTIONS[0]))
