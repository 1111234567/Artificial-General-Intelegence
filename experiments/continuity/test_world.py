"""Continuity, physical timing, action causality, and complete checkpoint tests."""

from io import BytesIO
import json

import pytest
import torch

from experiments.continuity.world import CANCEL, NEW, REPLACE, OnlineWorld


def _world(**kwargs):
    defaults = dict(seed=17, batch_size=1, n_jobs=2, active_jobs=1,
                    duration_range=(1.0, 1.0), gap_range=(20.0, 20.0),
                    window_range=(2.0, 2.0), cancel=0, replace=0)
    defaults.update(kwargs)
    return OnlineWorld(**defaults)


def _first_new(world):
    for _ in range(32):
        observation = world.observe()
        if world.event_kind[0, 0] == NEW:
            return observation
        world.act(torch.zeros((world.batch_size, world.n_jobs), dtype=torch.long))
    raise AssertionError("initial NEW did not arrive")


def _assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_state_equal(a, b)
    else:
        assert left == right


def _actions(world):
    return (torch.arange(world.batch_size * world.n_jobs).reshape(
        world.batch_size, world.n_jobs) + world.tick) % 13 - 1


def test_public_features_are_one_shot_without_pending_or_readiness_labels():
    world = _world()
    pulse = _first_new(world).reshape(1, 2, 24)
    assert pulse[0, 0, 0] == 1
    assert pulse[0, 0, 9:11].tolist() == [1.0, 2.0]
    version = int(world.version[0, 0])
    world.act(torch.tensor([[1 + version, 0]]))
    receipt = world.observe().reshape(1, 2, 24)
    assert receipt[0, 0, :11].sum() == 0
    assert receipt[0, 0, 11 + 1 + version] == 1
    assert receipt[0, 0, 22] == 1
    world.act(torch.zeros((1, 2), dtype=torch.long))
    for _ in range(6):
        observation = world.observe()
        assert torch.count_nonzero(observation) == 0
        world.act(torch.zeros((1, 2), dtype=torch.long))
    assert world.active[0, 0] and world.started[0, 0]
    assert world.now >= world.ready_time[0, 0]


def test_arbitrary_chunk_boundaries_never_flush_pending_tasks_or_receipts():
    world = _world(duration_range=(4.0, 4.0))
    _first_new(world)
    request = world.request_id.clone()
    world.act(world.oracle())
    boundary = world.state_dict()
    assert boundary["tensors"]["_ack_result"][0, 0] == 1
    for chunk in (3, 2, 4):
        for _ in range(chunk):
            world.observe()
            world.act(torch.zeros((1, 2), dtype=torch.long))
        assert world.active[0, 0] and world.started[0, 0]
        assert torch.equal(world.request_id, request)
        assert world.metrics()["eligible_pending"] == 1
    assert world.metrics()["matured_eligible"] == 0
    assert world.metrics()["commands"] == 1


def test_matured_denominator_keeps_complete_and_incomplete_windows_pending():
    world = _world()
    _first_new(world)
    deadline = float(world.deadline[0, 0])
    world.act(world.oracle("earliest"))
    while world.active[0, 0]:
        world.observe()
        world.act(world.oracle("earliest"))
    early = world.metrics()
    assert early["eligible"] == early["completed_eligible"] == 1
    assert early["matured_eligible"] == 0
    assert early["eligible_pending"] == early["eligible_pending_completed"] == 1
    assert early["pending_active"] == 0
    assert early["completion_rate"] == 0
    while world.now <= deadline + 1e-6:
        world.observe()
        world.act(world.oracle())
    final = world.metrics()
    assert final["matured_eligible"] == final["completed_matured"] == 1
    assert final["clean_completed_matured"] == 1
    assert final["eligible_pending"] == 0
    assert final["completion_rate"] == final["clean_completion_rate"] == 1
    # Outcome fields remain available for an external request-ID boundary scorer.
    assert world.current_eligible[0, 0] and world.request_id[0, 0] >= 0
    json.dumps(final, allow_nan=False)


def test_maturity_pulse_retains_completed_old_request_when_next_new_arrives():
    world = _world(gap_range=(0.0, 0.0))
    _first_new(world)
    request = int(world.request_id[0, 0])
    issued = float(world.issued_at[0, 0])
    deadline = float(world.deadline[0, 0])
    world.act(world.oracle("earliest"))
    while world.now <= deadline + 1e-6:
        world.observe()
        if world.now > deadline + 1e-6:
            break
        assert not world.matured_mask.any()
        world.act(world.oracle("earliest"))
    assert world.matured_mask.tolist() == [[True, False]]
    assert world.matured_request_id[0, 0] == request
    assert world.matured_issued_at[0, 0] == issued
    assert world.matured_duration[0, 0] == 1
    assert world.matured_completed[0, 0] and world.matured_clean_completed[0, 0]
    assert world.event_kind[0, 0] == NEW
    assert world.request_id[0, 0] != request
    clone = OnlineWorld.from_state_dict(world.state_dict())
    assert torch.equal(clone.matured_request_id, world.matured_request_id)
    world.act(world.oracle())
    world.observe()
    assert not world.matured_mask.any()
    assert world.matured_request_id.tolist() == [[-1, -1]]
    world.act(world.oracle())


def test_matured_request_cohorts_are_policy_independent_including_expired_jobs():
    teacher = OnlineWorld(seed=43, batch_size=8)
    silent = OnlineWorld(seed=43, batch_size=8)
    matured_teacher = matured_silent = 0
    for _ in range(160):
        teacher.observe()
        silent.observe()
        for name in ("matured_mask", "matured_request_id", "matured_issued_at", "matured_duration"):
            assert torch.equal(getattr(teacher, name), getattr(silent, name))
        matured_teacher += int(teacher.matured_completed.sum())
        matured_silent += int(silent.matured_completed.sum())
        teacher.act(teacher.oracle())
        silent.act(torch.zeros((8, 4), dtype=torch.long))
    assert matured_teacher > 0 and matured_silent == 0
    assert teacher.metrics()["matured_eligible"] == silent.metrics()["matured_eligible"]


@pytest.mark.parametrize("probabilities", [(0.2, 0.3), (0.3, 0.55), (0.0, 0.0)])
def test_oracle_resolves_every_matured_final_request_without_errors(probabilities):
    world = OnlineWorld(seed=31, batch_size=12, n_jobs=4,
                        cancel=probabilities[0], replace=probabilities[1])
    for tick in range(512):
        if tick == 128:
            world.set_dt(0.125)
        elif tick == 256:
            world.set_regime(duration_range=(4.0, 8.0), gap_range=(0.25, 0.75), name="long")
            world.set_dt(0.375)
        world.observe()
        world.act(world.oracle())
    metrics = world.metrics(per_stream=True)
    assert metrics["matured_eligible"] > 100
    assert metrics["completed_matured"] == metrics["matured_eligible"]
    assert metrics["clean_completed_matured"] == metrics["matured_eligible"]
    assert metrics["rejected"] == 0
    assert metrics["completion_rate"] == metrics["clean_completion_rate"] == 1
    assert metrics["eligible"] == metrics["matured_eligible"] + metrics["eligible_pending"]
    assert metrics["canceled_correctly"] == metrics["canceled"]
    assert metrics["finish_precision"] == 1
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize("checkpoint_after_observe", [False, True])
def test_checkpoint_exactly_replays_scheduler_rng_pending_jobs_receipts_and_counts(checkpoint_after_observe):
    original = OnlineWorld(seed=7, batch_size=8, n_jobs=4)
    for tick in range(37):
        if tick == 16:
            original.set_dt(0.375)
        if tick == 23:
            original.set_regime(duration_range=(4.0, 8.0), name="changed")
        original.observe()
        original.act(_actions(original))
    if checkpoint_after_observe:
        original.observe()
    buffer = BytesIO()
    torch.save(original.state_dict(), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    resumed = OnlineWorld.from_state_dict(saved)
    _assert_state_equal(original.state_dict(), resumed.state_dict())
    if checkpoint_after_observe:
        assert torch.equal(original.oracle(), resumed.oracle())
        original.act(_actions(original))
        resumed.act(_actions(resumed))
    for tick in range(96):
        if tick == 17:
            original.set_dt(0.125)
            resumed.set_dt(0.125)
        if tick == 34:
            original.set_regime(gap_range=(0.5, 0.75), cancel=0.4, replace=0.1, name="again")
            resumed.set_regime(gap_range=(0.5, 0.75), cancel=0.4, replace=0.1, name="again")
        assert torch.equal(original.observe(), resumed.observe())
        assert torch.equal(original.oracle(), resumed.oracle())
        original.act(_actions(original))
        resumed.act(_actions(resumed))
    _assert_state_equal(original.state_dict(), resumed.state_dict())
    assert original.metrics(per_stream=True) == resumed.metrics(per_stream=True)


def test_actor_streams_survive_batch_size_changes_and_different_actions():
    small = OnlineWorld(seed=91, batch_size=64, n_jobs=4)
    large = OnlineWorld(seed=91, batch_size=128, n_jobs=4)
    for _ in range(160):
        small.observe()
        large.observe()
        for name in ("event_kind", "event_version", "event_payload", "event_duration",
                     "event_window", "event_request_id", "event_eligible"):
            assert torch.equal(getattr(small, name), getattr(large, name)[:64])
        small.act(torch.zeros((64, 4), dtype=torch.long))
        large.act(large.oracle())
    assert small.case_ids == large.case_ids[:64]
    assert small.metrics()["completed_all"] == 0
    assert large.metrics()["completed_all"] > 0


def test_selected_actor_identity_is_independent_of_lane_order():
    base = OnlineWorld(seed=123, batch_size=4, actor_ids=(100, 200, 300, 400))
    reordered = OnlineWorld(seed=123, batch_size=2, actor_ids=(400, 100))
    for _ in range(80):
        x, y = base.observe(), reordered.observe()
        assert torch.equal(x[[3, 0]], y)
        base.act(base.oracle())
        reordered.act(reordered.oracle())


def test_fixed_four_slot_world_supports_two_active_jobs():
    world = OnlineWorld(seed=2, batch_size=4, n_jobs=4, active_jobs=2)
    for _ in range(128):
        observation = world.observe().reshape(4, 4, 24)
        assert not observation[:, 2:].any()
        assert not world.active[:, 2:].any()
        assert not world.oracle()[:, 2:].any()
        world.act(world.oracle())
    assert world.input_size == 96 and world.n_actions == 11
    assert not world.counts["commands"][:, 2:].any()


def test_regime_changes_preserve_pending_modifications_and_already_sampled_gap():
    world = _world(duration_range=(4.0, 4.0), gap_range=(3.0, 3.0), replace=1)
    _first_new(world)
    issued = world.issued_at.clone()
    world.set_regime(duration_range=(8.0, 8.0), gap_range=(9.0, 9.0),
                     window_range=(3.0, 3.0), replace=0, name="later")
    assert world.duration[0, 0] == 4
    assert torch.equal(world.issued_at, issued)
    world.act(world.oracle())
    while True:
        world.observe()
        if world.event_kind[0, 0] == REPLACE:
            break
        world.act(world.oracle())
    assert world.duration[0, 0] == 4
    assert world.event_window[0, 0] == 2
    assert world.current_regime_id[0, 0] == 0
    due = world.state_dict()["scheduler"]["_next_new_time"][0, 0]
    assert due == pytest.approx(world.now + 4 + 2 + 3)
    world.act(world.oracle())
    while True:
        world.observe()
        if world.event_kind[0, 0] == NEW:
            break
        world.act(world.oracle())
    assert world.duration[0, 0] == 8
    assert world.event_window[0, 0] == 3
    assert world.current_regime_id[0, 0] == 1
    world.act(world.oracle())


def test_clock_changes_use_absolute_elapsed_time_without_rescheduling_pending_jobs():
    world = _world(duration_range=(4.0, 4.0))
    _first_new(world)
    world.act(world.oracle())
    origin, ready = world.now, world.ready_time.clone()
    world.set_dt(0.125)
    world.observe()
    assert world.now == origin + 0.125 and world.elapsed == 0.125
    assert torch.equal(world.ready_time, ready)
    world.act(world.oracle())
    world.set_regime(dt=0.375)
    world.observe()
    assert world.now == origin + 0.5 and world.elapsed == 0.375
    world.set_dt(0.5)
    assert world.elapsed == 0.375
    resumed = OnlineWorld.from_state_dict(world.state_dict())
    assert resumed.elapsed == 0.375 and resumed.dt == 0.5
    world.act(world.oracle())
    resumed.act(resumed.oracle())
    assert torch.equal(world.observe(), resumed.observe())
    assert world.now == origin + 1.0 and world.elapsed == 0.5
    assert torch.equal(world.ready_time, ready)
    world.act(world.oracle())
    resumed.act(resumed.oracle())


def test_cancellation_precedes_actions_and_invalidates_the_pending_request():
    world = _world(duration_range=(4.0, 4.0), cancel=1)
    _first_new(world)
    version, payload = int(world.version[0, 0]), int(world.payload[0, 0])
    world.act(world.oracle())
    while True:
        world.observe()
        if world.event_kind[0, 0] == CANCEL:
            break
        world.act(world.oracle())
    assert world.oracle().tolist() == [[0, 0]]
    world.act(torch.tensor([[3 + 4 * version + payload, 0]]))
    metrics = world.metrics()
    assert metrics["completed_all"] == 0 and metrics["rejected"] == 1
    assert metrics["canceled"] == 1 and metrics["canceled_correctly"] == 0
    assert metrics["eligible"] == metrics["matured_eligible"] == 0


def test_clean_completion_excludes_a_request_with_a_rejected_action():
    world = _world()
    _first_new(world)
    finish = 3 + 4 * int(world.version[0, 0]) + int(world.payload[0, 0])
    world.act(torch.tensor([[finish, 0]]))
    deadline = float(world.deadline[0, 0])
    while world.now <= deadline + 1e-6:
        world.observe()
        world.act(world.oracle("earliest"))
    metrics = world.metrics()
    assert metrics["before_begin"] == metrics["rejected"] == 1
    assert metrics["completed_matured"] == metrics["matured_eligible"] == 1
    assert metrics["clean_completed_matured"] == 0
    assert metrics["completion_rate"] == 1 and metrics["clean_completion_rate"] == 0


def test_oracle_does_not_consult_future_events_or_eligibility():
    world = _world()
    _first_new(world)
    label = world.oracle()
    world.current_eligible.zero_()
    world._next_new_time.fill(10000)
    world._modify_time.fill(10000)
    world._modify_payload.fill(3)
    assert torch.equal(world.oracle(), label)
    world.act(label)


def test_protocol_validation_and_state_dict_are_detached():
    world = _world()
    with pytest.raises(ValueError, match="follow"):
        world.act(torch.zeros((1, 2), dtype=torch.long))
    with pytest.raises(ValueError, match="between"):
        world.oracle()
    world.observe()
    with pytest.raises(ValueError, match="advance"):
        world.observe()
    with pytest.raises(ValueError, match="shape"):
        world.act(torch.zeros((1, 4), dtype=torch.long))
    with pytest.raises(ValueError, match="policy"):
        world.oracle("future")
    world.act(world.oracle())
    saved = world.state_dict()
    saved["tensors"]["payload"].fill_(99)
    assert (world.payload < 4).all()
    with pytest.raises(ValueError, match="dt"):
        world.set_dt(0)
    with pytest.raises(ValueError, match="probabilities|exceed"):
        world.set_regime(cancel=0.8, replace=0.8)
    with pytest.raises(ValueError, match="active_jobs"):
        OnlineWorld(n_jobs=2, active_jobs=4)
    with pytest.raises(ValueError, match="actor_ids"):
        OnlineWorld(batch_size=2, actor_ids=(1, 1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_hot_path_matches_cpu_and_preserves_serialized_state():
    cpu = OnlineWorld(seed=29, batch_size=4)
    cuda = OnlineWorld(seed=29, batch_size=4, device="cuda")
    for tick in range(48):
        if tick == 17:
            cpu.set_dt(0.375)
            cuda.set_dt(0.375)
        assert torch.equal(cpu.observe(), cuda.observe().cpu())
        actions = cpu.oracle()
        assert torch.equal(actions, cuda.oracle().cpu())
        cpu.act(actions)
        cuda.act(actions.to("cuda"))
    assert cpu.metrics() == cuda.metrics()
    _assert_state_equal(cpu.state_dict(), cuda.state_dict())
