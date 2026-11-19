"""Independent timing, command priority, and observation-channel checks."""

from dataclasses import replace
import json

import pytest
import torch

from experiments.dynamics.world import (
    CANCEL, INPUT_SIZE, NEW, NONE, REPLACE, StreamBatch, StreamConfig,
    WorldBatch, generate_stream,
)


def _manual(events, steps=32, dt=0.25):
    """Hand-authored absolute-time events, with no generator/oracle reuse."""
    shape = (steps, 1, 2)
    values = {name: torch.zeros(shape, dtype=dtype) for name, dtype in (
        ("kind", torch.long), ("version", torch.long), ("payload", torch.long),
        ("duration", torch.float32), ("window", torch.float32),
        ("eligible", torch.bool), ("interrupted", torch.bool))}
    values["request_id"] = torch.full(shape, -1, dtype=torch.long)
    for event in events:
        time, job, kind, version, payload, duration, window, eligible = event
        step = round(time / dt)
        assert abs(step * dt - time) < 1e-7
        for key, value in (("kind", kind), ("version", version), ("payload", payload),
                           ("duration", duration), ("window", window),
                           ("eligible", eligible), ("request_id", step * 2 + job)):
            values[key][step, 0, job] = value
    return StreamBatch(StreamConfig(condition="manual", dt=dt), **values,
                       times=torch.arange(steps, dtype=torch.float64) * dt)


def _run_oracle(stream):
    world = WorldBatch(stream)
    trajectory = []
    for t in range(stream.steps):
        world.observe(t)
        action = world.oracle()
        for job in range(2):
            if action[0, job]:
                trajectory.append((float(stream.times[t]), job, int(action[0, job])))
        world.act(action)
    return world, trajectory


@pytest.mark.parametrize("condition", ["train", "overlap", "conflict", "long_gap", "sparse", "long"])
def test_oracle_completes_all_eligible_requests_without_false_actions(condition):
    stream = generate_stream(17, 16, 512, condition=condition)
    world, _ = _run_oracle(stream)
    metrics = world.metrics(per_stream=True)
    assert metrics["eligible"] > 0
    assert metrics["completed_eligible"] == metrics["eligible"]
    assert metrics["completion_rate"] == 1.0
    assert metrics["clean_completion_rate"] == 1.0
    assert metrics["finish_precision"] == 1.0
    assert metrics["interrupted_completed"] == metrics["interrupted_eligible"]
    assert metrics["rejected"] == 0
    assert metrics["canceled_correctly"] == metrics["canceled"]
    assert all(delay is not None and 0 <= delay < stream.config.dt + 1e-5
               for delay in metrics["per_job_finish_delay"])
    json.dumps(metrics, allow_nan=False)


def test_independent_controller_uses_only_command_pulses_and_elapsed_time():
    stream = generate_stream(31, 4, 320, condition="conflict")
    world = WorldBatch(stream)
    orders = [[None, None] for _ in range(stream.batch_size)]
    for t in range(stream.steps):
        observation = world.observe(t).reshape(stream.batch_size, 2, 24)
        actions = torch.zeros((stream.batch_size, 2), dtype=torch.long)
        now = float(stream.times[t])
        for b in range(stream.batch_size):
            for job in range(2):
                pulse = observation[b, job]
                kind = int(pulse[:3].argmax()) + 1 if pulse[:3].sum() else NONE
                if kind in (NEW, REPLACE):
                    orders[b][job] = {
                        "version": int(pulse[3:5].argmax()),
                        "payload": int(pulse[5:9].argmax()),
                        "ready": None, "duration": float(pulse[9]),
                    }
                elif kind == CANCEL:
                    orders[b][job] = None
                order = orders[b][job]
                if order is None:
                    continue
                if order["ready"] is None:
                    actions[b, job] = 1 + order["version"]
                    order["ready"] = now + order["duration"]
                elif now >= order["ready"] - 1e-6:
                    actions[b, job] = 3 + 4 * order["version"] + order["payload"]
                    orders[b][job] = None
        assert torch.equal(actions, world.oracle())
        world.act(actions)
    assert world.metrics()["completion_rate"] == 1.0


def test_cancel_at_exact_ready_time_precedes_action():
    stream = _manual([(0.0, 0, NEW, 1, 2, 1.0, 1.0, False),
                      (1.0, 0, CANCEL, 1, 0, 0.0, 0.0, False)])
    world = WorldBatch(stream)
    for t in range(5):
        world.observe(t)
        if t == 0:
            world.act(torch.tensor([[2, 0]]))
        elif t == 4:
            assert world.oracle().tolist() == [[0, 0]]
            world.act(torch.tensor([[9, 0]]))
        else:
            world.act(torch.zeros((1, 2), dtype=torch.long))
    assert world.metrics()["completed_all"] == 0
    assert world.metrics()["rejected"] == 1
    assert world.metrics()["canceled_correctly"] == 0


def test_replace_invalidates_old_version_and_restarts_processing():
    stream = _manual([(0.0, 0, NEW, 0, 1, 2.0, 1.0, False),
                      (0.5, 0, REPLACE, 1, 3, 1.0, 1.0, True)])
    world = WorldBatch(stream)
    for t in range(8):
        world.observe(t)
        if t == 0:
            action = [[1, 0]]
        elif t == 2:
            action = [[4, 0]]  # old version/payload, not the replacement
        else:
            action = world.oracle()
        world.act(torch.as_tensor(action))
    metrics = world.metrics()
    assert metrics["stale"] == 1
    assert metrics["rejected"] == 1
    assert metrics["completed_eligible"] == 1
    assert world.ready_time[0, 0] == 1.75  # actual delayed BEGIN at t=.75


def test_early_wrong_payload_and_before_begin_cannot_complete_request():
    stream = _manual([(0.0, 0, NEW, 0, 2, 1.0, 2.0, True)])
    world = WorldBatch(stream)
    action_schedule = {0: 5, 1: 1, 2: 5, 5: 6, 6: 5}
    for t in range(7):
        world.observe(t)
        world.act(torch.tensor([[action_schedule.get(t, 0), 0]]))
    metrics = world.metrics()
    assert metrics["before_begin"] == 1
    assert metrics["early"] == 1
    assert metrics["wrong_payload"] == 1
    assert metrics["completed_eligible"] == 1
    assert metrics["clean_completed_eligible"] == 0
    assert metrics["clean_completion_rate"] == 0.0
    assert metrics["finish_precision"] == 0.25
    assert metrics["per_job_finish_delay"][0] == pytest.approx(0.25)


def test_idle_error_does_not_contaminate_next_request_clean_completion():
    stream = _manual([(0.5, 0, NEW, 1, 2, 1.0, 1.0, True)])
    world = WorldBatch(stream)
    for t in range(stream.steps):
        world.observe(t)
        world.act(torch.tensor([[3, 0]]) if t == 0 else world.oracle())
    metrics = world.metrics()
    assert metrics["false_action"] == 1
    assert metrics["completed_eligible"] == metrics["clean_completed_eligible"] == 1
    assert metrics["finish_precision"] == 0.5


def test_observations_are_one_shot_and_do_not_leak_ready_or_pending_state():
    stream = _manual([(0.0, 0, NEW, 1, 3, 1.0, 1.0, True)])
    world = WorldBatch(stream)
    first = world.observe(0)
    assert first.shape == (1, INPUT_SIZE)
    assert first[0, 9] == 1.0 and first[0, 10] == 1.0
    world.act(torch.tensor([[2, 0]]))
    receipt = world.observe(1).reshape(1, 2, 24)
    assert receipt[0, 0, :11].sum() == 0
    assert receipt[0, 0, 13] == 1  # prior BEGIN(1), not its correct continuation
    assert receipt[0, 0, 22] == 1
    world.act(torch.zeros((1, 2), dtype=torch.long))
    for t in range(2, 6):
        observation = world.observe(t)
        assert torch.count_nonzero(observation) == 0  # including readiness boundary
        world.act(torch.zeros((1, 2), dtype=torch.long))
    assert world.active[0, 0] and world.started[0, 0]


def test_old_order_returns_after_another_order_without_a_new_input():
    stream = _manual([(0.0, 0, NEW, 0, 3, 2.0, 1.0, True),
                      (0.5, 1, NEW, 1, 1, 0.5, 1.0, True)])
    world, trajectory = _run_oracle(stream)
    assert trajectory == [(0.0, 0, 1), (0.5, 1, 2),
                          (1.0, 1, 8), (2.0, 0, 6)]
    assert world.metrics()["completed_eligible"] == 2
    assert stream.kind[8].sum() == 0


def test_processing_depends_on_physical_elapsed_time_not_observe_count():
    event = [(0.0, 0, NEW, 0, 1, 1.0, 1.0, True)]
    fine, fine_actions = _run_oracle(_manual(event, dt=0.25))
    coarse, coarse_actions = _run_oracle(_manual(event, dt=0.5))
    assert fine_actions == coarse_actions == [(0.0, 0, 1), (1.0, 0, 4)]
    assert fine.metrics()["completion_rate"] == coarse.metrics()["completion_rate"] == 1


def test_readout_subdivision_preserves_aligned_event_and_action_semantics():
    events = [(0.0, 0, NEW, 1, 2, 1.0, 1.0, True),
              (0.5, 1, NEW, 0, 1, 1.5, 1.0, True)]
    first, actions = _run_oracle(_manual(events, steps=24, dt=0.25))
    second, subactions = _run_oracle(_manual(events, steps=47, dt=0.125))
    assert actions == subactions
    for key in ("eligible", "completed_eligible", "attempted", "accepted", "rejected"):
        assert first.metrics()[key] == second.metrics()[key]


def test_deadline_is_tied_to_arrival_and_cannot_be_deferred_by_late_begin():
    stream = _manual([(0.0, 0, NEW, 0, 1, 1.0, 0.5, True)])
    world = WorldBatch(stream)
    for t in range(9):
        world.observe(t)
        world.act(torch.tensor([[1 if t == 4 else (4 if t == 8 else 0), 0]]))
    assert world.deadline[0, 0] == 1.5
    assert world.metrics()["completed_eligible"] == 0
    assert world.metrics()["expired"] == 1
    assert world.metrics()["late"] == 1


def test_actions_never_change_future_events_or_shared_case_identity():
    stream = generate_stream(7, 2, 128)
    duplicate = generate_stream(7, 2, 128)
    snapshot = {name: getattr(stream, name).clone() for name in
                ("kind", "version", "payload", "duration", "window", "times")}
    silent, teacher = WorldBatch(stream), WorldBatch(stream)
    for t in range(stream.steps):
        silent.observe(t)
        silent.act(torch.zeros((2, 2), dtype=torch.long))
        teacher.observe(t)
        teacher.act(teacher.oracle())
    assert stream.fingerprint == duplicate.fingerprint
    assert stream.case_ids == duplicate.case_ids
    assert teacher.metrics()["world_hash"] == silent.metrics()["world_hash"]
    for name, value in snapshot.items():
        assert torch.equal(getattr(stream, name), value)


def test_observe_and_act_are_chronological_and_reset_is_explicit():
    stream = _manual([(0.0, 0, NEW, 1, 1, 1.0, 1.0, True)])
    world = WorldBatch(stream)
    with pytest.raises(ValueError, match="follow"):
        world.act(torch.zeros((1, 2), dtype=torch.long))
    world.observe(0)
    with pytest.raises(ValueError, match="advance"):
        world.observe(1)
    world.act(world.oracle())
    with pytest.raises(ValueError, match="advance"):
        world.observe(0)
    assert world.started[0, 0]
    world.reset()
    assert not world.started.any() and not world.active.any()
    assert world.metrics()["commands"] == 0
    world.observe(0)


def test_generator_conditions_and_invalid_inputs():
    ordinary = generate_stream(1, 2, 128)
    longer = generate_stream(1, 2, 128, condition="long_gap")
    assert longer.config.duration_min > ordinary.config.duration_min
    assert ordinary.to("cpu").fingerprint == ordinary.fingerprint
    with pytest.raises(ValueError, match="unknown condition"):
        generate_stream(0, 2, 32, condition="unknown")
    with pytest.raises(ValueError, match="dt"):
        generate_stream(0, 2, 32, dt=0)
    with pytest.raises(ValueError, match="positive integer"):
        generate_stream(0, 0, 32)
    with pytest.raises(ValueError, match="timestamps"):
        replace(ordinary, times=torch.zeros_like(ordinary.times))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_batch_matches_cpu_without_per_environment_runtime_loops():
    stream = generate_stream(3, 8, 128)
    cpu = WorldBatch(stream)
    cuda = WorldBatch(stream, "cuda")
    for t in range(stream.steps):
        assert torch.equal(cpu.observe(t), cuda.observe(t).cpu())
        actions = cpu.oracle()
        assert torch.equal(actions, cuda.oracle().cpu())
        cpu.act(actions)
        cuda.act(actions.to("cuda"))
    assert cpu.metrics() == cuda.metrics()
