"""Context rule, causal stream, complete continuation and evaluator checks."""

from copy import deepcopy
from io import BytesIO
import random

import numpy as np
import pytest
import torch

from experiments.continuity.model import AgentModel
from experiments.continuity.run import state_digest
from experiments.continuity.world import OnlineWorld
from experiments.lifelong.evaluate import CONDITIONS, evaluate
from experiments.lifelong.world import CANCEL, NEW, REPLACE, ContextWorld, FAMILY_MAPPINGS


def _world(**kwargs):
    args = dict(seed=17, batch_size=1, n_jobs=2, active_jobs=1,
                duration_range=(1.0, 1.0), gap_range=(20.0, 20.0),
                window_range=(2.0, 2.0), cancel=0, replace=0)
    args.update(kwargs)
    return ContextWorld(**args)


def _first_new(world):
    for _ in range(32):
        obs = world.observe()
        if world.event_kind[0, 0] == NEW:
            return obs
        world.act(torch.zeros((world.batch_size, world.n_jobs), dtype=torch.long))
    raise AssertionError("NEW not delivered")


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for name in left:
            _equal(left[name], right[name])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("family", range(4))
def test_same_raw_symbols_different_contexts_have_real_different_private_targets(family):
    worlds = [_world(seed=seed, future_family=family) for seed in range(24)]
    observed_raw = set()
    changed = False
    for world in worlds:
        pulse = _first_new(world).reshape(1, 2, 28)
        raw, target = int(world.raw_payload[0, 0]), int(world.payload[0, 0])
        observed_raw.add(raw)
        assert target == FAMILY_MAPPINGS[family][raw]
        assert pulse[0, 0, 5:9].argmax().item() == raw
        assert pulse[0, 0, 24:28].tolist() == [float(index == family) for index in range(4)]
        reference = _world(seed=world.seed, future_family=0)
        identity_obs = _first_new(reference).reshape(1, 2, 28)
        assert torch.equal(pulse[..., :24], identity_obs[..., :24])
        assert int(reference.raw_payload[0, 0]) == raw
        changed |= target != int(reference.payload[0, 0])
    assert observed_raw == {0, 1, 2, 3}
    assert changed == (family != 0)


def test_context_is_one_shot_and_no_private_target_enters_idle_observation():
    world = _world(future_family=3)
    _first_new(world)
    world.act(world.oracle())
    receipt = world.observe().reshape(1, 2, 28)
    assert torch.count_nonzero(receipt[..., :11]) == 0
    assert torch.count_nonzero(receipt[..., 24:]) == 0
    world.act(torch.zeros((1, 2), dtype=torch.long))
    for _ in range(4):
        assert torch.count_nonzero(world.observe()) == 0
        world.act(torch.zeros((1, 2), dtype=torch.long))


@pytest.mark.parametrize("modification", ["replace", "cancel"])
def test_pending_modification_preserves_context_and_raw_scheduler_across_policy_change(modification):
    world = _world(future_family=3, **{modification: 1})
    _first_new(world)
    pending_raw = int(world._modify_payload[0, 0])
    before_rng = world.state_dict()["context"]["rng_state"].clone()
    world.act(torch.zeros((1, 2), dtype=torch.long))
    world.set_family(1)
    assert int(world._modify_payload[0, 0]) == pending_raw
    for _ in range(16):
        obs = world.observe().reshape(1, 2, 28)
        if int(world.event_kind[0, 0]) in (REPLACE, CANCEL):
            assert world.event_family[0, 0] == 3
            assert int(world.raw_event_payload[0, 0]) == pending_raw
            assert int(world.event_payload[0, 0]) == FAMILY_MAPPINGS[3][pending_raw]
            if modification == "replace":
                assert world.current_family[0, 0] == 3
                assert obs[0, 0, 5:9].argmax() == pending_raw
                assert obs[0, 0, 27] == 1
            else:
                assert torch.count_nonzero(obs[0, 0, 24:]) == 0
            assert torch.equal(before_rng, world.state_dict()["context"]["rng_state"])
            return
        world.act(torch.zeros((1, 2), dtype=torch.long))
    raise AssertionError("modification did not arrive")


def test_blocked_and_mixed_consume_one_context_draw_per_new_with_same_raw_stream():
    fixed = _world(future_family=2, gap_range=(0.5, 0.5))
    mixed = _world(family_allowed=(0, 1, 2, 3), gap_range=(0.5, 0.5))
    for _ in range(90):
        fixed.observe()
        mixed.observe()
        assert torch.equal(fixed.raw_event_payload, mixed.raw_event_payload)
        assert torch.equal(fixed.event_kind, mixed.event_kind)
        assert torch.equal(fixed.state_dict()["context"]["rng_state"], mixed.state_dict()["context"]["rng_state"])
        fixed.act(torch.zeros((1, 2), dtype=torch.long))
        mixed.act(torch.zeros((1, 2), dtype=torch.long))
    assert fixed.event_hash != mixed.event_hash
    fixed.set_family_policy((0, 1, 2, 3))
    for _ in range(90):
        fixed.observe()
        mixed.observe()
        new = fixed.event_kind == NEW
        assert torch.equal(fixed.event_family[new], mixed.event_family[new])
        fixed.act(torch.zeros((1, 2), dtype=torch.long))
        mixed.act(torch.zeros((1, 2), dtype=torch.long))


@pytest.mark.parametrize("pre_action", [False, True])
def test_cpu_resume_is_exact_including_pending_context_rng_counts_and_receipts(pre_action):
    world = _world(batch_size=3, n_jobs=4, active_jobs=4, family_allowed=(0, 1, 2, 3), replace=0.5)
    for tick in range(25):
        world.observe()
        if tick == 24 and pre_action:
            break
        world.act(world.oracle())
    memory = BytesIO()
    torch.save(world.state_dict(), memory)
    memory.seek(0)
    restored = ContextWorld.from_state_dict(torch.load(memory, weights_only=True))
    _equal(world.state_dict(), restored.state_dict())
    if pre_action:
        assert torch.equal(world.oracle(), restored.oracle())
        world.act(world.oracle())
        restored.act(restored.oracle())
    for tick in range(80):
        if tick == 20:
            world.set_family(3)
            restored.set_family(3)
        assert torch.equal(world.observe(), restored.observe())
        assert torch.equal(world.oracle(), restored.oracle())
        actions = (torch.arange(12).reshape(3, 4) + tick) % 13 - 1
        world.act(actions)
        restored.act(actions)
        _equal(world.state_dict(), restored.state_dict())


def test_exogenous_raw_context_stream_and_scheduler_do_not_depend_on_actions_or_global_rng():
    left = _world(batch_size=2, n_jobs=4, active_jobs=4, family_allowed=(0, 2, 3), replace=0.3, cancel=0.3)
    right = left.clone()
    python_before, numpy_before, torch_before = random.getstate(), np.random.get_state(), torch.get_rng_state()
    for _ in range(90):
        left.observe()
        right.observe()
        assert left.event_hash == right.event_hash
        assert torch.equal(left.event_kind, right.event_kind)
        assert torch.equal(left.event_family, right.event_family)
        assert torch.equal(left.raw_event_payload, right.raw_event_payload)
        _equal(left.state_dict()["scheduler"], right.state_dict()["scheduler"])
        left.act(left.oracle())
        right.act(torch.full((2, 4), 10))
    assert random.getstate() == python_before
    assert np.array_equal(np.random.get_state()[1], numpy_before[1])
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_actor_job_context_prefix_is_independent_of_other_lanes_and_batch_size():
    solo = _world(seed=3, actor_ids=(42,), family_allowed=(1, 2, 3), gap_range=(0.5, 0.5))
    batch = _world(seed=3, batch_size=3, actor_ids=(8, 42, 20), family_allowed=(1, 2, 3), gap_range=(0.5, 0.5))
    for _ in range(110):
        a, b = solo.observe(), batch.observe()
        assert torch.equal(a[0], b[1])
        assert torch.equal(solo.event_family[0], batch.event_family[1])
        solo.act(solo.oracle())
        batch.act(batch.oracle())


def test_restore_is_strict_and_does_not_alias_saved_state():
    world = _world(future_family=2)
    _first_new(world)
    state = world.state_dict()
    restored = ContextWorld.from_state_dict(state)
    state["context"]["tensors"]["raw_payload"].zero_()
    state["context"]["assigned"].fill_(-1)
    assert int(restored.current_family[0, 0]) == 2
    assert int(restored._assigned_context[0, 0]) == 2
    for field in ("rng_state", "assigned", "history", "family_allowed"):
        broken = deepcopy(world.state_dict())
        broken["context"].pop(field)
        with pytest.raises(ValueError):
            ContextWorld.from_state_dict(broken)
    with pytest.raises(ValueError):
        ContextWorld.from_state_dict(OnlineWorld(batch_size=1).state_dict())


def test_load_rejects_inconsistent_raw_target_context_history_and_dtypes():
    world = _world(future_family=3)
    _first_new(world)
    changes = (
        lambda s: s["context"]["tensors"]["raw_payload"].add_(1).remainder_(4),
        lambda s: s["context"]["history"].clear(),
        lambda s: s["context"]["assigned"].fill_(1),
        lambda s: s["tensors"].update(payload=s["tensors"]["payload"].float()),
        lambda s: s["spec"].update(dt=-1.0),
    )
    for mutate in changes:
        saved = world.state_dict()
        mutate(saved)
        with pytest.raises(ValueError):
            ContextWorld.from_state_dict(saved)


def test_family_switch_changes_metadata_only_and_requires_action_boundary():
    world = _world()
    before = world.state_dict()
    world.set_family(2)
    after = world.state_dict()
    _equal(before["scheduler"], after["scheduler"])
    _equal(before["tensors"], after["tensors"])
    assert before["event_hash"] == after["event_hash"]
    assert before["context"]["rng_state"].equal(after["context"]["rng_state"])
    world.observe()
    with pytest.raises(ValueError):
        world.set_family(1)


def test_matured_family_survives_simultaneous_new_after_long_clock_step():
    world = _world(future_family=3, gap_range=(0.5, 0.5))
    _first_new(world)
    world.act(world.oracle())
    world.set_family(1)
    world.set_dt(10.0)
    world.observe()
    assert world.matured_mask[0, 0]
    assert world.matured_family[0, 0] == 3
    assert world.event_kind[0, 0] == NEW
    assert world.current_family[0, 0] == 1


def test_family_recency_retains_exact_withdrawn_family_after_every_slot_changes():
    world = _world(n_jobs=4, active_jobs=4, future_family=0, gap_range=(0.5, 0.5))
    observed_last_tick, observed_last_time = -1, -1.0
    for _ in range(12):
        world.observe()
        command = (world.event_kind == NEW) | (world.event_kind == REPLACE)
        if torch.any(command & (world.event_family == 0)):
            observed_last_tick, observed_last_time = world.tick, world.now
        world.act(world.oracle())
    assert torch.all(world.current_family == 0)
    world.set_family(1)
    for _ in range(50):
        world.observe()
        world.act(world.oracle())
    assert torch.all(world.current_family == 1)
    assert torch.all(world.last_external_tick > observed_last_tick)
    recency = world.family_recency()
    assert recency["last_command_tick"][0] == observed_last_tick
    assert recency["last_command_time"][0] == observed_last_time
    assert recency["last_command_tick"][1] > observed_last_tick
    assert recency["last_command_tick"][2:] == [-1, -1]
    assert world.metrics()["family_recency"] == recency
    restored = ContextWorld.from_state_dict(world.state_dict())
    assert restored.family_recency() == recency
    _equal(world.state_dict(), restored.state_dict())


def test_family_command_recency_includes_replace_and_strictly_validates_vectors():
    world = _world(future_family=3, replace=1)
    _first_new(world)
    first_tick = world.tick
    world.act(world.oracle())
    for _ in range(16):
        world.observe()
        world.act(world.oracle())
        if world.event_kind[0, 0] == REPLACE:
            break
    assert world.event_kind[0, 0] == REPLACE
    assert world.family_recency()["last_command_tick"][3] == world.tick > first_tick
    assert world.family_recency()["last_command_time"][3] == world.now
    for name in ("family_last_command_tick", "family_last_command_time"):
        state = world.state_dict()
        state["context"]["tensors"][name] = state["context"]["tensors"][name][None]
        with pytest.raises(ValueError):
            ContextWorld.from_state_dict(state)
    state = world.state_dict()
    state["context"]["tensors"]["family_last_command_tick"][3] = -1
    with pytest.raises(ValueError):
        ContextWorld.from_state_dict(state)


def test_eval_forks_complete_context_state_native_actions_and_preserves_source():
    torch.manual_seed(74)
    model = AgentModel(input_size=56, n_jobs=2, width=12)
    world = _world(future_family=3, replace=1)
    hidden = model.initial_state(1)
    for _ in range(32):
        x = world.observe()
        logits, hidden = model.step(x, hidden, world.elapsed)
        world.act(logits.argmax(-1))
        if world.event_kind[0, 0] == NEW:
            break
    before = state_digest({"world": world.state_dict(), "model": model.state_dict(), "h": hidden})
    torch_before = torch.get_rng_state().clone()
    record = evaluate(model, world, hidden, 1, "long_wait", 50, keep_trace=True)
    assert state_digest({"world": world.state_dict(), "model": model.state_dict(), "h": hidden}) == before
    assert torch.equal(torch_before, torch.get_rng_state())
    assert record["family"] == 1
    assert record["intervention"] == "none"
    assert record["action_policy"] == "model"
    assert record["world_resets"] == record["hidden_resets"] == record["parameter_updates"] == 0
    assert record["observation_hash"] == record["raw_observation_hash"]
    assert record["action_hash"] == record["model_action_hash"]
    assert record["carryover_contract_eligible_issued"] >= 1
    assert record["actual_family_exposures"]["replace"][3] >= 1
    assert len(record["trace"]) == 50
    assert record["regime"]["duration_range"] == (8.0, 16.0)
    assert set(CONDITIONS) == {"short", "narrow", "long_wait", "long_idle", "long_both", "far"}


@pytest.mark.parametrize("condition", tuple(CONDITIONS))
def test_all_eval_conditions_keep_native_28_feature_observation(condition):
    model = AgentModel(input_size=56, n_jobs=2, width=4)
    world = _world()
    record = evaluate(model, world, model.initial_state(1), 2, condition, 2)
    assert record["model"]["input_size"] == 56
    assert record["family_mapping"] == list(FAMILY_MAPPINGS[2])
