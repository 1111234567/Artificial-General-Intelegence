from dataclasses import replace

import numpy as np
import pytest

from experiments.pulse_game.scout_world import (
    SCOUT_HIT,
    SCOUT_OBS_SIZE,
    SCOUT_RANGE,
    SCOUT_SIDE,
    ScoutHistoryPolicy,
    ScoutOraclePolicy,
    ScoutWorld,
    make_context_replay,
    make_scout_tape,
    rollout_scout,
)
from experiments.pulse_game.world import (
    ACTIVE,
    CUE,
    CTX,
    CONTACT_ID,
    LEFT_DISTANCE,
    LEFT_ID,
    OBS_SIZE,
    POSITION,
    REWARD_NEG,
    REWARD_POS,
    RIGHT_DISTANCE,
    RIGHT_ID,
    TICKS,
    TRIAL_START,
    VELOCITY,
    HistoryPolicy,
    NoopPolicy,
    OraclePolicy,
    PulseWorld,
    RandomPolicy,
    assert_unseen,
    make_tape,
    make_yoke,
    rollout,
    swap_sides,
)


def test_seed_batching_prefixes_and_training_isolation():
    train = make_tape([5, 7, 13], distance_jitter=0.2, wind_scale=0.12, map_codes=[0, 2, 3, 8])
    prefix = make_tape([13, 5], trials=9, distance_jitter=0.2, wind_scale=0.12, map_codes=[0, 2, 3, 8])
    for name in ("ctx", "cue", "left_id", "good_id", "distances", "wind"):
        np.testing.assert_array_equal(getattr(train, name)[[2, 0], :9], getattr(prefix, name))
    np.testing.assert_array_equal(train.good_map[:, 0] + train.good_map[:, 1], 1)
    np.testing.assert_array_equal(train.good_id, train.good_map[np.arange(train.n)[:, None], train.ctx, train.cue])
    np.testing.assert_array_equal(train.right_id + train.left_id, 1)
    with pytest.raises(ValueError, match="share life seeds"):
        assert_unseen(train, prefix)

    unseen = make_tape([30, 31, 32], map_codes=[1, 4, 5, 15])
    assert_unseen(train, unseen, disjoint_maps=True)
    with pytest.raises(ValueError, match="share map codes"):
        assert_unseen(make_tape([5], map_codes=[2]), make_tape([6], map_codes=[2]), disjoint_maps=True)


def test_pulses_change_acceleration_and_both_cancel_net_motion():
    world = PulseWorld(make_tape([8, 9, 10], trials=1))
    obs = world.observe()
    assert obs.shape == (3, OBS_SIZE)
    assert obs.dtype == np.float32
    np.testing.assert_array_equal(obs[:, [POSITION, VELOCITY]], 0)
    np.testing.assert_array_equal(obs[:, [ACTIVE, TRIAL_START]], 1)

    obs = world.step([[1, 0], [0, 1], [1, 1]])
    np.testing.assert_allclose(obs[:, POSITION], [-0.15, 0.15, 0])
    np.testing.assert_allclose(obs[:, VELOCITY], [-0.15, 0.15, 0])
    obs = world.step([[1, 1], [0, 0], [1, 1]])
    np.testing.assert_allclose(obs[:, POSITION], [-0.2475, 0.2475, 0])
    np.testing.assert_allclose(obs[:, VELOCITY], [-0.0975, 0.0975, 0])
    np.testing.assert_array_equal(obs[:, REWARD_POS:CONTACT_ID.stop], 0)


def test_contact_is_causal_and_has_one_shot_identity_and_sign():
    tape = make_tape([10, 11], trials=1, map_codes=[0])
    result = rollout(OraclePolicy(tape), tape, record=True)
    assert result.obs.shape == (2, 1, TICKS, OBS_SIZE)
    np.testing.assert_array_equal(result.contact_tick, 3)
    np.testing.assert_array_equal(result.contact_id, tape.good_id)
    np.testing.assert_array_equal(result.contact_side, (tape.left_id != tape.good_id))
    np.testing.assert_array_equal(result.obs[:, :, :4, REWARD_POS:CONTACT_ID.stop], 0)
    np.testing.assert_array_equal(result.obs[:, :, 4, REWARD_POS], 1)
    np.testing.assert_array_equal(result.obs[:, :, 4, REWARD_NEG], 0)
    np.testing.assert_array_equal(result.obs[:, :, 4, CONTACT_ID].argmax(axis=-1), result.contact_id)
    np.testing.assert_array_equal(result.obs[:, :, 5:, REWARD_POS:CONTACT_ID.stop], 0)
    np.testing.assert_array_equal(result.obs[:, :, 4:, ACTIVE], 0)
    np.testing.assert_array_equal(result.obs[:, :, 4:, VELOCITY], 0)
    for field in (CTX, CUE, LEFT_ID, RIGHT_ID, POSITION, LEFT_DISTANCE, RIGHT_DISTANCE):
        settled = result.obs[:, :, 4:, field]
        np.testing.assert_array_equal(settled, np.repeat(settled[:, :, :1], TICKS - 4, axis=2))


def test_last_legal_contact_delivers_feedback_before_next_trial():
    tape = make_tape([7, 8], trials=2)
    world = PulseWorld(tape)
    for _ in range(11):
        obs = world.step([[0, 0], [0, 0]])
        np.testing.assert_array_equal(obs[:, ACTIVE], 1)
    for _ in range(4):
        obs = world.step([[1, 0], [0, 0]])
    assert world.tick_index == 15
    np.testing.assert_array_equal(obs[:, ACTIVE], 0)
    assert obs[0, CONTACT_ID].argmax() == tape.left_id[0, 0]
    assert bool(obs[0, REWARD_POS]) == bool(tape.left_id[0, 0] == tape.good_id[0, 0])
    assert obs[0, REWARD_POS] + obs[0, REWARD_NEG] == 1
    np.testing.assert_array_equal(obs[1, REWARD_POS:CONTACT_ID.stop], 0)

    obs = world.step([[0, 1], [0, 1]])
    assert world.trial_index == 1
    assert world.tick_index == 0
    np.testing.assert_array_equal(obs[:, [POSITION, VELOCITY, REWARD_POS, REWARD_NEG]], 0)
    np.testing.assert_array_equal(obs[:, [ACTIVE, TRIAL_START]], 1)
    for _ in range(TICKS):
        obs = world.step([[0, 0], [0, 0]])
    assert obs is None
    result = world.result()
    np.testing.assert_array_equal(result.contact_tick, [[14, -1], [-1, -1]])
    np.testing.assert_array_equal(result.motor_pulses[:, 0], [[4, 0], [0, 0]])


def test_hidden_preferences_change_only_the_result_for_fixed_motor_history():
    seeds = range(20, 40)
    a = make_tape(seeds, trials=12, map_codes=[0])
    b = make_tape(seeds, trials=12, map_codes=[15])
    result_a = rollout(RandomPolicy(seed=42), a, record=True)
    result_b = rollout(RandomPolicy(seed=42), b, record=True)
    assert result_a.contact_rate == result_b.contact_rate == 1
    np.testing.assert_array_equal(result_a.contact_id, result_b.contact_id)
    np.testing.assert_array_equal(result_a.pulses, result_b.pulses)
    public = [i for i in range(OBS_SIZE) if i not in (REWARD_POS, REWARD_NEG)]
    np.testing.assert_array_equal(result_a.obs[..., public], result_b.obs[..., public])
    np.testing.assert_array_equal(result_a.reward, -result_b.reward)


def test_open_loop_donor_replay_preserves_reward_timing_and_focal_outcomes():
    tape = make_tape([3, 8, 17, 40, 50, 97], trials=12, distance_jitter=0.2)
    donor = rollout(RandomPolicy(seed=51), tape)
    yoke = make_yoke(donor, seed=92)
    assert set(yoke.donor_rows) == set(range(tape.n))
    assert all(donor_seed != focal_seed for donor_seed, focal_seed in zip(yoke.donor_seeds, tape.seeds))
    np.testing.assert_array_equal(np.sort(yoke.reward, axis=0), np.sort(donor.observed_reward, axis=0))

    silent_body = rollout(NoopPolicy(), tape, feedback="yoked", yoke=yoke, record=True)
    np.testing.assert_array_equal(silent_body.contact_id, -1)
    np.testing.assert_array_equal(silent_body.reward, 0)
    np.testing.assert_array_equal(silent_body.obs[..., CONTACT_ID], 0)
    np.testing.assert_array_equal(silent_body.observed_reward, yoke.reward)
    np.testing.assert_array_equal(
        silent_body.obs[..., REWARD_POS] - silent_body.obs[..., REWARD_NEG], yoke.reward,
    )
    moving_body = rollout(RandomPolicy(seed=99), tape, feedback="yoked", yoke=yoke, record=True)
    matched = rollout(RandomPolicy(seed=99), tape, record=True)
    np.testing.assert_array_equal(moving_body.contact_id, matched.contact_id)
    np.testing.assert_array_equal(moving_body.reward, matched.reward)
    np.testing.assert_array_equal(moving_body.obs[..., CONTACT_ID], matched.obs[..., CONTACT_ID])
    np.testing.assert_array_equal(moving_body.observed_reward, yoke.reward)


def test_public_history_retains_learning_across_trials_and_swapped_sides():
    tape = make_tape(range(50, 82), distance_jitter=0.22, wind_scale=0.12)
    swapped = swap_sides(tape, first_trial=32)
    np.testing.assert_array_equal(swapped.good_id, tape.good_id)
    np.testing.assert_array_equal(swapped.left_id[:, :32], tape.left_id[:, :32])
    np.testing.assert_array_equal(swapped.left_id[:, 32:], 1 - tape.left_id[:, 32:])
    assert rollout(NoopPolicy(), swapped).contact_rate == 0
    assert rollout(OraclePolicy(swapped), swapped).accuracy == 1

    random = rollout(RandomPolicy(seed=32), swapped)
    assert random.contact_rate == 1
    assert 0.45 < random.accuracy < 0.55
    history = rollout(HistoryPolicy(seed=32), swapped)
    assert history.contact_rate == 1
    assert history.accuracy > 0.9
    for i in range(tape.n):
        seen = set()
        for j, condition in enumerate(zip(tape.ctx[i], tape.cue[i])):
            if condition in seen:
                assert history.correct[i, j]
            seen.add(condition)
    np.testing.assert_array_equal(history.contact_id, rollout(HistoryPolicy(seed=32), tape).contact_id)

    hidden = rollout(HistoryPolicy(seed=32), swapped, feedback="none")
    np.testing.assert_array_equal(hidden.observed_reward, 0)
    np.testing.assert_array_equal(hidden.contact_id, random.contact_id)
    late = rollout(HistoryPolicy(seed=32), swapped, feedback_from_trial=32)
    np.testing.assert_array_equal(late.observed_reward[:, :32], 0)
    np.testing.assert_array_equal(late.contact_id[:, :32], random.contact_id[:, :32])
    assert late.correct[:, 32:].mean() > 0.85


def test_policy_resets_once_and_receives_every_inter_trial_slot():
    class Counter(NoopPolicy):
        def __init__(self):
            self.resets = 0
            self.steps = 0
            self.starts = 0

        def reset(self, n):
            self.resets += 1
            super().reset(n)

        def act(self, obs):
            self.steps += 1
            self.starts += int(obs[0, TRIAL_START])
            return super().act(obs)

    policy = Counter()
    tape = make_tape([62, 63, 64])
    rollout(policy, tape)
    assert policy.resets == 1
    assert policy.steps == tape.trials * TICKS
    assert policy.starts == tape.trials


def test_scout_does_not_reveal_context_until_physical_crossing():
    tape = make_scout_tape(range(8), trials=1)
    np.testing.assert_array_equal(tape.sensitive.sum(axis=1), 2)
    np.testing.assert_array_equal(tape.good_map[:, 0] != tape.good_map[:, 1], tape.sensitive)
    with pytest.raises(ValueError, match="use the scout world"):
        PulseWorld(tape)
    world = ScoutWorld(tape)
    other = ScoutWorld(replace(tape, ctx=1 - tape.ctx))
    start = world.observe()
    assert start.shape == (8, SCOUT_OBS_SIZE)
    np.testing.assert_array_equal(start, other.observe())
    np.testing.assert_array_equal(start[:, CTX], 0)
    np.testing.assert_array_equal(start[:, SCOUT_SIDE].argmax(axis=1), tape.scout_side[:, 0])
    np.testing.assert_allclose(start[:, SCOUT_RANGE], 0.35)

    pulse = np.eye(2)[tape.scout_side[:, 0]]
    before = world.step(pulse)
    np.testing.assert_array_equal(before, other.step(pulse))
    np.testing.assert_array_equal(before[:, SCOUT_HIT], 0)
    after = world.step(pulse)
    counterfactual = other.step(pulse)
    np.testing.assert_array_equal(after[:, CTX].argmax(axis=1), tape.ctx[:, 0])
    np.testing.assert_array_equal(after[:, CTX] + counterfactual[:, CTX], 1)
    np.testing.assert_array_equal(after[:, SCOUT_HIT], 1)
    np.testing.assert_array_equal(after[:, REWARD_POS:CONTACT_ID.stop], 0)
    np.testing.assert_array_equal(after[:, ACTIVE], 1)
    settled = world.step(pulse)
    np.testing.assert_array_equal(settled[:, CTX], 0)
    np.testing.assert_array_equal(settled[:, SCOUT_HIT], 0)
    np.testing.assert_array_equal(settled[:, REWARD_POS:CONTACT_ID.stop], 0)
    endpoint = world.step(pulse)
    np.testing.assert_array_equal(endpoint[:, REWARD_POS] + endpoint[:, REWARD_NEG], 1)


def test_scout_sensor_controls_preserve_physics_and_recipient_scoring():
    tape = make_scout_tape(range(156, 172), trials=19, distance_jitter=0.22, wind_scale=0.12)
    live = rollout_scout(RandomPolicy(42), tape, record=True)
    off = rollout_scout(RandomPolicy(42), tape, sensor="none", record=True)
    replay = make_context_replay(live, seed=71)
    yoke = rollout_scout(RandomPolicy(42), tape, sensor="replay", replay=replay, record=True)
    np.testing.assert_array_equal(off.observed_context, -1)
    np.testing.assert_array_equal(yoke.observed_context, replay.context)
    np.testing.assert_array_equal(np.sort(replay.context, axis=0), np.sort(live.observed_context, axis=0))
    assert np.any((replay.context >= 0) & ~live.obs[..., SCOUT_HIT].astype(bool))
    assert np.any((replay.context >= 0) & (replay.context != tape.ctx[..., None]))
    for controlled in (off, yoke):
        np.testing.assert_array_equal(live.obs[..., CTX.stop:], controlled.obs[..., CTX.stop:])
        np.testing.assert_array_equal(live.scout_tick, controlled.scout_tick)
        np.testing.assert_array_equal(live.contact_tick, controlled.contact_tick)
        np.testing.assert_array_equal(live.reward, controlled.reward)


def test_scout_upper_bounds_and_public_history_on_unseen_perturbed_lives():
    train = make_scout_tape(range(15100, 15109))
    heldout = make_scout_tape(range(20200, 20264), distance_jitter=0.22, wind_scale=0.12)
    assert_unseen(train, heldout)
    tape = swap_sides(heldout, first_trial=32)
    oracle = rollout_scout(ScoutOraclePolicy(tape), tape)
    assert oracle.accuracy == oracle.contact_rate == 1
    assert oracle.probed[oracle.sensitive_trial].all()
    assert 0.20 < oracle.reverse_rate < 0.30
    assert np.min(oracle.latency_ticks[oracle.reversed]) > np.median(oracle.latency_ticks[~oracle.reversed])
    np.testing.assert_array_equal(oracle.timed_out, False)

    blind = rollout_scout(ScoutOraclePolicy(tape, probe=False), tape, sensor="none")
    assert blind.correct[~blind.sensitive_trial].all()
    assert 0.45 < blind.correct[blind.sensitive_trial].mean() < 0.55
    assert 0.72 < blind.accuracy < 0.78
    assert blind.reverse_rate == 0

    learned = rollout_scout(ScoutHistoryPolicy(901), tape)
    no_sensor = rollout_scout(ScoutHistoryPolicy(901), tape, sensor="none")
    assert learned.contact_rate == no_sensor.contact_rate == 1
    assert learned.accuracy > 0.92
    assert learned.correct[:, -16:].mean() > 0.99
    assert 0.7 < no_sensor.accuracy < 0.78
    silent = rollout_scout(NoopPolicy(), tape)
    assert silent.probe_rate == silent.contact_rate == 0
    np.testing.assert_array_equal(silent.latency_ticks, -1)
