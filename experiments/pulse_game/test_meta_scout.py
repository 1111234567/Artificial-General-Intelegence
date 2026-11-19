import numpy as np
import torch

from experiments.pulse_game import world
from experiments.pulse_game.meta_scout_env import map_parity, randomly_change_one_context_rule
from experiments.pulse_game.meta_scout_model import ScoutMetaPolicy, ScoutModelConfig, ScoutRawAgent
from experiments.pulse_game.meta_scout_train import (
    actual_body_prediction_loss, collect, discounted_public_returns, make_batch, observed_body_rewards, parser, self_labels, sequence,
    targets_from_public, train, tried_endpoint_loss, value_progress_rewards,
)
from experiments.pulse_game.scout_world import SCOUT_HIT, SCOUT_OBS_SIZE, SCOUT_SIDE, ScoutWorld, rollout_scout


def test_receipt_targets_require_an_earlier_public_context_and_receipt():
    obs = np.zeros((1, 4, world.TICKS, SCOUT_OBS_SIZE), dtype=np.float32)
    obs[..., 2] = 1
    obs[0, 0, 4, 0] = 1
    obs[0, 0, 8, [world.REWARD_POS, 15]] = 1
    obs[0, 1, 8, [world.REWARD_NEG, 15]] = 1
    obs[0, 2, 4, 1] = 1
    obs[0, 2, 8, [world.REWARD_NEG, 15]] = 1

    targets = targets_from_public(obs, "cpu")
    assert targets.known[0].tolist() == [[False, False], [True, False], [True, False], [True, True]]
    assert targets.context_map[0, 3].tolist() == [0.0, 1.0]
    assert int(targets.context[0, 0, 3]) == 0
    assert int(targets.context[0, 0, 8]) == 1
    assert int(targets.context[0, 1, 8]) == 0
    assert int(targets.context[0, 2, 8]) == 2


def test_raw_scout_model_replays_native_physical_actions():
    torch.manual_seed(9983)
    model = ScoutMetaPolicy(ScoutModelConfig(hidden=16, degree=4, projection=16, keys=8, values=6))
    tape = make_batch(8240199, 6, 4)
    recorded = rollout_scout(ScoutRawAgent(model), tape, record=True)
    obs = torch.from_numpy(recorded.obs)
    with torch.no_grad():
        logits, *_ = sequence(model, obs)
    selected = logits.argmax(dim=-1).numpy()
    expected = np.where(recorded.pulses[..., 0], 1, np.where(recorded.pulses[..., 1], 2, 0))
    np.testing.assert_array_equal(selected, expected)
    assert int(model.edges.sum()) == model.config.degree * model.config.hidden


def test_student_collects_own_history_and_teacher_is_a_label_only():
    class Silent:
        def reset(self, n):
            self.pulses = np.zeros((n, 2), bool)

        def act(self, obs):
            return self.pulses

    tape = make_batch(8370161, 8, 5)
    examples = collect(tape, seed=9818, student=Silent(), teacher_mix=0)
    assert not examples.actions.any()
    assert examples.targets.any()
    assert examples.metrics["contact"] == 0.0
    assert not examples.obs[..., world.CTX].any()


def test_scout_fast_intervention_does_not_change_observation_shape():
    model = ScoutMetaPolicy(ScoutModelConfig(hidden=16, degree=4, projection=16, keys=8, values=6))
    tape = make_batch(8370161, 8, 5)
    agent = ScoutRawAgent(model, write_enabled=False)
    env = ScoutWorld(tape)
    agent.reset(tape.n)
    obs = env.observe()
    while obs is not None:
        assert obs.shape == (tape.n, SCOUT_OBS_SIZE)
        obs = env.step(agent.act(obs))
    assert not agent.state.fast.any()


def test_counterfactual_training_keeps_the_physical_trajectory_state_intact():
    torch.manual_seed(38801)
    model = ScoutMetaPolicy(ScoutModelConfig(hidden=12, degree=3, projection=12, keys=8, values=6))
    obs = torch.zeros((3, 2, world.TICKS, SCOUT_OBS_SIZE))
    obs[:, :, 5, 0] = 1
    alternate = torch.full((3, 2 * world.TICKS), -1)
    alternate[:, 5] = 2
    normal = sequence(model, obs)
    forked = sequence(model, obs, alternate)
    for before, after in zip(normal, forked):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert forked[-1].isfinite().item()
    assert forked[-1].requires_grad


def test_private_training_variants_preserve_the_observation_schedule():
    tape = make_batch(81884418, 64, 21)
    even = map_parity(tape, 0)
    odd = map_parity(tape, 1)
    assert np.all(even.good_map[:, 0].sum(axis=1) % 2 == 0)
    assert np.all(odd.good_map[:, 0].sum(axis=1) % 2 == 1)
    assert not set(even.map_codes).intersection(odd.map_codes)
    changed = randomly_change_one_context_rule(even, 1.0)
    np.testing.assert_array_equal(even.good_id[:, :7], changed.good_id[:, :7])
    assert np.any(even.good_id != changed.good_id)
    for variant in (even, odd, changed):
        for field in ("ctx", "cue", "left_id", "scout_side", "distances", "wind"):
            np.testing.assert_array_equal(getattr(variant, field), getattr(tape, field))


def test_scout_masked_teacher_histories_never_restore_missing_rewards():
    tape = make_batch(81884418, 12, 9)
    first = collect(map_parity(tape, 0), seed=1118, receipt_dropout=1.0)
    second = collect(map_parity(tape, 1), seed=1118, receipt_dropout=1.0)
    np.testing.assert_array_equal(first.obs, second.obs)
    np.testing.assert_array_equal(first.targets, second.targets)
    assert first.metrics["contact"] > 0.95
    assert not targets_from_public(first.obs, "cpu").known.any()


def test_self_label_has_no_unknown_post_probe_target_without_a_real_contact():
    obs = np.zeros((1, 1, world.TICKS, SCOUT_OBS_SIZE), np.float32)
    obs[:, :, :7, world.ACTIVE] = 1
    obs[..., 2] = 1
    obs[..., 6] = 1
    obs[..., 21] = 1  # Only the right-hand beacon is publicly available.
    obs[:, :, 4, 0] = 1
    result, weight = self_labels(torch.from_numpy(obs), targets_from_public(obs, "cpu"))
    assert result[0, 0, :4].tolist() == [2, 2, 2, 2]
    assert weight[0, 0, 4:7].tolist() == [0.0, 0.0, 0.0]


def test_predicted_return_has_no_counterfactual_endpoint_target():
    obs = torch.zeros((1, 2, world.TICKS, SCOUT_OBS_SIZE))
    obs[:, :, :7, world.ACTIVE] = 1
    obs[:, :, 4, 0] = 1
    obs[0, 0, 7, 15] = 1
    obs[0, 0, 7, world.REWARD_NEG] = 1
    obs[0, 1, 7, world.REWARD_POS] = 1  # No observed contact identity in this fixture.
    scores = torch.zeros((1, 2, world.TICKS, 2), requires_grad=True)
    tried_endpoint_loss(scores, obs).backward()
    assert (scores.grad[0, 0, :7, 0] > 0).all()
    assert not scores.grad[0, 0, :, 1].any()
    assert not scores.grad[0, 1].any()


def test_post_scout_exploration_is_only_logged_after_physical_probe():
    class ProbeDuringFixture:
        def reset(self, n):
            self.n = n

        def logits(self, obs):
            scores = torch.zeros((self.n, 3))
            scores[torch.arange(self.n), 1 + torch.from_numpy(obs[:, SCOUT_SIDE].argmax(axis=1))] = 9
            return scores

    tape = make_batch(81884418, 6, 4)
    batch = collect(tape, seed=3618, student=ProbeDuringFixture(), teacher_targets=False, temperature=0.4,
                    post_scout_noise=1.2)
    seen = np.maximum.accumulate(batch.obs[..., SCOUT_HIT] > 0, axis=2)
    assert np.any(seen)
    assert not batch.action_bias[~seen].any()
    assert np.any(batch.action_bias[seen] != 0)
    np.testing.assert_array_equal(batch.action_bias[..., 1], -batch.action_bias[..., 2])


def test_negative_indexed_receipt_changes_only_the_physically_tried_value():
    model = ScoutMetaPolicy(ScoutModelConfig(hidden=12, degree=3, projection=12, keys=8, values=2,
                                            indexed_outcomes=True))
    with torch.no_grad():
        model.associate[-1].bias[-1] = 2
    obs = torch.zeros((1, SCOUT_OBS_SIZE))
    obs[0, [0, 2, 16]] = 1
    initial = model.initial(1)
    unsigned = model(obs, initial)
    torch.testing.assert_close(initial.fast, unsigned.state.fast, rtol=0, atol=0)
    obs[0, world.REWARD_NEG] = 1
    negative = model(obs, initial)
    assert torch.count_nonzero(negative.state.fast[0, :, 0]) == 0
    assert bool(negative.read[0, 1] < 0)


def test_resume_scout_keeps_the_same_optimizer_trajectory(tmp_path):
    settings = [
        "--device", "cpu", "--cpu-threads", "1", "--lives", "4", "--trials", "4", "--eval-lives", "5",
        "--eval-trials", "6", "--hidden", "12", "--degree", "3", "--keys", "8", "--values", "6",
        "--projection", "12", "--forced-steps", "1", "--mixed-through", "0", "--seed", "8471",
        "--train-seed", "6260031", "--dev-seed", "6260881",
    ]
    full = tmp_path / "one_run"
    resumed = tmp_path / "resume_run"
    train(parser().parse_args([*settings, "--output", str(full), "--steps", "3"]))
    train(parser().parse_args([*settings, "--output", str(resumed), "--steps", "1"]))
    train(parser().parse_args([*settings, "--output", str(resumed), "--steps", "3", "--resume"]))
    normal = torch.load(full / "last.pt", map_location="cpu", weights_only=True)
    checkpoint = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)
    assert checkpoint["step"] == normal["step"] == 3
    for name, value in normal["model"].items():
        torch.testing.assert_close(value, checkpoint["model"][name], rtol=0, atol=0)


def test_body_warmup_uses_physical_receipt_and_actual_forecast_windows():
    obs = torch.zeros((1, 1, world.TICKS, SCOUT_OBS_SIZE))
    obs[..., 20] = 1
    obs[0, 0, 1, world.ACTIVE] = 1
    obs[0, 0, 2, world.POSITION] = 0.2
    obs[0, 0, 3, SCOUT_HIT] = 1
    obs[0, 0, 6, world.POSITION] = 0.7
    obs[0, 0, 6, 15] = 1
    obs[0, 0, 6, world.REWARD_NEG] = 1
    terms = dict(contact_bonus=0.4, scout_bonus=0.2, reverse_bonus=1.0, travel_cost=0.0)
    body = observed_body_rewards(obs, **terms)
    assert float(body[0, 0, 6]) > 1.39
    changed = obs.clone()
    changed[0, 0, 6, [world.REWARD_POS, world.REWARD_NEG]] = torch.tensor([1.0, 0.0])
    torch.testing.assert_close(observed_body_rewards(changed, **terms), body, rtol=0, atol=0)
    forecast = torch.zeros((1, 1, world.TICKS, 4), requires_grad=True)
    motion, events = actual_body_prediction_loss(forecast, obs)
    (motion + events).backward()
    assert forecast.grad[0, 0, 1, 0] < 0  # The next observed position is positive.
    assert forecast.grad[0, 0, 1, 2] < 0  # The physical sensor lies within the next three ticks.
    assert forecast.grad[0, 0, 1, 3] > 0  # The physical endpoint is outside that window.


def test_discount_can_credit_earlier_trials_without_crossing_lives():
    rewards = torch.zeros((2, 3, world.TICKS))
    rewards[0, 2, 0] = 2
    ordinary = discounted_public_returns(rewards, tick_discount=1.0, trial_discount=0.0)
    torch.testing.assert_close(ordinary, rewards)
    linked = discounted_public_returns(rewards, tick_discount=1.0, trial_discount=0.5)
    assert (linked[0, 0] == 0.5).all()
    assert (linked[0, 1] == 1.0).all()
    assert not linked[1].any()
    assert float(linked[0, 2, 1:].abs().max()) == 0.0


def test_value_progress_uses_sensed_values_and_actual_displacement_only():
    obs = torch.zeros((1, 1, world.TICKS, SCOUT_OBS_SIZE))
    obs[..., [6, 9, 18, 19, world.ACTIVE]] = 1
    obs[0, 0, 3, [0, SCOUT_HIT]] = 1
    obs[0, 0, 1, world.POSITION] = 0.1
    obs[0, 0, 3, world.POSITION] = 0.3
    obs[0, 0, 4, world.POSITION] = 0.5
    read = torch.zeros((1, 1, world.TICKS, 2))
    read[0, 0, 3] = torch.tensor([-1.0, 1.0])
    read[0, 0, 4:] = torch.tensor([1.0, -1.0])
    signal = value_progress_rewards(obs, read)
    assert not signal[:, :, :3].any()
    torch.testing.assert_close(signal[0, 0, 3], torch.tensor(0.2))
    shifted = value_progress_rewards(obs, read.flip(-1))
    torch.testing.assert_close(shifted, -signal)


def test_same_read_response_preserves_the_actual_actor_step():
    torch.manual_seed(17938)
    model = ScoutMetaPolicy(ScoutModelConfig(hidden=12, degree=3, projection=12, keys=8, values=2,
                                            readout_motor=True, indexed_outcomes=True))
    obs = torch.randn((4, SCOUT_OBS_SIZE)) * 0.1
    obs[:, world.CONTACT_ID] = 0
    previous = model.initial(4)
    previous.fast.copy_(torch.randn_like(previous.fast))
    step = model(obs, previous)
    logits, h = model.response_with_read(obs, previous.activity, model.prepared(), step.read)
    torch.testing.assert_close(logits, step.logits, rtol=0, atol=0)
    torch.testing.assert_close(h, step.state.activity, rtol=0, atol=0)
    altered, _ = model.response_with_read(obs, previous.activity, model.prepared(), step.read.flip(1))
    assert not torch.allclose(altered, logits)

