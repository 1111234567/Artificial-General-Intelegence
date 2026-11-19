import numpy as np
import pytest
import torch

from experiments.pulse_game import world
from experiments.pulse_game.meta_env import BatchTape
from experiments.pulse_game.meta_eval import CueVisibility, WrongAddress
from experiments.pulse_game.meta_model import ModelConfig, Policy, RolloutAgent, create_policy
from experiments.pulse_game.meta_train import parser, rollout, train


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"))])
def test_torch_world_matches_reference_observations_and_outcomes(device):
    tape = world.make_tape(range(20, 40), 12, distance_jitter=0.2, wind_scale=0.24)
    batch = BatchTape.from_tape(tape, device)
    reference = world.PulseWorld(tape)
    rng = np.random.default_rng(2791)
    for index in range(12):
        trial = batch.trial(index)
        static = trial.static_observation(device, torch.float32)
        vehicle = trial.initial()
        for tick in range(16):
            got = vehicle.observe(trial, static, tick).cpu().numpy()
            np.testing.assert_array_equal(got, reference.observe())
            pulses = rng.integers(0, 2, (20, 2), dtype=np.int8)
            # Include one-sided, parked, and randomized vehicle trajectories.
            pulses[:4] = [1, 0]
            pulses[4:8] = [0, 1]
            pulses[8:12] = [0, 0]
            reference.step(pulses)
            if tick < 15:
                drive = torch.as_tensor(pulses[:, 1] - pulses[:, 0], device=device)
                vehicle = vehicle.step(trial, drive, tick)
        np.testing.assert_array_equal(vehicle.last_contact.cpu().numpy(), reference._contact_id[:, index])


def test_reference_adapter_and_batched_policy_agree():
    torch.manual_seed(7343)
    tape = world.make_tape(range(8), 6, distance_jitter=0.1, wind_scale=0.2)
    model = Policy(ModelConfig(hidden=12, degree=3, keys=8, values=4))
    cpu_run = world.rollout(RolloutAgent(model), tape)
    with torch.no_grad():
        batched_run = rollout(model, BatchTape.from_tape(tape))
    np.testing.assert_array_equal(cpu_run.correct, batched_run.hit.T.numpy())
    np.testing.assert_array_equal(cpu_run.contact_id >= 0, batched_run.contacted.T.numpy())


def test_temporal_adapter_only_sees_one_tick_of_the_cue():
    torch.manual_seed(73123)
    tape = world.make_tape(range(19, 34), 10)
    model = create_policy(ModelConfig(hidden=12, degree=4, keys=8, values=4, memory_kind="temporal"))
    adapter = CueVisibility(RolloutAgent(model), 1)
    reference = world.rollout(adapter, tape, record=True)
    with torch.no_grad():
        traced = rollout(model, BatchTape.from_tape(tape), cue_ticks=1)
    np.testing.assert_array_equal(reference.correct, traced.hit.T.numpy())
    np.testing.assert_array_equal(reference.contact_id >= 0, traced.contacted.T.numpy())

    obs = np.zeros((1, 20), dtype=np.float32)
    obs[:, [0, 4, 17]] = 1
    adapter.reset(1)
    adapter.act(obs)
    first = adapter.agent.state.trace.clone()
    # Input sent to the same agent after tick 0 is indistinguishable for two different hidden cues.
    altered = obs.copy()
    altered[:, [4, 5]] = [0, 1]
    altered[:, world.TRIAL_START] = 0
    adapter.act(altered)
    hidden_a = adapter.agent.state.trace.clone()
    adapter.reset(1)
    plain = obs.copy()
    plain[:, world.TRIAL_START] = 0
    adapter.act(obs)
    torch.testing.assert_close(adapter.agent.state.trace, first, rtol=0, atol=0)
    adapter.act(plain)
    torch.testing.assert_close(adapter.agent.state.trace, hidden_a, rtol=0, atol=0)


def test_fast_updates_require_a_public_reward_and_recurrence_is_directed():
    policy = Policy(ModelConfig(hidden=16, degree=4, keys=8, values=4))
    weights, leak = policy.prepared_recurrence()
    assert int(policy.edges.sum()) == 64
    assert torch.all(policy.edges.diag() == 0)
    assert torch.all(weights[policy.edges == 0] == 0)
    assert torch.all((leak > 0) & (leak < 1))
    assert torch.count_nonzero(policy.edges - policy.edges.T) > 0

    obs = torch.zeros((3, 20))
    obs[:, [0, 2, 6, 9, 12, 17, 18, 19]] = 1
    initial = policy.initial_state(3)
    _, neutral = policy(obs, initial)
    torch.testing.assert_close(neutral.fast, initial.fast, rtol=0, atol=0)
    obs[[1, 2], 15] = 1
    obs[1, 13] = 1
    obs[2, 14] = 1
    _, receipt = policy(obs, neutral)
    torch.testing.assert_close(receipt.fast[0], neutral.fast[0], rtol=0, atol=0)
    assert torch.count_nonzero(receipt.fast[1]) > 0
    assert torch.count_nonzero(receipt.fast[1] - receipt.fast[2]) > 0
    _, deleted = policy(obs, neutral, write_enabled=False)
    torch.testing.assert_close(deleted.fast, neutral.fast, rtol=0, atol=0)


def test_hidden_rule_has_no_effect_before_a_physical_receipt():
    tape_a = world.make_tape([818], 1, map_codes=[0])
    tape_b = world.make_tape([818], 1, map_codes=[15])
    model = Policy()
    world_a = world.rollout(RolloutAgent(model), tape_a, record=True)
    world_b = world.rollout(RolloutAgent(model), tape_b, record=True)
    assert world_a.contact_tick[0, 0] == world_b.contact_tick[0, 0]
    assert world_a.contact_tick[0, 0] >= 0
    outcome = int(world_a.contact_tick[0, 0]) + 1
    np.testing.assert_array_equal(world_a.obs[:, :, :outcome], world_b.obs[:, :, :outcome])
    np.testing.assert_array_equal(world_a.pulses[:, :, :outcome], world_b.pulses[:, :, :outcome])
    assert world_a.correct[0, 0] != world_b.correct[0, 0]


def test_masked_receipts_produce_no_preference_label_even_with_real_contacts():
    first = world.make_tape(range(7), 8, map_codes=[0])
    second = world.make_tape(range(7), 8, map_codes=[15])
    torch.manual_seed(8911)
    policy = create_policy(ModelConfig(memory_kind="temporal", aux_probe=True))
    options = dict(training=True, seed=1900, explore=0.3, cue_ticks=1, receipt_dropout=1.0,
                   memory_loss_weight=1.0, receipt_loss_weight=1.0)
    a = rollout(policy, BatchTape.from_tape(first), **options)
    b = rollout(policy, BatchTape.from_tape(second), **options)
    assert not a.known.any() and not b.known.any()
    assert a.contacted.any()
    assert torch.all((a.hit ^ b.hit) == a.contacted)
    torch.testing.assert_close(a.loss, b.loss, rtol=0, atol=0)


def test_wrong_address_changes_only_the_fast_write_after_projection():
    torch.manual_seed(177616)
    policy = create_policy(ModelConfig(memory_kind="temporal"))
    normal, altered = RolloutAgent(policy), WrongAddress(policy)
    layout = np.zeros((2, 20), np.float32)
    layout[:, [0, 6, 9, 12, 17, 18, 19]] = 1
    layout[0, 2] = 1
    layout[1, 4] = 1
    for agent in (normal, altered):
        agent.reset(2)
        agent.act(layout)
    event = layout.copy()
    event[:, :6] = 0
    event[:, [12, 17]] = 0
    event[:, [13, 15]] = 1
    x, y = normal.act(event), altered.act(event)
    np.testing.assert_array_equal(x, y)
    torch.testing.assert_close(normal.state.trace, altered.state.trace, rtol=0, atol=0)
    torch.testing.assert_close(normal.state.activity, altered.state.activity, rtol=0, atol=0)
    assert not torch.allclose(normal.state.fast, altered.state.fast)


def test_resuming_training_matches_uninterrupted_weights(tmp_path):
    settings = ["--device", "cpu", "--cpu-threads", "1", "--lives", "5", "--trials", "4", "--eval-lives", "6",
                "--eval-trials", "4", "--hidden", "12", "--degree", "3", "--keys", "8", "--values", "4",
                "--forced-steps", "1", "--seed", "88712", "--memory-loss", "1", "--receipt-loss", "1"]
    first, second = tmp_path / "first", tmp_path / "second"
    train(parser().parse_args([*settings, "--output", str(first), "--steps", "3"]))
    train(parser().parse_args([*settings, "--output", str(second), "--steps", "1"]))
    train(parser().parse_args([*settings, "--output", str(second), "--steps", "3", "--resume"]))
    a = torch.load(first / "last.pt", weights_only=True)
    b = torch.load(second / "last.pt", weights_only=True)
    assert a["step"] == b["step"] == 3
    for name in a["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
