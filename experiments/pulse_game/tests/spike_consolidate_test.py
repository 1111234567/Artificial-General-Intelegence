"""Regression checks for original trajectories and the eligibility filter."""

import numpy as np

from experiments.pulse_game.spike_consolidate import ConsolidatingLIF, Consolidation
from experiments.pulse_game.spike_policy import SpikeConfig, SpikingPolicy
from experiments.pulse_game.world import PulseWorld, make_tape


def test_ordinary_mechanism_preserves_frozen_controller():
    cfg = SpikeConfig(use_residual=True)
    tape = make_tape([97100], 16)
    world = PulseWorld(tape)
    baseline = SpikingPolicy(cfg, network_seed=3, action_seed=761)
    new = ConsolidatingLIF(cfg, Consolidation(), network_seed=3, action_seed=761)
    obs = world.observe()
    while obs is not None:
        reference = baseline.act(obs[0])
        np.testing.assert_array_equal(new.act(obs[0]), reference)
        np.testing.assert_array_equal(new.weights, baseline.weights)
        obs = world.step(reference[None, :])
    assert len(new.write_norms) == 16


def test_lateral_antihebb_decreases_exposed_activity_direction():
    cfg = SpikeConfig(use_residual=True)
    rule = Consolidation(kind="decorrelated", covariance_rate=.15, direction="raw")
    policy = ConsolidatingLIF(cfg, rule, network_seed=3, action_seed=761)
    activation = np.random.default_rng(678).standard_normal(cfg.units).astype(np.float32)
    original = float(activation @ policy.lateral @ activation)
    policy.command_spikes[0] = 1
    for _ in range(48):
        policy._filter_eligibility(activation)
    final = float(activation @ policy.lateral @ activation)
    assert .0 < final < .03 * original
    np.testing.assert_allclose(policy.lateral, policy.lateral.T, rtol=0, atol=1e-7)
    assert np.linalg.eigvalsh(policy.lateral).min() > 0
    assert np.count_nonzero(policy.eligibility[~policy.synapses]) == 0
    assert not policy.eligibility[cfg.heads:].any()


def test_scalar_entry_points_all_apply_the_lateral_update():
    cfg = SpikeConfig(use_residual=True)
    rule = Consolidation(kind="decorrelated", covariance_rate=.055)
    agents = [ConsolidatingLIF(cfg, rule, network_seed=11, action_seed=761) for _ in range(3)]
    world = PulseWorld(make_tape([97100], 2))
    obs = world.observe()
    while obs is not None:
        reference = agents[0].act(obs[0])
        np.testing.assert_array_equal(agents[1].step(obs[0]), reference)
        np.testing.assert_array_equal(agents[2](obs[0]), reference)
        obs = world.step(reference[None, :])
    for agent in agents[1:]:
        np.testing.assert_array_equal(agent.lateral, agents[0].lateral)
        np.testing.assert_array_equal(agent.weights, agents[0].weights)
        assert agent.covariance_updates == 2


def test_write_budget_matches_realized_clipped_norm():
    cfg = SpikeConfig(use_residual=True)
    policy = ConsolidatingLIF(cfg, Consolidation(), network_seed=3, action_seed=761)
    policy.write_budget = .24
    initial = np.array([.64, .03, .00], dtype=np.float32)
    gradient = np.array([.1, .06, .02], dtype=np.float32)
    revised = policy._within_budget(initial, gradient)
    assert np.all(np.abs(revised) <= .65)
    np.testing.assert_allclose(np.linalg.norm(revised - initial), .24, rtol=1e-5)
