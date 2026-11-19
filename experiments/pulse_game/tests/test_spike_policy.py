"""Closed-loop tests for credit routing and state continuity."""

import copy
from dataclasses import replace

import numpy as np

from experiments.pulse_game.spike_policy import BatchSpikingPolicy, SpikeConfig, SpikingPolicy
from experiments.pulse_game.world import CONTACT_ID, REWARD_NEG, REWARD_POS, TICKS, TRIAL_START, PulseWorld, make_tape, rollout


def _first_receipt(agent):
    world = PulseWorld(make_tape([501], trials=2))
    obs = world.observe()[0]
    while not obs[CONTACT_ID].any():
        obs = world.step(agent.act(obs)[None, :])[0]
    return world, obs


def test_only_completed_emitted_identity_receives_weight_credit():
    config = SpikeConfig(use_residual=True)
    agent = SpikingPolicy(config, network_seed=7, action_seed=11)
    _, receipt = _first_receipt(agent)
    assert np.count_nonzero(agent.weights) == 0
    assert receipt[REWARD_POS] + receipt[REWARD_NEG] == 1
    actual = int(receipt[CONTACT_ID].argmax())
    assert agent.command_spikes[actual] == 1
    wrong_identity = receipt.copy()
    wrong_identity[CONTACT_ID] = wrong_identity[CONTACT_ID][::-1]
    mistagged = copy.deepcopy(agent)
    mistagged.act(wrong_identity)
    assert not mistagged.weights.any()

    pulses = agent.act(receipt)
    by_identity = agent.weights.reshape(2, config.heads, config.units)
    assert by_identity[actual].any()
    assert not by_identity[1 - actual].any()
    assert not agent.weights[~agent.synapses].any()
    assert not pulses.any()


def test_reservoir_state_crosses_boundary_and_fires_with_reset():
    agent = SpikingPolicy(SpikeConfig(use_residual=True), network_seed=7, action_seed=11)
    world, obs = _first_receipt(agent)
    while not obs[TRIAL_START]:
        obs = world.step(agent.act(obs)[None, :])[0]
    severed = copy.deepcopy(agent)
    severed.v.fill(0)
    severed.spikes.fill(0)
    severed.trace.fill(0)
    agent.act(obs)
    severed.act(obs)
    assert agent.statistics()["ticks"] == TICKS + 1
    assert agent.statistics()["updates"] == 1
    assert not np.array_equal(agent.trace, severed.trace)
    assert agent.spikes.any()
    assert np.all(agent.v[agent.spikes.astype(bool)] == 0)


def test_batch_replays_and_freeze_prevents_signed_outcome_learning():
    tape = make_tape([9, 10], trials=24)
    config = SpikeConfig(use_residual=True)
    frozen = BatchSpikingPolicy(replace(config, learning_rate=0), network_seed=7, action_seeds=[91, 92])
    first = rollout(frozen, tape, record=True)
    replay = rollout(frozen, tape, feedback="none", record=True)
    assert first.contact_rate == 1
    np.testing.assert_array_equal(first.pulses, replay.pulses)
    for agent in frozen.agents:
        assert agent.statistics()["reservoir_spikes"] > 0
        assert agent.statistics()["weight_l2"] == 0
