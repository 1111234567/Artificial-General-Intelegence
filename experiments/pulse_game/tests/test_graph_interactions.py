"""Exercise nonlinear node derivatives and the physical interaction-only task."""

from dataclasses import asdict

import numpy as np

from experiments.pulse_game import graph_interactions as model, mixture_world as food, world
from experiments.pulse_game.graph_interactions_study import run
from experiments.pulse_game.unified_live_graph import MOTOR, Graph, Recipe


def world_inputs():
    rng = np.random.default_rng(931)
    public = np.zeros((10, 1, world.OBS_SIZE), dtype=np.float32)
    public[..., food.LEFT.start:food.RIGHT.stop] = rng.random((10, 1, 8))
    public[:8, :, world.ACTIVE] = 1
    public[8, :, world.CONTACT_ID.start + 1] = 1
    public[8, :, world.REWARD_POS] = 1
    return public


def test_zero_scale_preserves_port_dynamics_and_gradients():
    left = Graph([96])
    right = model.SquareGraph([96], Recipe(), scale=0)
    for graph in (left, right):
        graph.start(np.array([.1], dtype=np.float32))
    for observation in world_inputs():
        np.testing.assert_array_equal(left.advance(observation), right.advance(observation))
        np.testing.assert_array_equal(left.state, right.state)
    np.testing.assert_array_equal(left.receipt, right.receipt)
    np.testing.assert_array_equal(left.side, right.side)
    np.testing.assert_array_equal(left.gradient(), right.gradient())


def test_even_node_gradient_matches_finite_differences_with_receipt_held():
    observations = world_inputs()
    full = model.SquareGraph([96], Recipe(), scale=.85)
    full.start(np.array([.1], dtype=np.float32))
    for observation in observations:
        full.advance(observation)
    gradient = full.gradient()[0]
    # One incoming edge to an ordinary node and one incoming edge to the right actuator.
    candidates = np.argwhere(full.mask[0] & (np.abs(gradient) > 1e-5))
    candidate = candidates[np.argmax(np.abs(gradient[candidates[:, 0], candidates[:, 1]]))]
    targets = ((14, 2), tuple(candidate))

    def loss(target, source, step):
        replica = model.SquareGraph([96], Recipe(), scale=.85, learning=False, editing=False)
        replica.weight[0, target, source] += step
        replica.start(np.array([.1], dtype=np.float32))
        contrasts = []
        for obs in observations:
            replica.advance(obs)
            if obs[0, world.ACTIVE]:
                motor = replica.state[0, MOTOR]
                contrasts.append(float(motor[1]) - float(motor[0]))
        amplitude = float(full.receipt[0])
        side = float(full.side[0])
        return .5 * amplitude * np.logaddexp(0, -2 * side * np.mean(contrasts))

    step = 0.001
    for target, source in targets:
        finite = (loss(target, source, -step) - loss(target, source, step)) / (2 * step)
        np.testing.assert_allclose(gradient[target, source], finite, atol=4e-6, rtol=1e-3)


def test_private_pair_weights_hide_switch_and_have_no_univariate_taste_signal():
    roots = [90, 91, 92]
    stable, change, noise, sham = model.make_menu(roots, "stable", 120)
    altered, other_change, other_noise, other_sham = model.make_menu(roots, "reverse", 120)
    for a, b in ((stable.chemicals, altered.chemicals), (stable.weights, altered.weights), (change, other_change),
                 (noise, other_noise), (sham, other_sham)):
        np.testing.assert_array_equal(a, b)
    samples = stable.chemicals * 2 - 1
    inputs = model.interactions(samples)
    np.testing.assert_allclose(model.interactions(-samples), inputs)
    for axes in ((0, 1), (2, 3)):
        antipodes = samples.copy()
        antipodes[..., axes] *= -1
        np.testing.assert_allclose(model.interactions(antipodes), -inputs)
    for lane, switch in enumerate(change):
        np.testing.assert_array_equal(stable.probability[lane, :switch], altered.probability[lane, :switch])
        np.testing.assert_allclose(stable.probability[lane, switch:], 1 - altered.probability[lane, switch:], atol=1e-7)


def test_graphs_grow_and_train_own_weights_only_with_actual_receipts():
    for even in (0, .85):
        config = dict(recipe=asdict(Recipe(grow_every=6, retire_every=5)), weight_gain=1, even=even)
        values = run([31, 32], [501, 502, 503], "reverse", "test", config, trials=80, tail=12)
        assert (values["edits"][..., 0] >= 0).any()
        assert (values["edits"][..., 1] >= 0).any()
        assert values["writes"][..., 1].sum() > 0
        np.testing.assert_array_equal(values["taste"][:, :-12], values["shown"][:, :-12])
        assert not values["shown"][:, -12:].any() and not values["writes"][:, -12:].any()
