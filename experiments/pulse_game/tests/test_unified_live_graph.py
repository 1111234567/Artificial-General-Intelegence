"""Check the physical feedback boundary and the mutable graph's invariants."""

import numpy as np

from experiments.pulse_game import mixture_world as food, world
from experiments.pulse_game.unified_live_graph import Graph, MOTOR, NO_CUT, Recipe, TASTE, births, strongly_connected
from experiments.pulse_game.unified_live_graph_study import make_menu, run


def test_random_birth_has_independent_multicycle_arcs():
    masks, weights, rates, biases = births([11, 12, 13, 14])
    assert (masks == (weights != 0)).all()
    for arcs, weight, rate, bias in zip(masks, weights, rates, biases, strict=True):
        assert strongly_connected(arcs)
        assert arcs.sum() > 2 * len(arcs)
        assert len(np.unique(weight[arcs])) == arcs.sum()
        assert len(np.unique(rate)) == len(rate)
        assert len(np.unique(bias)) == len(bias)


def test_motor_has_outgoing_effect_and_real_taste_hop_requires_its_edges():
    seeds = list(range(10, 18))
    noise = np.full(len(seeds), .2, dtype=np.float32)
    physical = np.zeros((len(seeds), world.OBS_SIZE), dtype=np.float32)
    physical[:, food.LEFT.start:food.RIGHT.stop] = .8
    physical[:, world.ACTIVE] = 1
    receipt = physical.copy()
    receipt[:, world.REWARD_POS] = 1
    receipt[:, world.CONTACT_ID.start] = 1
    no_taste = receipt.copy()
    no_taste[:, world.REWARD_POS] = 0
    port_cut = NO_CUT.copy()
    port_cut[TASTE] = 0
    motor_cut = NO_CUT.copy()
    motor_cut[MOTOR] = 0
    nodes = np.ones(len(NO_CUT), dtype=bool)
    nodes[TASTE] = False

    def states(vary_receipt, cut):
        graph = Graph(seeds, learning=False, editing=False)
        graph.start(noise)
        graph.advance(physical)
        pulses = graph.advance(vary_receipt)
        right = graph.state[:, MOTOR.start + 1] > graph.state[:, MOTOR.start]
        np.testing.assert_array_equal(pulses, np.column_stack((~right, right)))
        graph.advance(physical, cut=cut)
        return graph.state[:, nodes]

    assert np.max(np.abs(states(receipt, NO_CUT) - states(no_taste, NO_CUT))) > .005
    np.testing.assert_array_equal(states(receipt, port_cut), states(no_taste, port_cut))
    assert np.max(np.abs(states(receipt, NO_CUT) - states(receipt, motor_cut))) > .005


def test_world_switch_is_hidden_and_receipt_is_only_learning_event():
    seeds = [5011, 5022, 5033]
    births_ = [9011, 9022]
    stable, first, _, stable_noise, _ = make_menu(seeds, "stable", trials=80)
    reverse, _, _, reverse_noise, _ = make_menu(seeds, "reverse", trials=80)
    np.testing.assert_array_equal(stable.chemicals, reverse.chemicals)
    np.testing.assert_array_equal(stable.body.distances, reverse.body.distances)
    np.testing.assert_array_equal(stable_noise, reverse_noise)
    for lane, change in enumerate(first):
        np.testing.assert_array_equal(stable.probability[lane, :change], reverse.probability[lane, :change])
        np.testing.assert_allclose(stable.probability[lane, change:], 1 - reverse.probability[lane, change:], atol=1e-7)

    options = dict(trials=80, probe_tail=9)
    recipe = Recipe(grow_every=6, retire_every=5)
    given, _ = run(births_, seeds, "reverse", "joint", recipe, **options)
    silent, _ = run(births_, seeds, "reverse", "no_receipt", recipe, **options)
    for lane, change in enumerate(given["change"]):
        assert np.array_equal(given["shown"][lane, :change], given["reward"][lane, :change])
    assert (given["edits"][:, :, 0] >= 0).any()
    assert (given["edits"][:, :, 1] >= 0).any()
    assert (given["writes"][:, :, 1] > 0).any()
    assert not given["shown"][:, -9:].any()
    assert not given["writes"][:, -9:].any()
    assert not silent["shown"].any()
    assert not silent["writes"].any()
    np.testing.assert_array_equal(silent["initial_active"], silent["final_active"])
    np.testing.assert_array_equal(silent["initial_weight"], silent["final_weight"])


def test_dense_credit_matches_finite_difference_with_carried_start():
    rng = np.random.default_rng(1247)
    public = np.zeros((5, 2, world.OBS_SIZE), dtype=np.float32)
    public[:, :, food.LEFT.start:food.RIGHT.stop] = rng.random((5, 2, 8))
    public[:4, :, world.ACTIVE] = 1
    public[4, 0, world.REWARD_POS] = 1
    public[4, 1, world.REWARD_NEG] = 1
    public[4, 0, world.CONTACT_ID.start] = 1
    public[4, 1, world.CONTACT_ID.start + 1] = 1
    initial = rng.uniform(-.4, .4, (2, len(NO_CUT))).astype(np.float32)
    noise = np.array([.15, -.1], dtype=np.float32)

    def play(indices=None, step=0.):
        graph = Graph([123, 321], editing=False)
        graph.state = initial.copy()
        if indices is not None:
            graph.weight[indices] += step
        graph.start(noise)
        for observed in public:
            graph.advance(observed)
        motor = np.stack([tick[2] for tick in graph.history[:4]])
        return graph, (motor[:, :, 1] - motor[:, :, 0]).mean(axis=0)

    graph, _ = play()
    slope = graph.gradient()

    def objective(mean):
        # Hold scalar credit and the initial state fixed when differentiating a request.
        return -.5 * graph.receipt * np.logaddexp(0, -2 * graph.side * mean)

    for indices in ((0, 9, 0), (0, 8, 5), (1, 9, 3), (1, 4, 12)):
        _, hi = play(indices, .001)
        _, lo = play(indices, -.001)
        numerical = ((objective(hi) - objective(lo)) / .002)[indices[0]]
        np.testing.assert_allclose(slope[indices], numerical, rtol=.004, atol=2e-5)
