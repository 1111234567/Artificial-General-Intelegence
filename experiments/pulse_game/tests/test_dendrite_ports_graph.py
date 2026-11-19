"""Validate the additional receiver-local product gradient and structural support."""

import numpy as np
import pytest

from experiments.pulse_game import mixture_world as food, world
from experiments.pulse_game.dendrite_ports_graph import DendriteGraph, FIRST, SECOND, widen_birth
from experiments.pulse_game.dendrite_ports_study import SETUPS, run
from experiments.pulse_game.unified_live_graph import Graph, SIZE, Recipe


@pytest.mark.parametrize("product", [True, False])
def test_pair_credit_matches_finite_difference(product):
    rng = np.random.default_rng(9917)
    roots = [317, 319]
    state = rng.uniform(-.5, .5, (len(roots), SIZE)).astype(np.float32)
    physical = np.zeros((6, len(roots), world.OBS_SIZE), dtype=np.float32)
    physical[:, :, food.LEFT.start:food.RIGHT.stop] = rng.random((6, len(roots), 8))
    physical[:5, :, world.ACTIVE] = 1
    physical[5, 0, world.REWARD_POS] = 1
    physical[5, 1, world.REWARD_NEG] = 1
    physical[5, 0, world.CONTACT_ID.start] = 1
    physical[5, 1, world.CONTACT_ID.start + 1] = 1
    initial = DendriteGraph(roots, Recipe(), product=product, pair_rate=1., pair_decay=.99)
    pair = (initial.valid_pairs * rng.normal(0, .03, initial.pair.shape)).astype(np.float32)

    def play(q):
        graph = DendriteGraph(roots, Recipe(), product=product, pair_rate=1., pair_decay=.99)
        graph.state = state.copy()
        graph.pair[:] = q
        graph.refresh_pair_support()
        graph.start(np.array([.17, -.02], dtype=np.float32))
        for obs in physical:
            graph.advance(obs)
        motors = np.stack([tick[3] for tick in graph.history[:5]])
        return graph, (motors[:, :, 1] - motors[:, :, 0]).mean(axis=0)

    graph, _ = play(pair)
    gradient = graph.gradients()[1]

    def objective(mean):
        return -.5 * graph.receipt * np.logaddexp(0, -2 * graph.side * mean)

    index = np.unravel_index(np.argsort((np.abs(gradient) * graph.valid_pairs).ravel())[-8:], pair.shape)
    for address in zip(*index, strict=True):
        change = np.zeros_like(pair)
        change[address] = .001
        _, high = play(pair + change)
        _, low = play(pair - change)
        finite = ((objective(high) - objective(low)) / .002)[address[0]]
        np.testing.assert_allclose(gradient[address], finite, atol=3e-5, rtol=.006)


def test_matched_birth_and_actual_pair_writes():
    seeds = [318, 321]
    recipe = Recipe(**SETUPS[0]["recipe"])
    paired = DendriteGraph(seeds, recipe, product=True, pair_rate=1., pair_decay=.99)
    baseline = Graph(seeds, recipe)
    widen_birth(baseline, seeds)
    np.testing.assert_array_equal(paired.mask, baseline.mask)
    np.testing.assert_array_equal(paired.weight, baseline.weight)
    for graph in (paired, baseline):
        graph.start(np.array([.25, -.5], dtype=np.float32))
    for _ in range(world.TICKS):
        obs = np.full((len(seeds), world.OBS_SIZE), .4, dtype=np.float32)
        np.testing.assert_array_equal(paired.advance(obs), baseline.advance(obs))
        np.testing.assert_array_equal(paired.state, baseline.state)
    raw = run(seeds, [533, 544, 555], "reverse", "product", SETUPS[0], trials=128, tail=17)
    assert raw["writes"][:, :, 3].sum() > 0
    assert raw["writes"][:, :, 4].sum() > 0
    assert not raw["shown"][:, -17:].any()
    assert not raw["writes"][:, -17:].any()
    live = raw["final_mask"]
    assert not raw["final_pair"][~(live[:, :, FIRST] & live[:, :, SECOND])].any()
