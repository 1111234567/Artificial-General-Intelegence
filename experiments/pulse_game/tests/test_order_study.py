"""Regression checks for the symmetric order task and its physical controls."""

import numpy as np
import torch

from experiments.pulse_game import order_model as model, order_study as study
from experiments.pulse_game import order_world as env, world


def test_order_swap_changes_only_order_and_twin_controls_change_no_sensory_ticks():
    tape = env.make(range(35), 4, "long")
    swapped = env.fork(tape, "swap")
    np.testing.assert_array_equal(tape.observations[:-1], swapped.observations[:-1])
    np.testing.assert_array_equal(tape.observations[..., env.MARKER:],
                                  swapped.observations[..., env.MARKER:])
    np.testing.assert_array_equal(tape.target_vectors[:, -1, ::-1], swapped.target_vectors[:, -1])
    np.testing.assert_array_equal(tape.target_ticks, swapped.target_ticks)
    np.testing.assert_array_equal(env._sides(tape.target_symbols)[:, -1],
                                  1 - env._sides(swapped.target_symbols)[:, -1])
    for key in ("first_only", "second_only", "timing_only"):
        base, twin = (env.fork(tape, change) for change in (key, key + "_flip"))
        np.testing.assert_array_equal(base.observations, twin.observations)
        np.testing.assert_array_equal(tape.body.good_id[:, -1], base.body.good_id[:, -1])
        np.testing.assert_array_equal(base.body.good_id[:, -1], 1 - twin.body.good_id[:, -1])


def test_only_physical_contacts_produce_exploration_targets():
    tape = env.make(range(16), 3, "standard")
    result = env.explore(tape)
    side, taste = env.contact_receipts(result)
    np.testing.assert_array_equal((taste == 1), (side.T == env._sides(tape.target_symbols)).T)
    assert env.chronological_oracle(tape).accuracy == 1.0
    alternating = np.broadcast_to((np.arange(world.TICKS)[None, :, None] % 2 == 0),
                                  (tape.body.trials, world.TICKS, tape.body.n))
    timeout = env.physics(tape, alternating)
    assert not timeout.correct.any()
    assert timeout.contact_rate == 0
    assert not env.contact_receipts(timeout)[1].any()


def test_parameters_are_identical_at_birth_and_only_placement_of_candidate_edges_differs():
    graph, single = (model.Circuit(33, name) for name in study.TOPOLOGIES)
    assert sum(study.budgets()["self"].values()) == sum(study.budgets()["graph"].values()) == 449
    for graph_parameter, self_parameter in zip(graph.parameters(), single.parameters(), strict=True):
        torch.testing.assert_close(graph_parameter, self_parameter)
    assert not torch.any(graph.edges == single.edges)
    observation = torch.from_numpy(env.make(range(5), 2, "standard").observations)
    a, end = graph(observation[0], graph.start(5), intervention=1)
    b, _ = single(observation[0], single.start(5), intervention=1)
    torch.testing.assert_close(a, b)
    continuous, _ = graph(observation[1], end, intervention=1)
    joined, _ = graph(observation.flatten(0, 1), graph.start(5), intervention=1)
    torch.testing.assert_close(continuous, joined[len(a):])
    cold, _ = graph(observation[1], graph.start(5), intervention=1)
    assert not torch.allclose(continuous, cold)
