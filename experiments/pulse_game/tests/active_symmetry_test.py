"""Regression checks for paid actions, real receipts and matched budgets."""

import numpy as np

from experiments.pulse_game import active_symmetry_agent as p, active_symmetry_world as w
from experiments.pulse_game.active_symmetry_train import matched_schedule


def test_gate_routes_actual_contact_and_charges_same_arena_budget():
    menu = w.build(range(1331, 1341))
    chosen = np.arange(len(menu.seeds)) % 2 == 0
    _, hit, physical = w.reach_gate(menu, np.arange(len(menu.seeds)), 0, chosen)
    assert physical.all()
    np.testing.assert_array_equal(hit, chosen)
    a = p.lifetime(menu, p.Gate("once"), 23)
    shuffled = matched_schedule(menu, a["counts"], 878)
    b = p.lifetime(menu, p.Gate("matched", quantum=shuffled), 23)
    np.testing.assert_array_equal(a["counts"], b["counts"])
    np.testing.assert_allclose(a["paid"].sum(1), b["paid"].sum(1), atol=1e-12, rtol=0)
    np.testing.assert_array_equal(a["net"], (a["rewards"] - a["paid"]).sum(1))


def test_harvest_pays_a_real_result_without_giving_a_query_receipt():
    menu = w.build([3441, 3442, 3443, 3444])
    ids = np.arange(4)
    controller = p.Cortex(menu.seeds, 13)
    observation = w.assembly(menu, 0, 0, ids, True)
    explored = controller.session(ids, 0, 0, observation, True)
    assert (explored.observed_reward.sum(2) != 0).all()
    assert np.count_nonzero(controller.gain[:, 1]) == 0
    harvest = w.assembly(menu, 0, 1, ids, False)
    out = controller.session(ids, 0, 1, harvest, False)
    assert (out.contact_side[:, 1] >= 0).all()
    assert np.all(out.observed_reward[:, 1] == 0)
    assert np.all(np.abs(out.reward[:, 1]) == 1.)


def test_public_backward_plan_never_buys_a_probe_on_the_last_ticket():
    sequence, center, expected = p.ideal(.1, .72, 14)
    assert np.all(sequence[1] < 0)
    assert expected[1] == 0 and expected[-1] > 0
    assert sequence[12][0, center, center] > 0
