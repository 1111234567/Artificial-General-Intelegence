"""Check that the repeated rollout keeps one actual trajectory and the full taste ledger."""

import torch

from experiments.pulse_game import local_trace_graph as native
from experiments.pulse_game import noon_readback_fit as models
from experiments.pulse_game import noon_review_repeated as life
from experiments.pulse_game import noon_review_repeated_fit as fitting


def test_fair_shared_readout_and_random_have_identical_complete_physical_lives():
    tape = life.make_tape(1, 3)
    graph, rule = native.spawn(101), native.Rule(101)
    readout = models.Readout("logged").requires_grad_(False)
    with torch.no_grad():
        fair = life.run_life(graph, rule, tape, "learned", readout, audit_graph=True)
        random = life.run_life(graph, rule, tape, "random", audit_graph=True)
    assert all(torch.equal(fair[key], expected) for key, expected in random.items())
    assert (tape.candidates.max(2).values < life.WARM + torch.arange(life.ROUNDS)).all()
    assert fair["earned"].shape == (6, 6, 3)
    assert (fair["final_graph_change"][:, 0] > 0).all()
    totals = life.resources(fair)
    assert torch.equal(totals["total"], fair["warm"].sum(1) + fair["earned"].sum((1, 2)) + fair["final"])
    logged = {arm: {name: value.numpy() for name, value in row.items()} for arm, row in
              (("learned", fair), ("random", random))}
    report = life.nearby_effects(logged, "learned", "random")
    assert not any(report["choice_disagreement_by_round"])


def test_credit_difference_is_only_current_and_later_earned_review_taste():
    actual = {"earned": torch.ones(2, life.ROUNDS, 3), "final": torch.tensor([1., -1.])}
    actual["earned"][1, ::2, 0] = -1
    scale = 3 * torch.arange(life.ROUNDS, 0, -1) + 1
    review_return = actual["earned"][:, :, 0].flip(1).cumsum(1).flip(1).double() / scale
    torch.testing.assert_close(fitting.credit(actual, "total") - fitting.credit(actual, "later"), review_return)
