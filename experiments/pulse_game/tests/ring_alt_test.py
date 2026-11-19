"""Tests for taste eligibility, queue timing, and evaluation splits."""

from dataclasses import replace

import numpy as np

from experiments.pulse_game.ring_alt_policy import ForagingSpec, ReceptorForager
from experiments.pulse_game.ring_alt_run import contact_yoke, preference_code, world_panel
from experiments.pulse_game.ring_world import ACTIVE, FRUIT, NEGATIVE, OBS, TASTED, make_tape, rollout


def test_only_actual_wrong_fruit_cohort_changes_with_negative_feedback():
    tape = make_tape(range(201, 225), 1)
    spec = ForagingSpec(outcomes="negative", scoring="linear")
    actor = ReceptorForager(spec, birth_seed=12)
    result = rollout(actor, tape)
    assert (result.bites == 1).all()
    wrong = ~result.first_correct[:, 0]
    assert wrong.any() and not wrong.all()
    weighted = (actor.weights != 0).any(axis=(2, 3))
    wanted = np.zeros_like(weighted)
    wanted[wrong, result.first[wrong, 0]] = True
    np.testing.assert_array_equal(weighted, wanted)
    np.testing.assert_array_equal(actor.writes[:, 1], wrong)
    assert not actor.writes[:, 0].any()


def test_delayed_receipts_for_distinct_contacts_do_not_overwrite():
    spec = ForagingSpec(outcomes="negative", feedback_delay=4)
    actor = ReceptorForager(spec, birth_seed=12)
    actor.reset(1)
    actor.eligibility[:] = .5
    for tick in range(7):
        inp = np.zeros((1, OBS), dtype=np.float32)
        inp[0, ACTIVE] = 1
        if tick < 2:
            inp[0, TASTED.start + tick] = 1
            inp[0, FRUIT.start + tick] = 1
            inp[0, NEGATIVE] = 1
            actor.last_motor_receptor.fill(0)
            actor.last_motor_receptor[0, tick] = 1
        actor.act(inp)
        assert actor.writes[0, 1] == max(0, min(2, tick - 3))
    assert actor.weights[0, :2].any(axis=(1, 2)).all()
    assert not actor.weights[0, 2:].any()


def test_evaluation_maps_are_disjoint_and_yoke_preserves_one_sign_per_bite():
    train = world_panel(201001, 12, 3, panel="development")
    hold = world_panel(202001, 12, 3, panel="evaluation")
    assert not set(preference_code(train.preferences)) & set(preference_code(hold.preferences))
    spec = replace(ForagingSpec(), scoring="blind")
    result = rollout(ReceptorForager(spec), hold, retry=True)
    signs = contact_yoke(result)
    swapped = np.arange(hold.n) ^ 1
    np.testing.assert_array_equal((signs != 0).sum(axis=2), result.bites[swapped])
    np.testing.assert_array_equal((signs > 0).sum(axis=2), result.positive[swapped])
    np.testing.assert_array_equal((signs < 0).sum(axis=2), result.negative[swapped])
