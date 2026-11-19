import numpy as np

from experiments.pulse_game.sparse_location_roles import PROBE_TRIALS, fresh_menu, role_masks
from experiments.pulse_game.sparse_location_study import FIRST, PAIR_COUNT, SECOND, SIZE
from experiments.pulse_game.sparse_pairs_model import RECORD


def test_motif_requires_two_sensory_senders_and_a_motor_receiver():
    roles = [(0, 8, 0, 7), (0, 9, 6, 10), (1, 7, 0, 1), (1, 8, 9, 10)]
    active = np.zeros(len(roles), dtype=RECORD)
    for row, (lane, receiver, first, second) in enumerate(roles):
        site, = np.flatnonzero((FIRST == first) & (SECOND == second))
        active["key"][row] = (lane * SIZE + receiver) * PAIR_COUNT + site
    receiver, motor, sensory_motor = role_masks(active)
    np.testing.assert_array_equal(receiver, [8, 9, 7, 8])
    np.testing.assert_array_equal(motor, [True, True, False, True])
    np.testing.assert_array_equal(sensory_motor, [True, False, False, False])


def test_fresh_foods_share_the_physics_and_reverse_the_held_taste_rule():
    roots = [17, 23, 17, 23]
    stable, distinct = fresh_menu(roots, "stable")
    reverse, distinct_reverse = fresh_menu(roots, "reverse")
    assert distinct == distinct_reverse == 2 * PROBE_TRIALS * 2
    assert stable.body.trials == PROBE_TRIALS
    np.testing.assert_array_equal(stable.chemicals, reverse.chemicals)
    np.testing.assert_array_equal(stable.body.distances, reverse.body.distances)
    np.testing.assert_array_equal(stable.body.wind, reverse.body.wind)
    np.testing.assert_allclose(stable.probability + reverse.probability, 1, rtol=0, atol=np.finfo(np.float32).eps)
