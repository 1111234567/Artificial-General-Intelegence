import numpy as np

from experiments.pulse_game import active_symmetry_world as game, world
from experiments.pulse_game.context_local_active import frozen_brain, lifetime, randomized_nonfinal, taste_pack


def test_packed_tasting_has_only_the_actual_selected_menu_rewards():
    journey = game.build(range(171, 183))
    where = journey.arena[:, 0]
    bought = np.arange(len(where)) % 2 == 0
    found = np.arange(len(where)) % 5 != 0
    packed, available = taste_pack(journey, where, 0, bought, found)
    for seat in range(len(where)):
        if not found[seat]:
            assert not available[seat].any()
            np.testing.assert_array_equal(packed.realized[seat], 0)
            continue
        expected = game.assembly(journey, where[seat], 0, [seat], bool(bought[seat]))
        count = expected.body.trials
        assert available[seat].sum() == count
        np.testing.assert_array_equal(packed.chemicals[seat, :count], expected.chemicals[0])
        np.testing.assert_array_equal(packed.realized[seat, :count], expected.realized[0])
        np.testing.assert_array_equal(packed.realized[seat, count:], 0)


def test_nonfinal_random_matches_counts_in_each_physical_place():
    journey = game.build(range(293, 317))
    t = np.arange(journey.ticks)[None, :]
    places = journey.arena[..., None] == np.arange(2)
    counts = (places & (t < journey.horizon[:, None] - 1)[..., None]).sum(1).clip(0, 2)
    selected = randomized_nonfinal(journey, counts, 1)
    np.testing.assert_array_equal((places & selected[..., None]).sum(1), counts)
    assert not (selected & (t >= journey.horizon[:, None] - 1)).any()


def test_same_brain_advances_through_plates_and_tastes_without_clock_reset():
    frozen, _ = frozen_brain()
    journey = game.build([3101, 3102, 3103, 3104], horizons=(4,))
    record = lifetime(frozen, journey, method="once")
    assert record["continuous_neural_ticks"] == journey.ticks * (1 + (1 + game.BLOCK) * world.TICKS)
    assert record["landed"][record["active"]].all()
    np.testing.assert_array_equal(record["intentions"], record["probe"])
    np.testing.assert_allclose(record["net"], (record["rewards"] - record["paid"]).sum(1))
