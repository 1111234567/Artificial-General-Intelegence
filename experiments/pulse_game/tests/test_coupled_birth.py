"""Mechanism and invariants for a receipt-trained newborn connection."""

import numpy as np

from experiments.pulse_game.coupled_birth import Population, WIDTH
from experiments.pulse_game.live_module_edits import Schedule


def make_population(*, independent_sign=False):
    rng = np.random.default_rng(30)
    mask = np.zeros((1, WIDTH, WIDTH), dtype=bool)
    mask[0, np.arange(WIDTH), (np.arange(WIDTH) - 1) % WIDTH] = True
    mask[0, 0, 1] = True
    parameters = {key: (rng.standard_normal(shape) * .4).astype(np.float32) for key, shape in (
        ("gain", (1, WIDTH)), ("drive", (1, 4, WIDTH * 2)), ("bias", (1, WIDTH * 2)),
        ("readout", (1, WIDTH)), ("offset", (1,)))}
    return Population(parameters, mask, Schedule(1, 999, .5), [13], learning_rate=.03,
                      birth_scale=3., independent_sign=independent_sign, audit=True)


def request(brain, seed, contacted=1, taste=-1):
    rng = np.random.default_rng(seed)
    brain.begin_request(learn_edges=True)
    for i in range(9):
        obs = rng.standard_normal((1, 4)).astype(np.float32)
        brain.act(obs, np.ones(1) if i >= 4 else None, record=True)
    return brain.finish_request(np.array([contacted]), np.array([taste]))


def test_signed_birth_is_the_current_receipt_gradient_and_other_steps_match():
    aligned = make_population()
    control = make_population(independent_sign=True)
    initial = aligned.matrix.copy()
    a_edit, a_records = request(aligned, 56)
    b_edit, _ = request(control, 56)
    np.testing.assert_array_equal(a_edit, b_edit)
    t, s = divmod(int(a_edit[0, 0]), WIDTH)
    assert initial[0, t, s] == 0 and not aligned.initial[0, t, s]
    a, b = (brain.birth_checks[0][0] for brain in (aligned, control))
    assert a[2] > 0 and a[3] > 0
    np.testing.assert_allclose(a[2], a[3], rtol=.08)
    np.testing.assert_allclose(abs(a[1]), abs(b[1]), rtol=0., atol=0.)
    active = aligned.initial
    np.testing.assert_array_equal(aligned.matrix[active], control.matrix[active])
    np.testing.assert_array_equal(aligned.state, control.state)
    assert a_records[0, 0, 2] == aligned.matrix[0, t, s] == a[1]
    assert aligned.newborn_changes[0, t, s] == 0

    request(aligned, 57)
    request(aligned, 58)
    assert aligned.newborn_changes[0, t, s] == 2
    assert aligned.newborn_first_change[0, t, s] == 1
    assert aligned.raw[0, t, s] != a[1]


def test_retirement_allows_two_adjustments_and_leaves_all_other_effective_values_intact():
    brain = make_population()
    events, records = request(brain, 56)
    t, s = divmod(int(events[0, 0]), WIDTH)
    brain.retire_scores[0] = 0
    brain.retire_scores[0, t, s] = 10
    current = brain.matrix.copy()

    brain.receive_count[0] += 1
    brain._retire([True], events, records)
    np.testing.assert_array_equal(brain.matrix, current)
    brain.receive_count[0] += 1
    brain._retire([True], events, records)
    assert events[0, 1] == t * WIDTH + s
    current[0, t, s] = 0
    np.testing.assert_array_equal(brain.matrix, current)


def test_no_public_taste_means_no_weight_or_support_update():
    brain = make_population()
    current = brain.matrix.copy()
    events, records = request(brain, 56, contacted=-1, taste=0)
    assert (events == -1).all() and np.isnan(records).all()
    assert brain.receive_count[0] == 0 and (brain.weight_step == 0).all()
    np.testing.assert_array_equal(brain.matrix, current)
    np.testing.assert_array_equal(brain.mask, brain.initial)
