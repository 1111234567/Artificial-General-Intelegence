"""Check that retention scoring is paired, receipt-delayed and separate from weight SGD."""

import numpy as np

from experiments.pulse_game.coupled_birth import Population as Original
from experiments.pulse_game.live_module_edits import Schedule
from experiments.pulse_game.prospective_edges import Population, WIDTH


def make(arm, growth=9):
    rng = np.random.default_rng(43)
    mask = np.zeros((1, WIDTH, WIDTH), dtype=bool)
    mask[0, np.arange(WIDTH), (np.arange(WIDTH) - 1) % WIDTH] = True
    mask[0, 0, 1] = True
    params = {key: (rng.standard_normal(shape) * .4).astype(np.float32) for key, shape in (
        ("gain", (1, WIDTH)), ("drive", (1, 4, WIDTH * 2)), ("bias", (1, WIDTH * 2)),
        ("readout", (1, WIDTH)), ("offset", (1,)),
    )}
    args = params, mask, Schedule(growth, 6, .5), [30]
    options = dict(learning_rate=.015, birth_scale=4., audit=True)
    return Original(*args, **options) if arm is None else Population(*args, arm=arm, **options)


def move(life, seed):
    life.begin_request(learn_edges=True)
    rng = np.random.default_rng(seed)
    for tick in range(11):
        life.act(rng.standard_normal((1, 4), dtype=np.float32), np.ones(1, dtype=np.float32) if tick >= 5 else None,
                 record=True)


def request(life, seed, *, valid=True):
    move(life, seed)
    return life.finish_request(np.array([1 if valid else -1]), np.array([(-1) ** seed if valid else 0]))


def test_probe_matches_direct_paired_pre_update_replay():
    life = make("later")
    move(life, 71)
    life.receipt_direction = np.array([1])
    lanes = np.array([0])
    lane, target, source, effects = life.marginal_removal(lanes)
    intact = np.logaddexp(0., -life._mean_logit(lanes, life.matrix, life.parameters) / life.trace_ticks)
    assert len(effects) == life.mask.sum()
    direct = []
    for l, t, s in zip(lane, target, source, strict=True):
        silenced = life.matrix.copy()
        silenced[l, t, s] = 0.
        loss = np.logaddexp(0., -life._mean_logit(lanes, silenced, life.parameters) / life.trace_ticks)
        direct.append((intact - loss)[0])
    np.testing.assert_allclose(effects, direct, atol=1e-6)


def test_instrumented_baseline_preserves_original_actions_and_writes():
    instrumented, original = (make(arm) for arm in ("instant", None))
    for seed in range(33):
        measured = request(instrumented, seed, valid=seed != 2)
        expected = request(original, seed, valid=seed != 2)
        for a, b in zip(measured, expected, strict=True):
            np.testing.assert_array_equal(a, b)
        for key in ("raw", "mask", "state", "weight_step", "receive_count"):
            np.testing.assert_array_equal(getattr(instrumented, key), getattr(original, key))


def test_newborn_collects_only_later_receipts_and_no_contact_is_ignored():
    life = make("later", growth=4)
    for seed in range(4):
        growth, _ = request(life, seed)
    assert growth[0, 0] >= 0
    target, source = divmod(int(growth[0, 0]), WIDTH)
    arc = 0, target, source
    assert life.future_observations[arc] == 0
    assert life.future_score[arc] == 0
    assert life.receipt_writes[0, 2, 1] == 1
    request(life, 5, valid=False)
    assert life.future_observations[arc] == 0
    assert life.receive_count[0] == 4
    assert (life.receipt_writes == 0).all()
    request(life, 6)
    assert life.receive_count[0] == 5
    assert life.future_observations[arc] == 1
    growth, _ = request(life, 7)
    assert growth[0, 1] != target * WIDTH + source


def test_holdout_probe_cannot_change_retirement(monkeypatch):
    a, b = (make("later") for _ in range(2))
    for life, heldout in ((a, 1000.), (b, -1000.)):
        def controlled(lanes, life=life, heldout=heldout):
            lane, target, source = np.nonzero(life.mask)
            is_extra = (target == 0) & (source == 1)
            is_sixth = life.receive_count[0] == 5
            effect = np.where(is_extra, heldout if is_sixth else 1., 0.)
            return lane, target, source, effect
        monkeypatch.setattr(life, "marginal_removal", controlled)
        for seed in range(6):
            edits, _ = request(life, seed)
        assert edits[0, 1] == 1
        assert life.future_observations[0, 0, 1] == 0
        assert life.future_score[0, 0, 1] == 0
    assert not np.array_equal(a.evidence_values[0], b.evidence_values[0])
    np.testing.assert_array_equal(a.matrix, b.matrix)
