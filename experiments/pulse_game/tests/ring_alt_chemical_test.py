"""Check receptor recruitment, inactivity eviction, and real taste bonding."""

from dataclasses import replace

import numpy as np

from experiments.pulse_game.ring_alt_chemical import ChemSpec, ChemicalForager, ReceptorPopulation
from experiments.pulse_game.ring_alt_chemical_forward import AgeCompetingReceptors
from experiments.pulse_game.ring_alt_chemical_shift import PaletteShift, audit_tape
from experiments.pulse_game.ring_alt_chemical_run import actor
from experiments.pulse_game.ring_sensors import ChemicalInterface
from experiments.pulse_game.ring_world import make_tape, rollout


def test_capacity_probes_share_the_same_cue_and_lif_birth():
    lower = ChemSpec(cue_bank="growing", capacity=2)
    a = ChemicalForager(lower, birth_seed=42)
    b = ChemicalForager(replace(lower, capacity=12), birth_seed=42)
    np.testing.assert_array_equal(a.recurrent, b.recurrent)
    np.testing.assert_array_equal(a.aff, b.aff)
    np.testing.assert_array_equal(a.threshold, b.threshold)
    np.testing.assert_array_equal(a.cue.seed_weights, b.cue.seed_weights)
    np.testing.assert_array_equal(a.fruit.seed_weights, b.fruit.seed_weights[:2])
    np.testing.assert_array_equal(a.links, b.links[:2])


def test_unlabeled_branches_grow_only_on_novel_contacts_and_retire_when_stale():
    spec = ChemSpec(capacity=2, novelty=.96, retire_age=3)
    bank = ReceptorPopulation("growing", 2, spec, np.random.default_rng(91))
    bank.reset(1)
    one = np.eye(4, dtype=np.float32)
    seen = np.array([True])
    np.testing.assert_array_equal(bank.match(one[0:1]), [[0, 0]])
    for idx, trial in ((0, 0), (0, 0), (1, 1), (2, 2)):
        bank.observe(one[idx:idx + 1], seen, trial)
    assert bank.births[0] == 2
    assert bank.retirements[0] == 0
    assert bank.centroid_writes[0] == 3
    owners, branches = bank.observe(one[2:3], seen, 5)
    np.testing.assert_array_equal(owners, [0])
    np.testing.assert_array_equal(branches, [0])
    assert bank.retirements[0] == 1
    np.testing.assert_array_equal(bank.prototype[0, 0], one[2])


def test_forward_age_decreases_cross_cohort_responses_without_changing_templates():
    rng = np.random.default_rng(62)
    spec = ChemSpec(capacity=2, width=.35, novelty=.97)
    bank = AgeCompetingReceptors("growing", 2, spec, rng, "ordinary", 32)
    bank.reset(1)
    older = np.array([[1., 0., 0., 0.]], dtype=np.float32)
    newer = np.array([[.8, .6, 0., 0.]], dtype=np.float32)
    seen = np.array([True])
    bank.observe(older, seen, 0)
    bank.observe(newer, seen, 128)
    before = bank.match(older)
    bank.credit = "age"
    after = bank.match(older)
    assert before[0, 1] > .2 and after[0, 1] < .02
    np.testing.assert_allclose(after.sum(), 1., atol=1e-7)
    np.testing.assert_array_equal(bank.prototype[0], np.concatenate((older, newer), axis=0))


def test_real_novel_taste_is_credited_only_after_actual_bite():
    tape = make_tape(range(1200, 1216), 12)
    spec = ChemSpec(cue_bank="growing")
    learner = ChemicalForager(spec, birth_seed=29)
    result = rollout(ChemicalInterface(learner, tape.seeds, "novel"), tape)
    assert (result.bites == 1).all()
    assert not learner.misaligned_receipts.any()
    np.testing.assert_array_equal(learner.weight_updates[:, 1], result.negative.sum(axis=1))
    assert not learner.weight_updates[:, 0].any()
    np.testing.assert_array_equal(learner.fruit.alive.sum(axis=1), np.full(tape.n, 4))
    np.testing.assert_array_equal(learner.cue.alive.sum(axis=1), np.full(tape.n, 4))


def test_quiet_palette_probes_do_not_change_training_state():
    tape = make_tape([521, 522], 9)
    measure = audit_tape(tape, 252001, tape.n, 4, "development")
    spec = ChemSpec(cue_bank="growing")
    ref = actor(tape, spec, 17)
    forked = actor(tape, spec, 17)
    a = rollout(PaletteShift(ref, tape.seeds, "novel", cut=4), tape)
    b = rollout(PaletteShift(forked, tape.seeds, "novel", cut=4, assay=measure), tape)
    np.testing.assert_array_equal(a.first, b.first)
    np.testing.assert_array_equal(a.first_tick, b.first_tick)
    np.testing.assert_array_equal(ref.weights, forked.weights)
    np.testing.assert_array_equal(ref.fruit.prototype, forked.fruit.prototype)
