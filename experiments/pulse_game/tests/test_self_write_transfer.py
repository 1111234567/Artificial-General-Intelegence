import numpy as np

from experiments.pulse_game import self_write_core as core
from experiments.pulse_game import self_write_transfer as transfer


def test_matched_contacts_preserve_private_complements_and_deliver_real_reversal():
    tape = transfer.build_tape(81, 8, 6, 4, 2, 0.35, complementary=True)
    np.testing.assert_array_equal(tape.odors[::2], tape.odors[1::2])
    np.testing.assert_array_equal(tape.targets[::2], -tape.targets[1::2])
    np.testing.assert_array_equal(tape.scripted_arms[::2], -tape.scripted_arms[1::2])
    np.testing.assert_array_equal(np.sort(tape.cues.reshape(8, 4, 6), axis=2), np.tile(np.arange(6), (8, 4, 1)))
    np.testing.assert_array_equal(tape.cues[:, 11], tape.cues[:, 12])
    np.testing.assert_array_equal(tape.scripted_arms[:, 11], tape.scripted_arms[:, 12])
    np.testing.assert_array_equal(tape.taste(11, tape.scripted_arms[:, 11]), 1.0)
    np.testing.assert_array_equal(tape.taste(12, tape.scripted_arms[:, 12]), -1.0)


def test_transplant_and_scramble_only_edit_writer_on_matched_weight_scale():
    base = core.spawn_graph(14, 4)
    donor = base.clone()
    donor.b[:, 0] += 0.1
    donor.b[:, 12:16] += np.array([0.15, -0.10, 0.05, -0.01], dtype=np.float32)
    donor.w[:, 12, :6] += np.linspace(-0.2, 0.2, 6, dtype=np.float32)
    active = core.transplant_writer(base, donor, base)
    transfer.validate_transplant(base, active, donor)
    scrambled = transfer.scramble_writer(base, active, 14)
    assert not np.array_equal(scrambled.w, active.w)
    np.testing.assert_array_equal(scrambled.w[:, :12], base.w[:, :12])
    np.testing.assert_array_equal(scrambled.b[:, :12], base.b[:, :12])
    np.testing.assert_array_equal(scrambled.mask, active.mask)
    np.testing.assert_allclose(transfer.writer_delta(base, active)[0], transfer.writer_delta(base, scrambled)[0])
