"""Check native sensory isolation and the writer-state causal branch."""

import numpy as np

from experiments.pulse_game import self_write_look as look


def test_current_port_only_drives_the_three_inspection_nodes():
    case = look.encounter(455, 24)
    old = case.graph.clone()
    own = look.read_stimulus(case.graph, "own", case.donor)
    offset_only = look.read_stimulus(case.graph, "bias_only", case.donor)
    donor = look.read_stimulus(case.graph, "yoked", case.donor)
    t, s = look.PORT
    np.testing.assert_array_equal(own[:, 6], old.mask[:, t, s])
    np.testing.assert_array_equal(own[:, 7], old.w[:, t, s] * old.mask[:, t, s])
    np.testing.assert_array_equal(own[:, 8], old.b[:, t])
    np.testing.assert_array_equal(own[:, :6], 0)
    np.testing.assert_array_equal(own[:, 9:], 0)
    np.testing.assert_array_equal(offset_only[:, 6:8], 0)
    np.testing.assert_array_equal(offset_only[:, 8], own[:, 8])
    np.testing.assert_array_equal(donor, own[case.donor])
    assert np.all(case.donor != np.arange(len(own)))
    np.testing.assert_array_equal(look.read_stimulus(case.graph, "sham", case.donor), 0)
    np.testing.assert_array_equal(case.graph.w, old.w)
    np.testing.assert_array_equal(case.graph.b, old.b)
    np.testing.assert_array_equal(case.graph.mask, old.mask)


def test_writer_state_and_persistent_write_mediate_the_physical_choice():
    case = look.encounter(9217, 128)
    head = np.array([0, 1.0, 0.8, 0, -1.0, -0.8, 0, 0], dtype=np.float32)
    native = look.with_head(case.graph, head)
    own = look.inspect(native, look.read_stimulus(native, "own", case.donor))
    sham = look.inspect(native, look.read_stimulus(native, "sham", case.donor))
    transplanted = sham.copy()
    transplanted[:, look.SIGNAL_NODES] = own[:, look.SIGNAL_NODES]
    own_commit = look.execute(native, own, case.odor)
    sham_commit = look.execute(native, sham, case.odor)
    transplanted_commit = look.execute(native, transplanted, case.odor)
    own_blocked = look.execute(native, own, case.odor, allow=False)
    sham_blocked = look.execute(native, sham, case.odor, allow=False)
    assert np.any(own[:, look.SIGNAL_NODES] != sham[:, look.SIGNAL_NODES])
    assert np.any(own_commit["write"] != sham_commit["write"])
    assert np.any(own_commit["choices"] != own_blocked["choices"])
    np.testing.assert_array_equal(own_blocked["proposal"], own_commit["proposal"])
    np.testing.assert_array_equal(own_blocked["write"], 0)
    for field in ("proposal", "write", "margins", "choices", "rewards"):
        np.testing.assert_array_equal(own_commit[field], transplanted_commit[field])
    np.testing.assert_array_equal(own_blocked["choices"], sham_blocked["choices"])
