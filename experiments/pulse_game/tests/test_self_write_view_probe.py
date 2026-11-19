"""Check that a transient probe and a committed write are distinct transactions."""

import numpy as np

from experiments.pulse_game import self_write_view as graph
from experiments.pulse_game import self_write_view_probe as probe


def test_matched_probe_keeps_ordinary_stream_fixes_ports_and_does_not_write():
    case = probe.birth(752, 108)
    before = case.body.values.copy()
    support = case.body.exists.copy()
    features, baseline, response = probe.observations(case)
    public = graph.SLOTS + graph.WIDTH * (1 + probe.TICKS)
    pulse = slice(public, public + probe.TICKS * graph.WIDTH)
    structure = slice(pulse.stop, None)
    for observation in features.values():
        assert observation.shape == (108, probe.FEATURES)
        np.testing.assert_array_equal(observation[:, :public], features["sham"][:, :public])
    np.testing.assert_array_equal(features["yoked_pulse"][:, pulse], response[case.donor])
    np.testing.assert_array_equal(features["sham"][:, pulse], 0)
    np.testing.assert_array_equal(features["current_structure"][:, pulse], 0)
    for mode in ("true_pulse", "sham", "yoked_pulse"):
        np.testing.assert_array_equal(features[mode][:, structure], 0)
    assert (case.donor != np.arange(len(case.donor))).all()
    np.testing.assert_array_equal(case.kind[case.donor], case.kind)
    np.testing.assert_array_equal(case.port[case.donor], case.port)
    np.testing.assert_array_equal(case.body.values, before)
    np.testing.assert_array_equal(case.body.exists, support)
    np.testing.assert_array_equal(baseline, probe.advance(case.body, case.start, np.zeros_like(case.start)))


def test_probe_matches_only_first_transition_of_small_positive_write():
    case = probe.birth(29, 108)
    _, base, response = probe.observations(case)
    support = case.body.exists.copy()
    values = case.body.values.copy()
    rows = np.arange(len(case.port))
    support[rows, case.port] = True
    values[rows, case.port] += probe.PROBE
    sustained = probe.advance(graph.Body(support, values), case.start, np.zeros_like(case.start))
    pulsed = base + probe.PROBE * response.reshape(base.shape)
    np.testing.assert_allclose(pulsed[:, 0], sustained[:, 0], atol=2e-7, rtol=0)
    assert np.max(np.abs(pulsed[:, 1:] - sustained[:, 1:])) > 0.02
    plus = probe.motor(probe.actual_edit(case, np.ones(len(rows), dtype=bool))) - probe.motor(base)
    assert (plus > 0.01).any() and (plus < -0.01).any()
