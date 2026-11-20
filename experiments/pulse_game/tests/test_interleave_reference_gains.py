from dataclasses import replace

import numpy as np
import torch

from experiments.pulse_game import interleave_adaptive_reference as adaptive
from experiments.pulse_game import interleave_compact_reference as compact
from experiments.pulse_game import interleave_slow_reference as slow
from experiments.pulse_game import local_trace_graph as graph


def test_compact_templates_and_native_outer_range():
    body = graph.spawn(731)
    body = replace(body, weights=body.weights * 7)
    signed, size = compact.reference(body, "signed_rows")
    assert size == 185
    assert torch.equal(signed[..., :-1].sign(), (body.weights * body.present).sign())
    assert torch.equal(signed[..., -1].sign(), body.bias.sign())
    for mode, nbytes in (("global_radius", 4), ("row_radius", 80), ("indexed_exact", 1680)):
        target, size = compact.reference(body, mode)
        assert size == nbytes
        after = compact.apply_numeric(body, target, torch.zeros(1, 1, 1), mode)
        assert torch.equal(after.weights[body.present], body.weights[body.present])
        assert torch.equal(after.bias, body.bias)
        assert torch.equal(after.present, body.present)


def test_fixed_adaptive_gain_preserves_the_reference_and_one_action_payout():
    body, rule = graph.spawn(731), graph.Rule([731])
    tapes = slow.public_tapes(3, 44831, 44837)
    theta = np.array([[adaptive.GAIN0, 0., 0., 0.]])
    result = adaptive.life(body, rule, tapes, theta, [4], "absent_a", 1)[0, 0]
    ordinary = slow.live(body, rule, tapes, "absent_a", 1, .12, .02)
    np.testing.assert_array_equal(result, ordinary["correct"][slow.AGES.index(4), ..., :6])
    q = np.arange(3)
    earned = adaptive.life(body, rule, tapes, theta, [4], "absent_a", 1, query=q)[0]
    expected = np.stack([result[i, :, :, question+1].ravel() for i, question in enumerate(q)])
    np.testing.assert_array_equal(earned > 0, expected)


def test_two_realizations_of_zero_gain_match_in_one_long_life():
    body, rule = graph.spawn(811), graph.Rule([811])
    tapes = slow.public_tapes(2, 9871, 9877)
    baseline, _, _ = compact.evaluate(body, rule, tapes, "recurring", 4, "native", [0.])
    radii, _, _ = compact.evaluate(body, rule, tapes, "recurring", 4, "row_radius", [0., .12])
    np.testing.assert_array_equal(radii[0], baseline[0])
