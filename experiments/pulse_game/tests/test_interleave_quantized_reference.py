import numpy as np
import torch

from experiments.pulse_game import interleave_compact_reference as compact
from experiments.pulse_game import interleave_quantized_reference as ref


def example():
    values = np.arange(1, ref.COUNT + 1, dtype=np.float32).reshape(ref.SHAPE)
    values = (values - 17) / 88
    eligible = np.ones((ref.WIDTH, ref.WIDTH), dtype=bool)
    eligible[0, 0] = False
    values[0, 0] = 0.
    values[0, 1] = -1e-10
    return values, eligible


def test_actual_payload_sizes_and_absence_versus_numeric_zero():
    values, eligible = example()
    for mode in ref.FORMATS:
        payload = ref.encode(values, eligible, mode)
        assert len(payload) == ref.STORAGE[mode]
        decoded = ref.decode(payload, mode)
        assert not decoded.eligible[0, 0]
        if mode != "sparse40":
            np.testing.assert_array_equal(decoded.eligible.numpy(), eligible)
        if mode.startswith("q") or mode == "fp16":
            assert decoded.eligible[0, 1] and decoded.values[0, 1] == 0
        if mode == "exact":
            np.testing.assert_array_equal(decoded.values.numpy(), values)


def test_same_byte_sparse_keeps_top_40_original_weights_and_biases():
    values, eligible = example()
    decoded = ref.decode(ref.encode(values, eligible, "sparse40"), "sparse40")
    retained = decoded.eligible.numpy()
    assert retained.sum() == 40
    assert np.min(np.abs(values[:, :-1][retained])) >= np.max(np.abs(values[:, :-1][eligible & ~retained]))
    np.testing.assert_array_equal(decoded.values[:, -1].numpy(), values[:, -1])
    np.testing.assert_array_equal(decoded.values[:, :-1][retained].numpy(), values[:, :-1][retained])


def test_zero_target_keeps_reference_pull_and_live_fp32_support():
    values, eligible = example()
    decoded = ref.decode(ref.encode(values, eligible, "q2"), "q2")
    graph = ref.net.spawn(731)
    weight = torch.ones_like(graph.weights)
    shown = torch.ones_like(graph.present)
    graph = ref.net.Graph(weight, graph.bias, shown)
    output = ref.remap(graph, decoded, graph.bias.new_tensor(ref.GAMMA))
    assert torch.equal(output.present, graph.present)
    assert output.weights.dtype == graph.weights.dtype == torch.float32
    assert output.weights[0, 0, 0] == 1.
    assert torch.isclose(output.weights[0, 0, 1], torch.tensor(1. - ref.GAMMA))


def test_exact_indexed_reference_matches_original_assay_bitwise():
    torch.set_num_threads(1)
    g, r = ref.net.spawn(731), ref.net.Rule(731)
    tapes = ref.slow.public_tapes(4, 912830, 7)
    values, eligible = ref.birth_values(g)
    decoded = ref.decode(ref.encode(values, eligible, "exact"), "exact")
    compared = ref.replay(g, r, ref.physical(tapes), decoded)
    previous, sides, size = compact.evaluate(g, r, tapes, "absent_a", 4, "indexed_exact", [ref.GAMMA])
    selected = [ref.slow.AGES.index(age) for age in ref.AGES]
    np.testing.assert_array_equal(compared["correct"], previous[0, selected])
    np.testing.assert_array_equal(compared["side"], sides[0, selected])
    assert size == ref.STORAGE["exact"]


def test_full_interval_inside_margin_required():
    assert ref._inside((-.019, .019))
    assert not ref._inside((-.02, .019))
    assert not ref._inside((-.019, .02))
    assert not ref._inside((-.024, .001))
