import numpy as np

from src.core import Network as LegacyNetwork
from src.neurons.lif import NetSpec, Network, Plasticity, build_network


def test_build_network_matches_legacy_initialization_and_rng():
    spec = NetSpec(N=24, n_in=4, e0_init=80, use_std=True)
    plasticity = Plasticity(tau_trace=7.0, trace_decay=0.85, tau_theta=16.0)
    old_rng = np.random.default_rng(19)
    new_rng = np.random.default_rng(19)
    expected = LegacyNetwork(spec, old_rng, [10, 11])
    expected.tau_trace = plasticity.tau_trace
    expected.trace_decay = plasticity.trace_decay
    expected.tau_theta = plasticity.tau_theta
    actual = build_network(spec, new_rng, [10, 11], plasticity=plasticity)

    assert type(actual) is Network is LegacyNetwork
    assert actual.tau_trace == expected.tau_trace
    assert actual.trace_decay == expected.trace_decay
    assert actual.tau_theta == expected.tau_theta
    for field in ("pre", "post", "w", "key", "age"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    np.testing.assert_array_equal(new_rng.random(4), old_rng.random(4))

    inputs = np.random.default_rng(21).uniform(0, 1.5, (10, spec.N)).astype(np.float32)
    for xin in inputs:
        assert actual.step(xin) == expected.step(xin)
        for field in ("v", "state", "ref", "trace", "theta", "x_rec", "rate_ema"):
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))


def test_task_can_advance_and_train_lif_from_feedback():
    spec = NetSpec(N=2, n_in=1, e_ratio=1.0, e0_init=20)
    net = build_network(spec, np.random.default_rng(7), [1])
    np.testing.assert_array_equal(net.pre, [0])
    np.testing.assert_array_equal(net.post, [1])
    assert net.step(np.array([1.25, 0.0], dtype=np.float32)) == 1
    np.testing.assert_array_equal(net.state, [1, 0])
    assert net.step(np.array([0.0, 0.75], dtype=np.float32)) == 1
    np.testing.assert_array_equal(net.state[net.out_idx], [1])

    net.apply_weight(m_vec=np.array([0.5], dtype=np.float32), m_glob=None, eta=0.25)
    np.testing.assert_array_equal(net.w, np.array([0.625], dtype=np.float32))


def test_task_can_grow_and_prune_active_edges():
    spec = NetSpec(N=2, n_in=1, e_ratio=1.0, e0_init=0)
    net = build_network(spec, np.random.default_rng(7), [])
    net.step(np.array([1.25, 1.25], dtype=np.float32))
    assert net.apply_topology(thresh=0.3, budget=1, p_new=1.0, max_active=2, min_age=1) == 0
    assert net.last_born == 1
    np.testing.assert_array_equal(net.pre, [0])
    np.testing.assert_array_equal(net.post, [1])

    assert net.apply_topology(thresh=0.3, budget=0, p_new=1.0, max_active=2,
                              min_age=1, grow=False) == 1
    assert net.w.size == 0
