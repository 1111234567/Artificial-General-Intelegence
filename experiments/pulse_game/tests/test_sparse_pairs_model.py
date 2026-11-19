"""Check sparse physics, signed birth gating, retirement, and native runs."""

import numpy as np

from experiments.pulse_game import mixture_world as food, world
from experiments.pulse_game.sparse_pairs_model import (
    BIRTH, EVIDENCE, RECORD, SUPPORT, DenseWGraph, PairPolicy, SparseDendriteGraph, coordinates,
)
from experiments.pulse_game.sparse_pairs_study import BASES, POLICIES, run, setup
from experiments.pulse_game.unified_live_graph import SIZE, Recipe


def physical_records(lanes, rng):
    obs = np.zeros((7, lanes, world.OBS_SIZE), dtype=np.float32)
    obs[:, :, food.LEFT.start:food.RIGHT.stop] = rng.random((len(obs), lanes, 8))
    obs[:-1, :, world.ACTIVE] = 1
    obs[-1, :, world.REWARD_POS] = 1
    obs[-1, :, world.CONTACT_ID.start] = 1
    return obs


def test_active_products_and_receipt_gradient_match_dense_at_the_same_addresses():
    rng = np.random.default_rng(3945542)
    seeds = [435, 503]
    dense = DenseWGraph(seeds, Recipe(), product=True, pair_rate=.65, pair_decay=.993)
    sparse = SparseDendriteGraph(seeds, Recipe(), world_seeds=[702, 799], pair_rate=.65, pair_decay=.993)
    assert not len(sparse.active)
    assert not any(hasattr(sparse, attr) for attr in ("pair", "pair_credit", "symmetric", "valid_pairs"))
    np.testing.assert_array_equal(sparse.weight, dense.weight)
    keys = np.sort(rng.choice(np.flatnonzero(dense.valid_pairs), 106, replace=False))
    weights = rng.normal(0, .08, len(keys)).astype(np.float32)
    records = np.zeros(len(keys), dtype=RECORD)
    records["key"], records["weight"] = keys, weights
    sparse.set_active(records)
    dense.pair.flat[keys] = weights
    dense.refresh_pair_support()
    states = rng.uniform(-.45, .45, (len(seeds), SIZE)).astype(np.float32)
    noise = np.array([.08, -.21], dtype=np.float32)
    for graph in (dense, sparse):
        graph.state[:] = states
        graph.start(noise)
    for tick, obs in enumerate(physical_records(len(seeds), rng)):
        cut = np.ones(SIZE, dtype=np.float32)
        cut[tick] = .25
        np.testing.assert_array_equal(dense.advance(obs, cut=cut), sparse.advance(obs, cut=cut))
        np.testing.assert_allclose(dense.state, sparse.state, atol=1e-7)
    dense_w, dense_q = dense.gradients()
    local, sent = sparse.receipt_credit()
    np.testing.assert_allclose(sparse.gradient(), dense_w, rtol=.00001, atol=2e-8)
    np.testing.assert_allclose(sparse.pair_gradient(keys, local, sent), dense_q.ravel()[keys], rtol=.00001, atol=2e-8)


def test_birth_and_later_updates_are_receipt_gated_and_not_count_locked():
    data = run([191, 225], [405, 466, 511], "stable", "guided", setup(BASES[0], POLICIES[1]), trials=288, tail=32)
    records = data["pair_events"]
    births = records[records["type"] == BIRTH]
    support = records[records["type"] == SUPPORT]
    evidence = records[records["type"] == EVIDENCE]
    assert len(births) and len(support) and len(evidence)
    assert not data["active"][:, 0].any()
    assert np.any(np.diff(data["active"], axis=1) > 0) and np.any(np.diff(data["active"], axis=1) < 0)
    lane, receiver, a, b = coordinates(data["final_active"]["key"])
    assert (data["final_mask"][lane, receiver, a] & data["final_mask"][lane, receiver, b]).all()
    np.testing.assert_allclose(births["weight"], (.65 * births["gradient"]).clip(-1.25, 1.25), atol=1e-8)
    assert (births["gradient"] < 0).any() and (births["gradient"] > 0).any()
    assert (data["final_active"]["writes"] > 0).mean() > .9
    assert not data["final_active"]["writes"][data["final_active"]["born"] == 255].any()
    assert not data["shown"][:, -32:].any()
    assert not data["writes"][:, -32:].any()
    assert not data["samples"][:, -32:].any()
    assert data["writes"][:, :, 4].sum() > 0


def test_suppressing_new_proposals_preserves_active_q_and_ordinary_learning():
    seed = [435]
    graph = SparseDendriteGraph(seed, Recipe(grow_every=1), world_seeds=[702], pair_rate=.65,
                                pair_decay=.993, policy=PairPolicy(samples=250, threshold=0, floor=0))
    rng = np.random.default_rng(993565)
    for trial in range(2):
        graph.start(np.asarray([.04], dtype=np.float32))
        for obs in physical_records(1, rng):
            graph.advance(obs)
        changed, _, ordinary, _ = graph.update(trial)
        assert changed[0] > 0 and ordinary[0, 0] >= 0
        if not trial:
            old = graph.active["key"].copy()
            assert len(old) > 10 and (graph.pair_events["type"] == BIRTH).all()
            graph.proposals_enabled = False
    np.testing.assert_array_equal(graph.active["key"], old)
    assert graph.active["writes"].sum() > 0
    assert not len(graph.pair_events) and not graph.proposal_count.any()
