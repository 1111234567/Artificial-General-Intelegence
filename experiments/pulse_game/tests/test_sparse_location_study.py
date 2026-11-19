import numpy as np
import pytest

from experiments.pulse_game import graph_interactions
from experiments.pulse_game.sparse_location_study import FIRST, PAIR_COUNT, SECOND, SIZE, reassign, score_fork
from experiments.pulse_game.sparse_pairs_model import RECORD, SparseDendriteGraph
from experiments.pulse_game.unified_live_graph import Recipe
from experiments.pulse_game.unified_live_graph_study import Body


def test_reassign_pairs_preserves_receiver_budget_and_signed_weights():
    mask = np.zeros((3, SIZE, SIZE), dtype=bool)
    mask[0, 8, :7] = True
    mask[0, 9, :2] = True
    mask[1, 8, 2:7] = True
    legal = mask[:, :, FIRST] & mask[:, :, SECOND]
    keys = []
    for lane, receiver, count in ((0, 8, 5), (0, 9, 1), (1, 8, 3)):
        keys.extend((lane * SIZE + receiver) * PAIR_COUNT + np.flatnonzero(legal[lane, receiver])[:count])
    source = np.array(list(zip(keys, [.2, -.4, .2, 0., -.1, -.7, .9, -.4, 0.])),
                      dtype=[("key", np.int32), ("weight", np.float32)])[::-1].copy()
    original = source.copy()
    selected, eligible, counts = reassign(source, mask, [1, 2, 3], [4, 5, 6], 3)

    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(selected["key"], original["key"])
    for replacement in (selected, eligible):
        assert len(np.unique(replacement["key"])) == len(replacement)
        receiver, site = np.divmod(replacement["key"], PAIR_COUNT)
        assert legal[receiver // SIZE, receiver % SIZE, site].all()
        for head in np.unique(receiver):
            expected = source["weight"][source["key"] // PAIR_COUNT == head]
            np.testing.assert_array_equal(np.sort(replacement["weight"][receiver == head]), np.sort(expected))
    np.testing.assert_array_equal(counts[:, :6], [[6, 5, 22, 22, 5, 5], [3, 2, 10, 10, 3, 3], [0] * 6])
    assert counts[:, 6:].sum() > 0


def test_reassign_rejects_ineligible_or_repeated_pairs():
    mask = np.zeros((1, SIZE, SIZE), dtype=bool)
    mask[0, 8, :2] = True
    active = np.array([(8 * PAIR_COUNT, .4), (8 * PAIR_COUNT, -.1)],
                      dtype=[("key", np.int32), ("weight", np.float32)])
    with pytest.raises(ValueError, match="distinct legal"):
        reassign(active, mask, [1], [2], 0)
    active["key"][1] += 1
    with pytest.raises(ValueError, match="distinct legal"):
        reassign(active, mask, [1], [2], 0)


def test_reassign_preserves_empty_receivers():
    mask = np.zeros((1, SIZE, SIZE), dtype=bool)
    mask[0, 8, :2] = True
    active = np.empty(0, dtype=[("key", np.int32), ("weight", np.float32)])
    selected, eligible, counts = reassign(active, mask, [1], [2], 0)
    assert not len(selected) and not len(eligible)
    np.testing.assert_array_equal(counts, [[0, 0, 1, 0, 0, 0, 0, 0, 0]])


def test_zero_fork_keeps_ordinary_policy_and_matches_an_empty_pair_list():
    graph = SparseDendriteGraph([1, 2], Recipe(), world_seeds=[4, 5], pair_rate=1., pair_decay=.996)
    graph.state[:] = np.linspace(-.4, .4, SIZE, dtype=np.float32)
    state = graph.state.copy()
    menu, _, _, _ = graph_interactions.make_menu([4, 5], "stable", 4)
    body = Body(menu, None, 4)
    obs = body.observe()
    legal = graph.mask[:, :, FIRST] & graph.mask[:, :, SECOND]
    active = np.zeros(2, dtype=RECORD)
    active["key"] = np.flatnonzero(legal.ravel())[:2]
    active["weight"] = [.7, -.4]
    graph.set_active(active)
    zero = active.copy()
    zero["weight"] = 0
    zeros = score_fork(graph, body, obs, zero, 4)
    empty = score_fork(graph, body, obs, np.empty(0, dtype=RECORD), 4)
    for key in zeros:
        np.testing.assert_array_equal(zeros[key], empty[key])
    np.testing.assert_array_equal(graph.state, state)
    np.testing.assert_array_equal(graph.active["weight"], np.array([.7, -.4], dtype=np.float32))
    assert (body.trial_index, body.tick_index) == (0, 0)
