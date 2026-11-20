"""Check isolation, live timing and numerical dose in the one-draft mechanism."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from experiments.pulse_game import draft_rehearse as draft
from experiments.pulse_game import edge_site_eval
from experiments.pulse_game.interleave_interference import worlds


@pytest.fixture
def prepared():
    torch.set_num_threads(1)
    graph, rule = draft.net.spawn(731), draft.net.Rule(731)
    world = worlds(1571, 0, 12, 15)
    prefix = draft.prepare(graph, rule, world, seed=281168, lanes_per_geometry=4)
    return graph, rule, world, prefix


def test_temporary_graphs_share_prewrite_activity_and_never_enter_live_state(prepared):
    _, _, _, prefix = prepared
    net = draft.net
    assert prefix.ready_b.graph is prefix.experience_b.state.graph
    post = draft.rehearse(prefix, "post")
    pre = draft.rehearse(prefix, "pre")
    assert post.start.graph is prefix.native_b.state.graph
    assert pre.start.graph is prefix.ready_b.graph
    assert not torch.equal(post.start.graph.weights, pre.start.graph.weights)
    assert torch.equal(post.start.hidden, prefix.native_b.state.hidden)
    ready = net.wait(prefix.rule, prefix.experience_b.state, net.PRE_WRITE_TICKS)
    ticks = net.POST_WRITE_TICKS + draft.REHEARSAL_TICKS
    uninterrupted = net.wait(prefix.rule, ready, ticks)
    assert torch.equal(pre.start.hidden, net.wait(prefix.rule, ready, net.POST_WRITE_TICKS).hidden)
    assert prefix.live.graph is uninterrupted.graph
    assert draft.apply(prefix, mode="native").graph is prefix.native_b.state.graph
    assert not torch.equal(prefix.live.hidden, post.end.hidden)
    for name in ("hidden", "sender_trace", "receiver_trace"):
        assert torch.equal(getattr(prefix.live, name), getattr(uninterrupted, name))
        assert torch.equal(getattr(prefix.live, name), getattr(draft.apply(prefix, mode="native"), name))


def test_replay_uses_exactly_two_emitted_sides_and_real_archives(prepared):
    _, _, world, prefix = prepared
    net = draft.net
    root_orders = prefix.retained_first.reshape(-1, 4)
    assert (root_orders[:, :1] == root_orders).all()
    assert root_orders[:, 0].sum() == len(root_orders) // 2
    for vector in (world.a, world.b):
        torch.testing.assert_close((prefix.c * vector).sum(1), torch.zeros(len(vector)), atol=4e-7, rtol=0)
    assert prefix.study_c.commit.state.graph is prefix.study_a.before.graph
    assert prefix.study_a.commit.state.graph is prefix.study_b.before.graph
    assert prefix.study_c.archived.taste is prefix.study_c.experience.taste
    assert (prefix.study_c.commit.receipt.weight_delta != 0).any()
    for mode in draft.MODES:
        with patch.object(net, "act", wraps=net.act) as act, patch.object(net, "commit") as commit:
            reads = draft.rehearse(prefix, mode)
        assert act.call_count == 2
        commit.assert_not_called()
        retained = prefix.study_c if "wrong" in mode else prefix.study_a
        assert reads.retained.archived is retained.archived
        assert reads.current.archived is prefix.study_b.archived
        for read in (reads.retained, reads.current):
            stored = read.archived
            assert torch.equal(read.experience.taste, stored.taste * stored.side * read.action.side)
            assert torch.equal(read.experience.side, read.action.side)


def test_final_probe_fields_are_outside_rehearsal_and_selector(prepared):
    graph, rule, world, prefix = prepared
    orders = world.orders.clone()
    draws = world.draws.clone()
    orders[:, 2:] *= -1
    draws[:, 2:] = 1 - draws[:, 2:]
    changed = draft.prepare(graph, rule, replace(world, orders=orders, draws=draws), seed=281168, lanes_per_geometry=4)
    for mode in draft.MODES:
        before = draft.edge_features(prefix, draft.rehearse(prefix, mode))
        after = draft.edge_features(changed, draft.rehearse(changed, mode))
        assert torch.equal(before, after)


def test_equal_dose_only_changes_selected_addresses(prepared):
    _, _, _, prefix = prepared
    candidate = prefix.candidates
    alternatives = [torch.full_like(candidate.valid, ix, dtype=torch.long) for ix in (0, 3)]
    states = [draft.apply(prefix, addresses) for addresses in alternatives]
    physical = [draft.physical_dose(prefix, state, addresses, "selected")
                for state, addresses in zip(states, alternatives, strict=True)]
    selected = candidate.valid.numpy()
    assert np.all(physical[0][1][selected] != physical[1][1][selected])
    columns = edge_site_eval.DOSE_NAMES
    for name in ("deferred_count", "selected_count"):
        np.testing.assert_equal(physical[0][0][:, columns.index(name)], selected.sum(1))
    for suffix in ("positive_l1", "negative_l1"):
        for dose, _ in physical:
            np.testing.assert_allclose(dose[:, columns.index(f"deferred_{suffix}")],
                                       dose[:, columns.index(f"plan_{suffix}")], atol=1.5e-6, rtol=0)
    accept = torch.arange(len(candidate.valid)) % 2 == 0
    mixed = draft.apply(prefix, alternatives[0], accept=accept)
    native = draft.apply(prefix, mode="native")
    assert torch.equal(mixed.graph.weights[accept], native.graph.weights[accept])
    assert torch.equal(mixed.graph.weights[~accept], states[0].graph.weights[~accept])


def test_shared_local_score_matches_the_sampled_gradient(prepared):
    _, _, _, prefix = prepared
    weights = torch.linspace(-1, 1, len(draft.FEATURES), requires_grad=True)
    draw = torch.full_like(prefix.candidates.amount, 0.7)
    selected = draft.choose(prefix, draft.rehearse(prefix), weights, draw=draw, gradient=True)
    logits = selected.features @ weights
    chosen = logits.log_softmax(-1).gather(-1, selected.index[..., None]).squeeze(-1)
    expected, = torch.autograd.grad(chosen[prefix.candidates.valid].sum(), weights)
    torch.testing.assert_close(selected.score_gradient.sum(0), expected, atol=1e-4, rtol=1e-5)


def test_followup_b_uses_same_real_life_and_original_c_a_archives(prepared):
    _, _, world, prefix = prepared
    state = draft.apply(prefix, mode="native")
    again = draft.prepare_next_b(prefix, state, world, seed=754, lanes_per_geometry=4)
    ready = draft.net.wait(prefix.rule, state, draft.PREPARE_GAP)
    assert again.study_c is prefix.study_c
    assert again.study_a is prefix.study_a
    assert again.study_b.before.graph is state.graph
    assert torch.equal(again.study_b.before.hidden, ready.hidden)
