from dataclasses import replace

import numpy as np
import torch

from experiments.pulse_game import interleave_homeostasis as home, local_trace_graph as net


def test_interpolation_replays_native_clock_and_preserves_the_ordinary_support():
    graph = net.spawn(731)
    rule = net.Rule(731).expand(1)
    template = graph
    state = net.start(graph)
    right = torch.tensor([[.5, 1., .5]])
    act = net.act(rule, state, -right, right, torch.tensor([.3]))
    seen = net.observe(rule, act, -right, right, act.side)
    committed = net.commit(rule, seen, torch.tensor([[0., .3, 0., .7]]))
    ancestral = graph.present.clone()
    inherited = (ancestral & committed.state.graph.present).nonzero()[0]
    ancestral[tuple(inherited)] = False
    ordinary, mask = home.interpolate(rule, seen, committed, template, ancestral, "birth", 0.)
    assert ordinary is committed.state
    assert torch.equal(mask, ancestral & ordinary.graph.present)

    after_write = committed.state.graph
    ready = net.wait(rule, seen.state, net.PRE_WRITE_TICKS)
    for ref, w, b in (("birth", template.weights, template.bias),
                      ("zero", torch.zeros_like(template.weights), torch.zeros_like(template.bias)),
                      ("prewrite", seen.state.graph.weights, seen.state.graph.bias)):
        result, surviving = home.interpolate(rule, seen, committed, template, ancestral, ref, .35)
        weights = torch.where(mask, net._nonzero(after_write.weights.lerp(w, .35)), after_write.weights)
        bias = after_write.bias.lerp(b, .35)
        predicted = net.wait(rule, replace(ready, graph=replace(after_write, weights=weights, bias=bias)),
                             net.POST_WRITE_TICKS)
        assert torch.equal(surviving, mask)
        assert torch.equal(result.graph.present, after_write.present)
        assert torch.equal(result.graph.weights, weights)
        assert torch.equal(result.graph.bias, bias)
        assert torch.equal(result.hidden, predicted.hidden)
        assert torch.equal(result.sender_trace, predicted.sender_trace)
        assert torch.equal(result.receiver_trace, predicted.receiver_trace)


def test_matched_ecologies_and_novel_taste_share_prefixes():
    tape = home.make_tape(6, 44, 15, 16)
    recurrent, absent = tape.visits
    assert torch.equal(recurrent == 1, absent == 1)
    assert (recurrent[2:] == 0).any() and not (absent[2:] == 0).any()
    assert not (tape.visits == 2).any()
    assert torch.equal(recurrent[:2], absent[:2])
    graph, rule = net.spawn(731), net.Rule(731)
    result = home.replay(graph, rule, tape, "absent_a", (0, 4, 16), "birth", .12)
    assert result["side"].shape == (3, 6, 2, 2, 3)
    assert np.array_equal(result["a_before_side"][..., 0], result["a_before_side"][..., 1])
    assert np.array_equal(result["target"][..., 0, :2], result["target"][..., 1, :2])
    assert np.array_equal(result["target"][..., 0, 2], -result["target"][..., 1, 2])
    assert (result["ancestral_after_x"] <= result["support_after_x"]).all()
    assert home.correctness(result).shape == (3, 6, 2, 2, len(home.METRICS))


def test_gain_selection_uses_all_threes_aged_cells_and_smaller_ties():
    births = (731, 732)
    grid = (0., .01, .04)
    data = {}
    for birth in births:
        for ecology in home.ECOLOGIES:
            for label in ("unchanged", *(home._label(ref, gain) for ref in home.REFERENCES for gain in grid[1:])):
                data[birth, ecology, label] = np.zeros((len(home.AGES), 5, len(home.METRICS)))
            data[birth, ecology, "zero-0.04"][0, :, -1] = 1.
            for gain in grid[1:]:
                data[birth, ecology, home._label("birth", gain)][2:, :, -1] = .5
    chosen = home.select(data, births, home.AGES, grid)
    assert chosen["zero"]["selected_gain"] == 0.
    assert chosen["birth"]["selected_gain"] == .01
