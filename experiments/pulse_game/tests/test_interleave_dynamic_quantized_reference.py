from dataclasses import replace

import numpy as np
import torch

from experiments.pulse_game import interleave_dynamic_quantized_reference as quant


def test_nearest_dead_zone_and_unbiased_stochastic_rounding():
    base = torch.tensor([0., 1., -1., 2.])[:, None, None]
    bound = torch.tensor([[7.]])
    proposal = base + .02
    nearest = quant.code(proposal, bound, quant.MODES["q4_nearest"])
    assert torch.equal(quant.decode(nearest, bound, quant.MODES["q4_nearest"]), base)
    grid = ((torch.arange(10000) + .5) / 10000)[:, None, None, None]
    stochastic = quant.code(proposal, bound, quant.MODES["q4_stochastic"], grid)
    expected = quant.decode(stochastic, bound, quant.MODES["q4_stochastic"]).mean(0)
    torch.testing.assert_close(expected, proposal, rtol=0, atol=1e-5)


def test_snapshots_include_packed_surviving_mask_and_shared_radius():
    graph = quant.net.spawn(731)
    for mode in quant.MODES.values():
        state, radius = quant._initial(graph, 2, mode)
        packed = quant.pack_snapshot(state, radius, mode)
        counts = quant.storage(mode)
        assert counts["row_scales"] == 80 and counts["packed_ancestry"] == 50
        assert len(packed) == counts["values"] + 130
        mask = quant.frozen.unpack_words(packed[-50:], 1, quant.net.NODES**2)
        np.testing.assert_array_equal(mask, state.ancestry[0].flatten().numpy())


def test_transplant_keeps_recipient_live_graph_and_intersects_masks():
    memory, _ = quant._initial(quant.net.spawn(731), 4, quant.MODES["q4_nearest"])
    masks = memory.ancestry.clone()
    place = masks[0].nonzero()[0].tolist()
    masks[2, place[0], place[1]] = False
    codes = memory.ref.clone()
    codes[0, place[0], place[1]], codes[2, place[0], place[1]] = -6, 5
    codes[0, 0, -1], codes[2, 0, -1] = 2, -2
    memory = replace(memory, ref=codes, ancestry=masks)
    other, changed = quant.swapped(memory, torch.arange(4) ^ 2)
    assert other.live is memory.live and other.ancestry is memory.ancestry
    assert other.ref[0, place[0], place[1]] == -6
    assert other.ref[0, 0, -1] == -2
    assert changed[0, 1] == 1


def test_replay_retains_actual_draws_and_reproduces_stochastic_bytes():
    torch.set_num_threads(1)
    info = dict(seed=7, domain=8)
    tape = quant._physical(quant.slow.public_tapes(3, info["seed"], info["domain"]), "absent_a")
    g, r = quant.net.spawn(731), quant.net.Rule(731)
    raw, stored = quant.replay(g, r, tape, "q4_stochastic", info, ages=(1, 4))
    repeated, stored_again = quant.replay(g, r, tape, "q4_stochastic", info, ages=(1, 4))
    assert stored == stored_again and all(np.array_equal(raw[k], repeated[k]) for k in raw)
    assert raw["side"].shape == (2, 2, 3, 4, 2, 3)
    assert np.all(np.abs(raw["x_taste"]) == 1)
    assert np.array_equal(raw["correct"][..., 0][:, 0], raw["correct"][..., 0][:, 1])
    assert raw["history_traffic"][..., 1].sum() > 0
