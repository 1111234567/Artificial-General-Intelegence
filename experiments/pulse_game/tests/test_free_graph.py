"""Specific correctness checks for graph draws, tying, and paired selection."""

from itertools import product
import random

import numpy as np
import torch

from experiments.pulse_game import free_graph_model as m, free_graph_study as s
from experiments.pulse_game import ring_logic_model as ring, ring_logic_world as env


def test_exact_sequence_completion_counts_match_exhaustive_enumeration():
    n = 4
    length = 6
    counts = m._completion_counts(n, length)
    for rest in range(length + 1):
        for missing in range(n + 1):
            total = sum(set(range(missing)).issubset(seq) for seq in product(range(n), repeat=rest))
            assert counts[rest][missing] == total
    rng = random.Random(123)
    sequences = [m.covering_sequence(rng, n, length) for _ in range(20)]
    assert all(set(seq) == set(range(n)) for seq in sequences)


def test_graphs_match_edge_budget_and_distinguish_degree_and_backbone_rules():
    seed = 51
    records = {arm: m.graph_record(seed, arm) for arm in m.ARMS}
    assert all(meta["strongly_connected"] and meta["self_edges"] == 0 and meta["arc_count"] == 64
               for meta in records.values())
    assert not m.strongly_connected(4, [(0, 1), (1, 0), (2, 3), (3, 2)])
    regular = records["regular"]
    assert set(regular["in_degree"]) == set(regular["out_degree"]) == {2}
    assert records["free_chords"]["ordered_backbone_arcs_present"] == 32
    assert records["fully_free"]["ordered_backbone_arcs_present"] < 32
    for meta in (records["free_chords"], records["fully_free"]):
        assert min(meta["in_degree"]) == min(meta["out_degree"]) == 1
        assert max(meta["in_degree"]) > 2 and max(meta["out_degree"]) > 2


def test_parameter_counts_matched_initial_scales_and_existing_regular_arm():
    seed = 441
    tied = m.Bank([seed] * 3, m.ARMS).double()
    untied = m.Bank([seed] * 3, m.ARMS, independent=True).double()
    original = ring.Bank([seed], ["multiloop"]).double()
    assert sum(p[0].numel() for p in tied.parameters()) == 449
    assert sum(p[0].numel() for p in untied.parameters()) == 481
    torch.testing.assert_close(tied.matrix(), untied.matrix(), rtol=0, atol=0)
    torch.testing.assert_close(tied.matrix().norm(dim=-1), tied.gain.abs())
    torch.testing.assert_close(tied.matrix()[:1], original.matrix())
    for param in tied.parameters():
        torch.testing.assert_close(param, param[:1].expand_as(param), rtol=0, atol=0)
    generator = torch.Generator().manual_seed(112)
    query = torch.randn(1, 5, 11, env.SENSES, dtype=torch.float64, generator=generator)
    old, _ = original(query, original.start(5), retain=0)
    logits, _ = tied(query, tied.start(5), retain=0)
    torch.testing.assert_close(logits[:1], old)
    objective = untied(query, untied.start(5), retain=0)[0].square().sum()
    objective.backward()
    assert untied.gain.grad is not None and torch.all(torch.isfinite(untied.gain.grad))
    same_receiver = untied.receiver[0] == untied.receiver[0, 0]
    assert torch.unique(untied.gain.grad[0, same_receiver]).numel() == 2


def test_selection_and_checkpoint_roundtrip(tmp_path):
    design = "independent"
    rows = s.specs(s.births()[:2], design, 4)
    selected = [[[0, 1], [1, 0], [1, 1]], [[1, 1], [0, 1], [0, 0]]]
    pick = s.selection_indices({"design": design, "calibration_selected": selected})
    bank = s.make_bank(rows, design, "cpu")
    path = tmp_path / "weights.npz"
    s.save_checkpoint(path, bank, rows, design)
    reloaded, chosen = s.load_checkpoint(path, "cpu", pick)
    assert [row[1] for row in chosen] == np.asarray(selected).ravel().tolist()
    assert [row[0] for row in chosen] == np.repeat([4, 5], 6).tolist()
    torch.testing.assert_close(reloaded.matrix(), bank.matrix()[pick], rtol=0, atol=0)
