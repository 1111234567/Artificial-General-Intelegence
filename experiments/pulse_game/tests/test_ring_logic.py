"""Fairness, causal input, and exact edge constraints for the ring logic assay."""

import numpy as np
import torch

from experiments.pulse_game import ring_logic_model as m, ring_logic_study as s
from experiments.pulse_game import ring_logic_world as e


def test_rule_panel_balances_both_single_cues_and_has_three_swap_signatures():
    truth = e.truth()
    for odor in range(len(e.ODORS)):
        np.testing.assert_array_equal(truth[:, e.PAIRS[:, 0] == odor].sum(axis=1), [2, 2, 2])
        np.testing.assert_array_equal(truth[:, e.PAIRS[:, 1] == odor].sum(axis=1), [2, 2, 2])
    np.testing.assert_array_equal((truth != truth[:, e.SWAP]).sum(axis=1), [20, 0, 8])


def test_forks_share_prefix_background_and_geometry_and_deliver_public_taste():
    roots = range(15)
    prefix, observations, ticks = e.sensory(roots, "standard")
    tape = e.body_tape(roots)
    for i, positions in enumerate(ticks):
        odors = observations[i][:, positions, :3]
        np.testing.assert_array_equal(odors, odors[e.SWAP][:, ::-1])
        shared = observations[i].copy()
        shared[:, positions, :3] = 0
        np.testing.assert_array_equal(shared, np.broadcast_to(shared[0], shared.shape))
    np.testing.assert_array_equal(prefix, e.sensory(roots, "long")[0])
    side, taste = e.exploration(tape, 57)
    valid = side >= 0
    expected = np.where(side[None] == e.truth()[:, None, :], 1, -1)
    np.testing.assert_array_equal(taste, np.where(valid, expected, 0))
    for rule in range(len(e.RULES)):
        assert (e.oracle(tape, observations, rule)[:, 2] == 1).all()


def test_all_arms_have_identical_birth_and_candidate_scale_with_correct_graphs():
    seed = 51
    bank = m.Bank([seed] * len(m.ARMS), m.ARMS).double()
    count = sum(p[0].numel() for p in bank.parameters())
    assert count == 449
    for parameter in bank.parameters():
        for row in parameter[1:]:
            torch.testing.assert_close(row, parameter[0], rtol=0, atol=0)
    directed, reciprocal, independent, multiloop = bank.matrix()
    torch.testing.assert_close(reciprocal, reciprocal.T, rtol=0, atol=0)
    torch.testing.assert_close(reciprocal, (directed + directed.T) / np.sqrt(2), rtol=3e-16, atol=0)
    torch.testing.assert_close(independent, torch.diag(bank.gain[0]), rtol=0, atol=0)
    torch.testing.assert_close(bank.matrix().norm(dim=(-2, -1)), bank.gain.norm(dim=-1))
    assert not torch.equal(directed, directed.T)
    assert (multiloop.diag() == 0).all()
    for dim in (0, 1):
        assert (torch.count_nonzero(multiloop, dim=dim) == 2).all()
    graph = s.topology(seed, "multiloop")
    assert graph["strong_components"] == 1
    assert 1 not in graph["shortest_directed_return"]
    assert len(set(graph["shortest_directed_return"])) > 1


def test_vectorized_and_sparse_paths_match_explicit_recurrence():
    torch.manual_seed(45)
    seeds = [19, 19, 19, 19]
    bank = m.Bank(seeds, m.ARMS).double()
    x = torch.randn(1, 8, 9, e.SENSES, dtype=torch.float64)
    dense_logits, dense_state = bank(x, bank.start(8), retain=0)
    matrices = bank.matrix()
    state = bank.start(8)
    states = []
    for tick in x[0].transpose(0, 1):
        drive, keep = (tick @ bank.drive + bank.bias[:, None]).chunk(2, dim=-1)
        drive = drive + state @ matrices.transpose(-2, -1)
        state = torch.lerp(drive.tanh(), state, keep.sigmoid())
        states.append((state @ bank.readout[..., None]).squeeze(-1) + bank.offset[:, None])
    torch.testing.assert_close(dense_logits, torch.stack(states, dim=-1))
    torch.testing.assert_close(dense_state, state)
    for path, (seed, arm) in enumerate(zip(seeds, m.ARMS, strict=True), start=1):
        single = m.Bank([seed], [arm]).double()
        actual, state = single(x, single.start(8), retain=0, path=path)
        torch.testing.assert_close(actual[0], dense_logits[path - 1])
        torch.testing.assert_close(state[0], dense_state[path - 1])


def test_controls_hit_exact_conditional_chance_even_with_timeouts():
    torch.set_num_threads(1)
    torch.manual_seed(73)
    rows = [(0, 0, 31, 0, "directed"), (0, 0, 31, 2, "multiloop")]
    bank = s.bank_for(rows, "cpu")
    roots = range(8)
    p, q, tick = e.sensory(roots, "standard")
    episode = e.body_tape(roots), p, q, tick, "standard"
    for case in ("first_only", "second_only", "timing_only"):
        score = s.score(bank, rows, episode, case)
        receipt = score[..., 2]
        np.testing.assert_array_equal(2 * (receipt > 0).sum(axis=-1), (receipt != 0).sum(axis=-1))
