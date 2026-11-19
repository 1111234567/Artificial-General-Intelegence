import numpy as np
import torch

from experiments.pulse_game import plastic_topology_model as model, plastic_topology_study as study


def test_edit_updates_actual_degree_without_replacing_captured_buffer():
    width = 4
    birth = np.roll(np.eye(width, dtype=bool), 1, axis=0)
    changed = birth.copy()
    changed[0, 1] = True
    bank = model.Bank([17], birth[None], width=width)
    pointer = bank.support.data_ptr()
    bank.edit(changed[None])
    assert bank.support.data_ptr() == pointer
    assert model.strong(changed)
    torch.testing.assert_close(bank.support.square().sum(dim=-1), torch.ones(1, width))
    assert bank.support[0, 0, 1].item() > 0


def test_edit_proposals_allow_variable_count_and_protect_reachability():
    width = 5
    ring = np.roll(np.eye(width, dtype=bool), 1, axis=0)
    no_retire = study.propose(ring, np.random.default_rng(3172))
    assert all(p is None for p in no_retire[5:])
    dense = ~np.eye(width, dtype=bool)
    no_growth = study.propose(dense, np.random.default_rng(1290))
    assert all(p is None for p in no_growth[1:5])
    assert all(model.strong(p) and p.sum() == dense.sum() + size
               for p, size in zip(no_growth[5:], study.PROPOSALS[5:], strict=True))


def test_selector_keeps_when_no_gain_and_uses_matched_control_size():
    same = np.ones(len(study.PROPOSALS))
    available = [object()] * len(same)
    assert study.choice(same, available, np.random.default_rng(10)) == (0, 0)
    loss = same.copy()
    loss[6] = .9
    loss[8] = .8
    available[7:] = [None, None]
    guided, random = study.choice(loss, available, np.random.default_rng(10))
    assert guided == 6
    assert study.PROPOSALS[guided] == study.PROPOSALS[random] == -1
