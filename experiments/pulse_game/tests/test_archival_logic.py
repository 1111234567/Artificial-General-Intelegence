"""Regression checks for archive decoding, paired starts, and directed dynamics."""

import numpy as np
import torch

from experiments.pulse_game import archival_logic_model as model, archival_logic_study as study


def test_champion_support_matches_archived_loci():
    candidate = torch.load(study.ARCHIVE, map_location="cpu", weights_only=True)["genome"]
    active = candidate["birth_mask"] & candidate["birth_w"].ne(0)
    support = study.load_support()
    targets, sources = torch.triu_indices(96, 96, 1).numpy()
    np.testing.assert_array_equal(support[targets, sources], active[:4560].numpy())
    np.testing.assert_array_equal(support[sources, targets], active[4560:].numpy())
    assert support.sum() == 1218 and not support.trace()
    assert (support.sum(axis=1).min(), support.sum(axis=1).max()) == (4, 22)
    assert (support.sum(axis=0).min(), support.sum(axis=0).max()) == (5, 20)


def test_switched_control_protects_degrees_and_fits_equal_parameters():
    archive = study.load_support()
    control, mix = study.switched(archive, 796356981)
    assert mix[-1]["accepted"] >= 20 * study.EDGES
    assert int((archive & control).sum()) < study.EDGES / 4
    assert study.strongly_connected(control) and not control.trace()
    np.testing.assert_array_equal(control.sum(axis=0), archive.sum(axis=0))
    np.testing.assert_array_equal(control.sum(axis=1), archive.sum(axis=1))
    bank = model.Bank([123, 123], [archive, control])
    assert sum(p[0].numel() for p in bank.parameters()) == 2467
    for parameter in bank.parameters():
        torch.testing.assert_close(parameter[0], parameter[1], rtol=0, atol=0)
    np.testing.assert_array_equal(bank.matrix().detach().ne(0).numpy(), [archive, control])


def test_recurrence_has_directed_weight_gradient_on_every_active_arc():
    support = np.array([[0, 1, 1], [0, 0, 1], [1, 0, 0]], dtype=bool)
    bank = model.Bank([63], [support])
    start = torch.tensor([[[.5, .7, .9]]])
    blank = torch.zeros(1, 1, 1, 3)
    matrix = bank.matrix().transpose(-1, -2)
    [state], last = model.advance(blank, blank, start, matrix, torch.empty(0), 0)
    v0, v1, v2, v3 = bank.edges.unbind(-1)
    expected = torch.stack((start[0, 0, 1] * v0 + start[0, 0, 2] * v1,
                            start[0, 0, 2] * v2, start[0, 0, 0] * v3), dim=-1)[:, None, :].tanh()
    torch.testing.assert_close(last, expected)
    state.sum().backward()
    assert bank.edges.grad.ne(0).all()
