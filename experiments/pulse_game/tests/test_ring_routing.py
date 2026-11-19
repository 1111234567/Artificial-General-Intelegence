"""Check matched birth and parameter accounting in the routing sensitivity."""

import math

import torch

from experiments.pulse_game import ring_logic_model as model, ring_routing_study as routing


def test_all_routing_arms_start_at_the_native_multiloop_function():
    rows = [(0, 0, 429, 0, arm) for arm in routing.ARMS]
    routed = routing.RoutingBank(rows)
    native = model.Bank([429] * 3, ["multiloop"] * 3)
    torch.testing.assert_close(routed.matrix(), native.matrix(), atol=0, rtol=0)
    torch.testing.assert_close(routed.secondary, native.secondary, atol=0, rtol=0)
    observation = torch.linspace(-1., 1., 55).reshape(1, 1, 11, 5)
    active, state = routed(observation, routed.start(1), retain=0)
    expected, expected_state = native(observation, native.start(1), retain=0)
    torch.testing.assert_close(active, expected, atol=0, rtol=0)
    torch.testing.assert_close(state, expected_state, atol=0, rtol=0)


def test_angle_conserves_row_norm_and_free_chord_does_not_change_backbone():
    rows = [(0, 0, 17, 0, arm) for arm in routing.ARMS]
    bank = routing.RoutingBank(rows).double()
    assert sum(p.numel() for p in bank.parameters()) == 449 + 449 + 481
    birth = bank.matrix().detach()
    with torch.no_grad():
        bank.gain[1].fill_(-math.pi / 4.)
        bank.chord_gain[0].add_(.5)
    moved = bank.matrix()
    torch.testing.assert_close(moved[0], birth[0], atol=0, rtol=0)
    torch.testing.assert_close(moved[1].norm(dim=-1), birth[1].norm(dim=-1))
    target = torch.arange(bank.width)
    torch.testing.assert_close(moved[1, target, bank.secondary[1]], torch.zeros_like(bank.gain[1]))
    torch.testing.assert_close(moved[2, target, (target - 1) % bank.width],
                               birth[2, target, (target - 1) % bank.width], atol=0, rtol=0)
    assert torch.all(moved[2, target, bank.secondary[2]] > birth[2, target, bank.secondary[2]])
