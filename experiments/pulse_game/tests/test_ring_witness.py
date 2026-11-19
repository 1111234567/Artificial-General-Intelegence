"""Check exact local witnesses against the actual recurrent forward path."""

import math

import torch

from experiments.pulse_game import ring_logic_model as model, ring_witness as exact


def test_equilibrium_impulse_moments_match_the_nonlinear_model_derivative():
    cases = tuple(exact.witnesses().values())
    width = exact.WIDTH
    ticks = 2 * width
    bank = model.Bank([0] * len(cases), [case.arm for case in cases], width=width).double()
    with torch.no_grad():
        bank.drive.zero_()
        bank.bias.zero_()
        bank.offset.zero_()
        for row, case in enumerate(cases):
            drive = torch.tensor(case.drive, dtype=torch.float64)
            keep = float(case.keep)
            branch_scale = math.sqrt(2) if case.arm == "multiloop" else 1.
            bank.drive[row, 0, :width] = drive / (1. - keep)
            bank.bias[row, width:] = math.log(keep / (1. - keep))
            bank.gain[row] = torch.tensor(list(map(float, case.scaled_gain)), dtype=torch.float64)
            bank.gain[row] *= branch_scale / (1. - keep)
            bank.readout[row] = drive
            if case.arm == "multiloop":
                bank.secondary[row] = torch.tensor(exact.ANTIPODE)

    base = bank.drive.new_zeros((len(cases), 1, ticks, model.SENSES))
    base[:, :, 0, 0] = 1.

    def run(pulse):
        logits, _ = bank(base * pulse[:, None, None, None], bank.start(1), retain=0)
        return logits[:, 0]

    start = bank.drive.new_zeros(len(cases))
    baseline, tangent = torch.autograd.functional.jvp(run, start, torch.ones_like(start))
    torch.testing.assert_close(baseline, torch.zeros_like(baseline))
    for row, case in enumerate(cases):
        expected = tangent.new_tensor(list(map(float, case.moments())))
        torch.testing.assert_close(tangent[row, :len(expected)], expected, atol=1e-12, rtol=1e-10)


def test_fixed_ports_separate_three_one_tick_boolean_operations():
    bank = model.Bank([0, 0, 0], ["self", "directed", "multiloop"]).double()
    with torch.no_grad():
        bank.drive.zero_()
        bank.bias.zero_()
        bank.gain[0].fill_(-2.)
        bank.gain[1:].fill_(1.)
        bank.secondary[2] = torch.tensor(exact.ANTIPODE)

    state = bank.start(8)
    state[:, 0:2, 0] = state.new_tensor([.5, -.5])
    state[:, 2:6, 0] = state.new_tensor([.5, .5, -.5, -.5])
    state[:, 2:6, 17] = state.new_tensor([.5, -.5, .5, -.5])
    state[:, 6:8, 3] = state.new_tensor([.5, -.5])
    observation = state.new_zeros((3, 8, 1, model.SENSES))
    _, end = bank(observation, state)

    assert (end[:, 0:2, 16] > 0).tolist() == [[False, False], [False, False], [True, False]]
    assert (end[:, 2:6, 1] > 0).tolist() == [
        [False, False, False, False], [True, True, False, False], [True, False, False, False],
    ]
    assert (end[:, 6:8, 3] > 0).tolist() == [[False, True], [True, False], [True, False]]
