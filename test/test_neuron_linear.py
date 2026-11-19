import pytest
import torch
from torch import nn

from experiments.pulse_game.mixture_circuit import Config, TasteCircuit
from src.neurons.linear import LinearGraph
from src.neurons.training import train_chunk


@pytest.mark.parametrize("graph", ["rotations", "diagonal"])
def test_frozen_layout_and_future_blind_pulse_reads(graph):
    torch.manual_seed(7)
    old = TasteCircuit(Config(nodes=8, graph=graph)).double()
    cell = LinearGraph(6, 8, graph=graph).double()
    cell.load_state_dict({name: old.state_dict()[name] for name in cell.state_dict()})
    torch.testing.assert_close(cell.matrix(), old.matrix(), rtol=0, atol=0)
    batch, blocks = 3, 5
    raw = {"odor": torch.randn(batch, blocks, 2, 4, dtype=torch.double),
           "ctx": torch.randn(batch, blocks, 2, dtype=torch.double),
           "ot": torch.randint(0, 16, (batch, blocks)), "ct": torch.randint(0, 16, (batch, blocks))}
    when = torch.randint(0, 16, (batch, blocks))
    # The original task shares a graph across its two physical sides.
    packed = torch.zeros(blocks, batch, 2, 2, 6, dtype=torch.double)
    packed[:, :, :, 0, :4] = raw["odor"].transpose(0, 1)
    packed[:, :, :, 1, 4:] = raw["ctx"].transpose(0, 1).unsqueeze(2)
    events = torch.stack((raw["ot"], raw["ct"]), -1).transpose(0, 1)
    ticks = events.repeat_interleave(2, dim=1)
    reads = when.T.repeat_interleave(2, dim=1)
    state = cell.initial_state(batch * 2)
    result, end = cell.pulse_blocks(packed.flatten(1, 2), ticks, reads, state, block_ticks=16)
    expected = old.states(raw, when).transpose(0, 1).flatten(1, 2)
    torch.testing.assert_close(result, expected)
    last = old.states(raw, torch.full_like(when, 15))[:, -1].flatten(0, 1)
    torch.testing.assert_close(end, last)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_scan_outputs_state_and_gradients_across_cut(device):
    torch.manual_seed(9)
    cell = LinearGraph(3, 10).to(device=device, dtype=torch.float64)
    x = torch.randn(9, 4, 3, device=device, dtype=torch.float64, requires_grad=True)
    h = torch.randn(4, 10, device=device, dtype=torch.float64, requires_grad=True)
    first, carried = cell(x[:3], h)
    second, end = cell(x[3:], carried)
    scanned = torch.cat((first, second))
    loop, loop_end = cell(x, h, parallel=False)
    torch.testing.assert_close(scanned, loop)
    torch.testing.assert_close(end, loop_end)
    variables = (x, h, *cell.parameters())
    for v, ref in zip(torch.autograd.grad(scanned.square().sum(), variables),
                      torch.autograd.grad(loop.square().sum(), variables), strict=True):
        torch.testing.assert_close(v, ref)


def test_padded_and_simultaneous_inputs_match_tickwise_updates():
    torch.manual_seed(3)
    model = LinearGraph(3, 6).double()
    h = torch.randn(2, 6, dtype=torch.double, requires_grad=True)
    initial = h
    inputs = torch.randn(3, 2, 3, 3, dtype=torch.double)
    tick = torch.tensor([[[1, 1, -1], [3, -1, 2]]] * 3)
    inputs[tick == -1] = float("nan")
    inputs.requires_grad_()
    reads = torch.tensor([[0, 3], [1, 1], [2, 0]])
    sampled, end = model.pulse_blocks(inputs, tick, reads, h, block_ticks=4)
    expected = []
    for i in range(3):
        obs = torch.stack([torch.where(tick[i, :, :, None] == t, inputs[i], 0).sum(1) for t in range(4)])
        seq, h = model(obs, h, parallel=False)
        expected.append(seq[reads[i], torch.arange(2)])
    torch.testing.assert_close(sampled, torch.stack(expected))
    torch.testing.assert_close(end, h)
    variables = (inputs, initial, *model.parameters())
    fast_grad = torch.autograd.grad(sampled.square().sum() + end.square().sum(), variables)
    ref_grad = torch.autograd.grad(torch.stack(expected).square().sum() + h.square().sum(), variables)
    for actual, reference in zip(fast_grad, ref_grad, strict=True):
        torch.testing.assert_close(actual, reference)
    assert not fast_grad[0][tick == -1].any()
    with pytest.raises(ValueError, match="outside"):
        model.pulse_blocks(inputs, tick, reads + 4, h, block_ticks=4)


def test_train_chunk_updates_readout_and_truncates_only_between_chunks():
    torch.manual_seed(11)
    cell = LinearGraph(2, 4)
    head = nn.Linear(4, 1)
    params = [*cell.parameters(), *head.parameters()]
    originals = [p.detach().clone() for p in params]
    opt = torch.optim.Adam(params, lr=.01)
    start = cell.initial_state(2).requires_grad_()
    x = torch.randn(5, 2, 2)
    with torch.no_grad():
        _, expected = cell(x, start)
    loss, carried = train_chunk(cell, opt, x, start, lambda y: head(y).square().mean(), max_grad_norm=.25)
    torch.testing.assert_close(carried, expected)
    assert start.grad is None and not carried.requires_grad and not loss.requires_grad
    assert all(not torch.equal(p, old) for p, old in zip(params, originals, strict=True))
    _, next_state = train_chunk(cell, opt, x, carried, lambda y: head(y).square().mean())
    assert next_state.shape == start.shape


def test_empty_forward_and_invalid_loss_do_not_update_parameters():
    cell = LinearGraph(2, 3, graph="diagonal")
    opt = torch.optim.AdamW(cell.parameters())
    state = cell.initial_state(2)
    empty = torch.empty(0, 2, 2)
    outputs, last = cell(empty, state)
    assert outputs.shape == (0, 2, 3) and last is state
    with pytest.raises(ValueError, match="time step"):
        train_chunk(cell, opt, empty, state, torch.mean)
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        train_chunk(cell, opt, torch.ones(1, 2, 2), state, lambda y: y.mean() * float("nan"))
    assert not opt.state
