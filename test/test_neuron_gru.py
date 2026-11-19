"""Compatibility with the frozen GRU model, trainer and dense checkpoints."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from experiments.learn_remember import gru_train as historical
from experiments.learn_remember import symmetric_gru as historical_shared
from src.neurons.gru import (GRUAgent, attach_frozen_encoder_cache, deploy_dense_gru,
                             fit_gru_sequence, share_recurrent_weights)


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    count = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(count)


def test_plain_gru_uses_historical_initialization_readout_and_state():
    dimensions = dict(width=7, input_size=5, n_jobs=3, n_actions=4)
    with torch.random.fork_rng():
        torch.manual_seed(19)
        old = historical.GRUAgent(**dimensions)
        torch.manual_seed(19)
        model = GRUAgent(**dimensions)
    assert type(model.core) is nn.GRU
    assert model.config() == old.config()
    assert model.parameter_report() == old.parameter_report()
    torch.testing.assert_close(model.state_dict(), old.state_dict(), rtol=0, atol=0)

    obs = torch.randn(7, 2, dimensions["input_size"])
    anchor = model.initial_state(2)
    logits, hidden = model(obs, anchor)
    torch.testing.assert_close((logits, hidden), old(obs, anchor), rtol=0, atol=0)
    torch.testing.assert_close(model.step(obs[0], anchor, dt=0.25),
                               old.step(obs[0], anchor, dt=0.25), rtol=0, atol=0)

    first, carried = model(obs[:3], anchor)
    last, final = model(obs[3:], carried)
    torch.testing.assert_close(torch.cat((first, last)), logits)
    torch.testing.assert_close(final, hidden)


def test_shared_weights_pack_both_directions_and_export_dense():
    width = 6
    model = share_recurrent_weights(GRUAgent(width=width, input_size=5)).double()
    historical_model = historical_shared.constrain(historical.GRUAgent(**model.config())).double()
    historical_model.load_state_dict(model.state_dict())
    model.load_state_dict(historical_model.state_dict())
    rng_before = torch.get_rng_state()
    public = deploy_dense_gru(model)
    assert torch.equal(rng_before, torch.get_rng_state())
    torch.testing.assert_close(public.state_dict(), historical_shared.deploy(historical_model).state_dict(),
                               atol=0, rtol=0)
    assert model.parameter_report()["unique_hidden_coordinates"] == 3 * width * (width + 1) // 2
    assert public.parameter_report()["total_parameters"] - model.parameter_report()["total_parameters"] == (
        3 * width * (width - 1) // 2)
    weights = model.core.weight_hh_l0.reshape(3, width, width)
    assert torch.equal(weights, weights.transpose(-1, -2))

    obs = torch.randn(7, 2, 5, dtype=torch.float64)
    anchor = torch.randn(2, width, dtype=torch.float64)
    actual = model(obs, anchor)
    expected = public(obs, anchor)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    actual[0].square().sum().backward()
    expected[0].square().sum().backward()
    packing = model.core.parametrizations.weight_hh_l0
    rule = packing[0]
    grad = public.core.weight_hh_l0.grad.reshape(3, width, width)
    expected_grad = grad[:, rule.i, rule.j] + grad[:, rule.j, rule.i] * (rule.i != rule.j)
    torch.testing.assert_close(packing.original.grad, expected_grad, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("shared", [False, True])
def test_fit_matches_historical_adam_updates(shared):
    width, block, epochs = 7, 4, 3
    legacy = historical.Trainer.__new__(historical.Trainer)
    legacy.cfg = historical.Config(width=width, epochs=epochs, block=block, device="cpu")
    legacy.model = historical.GRUAgent(width=width, input_size=6)
    model = GRUAgent(**legacy.model.config())
    if shared:
        historical_shared.constrain(legacy.model)
        share_recurrent_weights(model)
    model.load_state_dict(deepcopy(legacy.model.state_dict()))
    legacy.optimizer = torch.optim.Adam(legacy.model.parameters(), lr=legacy.cfg.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=legacy.cfg.lr)
    legacy.work = dict(gradient_updates=0, gradient_lane_ticks=0)

    batch, ticks = 3, 8
    data = dict(obs=torch.randn(ticks, batch, 6), labels=torch.randint(11, (ticks, batch, 2)),
                anchor=torch.randn(batch, width))
    old_metrics = legacy.fit(data)
    weights = torch.ones(model.n_actions)
    weights[0] = legacy.cfg.wait_weight
    new_metrics = fit_gru_sequence(model, optimizer, data["obs"], data["labels"], data["anchor"],
                                   block_size=block, epochs=epochs, class_weights=weights)
    hidden = new_metrics.pop("hidden")
    assert hidden.shape == data["anchor"].shape and not hidden.requires_grad
    assert new_metrics == {**old_metrics, **legacy.work}
    torch.testing.assert_close(model.state_dict(), legacy.model.state_dict(), atol=0, rtol=0)
    torch.testing.assert_close(optimizer.state_dict(), legacy.optimizer.state_dict(), atol=0, rtol=0)


def test_fit_uses_short_final_block_and_detaches_anchor():
    model = GRUAgent(width=7, input_size=5, n_jobs=1, n_actions=3)
    optimizer = torch.optim.Adam(model.parameters())
    parameter = model.head.weight.detach().clone()
    anchor = model.initial_state(2).requires_grad_()
    obs, targets = torch.randn(5, 2, 5), torch.randint(3, (5, 2, 1))
    fit = fit_gru_sequence(model, optimizer, obs, targets, anchor, block_size=2, epochs=2)
    assert fit["gradient_updates"] == 6
    assert fit["gradient_lane_ticks"] == 20
    assert anchor.grad is None and not fit["hidden"].requires_grad
    assert not torch.equal(parameter, model.head.weight)


def test_scoped_frozen_encoder_cache_preserves_native_outputs():
    model = nn.Module()
    model.encoder = nn.GRU(3, 4).requires_grad_(False)
    obs = torch.randn(5, 2, 3)
    expected = model.encoder(obs)
    with attach_frozen_encoder_cache(model) as cache:
        torch.testing.assert_close(model.encoder(obs), expected, atol=0, rtol=0)
        torch.testing.assert_close(model.encoder(obs), expected, atol=0, rtol=0)
        assert (cache.stats["hits"], cache.stats["misses"]) == (1, 1)
    assert "forward" not in vars(model.encoder)
def test_gru_without_task_head_can_train_and_export():
    from src.neurons.gru import GRU, deploy_dense_gru, share_recurrent_weights
    from src.neurons.training import train_chunk

    torch.manual_seed(53)
    core = share_recurrent_weights(GRU(input_size=3, width=5))
    x = torch.randn(4, 2, 3)
    h = core.initial_state(2)
    opt = torch.optim.Adam(core.parameters(), lr=.01)
    before = core.core.weight_hh_l0.detach().clone()
    _, carried = train_chunk(core, opt, x, h, lambda values: values.square().mean())
    assert not torch.equal(before, core.core.weight_hh_l0)
    expected, end = core(x, carried)
    actual, end_dense = deploy_dense_gru(core)(x, carried)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(end_dense, end, atol=0, rtol=0)
