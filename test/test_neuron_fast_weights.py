import pytest
import torch
from torch.nn import functional as F

from experiments.pulse_game import habitat_meta, legacy_grounded_outcomes, ring_metric
from src.neurons.fast_weights import (
    delta_prefix_predictions,
    delta_step,
    initial_rls_state,
    read_fast_weights,
    rls_prefix_predictions,
    rls_update,
)


def test_rls_prefix_matches_online_writes_and_gradients_with_hidden_receipts():
    torch.manual_seed(43)
    mask = torch.tensor([[True, False, True, True, False], [False, True, True, False, True]])
    keys = torch.randn(2, 5, 3, dtype=torch.double).masked_fill(~mask[..., None], float('nan')).requires_grad_()
    queries = torch.randn(2, 5, 3, dtype=torch.double, requires_grad=True)
    targets = torch.randn(2, 5, dtype=torch.double).masked_fill(~mask, float('nan')).requires_grad_()
    ridge = torch.tensor(.4, dtype=torch.double, requires_grad=True)
    batched = rls_prefix_predictions(keys, targets, ridge, queries=queries, mask=mask)
    state = initial_rls_state(keys[:, 0], ridge)
    online = []
    for key, target, query, visible in zip(keys.unbind(1), targets.unbind(1), queries.unbind(1), mask.unbind(1)):
        online.append(read_fast_weights(state.weights, query))
        previous = state
        state = rls_update(state, key, target, mask=visible)
        for current, initial in zip(state, previous):
            torch.testing.assert_close(current[~visible], initial[~visible], rtol=0, atol=0)
    online = torch.stack(online, dim=1)
    torch.testing.assert_close(batched, online, rtol=1e-12, atol=1e-12)
    changed = targets.detach().clone()
    changed[:, 2:] = 100.
    counterfactual = rls_prefix_predictions(keys, changed, ridge, queries=queries, mask=mask)
    torch.testing.assert_close(batched[:, :3], counterfactual[:, :3], rtol=0, atol=0)
    variables = keys, queries, targets, ridge
    actual = torch.autograd.grad(batched.square().sum(), variables)
    expected = torch.autograd.grad(online.square().sum(), variables)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)
    assert actual[0][~mask].count_nonzero() == actual[2][~mask].count_nonzero() == 0


def test_rls_matches_habitat_and_legacy_action_specific_prefixes():
    torch.manual_seed(44)
    model = habitat_meta.MetaSynapses(habitat_meta.Config(nodes=3, width=5, slots=2, updater='rls')).double()
    state = torch.randn(2, 6, 3, dtype=torch.double)
    context = torch.randn(2, 6, 2, dtype=torch.double)
    query = torch.randn_like(state)
    at, now = torch.randn_like(state[..., 0]), torch.randn_like(state[..., 0])
    targets = state[..., 0].sign()
    ridge = .015 + model.log_ridge.exp()
    prediction = rls_prefix_predictions(model.code(state, context) * at.unsqueeze(-1), targets, ridge,
                                        queries=model.code(query, context))
    expected = model(state, context, query, at, now, targets)
    gain = .4 + 4. * model.scale.sigmoid()
    torch.testing.assert_close((gain * prediction).tanh() * now, expected, rtol=1e-12, atol=1e-12)

    attempts = torch.tensor([[0, 1, 0, 3, 1, 2], [2, 0, 1, 2, 2, 1]])
    slots = F.one_hot(attempts, 4).unsqueeze(-1)
    action_keys = (state.unsqueeze(-2) * slots).flatten(-2)
    expected = legacy_grounded_outcomes.causal_predictions(state, attempts, targets, ridge)
    torch.testing.assert_close(rls_prefix_predictions(action_keys, targets, ridge), expected, rtol=1e-12, atol=1e-12)


def test_delta_prefix_matches_steps_gradients_and_chunked_state():
    torch.manual_seed(45)
    keys = F.normalize(torch.randn(2, 6, 4, dtype=torch.double), dim=-1).requires_grad_()
    weights = torch.randn_like(keys[:, 0]).requires_grad_()
    mask = torch.tensor([[True, False, True, False, False, True], [False, False, True, True, True, True]])
    targets = torch.randn(2, 6, dtype=torch.double).masked_fill(~mask, float('nan')).requires_grad_()
    rate = torch.tensor(.47, dtype=torch.double, requires_grad=True)
    prior = torch.tensor(-.31, dtype=torch.double, requires_grad=True)
    kwargs = dict(rate=rate, prior=prior)
    predictions, end = delta_prefix_predictions(keys, targets, mask=mask, weights=weights, **kwargs)
    current, online = weights, []
    for key, target, write in zip(keys.unbind(1), targets.unbind(1), mask.unbind(1)):
        prediction, current = delta_step(current, key, target, mask=write, **kwargs)
        online.append(prediction)
    online = torch.stack(online, dim=1)
    torch.testing.assert_close(predictions, online, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(end, current, rtol=1e-12, atol=1e-12)

    first, carry = delta_prefix_predictions(keys[:, :3], targets[:, :3], mask=mask[:, :3], weights=weights, **kwargs)
    last, continued = delta_prefix_predictions(keys[:, 3:], targets[:, 3:], mask=mask[:, 3:], weights=carry, **kwargs)
    torch.testing.assert_close(torch.cat((first, last), dim=1), predictions)
    torch.testing.assert_close(continued, end)

    variables = keys, targets, weights, rate, prior
    actual = torch.autograd.grad(predictions.square().sum() + end.square().sum(), variables)
    expected = torch.autograd.grad(online.square().sum() + current.square().sum(), variables)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)
    assert actual[1][~mask].count_nonzero() == 0


def test_delta_matches_ring_strictly_lower_legacy_solve():
    torch.manual_seed(46)
    config = ring_metric.Settings(neurons=6, hidden=8, latent=4)
    model = ring_metric.Metric(config).double()
    inputs = torch.randn(2, 8, config.inputs, dtype=torch.double)
    targets = torch.randn(2, 8, dtype=torch.double).sign()
    mask = torch.rand_like(targets) > .25
    keys = model.keys(inputs)
    pred, _ = delta_prefix_predictions(keys, targets, mask=mask, rate=1.8*model.rate.sigmoid(), prior=model.prior)
    torch.testing.assert_close(pred, model.predict_events(inputs, targets, mask), rtol=1e-12, atol=1e-12)


def test_empty_sequences_and_zero_valued_public_targets():
    keys = torch.empty(1, 0, 3)
    targets = torch.empty(1, 0)
    start = torch.randn(1, 3)
    assert rls_prefix_predictions(keys, targets, .3).shape == targets.shape
    outputs, end = delta_prefix_predictions(keys, targets, rate=.6, weights=start)
    assert outputs.shape == targets.shape and end is start
    key = torch.ones(1, 3)
    initial = initial_rls_state(key, .3)
    successor = rls_update(initial, key, torch.zeros(1))
    assert torch.equal(successor.weights, initial.weights)
    assert not torch.equal(successor.inverse_covariance, initial.inverse_covariance)
    with pytest.raises(ValueError, match='positive finite scalar'):
        initial_rls_state(key, 0.)
