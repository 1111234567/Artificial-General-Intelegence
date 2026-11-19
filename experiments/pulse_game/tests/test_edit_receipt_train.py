from dataclasses import replace

import numpy as np
import torch

from experiments.pulse_game import edit_receipt_train as train


def differentiable_births(births, writers):
    graphs = [train.fw.spawn(birth, writers) for birth in births]
    return train.fw.Graph(torch.cat([graph.weights for graph in graphs]).requires_grad_(),
                          torch.cat([graph.bias for graph in graphs]).requires_grad_(),
                          torch.cat([graph.present for graph in graphs]))


def test_targets_shuffle_only_eligible_values_with_no_self_donors():
    target = torch.arange(5, dtype=torch.float32)
    eligible = torch.tensor([True, False, True, True, False])
    order = torch.tensor([2, 4, 0, 1, 3])
    swapped, used = train.paired_targets(target, eligible, order, shuffled=True)
    assert torch.equal(eligible, used)
    assert torch.equal(swapped[~eligible], target[~eligible])
    assert torch.equal(swapped[eligible].sort().values, target[eligible])
    assert torch.all(swapped[eligible] != target[eligible])
    _, singleton = train.paired_targets(target, eligible & (target == 2), order, shuffled=False)
    assert not singleton.any()


def test_clean_private_answer_changes_diagnostics_but_has_no_training_gradient():
    config = train.Config(batch=8, contacts=6, horizon=2)
    graph = differentiable_births((421,), config.writers)
    tape, focal, shuffle = train.world_for_update(config, 421, 39)
    first_loss, first_metrics = train.training_step(graph, tape, focal, shuffle, "causal", config)
    counterfactual_answer = replace(tape, truth=-tape.truth)
    second_loss, second_metrics = train.training_step(graph, counterfactual_answer, focal, shuffle, "causal", config)
    torch.testing.assert_close(first_loss, second_loss, rtol=0, atol=0)
    first = torch.autograd.grad(first_loss, (graph.weights, graph.bias))
    second = torch.autograd.grad(second_loss, (graph.weights, graph.bias))
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    physical = train.METRICS.index("physical_accuracy")
    earned = train.METRICS.index("fork_taste_mean")
    np.testing.assert_allclose(first_metrics[:, physical] + second_metrics[:, physical], 1)
    np.testing.assert_array_equal(first_metrics[:, earned], second_metrics[:, earned])


def test_grouped_batches_have_independent_birth_gradients():
    config = train.Config(batch=6, contacts=5, horizon=2)
    births = (421, 423)
    worlds, foci, shuffles = zip(*(train.world_for_update(config, birth, 7)
                                  for birth in births), strict=True)
    assert foci[0] == foci[1]
    grouped = differentiable_births(births, config.writers)
    total, grouped_metrics = train.training_step(grouped, train.Tape.concatenate(worlds), foci[0],
                                                 torch.cat(shuffles), "shuffled", config)
    grouped_grads = torch.autograd.grad(total, (grouped.weights, grouped.bias))
    for lane, (birth, world, shuffle) in enumerate(zip(births, worlds, shuffles, strict=True)):
        graph = differentiable_births((birth,), config.writers)
        loss, metrics = train.training_step(graph, world, foci[0], shuffle, "shuffled", config)
        grads = torch.autograd.grad(loss, (graph.weights, graph.bias))
        np.testing.assert_allclose(grouped_metrics[lane], metrics[0], atol=2e-7, rtol=2e-6)
        torch.testing.assert_close(tuple(grad[lane] for grad in grouped_grads),
                                   tuple(grad[0] for grad in grads), atol=2e-7, rtol=2e-5)
