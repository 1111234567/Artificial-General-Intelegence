import numpy as np
import torch

from experiments.pulse_game import self_write_core as core
from experiments.pulse_game.self_write_repeat_bootstrap import differentiable_write


def test_training_write_matches_native_edit_through_multiple_rounds():
    rng = np.random.default_rng(704)
    graph = core.spawn_graph(101, 32)
    start_w = torch.tensor(graph.w, requires_grad=True)
    start_b = torch.tensor(graph.b, requires_grad=True)
    w, b = start_w, start_b
    pre = torch.tensor(rng.normal(size=graph.b.shape), dtype=torch.float32)
    post = torch.tensor(rng.normal(size=graph.b.shape), dtype=torch.float32)
    accepted_structure = 0
    for key in (70, 71, 72):
        proposal = core.proposal(graph, pre.numpy(), post.numpy(), key, view="own")
        expected, receipt = core.apply(graph, proposal)
        w, b, mask = differentiable_write(w, b, graph.mask, pre, post, key)
        np.testing.assert_allclose(w.detach().numpy(), expected.w, atol=1e-7, rtol=1e-6)
        np.testing.assert_allclose(b.detach().numpy(), expected.b, atol=1e-7, rtol=1e-6)
        np.testing.assert_array_equal(mask, expected.mask)
        accepted_structure += (receipt.birth_status == 1).sum() + (receipt.retire_status == 1).sum()
        graph = expected
    assert accepted_structure > 0
    (w.square().mean() + b.square().mean()).backward()
    assert torch.count_nonzero(start_w.grad) > 0
    assert torch.count_nonzero(start_b.grad) > 0
