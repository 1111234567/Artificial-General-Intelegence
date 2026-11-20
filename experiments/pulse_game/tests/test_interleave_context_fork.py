import numpy as np
import torch

from experiments.pulse_game import interleave_context_fork as fork, interleave_train as train


def test_cue_only_queries_share_odors_orientation_draw_and_have_independent_targets():
    config = train.Config(pairs=16, max_wait=0)
    tape = train.make_tape((0,), config, 501, domain=fork.DOMAIN)
    cutoff, cues, odor, draw, target, newer_is_b = fork._cue_queries(tape)
    assert fork.CONTACT_TICKS == 17
    assert torch.all(torch.any(cues[:, 0] != cues[:, 1], dim=1))
    assert torch.allclose(odor.norm(dim=1), torch.full_like(draw, 1.35))
    assert torch.equal(cutoff[::2], cutoff[1::2])
    assert torch.equal(draw[::2], draw[1::2])
    assert torch.equal(target[::2], -target[1::2])
    assert torch.equal(newer_is_b[::2], newer_is_b[1::2])
    assert set((target[:, 0] == target[:, 1]).tolist()) == {True, False}
    graph = train.local.spawn(731)
    rule = train.local.Rule(731)
    side, logit, returned_target, returned_newer = fork.replay(graph, rule, tape, 0)
    assert side.shape == logit.shape == (len(fork.BRANCHES), 16, 2, 2)
    assert np.isfinite(logit).all()
    assert np.array_equal(returned_target.reshape(-1, 2), target.numpy())
    assert np.array_equal(returned_newer.reshape(-1), newer_is_b.numpy())
    measured = fork._measure(side, logit, returned_target[None], returned_newer[None], agreeing=False)
    included = returned_target[:, 0, 0] != returned_target[:, 0, 1]
    assert np.array_equal(measured[:, included, -1], 2 * measured[:, included, 2] - measured[:, included, 3])
