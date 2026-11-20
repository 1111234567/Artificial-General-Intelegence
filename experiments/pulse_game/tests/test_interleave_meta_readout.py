import numpy as np
import torch

from experiments.pulse_game import interleave_meta_evidence as task, interleave_meta_readout as readout
from experiments.pulse_game import local_trace_graph as net


def test_observer_preserves_replay_and_matches_three_real_x_receipts():
    torch.set_num_threads(1)
    g, r = net.spawn(731), net.Rule(731)
    worlds, n = 4, 8
    tape = readout._bank(worlds, n, "test")
    for scorer, replay_mode in (("yoked", "same_new_contacts"), ("natural", "natural")):
        observed = readout.observe(g, r, tape, n, scorer)
        with torch.no_grad():
            _, reward, act, hist = task.replay(g, r, tape, n, task.Config(worlds=worlds), mode=replay_mode, return_history=True)
        assert torch.equal(observed.side, act.side)
        assert torch.equal(observed.reward, reward)
        assert torch.equal(observed.history, hist)
        assert set(observed.snapshots) == set(readout.STAGES)


def test_calibration_tape_and_whole_world_null_keep_regime_and_sign_lanes():
    torch.set_num_threads(1)
    g, r = net.spawn(731), net.Rule(731)
    train = readout._bank(4, 8, "cal")
    test = readout._bank(6, 8, "test")
    assert not torch.equal(train["right"], test["right"][:len(train["right"])])
    labels = readout._targets(train)
    null = readout._shuffled_y(labels, 8).reshape(4, 8, -1)
    signs = null[:, :, 1:] * labels[..., None]
    assert np.all(signs == signs[:, :1])
    for n, branch in ((0, "no_prefix"), (8, "blank_prefix")):
        a, z = (readout._bank(count, n, split) for count, split in ((4, "cal"), (6, "test")))
        if n == 0:
            for key in ("right", "factor", "u", "gap", "canonical", "orient"):
                assert torch.equal(z[key], test[key][:, -4:])
        ca, te = (readout.observe(g, r, item, n, branch) for item in (a, z))
        for phase in readout.STAGES:
            scores, _ = readout._readout(ca, te, readout._shuffled_y(readout._targets(a), n), readout._targets(z),
                                        branch, "combined", phase)
            np.testing.assert_array_equal(scores["accuracy"], .5)


def test_ridge_standardizes_using_only_the_calibration_bank():
    torch.set_num_threads(1)
    labels = np.array([[-1.], [1.]])
    x = np.array([[-1.], [1.]])
    s = readout.ridge_scores(x, np.array([[10.], [14.]]), labels)
    assert s[0, 0] > 0 and s[1, 0] > s[0, 0]
