import numpy as np

from experiments.content_timing.probe import evaluate


def test_heldout_phase_and_anchor_accounting():
    anchors = [
        dict(identity=(lane, job, family), lane=lane, job=job, family=family)
        for lane in range(4, 8) for job in range(2) for family in range(4)
    ]
    states = np.zeros((len(anchors), 1, 4, 3, 24))
    # The signal moves to other features at mid phase and disappears at first legal.
    states[:, 0, :, 0, :4] = np.eye(4)
    states[:, 0, :, 1, 4:8] = np.eye(4)
    folds, summary = evaluate(anchors, states, [8.0], range(4))
    result = {(r["train_phase"], r["test_phase"]):r for r in summary if r["label"] == "raw"}
    assert result["cue", "cue"]["accuracy"] == result["mid", "mid"]["accuracy"] == 1
    assert result["cue", "mid"]["accuracy"] == result["first_legal", "first_legal"]["accuracy"] == .25
    assert all(r["total"] == len(anchors)*4 and r["null_correct"] == len(anchors) for r in summary)
    assert all(r["train_anchors"] == r["test_anchors"] == 8 for r in folds)
