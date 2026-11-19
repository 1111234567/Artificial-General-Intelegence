from dataclasses import replace

import numpy as np

from experiments.pulse_game import edit_receipt_eval as ev
from experiments.pulse_game import edit_receipt_reference as reference


def test_swapped_odors_share_evidence_and_recent_three_adapts():
    left, right = (1., 0., .5), (-1., 0., -.5)
    memory_all = reference.TasteHistory(None)
    memory_recent = reference.TasteHistory(3)
    for memory in (memory_all, memory_recent):
        for _ in range(5):
            key, orientation, action = memory.act(left, right, .25)
            memory.receive(key, orientation, action, action)
        assert memory.act(left, right, .95)[2] == 1
        assert memory.act(right, left, .05)[2] == -1
        for _ in range(3):
            key, orientation, action = memory.act(right, left, .25)
            memory.receive(key, orientation, action, action)
    assert memory_all.act(left, right, .95)[2] == 1
    assert memory_recent.act(left, right, .05)[2] == -1


def test_empirical_actions_ignore_private_truth_and_cue_indices():
    settings = ev.Settings(worlds=8, contacts=12, channels=3)
    tape = ev.environment_tapes(settings, 421, "deployment")["reversal"]
    privacy_test = replace(tape, truth=-tape.truth, cue=tape.cue + 100)
    for limit in reference.REFERENCES.values():
        first = reference.empirical_deploy(tape, reference.new_memories(tape, limit))
        altered = reference.empirical_deploy(privacy_test, reference.new_memories(privacy_test, limit))
        np.testing.assert_array_equal(first["side"], altered["side"])
        np.testing.assert_array_equal(first["taste"], altered["taste"])
        np.testing.assert_array_equal(first["correct"], ~altered["correct"])
