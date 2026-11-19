from types import SimpleNamespace

import numpy as np

from experiments.pulse_game.paired_choice import first_contact_pairs


def test_timeout_never_counts_as_a_changed_physical_choice():
    query = np.array([[False, True, True]] * 6)
    related = np.array([[-1, 0, 0]] * 6)
    # A valid reverse, a timeout against the related choice, a valid repeat.
    sides = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0],
                      [0, 1, 0], [0, -1, 0], [0, 1, 0]])
    tape = SimpleNamespace(query=query, related_side=related)
    result = SimpleNamespace(contact_side=sides)
    metrics = first_contact_pairs(tape, result)
    assert metrics["first_side_even"].tolist() == [0, 0, 1]
    assert metrics["first_side_odd"].tolist() == [1, -1, 1]
    assert metrics["paired_first_contact"].tolist() == [1, 0, 1]
    assert metrics["switched"].tolist() == [1, 0, 0]
    assert metrics["switch_given_paired_first_contact"].tolist() == 0.5


def test_conditional_change_is_unavailable_without_any_paired_contacts():
    tape = SimpleNamespace(query=np.ones((2, 1), bool), related_side=np.zeros((2, 1), int))
    metrics = first_contact_pairs(tape, SimpleNamespace(contact_side=np.array([[0], [-1]])))
    assert metrics["switched"].tolist() == [0]
    assert metrics["switch_given_paired_first_contact"].tolist() is None
