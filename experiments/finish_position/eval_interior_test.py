"""Cardinality and scope regression for the unique post-hoc interior test."""
from experiments.finish_position import eval_interior as e


def test_exact_twenty_one_bindings_and_no_position_sweep():
    assert len(e.expected_keys()) == 21
    assert {key[1] for key in e.expected_keys()} == {'interior'}
    assert e.scenarios('B000') == ('B',)
    assert e.scenarios('B008') == ('B', 'mixed', 'history')


def test_references_match_the_same_twenty_one_scenarios():
    paths = e.reference_rows()
    assert len(paths) == 84 and len(set(paths)) == 84
