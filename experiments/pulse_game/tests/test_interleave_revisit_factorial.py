from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.pulse_game import interleave_revisit_factorial as factorial
from experiments.pulse_game import interleave_revisit_train as revisit
from experiments.pulse_game import local_trace_eval as prior
from experiments.pulse_game import local_trace_graph as native


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_each_recency_window_contains_one_once_and_one_twice_object():
    tape = factorial.make_tape(factorial.Config(pairs=32), (731,), 1)
    for roles, sequence in zip(tape.roles[0].tolist(), tape.physical.sequence[0].tolist(), strict=True):
        assert set(sequence[:2]) == {roles[1], roles[3]}
        assert set(sequence[2:4]) == {roles[0], roles[1]}
        assert set(sequence[4:]) == {roles[2], roles[3]}
        assert [sequence.count(item) for item in roles] == [1, 2, 1, 2]
    for roles, query in zip(tape.roles[0].tolist(), tape.physical.query[0].tolist(), strict=True):
        assert query[0] == roles[3]
        assert query[1] in (roles[2], roles[3])
        assert query[2] in (roles[1], roles[3])
        assert (query[3] == roles[2]) or (query[3] == roles[0]) or (query[3] == query[2])
        assert (query[3] in roles[2:]) == (query[2] in roles[2:])
        assert (query[3] in (roles[1], roles[3])) == (query[1] == roles[3])


def test_factorial_queries_train_all_four_arms_from_same_native_support(monkeypatch):
    config = factorial.Config(pairs=8)
    tape = factorial.make_tape(config, (731,), 2)
    graph, rule = native.spawn(731), native.Rule(731)
    calls = []
    original = native.act

    def record(rule, state, left, right, draw):
        calls.append((state.hidden.detach().clone(), left.clone(), draw.clone()))
        return original(rule, state, left, right, draw)

    monkeypatch.setattr(native, "act", record)
    for arm in factorial.ARMS:
        loss, stats = revisit.training_step(graph, rule, factorial._for_arm(tape, arm), "recent", config.max_wait)
        assert torch.isfinite(loss) and torch.isfinite(stats).all()
    size = factorial.SUPPORT + 1
    for block_start in range(size, len(calls), size):
        for t in range(size):
            cols = (0, 2) if t == size - 1 else (0, 1, 2)
            for index in cols:
                torch.testing.assert_close(calls[t][index], calls[block_start+t][index], atol=0, rtol=0)


def test_heldout_crosses_all_laws_and_measures_last_within_late_slots():
    tape = factorial.make_tape(factorial.Config(pairs=32), (731,), 0, full_laws=True)
    np.testing.assert_array_equal(tape.physical.law[0, 0], factorial.LABELS)
    raw = factorial.evaluate_cells(native.spawn(731), native.Rule(731), tape)
    assert not raw["actual_last"][:2].any()
    assert raw["actual_last"][2:].sum(0).min() == raw["actual_last"][2:].sum(0).max() == 1
    stats = factorial._cell_metrics(raw)
    assert set(stats) == {*factorial.CELLS, "all_four"}
    assert "actual_last" in stats["late_once"]


def test_effects_are_paired_main_factors_and_difference_in_differences():
    intervals = prior.BirthIntervals((731, 732), 30, 9)
    data = (.1, .2, .3, .6)
    rates = {arm: intervals.report(np.array([value, value + .1]))
             for arm, value in zip(factorial.ARMS, data, strict=True)}
    estimate = factorial._factors(rates, intervals)
    assert estimate["allow_early_query"]["mean"] == pytest.approx(.3)
    assert estimate["allow_once_query"]["mean"] == pytest.approx(.2)
    assert estimate["interaction"]["mean"] == pytest.approx(.2)
