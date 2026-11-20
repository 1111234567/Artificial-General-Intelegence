from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.pulse_game import interleave_revisit_train as revisit
from experiments.pulse_game import local_trace_graph as native
from experiments.pulse_game import local_trace_train as old_fit


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_curriculum_matches_object_history_and_balances_private_laws():
    config = revisit.Config(pairs=128, two_object_updates=4)
    for step, possible_counts in ((4, {2}), (5, {2, 3, 4})):
        tape = revisit.make_tape(config, (731,), step)
        assert torch.equal(tape.law[:, :, 0], -tape.law[:, :, 1])
        assert torch.equal(tape.query[:, :, 0], tape.sequence[:, :, -1])
        observed = set()
        for sequence, (_, uniform) in zip(tape.sequence[0].tolist(), tape.query[0].tolist(), strict=True):
            count = len(set(sequence))
            observed.add(count)
            assert len(set(sequence[:count])) == count < len(sequence)
            assert uniform in sequence[:count]
        assert observed == possible_counts


def test_terminal_query_changes_menu_after_identical_physically_earned_tastes(monkeypatch):
    config = revisit.Config(pairs=8, two_object_updates=4)
    tape = revisit.make_tape(config, (731,), 1)
    warm = native.spawn(731)
    initial = native.Graph(warm.weights.requires_grad_(True), warm.bias.requires_grad_(True), warm.present)
    rule = native.Rule(731)
    calls = []
    original = native.act

    def record(rule, state, left, right, draw):
        action = original(rule, state, left, right, draw)
        calls.append((left.clone(), draw.clone(), action.side.clone(), state.hidden.detach().clone(),
                      state.graph.weights.detach().clone(), state.sender_trace.detach().clone()))
        return action

    monkeypatch.setattr(native, "act", record)
    results = [revisit.training_step(initial, rule, tape, arm, config.max_wait) for arm in revisit.ARMS]
    n_actions = tape.sequence.shape[-1] + 1
    for first, second in zip(calls[:n_actions - 1], calls[n_actions:-1], strict=True):
        for x, y in zip(first, second, strict=True):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
    recent, uniform = calls[n_actions - 1], calls[-1]
    assert not torch.equal(recent[0], uniform[0])
    for x, y in ((recent[i], uniform[i]) for i in (1, 3, 4, 5)):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    for loss, values in results:
        assert torch.isfinite(loss) and torch.isfinite(values).all()
    assert results[0][1][0, 8].item() == 0
    assert results[1][1][0, 8].item() > 0


def test_gradient_norm_stays_finite_on_large_float32_values_and_clips_per_birth():
    parameter = torch.nn.Parameter(torch.zeros((2, 2), dtype=torch.float32))
    parameter.grad = torch.tensor([[3e37, 4e37], [0.3, 0.4]])
    norms = revisit._gradients((parameter,), 2.0)
    torch.testing.assert_close(norms, torch.tensor([5e37, .5], dtype=torch.float64))
    torch.testing.assert_close(parameter.grad, torch.tensor([[1.2, 1.6], [.3, .4]]))


def test_optimizer_resume_matches_uninterrupted_fit(tmp_path):
    warm = tmp_path / "warm"
    birth = 731
    graph = native.spawn(birth)
    old_fit.save_checkpoint(warm, "odor", "active", birth, 8000, graph, native.Rule(birth), 0)
    config = revisit.Config(updates=4, pairs=2, two_object_updates=2, autosave=1)
    uninterrupted, resumed = tmp_path / "one", tmp_path / "two"
    def fit(path, stop):
        revisit._fit_group(path, warm, config, (birth,), stop, "test-manifest", 0, 1)
    fit(uninterrupted, 4)
    fit(resumed, 2)
    fit(resumed, 4)
    for arm in revisit.ARMS:
        a = revisit._checkpoint(uninterrupted, arm, birth, 4)
        b = revisit._checkpoint(resumed, arm, birth, 4)
        with np.load(a) as left, np.load(b) as right:
            assert left.files == right.files
            for key in left.files:
                np.testing.assert_array_equal(left[key], right[key])


def test_review_probes_use_same_targets_and_raw_ab_matches_capacity():
    graph, rule = native.spawn(731), native.Rule(731)
    tape = revisit.capacity.make_tape("odor", 7, 12, 2)
    baseline = revisit.capacity.evaluate(graph, rule, tape, (0, 1, 2))
    reviews = revisit.evaluate_reviews(graph, rule, tape)
    np.testing.assert_array_equal(reviews["side"][0], baseline["side"][0, 1, :2])
    np.testing.assert_array_equal(reviews["truth"], tape.right_value[:, :2, -1].numpy().T)
    np.testing.assert_array_equal(reviews["contact"][2], reviews["contact"][3])
