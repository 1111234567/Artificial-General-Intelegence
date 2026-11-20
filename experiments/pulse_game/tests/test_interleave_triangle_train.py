import numpy as np
import pytest
import torch

from experiments.pulse_game import interleave_triangle_train as triangle
from experiments.pulse_game import local_trace_graph as native


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("rho", triangle.GRAM_LEVELS)
def test_rotated_keys_have_declared_gram_matrix_and_crossed_labels(rho):
    tape = triangle.make_tape(17, 731, 6, rho)
    right = tape["right"].reshape(6, 8, 3, 3).numpy()
    keys = right[:, 0] / 1.35
    gram = keys @ keys.transpose(0, 2, 1)
    expected = np.eye(3) * (1 - rho) + rho
    np.testing.assert_allclose(gram, np.broadcast_to(expected, gram.shape), atol=2e-7)
    if rho == -.5:
        np.testing.assert_allclose(keys.sum(axis=1), 0., atol=2e-7)
    np.testing.assert_array_equal(right, np.repeat(right[:, :1], 8, axis=1))
    np.testing.assert_array_equal(tape["law"].numpy(), np.tile(triangle.LABELS, (6, 1)))


def test_curriculum_pairs_public_draws_and_becomes_identical_at_morph_endpoint():
    assert triangle.rho_for("morph", 1, 4) == 0
    assert triangle.rho_for("morph", 3, 4) == -1 / 3
    for step in (1, 4, 5):
        tapes = [triangle.make_tape(17, 731, 6, triangle.rho_for(arm, step, 4), update=step)
                 for arm in triangle.ARMS]
        for field in tapes[0]:
            matched = torch.equal(tapes[0][field], tapes[1][field])
            assert matched == (step >= 4 or field not in ("left", "right"))


def test_shuffled_objects_reconstructs_its_donor_law_and_probes_share_snapshot(monkeypatch):
    tape = triangle.make_tape(17, 731, 3, -.5)
    calls = []
    original = native.act

    def record(rule, state, left, right, draw):
        calls.append(state)
        return original(rule, state, left, right, draw)

    monkeypatch.setattr(native, "act", record)
    graph, rule = native.spawn(731), native.Rule(731)
    truth = tape["law"].numpy()
    shuffled = triangle.trials(graph, rule, tape, "shuffled_objects")
    assert len(calls) == 9 and all(state is calls[3] for state in calls[3:])
    expected = np.take_along_axis(truth, (np.arange(3) + tape["donor_cycle"].numpy()[:, None]) % 3, axis=1)
    np.testing.assert_array_equal(shuffled["observed"], truth)
    np.testing.assert_array_equal(shuffled["delivered"], expected)
    inverted = triangle.trials(graph, rule, tape, "false_sign")
    np.testing.assert_array_equal(inverted["delivered"], -truth)


def test_margin_gate_identifies_a_pair_only_when_both_frustrated_assignments_pass():
    tape = triangle.make_tape(17, 731, 2, -.5)
    truth = tape["law"].numpy()
    raw = dict(logit=np.stack((truth, -truth), axis=-1), observed=truth, delivered=truth,
               selected=np.repeat(truth[:, :, None], 2, axis=2))
    assert triangle._analyze(tape, raw)["both_uniform_histories_pass"] == 2
    raw["logit"][7, 0, :] *= .02
    assert triangle._analyze(tape, raw)["both_uniform_histories_pass"] == 1
