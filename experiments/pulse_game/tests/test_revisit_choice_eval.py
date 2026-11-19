from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from experiments.pulse_game import revisit_choice_eval as ev
from experiments.pulse_game import revisit_choice_train as training
from experiments.pulse_game import self_param_writer as fw


@pytest.fixture(autouse=True)
def single_thread_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_holdout_crosses_first_side_and_reorder_with_matching_draws():
    tape = ev.make_holdout(70, 731, 12)
    assert tape.lanes == 48
    assert tape.first_side.reshape(-1, 4).unique(dim=0).tolist() == [[-1, -1, 1, 1]]
    assert tape.swap.reshape(-1, 4).unique(dim=0).tolist() == [[False, True, False, True]]
    assert torch.equal(tape.second_left("same")[::2], -tape.second_left("same")[1::2])
    assert torch.equal(tape.second_left("novel")[::2], -tape.second_left("novel")[1::2])
    assert ((tape.first_left * tape.novel_left).sum(-1).abs() <= .75 * 1.35 ** 2).all()
    for draw in (tape.first_draw, tape.second_draw, tape.gate_draw, tape.proposal_draws, tape.donor_taste):
        for lane in range(1, 4):
            assert torch.equal(draw[::4], draw[lane::4])
    assert not torch.equal(tape.first_left, ev.make_holdout(70, 732, 12).first_left)


def test_donor_labels_do_not_reach_rollout_and_selected_forks_match_actual_actions():
    tape = ev.make_holdout(111, 731, 24)
    graph = fw.spawn(731, 4)
    raw = ev.evaluate(graph, tape, [0, 2])
    donor_flip = ev.evaluate(graph, replace(tape, donor_taste=-tape.donor_taste), [0, 2])
    for name in raw:
        if name != "donor_taste":
            np.testing.assert_array_equal(raw[name], donor_flip[name])

    actual = raw["motor_side"]
    np.testing.assert_array_equal(raw["first_actual_side"], raw["first_side"])
    np.testing.assert_array_equal(raw["first_natural_side"], np.where(raw["first_draw"] < ev.logistic(raw["first_logit"]), 1, -1))
    np.testing.assert_array_equal(actual, np.where(raw["second_draw"] < ev.logistic(raw["motor_logit"]), 1, -1))
    for stream, gates in enumerate(raw["gate_index"]):
        for lane, gate in enumerate(gates):
            np.testing.assert_array_equal(actual[stream, :, :, 0, lane], actual[stream, :, :, ev.FORCED[gate], lane])
            assert raw["receipt_status"][stream, 0, lane] == raw["receipt_status"][stream, ev.FORCED[gate], lane]
    disabled = [ev.BRANCHES.index(name) for name in ("disabled", "receipt_matched_disabled",
                                                   "receipt_matched_disabled_reset_h", "skip")]
    assert not (raw["receipt_status"][:, disabled] == fw.Status.APPLIED).any()


def test_natural_first_slice_replays_two_unforced_motor_actions():
    tape = ev.make_holdout(388, 731, 15)
    graph = fw.spawn(731, 4)
    raw = ev.evaluate(graph, tape, [0, 2])
    selected = raw["first_actual_side"] == raw["first_natural_side"]
    assert selected.sum() == tape.lanes // 2
    state = ev.native.start(graph, tape.lanes)
    first = ev.native.act(state, tape.first_left, -tape.first_left, tape.first_draw)
    experience = ev.native.taste(first, tape.first_left, -tape.first_left, 1)
    proposals = ev.native.propose(experience, tape.proposal_draws)
    decision = ev.native.choose(proposals, tape.gate_draw)
    result = ev.native.commit(experience, proposals, decision)
    state = ev.native.wait(result.state, 2)
    left = tape.second_left("same")
    acted = ev.native.act(state, left, -left, tape.second_draw)
    np.testing.assert_array_equal(raw["motor_side"][2, 0, 1, 0, selected], acted.side.numpy()[selected])


def test_recurrent_reset_keeps_edited_parameters_and_occurs_after_receipt():
    tape = ev.make_holdout(110, 731, 36)
    graph = fw.spawn(731, 4)
    start = ev.native.start(graph, tape.lanes)
    first = ev.native.observe(start, tape.first_left, -tape.first_left, tape.first_side, tape.first_draw,
                              side_override=tape.first_side)
    proposals = ev.native.propose(first, tape.proposal_draws)
    decision = ev.native.choose(proposals, tape.gate_draw, override=0)
    commit = ev.native.commit(first, proposals, decision)
    assert (commit.receipt.status == fw.Status.APPLIED).any()
    assert not torch.equal(commit.state.graph.weights, start.graph.weights)
    assert torch.count_nonzero(commit.state.hidden)
    reset = ev.clear_hidden(commit)
    assert reset.state.graph is commit.state.graph
    assert reset.receipt is commit.receipt
    assert not torch.count_nonzero(reset.state.hidden)


def test_repeat_switch_are_odor_relative_and_noise_controls_score_chance():
    tape = ev.make_holdout(302, 731, 4)
    raw = {name: tensor.numpy() for name, tensor in vars(tape).items()}
    raw["first_actual_side"] = tape.first_side.numpy().astype(np.int8)
    raw["first_natural_side"] = np.ones(tape.lanes, dtype=np.int8)
    shape = (3, 2, 1, len(ev.BRANCHES), tape.lanes)
    side = np.ones(shape, dtype=np.int8)
    repeat = raw["first_actual_side"] * (1 - 2 * raw["swap"])
    side[0, 0] = -repeat
    side[2, 0] = repeat
    raw.update(motor_side=side, motor_logit=20.0 * side,
               receipt_status=np.zeros((3, len(ev.BRANCHES), tape.lanes)),
               gate_index=np.zeros((3, tape.lanes), dtype=int),
               gate_logits=np.zeros((3, tape.lanes, 3)), benefit=np.zeros((3, tape.lanes, 2)))
    bootstrap = ev.BirthBootstrap([731, 732], 30, 100)
    behavior = ev.summarize_group([raw, raw], bootstrap, 0, 0)["behavior"]
    assert behavior["true"]["natural"]["actual"]["mean"] == 1
    assert behavior["true"]["natural_repeat"]["motor"]["mean"] == .5
    assert behavior["flipped"]["natural"]["actual"]["mean"] == 0
    for control in ("zero", "yoked"):
        assert behavior[control]["natural"]["actual"]["mean"] == .5
    for cell in behavior["true"]["natural_side_swap"].values():
        assert cell["positive_repeat"]["mean"] == cell["negative_switch"]["mean"] == 1
    novel = ev.summarize_group([raw, raw], bootstrap, 1, 0)["behavior"]
    assert novel["true"]["natural"]["actual"]["mean"] == .5


def test_checkpoints_share_birth_tapes_and_emit_replayable_traces(tmp_path, monkeypatch):
    births = [731, 732]
    arms = ["ranked", "shuffled"]
    manifest = {"births": births, "arms": arms, "checkpoints": [0, 2], "config": {"keep_connected": True}}
    (tmp_path / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    loaded = []

    def checkpoint(path, arm, birth, step):
        loaded.append((path, arm, birth, step))
        return fw.spawn(birth, 4)

    monkeypatch.setattr(training, "load_checkpoint", checkpoint)
    output, trials = tmp_path / "results.json", tmp_path / "trials.npz"
    ev.run(tmp_path, output, trials, pairs=3, delays=[0, 1], steps=["last"], bootstrap_draws=10)
    assert {(arm, birth, step) for _, arm, birth, step in loaded} == {
        (arm, birth, 2) for arm in arms for birth in births}
    report = json.loads(output.read_text(encoding="utf-8"))
    comparison = report["paired_arms"]["ranked_minus_shuffled"]["2"]["0"]["true_natural_actual"]
    assert comparison["ci95"] == [0, 0]
    assert comparison["birth_values"] == [0, 0]
    with np.load(trials) as data:
        assert data["motor_side"].shape == (2, 1, 2, 3, 2, 2, len(ev.BRANCHES), 12)
        assert np.array_equal(data["motor_side"][0], data["motor_side"][1])
        assert np.array_equal(data["motor_logit"][0], data["motor_logit"][1])
