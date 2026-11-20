"""Guard the information boundary and the matched review intervention."""

from dataclasses import replace

import torch

from experiments.pulse_game import local_trace_graph as native
from experiments.pulse_game import noon_review_eval as replay
from experiments.pulse_game import noon_review_memory as memory
from experiments.pulse_game import noon_review_preview as previewing
from experiments.pulse_game import noon_review_world as world


def test_public_menu_and_decision_do_not_access_private_labels(monkeypatch):
    tape = world.make_tape(71, 5, 4)
    private_flip = replace(tape, law=-tape.law, final_item=(tape.final_item + 1) % tape.count)
    graph, rule = native.spawn(111), native.Rule(111)
    snapshot = native.start(graph, tape.lanes)
    coeff = rule.expand(tape.lanes)
    recorded = []
    advance = native._advance

    def record(r, state, stimulus, ticks):
        recorded.append((stimulus, ticks))
        return advance(r, state, stimulus, ticks)

    monkeypatch.setattr(native, "_advance", record)
    a = world.select(coeff, snapshot, *world.candidate_menu(tape), tape.decision_draw)
    b = world.select(coeff, snapshot, *world.candidate_menu(private_flip), tape.decision_draw)
    assert torch.equal(a.side, b.side)
    assert torch.equal(a.motor_logit, b.motor_logit)
    assert all(not ports[:, 6:10].any() for ports, _ in recorded)
    assert [ticks for _, ticks in recorded[:2]] == [world.MODE_TICKS, native.REACTION_TICKS]

    world.echo_choice(coeff, a.state, a.side)
    ports, ticks = recorded[-1]
    assert ticks == world.ECHO_TICKS
    assert not ports[:, :6].any() and not ports[:, 8:10].any()
    assert torch.equal(ports[:, 7] > 0, a.side > 0)


def test_review_effects_use_one_real_selected_contact_and_preserve_sham_state():
    tape = world.make_tape(71, 3, 4)
    graph, rule = native.spawn(111), native.Rule(111)
    view = rule.expand(tape.lanes)
    history, _ = world.introduce(view, native.start(graph, tape.lanes), tape)
    uncertain, confidence = world.parent_uncertainty(view, history, tape)
    raw = replay.evaluate(graph, rule, tape, uncertain, confidence)
    rows = torch.arange(tape.lanes)
    for name in ("native", "flip_action", "swap_content", "swap_echo", "random", "uncertain"):
        obj = torch.from_numpy(raw[f"{name}/object"])
        actual_side = torch.from_numpy(raw[f"{name}/contact_side"])
        expected = actual_side * tape.review_orientation * tape.law[rows, obj]
        assert torch.equal(torch.from_numpy(raw[f"{name}/taste"]), expected)
        assert not raw[f"{name}_sham/graph_distance"].any()
    assert (raw["native/executed_side"] == raw["swap_content/executed_side"]).all()
    assert (raw["native/object"] != raw["swap_content/object"]).all()
    assert (raw["native/object"] == raw["swap_echo/object"]).all()
    assert (raw["native/executed_side"] != raw["swap_echo/executed_side"]).all()


def test_tape_carries_two_publicly_identical_histories_per_world():
    tape = world.make_tape(world.TRAIN_SEED, 8, 4, births=(731, 732), update=11)
    assert tape.lanes == 32
    assert torch.equal(tape.law[::2], -tape.law[1::2])
    assert (tape.candidates[:, 0] != tape.candidates[:, 1]).all()
    for key, values in vars(tape).items():
        if key != "law":
            assert torch.equal(values[::2], values[1::2])


def test_candidate_preview_matches_clock_without_side_taste_or_write(monkeypatch):
    tape = world.make_tape(91, 3, 4)
    preview = previewing.make_preview(91, 3)
    graph = native.spawn(111)
    state = native.start(graph, tape.lanes)
    coeff = native.Rule(111).expand(tape.lanes)
    recorded = []
    advance = native._advance

    def record(r, active, stimulus, ticks):
        recorded.append((stimulus, ticks))
        return advance(r, active, stimulus, ticks)

    monkeypatch.setattr(native, "_advance", record)
    actual, _ = previewing.sweep(coeff, state, tape, preview, True)
    unmatched, _ = previewing.sweep(coeff, state, tape, preview, False)
    assert actual.graph is state.graph and unmatched.graph is state.graph
    assert [ticks for _, ticks in recorded] == [previewing.PREVIEW_CUE_TICKS, native.REACTION_TICKS,
                                                native.REACTION_TICKS] * 2
    assert all(not ports[:, 6:10].any() for ports, _ in recorded)
    assert torch.equal(recorded[0][0], recorded[3][0])
    rows = torch.arange(tape.lanes)
    for index in range(2):
        target = tape.candidates[rows, preview.order[:, index]]
        signal = tape.odor[rows, target] * preview.sign[:, index, None]
        assert torch.equal(recorded[index + 1][0][:, :3], signal)
        control = preview.distractor[:, index] * preview.sign[:, index, None]
        assert torch.equal(recorded[index + 4][0][:, :3], control)


def test_taste_donor_scores_choose_between_the_residents_actual_outcomes():
    tape = world.make_tape(91, 3, 4)
    raw = memory.evaluate(native.spawn(111), native.Rule(111), tape)
    endings = raw["actual_option_correct"].mean(2)
    for target in memory.DONORS:
        for component in memory.COMPONENTS:
            side = raw[f"{target}/{component}/side"]
            actual = endings[torch.arange(tape.lanes), (side > 0).astype(int)]
            assert (raw[f"{target}/{component}/score"][:, -1] == actual).all()
