from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from experiments.pulse_game import local_trace_eval as ev
from experiments.pulse_game import local_trace_graph as native
from experiments.pulse_game import local_trace_train as training


@pytest.fixture(autouse=True)
def single_thread_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_odor_tape_balances_return_order_and_opposite_private_laws():
    tape = ev.make_tape("odor", 7, 12)
    twin = tape.twin
    for field in ("left", "right", "draw", "edit_draw", "legacy_choice_draw", "shuffle_draw"):
        values = getattr(tape, field)
        assert torch.equal(values, values[twin])
    assert torch.equal(tape.right_value, -tape.right_value[twin])
    assert torch.equal(tape.left[tape.swap, -1], -tape.left[tape.swap, 0])
    assert torch.equal(tape.left[~tape.swap, -1], tape.left[~tape.swap, 0])
    assert tape.swap.sum().item() == tape.lanes // 2


def test_feeder_novel_matches_fixed_laws_and_changes_physical_mark():
    fixed = ev.make_tape("feeder", 7, 12)
    novel = ev.make_tape("feeder_novel", 7, 12)
    for field in ("first_mark", "constant_value", "right_value", "draw", "edit_draw", "legacy_choice_draw"):
        assert torch.equal(getattr(fixed, field), getattr(novel, field))
    assert torch.all(novel.left[:, 0, 0].abs() <= .675)
    assert torch.equal(novel.left[:, :2], novel.right[:, :2])
    assert torch.equal(novel.left[:, 0], -novel.left[:, 1])
    assert torch.equal(novel.left[:, -1], -novel.right[:, -1])
    assert torch.all((novel.right[:, -1] == novel.left[:, 0]).all(1)
                     | (novel.right[:, -1] == novel.left[:, 1]).all(1))


def test_odor_scores_repeat_and_switch_by_identity():
    tape = ev.make_tape("odor", 8, 4)
    raw = tape.pack()
    raw["side"] = np.array([[raw["right_value"][:, -1]]])
    raw["logit"] = raw["side"] * 10
    choices = np.ones(tape.lanes, dtype=np.int8)
    raw["contact_side"] = choices[None, None]
    raw["actual_feedback"] = (raw["right_value"][:, 0] * choices)[None, None]
    metrics = ev.branch_metrics(raw, "odor", 0, 0)
    for metric in ("positive_repeat", "negative_switch", "positive_repeat_same", "negative_switch_swap"):
        assert metrics[metric].all()
    assert metrics["motor_repeat"].mean() == .5


def test_native_interventions_use_twin_and_value_sham_at_intended_phases():
    tape = ev.make_tape("odor", 8, 8)
    graph, rule = native.spawn(731), native.Rule(731)
    raw = ev.evaluate(graph, rule, tape, (0, 4, 16))
    expanded = ev.evaluate(graph, rule.expand(tape.lanes), tape, (0, 4, 16))
    np.testing.assert_array_equal(raw["side"], expanded["side"])
    audit = ev.audit(raw)
    assert audit["matched_twin_pairs"] == tape.lanes // 2
    assert audit["donor_receipts"] == tape.lanes // 2
    assert audit["precommit_twin_difference_lanes"]["sender_trace"] == tape.lanes
    assert audit["precommit_twin_difference_lanes"]["weights"] == 0
    assert audit["shuffle_values_last_commit"]["weight_count_mismatch_lanes"] == 0
    assert audit["shuffle_values_last_commit"]["additive_payload_mismatch_lanes"] == 0
    assert audit["shuffle_values_last_commit"]["different_additive_values_lanes"] > 0
    assert raw["final_delta"].shape == (3, tape.lanes, native.SHUFFLE_DRAWS)
    damaged = {**raw, "donor_feedback": raw["donor_feedback"].copy()}
    damaged["donor_feedback"][0, 0] *= -1
    with pytest.raises(AssertionError, match="Donor receipt"):
        ev.audit(damaged)


def test_paired_checkpoint_roundtrip_and_fingerprint_validation(tmp_path):
    task, arm, step = "feeder", "active", 0
    birth = 731
    paths = {controller: tmp_path / controller for controller in ("local", "legacy")}
    for controller, source in paths.items():
        graph = training.birth_graph((birth,), controller)
        rule = native.Rule(birth) if controller == "local" else None
        training.save_checkpoint(source, task, arm, birth, step, graph, rule, lane=0)
        file = training.checkpoint_path(source, task, arm, birth, step)
        manifest = dict(births=[birth], arms=[arm], checkpoints=[step],
                        config=asdict(training.Config(controller=controller)),
                        checkpoint_sha256={file.relative_to(source).as_posix(): ev.digest(file)})
        (source / "run.json").write_text(json.dumps(manifest), encoding="utf-8")

    out = tmp_path / "evaluation"
    result = ev.run(paths["local"], out, pairs=4, tasks=(task,), delays=(0,), bootstrap_draws=10,
                    legacy_run=paths["legacy"])
    assert len(result["checkpoint_sha256"]["local"]) == len(result["checkpoint_sha256"]["legacy"]) == 1
    assert result["local_minus_legacy"]["0"][task][arm]["0"]["zero"]["correct"]["ci95"] == [0, 0]
    weaker = result["results"]["0"][task][arm]["0"]["branches"]["intact"]
    assert weaker["weaker_law"]["mean"] == min(weaker["law_negative"]["mean"], weaker["law_positive"]["mean"])
    with np.load(out / f"{task}-{arm}-{birth}-{step}.npz") as a:
        with np.load(out / f"legacy-{task}-{arm}-{birth}-{step}.npz") as b:
            for name in ("left", "right", "right_value", "draw", "edit_draw"):
                np.testing.assert_array_equal(a[name], b[name])
            assert b["no_write_graph_unchanged"].item()

    control = paths["legacy"]
    changed = ev.load_manifest(control)
    checkpoint_name = next(iter(changed["checkpoint_sha256"]))
    changed["checkpoint_sha256"][checkpoint_name] = "incorrect"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        ev.verified_checkpoint(training, control, changed, task, arm, birth, step)
