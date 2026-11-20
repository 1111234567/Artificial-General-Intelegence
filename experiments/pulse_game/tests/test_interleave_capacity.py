from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from experiments.pulse_game import interleave_capacity as cap
from experiments.pulse_game import local_trace_eval as prior
from experiments.pulse_game import local_trace_graph as native
from experiments.pulse_game import local_trace_train as training
from experiments.pulse_game import revisit_choice_graph as legacy


@pytest.fixture(autouse=True)
def single_thread_torch():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.mark.parametrize("task", cap.TASKS)
def test_heldout_tape_has_matched_signs_and_stable_object_prefix(task):
    first = cap.make_tape(task, 919, 16, 1)
    extended = cap.make_tape(task, 919, 16, 5)
    for field in ("directions", "law", "left", "right", "right_value", "constant_value", "motor_draw",
                  "edit_draw", "legacy_draw"):
        values = getattr(extended, field)
        torch.testing.assert_close(getattr(first, field), values[:, :2])
    for field in ("left", "right", "motor_draw", "edit_draw", "legacy_draw"):
        values = getattr(extended, field)
        torch.testing.assert_close(values, values[::4].repeat_interleave(4, dim=0))
    torch.testing.assert_close(extended.law[::2, 0], -extended.law[1::2, 0])
    torch.testing.assert_close(extended.law[::2, 1:], extended.law[1::2, 1:])
    torch.testing.assert_close(extended.law[::4, 1:], -extended.law[2::4, 1:])
    assert (extended.right[:, :, 0] == extended.right[:, :, -1]).all(dim=-1).any()
    assert (extended.right[:, :, 0] == -extended.right[:, :, -1]).all(dim=-1).any()
    if task.startswith("feeder"):
        torch.testing.assert_close(extended.right[:, :, 0], -extended.right[:, :, 1])
        assert (extended.directions[:, 1:, 0].abs() <= .675).all()


@pytest.mark.parametrize("controller", ("local", "legacy"))
def test_probes_branch_from_same_carried_state_and_neutral_ticks_match(controller):
    tape = cap.make_tape("feeder_novel", 919, 3, 2)
    graph = training.birth_graph((731,), controller)
    rule = native.Rule(731) if controller == "local" else None
    raw = cap.evaluate(graph, rule, tape, (0, 1, 2), probe_wait=3, snapshot_worlds=tape.worlds)
    assert not raw["parameter_distance_from_post_A"][1:].any()
    assert raw["parameter_distance_from_post_A"][0, 2].any()
    assert np.isnan(raw["contact"][2, 1:]).all()
    for branch in (1, 2):
        np.testing.assert_array_equal(raw["side"][0, 0], raw["side"][branch, 0])

    def state(branch, cut):
        parameters = native.Graph(*(torch.from_numpy(raw[f"snapshot_{field}"][branch, cut])
                                    for field in cap.GRAPH_FIELDS))
        hidden = torch.from_numpy(raw["snapshot_hidden"][branch, cut])
        if rule is None:
            return legacy.State(parameters, hidden)
        traces = tuple(torch.from_numpy(raw[f"snapshot_{name}_trace"][branch, cut])
                       for name in ("sender", "receiver"))
        return native.State(parameters, hidden, *traces)

    birth_rule = None if rule is None else prior.birth_rule(rule)
    baseline, neutral = state(2, 0), state(2, 2)
    duration = 2 * 2 * cap.CONTACT_TICKS
    assert duration == 68
    waited = cap._wait(birth_rule, baseline, duration)
    torch.testing.assert_close(neutral.hidden, waited.hidden, atol=1e-6, rtol=1e-6)
    if rule is not None:
        torch.testing.assert_close(neutral.sender_trace, baseline.sender_trace)
        torch.testing.assert_close(neutral.receiver_trace, baseline.receiver_trace)
        assert not torch.equal(state(1, 2).sender_trace, baseline.sender_trace)

    for branch in range(len(cap.BRANCHES)):
        snapshot = state(branch, 2)
        for obj in (2, 0, 1):
            probe = cap._probe(birth_rule, snapshot, tape, obj)
            np.testing.assert_array_equal(probe.side.detach().numpy(), raw["side"][branch, 2, obj])
            np.testing.assert_array_equal(probe.motor_logit.detach().numpy(), raw["logit"][branch, 2, obj])


def test_replay_archives_fingerprinted_input_and_paired_metrics(tmp_path):
    birth, train_task, arm, step = 731, "other-task", "active", 0
    sources = {controller: tmp_path / controller for controller in ("local", "legacy")}
    for controller, path in sources.items():
        graph = training.birth_graph((birth,), controller)
        rule = native.Rule(birth) if controller == "local" else None
        training.save_checkpoint(path, train_task, arm, birth, step, graph, rule, lane=0)
        file = training.checkpoint_path(path, train_task, arm, birth, step)
        manifest = dict(births=[birth], arms=[arm], checkpoints=[step],
                        config=asdict(training.Config(controller=controller)),
                        checkpoint_sha256={file.relative_to(path).as_posix(): prior.digest(file)})
        (path / "run.json").write_text(json.dumps(manifest), encoding="utf-8")

    out = tmp_path / "replay"
    report = cap.run(sources, out, tasks=("odor",), source_task=train_task, worlds=4,
                     capacities=(0, 1), bootstrap_draws=8)
    assert len(report["checkpoint_sha256"]) == 2
    assert len(report["raw_sha256"]) == 2
    assert all(prior.digest(out / file) == sha for file, sha in report["checkpoint_sha256"].items())
    neutral = report["results"]["local"]["odor"]["1"]["branches"]["neutral"]
    assert neutral["B_mean"]["mean"] == .5
    difference = report["local_minus_legacy"]["odor"]["1"]["write"]["A"]
    means = [report["results"][controller]["odor"]["1"]["branches"]["write"]["A"]["mean"]
             for controller in sources]
    assert difference["mean"] == means[0] - means[1]
