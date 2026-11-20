from dataclasses import fields

import torch

from experiments.pulse_game import interleave_train as common, interleave_wide_capacity as wide


def test_matched_contacts_and_fully_crossed_worlds():
    cfg = wide.Config()
    tape = wide.make_tape(cfg, (731, 732), 4001, contacts=16)
    any_tape, recent, few = (wide.physical(tape, arm) for arm in wide.ARMS)
    for physical in (recent, few):
        assert torch.equal(any_tape[2], physical[2])
        assert torch.equal(any_tape[3], physical[3])
    for field in (0, 1):
        assert torch.equal(any_tape[field][:-1], recent[field][:-1])
        assert torch.equal(any_tape[field][:4], few[field][:4])
    assert tape.queries[..., 1].min() >= tape.contacts - 2
    assert tape.repeat_visits.max() == 3
    vectors, payout = any_tape[:2]
    assert torch.equal(vectors[:, ::4], -vectors[:, 1::4])
    assert torch.equal(vectors[:, ::4], vectors[:, 2::4])
    assert torch.equal(payout[:, ::4], -payout[:, 1::4])
    assert torch.equal(payout[:, ::4], -payout[:, 2::4])
    assert torch.equal(payout[:, ::4], payout[:, 3::4])


def test_block_cut_retains_all_sampled_scores_and_cuts_old_pathwise_gradient():
    torch.set_num_threads(1)
    cfg = wide.Config(worlds=2, max_wait=0)
    tape = wide.make_tape(cfg, (731,), 4001, contacts=8)
    graph = wide.net.spawn(731)
    graph.weights.requires_grad_()
    rule = wide.net.Rule(731)
    exact, _ = wide.trajectory(graph, rule, tape, "wide_any", 0, truncate=False)
    cut, _ = wide.trajectory(graph, rule, tape, "wide_any", 0, truncate=True)
    assert all(torch.equal(a, b) for a, b in zip(exact, cut, strict=True))
    assert torch.autograd.grad(exact[-1].sum(), graph.weights)[0].abs().max() > 0
    assert torch.autograd.grad(cut[-1].sum(), graph.weights, allow_unused=True)[0] is None
    assert torch.autograd.grad((cut[1] + cut[2]).sum(), graph.weights)[0].abs().max() > 0


def test_capacity_extension_preserves_every_standard_field():
    short = wide.capacity.make_tape("odor", 871831, 5, 16)
    extended = wide.capacity.make_tape("odor", 871831, 5, 32)
    for field in fields(short):
        a, b = getattr(short, field.name), getattr(extended, field.name)
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b[:, :17])


def test_resume_keeps_adam_and_the_indexed_world_stream(tmp_path):
    birth = 731
    source = tmp_path / "source"
    wide.prior.save_checkpoint(source, "odor", "uniform", birth, 8000, wide.net.spawn(birth),
                               wide.net.Rule(birth), 0)
    a, b = tmp_path / "interrupted", tmp_path / "main"
    cfg = wide.Config(updates=6, worlds=2, max_wait=0, checkpoint_every=3)
    arm = "wide_any"
    wide.fit(a, cfg, (birth,), source, 8000, arm, 3, 0)
    wide.fit(a, cfg, (birth,), source, 8000, arm, 6, 0)
    wide.fit(b, cfg, (birth,), source, 8000, arm, 6, 0)
    a, b = (torch.load(common.checkpoint_path(path, arm, 6), weights_only=True) for path in (a, b))
    assert torch.equal(a["metrics"], b["metrics"])
    assert torch.equal(a["contacts"], b["contacts"])
    for key in a["optimizer"]["state"]:
        assert all(torch.equal(v, b["optimizer"]["state"][key][k]) for k, v in a["optimizer"]["state"][key].items())
