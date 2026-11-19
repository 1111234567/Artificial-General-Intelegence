"""Offline checks of held-out fork construction and real-event sign control."""

import torch

from . import uncapped_private_symbol_audit as audit
from .evolved_plasticity_world import CONTEXT, FIRST_FINISH, NEW, RAW
from .uncapped_directed_reward import _DirectedRewardPolicy


def test_fork_replays_six_visits_then_changes_only_context_and_target():
    args = dict(seed=96_700_000, actors=8, device="cpu", visits=8)
    a = audit.fork_schedule(**args, question_context=0)
    b = audit.fork_schedule(**args, question_context=1)
    for field in ("public", "stage", "target", "context", "raw"):
        assert torch.equal(getattr(a, field)[:24], getattr(b, field)[:24]), field
    assert a.public.shape[0] == b.public.shape[0] == 25
    assert torch.equal(a.stage, b.stage)
    assert torch.equal(a.raw, b.raw)
    assert torch.equal(a.maps, b.maps)
    assert a.target[24].ne(b.target[24]).all()
    neutral_a, neutral_b = a.public[24].clone(), b.public[24].clone()
    neutral_a[..., CONTEXT].zero_()
    neutral_b[..., CONTEXT].zero_()
    assert torch.equal(neutral_a, neutral_b)
    for context, tape in enumerate((a, b)):
        port = tape.public[24, ..., 0, :]
        assert torch.equal(port[..., CONTEXT.start + context], port[..., NEW])
        assert torch.equal(port[..., RAW].sum(-1), port[..., NEW])
        expected = FIRST_FINISH + tape.maps[torch.arange(8), context, torch.arange(8) % 4]
        assert torch.equal(tape.target[24], expected)


def test_sign_control_permutes_only_actual_events_and_restores_public_reward(monkeypatch):
    policy = object.__new__(audit._ShuffledRewardPolicy)
    policy.actors, policy.device = 4, torch.device("cpu")
    policy.reward = torch.tensor([[1., -1., 0., -1.], [-1., 0., 0., 0.]])
    original = policy.reward.clone()
    event = policy.reward.ne(0)
    policy.sign_control_events = torch.zeros_like(event, dtype=torch.long)
    policy.sign_control_changed_events = torch.zeros_like(event, dtype=torch.long)
    received = []
    monkeypatch.setattr(_DirectedRewardPolicy, "_change_weights",
                        lambda self, coactivity, event: received.append(self.reward.clone()))
    policy._change_weights(None, event)
    assert torch.equal(policy.reward, original)
    assert torch.equal(received[0].ne(0), event)
    for original_row, shuffled_row in zip(original, received[0], strict=True):
        assert torch.equal(original_row.sort().values, shuffled_row.sort().values)
    assert policy.sign_control_events.sum().item() == 4
    assert policy.sign_control_changed_events.sum().item() == 2


def test_single_seed_aggregates_remain_paired_and_have_no_interval():
    left = [{"seed": 96_700_000, "symbol_late_mean": .75}]
    right = [{"seed": 96_700_000, "symbol_late_mean": .25}]
    summary = audit._aggregate({"0": left, "1": right})
    assert summary["paired_scores"] == [{"seed": 96_700_000, "symbol_late_mean": .5}]
    assert audit._delta(left, right, None)["symbol_late_mean"] == {
        "mean": .5, "paired_seed_ci95": None,
    }
