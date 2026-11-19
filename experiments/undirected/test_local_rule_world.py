import torch

from . import local_rule_world as world
from .evolved_plasticity_world import FIRST_FINISH, RAW, NEW
from .local_rule_evaluator import score


def test_revisits_only_after_both_keys_and_one_use_per_request():
    tape = world.schedule(seed=91, actors=4, visits=16)
    actual = world.partners(tape.raw, tape.context)
    expected = torch.full_like(actual, -1)
    for actor in range(4):
        for raw in range(4):
            presentations = [[q for q in range(len(actual)) if tape.raw[q, actor] == raw
                              and tape.context[q, actor] == ctx] for ctx in (0, 1)]
            if all(presentations):
                start = max(times[0] for times in presentations)
                a, b = [[q for q in times if q > start] for times in presentations]
                for left, right in zip(a, b):
                    expected[left, actor] = right
    assert torch.equal(actual, expected)
    cue = tape.public[..., 0, RAW]
    assert torch.equal(cue.sum(1).argmax(-1), tape.raw)
    assert torch.equal(cue.sum(-1), tape.public[..., 0, NEW])
    assert torch.equal(tape.target, FIRST_FINISH + tape.maps[torch.arange(4), tape.context, tape.raw])
    assert tape.raw[1:4].eq(tape.raw[:3]).any()


def test_one_pair_one_point_and_timeouts_stay_in_the_budget():
    row = dict(strict=2/512, completion=3/512, post_answer_actions=2,
               pairs_both_strict=1, symbol_late_a=0., symbol_late_b=0., mean_edges=1400)
    assert abs(score(row)["selection_score"] - (2 + .1 - .2 + 1) / 512) < 1e-12
