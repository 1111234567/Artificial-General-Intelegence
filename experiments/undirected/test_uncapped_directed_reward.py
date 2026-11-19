from __future__ import annotations

import pytest
import torch

from .evolved_plasticity import INPUTS, RULE_FEATURES
from .evolved_plasticity_world import FIRST_FINISH
from .plasticity_continuous_world import make_continuous_schedule
from .uncapped_directed_reward import RewardEvaluation, _DirectedRewardPolicy, _stack_directed_genomes
from .uncapped_reward_precision import PrecisionEvaluation


NODES = 78
PAIRS = NODES * (NODES - 1) // 2


def _genome(*, directed, rate=.02):
    return {
        "birth_w": torch.zeros(2 * PAIRS),
        "birth_mask": torch.zeros(2 * PAIRS, dtype=torch.bool),
        "node_bias": torch.zeros(NODES),
        "rule": torch.zeros((2, len(RULE_FEATURES))),
        "directed": torch.tensor(directed),
        "reward_rate": torch.tensor(rate, dtype=torch.float32),
    }


def _trial_genome(*, directed, rate=.02, support=True):
    genome = _genome(directed=directed, rate=rate)
    genome["birth_mask"][::PAIRS] = True
    genome["birth_w"][::PAIRS] = .25
    genome["node_bias"][INPUTS + FIRST_FINISH] = .31
    genome["node_bias"][:2] = .2
    if support:
        genome["rule"][1, 0] = 1.25  # Empty arcs are admitted during this tape.
    return genome


def test_present_arc_propagates_and_writes_only_in_its_own_direction():
    genome = _genome(directed=True)
    genome["birth_w"][0] = .25  # W[0,1]: source 1 sends to destination 0.
    genome["birth_mask"][0] = True
    _, birth = _stack_directed_genomes([genome])
    policy = _DirectedRewardPolicy(birth, 2, torch.device("cpu"), "full")
    stimulus = torch.zeros((1, 2, INPUTS))
    stimulus[0, 0, 1] = stimulus[0, 1, 0] = 1.
    policy.act(stimulus)
    policy.act(torch.zeros_like(stimulus))
    assert policy.h[0, 0, 0] > 0  # Actor 0: the forward path exists.
    assert policy.h[0, 1, 1] == 0  # Actor 1: the reverse path is absent.
    assert policy.summary()["final_reciprocity"].item() == 0.

    policy.reward.fill_(1.)
    policy._change_weights(torch.ones((1, 2, 1, PAIRS), dtype=torch.float16), policy.reward.bool())
    assert (policy.w[..., 0, 0] > .25).all()
    assert not policy.w[..., 1, :].any()
    assert not policy.eligibility[..., 1, :].any()


def test_reciprocal_twins_and_old_half_evaluator_agree():
    genome = _trial_genome(directed=False)
    mixed = [genome, {**genome, "directed": torch.tensor(True)}]
    args = dict(seed=372, actors=2, visits=4, device="cpu", capture=False)
    with torch.inference_mode():
        new_metrics = RewardEvaluation(make_continuous_schedule)(mixed, **args)
        original = {**genome, **{key: genome[key][:PAIRS] for key in ("birth_w", "birth_mask")}}
        old_metrics = PrecisionEvaluation(make_continuous_schedule)([original], dtype=torch.float16, **args)
    for name, value in new_metrics.items():
        assert torch.equal(value[:1], value[1:]), name
    assert new_metrics["final_reciprocity"].tolist() == [1., 1.]
    for name in ("first_finish_correct", "ready_coverage", "reward_events", "first_finish_correct_a_first_return",
                 "first_finish_correct_b_first_return", "changed_writes"):
        factor = 2 if name == "changed_writes" else 1
        torch.testing.assert_close(new_metrics[name][:1], factor * old_metrics[name])
    for name in ("initial_edges", "final_edges", "mean_edges", "births", "deaths", "attempted_writes"):
        torch.testing.assert_close(new_metrics[name][:1], 2 * old_metrics[name])


def test_ablations_keep_public_receipts_and_only_disable_requested_updates():
    evaluate = RewardEvaluation(make_continuous_schedule)
    genome = _trial_genome(directed=True)
    args = dict(seed=32, actors=1, device="cpu", capture=False)
    full, unwritten, frozen = (evaluate([genome], mode=mode, **args)
                                for mode in ("full", "no_reward_write", "frozen"))
    assert full["changed_writes"].item() > 0
    for metrics in (unwritten, frozen):
        assert metrics["changed_writes"].item() == metrics["attempted_writes"].item() == 0
        assert metrics["reward_events"].item() > 0
        assert 0 < metrics["wrong_answer_rate"].item() <= metrics["ready_coverage"].item()
        assert metrics["premature_request_rate"].item() > 0
    assert unwritten["births"].item() > 0 and unwritten["final_edges"].item() > unwritten["initial_edges"].item()
    assert frozen["births"].item() == frozen["deaths"].item() == 0
    assert frozen["final_edges"].item() == frozen["mean_edges"].item() == frozen["initial_edges"].item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph requires CUDA")
def test_cuda_capture_same_shape_replay_refreshes_seed_birth_and_rate():
    evaluate = RewardEvaluation(make_continuous_schedule)
    genomes = [_trial_genome(directed=True, support=False), _trial_genome(directed=False, support=False)]
    # Make the directed member asymmetric and change both its weights and
    # reward coefficient on the next call with the same batch shape.
    genomes[0]["birth_mask"][PAIRS] = False
    seen = None
    for seed, rate in ((91, .02), (108, .14)):
        genomes[0]["reward_rate"].fill_(rate)
        genomes[0]["birth_w"][0] = rate
        args = dict(seed=seed, actors=2, visits=4, device="cuda")
        eager = evaluate(genomes, **args, capture=False)
        replay = evaluate(genomes, **args, capture=True)
        for name, expected in eager.items():
            torch.testing.assert_close(replay[name], expected, msg=name)
        cpu = evaluate(genomes, **{**args, "device": "cpu"}, capture=False)
        for name in ("first_finish_correct", "ready_coverage", "wrong_answer_rate", "premature_request_rate",
                     "initial_edges", "final_edges", "final_reciprocity", "reward_events", "changed_writes"):
            torch.testing.assert_close(replay[name].cpu(), cpu[name], msg=name)
        if seen is not None:
            assert seen["weight_change"][0].item() != replay["weight_change"][0].item()
        seen = replay
    evaluate.clear_cache()
