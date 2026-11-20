import numpy as np
import torch

from experiments.pulse_game import interleave_binding as binding, interleave_code_probe as probe
from experiments.pulse_game import interleave_capacity as capacity, interleave_scorecard as card
from experiments.pulse_game import interleave_evidence as evidence
from experiments.pulse_game import interleave_readback as readback, local_trace_graph as local


def test_sum_of_two_public_signed_unit_vectors_recalls_both_nonparallel_pairs():
    tape=capacity.make_tape("odor",912110,64,1)
    out=card.reference(tape,"sum")
    assert out[1]["all"]==1


def test_exhaustive_label_measure_does_not_reward_global_flip_as_item_selectivity():
    tape=binding.geometry_tape(123,4,3,"random")
    laws=tape["law"].numpy().astype(np.int8)
    ideal=dict(answer=laws,p_plus=(laws+1)/2)
    metrics=binding.summarize(ideal,laws)
    assert metrics["distinct_output_patterns"]==8
    np.testing.assert_array_equal(metrics["change_one_input_flip_rate"],np.eye(3))
    assert metrics["targeted_minus_collateral"]==1


def test_frustrated_triangle_has_no_strictly_positive_linear_solution():
    tape=probe.make_scene(4,5,8,"triangle")
    right=tape["right"][::8]
    assert right.sum(1).abs().max()<1e-6
    weights=torch.randn(len(right),3,1)
    margin=torch.bmm(right,weights).squeeze(-1)
    assert not (margin>.00001).all(1).any()
    assert not (margin<-.00001).all(1).any()


def test_stationary_noise_benchmark_is_balanced_at_zero_contacts():
    assert evidence.optimal(0,.2)==.5
    np.testing.assert_allclose(evidence.optimal(3,.2),.896)


def test_zero_readback_scorecard_replays_original_controller():
    tape=capacity.make_tape("odor",417,3,1)
    g=local.spawn(4)
    r=readback.ReadRule([4],"own")
    before=capacity.evaluate(g,r.base,tape,(0,1),snapshot_worlds=0)
    replay=card.evaluate_readback(g,r,tape,(0,1),417)
    np.testing.assert_array_equal(replay["side"],before["side"])
    np.testing.assert_allclose(replay["logit"],before["logit"],rtol=0,atol=0,equal_nan=True)
