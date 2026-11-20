from dataclasses import replace

import numpy as np
import torch

from experiments.pulse_game import interleave_age_train as age
from experiments.pulse_game import interleave_interference as interference
from experiments.pulse_game import interleave_plasticity as plastic
from experiments.pulse_game import local_trace_graph as local


def test_scorer_only_fields_and_complementary_prefaces():
    cfg=age.Config(pairs=3)
    births=[831]
    tape=age.make_tape(cfg,births,10,2)
    assert torch.equal(tape.left[::2],tape.left[1::2])
    assert torch.equal(tape.right[::2],tape.right[1::2])
    assert torch.equal(tape.draws[::2],tape.draws[1::2])
    assert torch.equal(tape.factor[::2],-tape.factor[1::2])
    assert torch.equal(tape.fixed[::2],-tape.fixed[1::2])
    graph,rule=age.init(births,"cold")
    loss,_=age.objective(graph,rule,tape,"carry",cfg.pairs,cfg.max_wait)
    altered=replace(tape,laws=-tape.laws,obj=torch.zeros_like(tape.obj),kind=torch.zeros_like(tape.kind))
    same,_=age.objective(graph,rule,altered,"carry",cfg.pairs,cfg.max_wait)
    torch.testing.assert_close(loss,same,atol=0,rtol=0)


def test_age_arms_have_identical_initial_block():
    cfg=age.Config(pairs=2)
    tape=age.make_tape(cfg,[832],19,1)
    g,r=age.init([832],"cold")
    with torch.no_grad():
        a=age.rollout(g,r,tape,"carry",2,8)
        b=age.rollout(g,r,tape,"episodic",2,8)
    for left,right in zip(a,b,strict=True):
        torch.testing.assert_close(left,right,atol=0,rtol=0)


def test_write_rollback_matches_count_and_b_update_sign():
    d=torch.tensor([[[.001,.002,-.05,-.06],[.10,.12,-.14,0.]]])
    selected=torch.tensor([[[True,False,True,False],[False,False,True,False]]])
    matched=interference.stratified_mask(d,selected,np.random.default_rng(3).random((1,8)))
    for sign in (-1,1):
        assert ((d*sign>0)&matched).sum()==((d*sign>0)&selected).sum()


def test_unseen_law_never_reaches_choice_before_feedback():
    g,r=local.spawn(9),local.Rule(9)
    out=plastic.evaluate(g,r,5,9,3,"novel",(0,1))
    assert out["new_correct"].shape==(2,len(plastic.BRANCHES),6)
    i=plastic.BRANCHES.index("birth_everything")
    np.testing.assert_array_equal(out["new_side"][0,i],out["new_side"][1,i])
