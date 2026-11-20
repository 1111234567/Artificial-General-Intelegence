import numpy as np
import torch

from experiments.pulse_game import interleave_local_consolidation as learn
from experiments.pulse_game import interleave_slow_reference as slow,local_trace_graph as native


def test_equal_slow_rates_match_the_previous_slow_reference_exactly():
    torch.set_num_threads(1)
    tape=slow.public_tapes(2,51,922)
    g,r=native.spawn(731),native.Rule(731)
    with torch.no_grad():
        native_result=slow.live(g,r,tape,"absent_a",4,.12,.02)["correct"]
        trial=learn.run_bank(g,r,tape,"absent_a",4,[(.02,.02)],172)[0]
    ix=[slow.AGES.index(k) for k in learn.AGES]
    np.testing.assert_array_equal(trial,native_result[ix][...,[0,1,2,3,4,6]])


def test_rate_shuffle_preserves_exact_multiset_on_eligible_edges():
    x=torch.tensor([[.01,.02,.03,.08,.25],[0.,.02,.08,0.,.02]])
    mask=torch.tensor([[True,False,True,True,False],[False,False,False,True,False]])
    priority=torch.tensor([[.5,.4,.3,.2,.1],[.3,.2,.1,.5,.6]])
    y=learn.permute_eligible(x,mask,priority)
    for old,new,m in zip(x,y,mask,strict=True):
        assert torch.equal(old[m].sort().values,new[m].sort().values)
