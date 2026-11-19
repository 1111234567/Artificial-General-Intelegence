import torch
from torch import nn
from experiments.action_return.readout_swap import SwapController

class Scores(nn.Module):
    width=80;input_size=56;n_jobs=2;n_actions=11
    def step(self,obs,h,dt,max_step=None):return obs.reshape(-1,2,11),h
    def config(self):return {'kind':'test_scores'}

def test_swap_preserves_same_h_gate_and_nonbegin_with_wait_finish_and_begin_ties():
    logits=torch.zeros((6,2,11),dtype=torch.float64)
    logits[0,:,0]=3;logits[0,:,1:3]=torch.tensor([1.,2.])
    logits[1,:,3]=3;logits[1,:,1:3]=torch.tensor([1.,2.])
    logits[2,:,1:3]=torch.tensor([4.,2.])
    logits[3,:,1:3]=torch.tensor([2.,4.])
    logits[4,:,1:3]=4
    logits[5,:,3]=4;logits[5,:,1:3]=torch.tensor([4.,2.])
    h=torch.randn(6,80,dtype=torch.float64);actual,end=SwapController(Scores()).step(logits.reshape(6,22),h,.25)
    assert torch.equal(end,h)
    assert torch.equal(logits[...,[0,3,4,5,6,7,8,9,10]],actual[...,[0,3,4,5,6,7,8,9,10]])
    assert actual[:,0].argmax(-1).tolist()==[0,3,2,1,1,2]
