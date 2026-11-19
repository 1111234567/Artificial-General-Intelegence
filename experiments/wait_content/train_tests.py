"""Pre-freeze checks of the matched recurrent loss and state/gradient path."""
import math
from copy import deepcopy

import numpy as np
import torch

from experiments.wait_content import train as t


def test_all_position_action_loss_and_exact_fp64_wait_weight():
    begin=torch.zeros(1,1,2,2,dtype=torch.float64,requires_grad=True)
    finish=torch.zeros(1,1,2,8,dtype=torch.float64,requires_grad=True)
    actions=torch.tensor([[[0,3]]])
    anchors=np.empty((0,8),dtype=np.int64)
    loss,parts,budget=t.objective(begin,finish,actions,anchors,torch.empty((0,3),dtype=torch.long))
    assert budget['action_denominator']==1.1
    assert budget['conditional_weight_sum']==1.1
    assert float(parts['group'].detach())==math.log(3)
    assert abs(float(parts['conditional'].detach())-math.log(9))<1e-15
    assert abs(float(loss.detach())-(math.log(3)+math.log(9)))<1e-15
    loss.backward()
    assert finish.grad[0,0,0].gt(0).all()  # WAIT remains WAIT; lower all8 scores.
    assert finish.grad[0,0,1,0]<0         # Every true FINISH learns its content.


def test_auxiliary_moves_only_half_and_keeps_same_target_budget():
    anchors=np.array([[0,0,0,10,10,5,1,0],[0,0,1,20,20,6,7,1]],dtype=np.int64)
    left,lp=t.selected_anchors(anchors,0,'end_only')
    right,rp=t.selected_anchors(anchors,0,'wait_mixed')
    assert np.array_equal(left,right)
    assert lp.tolist()==[[10,0,0],[20,0,1]]
    assert rp.tolist()==[[10,0,0],[6,0,1]]
    begin=torch.zeros(128,1,2,2,dtype=torch.float64,requires_grad=True)
    finish=torch.arange(128*1*2*8,dtype=torch.float64).reshape(128,1,2,8)/100
    finish.requires_grad_(True)
    actions=torch.zeros(128,1,2,dtype=torch.long)
    a,pa,ba=t.objective(begin,finish,actions,left,lp)
    z,pz,bz=t.objective(begin,finish,actions,right,rp)
    assert torch.equal(pa['action'],pz['action'])
    assert ba==bz
    extra=torch.autograd.grad(pz['extra'],finish)[0]
    assert extra[6,0,1].ne(0).all()
    assert extra[20,0,1].eq(0).all()


def test_empty_auxiliary_still_has_finite_backward():
    begin=torch.randn(3,2,2,2,dtype=torch.float64,requires_grad=True)
    finish=torch.randn(3,2,2,8,dtype=torch.float64,requires_grad=True)
    actions=torch.tensor([[[1,2],[0,0]],[[3,10],[0,2]],[[5,0],[1,8]]])
    loss,parts,budget=t.objective(begin,finish,actions,np.empty((0,8),dtype=np.int64),torch.empty((0,3),dtype=torch.long))
    loss.backward()
    assert budget['action_positions']==12
    assert torch.isfinite(begin.grad).all() and torch.isfinite(finish.grad).all()
    assert float(parts['extra'].detach())==0


def test_real_inherited_model_promotion_flow_and_fixed_support_gradient():
    model,trainer=t.load_initial(0)
    original=deepcopy(model)
    before=t.b.state_digest(model.state_dict())
    mask=trainer.operator.mask.clone()
    params=t.trainable(model,mask)
    assert t.b.state_digest(model.state_dict())==before
    obs=torch.randn(3,4,56,generator=torch.Generator().manual_seed(10),dtype=torch.float64)
    elapsed=[.5,.75,.25]
    h0=trainer.h[:4].clone()
    with torch.no_grad():
        h=h0.clone();states=[]
        for o,d in zip(obs,elapsed):
            _,h=original.step(o,h,d);states.append(h.clone())
        _,_,end,fast=t.rollout(model,obs,h0,elapsed)
        assert torch.equal(torch.stack(states),fast)
        assert torch.equal(h,end)
    begin,finish,h,_=t.rollout(model,obs,h0,elapsed)
    loss,_,_=t.objective(begin,finish,torch.zeros(3,4,2,dtype=torch.long),np.empty((0,8),dtype=np.int64),torch.empty((0,3),dtype=torch.long))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    assert model.begin_head.base.edge_shared.grad[~mask].eq(0).all()
    assert [n for n,p in model.named_parameters() if p.requires_grad]==['j0_coef','j0_bias','j1_coef','j1_bias','begin_head.base.edge_shared']
