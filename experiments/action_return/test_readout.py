"""Tests of actual causal state preservation and scorer noninterference."""
from copy import deepcopy
from unittest.mock import patch
import torch
import pytest
from experiments.action_return.readout import BeginController, Shadow, fit
from experiments.continuity.run import state_digest
from experiments.lifelong.evaluate import evaluate
from experiments.lifelong.world import ContextWorld
from experiments.undirected.model import PortGraphModel

@pytest.mark.parametrize('arm,dimension',[('non_input24',24),('all80',80),('public_obs56',56)])
def test_controller_preserves_real_state_and_nonbegin_scores(arm,dimension):
    base=PortGraphModel().double();h=torch.randn(8,80,dtype=torch.float64);obs=torch.randn(8,56,dtype=torch.float64)
    indices=[i for i in range(80) if i not in base.input_ports.tolist()]
    fits=[dict(mean=torch.zeros(dimension,dtype=torch.float64),std=torch.ones(dimension,dtype=torch.float64),
        coeff=torch.zeros(2,dimension,dtype=torch.float64),bias=torch.tensor([.5,-.5],dtype=torch.float64)) for _ in range(2)]
    model=BeginController(base,arm,fits,indices)
    before=state_digest(base.state_dict());native,end=base.step(obs,h,.25);actual,actual_end=model.step(obs,h,.25)
    assert torch.equal(end,actual_end)
    assert torch.equal(native[...,[0,3,4,5,6,7,8,9,10]],actual[...,[0,3,4,5,6,7,8,9,10]])
    assert bool(actual.argmax(-1).eq(1).all())
    assert state_digest(base.state_dict())==before
    # The public interface accepts only obs/state/dt, with no private world.
    assert model.config()['backbone_state_flow_unchanged']

def test_fit_retains_original_wait_and_finish_row_weights():
    x=torch.zeros((10,2),dtype=torch.float64);labels=torch.tensor([0,0,0,0,0,0,0,3,1,2])
    plan={'train':{'optimizer':dict(name='LBFGS',lr=1,max_iter=120,max_eval=180,history_size=30,line_search_fn='strong_wolfe',tolerance_grad=1e-10,tolerance_change=1e-12)}}
    model,info=fit(x,labels,plan)
    # OTHER mass=.7 WAIT+1 FINISH, BEGIN masses=1 each. A pre-collapse weight
    # retains this ratio rather than treating the collapsed OTHER as WAIT.
    prob=torch.cat((torch.zeros(1,dtype=torch.float64),model['bias'])).softmax(0)
    assert torch.allclose(prob,torch.tensor([1.7,1,1],dtype=torch.float64)/3.7,atol=1e-6)
    assert info['class_counts']==[8,1,1]
    assert info['original_action_counts'][3]==1

def test_shadow_does_not_change_original_evaluator_and_tracks_matured_cases():
    world=ContextWorld(seed=123,batch_size=8,dt=.25,n_jobs=2);base=PortGraphModel().double();h=base.initial_state(8)
    obs=world.observe();_,h=base.step(obs,h,world.elapsed);world.act(torch.zeros((8,2),dtype=torch.long))
    before=state_digest((world.state_dict(),h,base.state_dict()))
    expected=evaluate(base,world,h,family=0,condition='short',steps=96,device='cpu')
    shadow=Shadow(world,[])
    def observed(live,actions):return shadow.act(live,actions)
    with patch.object(ContextWorld,'act',observed):actual=evaluate(base,world,h,family=0,condition='short',steps=96,device='cpu')
    assert expected==actual
    extra=shadow.result()
    assert extra['all']['counts']['primary_matured']==actual['primary']['matured']
    assert extra['all']['counts']['primary_completed']==actual['primary']['completed']
    assert extra['all']['counts']['primary_clean_completed']==actual['primary']['clean_completed']
    assert sum(extra[g]['counts']['primary_matured'] for g in ('heldout','overlap'))==actual['primary']['matured']
    assert state_digest((world.state_dict(),h,base.state_dict()))==before
