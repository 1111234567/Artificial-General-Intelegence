import torch

from experiments.pulse_game import interleave_age_train as age, interleave_adult_train as adult


def test_without_prefix_both_training_histories_get_same_loss_and_gradient():
    cfg=adult.Config(pairs=3)
    data=adult.batch(cfg,[731],17,0)
    g,r=age.init([731],"odor")
    left,ls=adult.rollout(g,r,data,cfg,0,"aged")
    right,rs=adult.rollout(g,r,data,cfg,0,"sham_age")
    assert torch.equal(ls,rs)
    assert torch.equal(left,right)
    x=torch.autograd.grad(left,tuple(r.parameters()))
    y=torch.autograd.grad(right,tuple(r.parameters()))
    assert all(torch.equal(a,b) for a,b in zip(x,y,strict=True))


def test_long_prefix_is_detached_but_suffix_rule_learns():
    cfg=adult.Config(pairs=3)
    data=adult.batch(cfg,[731],27,8)
    assert torch.equal(data["right"][::2],data["right"][1::2])
    assert torch.equal(data["law"][::2],-data["law"][1::2])
    g,r=age.init([731],"odor")
    loss,_=adult.rollout(g,r,data,cfg,8,"aged")
    gradient=torch.autograd.grad(loss,(g.weights,g.bias,*r.parameters()),allow_unused=True)
    assert gradient[0] is None and gradient[1] is None
    assert any(x is not None and x.abs().sum()>0 for x in gradient[2:])
