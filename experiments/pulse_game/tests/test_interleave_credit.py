from dataclasses import replace

import torch

from experiments.pulse_game import interleave_age_train as age, interleave_credit_train as credit


def test_truncation_preserves_every_motor_edit_score_reward_and_final_value():
    cfg=age.Config(pairs=2)
    tape=age.make_tape(cfg,[731],53,2)
    g,r=age.init([731],"odor")
    base=credit.rollout(g,r,tape,"full",cfg)
    for mode in ("hidden","graph","all"):
        other=credit.rollout(g,r,tape,mode,cfg)
        for a,b in zip(base[:3],other[:3],strict=True):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
        for k in ("hidden","sender_trace","receiver_trace"):
            assert torch.equal(getattr(base[3],k),getattr(other[3],k))
        assert torch.equal(base[3].graph.weights,other[3].graph.weights)
        assert torch.equal(base[3].graph.present,other[3].graph.present)
    x=credit.loss(g,r,tape,"full",cfg)[0]
    y=age.objective(g,r,tape,"carry",cfg.pairs,cfg.max_wait)[0]
    torch.testing.assert_close(x,y,atol=0,rtol=0)
    params=(g.weights,g.bias,*r.parameters())
    ga=torch.autograd.grad(x,params)
    gb=torch.autograd.grad(y,params)
    for a,b in zip(ga,gb,strict=True):
        torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_boundary_detaches_only_requested_derivatives():
    g,r=age.init([731],"odor")
    state=age.net.start(g,2)
    state=replace(state,hidden=state.hidden.requires_grad_(),sender_trace=state.sender_trace.requires_grad_(),
                  receiver_trace=state.receiver_trace.requires_grad_())
    after=credit.boundary(state,"graph")
    assert not after.graph.weights.requires_grad
    assert after.hidden is state.hidden
    after=credit.boundary(state,"hidden")
    assert after.graph.weights.requires_grad and not after.hidden.requires_grad
