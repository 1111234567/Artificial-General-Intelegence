import torch

from experiments.pulse_game import interleave_age_train as age
from experiments.pulse_game import interleave_readback as rb
from experiments.pulse_game import local_trace_graph as native


def test_zero_gain_is_exact_native_path_and_has_an_outer_learning_signal():
    cfg=age.Config(pairs=4)
    graph,base=age.init([731],"odor")
    r=rb.ReadRule([731],"own")
    r.base.load_state_dict(base.state_dict())
    tape=age.make_tape(cfg,[731],811,2)
    perm=rb.permutations([731],4,8,12,811,411)
    full=rb.rollout(graph,r,tape,perm,4,8)
    old=age.rollout(graph,base,tape,"carry",4,8)
    for x,y in zip(full,old,strict=True):
        torch.testing.assert_close(x,y,rtol=0,atol=0)
    loss,_=rb.loss_and_stats(graph,r,tape,perm,cfg)
    loss.backward()
    assert torch.isfinite(r.gain.grad).all() and r.gain.grad.abs().sum()>0


def test_readback_is_local_to_write_and_does_not_persist_as_an_extra_trace():
    graph=native.spawn(731)
    r=rb.ReadRule([731],"own")
    with torch.no_grad():r.gain.fill_(.85)
    cfg=age.Config(pairs=2)
    tape=age.make_tape(cfg,[731],321,1)
    view=r.expand(4)
    state=native.start(graph,4)
    x,y,u=tape.left[:,0],tape.right[:,0],tape.draws[:,0]
    a=native.act(view.base,state,x,y,u[:,0])
    obs=native.observe(view.base,a,x,y,a.side)
    permutation=rb.permutations([731],2,1,3,4,5)[:,0]
    out=rb.commit(view,obs,u[:,1:],permutation)
    assert torch.equal(out.state.sender_trace,obs.state.sender_trace)
    assert torch.equal(out.state.receiver_trace,obs.state.receiver_trace)
    assert torch.equal(permutation[::2],permutation[1::2])
