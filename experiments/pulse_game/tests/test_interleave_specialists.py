import numpy as np
import torch

from experiments.pulse_game import interleave_adult_eval as adult, interleave_context_time as context
from experiments.pulse_game import interleave_motor as motor, interleave_susceptibility as susceptibility
from experiments.pulse_game import interleave_stream as stream,local_trace_graph as graph


def test_motor_extension_is_exactly_native_at_birth_and_stays_bounded():
    torch.set_num_threads(1)
    g=graph.spawn(731);rule=graph.Rule(731)
    tape=context.make(context.Config(worlds=2),731,4,forced_angle=0)
    n=len(tape["law"]);s=graph.start(g,n);view=rule.expand(n)
    vec=tape["odor"][:,0]
    draw=tape["u"][:,0,0]
    base=graph.act(view,s,-vec,vec,draw)
    for mode in motor.ARMS:
        a=motor.action(view,s,-vec,vec,draw,torch.zeros(n),mode)
        assert torch.equal(a.side,base.side) and torch.equal(a.motor_logit,base.motor_logit)
        a=motor.action(view,s,-vec,vec,draw,torch.full((n,),20.),mode)
        assert (a.motor_logit.abs()<=5).all()


def test_susceptibility_controls_preserve_outcomes_when_gains_are_equal():
    torch.set_num_threads(1)
    g=graph.spawn(731);r=graph.Rule(731)
    t=susceptibility.tape(731,5,7,2,3,replicas=1)
    native,pre=susceptibility.simulate(g,r,t,torch.zeros(2,20),"native",all_queries=True,pre_new=True)
    for arm in susceptibility.ARMS:
        genes=torch.zeros(2,1 if arm=="global" else 20)
        sampled,old=susceptibility.simulate(g,r,t,genes,arm,all_queries=True,pre_new=True)
        assert torch.equal(sampled,native) and torch.equal(pre,old)
    target=t["truth"].reshape(2,8,3).numpy()
    for world in target:
        assert len({tuple(row) for row in world})==8


def test_exposure_history_does_not_reveal_two_previously_unseen_outcomes():
    tape=adult.make(195,731,3,4,"novel")
    assert tape["goal"].shape==(24,6)
    for group in range(3):
        block=slice(group*8,(group+1)*8)
        assert len(torch.unique(tape["goal"][block,-2:],dim=0))==4
    with torch.no_grad():
        z=adult.evaluate(graph.spawn(731),graph.Rule(731),tape,cuts=(0,4))
    np.testing.assert_array_equal(z["active/active/0"],z["sham/active/0"])


def test_stream_never_shares_probe_feedback_with_next_measurement():
    tape=stream.tape_for_worlds(731,3,2,742353)
    g=graph.spawn(731);r=graph.Rule(731)
    first=stream.evaluate(g,r,tape,742353,731,(0,1,2),("carry",))
    last=stream.evaluate(g,r,tape,742353,731,(0,2),("carry",))
    for key in ("side","credit"):
        np.testing.assert_array_equal(first["carry/"+key],last["carry/"+key])
    np.testing.assert_array_equal(first["carry/read"][-1],last["carry/read"][-1])
