from collections import Counter
import numpy as np
import torch

from experiments.pulse_game import interleave_cross_skill_eval as six
from experiments.pulse_game import interleave_geometry_curriculum as curriculum
from experiments.pulse_game import interleave_one_life as joint


def test_training_orders_pair_every_public_ticket_and_private_law():
    tickets=[joint.ticket(631,731,19,4,arm) for arm in joint.ARM]
    for name in ("law","ql","qr","query","qsign","qwait","qu"):
        assert torch.equal(tickets[0][name],tickets[1][name])
    for key in ("left","right","su","wait","orientation"):
        values=[]
        for t in tickets:
            order=t["support_order"].argsort(1)
            if t[key].ndim==3:order=order[:,:,None].expand_as(t[key])
            values.append(t[key].gather(1,order))
        assert torch.equal(*values)
    groups=tickets[1]["support_order"]>=3
    assert groups[:,::2].all()
    assert not groups[:,1::2].any()


def test_joint_pay_reconstructs_both_actions_and_independent_first_choice():
    torch.set_num_threads(1)
    g=joint.graph.spawn(811)
    r=joint.graph.Rule([811])
    t=joint.ticket(719,811,17,2,"blocked",all_laws=True)
    score,paid,side,history=joint.perform(g,r,t)
    truth=t["law"].gather(1,t["query"])*t["qsign"]
    torch.testing.assert_close(paid,side*truth,rtol=0,atol=0)
    first=history[:,0,0].reshape(2,64)
    assert torch.equal(first,first[:,:1].expand_as(first))
    assert torch.isfinite(score).all() and score.requires_grad


def test_six_choice_uniform_control_has_exact_private_law_denominator():
    tape=six.make(931,5)
    truth=six.LAWS[None]*tape["query_orientation"][:,None]
    for i in range(len(six.ORDERS)):
        sides,hist=six.simulate(None,None,tape,i,"uniform")
        assert hist.shape==(5,64,7,2)
        rate=six.metrics(sides,truth,"all_maps")
        np.testing.assert_array_equal(rate[:,0],np.full(5,1/64))
        np.testing.assert_array_equal(rate[:,-1],np.full(5,3.))


def test_training_geometry_change_preserves_process_and_exact_exposure_counts():
    base=curriculum.tape(937,731,137,"hard",2)
    for layout in ("orthogonal","random"):
        t=curriculum.tape(937,731,137,layout,2)
        for k in base:
            if k not in ("left","right"):assert torch.equal(t[k],base[k])
    schedule=[curriculum.geometry("interleaved",937,i) for i in range(1,65)]
    assert Counter(schedule)=={"hard":32,"orthogonal":16,"random":16}
