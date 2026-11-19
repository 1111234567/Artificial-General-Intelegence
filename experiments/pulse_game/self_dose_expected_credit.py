"""Exact two-contact expectation audit for discrete edit choice and continuous payload credit."""
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import self_dose_continuous as carried
from . import self_param_writer as fw, ring_logic_study as io


def objective(graph, *, stop_payload=False, guard=True):
    """Enumerate both motor decisions, every address and every definable command."""
    x=torch.tensor([[.8,-.5,.7]],dtype=graph.bias.dtype)
    h=fw.advance(graph,torch.zeros_like(graph.bias),fw.sensory(graph,x,-x),5)
    l1=2.5*(h[:,11]-h[:,10]); width=graph.width
    addresses=torch.arange(width*width+width)
    exact=score=no_struct=mass=l1.sum()*0
    path_count=0
    for s1 in (-1,1):
        side=l1.new_tensor([s1]);r1=side # right was rewarded, second presentation swaps sides
        m1_log=F.logsigmoid(side*l1)
        pre=fw.advance(graph,h,fw.sensory(graph,x,-x,side),3)
        post=fw.advance(graph,pre,fw.sensory(graph,x,-x,side,r1),3)
        alog=fw.address_logits(graph,pre,post).log_softmax(1)[0]
        bank=graph.repeat(len(addresses))
        states=post.expand(len(addresses),-1)
        olog,payload=fw.operation_logits(bank,states,addresses,view="own")
        olog=olog.log_softmax(1)
        loc,op=olog.isfinite().nonzero(as_tuple=True)
        chosen=addresses[loc]
        cmd=fw.Command(chosen,op,payload[loc].detach() if stop_payload else payload[loc])
        updated,_=fw.commit(graph.repeat(len(op)),cmd,keep_connected=guard,value_limit=None)
        carry=carried.inspection(bank,states,addresses)[loc]
        odor=x.expand(len(op),-1)
        after=fw.advance(updated,carry,fw.sensory(updated,-odor,odor),5)
        l2=2.5*(after[:,11]-after[:,10])
        author=alog[loc]+olog[loc,op]
        for s2 in (-1,1):
            reward2=-s2
            m2_log=F.logsigmoid(s2*l2)
            joint=(m1_log+author+m2_log).exp()
            mass=mass+joint.sum()
            exact=exact-(joint*(s1+reward2)).sum()/2
            common=(s1+reward2)*m1_log+reward2*m2_log
            # Weight rewards by the detached path mass so the production
            # estimator's batch mean becomes its exact sampling expectation.
            zero=torch.zeros_like(m2_log)
            gains=joint.detach()[:,None]*l1.new_tensor([[s1,reward2]])
            score=score+len(op)*fw.policy_loss(torch.stack((m1_log.expand_as(m2_log),m2_log),1),
                        torch.stack((alog[loc],zero),1),torch.stack((olog[loc,op],zero),1),gains,
                        zero.new_zeros(2),zero.new_zeros(2))
            no_struct=no_struct-(joint.detach()*common).sum()/2
            path_count+=len(op)
    return exact,score,no_struct,mass,path_count


def gradients(loss,g):
    w,b=torch.autograd.grad(loss,(g.weights,g.bias),retain_graph=True)
    return torch.cat((w.flatten(),b.flatten()))


def audit_one(root,width):
    orig=fw.spawn(root,width)
    g=fw.Graph(orig.weights.double().requires_grad_(),orig.bias.double().requires_grad_(),orig.present)
    expected,scored,omitted,mass,count=objective(g)
    exact,score,without=[gradients(x,g) for x in (expected,scored,omitted)]
    stopped=gradients(objective(g,stop_payload=True)[0],g)
    assert abs(float(mass.detach())-1)<1e-10
    err=float((exact-score).abs().max())
    assert err<1e-9
    rng=np.random.default_rng([root,43133])
    dw=torch.tensor(rng.normal(size=g.weights.shape),dtype=torch.float64)*g.present
    db=torch.tensor(rng.normal(size=g.bias.shape),dtype=torch.float64)
    v=torch.cat((dw.flatten(),db.flatten()));norm=v.norm(); dw=dw/norm;db=db/norm;v=v/norm
    eps=1e-5
    with torch.no_grad():
        values=[float(objective(fw.Graph(g.weights+e*dw,g.bias+e*db,g.present))[0]) for e in (-eps,eps)]
    fd=(values[1]-values[0])/(2*eps)
    assert abs(fd-float(exact@v))<1e-7
    return dict(birth=root,graph_nodes=g.width,paths=count,mass=float(mass.detach()),
                expected_loss=float(expected.detach()),exact_gradient_norm=float(exact.norm()),
                expected_score_max_error=err,missing_discrete_score_norm=float((without-exact).norm()),
                stop_payload_missing_norm=float((stopped-exact).norm()),finite_difference=fd,
                analytic_direction=float(exact@v))


if __name__=="__main__":
    torch.set_num_threads(2)
    cases=[audit_one(seed,width) for seed,width in ((421,4),(422,8))]
    io.write(carried.EVIDENCE/"exact-two-contact-credit.json",dict(cases=cases,
        scope="Two contacts, all sampled motor/address/operation outcomes including refused commands. Executed payloads and carried activity differentiable, initial presence fixed; no numerical cap, SCC guard on, comparison to exact expected taste gradient without baseline.",
        code=io.sha(Path(__file__)),writer_source=io.sha(Path(fw.__file__))))
    print(cases)
