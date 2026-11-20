"""Route a reward-trained motor readback into the native recurrent circuit; continue from actual returns."""

from argparse import ArgumentParser
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import torch
from torch.nn import functional as F

from . import interleave_one_life_repair as archive, noon_online_readout as online
from . import noon_readback as bank, noon_readback_fit as fitted, noon_review_world as w
from . import local_trace_graph as net,local_trace_train as old,revisit_choice_graph as motor


OUT=w.DATA/"motor-feedback";PLAN=w.EVIDENCE/"motor-feedback-plan.json"
ARMS=("attached","fade","unassisted");STEPS=(0,1500,3000,4500,6000)


def strength(arm,step):
    if arm=="attached":return 1.
    if arm=="unassisted":return 0.
    return min(1.,max(0.,(4500-step)/3000))


def precursor():
    f=online.OUT/str(online.SEEDS[0])/"live/step-4000.pt"
    m=fitted.Readout("logged")
    m.load_state_dict(torch.load(f,weights_only=True,map_location="cpu")["weights"])
    return m.requires_grad_(False),f


def parents(cohort):
    ids=w.COHORTS[cohort];root=w.PARENT_DATA/"one-life-repair"/cohort
    all_=[old.load_checkpoint(root,"odor","mixed",b,2000) for b in ids]
    a,b,p=(torch.cat([getattr(g,k) for g,r in all_]) for k in ("weights","bias","present"))
    g=net.Graph(torch.nn.Parameter(a),torch.nn.Parameter(b),p);r=net.Rule(ids)
    r.load_state_dict({n:torch.cat([rule.state_dict()[n] for _,rule in all_]) for n in r.state_dict()})
    return g,r


def begin(g,r,t,first,readout):
    n=t.lanes//len(g.bias);view=r.expand(n);s=net.start(g,n)
    state,visits=w.introduce(view,s,t)
    state,z=bank.scan(view,state,t,first)
    score=readout.scores(z.double()).to(g.bias.dtype)
    signal=((score[:,1]-score[:,0])*.5).tanh()
    return view,state,visits,signal,z


def emit(view,state,t,signal,amplitude):
    marker=state.graph.bias.new_tensor((0.,0.,.85))[None].expand_as(t.odor[:,0])
    state=net._advance(view,state,net.native.sensory(state.graph,marker,marker),w.MODE_TICKS)
    l,right=w.candidate_menu(t)
    stimulus=net.native.sensory(state.graph,l,right)
    current=torch.zeros_like(stimulus);current[:,10]=-signal*amplitude;current[:,11]=signal*amplitude
    result=net._advance(view,state,stimulus+current,net.REACTION_TICKS)
    logits=2.5*(result.hidden[:,11]-result.hidden[:,10])
    side=(t.decision_draw<logits.detach().sigmoid()).to(logits.dtype)*2-1
    return net.Action(result,side,logits,F.logsigmoid(side*logits))


def review_life(g,r,readout,packet,amplitude):
    tape,first=packet
    view,state,visits,signal,_=begin(g,r,tape,first,readout)
    a=emit(view,state,tape,signal,amplitude);s,ev=w.review(view,a.state,tape,a.side)
    row=w._rows(tape);object_id=tape.final_item;sign=tape.final_orientation[row,object_id]
    l,right=w.sensory_for(tape,object_id,sign);final=motor.act(s,l,right,tape.final_draw[row,object_id])
    paid=final.side*tape.law[row,object_id]*sign>0
    history=sum(v.motor.motor_log_prob+v.edit.edit_log_prob for v in visits)
    return history+a.motor_log_prob+ev.motor.motor_log_prob+ev.edit.edit_log_prob+final.motor_log_prob,paid


def schedule(seed,step):
    block,slot=divmod(step-1,4)
    ix=np.random.default_rng([472981,seed,block]).permutation(4)[slot]
    return ("review","review","short","mixed")[ix]


def review_tape(seed,ids,step,worlds=4):
    items=[]
    for b in ids:
        entropy=np.random.SeedSequence([197621,seed,b,step]).generate_state(1)[0]
        items.append(bank.tape(int(entropy),worlds,4))
    return (w.Tape(**{f.name:torch.cat([getattr(q[0],f.name) for q in items]) for f in online.fields(w.Tape)}),
            torch.cat([q[1] for q in items]))


def update(g,r,opt,critic,packet,which,amp,worlds):
    if which=="review":work=lambda m,rule,t:review_life(m,rule,critic,t,amp)
    elif which=="short":work=archive.short_life
    else:work=archive.long_life
    return archive.update(g,r,opt,work,packet,worlds)


def fit(cohort,arm):
    torch.set_num_threads(1);ids=list(w.COHORTS[cohort]);seed=1879793 if cohort=="original" else 1879807
    root=OUT/cohort;folder=root/arm;folder.mkdir(parents=True,exist_ok=True)
    g,r=parents(cohort);critic,cp=precursor();parents_root=w.PARENT_DATA/"one-life-repair"/cohort
    sha=lambda f:sha256(Path(f).read_bytes()).hexdigest()
    manifest=dict(cohort=cohort,arm=arm,seed=seed,steps=6000,lr=.002,plan=sha(PLAN),precursor=sha(cp),
                  parents={str(b):sha(old.checkpoint_path(parents_root,"odor","mixed",b,2000)) for b in ids},
                  sources={Path(f).name:sha(f) for f in (__file__,bank.__file__,w.__file__,net.__file__,archive.__file__)})
    mf=folder/"run.json"
    if mf.exists() and json.loads(mf.read_text())!=manifest:raise ValueError("Motor afferent study changed")
    if not mf.exists():bank.lesson.storage._write_json(mf,manifest)
    digest=sha(mf);opt=torch.optim.Adam((g.weights,g.bias,*r.parameters()),lr=.002)
    step=0;trace=[];resume=folder/"resume.pt";counts={k:0 for k in ("review","short","mixed")}
    if resume.exists():
        d=torch.load(resume,weights_only=True,map_location="cpu")
        if d["plan"]!=digest:raise ValueError("Wrong motor feedback optimizer")
        with torch.no_grad():g.weights.copy_(d["weights"]);g.bias.copy_(d["bias"])
        r.load_state_dict(d["rule"]);opt.load_state_dict(d["optimizer"]);step=d["step"];trace=d["trace"].tolist();counts=d["counts"]
    def save():
        if step in STEPS:
            for i,b in enumerate(ids):old.save_checkpoint(root,"odor",arm,b,step,g,r,i)
        d=dict(step=step,plan=digest,weights=g.weights.detach(),bias=g.bias.detach(),present=g.present,
               rule=r.state_dict(),optimizer=opt.state_dict(),trace=torch.tensor(trace),counts=counts)
        bank.lesson.storage._atomic(resume,lambda f:torch.save(d,f))
    if not resume.exists():save()
    first=step;t0=perf_counter()
    for step in range(first+1,6001):
        mode=schedule(seed,step);counts[mode]+=1
        if mode=="review":
            packet=review_tape(seed,ids,step);worlds=4
        elif mode=="short":
            packet=archive.join([archive.short_ticket(seed,b,step) for b in ids]);worlds=8
        else:
            packet=archive.join([archive.prior.ticket(seed,b,step,4,"interspersed",domain=386717) for b in ids]);worlds=4
        reward,norm,fallback=update(g,r,opt,critic,packet,mode,strength(arm,step),worlds)
        trace.append(torch.stack((reward,norm,torch.full_like(norm,fallback)),1).detach().tolist())
        if step%250==0:
            save();print(arm,cohort,step,counts,round((perf_counter()-t0)/(step-first),3),flush=True)
    if counts!={"review":3000,"short":1500,"mixed":1500}:raise ValueError("Training mixture differed")


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--cohort",choices=w.COHORTS,required=True)
    p.add_argument("--arm",choices=ARMS,required=True);a=p.parse_args();fit(a.cohort,a.arm)
