"""Separate study-route and food-collection motor readouts without an answer or uncertainty input."""

from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import json

import torch
from torch.nn import functional as F

from . import noon_review_world as w, noon_review_train as core
from . import local_trace_graph as net, local_trace_train as old


ARMS=("reserved","review","shared")
OUT=w.DATA/"motor-roles";PLAN=w.EVIDENCE/"motor-roles-plan.json"
STAGES=(0,1000,4000,8000)


def action(initial,draw,head,role,arm,*,override=None,lesion=False):
    h=initial.state.hidden
    if arm=="reserved" and role=="review" and not lesion:
        logit=2.5*(h[:,19]-h[:,18])
    elif arm in ("review","shared") and (arm=="shared" or role=="review") and not lesion:
        logit=initial.motor_logit+(h*head[:,:-1]).sum(1)+head[:,-1]
    else:
        logit=initial.motor_logit
    side=2*(draw<logit.detach().sigmoid()).to(logit.dtype)-1 if override is None else override
    return net.Action(initial.state,side,logit,F.logsigmoid(side*logit))


def collect(rule,state,left,right,truth,draw,head,arm,*,write=True):
    a=action(net.act(rule,state,left,right,draw[:,0]),draw[:,0],head,"collect",arm)
    earned=a.side*truth
    e=net.observe(rule,a,left,right,earned)
    edit=net.commit(rule,e,draw[:,1:],intervention="active" if write else "sham")
    return w.Contact(net.wait(rule,edit.state,w.GAP),a,earned,edit)


def start(g,r,t,head,arm):
    mult=t.lanes//len(g.bias);view=r.expand(mult)
    weight=head.repeat_interleave(mult,0);state=net.start(g,mult);contacts=[]
    for i in range(t.count):
        sign=t.study_orientation[:,i];left=t.odor[:,i]*sign[:,None]
        event=collect(view,state,left,-left,t.law[:,i]*sign,t.study_draw[:,i],weight,arm)
        state=event.state;contacts.append(event)
    return view,weight,state,contacts


def select(view,state,t,weight,arm,*,lesion=False):
    initial=w.select(view,state,*w.candidate_menu(t),t.decision_draw)
    return action(initial,t.decision_draw,weight,"review",arm,lesion=lesion)


def review(view,state,t,side,weight,arm,*,write=True):
    echoed=w.echo_choice(view,state,side);item=w.selected_item(t,side)
    l,right=w.sensory_for(t,item,t.review_orientation)
    return collect(view,echoed,l,right,t.law[w._rows(t),item]*t.review_orientation,
                   t.review_draw,weight,arm,write=write)


def query(view,state,t,item,weight,arm):
    row=w._rows(t);sg=t.final_orientation[row,item]
    l,right=w.sensory_for(t,item,sg)
    initial=net.act(view,state,l,right,t.final_draw[row,item])
    return action(initial,t.final_draw[row,item],weight,"collect",arm),t.law[row,item]*sg


def objective(g,r,head,t,arm,retention,cfg):
    view,weight,state,contacts=start(g,r,t,head,arm)
    score=torch.zeros(t.lanes,dtype=g.bias.dtype);right=score
    if not retention:
        a=select(view,state,t,weight,arm);score=a.motor_log_prob
        right=(a.side>0).float()
        ev=review(view,a.state,t,a.side,weight,arm);contacts.append(ev);state=ev.state
    last,target=query(view,state,t,t.final_item,weight,arm)
    shape=(len(g.bias),cfg.pairs,2)
    reward=(last.side*target).detach().reshape(shape)
    advantage=old.leave_pair_out_advantage(reward)
    collection=sum(c.motor.motor_log_prob for c in contacts);editing=sum(c.edit.edit_log_prob for c in contacts)
    loss=sum(old.score_loss(s.reshape(shape),advantage) for s in (last.motor_log_prob,collection,editing,score))
    stats=torch.stack(((reward>0).float().mean((1,2)),right.reshape(shape).mean((1,2))),1)
    return loss.sum(),stats.detach()


def update(g,r,head,opt,t,arm,retention,cfg):
    params=(g.weights,g.bias,*r.parameters(),head)
    for precise in (False,True):
        opt.zero_grad(set_to_none=True)
        if precise:
            m=net.Graph(torch.nn.Parameter(g.weights.detach().double()),torch.nn.Parameter(g.bias.detach().double()),g.present)
            rule=deepcopy(r).double();h=torch.nn.Parameter(head.detach().double())
        else:m,rule,h=g,r,head
        loss,metrics=objective(m,rule,h,t,arm,retention,cfg)
        if not loss.isfinite():continue
        loss.backward()
        grads=(m.weights,m.bias,*rule.parameters(),h)
        norm=core.archive._gradients(grads,cfg.grad_clip)
        if norm is None:continue
        if precise:
            for to,orig in zip(params,grads,strict=True):to.grad=None if orig.grad is None else orig.grad.to(to.dtype)
        opt.step()
        return torch.cat((metrics,norm[:,None],torch.full_like(norm[:,None],precise)),1)
    raise FloatingPointError("Nonfinite motor role fit")


def fit(cohort,arm):
    torch.set_num_threads(1);ids=list(w.COHORTS[cohort]);cfg=core.Config()
    p=OUT/cohort;d=p/arm;d.mkdir(parents=True,exist_ok=True)
    digest=lambda f:sha256(Path(f).read_bytes()).hexdigest()
    info=dict(cohort=cohort,arm=arm,births=ids,steps=STAGES[-1],lr=cfg.lr,worlds=cfg.pairs,
              plan_sha256=digest(PLAN),parents={str(b):digest(old.checkpoint_path(w.parent_path(cohort),"odor","uniform",b,8000)) for b in ids},
              sources={Path(f).name:digest(f) for f in (__file__,w.__file__,core.__file__,net.__file__,net.native.__file__)})
    file=d/"run.json"
    if file.exists() and json.loads(file.read_text())!=info:raise ValueError("Changed study motor protocol")
    if not file.exists():core.archive._write_json(file,info)
    key=digest(file);g,r=core.load_cohort(cohort)
    head=torch.nn.Parameter(torch.zeros(len(ids),net.NODES+1),requires_grad=arm!="reserved")
    opt=torch.optim.Adam((g.weights,g.bias,*r.parameters(),head),lr=cfg.lr)
    resume=d/"resume.pt";step=0;trace=[]
    if resume.exists():
        data=torch.load(resume,map_location="cpu",weights_only=True)
        if data["plan"]!=key:raise ValueError("Unexpected motor role checkpoint")
        with torch.no_grad():
            g.weights.copy_(data["weights"]);g.bias.copy_(data["bias"]);head.copy_(data["head"])
        r.load_state_dict(data["rule"]);opt.load_state_dict(data["optimizer"]);step=data["step"];trace=data["trace"].tolist()
    def save():
        if step in STAGES:
            for i,b in enumerate(ids):
                old.save_checkpoint(p,"odor",arm,b,step,g,r,i)
            core.archive._write_raw(d/f"head-{step}.npz",weights=head.detach().numpy(),births=ids)
        data=dict(step=step,plan=key,weights=g.weights.detach(),bias=g.bias.detach(),head=head.detach(),
                  present=g.present,rule=r.state_dict(),optimizer=opt.state_dict(),trace=torch.as_tensor(trace))
        core.archive._atomic(resume,lambda dst:torch.save(data,dst))
    if not resume.exists():save()
    beg=step;t0=perf_counter()
    for step in range(beg+1,STAGES[-1]+1):
        retention,count=core.practice_step(step,cfg)
        t=w.make_tape(w.TRAIN_SEED,cfg.pairs,count,births=ids,update=step)
        trace.append(update(g,r,head,opt,t,arm,retention,cfg).tolist())
        if step%250==0:
            save();print(cohort,arm,step,round((perf_counter()-t0)/(step-beg),3),flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--cohort",choices=w.COHORTS,required=True)
    p.add_argument("--arm",choices=ARMS,required=True)
    fit(**vars(p.parse_args()))
