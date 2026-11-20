"""Teach the primary organism's outgoing axons using only its own reported forecast error."""

import argparse
from dataclasses import fields, replace

import numpy as np
import torch
from torch.nn import functional as F

from . import fork_lesson_fit as baseline
from . import fork_lesson_world as world
from . import noon_market_stats as stats


DATA=baseline.DATA.parent/"sender"
EVIDENCE=baseline.EVIDENCE
PLAN=EVIDENCE/"sender-plan.md"
KINDS=("whole","cut_out")
ARMS=("true","sham")
STEPS=768
SEEDS=(7025311,7025383)
ROOTS=128


def path(kind,birth,arm):
    return DATA/kind/f"{birth}-{arm}.pt"


def load_parent(kind,birth):
    model,_,sha=baseline.load_final(kind,birth)
    return model.requires_grad_(False),sha


def train(kind,birth,arm):
    torch.set_num_threads(1)
    model,parent_sha=load_parent(kind,birth)
    model.return_weight.requires_grad_(True)
    old=model.return_weight.detach().clone()
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.Adam([model.return_weight],lr=.006)
    log=[]
    for step in range(1,STEPS+1):
        e=world.scenes([6238511,birth,step],64)
        # Deployment and fit see the same changed axons throughout. Credit for the
        # prediction cannot improve the stored history by backpropagating into support.
        with torch.no_grad():
            seen=world.cue(model,world.support(model,e),e)
        final,activity,_=world.choice_window(model,seen)
        answers=baseline.forecast_scores(activity)
        with torch.no_grad():
            action=world.choose(model,final,e,exploration=.25)
            _,outcome=world.commit(model,final,e,action)
            target=outcome.payoff if arm=="true" else outcome.payoff.roll(-1)
        chosen=answers.gather(1,action.side[:,None]).squeeze(1)
        loss=F.mse_loss(chosen,target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_([model.return_weight],2.,error_if_nonfinite=True)
        optimizer.step()
        log.append(torch.stack((e.draws,.75*action.logit.sigmoid()+.125,action.side,
                                outcome.payoff,target,chosen.detach(),grad.expand_as(chosen)),1))
    masks=model.mask[20:,:20]
    assert torch.equal(old[~masks],model.return_weight[~masks])
    for name,value in model.state_dict().items():
        if name!="return_weight":torch.testing.assert_close(value,initial[name],atol=0,rtol=0)
    ledger=torch.stack(log)
    draw,prob,actual,earned,wanted=ledger[:,:,:5].unbind(2)
    assert torch.equal(draw<prob,actual.bool())
    assert torch.equal(wanted,earned if arm=="true" else earned.roll(-1,dims=1))
    delta=model.return_weight.detach()-old
    output=dict(body=model.state_dict(),ledger=ledger,parent=parent_sha,source=baseline.digest(__file__),
                plan=baseline.digest(PLAN),step=STEPS,birth=birth,arm=arm,kind=kind,
                changed=int(delta.ne(0).sum()),possible=int(masks.sum()),max_change=float(delta.abs().max()))
    baseline.storage._atomic(path(kind,birth,arm),lambda tmp:torch.save(output,tmp))
    print("fit",kind,birth,arm,output["changed"],flush=True)


def factorial(seed):
    src=world.scenes(seed,ROOTS)
    cells=np.indices((2,2,2,2,2,2)).reshape(6,-1).T
    lu,ru,order,lmul,rmul,first=[torch.tensor(np.tile(cells[:,i],ROOTS)) for i in range(6)]
    e=world.Episode(**{field.name:getattr(src,field.name).repeat_interleave(64,0) for field in fields(src)})
    return replace(e,values=torch.stack((lu,ru),1).float()*2-1,door_order=torch.stack((order,1-order),1),
                   flips=torch.stack((lmul,rmul),1).float()*2-1,first=first)


@torch.no_grad()
def evaluate(kind,birth):
    torch.set_num_threads(1)
    parent,base_sha=load_parent(kind,birth)
    models={"start":parent}
    shas={"start":base_sha}
    for arm in ARMS:
        f=path(kind,birth,arm)
        saved=torch.load(f,weights_only=True,map_location="cpu")
        if saved["parent"]!=base_sha or saved["plan"]!=baseline.digest(PLAN) or saved["step"]!=STEPS:
            raise ValueError(f"Unverified fit {f}")
        model=world.ForkGraph(birth).requires_grad_(False)
        model.load_state_dict(saved["body"])
        models[arm]=model;shas[arm]=baseline.digest(f)
    for seed in SEEDS:
        all_e=factorial(seed);raw={}
        for i in range(0,len(all_e.first),256):
            e=world.Episode(**{f.name:getattr(all_e,f.name)[i:i+256] for f in fields(all_e)})
            for tag,model in models.items():
                prefix=world.cue(model,world.support(model,e),e)
                stages=(("natural",model),("original_sender",parent)) if tag!="start" else (("natural",model),)
                for condition,run in stages:
                    ready,hs,_=world.choice_window(run,prefix)
                    prediction=baseline.forecast_scores(hs)
                    act=world.choose(run,ready,e)
                    _,paid=world.commit(run,ready,e,act)
                    # Complete each route at evaluation only, using individual physical threshold and waiting ticks.
                    endpoints=[]
                    for j in range(2):
                        forced=world.choose(run,ready,e,side=j)
                        _,got=world.commit(run,ready,e,forced)
                        endpoints.append(got.payoff)
                    truth=torch.stack(endpoints,1)
                    assert torch.equal(paid.payoff,truth.gather(1,act.side[:,None])[:,0])
                    name=f"{tag}/{condition}"
                    for label,value in (("forecast",prediction),("side",act.side),("pay",paid.payoff),("truth",truth)):
                        raw.setdefault(f"{name}/{label}",[]).append(value.numpy())
        baseline.storage._write_raw(DATA/"eval"/str(seed)/f"{kind}-{birth}.npz",
                                    **{k:np.concatenate(v) for k,v in raw.items()},seed=seed,roots=ROOTS,
                                    plan=baseline.digest(PLAN),**{"sha_"+k:v for k,v in shas.items()})


def summarize():
    out={}
    for kind in KINDS:
        out[kind]={}
        for seed in SEEDS:
            groups={};raw_sha={}
            for birth in baseline.BIRTHS:
                p=DATA/"eval"/str(seed)/f"{kind}-{birth}.npz"
                raw_sha[p.name]=baseline.digest(p)
                with np.load(p) as z:
                    for tag in ("start/natural","true/natural","sham/natural","true/original_sender","sham/original_sender"):
                        y=z[tag+"/truth"];q=z[tag+"/forecast"];side=z[tag+"/side"];paid=z[tag+"/pay"]
                        strict=y[:,0]!=y[:,1];row=np.arange(len(y))
                        np.testing.assert_array_equal(y[row,side],paid)
                        loss=(y-q)**2
                        readings={"forecast":-loss.mean(1).reshape(ROOTS,64).mean(1),
                                  "unchosen_forecast":-loss[row,1-side].reshape(ROOTS,64).mean(1),
                                  "strict_preferred":(paid[strict]>0).reshape(ROOTS,32).mean(1),
                                  "net":paid.reshape(ROOTS,64).mean(1)}
                        group=groups.setdefault(tag,{})
                        for k,v in readings.items():group.setdefault(k,[]).append(v)
            banks={g:{k:np.stack(v) for k,v in data.items()} for g,data in groups.items()}
            comparisons=(("true/natural","sham/natural"),("true/natural","start/natural"),
                         ("true/natural","true/original_sender"),("sham/natural","sham/original_sender"))
            out[kind][str(seed)]=dict(scores=stats.summarize(banks,ROOTS,baseline.BIRTHS,seed+3,comparisons),
                                     raw_sha256=raw_sha)
    baseline.storage._write_json(EVIDENCE/"sender-results.json",dict(reports=out,plan=baseline.digest(PLAN),
        source=baseline.digest(__file__),metric="forecast metrics are negative MSE; higher is better; compare true minus control"))
    for kind,data in out.items():
        for seed,report in data.items():
            m=report["scores"]["true/natural minus sham/natural"]
            print(kind,seed,{k:round(v["mean"],4) for k,v in m.items()})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--kind",choices=KINDS);p.add_argument("--birth",type=int)
    p.add_argument("--arm",choices=ARMS);args=p.parse_args()
    if args.arm:train(args.kind,args.birth,args.arm)
    elif args.kind:evaluate(args.kind,args.birth)
    else:summarize()
