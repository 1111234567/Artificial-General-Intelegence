"""Remove one actual parameter transaction and tally downstream actions under common randomness."""
from pathlib import Path

import numpy as np
import torch

from . import self_dose_continuous as c

fw=c.fw


def worlds(root):
    n=len(root)
    rngs=[np.random.default_rng([r,82139]) for r in root]
    cues=[]; sides=[]; slips=[]; draws=[]
    for rng in rngs:
        odor=rng.normal(size=(16,3));odor=1.35*odor/np.linalg.norm(odor,axis=1,keepdims=True)
        orientation=rng.choice((-1,1),size=64)
        pref=rng.choice((-1,1),size=16)
        cues.append(odor.repeat(4,axis=0)*orientation[:,None])
        sides.append(pref.repeat(4)*orientation)
        slips.append(np.where(rng.random(64)<.1,-1,1))
        draws.append(rng.random((64,2),dtype=np.float32))
    x=torch.tensor(np.repeat(cues,2,axis=0),dtype=torch.float32)
    best=torch.tensor(np.repeat(sides,2,axis=0),dtype=torch.float32);best[1::2]*=-1
    return x,best,torch.tensor(np.repeat(slips,2,axis=0)),torch.from_numpy(np.repeat(draws,2,axis=0))


def step(graph,h,environment,t):
    x,best,noise,draws=environment
    left=x[:,t];right=-left
    choice=fw.advance(graph,h,fw.sensory(graph,left,right),5)
    logit=2.5*(choice[:,11]-choice[:,10]);side=(logit>0).float()*2-1
    taste=side*best[:,t]*noise[:,t]
    pre=fw.advance(graph,choice,fw.sensory(graph,left,right,side),3)
    post=fw.advance(graph,pre,fw.sensory(graph,left,right,side,taste),3)
    p=fw.propose(graph,choice,left,right,side,taste,draws[:,t],value_limit=None)
    activity=c.inspection(graph,post,p.command.address)
    return p,activity,side,logit


@torch.no_grad()
def run_one(root,environment,plan):
    with np.load(c.paths(root,"fresh","all"),allow_pickle=False) as z:
        graph=fw.Graph(*(torch.from_numpy(z[k]) for k in ("w","b","mask")))
    graph=graph.repeat(len(environment[0]))
    h=torch.zeros_like(graph.bias);records=[]
    for t in range(environment[0].shape[1]):
        before=graph
        p,h,_,_=step(graph,h,environment,t)
        graph=p.next_graph
        if t not in plan["forks"]: continue
        actual,blank=graph,before
        ha=hb=h
        changed=torch.zeros(len(h));direction=torch.zeros_like(changed);maxdiff=torch.zeros_like(changed)
        for offset in range(1,1+plan["horizon"]):
            at=t+offset;gold=environment[1][:,at]
            aa,ha,sa,la=step(actual,ha,environment,at);actual=aa.next_graph
            bb,hb,sb,lb=step(blank,hb,environment,at);blank=bb.next_graph
            changed+=(sa!=sb)
            direction+=(sa==gold).float()-(sb==gold).float()
            maxdiff=torch.maximum(maxdiff,(la-lb).abs())
        edge,receiver,sender=fw.decode(p.command.address,graph.width)
        incident=(receiver>=fw.BODY)|(edge&(sender>=fw.BODY))
        accepted=p.receipt.status==fw.Status.APPLIED
        assert not (changed[~accepted]!=0).any()
        assert not (maxdiff[~accepted]!=0).any()
        records.append(torch.stack((accepted,incident,p.command.address,p.command.op,p.receipt.status,changed,direction,maxdiff),1).numpy())
    return np.stack(records)


def run():
    plan=c.io.read(c.EVIDENCE/"fork-plan.json");torch.set_num_threads(1)
    env=worlds(plan["world_roots"])
    data=np.stack([run_one(s,env,plan) for s in c.REG["births"]])
    updated=data[...,0]!=0;incident=data[...,1]!=0; different=data[...,5]>0;net=data[...,6]
    by_scope={}
    for key,mask in (("all",np.ones_like(updated)),("committed",updated),("writer_incident",updated&incident),
                     ("body_only",updated&~incident)):
        by_scope[key]=dict(count=int(mask.sum()), changed_future_choice=int(different[mask].sum()),
             fraction_changed=float(different[mask].mean()) if mask.any() else None,
             more_correct=int((net[mask]>0).sum()),fewer_correct=int((net[mask]<0).sum()),
             tied_total=int((net[mask]==0).sum()),nonzero_logit_effect=int((data[...,7][mask]>1e-6).sum()))
    archive=c.EVIDENCE/"fork-traces.zip"
    from io import BytesIO
    from zipfile import ZipFile, ZIP_DEFLATED
    with ZipFile(archive,"w",ZIP_DEFLATED) as z:
        buf=BytesIO();np.save(buf,data,allow_pickle=False);z.writestr("records.npy",buf.getvalue())
    c.io.write(c.EVIDENCE/"fork-results.json",dict(scopes=by_scope,axis=["birth","fork","world×2 complementary values","fields"],
        fields=["committed","writer_incident","address","operation","status","different_future_sides","net_more_correct","max_logit_disagreement"],
        protocol_sha256=c.io.sha(c.EVIDENCE/"fork-plan.json"),code_sha256=c.io.sha(Path(__file__)),raw_sha256=c.io.sha(archive)))
    print(by_scope)


if __name__=="__main__":run()
