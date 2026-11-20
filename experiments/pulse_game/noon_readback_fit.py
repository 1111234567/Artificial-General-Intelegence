"""Fit a shared, low-dimensional review policy from observed action returns."""

from argparse import ArgumentParser
from hashlib import sha256
import json

import numpy as np
import torch
from torch import nn

from . import noon_readback as bank
from .interleave_capacity_report import Crossed


PLAN=bank.world.EVIDENCE/"readback-fit-plan.json"
MODES=("logged","paired","menu_only")


class Readout(nn.Module):
    def __init__(self,mode):
        super().__init__()
        generator=torch.Generator().manual_seed(661837)
        self.mode=mode
        self.gain=nn.Parameter(torch.zeros((),dtype=torch.float64))
        if mode!="menu_only":
            self.w=nn.Parameter(torch.randn(4,2,generator=generator,dtype=torch.float64)*1.2)
            self.b=nn.Parameter(torch.randn(4,generator=generator,dtype=torch.float64)*.65)
            self.v=nn.Parameter(torch.zeros(4,dtype=torch.float64))

    def scores(self,z):
        return torch.tanh(z@self.w.T/5+self.b)@self.v

    def forward(self,z,menu):
        logit=self.gain*menu
        if self.mode!="menu_only":
            scores=self.scores(z)
            logit=logit+scores[...,1]-scores[...,0]
        return logit.sigmoid()


def load(split,family,count):
    path=bank.DATA/split/family/str(count);info=json.loads((path/"index.json").read_text())
    rows=[]
    for b in info["births"]:
        f=path/f"{b}.npz"
        if sha256(f.read_bytes()).hexdigest()!=info["hashes"][str(b)]["trace"]:raise ValueError("Readout trace changed")
        with np.load(f) as z:
            rows.append({k:z[k] for k in z.files if k not in ("checkpoint","source","seed")})
    return rows,info


def fit():
    torch.set_num_threads(1);rows,manifest=load("train","uniform",4)
    z=torch.tensor(np.concatenate([r["z"] for r in rows]),dtype=torch.double)
    native=torch.tensor(np.concatenate([r["menu"] for r in rows]),dtype=torch.double)
    pay=torch.tensor(np.concatenate([r["paid"] for r in rows]),dtype=torch.double)
    logged=torch.tensor(np.concatenate([r["chosen"] for r in rows]),dtype=torch.long)
    observed=pay[torch.arange(len(pay)),logged]
    approx=2*(2*logged-1)*(observed-observed.mean())
    weights={};trace={}
    for mode in MODES:
        model=Readout(mode)
        optimizer=torch.optim.Adam(model.parameters(),lr=.025)
        y=pay[:,1]-pay[:,0] if mode=="paired" else approx
        record=[]
        for i in range(1200):
            optimizer.zero_grad(set_to_none=True)
            p=model(z,native)
            tiny=1e-12
            kl=(p*((2*p).clamp_min(tiny)).log()+(1-p)*(2*(1-p)).clamp_min(tiny).log()).mean()
            norm=torch.cat([v.reshape(-1) for v in model.parameters()]).square().mean()
            gain=((p-.5)*y).mean()
            loss=-gain+.002*kl+.0001*norm
            loss.backward()
            optimizer.step()
            if i%100==99:record.append([float(gain.detach()),float(loss.detach())])
        weights[mode]={k:v.detach().tolist() for k,v in model.state_dict().items()}
        trace[mode]=record
    design=dict(plan_sha256=sha256(PLAN.read_bytes()).hexdigest(),train_source_sha256=manifest["hashes"],
                weights=weights,trace=trace,n_subjects=len(rows),new_worlds=manifest["worlds"])
    bank.lesson.storage._write_json(bank.DATA/"readout.json",design)
    print("fit", {k:t[-1] for k,t in trace.items()},flush=True)


def restored(data,mode):
    m=Readout(mode)
    m.load_state_dict({k:torch.tensor(v,dtype=torch.double) for k,v in data["weights"][mode].items()})
    return m


def computations(d,models):
    z=torch.tensor(d["z"],dtype=torch.double);n=torch.tensor(d["menu"],dtype=torch.double)
    u=d["u"]
    with torch.no_grad():
        ps={k:model(z,n).numpy() for k,model in models.items()}
        weak=(z[:,:,0]-z[:,:,1]).abs().numpy()/2
        ps.update(native=n.sigmoid().numpy(),reverse_native=(-n).sigmoid().numpy(),
                  weak=np.where(weak[:,1]==weak[:,0],.5,(weak[:,1]<weak[:,0]).astype(float)),
                  random=np.full(len(z),.5))
        read_scores=models["logged"].scores(z).numpy()
        # Use a donor's actual motor observations, at the same geometric world,
        # candidate positions, public clock and random numbers. Keep both features
        # for each item paired; the private contents were independently crossed.
        cross={}
        for tag in ("readings","menu","all"):
            donors=[]
            for shift in range(8):
                ix=(np.arange(len(z))//8)*8+(np.arange(len(z))%8+shift)%8
                donors.append(models["logged"](z[ix] if tag in ("all","readings") else z,
                                              n[ix] if tag in ("all","menu") else n).numpy())
            cross[tag]=np.stack(donors)
    ok=d["correct"];row=np.arange(len(z))
    offered=np.take_along_axis(ok,np.broadcast_to(d["candidates"][:,None,:],(len(z),2,2)),axis=2)
    vals={"all":ok.all(2),"mean":ok.mean(2),"paid":d["paid"],"both":offered.all(2),
          "mean_expected_terminal":d["future_expected"].mean(2)}
    metrics={}
    for name,value in vals.items():
        left=value[:,0].astype(float);diff=value[:,1].astype(float)-left
        uniform=left+.5*diff
        for m,p in ps.items():
            select=(u<p).astype(int)
            sample=value[row,select].astype(float)
            if m=="random":sample=uniform   # exact expectation over an independent fair route
            metrics[f"{m}/{name}"]=sample
            metrics[f"{m}/minus_random/{name}"]=sample-uniform
            metrics[f"{m}/expected_route_minus_random/{name}"]=(p-.5)*diff
            if m=="logged":
                for key,y in cross.items():
                    donor=(u[None,:]<y).mean(0)
                    metrics[f"logged/minus_donor_{key}/{name}"]=sample-(left+donor*diff)
        for other in ("weak","menu_only","native","paired"):
            metrics[f"logged/minus_{other}/{name}"]=metrics["logged/"+name]-metrics[other+"/"+name]
    for name,p in ps.items():
        metrics[f"{name}/right"]=(u<p).astype(float)
        metrics[f"{name}/chose_weaker"]=np.where(u<p,weak[:,1]<=weak[:,0],weak[:,0]<=weak[:,1]).astype(float)
    return {k:v.reshape(-1,8).mean(1) for k,v in metrics.items()},dict(
        learned=ps["logged"],native=ps["native"],learned_scores=read_scores,difference_confidence=weak[:,1]-weak[:,0])


def evaluate(split,family,count):
    torch.set_num_threads(1)
    trained=json.loads((bank.DATA/"readout.json").read_text())
    models={m:restored(trained,m) for m in MODES}
    rows,manifest=load(split,family,count)
    stats={};hashes={};dest=bank.DATA/"scores"/split/family/str(count);dest.mkdir(parents=True,exist_ok=True)
    for b,r in zip(manifest["births"],rows,strict=True):
        values,raw=computations(r,models)
        for k,v in values.items():stats.setdefault(k,[]).append(v)
        f=dest/f"{b}.npz"
        bank.lesson.storage._write_raw(f,**raw,**{"metric/"+k:v for k,v in values.items()})
        hashes[str(b)]=sha256(f.read_bytes()).hexdigest()
    ci=Crossed(len(rows),manifest["worlds"],manifest["seed"]+count*41,3500)
    out=dict(split=split,family=family,count=count,births=manifest["births"],worlds=manifest["worlds"],
             trained_weights_sha256=sha256((bank.DATA/"readout.json").read_bytes()).hexdigest(),weights=trained["weights"],
             source_trace_sha256=manifest["hashes"],world_scores_sha256=hashes,
             scores={k:ci(np.stack(v)) for k,v in stats.items()},definition="A terminal motor choice is sampled "
                    "for each object from the same post-study snapshot. The paid column is the single actual final "
                    "query; mean and all probe the full set. Each model gets one actual review and the same 20 "
                    "state-changing preview ticks. Random is integrated over the two alternatives. Donor-only "
                    "shuffles retain the actual geometric world and exchange privately learned readings.")
    bank.lesson.storage._write_json(bank.world.EVIDENCE/f"readback-{split}-{family}-{count}.json",out)
    print(split,family,count,round(out["scores"]["logged/minus_random/all"]["mean"],4),flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--fit",action="store_true");p.add_argument("--split")
    p.add_argument("--family",default="uniform");p.add_argument("--count",type=int,default=4)
    a=p.parse_args()
    if a.fit:fit()
    if a.split:evaluate(a.split,a.family,a.count)
