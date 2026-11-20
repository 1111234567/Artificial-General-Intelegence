"""Does review accumulate several noisy resource receipts about the same object?"""

from argparse import ArgumentParser
from hashlib import sha256
from math import comb
from pathlib import Path
import json

import numpy as np
import torch

from . import interleave_age_train as age, interleave_stream as stream
from .interleave_interference import DATA, EVIDENCE, ROOT


VISITS=(0,1,2,3,4,5,8,16,32)
SLIPS=(0.,.2,.35)
ORDERS={"early_correct_3":(1,1,-1),"late_correct_3":(-1,1,1),
        "interleaved_3":(1,-1,1),"early_correct_7":(1,1,1,1,-1,-1,-1),
        "late_correct_7":(-1,-1,-1,1,1,1,1),"interleaved_7":(1,-1,1,-1,1,-1,1)}


def optimal(n,p):
    """Stationary antipodal tastes with independent slips and equal prior on the two laws."""
    return sum((1 if j>n/2 else .5 if j==n/2 else 0)*comb(n,j)*(1-p)**j*p**(n-j) for j in range(n+1))


def taps(seed,birth,worlds,length):
    rng=np.random.default_rng([373344,seed,birth])
    def twin(x):
        return torch.from_numpy(np.repeat(x,2,axis=0)).float()
    return dict(odor=twin(age.directions(rng,worlds)),
                order=twin(age.signs(rng,(worlds,length+1))),
                uniform=twin(rng.random((worlds,length+1,6),dtype=np.float32)),
                slips=twin(rng.random((worlds,length),dtype=np.float32)),
                law=torch.tensor([-1.,1.]).repeat(worlds))


@torch.no_grad()
def physical(g,rule,w,slips,visits,*,mode="active"):
    lanes=len(w["law"])
    state=stream.start(g,rule,lanes)
    coeff=None if rule is None else rule.expand(lanes)
    out={k:[] for k in ("correct","expected","side","public_sum","last_evidence_direction")}
    public_sum=torch.zeros(lanes)
    last=torch.zeros_like(public_sum)
    # Define a visible canonical side by the first presented mark vector, not by its value.
    first_visible=w["odor"]*w["order"][:,0,None]
    for t in range(max(visits)+1):
        if t in visits:
            right=w["odor"]*w["order"][:,-1,None]
            a=stream.action(coeff,state,-right,right,w["uniform"][:,-1,0])
            truth=w["law"]*w["order"][:,-1]
            for k,value in (("side",a.side.to(torch.int8)),("correct",a.side==truth),
                            ("expected",(a.motor_logit*truth).sigmoid()),("public_sum",public_sum),
                            ("last_evidence_direction",last)):
                out[k].append(value.clone().numpy())
        if t==max(visits):
            continue
        right=w["odor"]*w["order"][:,t,None]
        factor=w["law"]*w["order"][:,t]*slips[:,t]
        draw=w["uniform"][:,t]
        state,side,earned=stream.contact(coeff,state,-right,right,factor,torch.zeros_like(factor),draw,mode)
        # Only the sampled side, actual earned receipt and visible geometry enter this reference.
        contact_item=right*side[:,None]
        last=earned*torch.sign((contact_item*first_visible).sum(1))
        public_sum=public_sum+last
        state=stream.wait(coeff,state,4)
    out={k:np.stack(v) for k,v in out.items()}
    canonical_to_probe=(w["order"][:,0]*w["order"][:,-1]).numpy()
    goal=(w["law"]*w["order"][:,-1]).numpy()
    out.update(canonical_to_probe=canonical_to_probe,goal=goal)
    return out


def cell(out,idx):
    good=out["correct"][idx]
    evidence=out["public_sum"][idx]
    last=out["last_evidence_direction"][idx]
    preferred=np.sign(evidence)*out["canonical_to_probe"]
    latest=last*out["canonical_to_probe"]
    conflict=(preferred!=latest)&(evidence!=0)&(last!=0)
    majority=(preferred==out["goal"]).astype(float)
    majority[evidence==0]=.5
    return dict(actual=float(good.mean()),expected=float(out["expected"][idx].mean()),
                observed_majority=float(majority.mean()),cases=len(good),
                majority_last_disagree=int(conflict.sum()),
                followed_majority=int((out["side"][idx,conflict]==preferred[conflict]).sum()),
                sum_zero=int((evidence==0).sum()))


def run(source,label,births,seed,worlds):
    torch.set_num_threads(1)
    dest=DATA/"receipts"/label
    dest.mkdir(parents=True,exist_ok=True)
    report=dict(seed=seed,births=births,worlds_per_birth=worlds,
                visits=VISITS,slip_rates=SLIPS,orders=ORDERS,results={},sha256={})
    for birth in births:
        g,r=source.load(birth)
        w=taps(seed,birth,worlds,max(VISITS))
        metrics,raw={},{}
        for p in SLIPS:
            slip=torch.where(w["slips"]<p,-1.,1.)
            row=physical(g,r,w,slip,VISITS)
            metrics[str(p)]={n:{**cell(row,i),"stationary_optimum":optimal(n,p)} for i,n in enumerate(VISITS)}
            raw.update({f"{p}/{k}":v for k,v in row.items()})
        for name,sequence in ORDERS.items():
            length=len(sequence)
            changes=torch.tensor(sequence).expand(len(w["law"]),length)
            row=physical(g,r,w,changes,(length,))
            metrics[name]=cell(row,0)
            raw.update({f"{name}/{k}":v for k,v in row.items()})
        path=dest/f"{birth}.npz"
        np.savez_compressed(path,**raw)
        report["sha256"][path.name]=sha256(path.read_bytes()).hexdigest()
        report["results"][birth]=metrics
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/f"receipts-{label}.json").write_text(json.dumps(report,indent=1)+"\n",encoding="utf8")
    print(label,"complete",flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--label",required=True)
    p.add_argument("--family",choices=("joint","lifetime","objective","odor","feeder"),required=True)
    p.add_argument("--run",type=Path,default=ROOT/"data/local-trace-20261119/local")
    p.add_argument("--arm",default="active")
    p.add_argument("--step",type=int,default=8000)
    p.add_argument("--worlds",type=int,default=128)
    p.add_argument("--seed",type=int,default=978721)
    p.add_argument("--births",nargs="+",type=int,default=list(range(731,737)))
    a=p.parse_args()
    run(stream.Source(a.family,a.run,a.arm,a.step),a.label,a.births,a.seed,a.worlds)
