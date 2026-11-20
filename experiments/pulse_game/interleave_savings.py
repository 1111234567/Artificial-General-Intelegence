"""Compare A relearning after matching prior A, opposite A, or an unrelated prior object."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_age_train as age, interleave_stream as stream
from .interleave_interference import DATA,EVIDENCE,ROOT


GAPS=(0,1,4,16,64)
REVIEWS=(0,1,2,4)
HISTORIES=("true_a","opposite_a","unrelated_c")


def make(seed,birth,worlds):
    rng=np.random.default_rng([3962047,seed,birth])
    def copy(x):return torch.as_tensor(np.repeat(x,4,axis=0))
    x=copy(age.directions(rng,worlds,max(GAPS)+2))
    bits=np.repeat(age.signs(rng,(worlds,max(GAPS)+2)),4,axis=0)
    bits[:,0]=np.tile([-1,1,-1,1],worlds)
    bits[:,2:]*=np.tile([1,1,-1,-1],worlds)[:,None]
    steps=1+max(GAPS)+max(REVIEWS)
    return dict(x=x,law=torch.from_numpy(bits),orient=copy(age.signs(rng,(worlds,steps))),
                u=copy(rng.random((worlds,steps,6),dtype=np.float32)),probe=copy(age.signs(rng,(worlds,2))),
                probe_u=copy(rng.random((worlds,2),dtype=np.float32)))


@torch.no_grad()
def evaluate(g,rule,w):
    n=len(w["x"])
    r=None if rule is None else rule.expand(n)
    def experience(s,obj,law,event):
        flip=w["orient"][:,event]
        right=w["x"][:,obj]*flip[:,None]
        s,act,earned=stream.contact(r,s,-right,right,law*flip,torch.zeros_like(flip),w["u"][:,event],"active")
        return stream.wait(r,s,4),act,earned
    def trial(s,obj,probe):
        flip=w["probe"][:,probe]
        right=w["x"][:,obj]*flip[:,None]
        a=stream.action(r,s,-right,right,w["probe_u"][:,probe])
        goal=w["law"][:,obj]*flip
        return a.side==goal,(a.motor_logit*goal).sigmoid()
    rows={}
    for history in HISTORIES:
        s=stream.start(g,rule,n)
        obj=1 if history=="unrelated_c" else 0
        prior=w["law"][:,obj]*(-1 if history=="opposite_a" else 1)
        s,_,_=experience(s,obj,prior,0)
        for i in range(max(GAPS)+1):
            if i:
                s,_,_=experience(s,i+1,w["law"][:,i+1],i)
            if i not in GAPS:continue
            # Each intervention starts at this original uninterrupted post-distractor point.
            reviewed=s
            earned_prefix=[]
            for m in range(max(REVIEWS)+1):
                if m:
                    event=max(GAPS)+m
                    reviewed,_,earned=experience(reviewed,0,w["law"][:,0],event)
                    earned_prefix.append(earned)
                if m not in REVIEWS:continue
                good,p=trial(reviewed,0,0)
                stem=f"{history}/{i}/{m}"
                rows[stem+"/old"]=good.numpy()
                rows[stem+"/p_old"]=p.numpy()
                if i:
                    other,_=trial(reviewed,i+1,1)
                    rows[stem+"/both"]=(good&other).numpy()
                    rows[stem+"/latest_b"]=other.numpy()
                if m:rows[stem+"/earned"]=torch.stack(earned_prefix,1).numpy().astype(np.int8)
    return rows


def run(source,label,births,worlds,seed):
    torch.set_num_threads(1)
    dest=DATA/"savings"/label
    dest.mkdir(parents=True,exist_ok=True)
    out=dict(births=births,worlds=worlds,seed=seed,gaps=GAPS,reviews=REVIEWS,
             interpretation="Prior true and opposite A use the same physical A cue, with opposing feedback; C prior "
                            "uses an unrelated cue and independent law for equal lifetime length. Then follow up to 64 "
                            "studied distractors. Each delayed test and each number of genuine A reviews forks from the "
                            "same original snapshot. Subsequent A probes receive no further taste.",
             roots={},sha256={})
    for birth in births:
        g,r=source.load(birth)
        w=make(seed,birth,worlds)
        raw=evaluate(g,r,w)
        out["roots"][birth]={k:float(v.mean()) for k,v in raw.items()}
        path=dest/f"{birth}.npz"
        np.savez_compressed(path,**raw,law=w["law"].numpy())
        out["sha256"][path.name]=sha256(path.read_bytes()).hexdigest()
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/f"savings-{label}.json").write_text(json.dumps(out,indent=1)+"\n",encoding="utf8")
    print(label,"complete",flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--label",required=True)
    p.add_argument("--family",choices=("joint","lifetime","objective","odor","feeder"),required=True)
    p.add_argument("--run",type=Path,default=ROOT/"data/local-trace-20261119/local")
    p.add_argument("--arm",default="active")
    p.add_argument("--step",type=int,default=8000)
    p.add_argument("--worlds",type=int,default=96)
    p.add_argument("--seed",type=int,default=778231)
    p.add_argument("--births",nargs="+",type=int,default=list(range(731,737)))
    a=p.parse_args()
    run(stream.Source(a.family,a.run,a.arm,a.step),a.label,a.births,a.worlds,a.seed)
