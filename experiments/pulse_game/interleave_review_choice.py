"""Does existing motor confidence identify which one real review helps the whole memory?"""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_capacity as cap, interleave_stream as stream
from .interleave_interference import DATA,EVIDENCE,ROOT


@torch.no_grad()
def trials(src,birth,seed,worlds,objects):
    g,r=src.load(birth);t=cap.make_tape("odor",seed,worlds,objects-1)
    n=t.lanes
    state=stream.start(g,r,n);view=None if r is None else r.expand(n)
    for i in range(objects):state,_=cap._study(view,state,t,i,write=True)
    snap=stream.wait(view,state,4)
    rng=np.random.default_rng([seed,98175])
    review_sign=torch.from_numpy(np.repeat(rng.choice((-1.,1.),(worlds,objects)).astype(np.float32),4,0))
    u=torch.from_numpy(np.repeat(rng.random((worlds,objects,6),dtype=np.float32),4,0))
    tie=torch.from_numpy(np.repeat(rng.random((worlds,objects),dtype=np.float32),4,0))
    def probes(s):
        side=[];logit=[];greedy=[]
        for j in range(objects):
            a=cap._probe(view,s,t,j);rev=stream.action(view,s,t.right[:,j,-1],t.left[:,j,-1],1-t.motor_draw[:,j,-1])
            side.append(a.side);logit.append(a.motor_logit);greedy.append(rev.motor_logit)
        return torch.stack(side,1),torch.stack(logit,1),torch.stack(greedy,1)
    pre,pre_logit,opposite=probes(snap)
    target=t.right_value[:,:,-1]
    # Features are read before the review, with no true label or subsequent result.
    conf=(pre_logit-opposite).abs()/2
    geometry=t.directions.repeat_interleave(4,0)
    unit=geometry/geometry.norm(dim=2,keepdim=True);overlap=torch.bmm(unit,unit.transpose(1,2)).abs()
    overlap.diagonal(dim1=1,dim2=2).zero_()
    index=torch.arange(objects).float()[None].expand(n,-1)/max(1,objects-1)
    feature=torch.stack((conf,pre_logit.abs(),(pre_logit+opposite).abs()/2,index,overlap.max(2).values,
                         overlap.sum(2)/(objects-1)),2)
    outcomes=[]
    for mode in ("active","sham"):
        effects=[]
        for item in range(objects):
            # Fresh orientation preserves each physical object's sign law.
            right=-geometry[:,item]*review_sign[:,item,None]
            goal=t.law[:,item]*review_sign[:,item]
            after,_,_=stream.contact(view,snap,-right,right,goal,torch.zeros_like(goal),u[:,item],mode)
            waited=stream.wait(view,after,4)
            s,l,_=probes(waited)
            effects.append(torch.stack(((s==target).float(),(l*target).sigmoid()),2))
        outcomes.append(torch.stack(effects,1))
    return dict(features=feature.numpy(),tie=tie.numpy(),outcomes=torch.stack(outcomes).numpy(),
                pre=(pre==target).numpy(),pre_p=(pre_logit*target).sigmoid().numpy())


def analyze(cal,test):
    # [mode, lane, reviewed object, test object, actual or conditional expected].
    # Fit the external selector only to the fraction of real future choices correct.
    x=cal["features"].reshape(-1,cal["features"].shape[-1]).astype(float)
    center=x.mean(0);scale=x.std(0).clip(1e-7)
    x=(x-center)/scale
    y=cal["outcomes"][0,:,:,:,0].mean(2).reshape(-1).astype(float)
    coef=np.linalg.solve(x.T@x+len(x)*.002*np.eye(x.shape[1]),x.T@(y-y.mean()))
    z=(test["features"]-center)/scale
    tie=test["tie"]; n,objects=tie.shape; row=np.arange(n)
    prediction=z@coef+y.mean()
    outcomes=test["outcomes"]
    methods={
        "uniform":tie.argmax(1),
        "least_confident":(-test["features"][:,:,0]+tie*1e-7).argmax(1),
        "most_confident":(test["features"][:,:,0]+tie*1e-7).argmax(1),
        "oldest":np.zeros(n,dtype=int),"newest":np.full(n,objects-1),
        "calibrated":prediction.argmax(1),
        "hindsight_expected":outcomes[0,:,:,:,1].mean(2).argmax(1),
    }
    stats={}
    for name,i in methods.items():
        successful=outcomes[0,row,i,:,0]
        sham=outcomes[1,row,i,:,0]
        chosen=successful[row,i]
        stats[name]=dict(mean=float(successful.mean()),all=float(successful.all(1).mean()),reviewed=float(chosen.mean()),
                         unreviewed=float((successful.sum(1)-chosen).mean()/(objects-1)),
                         mean_minus_sham=float((successful-sham).mean()),
                         mean_minus_pre=float((successful-test["pre"]).mean()),
                         pre_average_chosen_confidence=float(test["features"][row,i,0].mean()))
    return dict(methods=stats,calibration_coef=coef.tolist(),prior_mean=float(test["pre"].mean()),
                prior_all=float(test["pre"].all(1).mean())),methods


def run(src,label,births,worlds,seed):
    torch.set_num_threads(1)
    dest=DATA/"review-choice"/label;dest.mkdir(parents=True,exist_ok=True)
    out=dict(seed=seed,worlds_per_split=worlds,births=births,
             note="After studying 4 or 8 independently valued objects, the external scheduler can inspect native motor "
                  "confidence on read-only copies. Every selection policy then buys exactly one true contact and write. "
                  "All final probes fork from that common post-review state. The calibrated scheduler is fitted on "
                  "separate geometries with physical sampled outcomes. No claim the network autonomously chose the review.",
             roots={},sha256={})
    for b in births:
        summary={}
        for count in (4,8):
            cal=trials(src,b,seed,worlds,count);test=trials(src,b,seed+1,worlds,count)
            measured,choices=analyze(cal,test);summary[count]=measured
            f=dest/f"{b}-{count}.npz";np.savez_compressed(f,**{f"test/{k}":v for k,v in test.items()},
                                                          **{f"selection/{k}":v for k,v in choices.items()})
            out["sha256"][f.name]=sha256(f.read_bytes()).hexdigest()
        out["roots"][b]=summary;print(label,b,flush=True)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/f"review-choice-{label}.json").write_text(json.dumps(out,indent=1)+"\n",encoding="utf8")


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--label",required=True);p.add_argument("--family",choices=("joint","lifetime","objective","odor","feeder"),required=True)
    p.add_argument("--run",type=Path,default=ROOT/"data/local-trace-20261119/local")
    p.add_argument("--arm",default="active");p.add_argument("--step",type=int,default=8000)
    p.add_argument("--births",nargs="+",type=int,default=list(range(731,737)))
    p.add_argument("--worlds",type=int,default=128);p.add_argument("--seed",type=int,default=631519)
    a=p.parse_args();run(stream.Source(a.family,a.run,a.arm,a.step),a.label,a.births,a.worlds,a.seed)
