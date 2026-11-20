"""Compare review spacing at identical first/last A times and matched distractor events."""

from argparse import ArgumentParser
from hashlib import sha256
import json

import numpy as np
import torch

from . import interleave_age_train as age
from . import interleave_stream as stream
from .interleave_interference import DATA, EVIDENCE, ROOT


SCHEDULES={"spaced":(0,4,8,12),"early":(0,1,2,12),"late":(0,10,11,12)}
SHAPES=(1,4,12)


def setup(birth,seed,worlds,n_other):
    rng=np.random.default_rng([721035,seed,birth,n_other])
    directions=age.directions(rng,worlds,1+n_other)
    law=age.signs(rng,(worlds,1+n_other))
    # Initial A label and sign of the entire independent B-label vector cross separately.
    physical=np.repeat(law,4,axis=0)
    physical[:,0]=np.tile([-1.,1.,-1.,1.],worlds)
    physical[:,1:]*=np.tile([1.,1.,-1.,-1.],worlds)[:,None]
    b_order=np.broadcast_to((np.arange(12)%n_other)+1,(worlds,12)).copy()
    b_order=rng.permuted(b_order,axis=1)
    def repeat(arr):
        return torch.from_numpy(np.repeat(arr,4,axis=0))
    return dict(x=repeat(directions),law=torch.from_numpy(physical),order=repeat(b_order),
                # Slots 0..3 are the four A events; 4..15 are B events in fixed order.
                orient=repeat(age.signs(rng,(worlds,16))),u=repeat(rng.random((worlds,16,6),dtype=np.float32)),
                probe=repeat(age.signs(rng,(worlds,1+n_other))),
                draw=repeat(rng.random((worlds,1+n_other),dtype=np.float32)))


@torch.no_grad()
def score(g,r,world,n_other,*,gaps=(4,64)):
    rows=torch.arange(len(world["x"]))
    rules=None if r is None else r.expand(len(rows))
    results={}
    for sched,a_times in SCHEDULES.items():
        for writes in ("all","middle_a_sham"):
            state=stream.start(g,r,len(rows))
            a=b=0
            achieved=[]
            for t in range(16):
                if t in a_times:
                    oid=torch.zeros_like(rows)
                    s=a
                    a+=1
                    do_write=writes=="all" or s in (0,3)
                else:
                    oid=world["order"][:,b]
                    s=4+b
                    b+=1
                    do_write=True
                right=world["x"][rows,oid]*world["orient"][:,s,None]
                goal=world["law"][rows,oid]*world["orient"][:,s]
                u=world["u"][:,s]
                state,_,earned=stream.contact(rules,state,-right,right,goal,torch.zeros_like(goal),
                                              u,"active" if do_write else "sham")
                achieved.append(earned>0)
                if t<15:
                    state=stream.wait(rules,state,4)
            if a!=4 or b!=12:
                raise AssertionError("Exposure counts changed across schedules")
            results[sched+"/"+writes+"/earned"]=torch.stack(achieved,1).numpy()
            for gap in gaps:
                final=stream.wait(rules,state,gap)
                side=[]
                for item in range(n_other+1):
                    right=world["x"][:,item]*world["probe"][:,item,None]
                    trial=stream.action(rules,final,-right,right,world["draw"][:,item])
                    side.append(trial.side.to(torch.int8))
                results[f"{sched}/{writes}/{gap}"]=torch.stack(side,1).numpy()
    return results


def run(source,label,births,seed,worlds):
    torch.set_num_threads(1)
    dest=DATA/"spacing"/label
    dest.mkdir(parents=True,exist_ok=True)
    out=dict(births=births,seed=seed,worlds_per_birth=worlds,positions=SCHEDULES,
             fixed_last_A=12,contacts=16,neutral_ticks_between=4,
             interpretation="Each schedule has the same A and B presentations and random draws by event, "
                            "including three B events after the last A. No fresh feedback during any final probes.",
             results={},sha256={})
    for n in SHAPES:
        out["results"][n]={}
        for birth in births:
            w=setup(birth,seed,worlds,n)
            g,r=source.load(birth)
            raw=score(g,r,w,n)
            truth=(w["law"]*w["probe"]).numpy()
            cells={}
            for key,side in raw.items():
                if key.endswith("earned"):continue
                correct=side==truth
                cells[key]=dict(a=float(correct[:,0].mean()),b_avg=float(correct[:,1:].mean()),
                                a_and_b_avg=float((correct[:,:1]&correct[:,1:]).mean()),
                                all=float(correct.all(1).mean()),
                                positive_a=float(correct[w["law"][:,0].numpy()>0,0].mean()),
                                negative_a=float(correct[w["law"][:,0].numpy()<0,0].mean()))
            p=dest/f"{n}-{birth}.npz"
            np.savez_compressed(p,**raw,truth=truth,border=w["order"].numpy(),law=w["law"].numpy())
            out["results"][n][birth]=cells
            out["sha256"][p.name]=sha256(p.read_bytes()).hexdigest()
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/f"spacing-{label}.json").write_text(json.dumps(out,indent=1)+"\n",encoding="utf8")
    return out


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--label",required=True)
    p.add_argument("--family",choices=("joint","lifetime","objective","odor","feeder"),required=True)
    p.add_argument("--run",type=age.Path,default=ROOT/"data/local-trace-20261119/local")
    p.add_argument("--arm",default="active")
    p.add_argument("--step",type=int,default=8000)
    p.add_argument("--worlds",type=int,default=64)
    p.add_argument("--seed",type=int,default=817811)
    p.add_argument("--births",nargs="+",type=int,default=list(range(731,737)))
    a=p.parse_args()
    run(stream.Source(a.family,a.run,a.arm,a.step),a.label,a.births,a.seed,a.worlds)
