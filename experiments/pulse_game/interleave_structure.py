"""Separate weight, bias, growth and retirement effects over a long physical life."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_stream as stream, local_trace_graph as native
from .interleave_interference import DATA,EVIDENCE,ROOT


MODES=("active","values","weights","structure","no_growth","no_retire","sham")
AGES=(0,1,2,4,8,16,32,64)


@torch.no_grad()
def run_one(g,rule,tape,seed,birth,modes=MODES,ages=AGES):
    n,t=tape.left.shape[:2]
    coeff,rule2=rule.expand(n),rule.expand(2*n)
    probes=stream.ColdProbes(tape,seed,birth)
    raw={}
    for mode in modes:
        state=native.start(g,n)
        earned,counts,snaps,health=[],[],[],[]
        for i in range(t+1):
            if i%4==0 and i//4 in ages:
                snaps.append(probes.read(coeff,rule2,state).numpy())
                support=state.graph.present
                common=support&g.present
                w=state.graph.weights
                health.append(torch.stack((support.flatten(1).sum(1),(support!=g.present).flatten(1).sum(1),
                                          (w-g.weights).abs().flatten(1).mean(1),
                                          ((w.abs()>=native.WEIGHT_LIMIT-1e-5)&common).flatten(1).sum(1)/
                                          common.flatten(1).sum(1).clamp_min(1)),1).numpy())
            if i==t:break
            if i:state=native.wait(coeff,state,tape.wait[:,i])
            u=tape.draws[:,i]
            left,right=tape.left[:,i],tape.right[:,i]
            action=native.act(coeff,state,left,right,u[:,0])
            reward=action.side*tape.factor[:,i]+tape.fixed[:,i]
            exp=native.observe(coeff,action,left,right,reward)
            draw=u[:,1:].clone()
            if mode=="no_growth":draw[:,0]=1
            if mode=="no_retire":draw[:,2]=1
            op={"values":"weights_and_bias","weights":"weights_only","structure":"topology_only",
                "sham":"sham"}.get(mode,"active")
            result=native.commit(coeff,exp,draw,intervention=op)
            state=result.state
            rc=result.receipt
            counts.append(torch.stack(((rc.birth>=0).float(),(rc.retired>=0).float(),
                                       rc.weight_delta.abs().flatten(1).sum(1),rc.bias_delta.abs().sum(1)),1).numpy())
            earned.append((reward>0).numpy())
        raw[mode+"/reads"]=np.stack(snaps)
        raw[mode+"/health"]=np.stack(health)
        raw[mode+"/writes"]=np.stack(counts,1)
        raw[mode+"/earned"]=np.stack(earned,1)
    return raw


def run(source,label,births,worlds,seed,ages=AGES):
    torch.set_num_threads(1)
    out=DATA/"structure"/label
    out.mkdir(parents=True,exist_ok=True)
    report=dict(births=births,worlds=worlds,seed=seed,blocks=ages,modes=MODES,readouts=stream.READS,
                comparator="Interventions apply to the entire history. The read-only old-item probes never feed the "
                           "stream. The fresh-item fork always allows full native writes for that one new study, "
                           "so a learner that spent the history with numeric or structural writes disabled can still "
                           "demonstrate whether its current mechanism learns a fresh object.",
                roots={},sha256={})
    for birth in births:
        g,r=source.load(birth)
        tape=stream.tape_for_worlds(birth,worlds,max(ages),seed)
        raw=run_one(g,r,tape,seed,birth,ages=ages)
        rows={}
        for mode in MODES:
            total=raw[mode+"/writes"]
            if mode in ("values","weights","sham") and total[:,:,:2].any():
                raise AssertionError("A blocked support edit changed the graph")
            if mode=="no_growth" and total[:,:,0].any() or mode=="no_retire" and total[:,:,1].any():
                raise AssertionError("A blocked structural event was applied")
            if mode in ("structure","sham") and total[:,:,2:].any():
                raise AssertionError("Unexpected change to numeric weights/biases")
            for j,ar in enumerate(ages):
                values=raw[mode+"/reads"][j].mean((1,2))
                row=dict(zip(stream.READS,values.tolist(),strict=True))
                if ar:
                    s=4*ages[j-1];e=4*ar
                    probes=(np.arange(s,e)%4)>=2
                    row["quiz"]=float(raw[mode+"/earned"][:,s:e][:,probes].mean())
                    row["births_per_life"]=float(total[:,:e,0].sum(1).mean())
                    row["retirements_per_life"]=float(total[:,:e,1].sum(1).mean())
                    row["changed_support"]=float(raw[mode+"/health"][j,:,1].mean())
                    row["limit_fraction"]=float(raw[mode+"/health"][j,:,3].mean())
                rows[f"{mode}/{ar}"]=row
        path=out/f"{birth}.npz"
        np.savez_compressed(path,**raw)
        report["sha256"][path.name]=sha256(path.read_bytes()).hexdigest()
        report["roots"][birth]=rows
        print(label,birth,flush=True)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/f"structure-{label}.json").write_text(json.dumps(report,indent=1)+"\n",encoding="utf8")


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--label",required=True)
    p.add_argument("--family",choices=("joint","lifetime","objective","odor","feeder"),required=True)
    p.add_argument("--run",type=Path,default=ROOT/"data/local-trace-20261119/local")
    p.add_argument("--arm",default="active")
    p.add_argument("--step",type=int,default=8000)
    p.add_argument("--worlds",type=int,default=48)
    p.add_argument("--seed",type=int,default=638309)
    p.add_argument("--births",nargs="+",type=int,default=list(range(731,737)))
    a=p.parse_args()
    run(stream.Source(a.family,a.run,a.arm,a.step),a.label,a.births,a.worlds,a.seed)
