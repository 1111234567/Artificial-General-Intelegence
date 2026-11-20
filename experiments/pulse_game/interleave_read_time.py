"""Does an unchanged graph solve the non-linear triangle if it acts at a different recurrent tick?"""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_triangle_train as tri, local_trace_graph as native
from .interleave_interference import DATA,EVIDENCE


TICKS=(1,2,3,4,5,6,8,12,16,24)


@torch.no_grad()
def response(g,r,tape):
    n=len(tape["law"]);coeff=r.expand(n);s=native.start(g,n);rows=torch.arange(n)
    for t in range(3):
        _,edit,_,_=tri._contact(coeff,s,tape,rows,t,"active")
        s=native.wait(coeff,edit.state,4)
    results={}
    for name in ("present","then_quiet"):
        collector={i:[] for i in TICKS}
        for item in range(3):
            orientation={i:[] for i in TICKS}
            for sign in (1.,-1.):
                l,rv=tri.binding.present(tape,torch.full((n,),item),torch.full((n,),sign))
                stimulus=native.native.sensory(s.graph,l,rv)
                h=s.hidden
                for t in range(1,max(TICKS)+1):
                    drive=stimulus if name=="present" or t<=5 else torch.zeros_like(stimulus)
                    h=native.native.advance(s.graph,h,drive,1)
                    if t in TICKS:orientation[t].append(2.5*(h[:,11]-h[:,10]))
            for t in TICKS:collector[t].append(torch.stack(orientation[t],1))
        for t in TICKS:
            logit=torch.stack(collector[t],1).numpy()
            results[f"{name}/{t}"]=logit
    # Native five-tick readout must agree exactly with the normal physical entry point.
    ordinary=tri.trials(g,r,tape,"active")["logit"]
    np.testing.assert_allclose(results["present/5"],ordinary,atol=1e-6,rtol=1e-6)
    return results


def score(logit,law):
    logits=logit.reshape(-1,8,3,2)
    label=law.reshape(-1,8,3)
    margin=label*(logits[...,0]-logits[...,1])/2
    ok=margin>.05
    return dict(strict_pairs=float(ok[:,[0,7]].all((1,2)).mean()),uniform_maps=float(ok[:,[0,7]].all(2).mean()),
                mixed_maps=float(ok[:,1:7].all(2).mean()),all_maps=float(ok.all(2).mean()))


def run(path,arm,label,births,step,seed,worlds):
    torch.set_num_threads(1);dest=DATA/"read-time"/label;dest.mkdir(parents=True,exist_ok=True)
    report=dict(births=births,step=step,seed=seed,worlds_per_split=worlds,ticks=TICKS,roots={},sha256={},
                note="Same complete state after three actual contacts. Fork each query orientation with a held menu "
                     "or remove the menu after five ticks. No future tastes or further writes. A separate calibration "
                     "set picks one constant read time per birth by highest strict paired frustrated success, then by "
                     "higher score on mixed maps and closeness to five ticks. These are external timing interventions.")
    for b in births:
        g,r=tri.old_fit.load_checkpoint(path,"odor",arm,b,step)
        split={}
        for name,s in (("cal",seed),("test",seed+1)):
            tape=tri.make_tape(s,b,worlds,-.5)
            data=response(g,r,tape);law=tape["law"].numpy()
            split[name]={key:score(v,law) for key,v in data.items()}
            f=dest/f"{b}-{name}.npz";np.savez_compressed(f,**data,laws=law)
            report["sha256"][f.name]=sha256(f.read_bytes()).hexdigest()
        picked=max(split["cal"],key=lambda k:(split["cal"][k]["strict_pairs"],
                                              split["cal"][k]["mixed_maps"],-abs(int(k.split("/")[-1])-5)))
        report["roots"][b]=dict(selected=picked,selected_test=split["test"][picked],native_test=split["test"]["present/5"],
                                calibration=split["cal"],test=split["test"])
        print(label,b,picked,flush=True)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/f"read-time-{label}.json").write_text(json.dumps(report,indent=1)+"\n",encoding="utf8")


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--label",required=True);p.add_argument("--run",type=Path,default=DATA/"triangle-objective")
    p.add_argument("--arm",default="direct");p.add_argument("--step",type=int,default=8000)
    p.add_argument("--births",nargs="+",type=int,default=list(range(731,737)))
    p.add_argument("--seed",type=int,default=743211);p.add_argument("--worlds",type=int,default=96)
    a=p.parse_args();run(a.run,a.arm,a.label,a.births,a.step,a.seed,a.worlds)
