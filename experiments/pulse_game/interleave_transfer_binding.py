"""Keep all actions and tastes identical and change only which past odors received them."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import torch

from . import interleave_age_train as age, interleave_stream as stream, local_trace_graph as net
from .interleave_interference import DATA,EVIDENCE,ROOT


@torch.no_grad()
def experiment(g,rule,birth,worlds,seed,k):
    rng=np.random.default_rng([771399,seed,birth,k])
    def twins(x):return torch.from_numpy(np.repeat(x,2,axis=0))
    # Opposite hidden preference directions share every object and every random draw.
    all_x=age.directions(rng,worlds,k+1);w=age.directions(rng,worlds)
    labels=np.where((all_x*w[:,None]).sum(2)>=0,1.,-1.).astype(np.float32)
    true=twins(labels)*torch.tensor([-1.,1.]).repeat(worlds)[:,None]
    x=twins(all_x);ori=twins(age.signs(rng,(worlds,k+1)))
    draws=twins(rng.random((worlds,k+1,5),dtype=np.float32))
    perm=rng.permuted(np.tile(np.arange(k),(worlds,1)),axis=1)
    # Cycle a random ordering so all prior objects are presented once, with no fixed positions.
    rotated=np.empty_like(perm);rotated[np.arange(worlds)[:,None],perm]=np.roll(perm,1,axis=1)
    future=twins(rotated)
    n=len(x);row=torch.arange(n)
    r=rule.expand(n)
    start=net.start(g,n)
    states={name:start for name in ("own","mirror","permute","sham","replay")}
    key_evidence=[];taste_evidence=[]
    for t in range(k):
        source=x[:,t]*ori[:,t,None];u=draws[:,t]
        own=net.act(r,states["own"],-source,source,u[:,0])
        taste=own.side*ori[:,t]*true[:,t]
        key_evidence.append(torch.nn.functional.normalize(source*own.side[:,None],dim=1));taste_evidence.append(taste)
        for name,state in tuple(states.items()):
            if name=="mirror":right=-source
            elif name=="permute":right=x[row,future[:,t]]*ori[:,t,None]
            else:right=source
            action=own if name=="own" else net.act(r,state,-right,right,u[:,0],side_override=own.side)
            exp=net.observe(r,action,-right,right,taste)
            if not torch.equal(exp.side,own.side) or not torch.equal(exp.taste,taste):
                raise AssertionError("Yoked history diverged in physical side or delivered taste")
            result=net.commit(r,exp,u[:,1:],intervention="sham" if name=="sham" else "active")
            states[name]=net.wait(r,result.state,4)
    anchor,mirrored=states["own"],states["mirror"]
    states.update(P_mirror=net.swap(anchor,mirrored,"graph"),
                  h_and_traces_mirror=net.swap(anchor,mirrored,("hidden","trace")))
    if not torch.equal(anchor.graph.weights,states["replay"].graph.weights):
        raise AssertionError("Identical-side original-sensor replay must reproduce parameters")
    right=x[:,-1]*ori[:,-1,None];goal=true[:,-1]*ori[:,-1];u=draws[:,-1,0]
    outputs={}
    for name,state in states.items():
        a=net.act(r,state,-right,right,u)
        outputs[name+"/side"]=a.side.numpy().astype(np.int8)
        outputs[name+"/logit"]=a.motor_logit.numpy()
    if not np.array_equal(outputs["own/logit"],outputs["replay/logit"]):
        raise AssertionError("Yoked original sensor path changed the output")
    keys=torch.stack(key_evidence,1);taste=torch.stack(taste_evidence,1)
    sim=(keys*right[:,None]).sum(2)
    sum_side=(taste*sim).sum(1).sign()
    closest=sim.abs().argmax(1)
    nn_side=sim[row,closest].sign()*taste[row,closest]
    return dict(**outputs,goal=goal.numpy().astype(np.int8),sum_side=sum_side.numpy().astype(np.int8),
                nearest_side=nn_side.numpy().astype(np.int8))


def run(src,label,births,worlds,seed):
    torch.set_num_threads(1)
    dest=DATA/"transfer-binding"/label;dest.mkdir(parents=True,exist_ok=True)
    out=dict(births=births,worlds=worlds,seed=seed,results={},sha256={},
             design="A uses real physical choices and tastes in a rotated linear resource world. Three controls replay exactly "
             "those actual choices and tastes with original, sign-mirrored, or permuted previous odor displays. A mirrored "
             "world with inverted latent direction produces the same physical taste for every one of these past actions. "
             "Then all act freely on the same never-before-seen item, with the same action draw. Scores use original law.")
    for b in births:
        g,r=src.load(b);data={};cells={}
        for k in (4,8,16):
            z=experiment(g,r,b,worlds,seed,k)
            row={}
            for n in ("own","mirror","permute","sham","P_mirror","h_and_traces_mirror"):
                side=z[n+"/side"]
                row[n]=float((side==z["goal"]).mean())
                row[n+"_changed"]=float((side!=z["own/side"]).mean())
            disagree=z["sum_side"]!=z["nearest_side"]
            row["reference_disagreement_n"]=int(disagree.sum())
            row["choose_sum_when_disagree"]=float((z["own/side"][disagree]==z["sum_side"][disagree]).mean()) if disagree.any() else None
            row["sum_correct_when_disagree"]=float((z["sum_side"][disagree]==z["goal"][disagree]).mean()) if disagree.any() else None
            cells[k]=row
            data.update({f"{k}/{key}":v for key,v in z.items()})
        path=dest/f"{b}.npz";np.savez_compressed(path,**data);out["sha256"][path.name]=sha256(path.read_bytes()).hexdigest()
        out["results"][b]=cells
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/f"transfer-binding-{label}.json").write_text(json.dumps(out,indent=1)+"\n",encoding="utf8")
    print(label,flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--label",required=True);p.add_argument("--family",choices=("joint","lifetime","objective","odor","feeder"),required=True)
    p.add_argument("--run",type=Path,default=ROOT/"data/local-trace-20261119/local")
    p.add_argument("--arm",default="active");p.add_argument("--step",type=int,default=8000)
    p.add_argument("--births",nargs="+",type=int,default=list(range(731,737)))
    p.add_argument("--worlds",type=int,default=192);p.add_argument("--seed",type=int,default=4771557)
    a=p.parse_args();run(stream.Source(a.family,a.run,a.arm,a.step),a.label,a.births,a.worlds,a.seed)
