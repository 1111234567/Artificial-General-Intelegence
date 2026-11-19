"""Fit sensory retrieval and local modulation on matched-stream, physically sampled trajectories."""
import argparse
from dataclasses import asdict
import os
import time
import numpy as np
import torch

from . import symmetry_memory as design, symmetry_train as source, mixture_circuit_train as helpers
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


def donor_rows(ctx):
    # Match observable context separately at every event. Rolling within
    # groups avoids the trivial cue "negative belongs to another habitat".
    n,t,_=ctx.shape
    order=(ctx[:,:,1]+torch.rand(n,t,device=ctx.device)*.3).argsort(0)
    label=ctx[:,:,1].gather(0,order)
    previous=order.roll(1,0)
    priorlabel=label.roll(1,0)
    last0=(label==0).sum(0)-1
    last1=n-1
    target=torch.where(label==0,order.gather(0,last0.clamp(0,n-1)[None]).expand(n,-1),order[last1:].expand(n,-1))
    prev=torch.where(priorlabel==label,previous,target)
    return torch.zeros_like(order).scatter(0,order,prev)


def loss(model,batch,frac,weight):
    times=helpers.early_times(batch,frac)
    donor=donor_rows(batch["ctx"]) if weight else None
    pred,sense,_,_=model(batch,times,donor)
    free=batch["odor"].diff(dim=2).abs().sum((2,3))>.015
    err=(pred-batch["reward"]).square()
    physical=.76*err[free].mean()+.24*err[~free].mean()
    return physical+weight*sense,physical,sense


def load(path,cut=None):
    d=torch.load(path,map_location="cpu",weights_only=True)
    model=design.Memory(design.Config(**d["spec"]));model.load_state_dict(d["state"])
    return model.export(cut)


def train(cfg,seed,data,val,args,folder):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=design.Memory(cfg).to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=.0025,weight_decay=.00030)
    name=f"m{cfg.capacity}-{cfg.change}-aux{int(cfg.sensory_loss*100)}-s{seed}"
    cp=folder/(name+"-progress.pt");out=folder/(name+".pt")
    first=0;best=float("inf");hist=[];start=time.perf_counter()
    if cp.exists():
        state=torch.load(cp,map_location=args.device,weights_only=False)
        model.load_state_dict(state["state"]);opt.load_state_dict(state["optimizer"])
        first=state["step"];hist=state["history"];best=state["best"];rng.bit_generator.state=state["rng"]
    for j in range(first+1,args.steps+1):
        b=helpers.batch(data,torch.as_tensor(rng.integers(len(data["side"]),size=args.batch),device=args.device))
        objective,phys,sens=loss(model,b,float(rng.choice([0,.5,1.,1.])),cfg.sensory_loss)
        opt.zero_grad(set_to_none=True);objective.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.8)
        if not torch.isfinite(grad):raise ValueError("Nonfinite training")
        opt.step()
        if j==1 or j%100==0 or j==args.steps:
            with torch.no_grad():
                values=[]
                for ix in np.array_split(np.arange(len(val["side"])),8):
                    part=helpers.batch(val,torch.as_tensor(ix,device=args.device))
                    for f in (.33,1.):values.append(float(loss(model,part,f,0.)[1]))
            v=float(np.mean(values));hist.append(dict(step=j,validation=v,sampled_error=float(phys.detach()),
                 sensory_error=float(sens.detach()),time=time.perf_counter()-start))
            def state():return dict(spec=asdict(cfg),state=model.state_dict(),step=j,seed=seed,history=hist,best=best,
                                    train_lives=len(data["side"]),feedback="sampled public bite signs")
            if v<best:
                best=v;tmp=out.with_suffix(".partial.pt");torch.save(state(),tmp);os.replace(tmp,out)
            tmp=cp.with_suffix(".partial.pt")
            torch.save({**state(),"optimizer":opt.state_dict(),"rng":rng.bit_generator.state},tmp);os.replace(tmp,cp)
            _write(folder/(name+"-history.json"),hist)
            print(name,j,round(v,4),round(best,4),round(time.perf_counter()-start,1),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder",default="symmetry-memory-v1");p.add_argument("--steps",type=int,default=1600)
    p.add_argument("--seeds",type=int,nargs="+",default=[9643,3137,8273]);p.add_argument("--capacity",type=int,default=20)
    p.add_argument("--changes",nargs="+",default=["delta","learned","fixed"])
    p.add_argument("--aux",type=float,nargs="+",default=[.30,0.]);p.add_argument("--batch",type=int,default=72)
    p.add_argument("--device",default="cuda");p.add_argument("--data",default="symmetry-circuit-v2")
    args=p.parse_args();torch.set_num_threads(1)
    folder=OUT/args.folder;folder.mkdir(exist_ok=True)
    raw=OUT/args.data
    data=helpers.torch_data(source.data(raw,5190761,6144,True),args.device)
    val=helpers.torch_data(source.data(raw,3741649,768,True),args.device)
    for seed in args.seeds:
        for aux in args.aux:
            for update in args.changes:
                train(design.Config(capacity=args.capacity,change=update,sensory_loss=aux),seed,data,val,args,folder)
    _write(folder/"complete.json",dict(seeds=args.seeds,changes=args.changes,aux=args.aux,steps=args.steps))


if __name__=="__main__":main()
