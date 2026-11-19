"""Select a strategy for transfer solely from public outcomes of physical actions."""
import argparse
from dataclasses import asdict
import os
import time
import numpy as np
import torch

from . import mixture_circuit_train as training, mixture_flash as flash, symmetry_world as game, symmetry_circuit as brain
from .mixture_metric import Explorer
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


def data(folder,seed,n,balanced=False):
    path=folder/f"public-explorers-{seed}-{n}-matched{int(balanced)}.npz"
    if path.is_file():
        with np.load(path) as stored:return {k:stored[k] for k in stored.files}
    bank={}
    rng=np.random.default_rng(seed)
    for start in range(0,n,96):
        # Exposure count, B probe count and total duration vary across batches;
        # fixing total public bouts accommodates efficient simultaneous training.
        # Every word and motor destination remains randomized per individual life.
        pairs=int(rng.choice([6,8,12,14]))
        probes=24-pairs
        span=int(rng.choice([1,1,2,3]))
        if balanced:
            pairs=int(rng.choice([8,12,16]))
            probes=24-pairs
            span=int(rng.choice([2,4,4,8]))
        seeds=rng.integers(10_000_000,470_000_000,min(n-start,96)).tolist()
        episode=game.make(seeds,pairs=pairs,probes=probes,interleave=span,balanced_stream=balanced,
                          noise=float(rng.uniform(.0,.10)),jitter=float(rng.uniform(.02,.14)))
        agent=Explorer(seeds,48)
        r=flash.rollout(agent,episode.menu,record=True)
        rows=game.public_rows(episode,r)
        for k,v in rows.items():bank.setdefault(k,[]).append(v)
    packed={k:np.concatenate(v) for k,v in bank.items()}
    tmp=path.with_suffix(".partial.npz")
    np.savez_compressed(tmp,**packed);os.replace(tmp,path)
    return packed


def loss(net,b,fraction):
    early=training.early_times(b,fraction)
    pred,_,_,_=net(b,early)
    # On free-choice bouts the two public foods differ. Weighting those
    # actual rewards higher matches the deployable decision objective. The
    # supporting forced-bout targets are also real signed tastes at contact.
    free=b["odor"].diff(dim=2).abs().sum((2,3))>.015
    target=b["reward"]
    err=(pred-target).square()
    return .76*err[free].mean()+.24*err[~free].mean()


def load(path,cut=None):
    saved=torch.load(path,map_location="cpu",weights_only=True)
    model=brain.Circuit(brain.Spec(**saved["spec"]))
    model.load_state_dict(saved["state"])
    return model.export(cut)


def train(spec,seed,train_set,val,args,folder):
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    net=brain.Circuit(spec).to(args.device)
    opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.0004)
    name=f"{spec.graph}-{spec.mechanism}-e{net.e}-s{seed}"
    start=time.perf_counter();best=float("inf")
    path=folder/(name+".pt");cp=folder/(name+"-progress.pt")
    hist=[];first=0
    if cp.is_file():
        s=torch.load(cp,map_location=args.device,weights_only=False)
        net.load_state_dict(s["state"]);opt.load_state_dict(s["optimizer"])
        rng.bit_generator.state=s["rng"];hist=s["history"];best=s["best"];first=s["step"]
    for step in range(first+1,args.steps+1):
        sl=torch.as_tensor(rng.integers(len(train_set["side"]),size=args.batch),device=args.device)
        b=training.batch(train_set,sl)
        frac=float(rng.choice([.0,.45,.8,1.,1.]))
        value=loss(net,b,frac)
        opt.zero_grad(set_to_none=True);value.backward()
        gn=torch.nn.utils.clip_grad_norm_(net.parameters(),1.5)
        if not torch.isfinite(gn):raise ValueError("Nonfinite outer update")
        opt.step()
        if step==1 or step%100==0 or step==args.steps:
            with torch.no_grad():
                sc=[]
                for indices in np.array_split(np.arange(len(val["side"])),8):
                    v=training.batch(val,torch.as_tensor(indices,device=args.device))
                    sc.append([float(loss(net,v,f)) for f in (.33,1.)])
                criterion=float(np.mean(sc))
            hist.append(dict(step=step,error=criterion,elapsed=time.perf_counter()-start))
            def state():return dict(spec=asdict(spec),state=net.state_dict(),step=step,seed=seed,best=best,history=hist,
                                    lives=len(train_set["side"]),labels="signed outcome of physically touched food")
            if criterion<best:
                best=criterion;tmp=path.with_suffix(".partial.pt")
                torch.save(state(),tmp);os.replace(tmp,path)
            tmp=cp.with_suffix(".partial.pt")
            torch.save({**state(),"rng":rng.bit_generator.state,"optimizer":opt.state_dict()},tmp);os.replace(tmp,cp)
            _write(folder/(name+"-history.json"),hist)
            print(name,step,round(criterion,4),round(best,4),round(time.perf_counter()-start,1),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder",default="symmetry-circuit-v1");p.add_argument("--seeds",nargs="+",type=int,default=[6371,2377,9091])
    p.add_argument("--mechanisms",nargs="+",default=["evidence","uniform","single"])
    p.add_argument("--graph",default="rotations");p.add_argument("--experts",type=int,default=2)
    p.add_argument("--steps",type=int,default=1600);p.add_argument("--lives",type=int,default=6144)
    p.add_argument("--device",default="cuda");p.add_argument("--batch",type=int,default=72)
    p.add_argument("--balanced-stream",action="store_true")
    args=p.parse_args();torch.set_num_threads(1)
    folder=OUT/args.folder;folder.mkdir(exist_ok=True)
    si=training.torch_data(data(folder,5190761,args.lives,args.balanced_stream),args.device)
    val=training.torch_data(data(folder,3741649,768,args.balanced_stream),args.device)
    for seed in args.seeds:
        for mechanism in args.mechanisms:
            train(brain.Spec(mechanism=mechanism,experts=args.experts,graph=args.graph),seed,si,val,args,folder)
    _write(folder/"finished.json",dict(seeds=args.seeds,mechanisms=args.mechanisms,steps=args.steps))


if __name__=="__main__":main()

