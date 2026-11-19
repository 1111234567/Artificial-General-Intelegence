"""Learn context sharing and conflict resolution in the same recurrent individual."""
import argparse
from dataclasses import asdict
import os
import time
import numpy as np
import torch

from . import habitat_world as body, unified_world as game, habitat_meta as neuronal
from . import mixture_circuit_train as helpers, symmetry_world as pulses, mixture_metric, research_audit as audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


def raw(folder,seed,n):
    path=folder/f"unified-public-{seed}-{n}.npz"
    if path.exists():
        with np.load(path) as data:return {k:data[k] for k in data.files}
    bank={};rng=np.random.default_rng(seed)
    for i in range(0,n,80):
        ss=rng.integers(10_000_000,470_000_000,size=min(n-i,80)).tolist()
        # Variable stage lengths at fixed batchable total duration. No stage
        # index enters either the learned state or the physical controller.
        initial=int(rng.choice([8,10,12,14,16]));end=24-initial
        middle=sorted(rng.choice(np.arange(1,end//2),size=2,replace=False)*2)
        ds=(0,*middle,end)
        ep=game.make(ss,initial=initial,depths=ds,span=2,noise=float(rng.uniform(0.,.14)),
                     jitter=float(rng.uniform(.02,.13)),appearance=rng.choice(["discrete","smooth","nearby"]),
                     same_prior=.77,nonce=89279619)
        w=ep.arena;agent=mixture_metric.Explorer(ss,w.menu.body.trials)
        result=body.rollout(agent,w,silent_probes=True,record=True)
        b=pulses.public_rows(w,result)
        # At every test the environment privately scores the ONE taste physically
        # chosen. It is withheld from the animal. This bit can enter the outer
        # objective but cannot enter later in-life retrievals or synaptic writes.
        b["shown"]=b["reward"].copy()
        b["reward"]=result.reward.astype(np.float32)
        if (b["shown"][w.probe]!=0).any():raise AssertionError("Test outcome exposed")
        for k,v in b.items():bank.setdefault(k,[]).append(v)
    data={k:np.concatenate(v) for k,v in bank.items()}
    tmp=path.with_suffix(".partial.npz");np.savez_compressed(tmp,**data);os.replace(tmp,path)
    return data


def retrieve(net,read,write,shown):
    temp,null,_,_,_=net.scalars()
    n=read.size(1);t=torch.arange(n,device=read.device)
    available=shown!=0
    ranks=available.cumsum(1)
    before=ranks-available.long()
    distance=before[:,:,None]-ranks[:,None,:]
    legal=(t[:,None]>t[None,:])[None,:,:] & available[:,None,:] & (distance<net.cfg.capacity)
    logit=(temp*(read@write.transpose(1,2))).masked_fill(~legal,-1e4)
    nulls=(temp*null).expand(*logit.shape[:2],1)
    prob=torch.cat((logit,nulls),2).softmax(2)
    return (prob[:,:,:-1]*shown[:,None,:]).sum(2)


def precompute(sensor,bank,device):
    raw=helpers.torch_data(bank,device);sensor=sensor.to(device);out={}
    with torch.no_grad():
        for ids in np.array_split(np.arange(len(raw["side"])),max(1,(len(raw["side"])+63)//64)):
            b=helpers.batch(raw,torch.as_tensor(ids,device=device))
            chosen=b["side"].long()[:,:,None,None].expand(-1,-1,1,sensor.cfg.nodes)
            touch=sensor.sensor.states(b,b["touch"].long());q,k=sensor.heads(touch.gather(2,chosen).squeeze(2))
            shown=b["shown"].float()
            denom=retrieve(sensor,q,k,shown)
            chunk=dict(s=touch.mean(2),ctx=b["ctx"],r=denom*(shown!=0),y=b["reward"].float(),
                       free=b["odor"].diff(dim=2).abs().sum((2,3))>.01)
            for phase,f in enumerate((0.,.33,.66,1.)):
                s=sensor.sensor.states(b,helpers.early_times(b,f))
                read,_=sensor.heads(s.gather(2,chosen).squeeze(2))
                chunk[f"s{phase}"]=s.mean(2);chunk[f"r{phase}"]=retrieve(sensor,read,k,shown)
            for k,v in chunk.items():out.setdefault(k,[]).append(v)
    return {k:torch.cat(v) for k,v in out.items()}


def train(data,val,seed,cfg,args,folder,source):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=neuronal.MetaSynapses(cfg).to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=.0027,weight_decay=.0002)
    name=f"{cfg.updater}-{cfg.routing}-s{seed}"
    path=folder/(name+".pt");history=[];best=float("inf");start=time.perf_counter()
    for i in range(args.steps+1):
        if i:
            b=helpers.batch(data,torch.as_tensor(rng.integers(len(data["s"]),size=128),device=args.device))
            loss=neuronal.objective(model,b,int(rng.integers(4)))
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.5);opt.step()
        if i%200==0 or i==args.steps:
            with torch.no_grad():
                vals=[float(neuronal.objective(model,helpers.batch(val,torch.as_tensor(ix,device=args.device)),phase))
                      for ix in np.array_split(np.arange(len(val["s"])),8) for phase in (1,3)]
            score=float(np.mean(vals));history.append(dict(step=i,score=score,elapsed=time.perf_counter()-start))
            if score<best:
                best=score;tmp=path.with_suffix(".partial.pt")
                torch.save(dict(config=asdict(cfg),state=model.state_dict(),step=i,score=score,seed=seed,
                                receptor_sha=audit.fingerprint(source),scoring="only actual touched food",probe_feedback=False),tmp)
                os.replace(tmp,path)
            _write(folder/(name+"-history.json"),history)
            print(name,i,round(score,4),round(best,4),round(time.perf_counter()-start,1),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--folder",default="unified-transfer-v1")
    p.add_argument("--lives",type=int,default=6400);p.add_argument("--steps",type=int,default=2200)
    p.add_argument("--seeds",type=int,nargs="+",default=[1373,8539,6547])
    p.add_argument("--device",default="cuda");args=p.parse_args();torch.set_num_threads(1)
    folder=OUT/args.folder;folder.mkdir(exist_ok=True)
    source=OUT/"symmetry-grounded-v1/public-delta-s7243-before.pt"
    sensor=neuronal.receptor(source)
    data=precompute(sensor,raw(folder,12492371,args.lives),args.device)
    val=precompute(sensor,raw(folder,31190571,960),args.device)
    print("actual public trajectories prepared",len(data["s"]),flush=True)
    for seed in args.seeds:
        for update in ("rls","delta"):
            for route in ("learned","single","public","shared_public","random"):
                train(data,val,seed,neuronal.Config(routing=route,updater=update),args,folder,source)
    _write(folder/"complete.json",dict(seeds=args.seeds,steps=args.steps,training_lives=args.lives))


if __name__=="__main__":main()
