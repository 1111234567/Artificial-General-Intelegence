"""Train a recurrent forager's fast-synapse similarity from its own sampled rewards."""

import os
for _k in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[_k]="1"

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.seed_evolution.large_runner import _write, _save
from . import ring_world as world
from .ring_metric import Settings, Appetite, Metric, MetricForager, event_batch
from .ring_search import measures


ROOT=Path(__file__).resolve().parents[2]


def map_parity(tape):
    row=tape.preferences[:,0]
    return sum((row[:,i]>row[:,j]).astype(np.int8) for i in range(4) for j in range(i+1,4))%2


def get_seeds(rng,n,parity):
    result=[]
    while len(result)<n:
        possible=rng.integers(1,480_000_000,size=max(64,2*(n-len(result)))).tolist()
        if parity is None:
            result.extend(possible)
        else:
            small=world.make_tape(possible,trials=1)
            result.extend(np.asarray(possible)[map_parity(small)==parity].tolist())
    return result[:n]


def collect(cfg,agent_params,n,trials,stream,exposures,*,retry_mix=True,chemistry="onehot"):
    rng=np.random.default_rng(stream)
    # Keep a whole-map held-out class; individual cue and fruit marginals remain balanced.
    seeds=get_seeds(rng,n,0)
    xs=[];ys=[];ms=[];count=0
    for block,i in enumerate(range(0,n,64)):
        env_seeds=seeds[i:i+64]
        tape=world.make_tape(env_seeds,trials)
        explorer=agent_params is None
        appetite=(Appetite(random_bite=float(rng.uniform(.24,.70))) if explorer else
                  Appetite(threshold=.08,gain=1.1,random_bite=.19,urgency=.018))
        p=MetricForager(cfg,agent_params,env_seeds,appetite=appetite,explorer=explorer,record=True)
        exposure=exposures[block%len(exposures)]
        style=(("onehot","anchored","novel","noisy")[(block//len(exposures))%4]
               if chemistry=="mixed" else chemistry)
        acting=p
        if style!="onehot":
            from .ring_sensors import ChemicalInterface
            acting=ChemicalInterface(p,env_seeds,style)
        world.rollout(acting,tape,retry=retry_mix and block%2==1,cue_ticks=exposure)
        x,y,valid=event_batch(p.events,trials)
        count+=int(valid.sum());xs.append(x);ys.append(y);ms.append(valid)
    return tuple(map(np.concatenate,(xs,ys,ms))),{"life_seeds":seeds,"event_count":count,
             "max_events_per_life":trials,"source":"closed_loop_policy" if agent_params is not None else "physical_random_explorer"}


def validate(model,seed,exposures,*,n=64,trials=80,parity=0,chemistry="onehot"):
    cfg=model.cfg;params=model.export()
    seeds=get_seeds(np.random.default_rng(seed),n,parity)
    tape=world.make_tape(seeds,trials)
    out={}
    for k in exposures:
        agent=MetricForager(cfg,params,tape.seeds)
        if chemistry!="onehot":
            from .ring_sensors import ChemicalInterface
            agent=ChemicalInterface(agent,tape.seeds,"novel" if chemistry=="mixed" else chemistry)
        result=world.rollout(agent,tape,cue_ticks=k)
        out[str(k)]=measures(result)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name",default="ring-metric-01")
    p.add_argument("--seed",type=int,default=330104)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--steps",type=int,default=600)
    p.add_argument("--batch",type=int,default=64)
    p.add_argument("--initial-lives",type=int,default=640)
    p.add_argument("--trials",type=int,default=80)
    p.add_argument("--exposures",nargs="+",type=int,default=[40])
    p.add_argument("--onset-trace",action="store_true")
    p.add_argument("--no-raw",action="store_true")
    p.add_argument("--chemistry",choices=("onehot","anchored","novel","mixed"),default="onehot")
    p.add_argument("--objective",choices=("logistic","balanced","mse"),default="logistic")
    p.add_argument("--refresh-every",type=int,default=240)
    p.add_argument("--audit-every",type=int,default=64)
    p.add_argument("--lr",type=float,default=.0018)
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    cfg=Settings(graph_seed=args.seed+23,onset_trace=args.onset_trace,raw_skip=not args.no_raw)
    model=Metric(cfg).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.001)
    out=ROOT/"data/pulse-game"/args.name;out.mkdir(parents=True,exist_ok=True)
    params={k:v for k,v in vars(args).items() if k not in ("steps","device")}
    data_file=out/"initial-experiences.npz"
    def device_data(data):
        return tuple(torch.from_numpy(v).to(args.device) for v in data)
    if data_file.is_file():
        with np.load(data_file) as z: initial=device_data([z[k] for k in ("x","y","valid")])
        archived=json.loads((out/"protocol.json").read_text())["params"]
        if {"chemistry":"onehot","objective":"logistic",**archived}!=params:
            raise ValueError("Cached dataset belongs to a different protocol")
    else:
        data,metadata=collect(cfg,None,args.initial_lives,args.trials,args.seed*33,args.exposures,
                              chemistry=args.chemistry)
        np.savez_compressed(data_file,x=data[0],y=data[1],valid=data[2])
        _write(out/"protocol.json",{"params":params,"settings":asdict(cfg),"initial":metadata,
               "training_A_permutation_parity":0,"loss":"actual bite sign; no unchosen identity labels",
               "training_sensory_source_sha256":hashlib.sha256(Path(__file__).with_name("ring_metric.py").read_bytes()).hexdigest()})
        initial=device_data(data)
    state=out/"resume.pt"; history=[];best=-1.;start=0;buffer=initial;last_refresh=0
    if state.exists():
        q=torch.load(state,map_location=args.device,weights_only=True)
        model.load_state_dict(q["model"]);optimizer.load_state_dict(q["optimizer"])
        start,best,history,rng.bit_generator.state,last_refresh=q["step"],q["best"],q["history"],q["rng"],q["last_refresh"]
        if last_refresh:
            with np.load(out/f"interactions-{last_refresh}.npz") as z:
                on=device_data([z[k] for k in ("x","y","valid")])
            buffer=tuple(torch.cat((v,x)) for v,x in zip(initial,on,strict=True))
    clock=time.perf_counter()
    for step in range(start+1,args.steps+1):
        if args.refresh_every and step%args.refresh_every==0:
            on,meta=collect(cfg,model.export(),max(128,args.initial_lives//2),args.trials,
                            args.seed*997+step,args.exposures,chemistry=args.chemistry)
            np.savez_compressed(out/f"interactions-{step}.npz",x=on[0],y=on[1],valid=on[2])
            _write(out/f"interactions-{step}.json",meta)
            fresh=device_data(on)
            buffer=tuple(torch.cat((v,x)) for v,x in zip(initial,fresh,strict=True))
            last_refresh=step
        index=torch.as_tensor(rng.integers(len(buffer[0]),size=args.batch),device=args.device)
        x,y,valid=(v.index_select(0,index) for v in buffer)
        q=model.predict_events(x,y,valid)
        if args.objective=="mse":
            loss=((y-q).square()*valid).sum()/valid.sum()
        else:
            weight=valid if args.objective=="logistic" else valid*torch.where(y>0,2.,1.)
            loss=(F.softplus(-y*q*F.softplus(model.logit_gain))*weight).sum()/weight.sum()
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
        optimizer.step()
        if step==1 or step%args.audit_every==0 or step==args.steps:
            scores=validate(model,args.seed*374+1,sorted(set(args.exposures)),chemistry=args.chemistry)
            val=float(np.mean([s["late"] for s in scores.values()]))
            history.append({"step":step,"loss":float(loss.detach()),"development":scores,"mean_late":val})
            snap={"model":model.state_dict(),"settings":asdict(cfg),"step":step}
            if val>best:
                best=val;_save(out/"best.pt",snap)
            _save(state,{**snap,"history":history,"best":best,"rng":rng.bit_generator.state,
                         "optimizer":optimizer.state_dict(),"last_refresh":last_refresh})
            _write(out/"history.json",history)
            print("step",step,"loss",round(history[-1]["loss"],4),"late",round(val,4),
                  "seconds",round(time.perf_counter()-clock,1),flush=True)


if __name__=="__main__":
    main()
