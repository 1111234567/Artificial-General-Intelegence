"""Independent developmental, sensory, reward-learning and policy-search births for one recipe."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import torch

from . import habitat_meta as route,unified_train as teach,unified_evolution as evolve,unified_physical as body
from . import symmetry_grounded as senses,symmetry_memory as perception,mixture_circuit_train as tensors
from . import unified_world, unified_audit, unified_confirm, longrun_world,longrun_plastic,longrun_study
from . import research_audit as audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


def receptor(folder,seed,graph,data,val,device):
    cfg=perception.Config(graph_seed=graph)
    path=folder/"receptor.pt"
    if path.is_file():return route.receptor(path),path
    torch.manual_seed(seed)
    mdl=senses.Grounded(cfg).to(device)
    x=tensors.torch_data(data,device)
    error,secs=senses.public_fit(mdl,x,360,device,seed)
    project,singular,corr=senses.measured_transform(data,20)
    with torch.no_grad():
        mdl.transform.copy_(torch.from_numpy(project).to(device))
        v=tensors.torch_data({k:z[:128] for k,z in val.items()},device)
        state=mdl.sensor.states(v,v["touch"].long())
        mse=float((mdl.actual_odor(state)-v["odor"]).square().mean())
    from dataclasses import asdict
    torch.save(dict(spec=asdict(cfg),state=mdl.state_dict(),seed=seed,graph_seed=graph,step=360,
                    current_validation_mse=mse,train_mse=error),path)
    _write(folder/"representation.json",dict(seed=seed,graph_seed=graph,transform=project.tolist(),coactivation=corr,
                    singular=singular,reconstruction_mse=mse,seconds=secs,supervision="public sensory flashes"))
    return mdl,path


def resources(folder):
    front=senses.load(folder/"receptor.pt")
    model=next(folder.glob("rls-learned-s*.pt"))
    return front,route.load(model),np.array([.39,1.6,.78,.13,1.08,1.28,0.],np.float32)


def fixed_reopening():
    # A single previously selected hyperparameter. Use only discovery scores
    # of the original three births and fix it before independent replication.
    folder=OUT/"long-life-plasticity-v1"
    files=[p for p in folder.glob("rls-learned*.json") if not p.stem.endswith("parity")]
    if len(files)!=3:raise RuntimeError(files)
    settings=[json.loads(f.read_text())["settings"]["uncertainty_direction"]["parameters"] for f in files]
    return np.median(np.asarray(settings),axis=0).tolist()


def confirm(folder,seed,alpha):
    f,r,m=resources(folder);frozen=next(folder.glob("*--full-*/frozen.json"))
    policy=json.loads(frozen.read_text());vec=np.asarray(policy["best"]["genome"])
    ss=np.random.default_rng(418673901).integers(7_000_000,475_000_000,256).tolist()
    root=folder/"external.json"
    if root.exists():return
    result={}
    for name in ("standard","similar_habitats","long_competing_run"):
        ep=unified_world.make(ss*4,force=np.repeat([[1,1],[1,-1],[-1,1],[-1,-1]],len(ss),axis=0),
                              nonce=9500131,**unified_audit.SETTINGS[name])
        c=body.Cohort(ep,f,r,m)
        result[name]={}
        for label,genes in (("selected",vec),("pre_action_training",np.zeros_like(vec))):
            out,_=c.run(genes,"full",detail=True)[0]
            result[name][label]=unified_confirm.data(ep,out,len(ss))
    long={}
    lst=np.random.default_rng(50763819).integers(7_000_000,475_000_000,96).tolist()
    af=body.phenotype(body.params(f,r,m),vec,"full")[0]
    for k in range(0,len(lst),24):
        w=longrun_world.make(lst[k:k+24],blocks=192,mode="reverse_return",noise=.05,nonce=75619319)
        c=longrun_plastic.Cohort(w,f,r,m)
        for method,pa in (("fixed_RLS",[0.,0.,0.,0.,0.]),("reopen",alpha)):
            out=c.adjustments(af,pa,detail=True)[0]
            v=longrun_world.count_regions(w,out);dest=long.setdefault(method,{})
            for reg,metrics in v.items():
                for mm,x in metrics.items():dest.setdefault(reg,{}).setdefault(mm,[]).extend(x)
    _write(root,dict(recipe_seed=seed,roots=ss,long_roots=lst,parameters=alpha,selected_generation=policy["best"]["generation"],
                      tests=result,continuous=long,frozen_sha=audit.fingerprint(frozen)))


def replicate(base,item,args,alpha):
    root=base/f"birth-{item}";root.mkdir(exist_ok=True)
    rng=np.random.default_rng(item)
    sensory_seed,topology,policy_seed,learn_seed,train_worlds,val_worlds=rng.integers(1,400_000_000,6).tolist()
    mark=root/"external.json"
    if mark.exists():return
    model=f"rls-learned-s{learn_seed}";weight=root/(model+".pt")
    if not weight.exists() or json.loads((root/(model+"-history.json")).read_text())[-1]["step"]!=args.steps:
        tr=teach.raw(root,train_worlds,args.lives);va=teach.raw(root,val_worlds,800)
        net,src=receptor(root,sensory_seed,topology,tr,va,args.device)
        data=teach.precompute(net,tr,args.device);val=teach.precompute(net,va,args.device)
        teach.train(data,val,learn_seed,route.Config(routing="learned",updater="rls"),args,root,src)
        del data,val,tr,va,net
    f,r,m=resources(root)
    evolve.optimize(root,model,policy_seed,"full",SimpleNamespace(width=args.width,gens=args.gens,lives=80),
                    frozen_inputs=(f,r,m),model_path=weight)
    confirm(root,item,alpha)
    print("completed independent birth",item,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--count",type=int,default=12)
    p.add_argument("--steps",type=int,default=2000);p.add_argument("--lives",type=int,default=4800)
    p.add_argument("--gens",type=int,default=160);p.add_argument("--width",type=int,default=1024)
    p.add_argument("--device",default="cuda");args=p.parse_args();torch.set_num_threads(1)
    root=OUT/"independent-lifelong-births-v1";root.mkdir(exist_ok=True)
    seeds=np.random.default_rng(921486313).integers(70_000,470_000_000,12).tolist()[:args.count]
    alpha=fixed_reopening()
    _write(root/"registered.json",dict(fixed_birth_seeds=seeds,reopening_rule=alpha,training=vars(args),
            evaluation="same 256 unobserved paired roots for all births; stable 96 independent long streams",
            every_birth_counts=True,selection_worlds_only=True))
    for i,item in enumerate(seeds):
        _write(root/"progress.json",dict(candidate=i+1,total=len(seeds),seed=item,completed=seeds[:i]))
        replicate(root,item,args,alpha)
    _write(root/"progress.json",dict(candidate=None,total=len(seeds),completed=seeds))


if __name__=="__main__":main()
