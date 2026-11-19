"""Search a second neuronal time scale without external cue latching or answer labels."""

import argparse
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.seed_evolution.large_runner import _write
from experiments.undirected._cuda_nvrtc import compile_kernels
from . import attractor_search as base


EXTRA=(("dendrite_leak",.65,.995),("dendrite_gain",-2.8,2.8),
       ("dendrite_add",-1.20,1.20),("dendrite_rec",-.75,1.25))
RANGES=base.RANGES+EXTRA


@lru_cache(maxsize=4)
def _second_inputs(net):
    rng=np.random.default_rng(np.random.SeedSequence([net,0xA9FC2]))
    return (rng.normal(.05,.70,(base.N,7))*(rng.random((base.N,7))<.73)).astype(np.float32)


def birth(v,net,mode,shuffle=False):
    original=base.genome(v[:len(base.RANGES)],net,shuffle=shuffle)
    extra=np.array([lo+(hi-lo)*x for (_,lo,hi),x in zip(EXTRA,v[len(base.RANGES):],strict=True)],np.float32)
    if mode=="basal":extra[1:]=0.
    elif mode=="add":extra[1]=0.
    elif mode=="gain":extra[2]=0.
    elif mode!="both":raise ValueError(mode)
    inputs=np.ascontiguousarray(np.concatenate((original[0],_second_inputs(net)),1))
    return (inputs,*original[1:5],np.concatenate((original[5],extra)),original[6])


def _source():
    s=Path(base.__file__).with_name("attractor_gpu.cu").read_text()
    snippets={
        "float membrane=0.f, nuisance=0.f":"float dendrite=0.f; float membrane=0.f, nuisance=0.f",
        "const float* sensory=inputs+unit*7;":"const float* sensory=inputs+unit*14; const float* basal=sensory+7;",
        """float drive=bias+recurrent;
            if(t==issued[ix])drive+=stimulus;
            if(t==raw_at[ix])drive+=sensory[2+c];
            if(t==intrusion_tick[ix])drive+=distractor;""": """
            float signal=0.f, second=0.f;
            if(t==issued[ix]){
                signal+=stimulus;second+=basal[ctx]+(raw_at[ix]==255?basal[2+c]:0.f)+basal[6];
            }
            if(t==raw_at[ix]){signal+=sensory[2+c];second+=basal[2+c];}
            if(t==intrusion_tick[ix]){signal+=distractor;second+=basal[noise/4]+basal[2+noise%4];}
            const float past=tanhf(dendrite);
            float drive=bias+recurrent;
            drive+=(g[2+6]*past)+signal*(1.f+g[1+6]*past);
            dendrite=fminf(12.f,fmaxf(-12.f,g[6]*dendrite+second+g[3+6]*recurrent));
            """,
    }
    for old,new in snippets.items():
        if s.count(old)!=1:raise ValueError("Sealed encoder changed: "+old[:45])
        s=s.replace(old,new)
    return f"#define NODES {base.N}\n#define DEGREE {base.D}\n#define KEYS {base.K}\n#define NPARAM 10\n"+s


@lru_cache(maxsize=1)
def kernels():
    return compile_kernels(_source(),("encode_lives","score_lives"),device=torch.device("cuda:0"))


def run(vectors,net,t,*,mode="both",cut=False,shuffle=False,features=False,feedback=True):
    dev=torch.device("cuda:0")
    genomes=[birth(v,net,mode,shuffle) for v in vectors]
    stack=lambda vs:torch.as_tensor(np.ascontiguousarray(np.stack(vs)),device=dev)
    a=[stack(vs) for vs in zip(*genomes,strict=True)]
    public=[torch.as_tensor(np.ascontiguousarray(v),device=dev) for v in (t.cue,t.ctx,t.issue,t.query,t.length,t.raw_at,t.distractor,t.at)]
    n,tr=t.goal.shape;pop=len(genomes)
    z=torch.empty((pop,n,tr,base.K),device=dev,dtype=torch.float32)
    count=torch.empty((pop,n,base.N),device=dev,dtype=torch.int32)
    ans=torch.empty((pop,n,tr),device=dev,dtype=torch.uint8)
    cu=kernels()
    cu["encode_lives"]((n,pop),(base.N,),*a[:6],*public,z,count,n,tr,int(cut),0)
    cu["score_lives"]((n*pop,),(128,),z,torch.as_tensor(t.goal,device=dev),torch.as_tensor(t.tie,device=dev),
                      a[6],ans,n,tr,int(feedback),1)
    return ans.cpu().numpy(),count.sum((1,2)).cpu().numpy()/(base.physical_time(t)*base.N),z.cpu().numpy() if features else None


def scores(vectors,net,t,mode,cut):
    out=[]
    for j in range(0,len(vectors),32):
        ans,f,_=run(vectors[j:j+32],net,t,mode=mode,cut=cut)
        correct=ans==t.goal
        out.extend(np.stack((correct[:,:,32:].mean((1,2)),correct[:,:,:8].mean((1,2)),correct.mean((1,2)),f),1))
    return np.stack(out)


def search(name,seed,mode,cut,budget,population,task,guide):
    if population%64:raise ValueError("budget must be divisible by 64")
    folder=base.OUT/name;folder.mkdir(exist_ok=True)
    path=folder/"checkpoint.json";rng=np.random.default_rng(seed)
    protocol=dict(seed=seed,mode=mode,cut=cut,population=population,task=task,guide=guide)
    width=len(RANGES);families=population//64
    val=base.task(np.random.default_rng(437091).integers(10_000_000,470_000_000,96).tolist(),task)
    if path.exists():
        state=json.loads(path.read_text());hist=state["history"];population_x=np.array(state["next"])
        if state["protocol"]!=protocol:raise ValueError("Protocol differs")
        rng.bit_generator.state=state["rng"];best=state["best"]
        previous_pairs=state["previous_pairs"]
    else:
        hist=[];population_x=rng.random((population,width));best=None;previous_pairs=None
    started=time.perf_counter()
    for gen in range(len(hist)+1,budget+1):
        graph=int(rng.integers(1,9_000_000));ws=rng.integers(10_000_000,470_000_000,48).tolist()
        result=scores(population_x,graph,base.task(ws,task),mode,cut)
        score=result[:,0];sort=np.argsort(-score,kind="stable")
        top=population_x[sort[:max(4,families*2)]]
        line={"generation":gen,"graph":graph,"worlds":ws,"top":result[sort[0]].tolist(),"mean":result.mean(0).tolist()}
        if gen==1 or gen%8==0 or gen==budget:
            valid=np.stack([scores(top[:4],n,val,mode,cut) for n in (709,571,1301)]).mean(0)
            pick=int(valid[:,0].argmax())
            if best is None or valid[pick,0]>best["late"]:
                best={"late":float(valid[pick,0]),"vector":top[pick].tolist(),"generation":gen,"metrics":valid[pick].tolist()}
            line["validation"]=valid.tolist()
        # Each family has one real incumbent (lane 0), one previous guided
        # and one matched-random proposal (1,2), 24 mirrored pairs (3..50),
        # and independent directions/immigrants (51..63).
        directions=[]
        if previous_pairs:
            line["validated_proposal_difference"]=[float(score[j*64+1]-score[j*64+2]) for j in range(families)]
            for j in range(families):
                idx=j*64
                dp=population_x[idx+3:idx+51:2]-population_x[idx+4:idx+51:2]
                advantage=score[idx+3:idx+51:2]-score[idx+4:idx+51:2]
                vec=(dp*advantage[:,None]).sum(0)
                directions.append(vec/(np.linalg.norm(vec)+1e-12))
        else:directions=[None]*families
        next_pop=[]
        for j in range(families):
            prev=j*64
            # Construct the directional test at the same point where the
            # directions were estimated. New families still explore other points.
            center=top[j] if j%2==0 else population_x[prev+np.argmax(score[prev:prev+64])]
            if j==0:center=np.asarray(best["vector"])
            old=population_x[prev]
            original_direction=rng.normal(size=width);original_direction/=np.linalg.norm(original_direction)
            d=directions[j] if guide and directions[j] is not None and np.linalg.norm(directions[j])>.5 else original_direction
            random_dir=rng.normal(size=width);random_dir/=np.linalg.norm(random_dir)
            # Reflect rather than clip at the cube edge, to retain exploration.
            reflect=lambda x:1.-np.abs(1.-np.mod(x,2.))
            guided=reflect(old+.10*d);random_child=reflect(old+.10*random_dir)
            children=[center,guided,random_child]
            for k in range(24):
                perturb=rng.normal(0,rng.choice([.026,.065,.13]),size=width)
                # Symmetric truncation keeps the central finite-difference pair exact.
                perturb=np.clip(perturb,-np.minimum(center,1.-center),np.minimum(center,1.-center))
                children.extend((center+perturb,center-perturb))
            while len(children)<64:
                donor=top[rng.integers(len(top))]
                if rng.random()<.40:child=rng.random(width)
                else:child=reflect(np.where(rng.random(width)<.2,donor,center)+rng.normal(0,.12,width))
                children.append(child)
            next_pop.extend(children)
        hist.append(line);population_x=np.stack(next_pop);previous_pairs=True
        _write(path,dict(protocol=protocol,history=hist,best=best,next=population_x.tolist(),
                         previous_pairs=previous_pairs,rng=rng.bit_generator.state))
        if gen%8==0:print(name,gen,"best",round(best["late"],3),"elapsed",round(time.perf_counter()-started),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--tag",default="dendrite")
    p.add_argument("--modes",nargs="+",default=["basal","add","gain","both"])
    p.add_argument("--tasks",nargs="+",default=["serial_noise"])
    p.add_argument("--seeds",type=int,nargs="+",default=[386051,619855,151829])
    p.add_argument("--population",type=int,default=1536);p.add_argument("--generations",type=int,default=144)
    p.add_argument("--no-guide",action="store_true");p.add_argument("--cut",action="store_true")
    args=p.parse_args();torch.set_num_threads(1)
    done=[]
    for seed in args.seeds:
        for task in args.tasks:
            for mode in args.modes:
                name=f"{args.tag}-{task}-{mode}-{'cut' if args.cut else 'full'}-{seed}"
                _write(base.OUT/(args.tag+"-sweep.json"),{"active":name,"done":done})
                search(name,seed,mode,args.cut,args.generations,args.population,task,not args.no_guide)
                done.append(name)
    _write(base.OUT/(args.tag+"-sweep.json"),{"active":None,"done":done})


if __name__=="__main__":
    main()
