"""Search a small developmental encoding of spiking recurrent graphs for delayed reward learning."""

import argparse
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.seed_evolution.large_runner import _write
from experiments.undirected._cuda_nvrtc import compile_kernels
from . import ring_world


N,D,K=64,12,16
OUT=Path(__file__).resolve().parents[2]/"data/pulse-game"
RANGES=(
    ("membrane_leak",.30,.985),("synapse_decay",.0,.95),("spike_trace_decay",.04,.98),
    ("adaptation_decay",.55,.995),("adaptation_strength",-.68,3.),("mean_rate",.001,.10),
    ("input_gain",.5,2.65),("tonic_bias",.015,.71),("tonic_spread",.015,.36),
    ("recurrent_gain",.05,4.),("assortment",-1.5,5.2),("inhibitory_fraction",.12,.53),
    ("inhibitory_gain",.56,3.),("start_gain",-.7,1.65),("plastic_rate",.28,1.65),
)


@dataclass
class Schedule:
    cue:np.ndarray
    ctx:np.ndarray
    issue:np.ndarray
    query:np.ndarray
    length:np.ndarray
    raw_at:np.ndarray
    distractor:np.ndarray
    at:np.ndarray
    goal:np.ndarray
    tie:np.ndarray
    seeds:tuple


def task(seeds,difficulty,trials=64):
    seeds=tuple(map(int,seeds))
    t=ring_world.make_tape(seeds,trials)
    ctx=t.ctx.astype(np.uint8);cue=t.cue.astype(np.uint8)
    issue=np.empty_like(cue);query=np.empty_like(cue);length=np.empty_like(cue)
    noise=np.empty_like(cue);at=np.full_like(cue,255);raw_at=np.full_like(cue,255)
    tie=np.empty((*cue.shape,4),np.float32)
    for a,s in enumerate(seeds):
        r=np.random.default_rng(np.random.SeedSequence([s,47705]))
        issue[a]=r.integers(0,2,size=trials)
        mode=difficulty if difficulty!="mixed" else ("short","interfere","long")[a%3]
        lo,hi={"short":(8,12),"interfere":(12,20),"long":(26,42),"extrapolate":(48,68),
               "serial":(8,15),"serial_noise":(12,20)}[mode]
        wait=r.integers(lo,hi+1,size=trials)
        serial=mode.startswith("serial")
        first_gap=r.integers(4,13,size=trials) if serial else 0
        if serial:raw_at[a]=issue[a]+first_gap
        query[a]=issue[a]+first_gap+wait
        length[a]=hi+6+(12 if serial else 0)
        noise[a]=r.integers(8,size=trials)
        if mode not in ("short","serial"):
            at[a]=issue[a]+first_gap+r.integers(2,wait-2,size=trials)
        tie[a]=r.random((trials,4),dtype=np.float32)
    return Schedule(cue,ctx,issue,query,length,raw_at,noise,at,t.goal.astype(np.uint8),tie,seeds)


def original_task(world_seed):
    from experiments.undirected import local_rule_world
    from experiments.undirected.evolved_plasticity_world import NEW,READY,FIRST_FINISH
    t=local_rule_world.schedule(seed=int(world_seed),actors=8,visits=16,device="cpu")
    trans=lambda v:np.ascontiguousarray(v.numpy().T,dtype=np.uint8)
    cue,ctx,goal=trans(t.raw),trans(t.context),trans(t.target-FIRST_FINISH)
    issue=trans(t.public[...,0,NEW].argmax(1));query=trans(t.public[...,0,READY].argmax(1))
    tie=np.random.default_rng(np.random.SeedSequence([int(world_seed),60971])).random((*cue.shape,4),dtype=np.float32)
    return Schedule(cue,ctx,issue,query,np.full_like(cue,18),np.full_like(cue,255),np.zeros_like(cue),np.full_like(cue,255),
                    goal,tie,tuple(int(world_seed)*8+i for i in range(8)))


@lru_cache(maxsize=4)
def template(seed):
    rng=np.random.default_rng(seed)
    sensing=rng.normal(.08,.62,(N,6))*(rng.random((N,6))<.73)
    norm=sensing/np.maximum(np.linalg.norm(sensing,axis=1,keepdims=True),.001)
    return (sensing,norm@norm.T,rng.gumbel(0.,1.,(N,N)),rng.uniform(.23,1.55,(N,N)),
            rng.random(N),rng.normal(0,1,N),rng.uniform(.85,1.13,N),rng.normal(.18,.54,N),
            (rng.normal(size=(K,N))/np.sqrt(N)).astype(np.float32))


def decode(vector):
    v=np.asarray(vector)
    return {name:float(low+(high-low)*x) for (name,low,high),x in zip(RANGES,v,strict=True)}


def genome(vector,seed,*,shuffle=False):
    p=decode(vector);s,similar,priors,amp,node_type,bias,threshold,new,projection=template(int(seed))
    indices=np.arange(N);ring=(indices-1)%N
    choice=priors+p["assortment"]*similar
    choice[indices,indices]=-np.inf;choice[indices,ring]=-np.inf
    src=np.column_stack((ring,np.argsort(-choice,axis=1,kind="stable")[:,:D-1])).astype(np.int32)
    signed=np.where(node_type<p["inhibitory_fraction"],-p["inhibitory_gain"],1.)
    weights=amp[indices[:,None],src]*signed[src]*(p["recurrent_gain"]/np.sqrt(D))
    if shuffle:
        # Permute the identities at both ends of the recurrent wiring only.
        # This conserves the recurrent graph and all weights, but breaks their
        # alignment with sensory tuning and cell identity.
        perm=np.random.default_rng(seed+57031).permutation(N)
        inverse=np.argsort(perm)
        src=inverse[src[perm]].astype(np.int32)
        weights=weights[perm]
    inputs=np.column_stack((p["input_gain"]*s,new*p["start_gain"]))
    offsets=np.stack((p["tonic_bias"]+p["tonic_spread"]*bias,threshold),axis=1)
    params=np.array([p[k] for k,_,_ in RANGES[:6]])
    return (inputs.astype(np.float32),offsets.astype(np.float32),src,weights.astype(np.float32),projection,
            params.astype(np.float32),np.float32(p["plastic_rate"]))


@lru_cache(maxsize=2)
def compiled(device):
    source=(f"#define NODES {N}\n#define DEGREE {D}\n#define KEYS {K}\n#define NPARAM 6\n"
            +Path(__file__).with_name("attractor_gpu.cu").read_text(encoding="utf-8"))
    return compile_kernels(source,("encode_lives","score_lives"),device=device)


def physical_time(t):
    return int(t.length.sum())


def gpu_batch(vectors,net,t,*,cut=False,feedback=True,rls=True,shuffle=False,features=False,device="cuda:0"):
    dev=torch.device(device);program=compiled(dev)
    genomes=[genome(v,net,shuffle=shuffle) for v in vectors]
    arrays=[torch.from_numpy(np.ascontiguousarray(np.stack(c))).to(dev) for c in zip(*genomes,strict=True)]
    inputs,offs,sources,edges,project,params,rate=arrays
    fields=[torch.from_numpy(np.ascontiguousarray(x)).to(dev) for x in
            (t.cue,t.ctx,t.issue,t.query,t.length,t.raw_at,t.distractor,t.at,t.goal,t.tie)]
    pop=len(vectors);acts,trials=t.goal.shape
    z=torch.empty((pop,acts,trials,K),device=dev,dtype=torch.float32)
    count=torch.empty((pop,acts,N),device=dev,dtype=torch.int32)
    out=torch.empty((pop,acts,trials),device=dev,dtype=torch.uint8)
    program["encode_lives"]((acts,pop),(N,),inputs,offs,sources,edges,project,params,*fields[:8],z,count,
                            acts,trials,int(cut),0)
    program["score_lives"]((pop*acts,),(128,),z,fields[8],fields[9],rate,out,acts,trials,int(feedback),int(rls))
    ans=out.cpu().numpy()
    pulse=count.sum((1,2)).cpu().numpy()/(physical_time(t)*N)
    arg={"answers":ans,"spike_fraction":pulse}
    if features:arg["z"]=z.cpu().numpy()
    return arg


def cpu_features(vector,seed,t,*,cut=False):
    inp,offset,src,w,proj,params,_=genome(vector,seed)
    leak,sd,zd,ad,adapt,meanrate=params
    result=np.empty((*t.goal.shape,K),np.float32);total=0
    for life in range(len(t.seeds)):
        h=np.zeros(N,np.float32);a=h.copy();z=h.copy();syn=h.copy();mean=np.full_like(h,.12)
        for j in range(t.cue.shape[1]):
            raw=inp[:,2+t.cue[life,j]]
            stimulus=inp[:,t.ctx[life,j]]+(raw if t.raw_at[life,j]==255 else 0.)+inp[:,6]
            d=t.distractor[life,j];noise=inp[:,d//4]+inp[:,2+d%4]
            for tick in range(t.length[life,j]):
                rec=np.zeros_like(h)
                if not cut:
                    for k in range(D):rec+=w[:,k]*syn[src[:,k]]
                drive=offset[:,0]+rec
                if tick==t.issue[life,j]:drive+=stimulus
                if tick==t.raw_at[life,j]:drive+=raw
                if tick==t.at[life,j]:drive+=noise
                h=np.maximum(-3.,leak*h+drive)
                sp=(h>=offset[:,1]+adapt*a).astype(np.float32)
                h[sp>0]=0
                a=ad*a+(1-ad)*sp
                z=zd*z+(1-zd)*sp
                mean+=meanrate*(z-mean)
                syn=sd*syn+(1-sd)*sp
                total+=int(sp.sum())
                if tick==t.query[life,j]:
                    read=np.zeros(K,np.float32)
                    for k in range(N):read+=proj[:,k]*(z[k]-mean[k])
                    result[life,j]=read
    return result,total/(physical_time(t)*N)


def cpu_answers(vector,features,t,*,feedback=True,rls=True):
    n,trials=t.goal.shape
    eta=np.float32(decode(vector)["plastic_rate"])
    w=np.zeros((n,4,K),np.float32);ids=np.arange(n)
    cov=np.broadcast_to(np.eye(K,dtype=np.float32)*4.,(n,4,K,K)).copy()
    out=np.empty_like(t.goal)
    for j in range(trials):
        f=features[:,j]
        mag=np.full(n,.01,np.float32)
        for k in range(K):mag+=f[:,k]*f[:,k]
        x=f*(1./np.sqrt(mag[:,None]))
        value=np.zeros((n,4),np.float32)
        for k in range(K):value+=w[:,:,k]*x[:,None,k]
        c=(value+1e-5*t.tie[:,j]).argmax(1);out[:,j]=c
        if feedback:
            if rls:
                old=cov[ids,c];z=np.zeros_like(x)
                for k in range(K):z+=old[:,:,k]*x[:,k,None]
            else:z=x
            scale=np.full(n,1. if rls else .1,np.float32)
            for k in range(K):scale+=x[:,k]*z[:,k]
            scale=1./scale
            if rls:cov[ids,c]=old-(scale[:,None,None]*z[:,:,None])*z[:,None,:]
            reward=np.where(t.goal[:,j]==c,1.,-1.).astype(np.float32)
            w[ids,c]+=(eta*(reward-value[ids,c])*scale)[:,None]*z
    return out


def evaluate(vectors,net,t,cut,batch):
    hit=[];fr=[]
    for i in range(0,len(vectors),batch):
        q=gpu_batch(vectors[i:i+batch],net,t,cut=cut)
        hit.extend((q["answers"]==t.goal).mean(1))
        fr.extend(q["spike_fraction"])
    arr=np.stack(hit)
    return np.stack((arr[:,32:].mean(1),arr[:,:8].mean(1),arr.mean(1),np.array(fr)),axis=1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name",required=True);p.add_argument("--seed",type=int,default=149221);p.add_argument("--cut",action="store_true")
    p.add_argument("--population",type=int,default=128);p.add_argument("--generations",type=int,default=64)
    p.add_argument("--actors",type=int,default=48);p.add_argument("--batch",type=int,default=32)
    p.add_argument("--difficulty",choices=("mixed","short","interfere","long","serial","serial_noise"),default="mixed")
    p.add_argument("--validate-every",type=int,default=4)
    args=p.parse_args()
    torch.set_num_threads(1)
    folder=OUT/args.name;folder.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed);state=folder/"search.json"
    proto={k:v for k,v in vars(args).items() if k not in ("generations","batch")}
    if state.exists():
        saved=json.loads(state.read_text());hist=saved["history"];pop=np.array(saved["next"])
        if saved["protocol"]!=proto:raise ValueError("search checkpoint differs")
        best=saved["best"];rng.bit_generator.state=saved["rng"]
    else:
        hist=[];best=None;pop=rng.random((args.population,len(RANGES)))
    start=time.perf_counter()
    val=task(tuple(map(int,np.random.default_rng(982376).integers(100000,8000000,size=96))),args.difficulty)
    for gen in range(len(hist)+1,args.generations+1):
        seeds=tuple(map(int,rng.integers(100_000,480_000_000,size=args.actors)))
        net=int(rng.integers(1,5_000_000))
        t=task(seeds,args.difficulty)
        scores=evaluate(pop,net,t,args.cut,args.batch)
        order=np.argsort(-scores[:,0],kind="stable"); elite=pop[order[:max(4,len(pop)//8)]]
        record={"gen":gen,"net":net,"worlds":seeds,"train":scores[order[0]].tolist(),"mean":scores.mean(0).tolist()}
        if gen==1 or gen%args.validate_every==0 or gen==args.generations:
            checks=np.stack([evaluate(elite[:4],b,val,args.cut,args.batch) for b in (127,541,1999)]).mean(0)
            ix=checks[:,0].argmax();v=float(checks[ix,0]);record["validation"]=checks.tolist()
            if best is None or v>best["late"]:
                vec=elite[ix]
                best=dict(gen=gen,late=v,vector=vec.tolist(),values=decode(vec),validation=checks[ix].tolist())
        hist.append(record)
        children=[elite[0],np.array(best["vector"]),*elite[1:3]]
        while len(children)<len(pop):
            if rng.random()<.12:
                x=rng.random(len(RANGES))
            else:
                i,j=rng.integers(len(elite),size=2)
                x=np.where(rng.random(len(RANGES))<.25,elite[i],elite[j])
                x=np.clip(x+rng.normal(0,rng.choice([.035,.10,.19]),size=len(x)),0,1)
            children.append(x)
        pop=np.stack(children)
        _write(state,{"protocol":proto,"best":best,"history":hist,"next":pop.tolist(),"rng":rng.bit_generator.state})
        print(gen,"late",round(scores[order[0],0],4),"mean",round(scores[:,0].mean(),4),"devbest",round(best["late"],4),
              "elapsed",round(time.perf_counter()-start,1),flush=True)


if __name__=="__main__":
    main()
