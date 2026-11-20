"""Select susceptibility of receiver traces by whole-life resource, including old and new objects."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import interleave_age_train as age, interleave_train as common, interleave_rule_search as search
from . import local_trace_graph as net, local_trace_train as previous
from .interleave_interference import DATA,EVIDENCE


ARMS=("receiver","global","permuted")
DOMAIN=347511


@dataclass(frozen=True)
class Config:
    updates:int=4000
    directions:int=4
    lr:float=.03
    sigma:float=.25
    seed:int=72583
    save_every:int=250


def length(seed,step):
    ks=(2,4,8) if step<=500 else (2,4,8,16,32) if step<=1500 else (2,4,8,16,32,64)
    return int(np.random.default_rng([DOMAIN,seed,step,13]).choice(ks))


def tape(root,step,seed,worlds,k,replicas=2,domain=DOMAIN):
    rng=np.random.default_rng([domain,root,seed,step])
    paths=age.directions(rng,worlds,k+4)
    # A and B recur; one randomly chosen familiar object is absent for the last
    # third of the prefix. The unknown final request is balanced over A, B and new X.
    u=rng.random((worlds,k))
    cold=rng.integers(2,size=worlds)
    known=rng.integers(2,size=(worlds,k))
    if k:
        known[:,2*k//3:]=1-cold[:,None]
    distract=np.where(rng.random((worlds,k))<.35,2,3+np.arange(k)[None,:])
    mid=np.where(u<.55,known,distract)
    first=rng.integers(2,size=worlds)
    novel=paths.shape[1]-1
    ids=np.concatenate((np.stack((first,1-first),1),mid,np.full((worlds,1),novel,dtype=int)),1)
    final=rng.integers(3,size=worlds)
    row=np.arange(worlds)[:,None]
    base_law=age.signs(rng,(worlds,k+4))
    bits=np.array([[a,b,c] for a in (-1,1) for b in (-1,1) for c in (-1,1)],dtype=np.float32)
    laws=np.repeat(base_law[:,None,:],8,axis=1)
    laws[:,:,0]=bits[:,0]; laws[:,:,1]=bits[:,1];laws[:,:,novel]=bits[:,2]
    flip=age.signs(rng,ids.shape)
    def share(x):
        return torch.from_numpy(np.repeat(np.repeat(x[:,None,None],replicas,1),8,2)
                                .reshape(worlds*replicas*8,*x.shape[1:]))
    factor=np.take_along_axis(laws,ids[:,None,:],axis=2)*flip[:,None,:]
    factor=np.repeat(factor[:,None],replicas,1).reshape(-1,ids.shape[1])
    test_target=laws[:,:,[0,1,novel]]
    test_flip=age.signs(rng,(worlds,3))
    return dict(right=share(paths[row,ids]*flip[...,None]),factor=torch.from_numpy(factor),
                draws=share(rng.random((*ids.shape,5),dtype=np.float32)),
                waits=share(rng.integers(0,9,size=ids.shape)),
                probe=share(paths[:,[0,1,novel]]*test_flip[...,None]),
                truth=torch.from_numpy(np.repeat((test_target*test_flip[:,None])[:,None],replicas,1).reshape(-1,3)),
                motor=share(rng.random((worlds,3),dtype=np.float32)),query=share(final),
                dormant=share(cold),permutation=share(rng.random((worlds,20)).argsort(1)))


def apply_gain(view,experience,draw,gain):
    original=experience.state
    tagged=replace(original,receiver_trace=original.receiver_trace*gain[:,:,None])
    applied=net.commit(view,replace(experience,state=tagged),draw)
    return replace(applied,state=replace(applied.state,receiver_trace=original.receiver_trace))


def parents(root,births):
    gs,rs=zip(*(previous.load_checkpoint(root,"odor","uniform",b,8000) for b in births),strict=True)
    g=net.Graph(*(torch.cat([getattr(v,k) for v in gs]) for k in ("weights","bias","present")))
    r=net.Rule(births)
    r.load_state_dict({k:torch.cat([v.state_dict()[k] for v in rs]) for k in r.state_dict()})
    return g,r.requires_grad_(False)


@torch.no_grad()
def simulate(g,rule,data,genes,mode,*,all_queries=False,pre_new=False):
    # One independent candidate or one paired perturbation per row; the 8 law histories
    # share all unordered observations and random numbers and differ only in true value.
    n=len(data["truth"]); lanes=n//len(g.bias)
    state=net.start(g,lanes); view=rule.expand(lanes)
    if mode=="global":
        gain=(2*genes.sigmoid()).expand(-1,20)
    elif mode=="native":
        gain=torch.ones(len(genes),20)
    else:
        gain=2*genes.sigmoid()
    gain=gain.repeat_interleave(8,0)
    if mode=="permuted":gain=gain.gather(1,data["permutation"])
    pre_score=None
    for t in range(data["factor"].shape[1]):
        if t:state=net.wait(view,state,data["waits"][:,t],max_ticks=8)
        if pre_new and t==data["factor"].shape[1]-1:
            pre_score=[]
            for i in range(3):
                obj=data["probe"][:,i]
                a=net.act(view,state,-obj,obj,data["motor"][:,i])
                pre_score.append(a.side==data["truth"][:,i])
            pre_score=torch.stack(pre_score,1)
        right=data["right"][:,t];u=data["draws"][:,t]
        action=net.act(view,state,-right,right,u[:,0])
        exp=net.observe(view,action,-right,right,action.side*data["factor"][:,t])
        state=apply_gain(view,exp,u[:,1:],gain).state
    state=net.wait(view,state,4)
    rows=torch.arange(n)
    selections=(0,1,2) if all_queries else (data["query"],)
    returned=[]
    for choice in selections:
        obj=data["probe"][rows,choice]
        a=net.act(view,state,-obj,obj,data["motor"][rows,choice])
        returned.append(a.side*data["truth"][rows,choice])
    return torch.stack(returned,1),pre_score


def fit(root,out,births,arm,cfg):
    torch.set_num_threads(1)
    g,r=parents(root,births)
    dim=1 if arm=="global" else 20
    gene=torch.nn.Parameter(torch.zeros(len(births),dim))
    opt=torch.optim.Adam((gene,),lr=cfg.lr)
    folder=out/arm;folder.mkdir(parents=True,exist_ok=True)
    snaps=sorted(folder.glob("step-*.pt"))
    step,track,contacts=0,[],0
    if snaps:
        d=torch.load(snaps[-1],weights_only=True,map_location="cpu")
        if d["births"]!=list(births) or d["config"]!=asdict(cfg) or d["arm"]!=arm:raise ValueError("Wrong fit")
        gene.data.copy_(d["gene"]);opt.load_state_dict(d["optimizer"])
        step,track,contacts=d["step"],d["trace"].tolist(),d["contacts"]
    def save():
        payload=dict(step=step,births=list(births),config=asdict(cfg),arm=arm,gene=gene.detach(),
                     optimizer=opt.state_dict(),trace=torch.as_tensor(track),contacts=contacts)
        common._atomic_save(age.ckpath(out,arm,step),lambda f:torch.save(payload,f))
    if not snaps:save()
    first=step;started=perf_counter()
    for step in range(first+1,cfg.updates+1):
        k=length(cfg.seed,step)
        samples=[tape(b,step,cfg.seed,cfg.directions,k) for b in births]
        batch={key:torch.cat([s[key] for s in samples]) for key in samples[0]}
        noise=torch.as_tensor(np.stack([age.signs(np.random.default_rng([DOMAIN,cfg.seed,b,step,117]),
                                                  (cfg.directions,dim)) for b in births]))
        with torch.no_grad():
            candidate=gene[:,None,None,:]+cfg.sigma*noise[:,:,None,:]*gene.new_tensor([1,-1])[None,None,:,None]
            returned,_=simulate(g,r,batch,candidate.flatten(0,2),arm)
            returns=returned.reshape(len(births),cfg.directions,2,8).mean(3)
            grad=-search.direction_gradient(returns,noise,cfg.sigma)
            opt.zero_grad(set_to_none=True);gene.grad=grad
            different=(returns[:,:,0]!=returns[:,:,1]).float().mean(1)
            stats=torch.stack(((returns.mean((1,2))+1)/2,different,grad.norm(dim=1),
                               (2*gene.sigmoid()).mean(1),(2*gene.sigmoid()).std(1,unbiased=False)),1)
        opt.step()
        contacts+=cfg.directions*2*8*(k+4)
        track.append(stats.tolist())
        if step%cfg.save_every==0 or step==cfg.updates:save()
        if step%250==0 or step in (1,cfg.updates):
            print(arm,step,np.mean(track[-100:],axis=(0,1)).round(3),
                  round((perf_counter()-started)/(step-first),3),flush=True)


@torch.no_grad()
def evaluate(root,out,births,arms,cfg,step,worlds,seed):
    torch.set_num_threads(1)
    dest=out/"validation";dest.mkdir(parents=True,exist_ok=True)
    result=dict(seed=seed,worlds_per_birth=worlds,births=births,step=step,roots={},sha256={})
    for b in births:
        g,r=parents(root,[b]);per_root={}
        for arm in ("native",*arms):
            if arm=="native":gene=torch.zeros(1,20)
            else:
                data=torch.load(age.ckpath(out,arm,step),weights_only=True,map_location="cpu")
                ix=data["births"].index(b);gene=data["gene"][ix:ix+1]
            modes=[arm]
            if arm=="receiver":modes+=["permuted"]
            for mode in modes:
                label=f"{arm}_as_{mode}"
                cells={};raw={}
                for k in (0,4,16,64,128):
                    # Generate independent chunks with unchanged seed addresses across all treatments.
                    correct=[];pre=[];dormant=[]
                    for n in range(0,worlds,16):
                        size=min(worlds-n,16)
                        t=tape(b,n,seed,size,k,replicas=1,domain=DOMAIN+7)
                        ret,prior=simulate(g,r,t,gene.expand(size,-1),mode,all_queries=True,pre_new=True)
                        correct.append((ret>0).numpy());pre.append(prior.numpy());dormant.append(t["dormant"].numpy())
                    c=np.concatenate(correct);before=np.concatenate(pre);idx=np.concatenate(dormant);rows=np.arange(len(c))
                    cells[k]=dict(old_pair=float(c[:,:2].all(1).mean()),old=float(c[:,:2].mean()),fresh=float(c[:,2].mean()),
                                  all=float(c.all(1).mean()),dormant=float(c[rows,idx].mean()),
                                  active_old=float(c[rows,1-idx].mean()),old_before_fresh=float(before[:,:2].mean()))
                    raw[str(k)]=c;raw[f"{k}/before_fresh"]=before;raw[f"{k}/dormant"]=idx
                per_root[label]=dict(gains=(2*gene.sigmoid()).squeeze(0).tolist(),scores=cells)
                p=dest/f"{label}-{b}-{step}.npz";np.savez_compressed(p,**raw)
                result["sha256"][p.name]=sha256(p.read_bytes()).hexdigest()
        result["roots"][b]=per_root
        print("evaluated",b,flush=True)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/f"susceptibility-{step}.json").write_text(json.dumps(result,indent=1)+"\n",encoding="utf8")


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("fit","eval"),required=True)
    p.add_argument("--out",type=Path,default=DATA/"susceptibility-fit")
    p.add_argument("--source",type=Path,default=DATA/"retention-objective")
    p.add_argument("--births",nargs="+",type=int,default=list(previous.BIRTHS))
    p.add_argument("--arms",nargs="+",choices=ARMS,default=ARMS)
    p.add_argument("--updates",type=int,default=4000);p.add_argument("--directions",type=int,default=4)
    p.add_argument("--workers",type=int,default=2);p.add_argument("--worlds",type=int,default=48)
    p.add_argument("--test-seed",type=int,default=741823)
    a=p.parse_args();cfg=Config(updates=a.updates,directions=a.directions)
    if a.mode=="eval":
        evaluate(a.source,a.out,a.births,a.arms,cfg,a.updates,a.worlds,a.test_seed)
        return
    a.out.mkdir(parents=True,exist_ok=True)
    plan=dict(config=asdict(cfg),births=a.births,arms=a.arms,
              source_sha256={str(b):sha256(previous.checkpoint_path(a.source,"odor","uniform",b,8000)
                                         .read_bytes()).hexdigest() for b in a.births},
              code_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
              note="Diagnostic outer search over only 20 intrinsic write susceptibilities (one common value for global). "
                   "All start at gain 1; every local lifetime write, including topology turnover, remains live. Gains scale "
                   "the receiver traces during commit only, not their stored values or motor activation. Nonzero continuous "
                   "gains have range (0,2). Eight private sign assignments of old A/B and new X crossed per world. "
                   "A or B is dormant during the last third of the interference. The terminal request samples old A, old B, "
                   "or X with equal probability and only its sampled physical resource guides antithetic rule selection.")
    manifest=a.out/"run.json"
    if manifest.exists() and json.loads(manifest.read_text())!=plan:raise ValueError("Frozen susceptibility protocol changed")
    common._atomic_save(manifest,lambda f:f.write_text(json.dumps(plan,indent=2)+"\n",encoding="utf8"))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in [pool.submit(fit,a.source,a.out,a.births,arm,cfg) for arm in a.arms]:f.result()


if __name__=="__main__":main()
