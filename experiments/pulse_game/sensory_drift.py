"""A changing public sensory signature with stable, apart (or close) private habitats."""
from pathlib import Path
import json
import numpy as np
import torch
from . import longrun_world as env,adaptive_local_gpu as control,adaptive_local as native
from . import unified_physical as physics,independent_births as births,longrun_plastic as distributed
from . import mixture_flash,symmetry_physics as gpu,research_audit as audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


def shifted(seeds,*,blocks,amplitude,gap,noise,nonce):
    w=env.make(seeds,blocks=blocks,mode="stable",noise=.05,nonce=nonce)
    n=len(seeds);tr=w.arena.menu.body.trials;time=np.arange(tr)[None,:]/tr
    directions=np.empty((n,tr),np.float32)
    for i,s in enumerate(seeds):
        rng=np.random.default_rng(np.random.SeedSequence([s,nonce,3173]))
        phase=rng.uniform(0.,2*np.pi);which=rng.integers(2)
        delta=amplitude*np.sin(2*np.pi*(1.7*time[0])+phase)
        base=.78+np.where(w.arena.menu.body.ctx[i]==which,gap/2,-gap/2)
        directions[i]=base+delta+rng.normal(0,noise,tr)
    # Both cues move together. With zero measurement noise the angular gap at
    # a particular time stays constant, including the close-habitat protocol.
    w.arena.appearance[:]=np.stack([np.cos(directions),np.sin(directions)],2)
    return w


def addresses(appearance,ct,cap,tau):
    n,l,_=appearance.shape;r=np.arange(n)
    p=np.zeros((n,cap,2),np.float32);cnt=np.zeros((n,cap),int);present=np.full(n,255,np.uint8)
    code=np.zeros((n,l,16),np.uint8);born=np.full((n,l),255,np.uint8)
    for j in range(l):
        x=appearance[:,j]/np.maximum(np.linalg.norm(appearance[:,j],axis=1,keepdims=True),1e-8)
        exist=cnt>0;sim=np.where(exist,(p*x[:,None]).sum(2),-np.inf)
        hit=sim.argmax(1);alloc=(sim.max(1)<np.cos(.09))&(~exist).any(1)
        hit[alloc]=(~exist[alloc]).argmax(1);born[alloc,j]=hit[alloc]
        w=cnt[r,hit] if tau<=0 else np.minimum(cnt[r,hit],tau)
        update=p[r,hit]*w[:,None]+x;p[r,hit]=update/np.maximum(np.linalg.norm(update,axis=1,keepdims=True),1e-8)
        cnt[r,hit]+=1
        code[:,j]=np.where(np.arange(16)[None,:]>=ct[:,j,None],hit[:,None],present[:,None])
        present=hit.astype(np.uint8)
    return code,born,cnt


def one(w,front,route,motion,body,alpha,constants):
    co=control.Cohort(w,front,route,motion)
    _,clock=mixture_flash.timing(w.arena.menu.body.seeds,co.tr)
    lut=np.vstack((np.eye(6,dtype=np.float32),np.zeros((1,6),np.float32)))
    rows={}
    for cap in constants:
        slots,born,counts=addresses(w.arena.appearance,clock,6,cap)
        co.address=gpu.tensor(lut[np.where(slots==255,6,slots)]);co.born=gpu.tensor(born)
        result=co.play(body,alpha);ix=w.query_at
        hit=(result.contact_side==w.arena.truths)&(result.contact_side>=0)
        indices=w.block_context
        at=slots[:,ix[:,1],5];dominant=[]
        for c in range(2):
            hits=np.stack([np.sum((at==s)&(indices==c),axis=1) for s in range(6)],axis=1)
            dominant.append(hits.max(1)/hits.sum(1).clip(min=1))
        rows[str(cap)]=dict(pre=hit[:,ix[:,0]].mean(1).tolist(),post=hit[:,ix[:,1]].mean(1).tolist(),
            last_pre=hit[:,ix[-24:,0]].mean(1).tolist(),last_post=hit[:,ix[-24:,1]].mean(1).tolist(),
            slots=(counts>0).sum(1).tolist(),dominant=np.mean(dominant,axis=0).tolist())
    return rows


def run():
    torch.set_num_threads(1);root=OUT/"independent-lifelong-births-v1";reg=json.loads((root/"registered.json").read_text())
    alpha=reg["reopening_rule"];selected=reg["fixed_birth_seeds"][:3];dest=audit.OUT/"sensory-drift";dest.mkdir(exist_ok=True)
    params=[]
    for b in selected:
        folder=root/f"birth-{b}";f,r,m=births.resources(folder)
        v=json.loads(next(folder.glob("*--full-*/frozen.json")).read_text())["best"]["genome"]
        params.append((f,r,m,physics.phenotype(physics.params(f,r,m),v,"full")[0]))
    ts=[0,2,4,8,16,32,64,128]
    dev=np.random.default_rng(56620341).integers(7_000_000,465_000_000,48).tolist()
    values=[]
    for amp,gap,noise in ((0.,.7,.06),(.34,.7,.025),(.34,.29,.04)):
        w=shifted(dev,blocks=96,amplitude=amp,gap=gap,noise=noise,nonce=825731)
        d=one(w,*params[0],alpha,ts)
        values.append([.5*np.mean(d[str(t)]["pre"])+.5*np.mean(d[str(t)]["post"]) for t in ts])
    best=int(ts[int(np.asarray(values).mean(0).argmax())])
    _write(dest/"frozen.json",dict(tau=best,grid=ts,seeds=dev,objective="actual pre and post physical choices",scores=values))
    roots=np.random.default_rng(31906219).integers(8_000_000,465_000_000,128).tolist()
    for seed,p in zip(selected,params,strict=True):
        for tag,amp,gap,noise in (("stable",0.,.7,.02),("drift",.34,.7,.025),("narrow_drift",.34,.29,.04),("narrow_static",0.,.29,.04)):
            rows={}
            for off in range(0,len(roots),32):
                ss=roots[off:off+32];w=shifted(ss,blocks=192,amplitude=amp,gap=gap,noise=noise,nonce=3173937)
                ans=one(w,*p,alpha,sorted(set((0,best))))
                full=distributed.Cohort(w,*p[:3]);o=full.adjustments(p[3],alpha,detail=True)[0]
                correct=(o.contact_side==w.arena.truths)&(o.contact_side>=0);q=w.query_at
                ans["distributed"]=dict(pre=correct[:,q[:,0]].mean(1).tolist(),post=correct[:,q[:,1]].mean(1).tolist(),
                    last_pre=correct[:,q[-24:,0]].mean(1).tolist(),last_post=correct[:,q[-24:,1]].mean(1).tolist())
                for a,ds in ans.items():
                    for k,v in ds.items():rows.setdefault(a,{}).setdefault(k,[]).extend(v)
            _write(dest/f"{seed}-{tag}.json",dict(birth=seed,case=tag,tau=best,seeds=roots,rows=rows))
    _write(dest/"complete.json",dict(births=selected,worlds=roots))


if __name__=="__main__":run()
