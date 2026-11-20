"""Can repeated interleaved reward samples be treated as evidence when some are misleading?"""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict,dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import interleave_age_train as age, interleave_credit_train as credit, interleave_train as common
from . import interleave_susceptibility as susceptibility, local_trace_graph as net, local_trace_train as previous
from .interleave_interference import DATA,EVIDENCE


ARMS=("clean","noisy","mixed")
DOMAIN=654281


@dataclass(frozen=True)
class Config:
    updates:int=8000
    worlds:int=8
    contacts:int=16
    lr:float=.0015
    seed:int=435168
    checkpoint_every:int=500


def make(config,birth,step,ecology,*,domain=DOMAIN):
    n,T=config.worlds,config.contacts
    rng=np.random.default_rng([domain,config.seed,birth,step])
    row=np.arange(n)
    v=age.directions(rng,n,2)
    first=rng.integers(2,size=n)
    ids=np.concatenate((np.stack((first,1-first),1),rng.integers(2,size=(n,T-2))),1)
    ori=age.signs(rng,(n,T))
    regime=rng.random(n)<.5
    flip_coin=rng.random((n,T))
    outcome_coin=rng.random((n,T))
    hazard=np.zeros(n)
    noise=np.zeros(n)
    if ecology=="noisy":noise[:]=.2
    elif ecology=="mixed":
        noise=np.where(regime,.10,.20)
        hazard=np.where(regime,.16,0.)
    elif ecology!="clean":raise ValueError(ecology)
    law=np.tile(np.array([[-1,-1],[-1,1],[1,-1],[1,1]],dtype=np.float32),(n,1,1))
    visited=np.zeros((n,2),dtype=bool)
    history=[]
    for t in range(T):
        item=ids[:,t]
        flip=visited[row,item]&(flip_coin[:,t]<hazard)
        law[row,:,item]*=np.where(flip,-1,1)[:,None]
        history.append(law[row,:,item].copy())
        visited[row,item]=True
    private=np.stack(history,2)
    factor=private*ori[:,None]
    slips=np.where(outcome_coin<noise[:,None],-1.,1.).astype(np.float32)
    def shared(x):return np.repeat(x,4,axis=0)
    return dict(right=shared(v[np.arange(n)[:,None],ids]*ori[...,None]),obj=shared(ids),orient=shared(ori),
                clean=factor.reshape(-1,T),earned=(factor*slips[:,None]).reshape(-1,T),
                draw=shared(rng.random((n,T,5),dtype=np.float32)),wait=shared(rng.integers(0,9,size=(n,T))),
                high_vol=shared(regime if ecology=="mixed" else np.zeros(n,dtype=bool)))


def batch(cfg,births,step,ecology,domain):
    parts=[make(cfg,b,step,ecology,domain=domain) for b in births]
    return {k:torch.from_numpy(np.concatenate([v[k] for v in parts])) for k in parts[0]}


def run_life(g,r,w,cfg,*,sham=False,blank=False):
    roots=len(g.bias);lanes=cfg.worlds*4
    state=net.start(g,lanes);rule=r.expand(lanes)
    returns=[];motor=[];edit=[];chosen=[]
    for t in range(cfg.contacts):
        if t and t%4==0:state=credit.boundary(state,"all")
        if t:state=net.wait(rule,state,w["wait"][:,t],max_ticks=8)
        x,u=w["right"][:,t],w["draw"][:,t]
        a=net.act(rule,state,-x,x,u[:,0])
        reward=(a.side*w["earned"][:,t]).detach()
        e=net.observe(rule,a,-x,x,torch.zeros_like(reward) if blank else reward)
        result=net.commit(rule,e,u[:,1:],intervention="sham" if sham else "active")
        state=result.state
        returns.append(reward);motor.append(a.motor_log_prob);edit.append(result.edit_log_prob);chosen.append(a.side)
    def pack(xs):return torch.stack(xs,1).reshape(roots,cfg.worlds,4,cfg.contacts)
    return pack(returns),pack(motor),pack(edit),pack(chosen)


def fit(out,source,births,arm,cfg,stop):
    torch.set_num_threads(1)
    g,r=susceptibility.parents(source,births)
    # The helper loads a frozen template. Explicitly re-enable every birth coefficient here.
    g=net.Graph(torch.nn.Parameter(g.weights),torch.nn.Parameter(g.bias),g.present)
    r.requires_grad_(True)
    params=(g.weights,g.bias,*r.parameters())
    opt=torch.optim.Adam(params,lr=cfg.lr)
    folder=out/arm;folder.mkdir(exist_ok=True,parents=True)
    files=sorted(folder.glob("step-*.pt"));step=0;log=[]
    if files:
        d=torch.load(files[-1],weights_only=True,map_location="cpu")
        if (d["births"],d["arm"],d["config"])!=(list(births),arm,asdict(cfg)):raise ValueError("Wrong fit")
        with torch.no_grad():g.weights.copy_(d["graph"]["weights"]);g.bias.copy_(d["graph"]["bias"])
        r.load_state_dict(d["rule"]);opt.load_state_dict(d["optimizer"]);step=d["step"];log=d["trace"].tolist()
    def save():
        payload=dict(births=list(births),step=step,arm=arm,config=asdict(cfg),trace=torch.as_tensor(log),
                     graph=dict(weights=g.weights.detach(),bias=g.bias.detach(),present=g.present),
                     rule=r.state_dict(),optimizer=opt.state_dict())
        common._atomic_save(age.ckpath(out,arm,step),lambda f:torch.save(payload,f))
    if not files:save()
    first=step;started=perf_counter()
    for step in range(first+1,stop+1):
        data=batch(cfg,births,step,arm,DOMAIN)
        opt.zero_grad(set_to_none=True)
        reward,ms,es,side=run_life(g,r,data,cfg)
        q=(torch.arange(cfg.contacts)>=2).float()/(cfg.contacts-2)
        adv=common.causal_advantage(reward,q.expand_as(reward))
        loss=-(ms*adv).sum(-1).mean((1,2)).sum()-(es[...,:-1]*adv[...,1:]).sum(-1).mean((1,2)).sum()
        loss.backward()
        with torch.no_grad():
            norm=sum(p.grad.flatten(1).double().square().sum(1) for p in params).sqrt()
            if not torch.isfinite(norm).all():raise FloatingPointError(f"Nonfinite {arm}/{step}")
            scale=(2/norm.clamp_min(1e-12)).clamp_max(1)
            for p in params:p.grad.mul_(scale.reshape((-1,)+(1,)*(p.ndim-1)).to(p.dtype))
            physical=side==data["clean"].reshape_as(side)
            stats=torch.stack(((reward[...,2:]>0).float().mean((1,2,3)),physical[...,2:].float().mean((1,2,3)),
                               physical[...,-4:].float().mean((1,2,3)),norm.float()),1)
        opt.step(); log.append(stats.tolist())
        if step%cfg.checkpoint_every==0 or step==stop:save()
        if step%500==0 or step in (1,stop):
            print(arm,step,np.mean(log[-100:],axis=(0,1)).round(3),round((perf_counter()-started)/(step-first),3),flush=True)


def reference(w,ecology):
    """Public-evidence filter. Unknown volatility is inferred jointly across both objects."""
    right=w["right"].numpy();u=w["draw"].numpy()[:,:,0]
    factor=w["earned"].numpy();n,T=factor.shape
    # Map visible directions to one of two remembered physical items and orientations.
    # The first two presentations are guaranteed distinct, so they supply the two keys.
    direction=right/np.linalg.norm(right,axis=2,keepdims=True)
    keys=direction[:,:2].copy()
    match=np.einsum("ntd,nkd->ntk",direction,keys)
    item=np.abs(match).argmax(2)
    orient=np.sign(np.take_along_axis(match,item[...,None],2)[...,0])
    if np.min(np.abs(np.take_along_axis(match,item[...,None],2)))<.99999:
        raise AssertionError("The visible reference was given an unknown cue")
    classes=[(.20,0),(.10,.16)] if ecology=="mixed" else [(.20,0)] if ecology=="noisy" else [(0.,0.)]
    posterior=np.full((n,len(classes),2),.5)
    mix=np.full((n,len(classes)),1/len(classes))
    seen=np.zeros((n,2),dtype=bool)
    rows=np.arange(n);side=np.zeros((n,T),dtype=np.int8)
    for t in range(T):
        obj=item[:,t];observed=seen[rows,obj]
        for j,(_,h) in enumerate(classes):
            x=posterior[rows,j,obj]
            posterior[rows,j,obj]=np.where(observed,h+(1-2*h)*x,x)
        expected=np.zeros(n)
        for j,(p,_) in enumerate(classes):
            expected+=mix[:,j]*(1-2*p)*(2*posterior[rows,j,obj]-1)*orient[:,t]
        side[:,t]=np.where(expected>1e-9,1,np.where(expected< -1e-9,-1,np.where(u[:,t]<.5,1,-1)))
        taste=factor[:,t]*side[:,t]
        ev=taste*side[:,t]*orient[:,t]
        for j,(p,_) in enumerate(classes):
            prior=posterior[rows,j,obj]
            positive=np.where(ev>0,1-p,p)
            likelihood=prior*positive+(1-prior)*(1-positive)
            posterior[rows,j,obj]=prior*positive/likelihood.clip(1e-20)
            mix[:,j]*=likelihood
        mix/=mix.sum(1,keepdims=True).clip(1e-20)
        seen[rows,obj]=True
    return side


@torch.no_grad()
def evaluate(out,source,births,arms,step,worlds,seed,T):
    torch.set_num_threads(1)
    cfg=Config(worlds=worlds,seed=seed,contacts=T)
    dest=out/"validation";dest.mkdir(parents=True,exist_ok=True)
    report=dict(seed=seed,contacts=T,worlds=worlds,step=step,births=births,results={},sha256={})
    for b in births:
        cells={};raw={}
        for ecology in ARMS:
            w=batch(cfg,[b],1,ecology,DOMAIN+1)
            truth=w["clean"].numpy()
            ref=reference(w,ecology)
            methods={"public_filter":ref}
            for model in ("parent",*arms):
                if model=="parent":g,r=susceptibility.parents(source,[b])
                else:g,r=age.load_checkpoint(out,model,b,step)
                for treatment in ("true","blank","sham"):
                    *_,s=run_life(g,r,w,cfg,sham=treatment=="sham",blank=treatment=="blank")
                    methods[model+"/"+treatment]=s.reshape_as(w["clean"]).numpy()
            for method,s in methods.items():
                good=s==truth;earned=s*w["earned"].numpy()
                cells[f"{ecology}/{method}"]=dict(correct=float(good[:,2:].mean()),
                    early=float(good[:,2:6].mean()),late=float(good[:,-8:].mean()),
                    positive_resource=float((earned[:,2:]>0).mean()))
                raw[ecology+"/"+method]=s.astype(np.int8)
            for field in ("clean","earned","obj","high_vol"):raw[ecology+"/world_"+field]=w[field].numpy()
        path=dest/f"{b}-{step}-{T}.npz";np.savez_compressed(path,**raw)
        report["sha256"][path.name]=sha256(path.read_bytes()).hexdigest()
        report["results"][b]=cells
        print("evaluated",b,flush=True)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/f"reliability-{step}-{T}.json").write_text(json.dumps(report,indent=1)+"\n",encoding="utf8")


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("fit","eval"),required=True)
    p.add_argument("--out",type=Path,default=DATA/"reliability-fit")
    p.add_argument("--source",type=Path,default=DATA/"retention-objective")
    p.add_argument("--births",nargs="+",type=int,default=list(previous.BIRTHS))
    p.add_argument("--arms",nargs="+",choices=ARMS,default=ARMS)
    p.add_argument("--updates",type=int,default=8000);p.add_argument("--worlds",type=int,default=8)
    p.add_argument("--stop",type=int);p.add_argument("--workers",type=int,default=2)
    p.add_argument("--eval-worlds",type=int,default=96);p.add_argument("--eval-contacts",type=int,default=16)
    p.add_argument("--eval-seed",type=int,default=1319277)
    a=p.parse_args();cfg=Config(updates=a.updates,worlds=a.worlds)
    stop=a.stop or a.updates
    if cfg.worlds<2 or not 1<=stop<=a.updates:raise ValueError("Invalid reliability budget")
    if a.mode=="eval":
        evaluate(a.out,a.source,a.births,a.arms,stop,a.eval_worlds,a.eval_seed,a.eval_contacts);return
    a.out.mkdir(parents=True,exist_ok=True)
    plan=dict(config=asdict(cfg),births=a.births,arms=a.arms,source=str(a.source),
              code_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
              note="All actual gains/losses are delivered, including stochastic slips. From contact 3, earned resources "
                   "count toward the outer criterion. Private clean correct-side is a diagnostic only. Two independent "
                   "hidden values are fully crossed within every world. States are never reset within a life; gradients "
                   "through h/trace/graph truncate every four contacts, previous sampled decisions retain future credit. "
                   "Clean has no slips or flips. Noisy has 20% iid slips. Mixed gives half the lives the same noisy law, "
                   "the rest 10% slips and .16 hidden flip probability at each re-encounter, with no regime cue.")
    manifest=a.out/"run.json"
    if manifest.exists() and json.loads(manifest.read_text())!=plan:raise ValueError("Frozen protocol mismatch")
    common._atomic_save(manifest,lambda f:f.write_text(json.dumps(plan,indent=2)+"\n",encoding="utf8"))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in [pool.submit(fit,a.out,a.source,a.births,arm,cfg,stop) for arm in a.arms]:f.result()


if __name__=="__main__":main()
