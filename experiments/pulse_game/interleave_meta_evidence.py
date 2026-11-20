"""Can other objects teach the graph whether to trust an anomalous receipt on a new one?"""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass,asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import json
import ast

import numpy as np
import torch

from . import interleave_age_train as age, interleave_credit_train as credit, interleave_train as common
from . import interleave_susceptibility as susceptibility, local_trace_graph as net, local_trace_train as old
from .interleave_interference import DATA,EVIDENCE


P=.15
H=2*P*(1-P)
DOMAIN=451739
ARMS=("direct","scaffold")


@dataclass(frozen=True)
class Config:
    updates:int=6000
    worlds:int=4
    seed:int=345981
    lr:float=.0015
    save_every:int=500


def parameters(arm,step):
    if arm=="direct":return P,H
    mix=min(1,max(0,(step-1000)/1800))
    return .01*(1-mix)+P*mix,.65*(1-mix)+H*mix


def make(cfg,birth,step,n,arm,*,domain=DOMAIN,conditioning=None):
    """Past history differs; the next object's observable three-evidence string is identical."""
    p,h=parameters(arm,step) if conditioning is None else conditioning
    rng=np.random.default_rng([domain,cfg.seed,birth,step])
    worlds=cfg.worlds
    x=age.directions(rng,worlds,3)
    # Across worlds, the two old objects are independent and the new one is unrelated to both.
    ids=rng.permuted(np.tile(np.repeat(np.arange(2),n),(worlds,1)),axis=1)
    old_sign=age.signs(rng,(worlds,2))
    coins=rng.random((worlds,2*n))
    stable=old_sign[np.arange(worlds)[:,None],ids]*np.where(coins<p,-1,1)
    volatile=old_sign.copy();past=[];seen=np.zeros((worlds,2),dtype=bool);row=np.arange(worlds)
    for t in range(2*n):
        j=ids[:,t]
        change=seen[row,j]&(coins[:,t]<h)
        volatile[row,j]=np.where(change,-volatile[row,j],volatile[row,j])
        past.append(volatile[row,j].copy());seen[row,j]=True
    volatility=np.stack(past,1) if past else np.zeros((worlds,0))
    pattern=np.array([(r,s,xs) for r in (0,1) for s in (-1,1) for xs in (-1,1)],dtype=np.int8)
    regime=np.tile(pattern[:,0],worlds)
    mirror=np.tile(pattern[:,1],worlds)
    s=np.tile(pattern[:,2],worlds)
    def common(v):return np.repeat(v,8,axis=0)
    prior=np.where(regime[:,None]==0,common(stable),common(volatility))*mirror[:,None]
    evidence=s[:,None]*np.array([1,1,-1],dtype=np.float32)
    # Under stable .15 iid noise, after ++- the posterior on the true + law is .85.
    # The probability of another + physical receipt is .85**2+.15**2=1-H.
    # In the reliable Markov regime the next sign remains the last sign with 1-H.
    final_sign=s*(1-2*regime)*common(np.where(rng.random(worlds)<H,-1,1))
    law=np.column_stack((prior,evidence,final_sign)).astype(np.float32)
    all_ids=np.column_stack((ids,np.full((worlds,4),2,dtype=np.int64)))
    orientation=age.signs(rng,all_ids.shape)
    right=x[row[:,None],all_ids]*orientation[...,None]
    return dict(right=common(right),factor=law*common(orientation),
                u=common(rng.random((*all_ids.shape,5),dtype=np.float32)),
                gap=common(rng.integers(0,9,size=all_ids.shape)),canonical=law,
                orient=common(orientation),regime=regime,mirror=mirror,x_sign=s,ids=common(all_ids))


def batch(cfg,births,step,n,arm,*,domain=DOMAIN,conditioning=None):
    records=[make(cfg,b,step,n,arm,domain=domain,conditioning=conditioning) for b in births]
    return {k:torch.from_numpy(np.concatenate([r[k] for r in records])) for k in records[0]}


def replay(g,r,t,n,cfg,*,mode="natural",cut=True,return_history=False):
    lanes=8*cfg.worlds;count=len(t["right"]);view=r.expand(lanes)
    state=net.start(g,lanes);score=g.weights.new_zeros(count)
    twin=torch.arange(count)^4
    history=[]
    for i in range(2*n+4):
        if cut and i and i%4==0:state=credit.boundary(state,"all")
        if i:state=net.wait(view,state,t["gap"][:,i],max_ticks=8)
        if i==2*n:
            if mode in ("swap_graph","swap_activity"):
                state=net.swap(state,state.subset(twin),"graph" if mode=="swap_graph" else ("hidden","trace"))
        x,u=t["right"][:,i],t["u"][:,i]
        a=net.act(view,state,-x,x,u[:,0])
        if mode=="same_new_contacts" and 2*n<=i<2*n+3:
            # Same physical side and same earned receipt for X under both history regimes.
            stable=torch.arange(count)&~4
            a=net.act(view,state,-x,x,u[:,0],side_override=a.side[stable])
        reward=(a.side*t["factor"][:,i]).detach()
        score=score+a.motor_log_prob
        if i<2*n+3:
            feedback=reward
            if i<2*n:
                if mode=="blank_prefix":feedback=torch.zeros_like(reward)
                elif mode=="opposite_prefix":feedback=a.side*t["factor"][twin,i]
            z=net.observe(view,a,-x,x,feedback)
            written=net.commit(view,z,u[:,1:],intervention="sham" if mode=="sham_prefix" and i<2*n else "active")
            state=written.state;score=score+written.edit_log_prob
            if return_history:history.append(torch.stack((a.side,feedback),1))
    return score,reward,a,(torch.stack(history,1) if return_history else None)


def reference(t,physical,n,p,h):
    # Identify objects and their sign relative to the first observed vector with that identity.
    # Object coordinates, latent signs and regime identifiers are never consulted.
    right=t["right"].numpy();hist=physical.numpy();count=len(right)
    likelihood=np.zeros(count)
    for lane in range(count):
        keys=[];evidence=[]
        for i in range(2*n):
            v=right[lane,i];chosen,taste=hist[lane,i]
            if not keys:found=-1
            else:
                matches=np.array([abs(float((q*v).sum()))/1.35**2 for q in keys])
                found=int(matches.argmax()) if matches.max()>.999995 else -1
            if found<0:
                keys.append(v);evidence.append([]);found=len(keys)-1
            sign=1 if (keys[found]*v).sum()>0 else -1
            evidence[found].append(taste*chosen*sign)
        for sequence in evidence:
            a=np.asarray(sequence);plus=(a>0).sum();minus=len(a)-plus
            log_stable=np.logaddexp(plus*np.log(1-p)+minus*np.log(p),
                                   minus*np.log(1-p)+plus*np.log(p))-np.log(2)
            changes=(a[1:]!=a[:-1]).sum()
            log_fast=-np.log(2)+changes*np.log(h)+(len(a)-1-changes)*np.log(1-h)
            likelihood[lane]+=log_stable-log_fast
    # Balanced experimental conditioning supplies the same new evidence under either regime;
    # it carries no additional information about which of the two calibration worlds occurred.
    # Re-express the three experienced tastes in the publicly visible orientation
    # of the current menu. No world identifier or hidden sign enters the decision.
    new_right = right[:, 2*n:2*n+3]
    orientation = np.sign((new_right * right[:, -1:, :]).sum(axis=2))
    new_history = hist[:, 2*n:2*n+3]
    majority = np.sign((new_history[:, :, 0] * new_history[:, :, 1] * orientation).sum(axis=1))
    u=t["u"][:,-1,0].numpy()
    choice=np.where(likelihood>1e-7,majority,np.where(likelihood< -1e-7,-majority,np.where(u<.5,1,-1)))
    return choice,likelihood


def fit(out,source,births,arm,cfg,stop):
    torch.set_num_threads(1)
    g,r=susceptibility.parents(source,births)
    g=net.Graph(torch.nn.Parameter(g.weights),torch.nn.Parameter(g.bias),g.present);r.requires_grad_(True)
    params=(g.weights,g.bias,*r.parameters())
    opt=torch.optim.Adam(params,lr=cfg.lr)
    folder=out/arm;folder.mkdir(parents=True,exist_ok=True)
    files=sorted(folder.glob("step-*.pt"));first=0;trace=[]
    if files:
        d=torch.load(files[-1],weights_only=True,map_location="cpu")
        if (d["births"],d["config"],d["arm"])!=(list(births),asdict(cfg),arm):raise ValueError("Wrong run")
        with torch.no_grad():g.weights.copy_(d["graph"]["weights"]);g.bias.copy_(d["graph"]["bias"])
        r.load_state_dict(d["rule"]);opt.load_state_dict(d["optimizer"])
        first,trace=d["step"],d["trace"].tolist()
    step=first
    def save():
        d=dict(config=asdict(cfg),births=list(births),step=step,arm=arm,trace=torch.tensor(trace),
               graph=dict(weights=g.weights.detach(),bias=g.bias.detach(),present=g.present),
               rule=r.state_dict(),optimizer=opt.state_dict())
        common._atomic_save(age.ckpath(out,arm,step),lambda f:torch.save(d,f))
    if not files:save()
    start=perf_counter()
    for step in range(first+1,stop+1):
        n=int(np.random.default_rng([DOMAIN,cfg.seed,step,711]).choice((3,5,8)))
        data=batch(cfg,births,step,n,arm)
        opt.zero_grad(set_to_none=True)
        score,earned,a,_=replay(g,r,data,n,cfg)
        rew=earned.reshape(len(births),cfg.worlds,8)
        means=rew.mean(2,keepdim=True)
        adv=rew-(means.sum(1,keepdim=True)-means)/(cfg.worlds-1)
        loss=-(score.reshape_as(rew)*adv).mean((1,2)).sum();loss.backward()
        params_grad=[p for p in params if p.grad is not None]
        with torch.no_grad():
            norm=sum(p.grad.flatten(1).double().square().sum(1) for p in params_grad).sqrt()
            if not torch.isfinite(norm).all():raise FloatingPointError(f"{arm}/{step}")
            weight=(2/norm.clamp_min(1e-12)).clamp_max(1)
            for p in params_grad:p.grad.mul_(weight.reshape((-1,)+(1,)*(p.ndim-1)).to(p.dtype))
            follow=(a.side==data["x_sign"]*data["orient"][:,-1]).reshape_as(rew)
            stats=torch.stack(((rew>0).float().mean((1,2)),follow[:,:,:4].float().mean((1,2)),
                               follow[:,:,4:].float().mean((1,2)),norm.float()),1)
        opt.step();trace.append(stats.tolist())
        if step%cfg.save_every==0 or step==stop:save()
        if step%250==0 or step in (1,stop):print(arm,step,np.mean(trace[-100:],axis=(0,1)).round(3),
                                               round((perf_counter()-start)/(step-first),3),flush=True)


@torch.no_grad()
def evaluate(out,model,label,births,worlds,seed):
    torch.set_num_threads(1)
    dest=out/"validation"/label;dest.mkdir(parents=True,exist_ok=True)
    cfg=Config(worlds=worlds,seed=seed);scores={};hashes={}
    for b in births:
        g,r=model(b);cells={};raw={}
        for task,(p,h) in (("matched",(P,H)),("easy",(.01,.65))):
            for n in (0,2,4,8,16):
                t=batch(cfg,[b],1,n,"direct",domain=DOMAIN+5,conditioning=(p,h))
                majority=(t["x_sign"]*t["orient"][:,-1]).numpy()
                reg=t["regime"].numpy().astype(bool); pay=t["factor"][:,-1].numpy()
                for mode in ("natural","same_new_contacts","blank_prefix","opposite_prefix","sham_prefix",
                             "swap_graph","swap_activity"):
                    _,earned,a,hist=replay(g,r,t,n,cfg,mode=mode,return_history=True)
                    side=a.side.numpy();follow=side==majority
                    if mode=="same_new_contacts":
                        part=hist[:,2*n:2*n+3].reshape(worlds,2,4,3,2)
                        assert torch.equal(part[:,0],part[:,1])
                    key=f"{task}/{n}/{mode}"
                    cells[key]=dict(positive=float((earned>0).float().mean()),stable_follow_majority=float(follow[~reg].mean()),
                                    volatile_follow_majority=float(follow[reg].mean()),
                                    history_effect=float(follow[~reg].mean()-follow[reg].mean()))
                    raw[key]=side.astype(np.int8)
                    if mode=="natural":
                        chosen,e=reference(t,hist,n,p,h)
                        ref=chosen==majority
                        cells[key]["reference_positive"]=float((chosen*pay>0).mean())
                        cells[key]["reference_history_effect"]=float(ref[~reg].mean()-ref[reg].mean())
                        raw[key+"/public_side"]=chosen.astype(np.int8)
                        raw[key+"/log_likelihood_ratio"]=e
                    if (mode=="blank_prefix" or n==0) and abs(cells[key]["positive"]-.5)>1e-8:
                        raise AssertionError("Regime was visible without informative calibration")
        f=dest/f"{b}.npz";np.savez_compressed(f,**raw);hashes[f.name]=sha256(f.read_bytes()).hexdigest()
        scores[b]=cells;print(label,b,flush=True)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/f"meta-evidence-{label}.json").write_text(json.dumps(dict(births=births,worlds=worlds,seed=seed,
                                            results=scores,sha256=hashes),indent=1)+"\n",encoding="utf8")


def fit_definitions(source):
    # An archived run may be reopened after a reporting-only change. Compare
    # the actual fitting definitions and module constants before reusing it.
    program=ast.parse(source)
    names={"Config","parameters","make","batch","replay","fit"}
    return [ast.dump(node) for node in program.body
            if isinstance(node,ast.Assign) or getattr(node,"name","") in names]


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("fit","eval"),required=True);p.add_argument("--out",type=Path,default=DATA/"meta-evidence")
    p.add_argument("--source",type=Path,default=DATA/"retention-objective")
    p.add_argument("--births",nargs="+",type=int,default=list(old.BIRTHS));p.add_argument("--arms",nargs="+",choices=ARMS,default=ARMS)
    p.add_argument("--updates",type=int,default=6000);p.add_argument("--stop",type=int);p.add_argument("--workers",type=int,default=1)
    p.add_argument("--worlds",type=int,default=4);p.add_argument("--eval-worlds",type=int,default=64);p.add_argument("--eval-seed",type=int,default=815139)
    a=p.parse_args();cfg=Config(updates=a.updates,worlds=a.worlds)
    stop=a.stop or a.updates
    if a.mode=="eval":
        for arm in a.arms:
            evaluate(a.out,lambda b:age.load_checkpoint(a.out,arm,b,stop),arm+"-"+str(stop),a.births,a.eval_worlds,a.eval_seed)
        return
    if cfg.worlds<2 or not 1<=stop<=a.updates:raise ValueError("Invalid fit budget")
    a.out.mkdir(parents=True,exist_ok=True)
    plan=dict(config=asdict(cfg),births=list(a.births),arms=list(a.arms),stable_slip=P,volatile_hazard=H,
              note="Calibrating objects have the same one-visit sign change probability 2p(1-p) in both final regimes. "
                   "Stable/noisy versus volatile/reliable is unobserved. New object's visible signed evidence is "
                   "always ++- or --+, matched across the two regimes; the final physical payoff favors majority or "
                   "last taste with equal and opposite odds. No private law or correct motor in the forward policy. "
                   "Scaffold starts with easier prior-object temporal statistics, with identical target X. Both use "
                   "matched natural statistics from step 2800. At evaluation, same-new-contacts yokes all actual X "
                   "support sides across regimes to hold the literal experienced X feedback exactly fixed.",
              parent_sha256={str(b):sha256(old.checkpoint_path(a.source,"odor","uniform",b,8000).read_bytes()).hexdigest()
                             for b in a.births},code_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    manifest=a.out/"run.json"
    if manifest.exists():
        previous=json.loads(manifest.read_text())
        if previous!=plan:
            digest=previous["code_sha256"]
            archive=a.out/f"source-used-{digest[:12]}.py"
            if ({**plan,"code_sha256":digest}!=previous or not archive.exists()
                    or sha256(archive.read_bytes()).hexdigest()!=digest
                    or fit_definitions(archive.read_text(encoding="utf8"))!=fit_definitions(Path(__file__).read_text(encoding="utf8"))):
                raise ValueError("Frozen fitting plan changed")
    else:
        common._atomic_save(manifest,lambda f:f.write_text(json.dumps(plan,indent=2)+"\n",encoding="utf8"))
    if a.workers==1:
        for arm in a.arms:fit(a.out,a.source,a.births,arm,cfg,stop)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for f in [pool.submit(fit,a.out,a.source,a.births,arm,cfg,stop) for arm in a.arms]:f.result()


if __name__=="__main__":main()
