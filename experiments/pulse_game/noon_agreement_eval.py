"""Evaluate shopping after contradictory tastes while holding exposure count and opportunity fixed."""

from argparse import ArgumentParser
from hashlib import sha256
from itertools import product
from pathlib import Path

import numpy as np
import torch

from . import noon_receipt_agreement as game,noon_market_stats as statistics


def scene(seed,worlds,price,visits,slip=.2):
    if visits==2:return game.tape(seed,0,0,worlds,domain=331631,price=price,slip=slip)
    base=game.tape(seed,0,0,worlds,domain=331631,price=price,slip=slip)
    labels=np.array(list(product((-1.,1.),repeat=visits+2)),np.float32)
    m=len(labels);base_rows=torch.arange(0,worlds*16,16).repeat_interleave(m)
    t={k:v[base_rows] for k,v in base.items()};rng=np.random.default_rng([183581,seed,visits])
    pattern=torch.from_numpy(np.tile(labels,(worlds,1)));t["pattern"]=pattern
    x=t["preview"];b=rng.normal(size=(worlds,3));b=torch.from_numpy(np.repeat((1.35*b/np.linalg.norm(b,axis=1,keepdims=True)).astype(np.float32),m,0))
    row=np.arange(visits+1)[None,:];bpos=rng.integers(visits+1,size=(worlds,1))
    order=np.repeat(np.where(row==bpos,visits,np.where(row<bpos,row,row-1)),m,0)
    order=torch.from_numpy(order);t["order"]=order
    orientation=torch.from_numpy(np.repeat(rng.choice(np.array([-1.,1.],np.float32),(worlds,visits+1)),m,0))
    t["study"]=torch.where(order[:,:,None]==visits,b[:,None,:],x[:,None,:])*orientation[:,:,None]
    true=pattern[:,0];signed=true[:,None]*pattern[:,2:]
    t["study_value"]=torch.cat((signed,pattern[:,1:2]),1).gather(1,order)*orientation
    t["final_value"]=true*torch.sign((t["final"]*x).sum(1));t["review_value"]=true*torch.sign((t["review"]*x).sum(1))
    t["weight"]=torch.where(pattern[:,2:]>0,1-slip,slip).prod(1)/4
    def pub(a):return torch.from_numpy(np.repeat(a,m,0))
    t["study_draws"]=pub(rng.random((worlds,visits+1,5),dtype=np.float32))
    t["study_gaps"]=pub(rng.integers(0,7,size=(worlds,visits+1)))
    t["scramble"]=pub(rng.choice(np.array([-1.,1.],np.float32),(worlds,visits+1)))
    return t


@torch.no_grad()
def measure(g,r,k,eta,t,arm,layout):
    view,s,_,trace,base,h=game.prefix(g,r,t,eta,arm)
    tariff=t["price"];seen=t["pattern"];n=len(seen);visits=seen.shape[1]-2;m=2**(2+visits);worlds=n//m
    lex=torch.arange(n)
    hidden_twin=lex^((1<<(visits+1))|((1<<visits)-1))
    second_twin=lex^1
    # Hidden X switched and all past receipt-relative signs switched: exactly
    # the same observable food history up to the ticket, despite opposite future law.
    if not torch.equal(h,h[hidden_twin]) or not torch.equal(t["final_value"],-t["final_value"][hidden_twin]):
        raise ValueError("The two hidden worlds are not observationally equivalent")
    def offer(tr):return game.sensed(t,tr,k,layout,1)
    normal=offer(trace);out=game.ticket.finish(view,s,normal,"native")
    if not torch.equal(out["choice"],out["choice"][hidden_twin]):raise ValueError("Route foresaw a hidden future value")
    ones=torch.ones_like(out["buy"]);a=game.ticket.finish(view,s,normal,"native",decision=ones)
    b=game.ticket.finish(view,s,normal,"native",decision=~ones)
    def chosen(flag):return torch.where(flag,a["utility"],b["utility"])
    torch.testing.assert_close(chosen(out["buy"]),out["utility"],rtol=0,atol=0)
    swap=game.ticket.finish(view,s,offer(trace[second_twin]),"native")
    zero=game.ticket.finish(view,s,offer(base),"native")
    wrong=game.ticket.finish(view,s,normal,"native",deliver="zero")
    weights=t["weight"].reshape(worlds,8,-1)
    matched=(out["buy"].reshape_as(weights).float()*weights).sum(2,keepdim=True)/weights.sum(2,keepdim=True)
    matched=matched.expand_as(weights).reshape(-1)
    # Count agreement using only the displayed odor and actual side/taste.
    matches=(t["order"]!=visits)
    observed=h[:,:,0]*h[:,:,1]*torch.sign((t["study"]*t["preview"][:,None]).sum(2))
    obs=observed[matches].reshape(n,visits)
    imbalance=obs.sum(1).abs()
    if not torch.equal(imbalance,seen[:,2:].sum(1).abs()):raise ValueError("Could not reconstruct receipt agreement")
    posterior=1/(1+(.2/.8)**imbalance)
    public=chosen(tariff<1-posterior)
    values=dict(buy=out["buy"].float(),gross=out["gross"],net=out["utility"],always_buy=a["utility"],
                always_skip=b["utility"],same_cost_gain=out["utility"]-(b["utility"]+matched*(a["utility"]-b["utility"])),
                public_history=public,extra_taste_gain=out["utility"]-wrong["utility"],value_of_information=a["gross"]-b["gross"],
                donor_trace_choice_loss=out["utility"]-chosen(swap["buy"]),remove_confirmation_choice_loss=out["utility"]-chosen(zero["buy"]),
                sensed=game.cov.projection(trace,t["preview"]),baseline_sensed=game.cov.projection(base,t["preview"]))
    w=t["weight"].reshape(worlds,m)
    subsets={"all":torch.ones_like(imbalance,dtype=torch.bool),"unanimous":imbalance==visits,"most_ambiguous":imbalance==visits%2}
    for x in range(visits%2,visits+1,2):subsets["margin"+str(x)]=imbalance==x
    stats={}
    for group,mask in subsets.items():
        ws=w*mask.reshape(worlds,m)
        if (ws.sum(1)==0).any():raise ValueError("Missing prior evidence cell")
        for key,v in values.items():stats[group+"/"+key]=((v.reshape(worlds,m)*ws).sum(1)/ws.sum(1)).numpy()
    stats["ambiguity_choice_gap"]=stats["most_ambiguous/buy"]-stats["unanimous/buy"]
    stats["ambiguity_signal_gap"]=stats["most_ambiguous/sensed"]-stats["unanimous/sensed"]
    # Brier of the graph's preceding motor tendency as a predictor of actual taste.
    for pos in range(visits):
        at=t["order"]==pos
        predicted=(h[:,:,2][at]+1)/2;actual=(h[:,:,1][at]>0).float()
        stats[f"taste_forecast_brier_{pos}"]=((predicted-actual).square().reshape(worlds,m)*w).sum(1).numpy()
    raw=dict(trace=trace.numpy(),count=base.numpy(),actual_history=h.numpy(),choice=out["choice"].numpy().astype(np.int8),
             terminal=out["final"].numpy().astype(np.int8),target=t["final_value"].numpy().astype(np.int8),
             always_buy=a["final"].numpy().astype(np.int8),always_skip=b["final"].numpy().astype(np.int8),
             donor=swap["choice"].numpy().astype(np.int8),weights=t["weight"].numpy())
    return raw,stats


def run(cohort,layout,stage,worlds,arms,*,offset=0,stress=False,label=None):
    torch.set_num_threads(1);ids=game.ticket.births(cohort);seed=(1059173 if cohort=="original" else 1059181)+offset
    label=label or (("transfer-" if stress else "")+cohort)
    dest=game.ticket.DATA/"agreement-eval"/layout/label/str(stage);dest.mkdir(parents=True,exist_ok=True)
    sha=lambda f:sha256(Path(f).read_bytes()).hexdigest();code=sha(__file__);series={};hashes={}
    bodypath=game.OUT/layout/cohort;params={}
    for arm in arms:
        hp=bodypath/arm/f"coupling-{stage}.npz";hh=sha(hp);hashes[hp.relative_to(game.ticket.DATA).as_posix()]=hh
        with np.load(hp) as z:k=torch.from_numpy(z["gain"]);eta=torch.from_numpy(z["eta"])
        params[arm]=dict(gain=k.flatten().tolist(),eta=eta.flatten().tolist())
        rates={}
        for i,b in enumerate(ids):
            g,r=game.ticket.old.load_checkpoint(bodypath,"odor",arm,b,stage)
            cp=game.ticket.old.checkpoint_path(bodypath,"odor",arm,b,stage);identity=sha(cp)
            hashes[cp.relative_to(game.ticket.DATA).as_posix()]=identity
            for count in ((3,4) if stress else (2,)):
                f=dest/f"{arm}-{b}-{count}.npz"
                prices=(.25,) if stress else (.05,.15,.25,.40,.55)
                if not f.exists():
                    raw={};values={}
                    for price in prices:
                        t=scene(seed,worlds,price,count);m=2**(count+2);parts=[];vbank=[]
                        for j in range(0,worlds,12):
                            sub={a:v[m*j:m*(j+12)] for a,v in t.items()}
                            d,z=measure(g,r,k[i],eta[i],sub,arm,layout);parts.append(d);vbank.append(z)
                        raw.update({f"{price}/{q}":np.concatenate([v[q] for v in parts]) for q in parts[0]})
                        values.update({f"{price}/{q}":np.concatenate([v[q] for v in vbank]) for q in vbank[0]})
                    primary=prices if stress else (.05,.25,.55)
                    for q in vbank[0]:values["pooled/"+q]=np.mean([values[f"{p}/{q}"] for p in primary],0)
                    game.ticket.storage._write_raw(f,**raw,**{"metric/"+q:v for q,v in values.items()},source=code,checkpoint=identity,
                                                   coupling=hh,seed=seed,worlds=worlds)
                with np.load(f) as z:
                    if (str(z["source"]),str(z["checkpoint"]),str(z["coupling"]),int(z["seed"]),int(z["worlds"]))!=(code,identity,hh,seed,worlds):
                        raise ValueError("Wrong agreement measurements")
                    for q in z:
                        if q.startswith("metric/"):rates.setdefault(str(count)+"/"+q[7:],[]).append(z[q])
                hashes[f.relative_to(game.ticket.DATA).as_posix()]=sha(f)
            print(layout,cohort,stage,arm,b,flush=True)
        series[arm]={q:np.stack(v) for q,v in rates.items()}
    result=dict(layout=layout,cohort=cohort,stage=stage,births=ids,worlds=worlds,seed=seed,hashes=hashes,parameters=params,
                scores=statistics.summarize(series,worlds,ids,seed+59,[("agreement","count"),("agreement","scrambled"),("scrambled","count")]),
                definition="Natural independent slip probability .2. Every fourfold law/error combination is "
                           "physically replayed with its generating probability. Sibling allocation preserves "
                           "resource price, first actual cue/evidence, true X and B and the weighted purchase frequency. "
                           "Unanimous vs maximally mixed histories have identical counts. Swapping the fast sensory "
                           "trace flips only the last support's evidence sign in the donor, then transfers that "
                           "decision back into the original body for realized resource scoring.")
    game.ticket.storage._write_json(game.ticket.EVIDENCE/f"agreement-{layout}-{label}-{stage}.json",result)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--cohort",choices=("original","independent"),required=True)
    p.add_argument("--layout",choices=("bilateral","local"),required=True);p.add_argument("--stage",type=int,required=True)
    p.add_argument("--worlds",type=int,default=160);p.add_argument("--arms",nargs="+",choices=game.ARMS,default=game.ARMS)
    p.add_argument("--offset",type=int,default=0);p.add_argument("--stress",action="store_true");p.add_argument("--label")
    run(**vars(p.parse_args()))
