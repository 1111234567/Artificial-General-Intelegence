"""Economic test of the route-bound sensory trace with matched actual movements and prices."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from . import noon_option_binding as bound, noon_coverage_eval as scenarios, noon_market_stats as stats


@torch.no_grad()
def evaluate(g,r,gain,t,role):
    view,s,_,trace,hist=bound.cov.physical_prefix(g,r,t,"evidence")
    q=t["preview"];v=bound.cov.projection(trace,q);n=len(v);twin=torch.arange(n)^4
    aug=bound.instrument(t,v,gain,role)
    chosen=bound.cov.ticket.finish(view,s,aug,"native")
    buys=chosen["buy"];force=torch.ones_like(buys)
    buy=bound.cov.ticket.finish(view,s,aug,"native",decision=force);skip=bound.cov.ticket.finish(view,s,aug,"native",decision=~force)
    def earned(side):return torch.where(side,buy["utility"],skip["utility"])
    torch.testing.assert_close(earned(buys),chosen["utility"],rtol=0,atol=0)
    report=dict(buy=buys.float(),gross=chosen["gross"],net=chosen["utility"],always_buy=buy["utility"],never_buy=skip["utility"],
                expected_information=buy["gross"]-skip["gross"],selection_gain=.5*(chosen["utility"]-earned(buys[twin])))
    exchanged=bound.instrument(t,v[twin],gain,role)
    flip=bound.instrument(t,v,gain,"opposite" if role=="local" else "local" if role=="opposite" else role)
    for name,u in (("donor",exchanged),("disabled",t),("side_flip",flip)):
        z=bound.cov.ticket.finish(view,s,u,"native")
        report[name+"_choice_loss"]=chosen["utility"]-earned(z["buy"])
        report[name+"_net_loss"]=chosen["utility"]-z["utility"]
    denied=bound.cov.ticket.finish(view,s,aug,"native",deliver="zero")
    if not torch.equal(denied["buy"],buys):raise ValueError("Failed no-peeking check")
    report["taste_gain"]=chosen["utility"]-denied["utility"]
    raw=dict(executed=chosen["choice"],purchased=buys,terminal=chosen["final"],target=t["final_value"],
             force_buy=buy["final"],force_skip=skip["final"],study=hist,trace=trace,feature=v)
    return {k:x.cpu().numpy() for k,x in raw.items()},{k:x.reshape(-1,2,4).mean(2).numpy() for k,x in report.items()}


def bank(g,r,gain,seed,worlds,role,extra=0,where="none",angle=None,prices=(.05,.25,.55)):
    raw={};vals={}
    for p in prices:
        both=[]
        for kind in ("unseen","omitted"):
            t=scenarios.scene(seed,worlds,p,kind,extra,where,angle);data=[];scores=[]
            for j in range(0,worlds,32):
                d,z=evaluate(g,r,gain,{k:v[j*8:(j+32)*8] for k,v in t.items()},role)
                data.append(d);scores.append(z)
            for k in data[0]:raw[f"{p}/{kind}/{k}"]=np.concatenate([d[k] for d in data])
            both.append({k:np.concatenate([v[k] for v in scores]) for k in scores[0]})
        for k in both[0]:
            a,b=both[0][k],both[1][k]
            states=np.stack(((a[:,0]+b[:,0])/2,a[:,1],b[:,1]),1)
            vals[f"{p}/all/{k}"]=states@np.array([.5,.25,.25])
            for i,n in enumerate(("known","novel","omitted")):vals[f"{p}/{n}/{k}"]=states[:,i]
            if k=="buy":
                vals[f"{p}/novel_minus_known"]=states[:,1]-states[:,0]
                vals[f"{p}/omitted_minus_known"]=states[:,2]-states[:,0]
    return raw,vals


def run(cohort,stage,roles,worlds,*,stress=False,offset=0,label=None):
    torch.set_num_threads(1);ids=bound.cov.ticket.births(cohort);seed=(10930117 if cohort=="original" else 10930131)+offset
    label=label or (("stress-" if stress else "")+cohort)
    dest=bound.cov.ticket.DATA/"binding-eval"/label/str(stage);dest.mkdir(parents=True,exist_ok=True)
    digest=lambda p:sha256(Path(p).read_bytes()).hexdigest()
    code=digest(__file__);hashes={};groups={}
    for role in roles:
        path=bound.cov.OUT/cohort if role=="bilateral" else bound.OUT/cohort
        arm="evidence" if role=="bilateral" else role;hfile=path/arm/f"gain-{stage}.npz"
        with np.load(hfile) as z:gains=torch.from_numpy(z["gain"]).flatten()
        hashes[hfile.relative_to(bound.cov.ticket.DATA).as_posix()]=digest(hfile)
        rates={}
        for i,b in enumerate(ids):
            g,r=bound.cov.ticket.old.load_checkpoint(path,"odor",arm,b,stage)
            cp=bound.cov.ticket.old.checkpoint_path(path,"odor",arm,b,stage);identity=digest(cp)
            hashes[cp.relative_to(bound.cov.ticket.DATA).as_posix()]=identity
            suites={"basic":(0,"none",None,(.05,.25,.55))} if not stress else {
                name:(k,pos,angle,prices) for name,(k,pos,angle,prices,_) in scenarios.scenarios(True).items()}
            for name,(k,pos,angle,prices) in suites.items():
                f=dest/f"{role}-{b}-{name}.npz"
                if not f.exists():
                    raw,values=bank(g,r,gains[i],seed,worlds,role,k,pos,angle,prices)
                    # Set each tariff-only decision on fresh calibration histories, earlier than test.
                    if not stress:
                        _,cal=bank(g,r,gains[i],seed+13,64,role)
                        for price in prices:
                            a=cal[f"{price}/all/always_buy"].mean()>cal[f"{price}/all/never_buy"].mean()
                            fixed=values[f"{price}/all/"+("always_buy" if a else "never_buy")]
                            values[f"{price}/all/over_price_only"]=values[f"{price}/all/net"]-fixed
                    for key in list(values):
                        if key.startswith(f"{prices[0]}/"):
                            suffix=key.split("/",1)[1];values["pooled/"+suffix]=np.mean([values[f"{p}/{suffix}"] for p in prices],0)
                    bound.cov.ticket.storage._write_raw(f,**raw,**{"metric/"+k:v for k,v in values.items()},checkpoint=identity,
                                                       selector=digest(hfile),source=code,worlds=worlds,seed=seed)
                with np.load(f) as z:
                    if (str(z["checkpoint"]),str(z["selector"]),str(z["source"]),int(z["worlds"]),int(z["seed"]))!=(identity,digest(hfile),code,worlds,seed):raise ValueError("Binding cache mismatch")
                    for key in z:
                        if key.startswith("metric/"):rates.setdefault(name+"/"+key[7:],[]).append(z[key])
                hashes[f.relative_to(bound.cov.ticket.DATA).as_posix()]=digest(f)
            print(cohort,stage,role,b,flush=True)
        groups[role]={k:np.stack(v) for k,v in rates.items()}
    report=dict(cohort=cohort,stage=stage,births=ids,seed=seed,worlds=worlds,hashes=hashes,
                results=stats.summarize(groups,worlds,ids,seed+97,[("local","bilateral"),("local","opposite"),("local","detached")]),
                notes="All policies sample the ordinary native motor. Donor-choice and side-attachment interventions "
                      "execute the donor side back in the unchanged ordinary recipient to isolate action selection.")
    bound.cov.ticket.storage._write_json(bound.cov.ticket.EVIDENCE/f"binding-{label}-{stage}.json",report)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--cohort",choices=("original","independent"),required=True)
    p.add_argument("--stage",type=int,required=True);p.add_argument("--roles",nargs="+",required=True)
    p.add_argument("--worlds",type=int,default=192);p.add_argument("--stress",action="store_true")
    p.add_argument("--offset",type=int,default=0);p.add_argument("--label");run(**vars(p.parse_args()))
