"""Frozen net-resource verification of the small, explicit ticket action readout."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from . import noon_resource_readout as policy,noon_ticket_speed as fast,noon_market_stats as bootstrap
from .interleave_capacity_report import Crossed


def conditions(stress):
    if not stress:return {"base":(0,"none",None), "before2":(2,"before",None),"after2":(2,"after",None),
                          "before6":(6,"before",None),"after6":(6,"after",None)}
    return {"before14":(14,"before",None),"after14":(14,"after",None),"close15":(0,"none",15),"far75":(0,"none",75)}


@torch.no_grad()
def bank(g,r,seed,n,condition,prices,check=True):
    extra,pos,angle=condition;all_choices=[];all_features=[];all_draws=[]
    for alt in ("unseen","omitted"):
        tapes=[policy.testworld.scene(seed,n,p,alt,extra,pos,angle) for p in prices]
        blocks=[];features=[];randoms=[]
        for j in range(0,n,32):
            batch=[{k:v[8*j:8*(j+32)] for k,v in t.items()} for t in tapes]
            view,state,_,trace,_=policy.cov.physical_prefix(g,r,batch[0],"evidence")
            outcomes=[];sensed=[];uniform=[]
            for t in batch:
                gate=policy.ticket.net.act(view,state,t["offer_left"],t["offer_right"],t["draws"][:,2,0])
                native=gate.side==t["offer_side"];all_yes=torch.ones_like(native)
                yes,_=fast.downstream(view,gate,t,t["offer_side"],all_yes)
                no,_=fast.downstream(view,gate,t,-t["offer_side"],~all_yes)
                if check and j==0:
                    actual=policy.ticket.finish(view,state,t,"native")
                    torch.testing.assert_close(torch.where(native[:,None],yes,no),fast.recorded(actual),atol=0,rtol=0)
                outcomes.append(torch.stack((no[:,6],yes[:,6],no[:,7],yes[:,7],native.float()),1))
                sensed.append(policy.features(t,trace,"directional"))
                uniform.append(t["draws"][:,2,0])
            blocks.append(torch.stack(outcomes,1).numpy())
            features.append(torch.stack(sensed,1).numpy())
            randoms.append(torch.stack(uniform,1).numpy())
        all_choices.append(np.concatenate(blocks).reshape(n,2,4,len(prices),5))
        all_features.append(np.concatenate(features).reshape(n,2,4,len(prices),4))
        all_draws.append(np.concatenate(randoms).reshape(n,2,4,len(prices)))
    return dict(outcomes=np.stack(all_choices,1),features=np.stack(all_features,1),draw=np.stack(all_draws,1),prices=np.asarray(prices))


def collect(cohort,stress):
    torch.set_num_threads(1);seed=(1277011 if cohort=="original" else 1277017)+(193701 if stress else 0)
    n=96 if stress else 192;base=policy.OUT/"evaluation"/cohort/("stress" if stress else "main");base.mkdir(parents=True,exist_ok=True)
    ids=policy.ticket.births(cohort);hashes={};sha=lambda f:sha256(Path(f).read_bytes()).hexdigest();code=sha(__file__)
    for b in ids:
        root=policy.ticket.DATA/"tickets"/cohort;cp=policy.ticket.old.checkpoint_path(root,"odor","native",b,8000)
        g,r=policy.ticket.old.load_checkpoint(root,"odor","native",b,8000);checksum=sha(cp)
        for condition,spec in conditions(stress).items():
            f=base/f"{b}-{condition}.npz";prices=policy.ticket.PRICES if condition=="base" else (.25,)
            if not f.is_file():
                raw=bank(g,r,seed,n,spec,prices)
                cal=bank(g,r,seed+971,64,spec,prices,False)
                policy.ticket.storage._write_raw(f,**raw,cal=cal["outcomes"],checkpoint=checksum,code=code,seed=seed,worlds=n)
            with np.load(f) as z:
                if (str(z["code"]),str(z["checkpoint"]),int(z["seed"]),int(z["worlds"]))!=(code,checksum,seed,n):raise ValueError("Resource bank changed")
            hashes[f.name]=sha(f)
        print(cohort,stress,b,flush=True)
    policy.ticket.storage._write_json(base/"index.json",dict(seed=seed,worlds=n,births=ids,hashes=hashes,source=code))


def means(value,subset="all"):
    # [world,alternative,known/alternative,private,tariff].
    v=value.mean(3)
    if subset=="all":return v.mean((1,2))
    if subset=="known":return v[:,:,0].mean(1)
    return v[:,0 if subset=="unseen" else 1,1]


def measures(z,theta,arm):
    v=z["outcomes"];cal=z["cal"];f=z["features"].copy()
    if arm=="price":f[...,2:]=0
    elif arm=="dose":f[...,3]=0
    p=torch.from_numpy(f).double().matmul(torch.as_tensor(theta,dtype=torch.double)).sigmoid().numpy()
    take=z["draw"]<p
    do=v[...,3];skip=v[...,2];delta=do-skip
    actual=np.where(take,do,skip)
    # Swap the selected decision between the seen and unseen/unfed matched twin.
    matched=.5*(actual+np.where(np.flip(take,axis=2),do,skip))
    fixed=means(cal[...,3]-cal[...,2]).mean(0)>0
    reference=np.where(fixed.reshape(1,1,1,1,-1),do,skip)
    native=np.where(v[...,4]>0,do,skip)
    raw=dict(net=actual,gross=np.where(take,v[...,1],v[...,0]),buy=take.astype(float),
             fee=take*z["prices"].reshape(1,1,1,1,-1),over_price_only=actual-reference,
             gain_over_matched=actual-matched,over_native=actual-native,price_only=reference,native=native,
             always_buy=do,always_skip=skip,expected_net=skip+p*delta)
    vals={}
    for k,value in raw.items():
        for group in ("all","known","unseen","unfed"):
            subtotal=means(value,group)
            vals[group+"/"+k]=subtotal.mean(1)
            for j,price in enumerate(z["prices"]):vals[f"{price}/{group}/{k}"]=subtotal[:,j]
    vals["unseen_m_known"]=vals["unseen/buy"]-vals["known/buy"]
    vals["unfed_m_known"]=vals["unfed/buy"]-vals["known/buy"]
    return vals


def score(cohort,stress=False):
    torch.set_num_threads(1)
    path=policy.OUT/"evaluation"/cohort/("stress" if stress else "main")
    import json
    info=json.loads((path/"index.json").read_text());parameters={};locked={}
    for arm in policy.ARMS:
        parameters[arm]=[]
        for seed in policy.SEEDS:
            f=policy.OUT/str(seed)/arm/"step-2000.pt";saved=torch.load(f,weights_only=True,map_location="cpu")
            if saved["step"]!=2000:raise ValueError("Premature resource fit")
            parameters[arm].append(saved["weight"].tolist());locked[f.relative_to(policy.ticket.DATA).as_posix()]=sha256(f.read_bytes()).hexdigest()
    results={}
    for cond in conditions(stress):
        tables={}
        for arm,weights in parameters.items():
            all_={}
            for b in info["births"]:
                f=path/f"{b}-{cond}.npz"
                if sha256(f.read_bytes()).hexdigest()!=info["hashes"][f.name]:raise ValueError("Unexpected resource scenes")
                with np.load(f) as z:
                    raw={k:z[k] for k in ("outcomes","cal","features","draw","prices")}
                items=[measures(raw,param,arm) for param in weights]
                for k in items[0]:all_.setdefault(k,[]).append(np.stack([q[k] for q in items]))
            tables[arm]={k:np.stack(v,axis=1) for k,v in all_.items()}
        for a,b in (("directional","price"),("directional","dose"),("dose","price")):
            tables[a+" minus "+b]={k:v-tables[b][k] for k,v in tables[a].items()}
        ci=Crossed(len(info["births"]),info["worlds"],info["seed"]+37,2500)
        results[cond]={arm:bootstrap.fit_seed_intervals(table,ci) for arm,table in tables.items()}
    label=("stress-" if stress else "")+cohort
    policy.ticket.storage._write_json(policy.ticket.EVIDENCE/f"resource-readout-{label}.json",dict(births=info["births"],
          worlds=info["worlds"],test_seed=info["seed"],seeds=list(policy.SEEDS),results=results,coefficients=parameters,
          learner_hashes=locked,raw_hashes=info["hashes"],notes="Buy logit is explicitly translated to the side with "
          "the observed ticket mark. All scalar readouts use the same native bodies and subsequent actual contact/terminal "
          "behavior. Price-only comparator selected from forced outcomes on separate geometries. Cross-validated "
          "benefit above that comparator and equal-price/equal-count redistribution have different denominators."))


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--cohort",choices=("original","independent"),required=True)
    p.add_argument("--mode",choices=("collect","score"),required=True);a=p.parse_args()
    for stress in (False,True):
        if a.mode=="collect":collect(a.cohort,stress)
        else:score(a.cohort,stress)
