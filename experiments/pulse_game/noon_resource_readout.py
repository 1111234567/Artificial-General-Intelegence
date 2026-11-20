"""Learn an option-bound resource decision directly from the consequences of a single taken action."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json
from time import perf_counter

import numpy as np
import torch

from . import noon_coverage as cov,noon_coverage_eval as testworld,noon_ticket as ticket


SEEDS=(1177001,1177013,1177033)
ARMS=("price","dose","directional")
OUT=ticket.DATA/"resource-readout";PLAN=ticket.EVIDENCE/"resource-readout-plan.json"


def inherited(ids):
    root=ticket.DATA/"tickets/original"
    rows=[ticket.old.load_checkpoint(root,"odor","native",b,8000) for b in ids]
    g=ticket.net.Graph(*(torch.cat([getattr(body,k) for body,_ in rows]) for k in ticket.capacity.GRAPH_FIELDS))
    r=ticket.net.Rule(ids)
    r.load_state_dict({k:torch.cat([rule.state_dict()[k] for _,rule in rows]) for k in r.state_dict()})
    return g.detach(),r.requires_grad_(False)


def features(t,trace,arm):
    # Only publicly displayed tariff and the circuit's own physically updated
    # six-entry trace supply the readout. No previous object identifier.
    price=(t["offer_left"][:,2]+t["offer_right"][:,2])*.25
    dose=trace[:,:3].sum(1)/3
    directional=cov.projection(trace,t["preview"])
    zero=torch.zeros_like(price)
    return torch.stack((torch.ones_like(price),(price-.25)*4,
                        dose if arm in ("dose","directional") else zero,
                        directional if arm=="directional" else zero),1)


def train_context(seed,step):
    block,j=divmod(step-1,4);rng=np.random.default_rng([149737,seed,block])
    k=(0,0,2,6)[rng.permutation(4)[j]]
    return k,("before" if rng.integers(2) else "after")


def fit(seed,arm):
    torch.set_num_threads(1);ids=ticket.births("original");g,r=inherited(ids)
    out=OUT/str(seed)/arm;out.mkdir(parents=True,exist_ok=True)
    sha=lambda f:sha256(Path(f).read_bytes()).hexdigest()
    protocol=dict(arm=arm,seed=seed,stages=[1000,2000],lr=.03,worlds=4,births=ids,plan=sha(PLAN),
                  parents={str(b):sha(ticket.old.checkpoint_path(ticket.DATA/"tickets/original","odor","native",b,8000)) for b in ids},
                  source={Path(f).name:sha(f) for f in (__file__,cov.__file__,ticket.__file__,testworld.__file__)})
    mf=out/"run.json"
    if mf.exists() and json.loads(mf.read_text())!=protocol:raise ValueError("Resource policy protocol changed")
    if not mf.exists():ticket.storage._write_json(mf,protocol)
    key=sha(mf);parameter=torch.nn.Parameter(torch.zeros(4,dtype=torch.float64))
    opt=torch.optim.Adam((parameter,),lr=.03);step=0;rows=[];file=out/"resume.pt"
    if file.exists():
        z=torch.load(file,weights_only=True,map_location="cpu")
        if z["meta"]!=key:raise ValueError("Invalid resource learner")
        with torch.no_grad():parameter.copy_(z["weight"])
        opt.load_state_dict(z["optimizer"]);step=z["step"];rows=z["trace"].tolist()
    first=step;t0=perf_counter()
    for step in range(first+1,2001):
        extra,pos=train_context(seed,step)
        ts=[]
        for b in ids:
            scene_seed=int(np.random.SeedSequence([53199,seed,b,step]).generate_state(1)[0])
            ts.append(testworld.scene(scene_seed,4,None,None,extra,pos,None))
        t=ticket.join(ts)
        with torch.no_grad():view,s,_,tr,_=cov.physical_prefix(g,r,t,"evidence");f=features(t,tr,arm).double()
        raw=f@parameter;p=raw.sigmoid()
        u=np.random.default_rng([48191,seed,step]).random(t["price"].numel())
        buy=torch.from_numpy(u)<p.detach()
        with torch.no_grad():
            public_side=(t["offer_right"][:,0]>t["offer_left"][:,0]).float()*2-1
            z=ticket.finish(view,s,t,"native",decision=buy)
            if not torch.equal(z["choice"],torch.where(buy,public_side,-public_side)):raise ValueError("Fee assigned to the wrong visible port")
            amount=z["utility"].double().reshape(len(ids),4,8);means=amount.mean(2,keepdim=True)
            adv=amount-(means.sum(1,keepdim=True)-means)/3
        logp=torch.nn.functional.logsigmoid(torch.where(buy,raw,-raw))
        entropy=(p*(2*p).clamp_min(1e-12).log()+(1-p)*(2*(1-p)).clamp_min(1e-12).log()).mean()
        loss=-(logp*adv.flatten()).mean()+.001*entropy+.00002*parameter.square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        rows.append([float(amount.mean()),float(z["gross"].mean()),float(buy.float().mean()),*parameter.detach().tolist()])
        if step%100==0:
            state=dict(step=step,meta=key,weight=parameter.detach(),optimizer=opt.state_dict(),trace=torch.tensor(rows))
            if step in (1000,2000):ticket.storage._atomic(out/f"step-{step}.pt",lambda f:torch.save(state,f))
            ticket.storage._atomic(file,lambda f:torch.save(state,f))
            print(seed,arm,step,parameter.detach().numpy().round(2),round((perf_counter()-t0)/(step-first),3),flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--arm",choices=ARMS,required=True);p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    a=p.parse_args();fit(a.seed,a.arm)
