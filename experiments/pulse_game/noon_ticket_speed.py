"""Reuse causal prefixes and the two actually possible route states in frozen market evaluation."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from . import noon_ticket as study


def recorded(out):
    fields=("choice","buy","logit","review_side","taste","final","gross","utility")
    return torch.stack([out[k].to(torch.float32) for k in fields],1)


@torch.no_grad()
def ordinary(g,r,tapes):
    values=[]
    for t in tapes:
        view,state,_,_=study.prefix(g,r,t);forced=torch.ones_like(t["coin"])
        rows=[study.finish(view,state,t,"native",decision=d) for d in (None,forced,~forced)]
        rows.append(study.finish(view,state,t,"native",deliver="zero"))
        values.append(torch.stack([recorded(v) for v in rows],1))
    return torch.stack(values,1)


def downstream(view,gate_state,t,side,access):
    fee=t["price"]*access
    a=study.net.Action(gate_state.state,side,gate_state.motor_logit,F.logsigmoid(side*gate_state.motor_logit))
    visited=study.net.observe(view,a,t["offer_left"],t["offer_right"],-fee)
    after=study.net.commit(view,visited,t["draws"][:,2,1:]).state
    def sample(info):
        state,_,review_side,taste=study.encounter(view,after,t["review"],t["review_value"]*info,t["draws"][:,3])
        state=study.net.wait(view,state,t["gaps"][:,3],max_ticks=6)
        final=study.motor.act(state,-t["final"],t["final"],t["draws"][:,4,0])
        gross=(final.side*t["final_value"]>0).float()
        return recorded(dict(choice=side,buy=access,logit=a.motor_logit,review_side=review_side,taste=taste,
                             final=final.side,gross=gross,utility=gross-fee))
    regular=sample(access)
    denied=sample(torch.zeros_like(access)) if access.any() else regular
    return regular,denied


@torch.no_grad()
def cached(g,r,tapes,augment=None):
    view,state,_,history=study.prefix(g,r,tapes[0]);values=[]
    if augment is not None:tapes=[augment(t,history) for t in tapes]
    for t in tapes:
        gate=study.net.act(view,state,t["offer_left"],t["offer_right"],t["draws"][:,2,0])
        buy=gate.side==t["offer_side"]
        yes,denied=downstream(view,gate,t,t["offer_side"],torch.ones_like(buy))
        no,_=downstream(view,gate,t,-t["offer_side"],torch.zeros_like(buy))
        natural=torch.where(buy[:,None],yes,no)
        blind=torch.where(buy[:,None],denied,no)
        values.append(torch.stack((natural,yes,no,blind),1))
    return torch.stack(values,1)


def run(worlds,repeats):
    torch.set_num_threads(1)
    cases=[];raw={}
    for cohort,birth in (("original",731),("original",735),("independent",811),("independent",815)):
        g,r=study.old.load_checkpoint(study.DATA/"tickets"/cohort,"odor","native",birth,8000)
        for alt in ("omitted","unseen"):
            bank=[study.tape(553891,birth,1,worlds,domain=813711,price=p,alternative=alt) for p in study.PRICES]
            for key in ("study","study_value","preview","draws","gaps","final","final_value","review_value"):
                if not all(torch.equal(bank[0][key],v[key]) for v in bank[1:]):raise ValueError("Different shared physics")
            old=ordinary(g,r,bank);new=cached(g,r,bank)
            torch.testing.assert_close(old,new,atol=0,rtol=0)
            raw[f"{birth}/{alt}"]=new.numpy()
            cases.append((g,r,bank))
    rng=np.random.default_rng(761231);order=rng.permutation(np.arange(repeats)%2)
    elapsed=[];checked=0
    for i,first in enumerate(order):
        data=cases[i%len(cases)]
        times=[0.,0.];vals=[None,None]
        for j in (int(first),1-int(first)):
            start=perf_counter();vals[j]=(ordinary,cached)[j](*data);times[j]=perf_counter()-start
        if not torch.equal(vals[0],vals[1]):raise ValueError("Frozen replay changed a recorded scalar")
        checked+=vals[0].numel();elapsed.append(times)
    dt=np.array(elapsed);idx=rng.integers(repeats,size=(5000,repeats));speed=dt[idx,0].mean(1)/dt[idx,1].mean(1)
    dest=study.DATA/"speed";dest.mkdir(exist_ok=True,parents=True)
    trace=dest/"frozen-ticket.npz";study.storage._write_raw(trace,**raw,seconds=dt,order=order)
    study.storage._write_json(study.EVIDENCE/"ticket-speed.json",dict(repeats=repeats,worlds=worlds,threads=1,births=[731,735,811,815],
           tariffs=study.PRICES,alternatives=["native","buy","skip","native-withheld"],mean_seconds=dt.mean(0).tolist(),
           ratio=float(dt[:,0].mean()/dt[:,1].mean()),paired_ci95=np.quantile(speed,[.025,.975]).tolist(),
           checked_scalar_count=checked,raw_sha256=sha256(trace.read_bytes()).hexdigest(),
           implementation_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),torch=str(torch.__version__),
           note="Both methods execute the same sample from the full four-branch, three-price endpoint battery. "
                "All eight per-life outcome fields including motor logit match exactly on every paired repeat. "
                "Measurements include all neural operations after tape generation and exclude optimizer/backward."))
    print("ratio",dt[:,0].mean()/dt[:,1].mean(),np.quantile(speed,[.025,.975]),flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--worlds",type=int,default=48);p.add_argument("--repeats",type=int,default=24)
    a=p.parse_args();run(a.worlds,a.repeats)
