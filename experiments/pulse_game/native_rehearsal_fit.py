"""Teach the recurrent graph on physical actions. No cue, reward, phase or query enters its silent window."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from . import native_rehearsal_graph as live
from . import interleave_revisit_train as storage


ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"data/native-rehearsal-20261120"
EVIDENCE=ROOT/"docs/evidence/native-rehearsal-20261120"
KINDS=("single","copy","permuted","guide_a","guide_c")
BIRTHS=(731,733,735)
BATCH=64
STEPS=1536


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def internal_cue(currents):
    # Direct neural current already entering the two opposed ordinary sensory port banks.
    return (currents[:,:,3:6]-currents[:,:,:3])/2


def update(model,opt,kind,birth,step,batch):
    e=live.scenes([4109327,birth,step],batch)
    _,s,support_scores,visits=live.support(model,e)
    s,_,currents=live.silence(model,s)
    _,action_scores,pays,_=live.queries(model,s,e)
    correct=(pays>0).to(s.h.dtype)
    both=correct.prod(1)
    reward=correct.sum(1)+both
    # Each action score receives only return causally downstream. The second
    # question cannot earn the earlier question's reward.
    first=e.first_query
    second=1-first
    rows=torch.arange(len(first))
    later=correct[rows,second]+both
    def centered(x):
        return (x-x.mean()).detach()*len(x)/(len(x)-1)
    score=((support_scores.sum(1)+action_scores[rows,first])*centered(reward)+
           action_scores[rows,second]*centered(later)).mean()
    loss=-score
    teaching=loss.new_tensor(0.)
    if kind in ("guide_a","guide_c"):
        target=live.recorded_favorable(visits,0 if kind=="guide_a" else 1).detach()/1.35
        teaching=F.mse_loss(internal_cue(currents)[-4:].mean(0),target)
        loss=loss+1.0*teaching
    opt.zero_grad(set_to_none=True)
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
    if not norm.isfinite():raise FloatingPointError(f"Nonfinite native update at {kind}/{birth}/{step}")
    opt.step()
    return [*correct.mean(0).tolist(),both.mean().item(),reward.mean().item(),float(teaching.detach()),float(norm)]


def fit(kind,birth,smoke=False):
    torch.set_num_threads(1);torch.manual_seed(birth)
    layout=kind if kind in ("single","permuted") else "copy"
    model=live.NativeGraph(birth,layout)
    opt=torch.optim.Adam(model.parameters(),lr=.003)
    folder=DATA/("smoke" if smoke else kind);folder.mkdir(parents=True,exist_ok=True)
    destination=folder/f"seed-{birth}.pt"
    signature=dict(source=digest(__file__),graph=digest(live.__file__),birth=birth,layout=layout,kind=kind,
                   plan=digest(EVIDENCE/"protocol.md") if not smoke else "",steps=STEPS,batch=BATCH)
    start=0;ledger=[]
    if destination.exists() and not smoke:
        saved=torch.load(destination,map_location="cpu",weights_only=True)
        if signature!=saved["signature"]:raise ValueError("Changed native training configuration")
        model.load_state_dict(saved["state"]);opt.load_state_dict(saved["optimizer"])
        start=saved["step"];ledger=saved["ledger"].tolist()
    begin=perf_counter()
    for step in range(start+1,(6 if smoke else STEPS)+1):
        ledger.append(update(model,opt,kind,birth,step,BATCH))
        if not smoke and (step%128==0 or step==STEPS):
            pack=dict(signature=signature,layout=layout,kind=kind,birth=birth,step=step,
                      state=model.state_dict(),optimizer=opt.state_dict(),ledger=torch.tensor(ledger))
            storage._atomic(destination,lambda temporary:torch.save(pack,temporary))
            print(kind,birth,step,np.mean(ledger[-32:],0)[:4].round(3),flush=True)
    print("elapsed",kind,birth,perf_counter()-begin,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind",choices=KINDS);p.add_argument("birth",type=int,choices=BIRTHS)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args();fit(args.kind,args.birth,args.smoke)
