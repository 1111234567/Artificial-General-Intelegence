"""Which receiver rows of a physical B write change A and B retrieval?"""

from argparse import ArgumentParser
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_capacity as cap, interleave_stream as stream, local_trace_graph as net
from .interleave_interference import DATA,EVIDENCE,ROOT


def expand_rows(x,n):
    return x.repeat((n,)+(1,)*(x.ndim-1))


@torch.no_grad()
def attribution(g,rule,tape):
    lanes=tape.lanes
    coeff=rule.expand(lanes)
    state=net.start(g,lanes)
    before,_=cap._study(coeff,state,tape,0,write=True)
    l,r=tape.left[:,1,0],tape.right[:,1,0]
    act=net.act(coeff,before,l,r,tape.motor_draw[:,1,0])
    exp=net.observe(coeff,act,l,r,act.side*tape.right_value[:,1,0])
    edit=net.commit(coeff,exp,tape.edit_draw[:,1,0])
    ready=net.wait(coeff,exp.state,net.PRE_WRITE_TICKS)
    P=edit.state.graph
    prev=before.graph
    shared=P.present&prev.present
    deltas=edit.receipt.weight_delta
    # A native baseline, then one postoperative graph per receiver. Only that row's
    # surviving numeric weights return to their physical pre-B values.
    size=P.width+1
    ids=torch.arange(size).repeat_interleave(lanes)
    all_graph=net.Graph(*(expand_rows(getattr(P,k),size) for k in ("weights","bias","present")))
    revert=(ids[:,None,None]==torch.arange(P.width)[None,:,None]+1)&expand_rows(shared,size)
    W=torch.where(revert,expand_rows(prev.weights,size),all_graph.weights)
    new=replace(all_graph,weights=W)
    after=net.State(new,expand_rows(ready.hidden,size),expand_rows(ready.sender_trace,size),
                    expand_rows(ready.receiver_trace,size))
    all_coeff=rule.expand(size*lanes)
    waited=net.wait(all_coeff,after,net.POST_WRITE_TICKS+4)
    sides,prob=[],[]
    for i in (0,1):
        left,right,u=(expand_rows(x,size) for x in
                      (tape.left[:,i,-1],tape.right[:,i,-1],tape.motor_draw[:,i,-1]))
        choice=net.act(all_coeff,waited,left,right,u)
        truth=expand_rows(tape.right_value[:,i,-1],size)
        sides.append((choice.side==truth).reshape(size,lanes))
        prob.append((choice.motor_logit*truth).sigmoid().reshape(size,lanes))
        baseline=cap._probe(coeff,net.wait(coeff,edit.state,4),tape,i)
        torch.testing.assert_close(choice.motor_logit[:lanes],baseline.motor_logit,rtol=1e-5,atol=1e-6)
        assert torch.equal(choice.side[:lanes],baseline.side)
    # Features used only by post-hoc calibration, not by the autonomous policy.
    features=torch.stack((deltas.abs().sum(2),deltas.sum(2),ready.hidden,ready.receiver_trace.mean(2),
                          exp.feedback[...,2],prev.weights.abs().sum(2),P.bias),2)
    return dict(correct=torch.stack(sides,2).numpy(),prob=torch.stack(prob,2).numpy(),
                features=features.numpy(),delta_rows_l1=deltas.abs().sum(2).numpy())


def effects(batch):
    p=batch["prob"].astype(float)
    both=p.prod(2)
    real=batch["correct"].all(2)
    return dict(pa=p[1:,:,0]-p[0,:,0],pb=p[1:,:,1]-p[0,:,1],
                p_both=both[1:]-both[0],both=real[1:].astype(float)-real[0],
                a=batch["correct"][1:,:,0].astype(float)-batch["correct"][0,:,0],
                b=batch["correct"][1:,:,1].astype(float)-batch["correct"][0,:,1])


def summary(cal,measured,seed):
    train=effects(cal);test=effects(measured)
    samples=test["both"].shape[1]
    ix=np.arange(samples)
    # Calibrate static and feature-conditional vetoes solely on separate geometries.
    static=int(np.argmax(train["p_both"].mean(1)))
    f=cal["features"].astype(float)
    h=measured["features"].astype(float)
    center=f.reshape(-1,f.shape[2]).mean(0)
    scale=f.reshape(-1,f.shape[2]).std(0).clip(1e-6)
    x=((f-center)/scale).reshape(-1,f.shape[2])
    z=((h-center)/scale)
    y=train["p_both"].T.reshape(-1)
    coefficient=np.linalg.solve(x.T@x+np.eye(x.shape[1])*len(x)*.001,x.T@(y-y.mean()))
    predicted=y.mean()+z@coefficient
    picked=predicted.argmax(1)
    proposed=predicted[ix,picked]>0
    if samples%4:raise AssertionError("Expected balanced groups of four world lanes")
    # Random choice shares each world's four labels and has no access to which label was scored.
    rng=np.random.default_rng(seed)
    random=np.repeat(rng.integers(net.NODES,size=samples//4),4)
    oracle=test["p_both"].argmax(0)
    all_modes={"static_calibrated":(np.full(samples,static),np.ones(samples,dtype=bool)),
               "feature_calibrated":(picked,proposed),
               "random_row":(random,np.ones(samples,dtype=bool)),
               "hindsight_row":(oracle,test["p_both"][oracle,ix]>0)}
    actual={}
    for name,(row,allow) in all_modes.items():
        actual[name]={k:float((v[row,ix]*allow).mean()) for k,v in test.items()}
        actual[name]["veto_rate"]=float(allow.mean())
    return dict(static_row=static,feature_coefficients=coefficient.tolist(),
                physical_veto_effects=actual,
                per_row_mean={k:np.mean(v,axis=1).tolist() for k,v in test.items()},
                per_row_abs={k:np.abs(v).mean(1).tolist() for k,v in test.items() if k in ("pa","pb","p_both")},
                per_row_dose_l1=measured["delta_rows_l1"].mean(0).tolist(),
                opposite_effects=float(((test["pa"]*test["pb"]<0)&(np.abs(test["pa"])>.002)
                                        &(np.abs(test["pb"])>.002)).mean()))


def run(source,label,births,worlds,seed):
    torch.set_num_threads(1)
    dest=DATA/"row-credit"/label
    dest.mkdir(parents=True,exist_ok=True)
    out=dict(seed=seed,births=births,worlds_per_split=worlds,roots={},sha256={},
             note="Physical receiver-row repeal applies after B, retains actual bias, support and birth weights, "
                  "then replays the two post-write ticks plus four neutral ticks. Branch probes are independent. "
                  "Reversion effects can interact. Calibrated selectors are external diagnostic interventions, "
                  "not additional controllers trained into the graph. Hindsight bound uses privileged future outcomes.")
    cal=cap.make_tape("odor",seed,worlds,1)
    holdout=cap.make_tape("odor",seed+1,worlds,1)
    for birth in births:
        g,r=source.load(birth)
        a=attribution(g,r,cal)
        b=attribution(g,r,holdout)
        out["roots"][birth]=summary(a,b,[birth,seed])
        p=dest/f"{birth}.npz"
        np.savez_compressed(p,**{f"cal/{k}":v for k,v in a.items()},**{f"test/{k}":v for k,v in b.items()})
        out["sha256"][p.name]=sha256(p.read_bytes()).hexdigest()
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/f"row-credit-{label}.json").write_text(json.dumps(out,indent=1)+"\n",encoding="utf8")
    print(label,"complete",flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--label",required=True)
    p.add_argument("--family",choices=("joint","lifetime","objective","odor","feeder"),required=True)
    p.add_argument("--run",type=Path,default=ROOT/"data/local-trace-20261119/local")
    p.add_argument("--arm",default="active")
    p.add_argument("--step",type=int,default=8000)
    p.add_argument("--worlds",type=int,default=96)
    p.add_argument("--seed",type=int,default=774599)
    p.add_argument("--births",nargs="+",type=int,default=list(range(731,737)))
    a=p.parse_args()
    run(stream.Source(a.family,a.run,a.arm,a.step),a.label,a.births,a.worlds,a.seed)
