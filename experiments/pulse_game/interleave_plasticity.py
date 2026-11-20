"""Distinguish forgetting old objects from loss of the ability to learn a new one."""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import local_trace_graph as net
from . import local_trace_train as old
from .interleave_interference import DATA, EVIDENCE, ROOT


BRANCHES = ("intact", "zero_h", "zero_traces", "fresh_h_traces", "restore_surviving_weights",
            "restore_bias", "birth_graph", "birth_everything", "no_new_write")
AGES = (0,1,2,4,8,16,32,64,128)
CONDITIONS = ("novel", "repeat", "sense_only", "idle")


def marks(rng, shape):
    raw = rng.normal(size=(*shape,3))
    return (1.35 * raw / np.linalg.norm(raw,axis=-1,keepdims=True)).astype(np.float32)


def support(rule, state, item, law, flip, u, mode):
    right = item * flip[:, None]
    act = net.act(rule, state, -right, right, u[:,0])
    taste = act.side * law * flip
    obs = net.observe(rule, act, -right, right, taste)
    return net.commit(rule, obs, u[:,1:], intervention=mode)


def choose(rule, state, item, law, flip, u):
    right = item * flip[:,None]
    action = net.act(rule,state,-right,right,u)
    truth = law * flip
    return action.side, action.side == truth, (action.motor_logit * truth).sigmoid()


def fork(state, birth_state, mode):
    graph = state.graph
    init = birth_state.graph
    if mode == "birth_everything":
        return birth_state
    if mode == "birth_graph":
        return replace(state, graph=init)
    if mode == "restore_bias":
        return replace(state,graph=replace(graph,bias=init.bias))
    if mode == "restore_surviving_weights":
        w = torch.where(graph.present & init.present, init.weights, graph.weights)
        return replace(state,graph=replace(graph,weights=w))
    h = torch.zeros_like(state.hidden) if mode in ("zero_h","fresh_h_traces") else state.hidden
    if mode in ("zero_traces","fresh_h_traces"):
        return replace(state,hidden=h,sender_trace=torch.zeros_like(state.sender_trace),
                       receiver_trace=torch.zeros_like(state.receiver_trace))
    return replace(state,hidden=h)


def health(state, born):
    g = state.graph
    w = g.weights
    def flat(x): return x.flatten(1)
    active = flat(g.present).sum(1)
    common = g.present & born.present
    return dict(edges=active, changed_support=flat(g.present != born.present).sum(1),
                weight_l1=flat((w-born.weights).abs()).sum(1),
                bias_l1=(g.bias-born.bias).abs().sum(1),
                frac_at_limit=(flat(common & (w.abs() >= net.WEIGHT_LIMIT-1e-5)).sum(1)
                               / flat(common).sum(1).clamp_min(1)),
                activity_abs=state.hidden.abs().mean(1),
                sender_trace_abs=flat(state.sender_trace.abs()).mean(1))


@torch.no_grad()
def evaluate(graph, rule, seed, root, pairs, condition, ages):
    # The unseen test odor, its sign, side layouts and draws are independent of the history.
    # Test both signs by duplicating each full history with exactly the same uniforms.
    H=max(ages)
    rng=np.random.default_rng([seed,root,71727])
    history=torch.from_numpy(marks(rng,(H,pairs)))
    private=torch.from_numpy(rng.choice((-1.,1.),size=(H,pairs)).astype(np.float32))
    flip=torch.from_numpy(rng.choice((-1.,1.),size=(H,pairs)).astype(np.float32))
    draws=torch.from_numpy(rng.random((H,pairs,5),dtype=np.float32))
    new=torch.from_numpy(marks(rng,(pairs,))).repeat_interleave(2,0)
    sign=torch.tensor([-1.,1.]).repeat(pairs)
    testflip=torch.from_numpy(rng.choice((-1,1),size=(pairs,2)).astype(np.float32)).repeat_interleave(2,0)
    testu=torch.from_numpy(rng.random((pairs,6),dtype=np.float32)).repeat_interleave(2,0)
    earlier_flip=torch.from_numpy(rng.choice((-1,1),size=pairs).astype(np.float32)).repeat_interleave(2)
    earlier_u=torch.from_numpy(rng.random(pairs,dtype=np.float32)).repeat_interleave(2)
    old_item=history[0].repeat_interleave(2,0)
    old_sign=private[0].repeat_interleave(2)
    state=net.start(graph,pairs)
    rules=rule.expand(pairs)
    rules2=rule.expand(2*pairs)
    birth2=net.start(graph,2*pairs)
    twins=torch.arange(pairs).repeat_interleave(2)
    samples={k:[] for k in ("new_side","new_correct","new_p","old_correct","old_p","fresh_delta_l1")}
    metrics={}
    for age in range(H+1):
        if age in ages:
            prefix=state.subset(twins)
            for key,value in health(state,net.start(graph,pairs).graph).items():
                metrics.setdefault(key,[]).append(value.numpy())
            branch_values={k:[] for k in samples}
            for branch in BRANCHES:
                entry=fork(prefix,birth2,branch)
                write=support(rules2,entry,new,sign,testflip[:,0],testu[:,:5],
                              "sham" if branch=="no_new_write" else "active")
                after=net.wait(rules2,write.state,4)
                side,good,p=choose(rules2,after,new,sign,testflip[:,1],testu[:,5])
                _,old_good,old_p=choose(rules2,after,old_item,old_sign,earlier_flip,earlier_u)
                l1=write.receipt.weight_delta.flatten(1).abs().sum(1)
                for key,v in zip(samples,(side.to(torch.int8),good,p,old_good,old_p,l1),strict=True):
                    branch_values[key].append(v.numpy())
            for k in samples:
                samples[k].append(np.stack(branch_values[k]))
        if age == H:
            break
        if condition=="idle":
            state=net.wait(rules,state,21)
        else:
            obj=0 if condition=="repeat" else age
            write=support(rules,state,history[obj],private[obj],flip[age],draws[age],
                          "sham" if condition=="sense_only" else "active")
            state=net.wait(rules,write.state,4)
    return {k:np.stack(v) for k,v in {**samples,**metrics}.items()}


def run(label,loader,births,pairs=96,seed=584321,ages=AGES):
    torch.set_num_threads(1)
    directory=DATA/"plasticity"/label
    directory.mkdir(parents=True,exist_ok=True)
    report=dict(ages=ages,branches=BRANCHES,births=births,independent_histories=pairs,
                private_novel_laws_per_history=2,seed=seed,results={},hashes={})
    for cond in CONDITIONS:
        rows=[]
        for birth in births:
            model,rule=loader(birth)
            obs=evaluate(model,rule,seed,birth,pairs,cond,ages)
            path=directory/f"{cond}-{birth}.npz"
            np.savez_compressed(path,**obs)
            report["hashes"][path.name]=sha256(path.read_bytes()).hexdigest()
            rows.append({k:v.mean(axis=-1).tolist() for k,v in obs.items() if k!="new_side"})
        report["results"][cond]=rows
        print(label,cond,flush=True)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/f"plasticity-{label}.json").write_text(json.dumps(report,indent=1)+"\n",encoding="utf8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label",default="prior-odor")
    p.add_argument("--run",type=Path,default=ROOT/"data/local-trace-20261119/local")
    p.add_argument("--joint",action="store_true")
    p.add_argument("--arm",default="local_active")
    p.add_argument("--step",type=int,default=8000)
    p.add_argument("--births",nargs="+",type=int,default=list(range(731,737)))
    p.add_argument("--pairs",type=int,default=96)
    cfg=p.parse_args()
    if cfg.joint:
        from . import interleave_train as training
        def load(b):
            return training.load_checkpoint(cfg.run,cfg.arm,b,cfg.step)
    else:
        def load(b):
            return old.load_checkpoint(cfg.run,"odor","active",b,cfg.step)
    run(cfg.label,load,cfg.births,cfg.pairs)
