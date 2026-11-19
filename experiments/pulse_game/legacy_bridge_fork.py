"""Counterfactual receipt and later unreinforced READY queries for the transferred LIF rule."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from experiments.seed_evolution.large_runner import _write
from experiments.undirected import local_rule_world as world
from experiments.undirected.evolved_plasticity_world import FIRST_FINISH, N_ACTIONS, READY, _receipt_table
from experiments.undirected.uncapped_private_symbol_receipt_probe import clear_receipts
from experiments.undirected.uncapped_private_symbol_receipt_world import native_inputs, probe_frame, set_keys
from experiments.undirected.uncapped_reward_precision import _ACK,_REJ
from .legacy_bridge import BridgeAgent,Options
from . import research_audit as audit,ring_spikes


TEACH=32


def group(seeds,graph,cfg,opt,novel,reinforced):
    tapes=[world.schedule(seed=s,actors=8,visits=16) for s in seeds]
    dims=dict(public=2,stage=2,raw=1,context=1,target=1,maps=0)
    tape=replace(tapes[0],**{k:torch.cat([getattr(t,k) for t in tapes],dim=d) for k,d in dims.items()})
    raw=tape.raw[TEACH]; context=tape.context[TEACH];other=(raw+1)%4
    if novel:
        before=slice(0,TEACH)
        tape=set_keys(tape,before,(raw+2+tape.raw[before]%2)%4,tape.context[before])
    policy=BridgeAgent(cfg,opt,[s*8+i for s in seeds for i in range(8)],graph)
    policy.population=1
    saved=[]
    obs,actions=native_inputs(policy,tape,_receipt_table("cpu").half(),TEACH,lambda:saved.append(deepcopy(policy)))
    covered,first=(actions[0].ge(FIRST_FINISH)&actions[0].lt(N_ACTIONS)&obs[0,...,READY].bool()).max(0)
    pulse=first+1;eligible=covered & pulse.lt(len(actions[0]))
    answer=(actions[0].gather(0,first[None])[0]-FIRST_FINISH)%4
    truth=answer.eq(tape.target[TEACH]-FIRST_FINISH)
    actual_ack=obs[0,...,_ACK].gather(0,pulse.clamp_max(actions.size(1)-1)[None])[0].bool()
    assert ((actual_ack==truth)|~eligible).all()
    keys=[(raw,context),(raw,1-context),(other,context)]
    goals=torch.stack([tape.maps[torch.arange(len(raw)),c,k] for k,c in keys])
    answers={}
    for positive in (0,1):
        branch=deepcopy(saved[0])
        mismatch=torch.zeros_like(eligible)
        for tick,frame in enumerate(obs[0]):
            public=frame.clone();at=eligible & pulse.eq(tick);after=eligible & pulse.lt(tick)
            public[...,READY].masked_fill_(at|after,0);clear_receipts(public,after)
            public[...,_ACK].masked_fill_(at,positive);public[...,_REJ].masked_fill_(at,1-positive)
            chosen=branch.act(public)[...,0]
            mismatch |= (~eligible | (tick<pulse)) & (chosen!=actions[0,tick])
        assert not mismatch.any()
        for lag in range(9):
            if lag in (0,1,8):
                for key,(symbol,ctx) in enumerate(keys):
                    query=deepcopy(branch)
                    picked=torch.full_like(answer,-1)
                    for frame in tape.public[TEACH+9]:
                        f=probe_frame(frame,symbol,ctx)
                        act=query.act(f.flatten(1)[None])[...,0]
                        legal=f[:,0,READY].bool() & act.ge(FIRST_FINISH) & act.lt(N_ACTIONS)
                        picked=torch.where(picked.lt(0)&legal,(act-FIRST_FINISH)%4,picked)
                    answers[positive,lag,key]=picked
            if lag<8:
                for tick,frame in enumerate(obs[lag+1]):
                    f=frame.clone()
                    if not reinforced or (lag==0 and tick==0):
                        clear_receipts(f,torch.ones_like(eligible))
                    branch.act(f)
    rows=[]
    for j,seed in enumerate(seeds):
        sl=slice(j*8,j*8+8)
        for lag in (0,1,8):
            row={"seed":seed,"graph":graph,"lag":lag,"eligible":int(eligible[:,sl].sum())}
            valids=eligible[:,sl]
            yeses=[];nos=[]
            for key,name in enumerate(("same","other_context","other_raw")):
                a,b=answers[1,lag,key][:,sl],answers[0,lag,key][:,sl]
                true=torch.where(truth[:,sl],a,b);inverted=torch.where(truth[:,sl],b,a)
                true_correct=true.eq(goals[key,sl]);false_correct=inverted.eq(goals[key,sl])
                for label,mask in (("repeat_ack",a.eq(answer[:,sl])),("repeat_rej",b.eq(answer[:,sl])),
                                   ("true_correct",true_correct),("flipped_correct",false_correct),
                                   ("both_covered",a.ge(0)&b.ge(0))):
                    row[name+"_"+label]=int((valids&mask).sum())
                yeses.append(true_correct);nos.append(false_correct)
            row["joint_true"]=int((valids&yeses[0]&yeses[1]).sum())
            row["joint_flipped"]=int((valids&nos[0]&nos[1]).sum())
            rows.append(row)
    return rows


def summarize(rows,seeds):
    report={}
    for lag in (0,1,8):
        values={}
        for w in seeds:
            data=[r for r in rows if r["lag"]==lag and r["seed"]==w]
            count=sum(r["eligible"] for r in data)
            if not count:continue
            rate=lambda k:sum(r[k] for r in data)/count
            v={"joint_true":rate("joint_true"),"joint_gain":rate("joint_true")-rate("joint_flipped")}
            for key in ("same","other_context","other_raw"):
                v[key+"_true_gain"]=rate(key+"_true_correct")-rate(key+"_flipped_correct")
                v[key+"_repeat_effect"]=rate(key+"_repeat_ack")-rate(key+"_repeat_rej")
            v["cue_specificity"]=v["same_repeat_effect"]-v["other_raw_repeat_effect"]
            v["context_specificity"]=v["same_repeat_effect"]-v["other_context_repeat_effect"]
            for k,x in v.items():values.setdefault(k,[]).append(x)
        report[str(lag)]={k:audit.estimates(x) for k,x in values.items()}
    return report


def main():
    torch.set_num_threads(1)
    root=audit.OUT
    cfg=ring_spikes.Config(**json.loads(Path("data/pulse-game/ring-value-search-v1/frozen.json").read_text())["best"]["config"])
    sel=json.loads((root/"legacy-bridge-selected.json").read_text())["selected"]["continuous_lif"]["options"]
    opt=Options(**sel)
    seeds=audit.split("replication")[128:160]
    out={"seeds":seeds,"options":sel,"source_sha256":audit.fingerprint(__file__),"conditions":{}}
    for novel in (True,False):
        for reinforced in (False,True):
            condition=("novel" if novel else "practised")+("-rewarded-interference" if reinforced else "-silent-interference")
            rows=[]
            for graph in (3,19,37,53):
                for i in range(0,len(seeds),8):rows.extend(group(seeds[i:i+8],graph,cfg,opt,novel,reinforced))
            stats=summarize(rows,seeds)
            out["conditions"][condition]={"rows":rows,"effects":stats}
            _write(root/"legacy-bridge-receipt-forks.json",out)
            print(condition,{k:round(stats["8"][k]["mean"],4) for k in
                             ("same_true_gain","joint_gain","cue_specificity","context_specificity")},flush=True)


if __name__=="__main__":
    main()
