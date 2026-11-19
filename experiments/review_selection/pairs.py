"""Free action when two consecutive requests in one slot repeat the raw symbol."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.continuity.world import OnlineWorld, NEW
from experiments.experience_transfer.train_transfer import SEEDS
from experiments.learn_remember.gru_train import GRUAgent
from experiments.lifelong.evaluate import CONDITIONS


ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"data/review-selection/pairs-v1"
MAIN=ROOT/"data/review-selection/train-v1"
OLD=ROOT/"data/experience-transfer/training-v1"
ARMS=("A_start","no_review","uniform","conflict","error")
PATTERNS=("AA","AB","BA","BB")
WARM,TEST=512,4096


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def declare():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"protocol.json").open("x",encoding="utf-8") as f:
        json.dump(dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
            seeds=[0,1,2],seed_offset=849230,actor_ids=list(range(55000,55016)),families="A,B",
            ages=dict(warm=WARM,scored=TEST,dt=.25),arms=ARMS,conditions=CONDITIONS["long_both"],
            cases="Native two-slot OnlineWorld timing, versions, controller action receipts and deadlines; no cancellations or replacements for this supplemental paired test. On each slot form nonoverlapping pairs of adjacent NEWs with the same raw. Within blocks randomly permute all4 raws x AA/AB/BA/BB; never expose pair markers.",
            allocation="Same seed/cue table/future commands across all arms. Only ordinary actions/receipts drive the state. Birth h initialized once; 512 free-action warm ticks. No teacher intervention. Score a pair only if both NEWs occurred after warmup and both matured before end.",
            primary="Both requests strictly completed (no errors on either) among AB or BA pairs where A and B demand different mapped outputs. Keep AA/BB and non-conflicting counts as controls, split directions. Independent unit is the training seed.",
            controls="Original learned A and previous A->B no-review endpoints; new trained checkpoints for uniform/conflict/error. Additional mixed schedule's absolute rates have a different testing population from the original random-interleaved benchmark.",
            source_world_sha256=sha(ROOT/"experiments/continuity/world.py")),f,indent=2)


class PairedWorld(OnlineWorld):
    def __init__(self,seed):
        super().__init__(seed=849230+seed,batch_size=16,actor_ids=tuple(range(55000,55016)),
                         n_jobs=2,active_jobs=2,dt=.25,cancel=0.,replace=0.,**CONDITIONS["long_both"])
        self.specification=SEEDS[seed]
        shape=(self.batch_size,self.n_jobs)
        self.number=np.full(shape,-1,dtype=np.int64)
        self.raw=np.zeros(shape,dtype=np.int64)
        self.family=np.zeros(shape,dtype=np.int64)
        # Each slot has its own random order across complete factorial blocks.
        rng=np.random.default_rng(849880+seed)
        self.design=np.argsort(rng.random((*shape,32,16)),axis=-1)

    def _events(self):
        event=super()._events()
        assert np.isin(event["kind"],[0,NEW]).all()
        lane,job=np.where(event["kind"]==NEW)
        self.number[lane,job]+=1
        pair=self.number[lane,job]//2
        code=self.design[lane,job,pair//16,pair%16]
        raw=code//4
        b=(code%4 >> (1-self.number[lane,job]%2)) & 1
        self.raw[lane,job]=raw
        self.family[lane,job]=np.where(b,self.specification["B_cue"],self.specification["history_cue"])
        event["payload"][lane,job]=np.where(b,np.array(self.specification["B"])[raw],np.array(self.specification["A"])[raw])
        return event

    def observe(self):
        obs=super().observe().reshape(self.batch_size,2,24)
        cmd=self.event_kind.eq(NEW)[...,None]
        obs[...,5:9]=F.one_hot(torch.from_numpy(self.raw),4)*cmd
        context=F.one_hot(torch.from_numpy(self.family),4)*cmd
        return torch.cat((obs,context),dim=-1).reshape(self.batch_size,56)


def path_for(seed,arm):
    if arm in ("A_start","no_review"):
        return OLD/f"seed_{seed:03d}/related"/("B000.pt" if arm=="A_start" else "B008.pt")
    return MAIN/f"seed_{seed:03d}/{arm}/B008.pt"


def pair_counts(rows):
    pairs={}
    for r in rows:
        group=(r["lane"],r["job"],r["seq"]//2)
        pairs.setdefault(group,{})[r["seq"]%2]=r
    totals={}
    for items in pairs.values():
        if len(items)!=2:
            continue
        a,b=items[0],items[1]
        if a["issued"]<WARM or not (a["matured"] and b["matured"]):
            continue
        assert a["raw"]==b["raw"]
        order=a["context"]+b["context"]
        part="conflict" if a["conflict"] else "shared"
        for group in (order,"switch" if order in ("AB","BA") else "repeat"):
            c=totals.setdefault(part+"_"+group,Counter())
            c.update(pairs=1,both_strict=a["strict"] and b["strict"],both_ordinary=a["done"] and b["done"],
                     first_strict=a["strict"],second_strict=b["strict"])
    return totals


@torch.inference_mode()
def rollout(seed,arm,stop=None):
    w=PairedWorld(seed)
    if arm not in ("oracle","wait"):
        cp=torch.load(path_for(seed,arm),weights_only=False,map_location="cpu")
        model=GRUAgent(**cp["model_config"]);model.load_state_dict(cp["model"]);model.eval()
        h=model.initial_state(w.batch_size)
    rows={}; actions=[]; receipts=[]; active=[]; ids=[]; events=hashlib.sha256()
    for t in range(WARM+TEST if stop is None else stop):
        x=w.observe()
        for lane,j in w.matured_mask.nonzero().tolist():
            r=rows[lane,j,int(w.matured_request_id[lane,j])]
            r.update(matured=True,done=bool(w.matured_completed[lane,j]),strict=bool(w.matured_clean_completed[lane,j]))
        for lane,j in w.event_kind.eq(NEW).nonzero().tolist():
            rid=int(w.request_id[lane,j]); raw=int(w.raw[lane,j])
            role="A" if int(w.family[lane,j])==w.specification["history_cue"] else "B"
            mapped=w.specification[role][raw]
            assert int(w.payload[lane,j])==mapped and bool(w.event_eligible[lane,j])
            public=x[lane].reshape(2,28)[j]
            assert int(public[5:9].argmax())==raw and int(public[24:28].argmax())==w.family[lane,j]
            r=dict(lane=lane,job=j,request=rid,raw=raw,context=role,target=mapped,version=int(w.version[lane,j]),
                   issued=t,seq=int(w.number[lane,j]),conflict=raw in w.specification["conflict_raw"],
                   duration=float(w.duration[lane,j]),deadline=float(w.deadline[lane,j]),matured=False,done=False,strict=False)
            events.update(json.dumps(r,separators=(",",":")).encode())
            rows[lane,j,rid]=r
        if arm=="oracle":
            act=w.oracle("midpoint")
        elif arm=="wait":
            act=torch.zeros_like(w.version)
        else:
            scores,h=model.step(x,h,w.elapsed)
            act=scores.argmax(-1)
        active.append(w.active.numpy().copy());ids.append(w.request_id.numpy().copy())
        w.act(act);actions.append(act.numpy().copy());receipts.append(w.last_result.numpy().copy())
    facts=dict(active=np.stack(active),actions=np.stack(actions),receipts=np.stack(receipts),ids=np.stack(ids))
    idle=~facts["active"][WARM:];rows=list(rows.values())
    out=dict(seed=seed,arm=arm,events_sha=events.hexdigest(),requests=rows,pair_counts=pair_counts(rows),
             idle_opportunities=int(idle.sum()),idle_actions=int((idle & (facts["actions"][WARM:]!=0)).sum()))
    return out,facts


def run(seed,arm):
    torch.set_num_threads(1)
    plan=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    assert plan["source_sha256"]==sha(__file__)
    path=OUT/f"seed_{seed:03d}_{arm}.json"
    if path.exists():raise FileExistsError(path)
    outcome,facts=rollout(seed,arm)
    facts_path=path.with_suffix(".npz");np.savez_compressed(facts_path,**facts)
    outcome.update(facts_sha=sha(facts_path),checkpoint_sha=sha(path_for(seed,arm)),protocol_sha=sha(OUT/"protocol.json"))
    path.write_text(json.dumps(outcome,separators=(",",":")),encoding="utf-8")
    print(seed,arm,outcome["pair_counts"]["conflict_switch"],flush=True)


def preflight():
    torch.set_num_threads(1)
    oracle,_=rollout(0,"oracle",1024);wait,_=rollout(0,"wait",1024)
    assert oracle["events_sha"]==wait["events_sha"]
    for key,row in oracle["pair_counts"].items():
        assert row["both_strict"]==row["pairs"]==wait["pair_counts"][key]["pairs"]
        assert wait["pair_counts"][key]["both_strict"]==0
    assert oracle["pair_counts"]["conflict_switch"]["pairs"]>0
    print("Teacher/WAIT controls and matched external stimuli passed")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=("declare","check","run"))
    parser.add_argument("--seed",type=int,choices=range(3));parser.add_argument("--arm",choices=ARMS)
    args=parser.parse_args()
    if args.command=="declare":declare()
    elif args.command=="check":preflight()
    else:run(args.seed,args.arm)
