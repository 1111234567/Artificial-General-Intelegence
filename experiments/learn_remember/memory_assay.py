"""Paired raw/context cue changes and hidden-state substitution after the cue."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch

from experiments.lifelong.world import FAMILY_MAPPINGS


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/"data/learn-remember"
OUT = BASE/"memory-v1"
DURATION = 8
LEGAL_TICK = 32
SWAP_TICK = 16
WARM_MIN = 128


def source(seed):
    return ROOT/f"data/content-timing/matched-v1/seed_{seed:03d}/background_f3.pt"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def declare():
    OUT.mkdir(parents=True,exist_ok=True)
    plan = dict(format=1, created_utc=datetime.now(timezone.utc).isoformat(),
        goal="Does a single raw/context cue causally change the later content decisions when every subsequent observation and feedback is held identical?",
        inputs={str(source(s)):sha(source(s)) for s in range(3)},
        anchors="All previously saved family3 backgrounds with at least128 ticks of continuous public history; selection is independent of new model scores. The learner never fits these assay variants.",
        conditions="For every real background keep its original slot/version/window and other-slot observations, vary public raw payload0..3 and family0..3 at its cue. Fix duration8. Lock target correct-version BEGIN at cue and WAIT thereafter. The only new feedback is the accepted BEGIN, which reveals version but not payload or family.",
        phases="Measure conditional 8-way/version+mapped payload ranking and 11-way FINISH choice at the first midpoint-teacher FINISH opportunity. These are scores under fixed actions, not free completions. Report any autonomous success independently.",
        state_intervention="After processing tick16 copy the entire hidden state between two matched branches with identical family/version/world suffix but raw=(raw+1) mod4. At the final read, compare against the recipient and donor labels. Require donor score equality within dtype tolerance.",
        aggregation="Each target/family has one variant per anchor. Family3 raw1 and raw3 targets differ from every retained family's target. Group by independent model seed. A rule-blind oracle that knows raw and version can get at most50% across the balanced four families.",
        stages="Compare untrained initialization, end of acquisition and end of withdrawal; include previous undirected end_only as a diagnostic comparator. Each candidate receives its own hidden state on the same reference history; no shared hidden dimensions required.")
    with (OUT/"protocol.json").open("x",encoding="utf-8") as f:
        json.dump(plan,f,indent=2)


def make_branches(background):
    anchors=[a for a in background["anchors"] if a["source_index"]>=WARM_MIN]
    if not anchors:
        raise ValueError("no warmed anchors")
    samples = [(a,f,raw) for a in anchors for f in range(4) for raw in range(4)]
    obs = background["observations"]
    horizon = max(LEGAL_TICK+int((a["window"]*4+1)//1) for a in anchors)
    public = torch.zeros(horizon+1,len(samples),56,dtype=obs.dtype)
    metadata = []
    for i,(a,f,raw) in enumerate(samples):
        t,lane,j=a["source_index"],a["actor_lane"],a["job"]
        w=a["window"]
        assert a["next_external_time"]>a["cue_time"]+8+w
        # Past the terminal teacher measurement the corresponding rows are unused.
        length=min(len(obs)-t,horizon+1)
        public[:length,i]=obs[t:t+length,lane]
        slot=public[:,i,j*28:(j+1)*28]
        slot[0,5:9]=0; slot[0,5+raw]=1
        slot[0,24:28]=0; slot[0,24+f]=1
        slot[0,9]=DURATION
        slot[1:]=0
        slot[1,11+1+a["version"]]=1
        slot[1,22]=1
        teacher_tick=int(torch.ceil(torch.tensor((8+w/2-1e-6)*4,dtype=torch.float64)))
        assert LEGAL_TICK<=teacher_tick<length
        metadata.append(dict(anchor=a["anchor_id"], source_index=t, lane=lane, job=j, family=f, raw=raw,
            version=a["version"], teacher_tick=teacher_tick, target=4*a["version"]+FAMILY_MAPPINGS[f][raw]))
    assert torch.equal(public[1:], public[1:].reshape(horizon,len(anchors),16,56)[:,:,:1].expand(-1,-1,16,-1).reshape_as(public[1:]))
    return anchors,public,metadata


@torch.inference_mode()
def measure(model, initial, background):
    anchors,public,rows = make_branches(background)
    dtype=next(model.parameters()).dtype
    prior=background["observations"].to(dtype)
    by_time={}
    for a in anchors:
        by_time.setdefault(a["source_index"],[]).append(a)
    state=initial.clone()
    histories={}
    for t in range(max(by_time)+1):
        for a in by_time.get(t,()):
            histories[a["anchor_id"]]=state[a["actor_lane"]].clone()
        if t < max(by_time):
            _,state=model.step(prior[t],state,.25)
    state=torch.stack([histories[r["anchor"]] for r in rows])
    public=public.to(dtype)
    donor=torch.arange(len(rows)).reshape(-1,4).roll(-1,dims=1).flatten()
    target_ticks=torch.tensor([r["teacher_tick"] for r in rows])
    jobs=torch.tensor([r["job"] for r in rows])
    normal_score=torch.zeros(len(rows),11,dtype=dtype)
    swap_score=torch.zeros_like(normal_score)
    for t in range(int(target_ticks.max())+1):
        current,state=model.step(public[t],state,.25)
        if t==SWAP_TICK:
            swapped=state.index_select(0,donor)
        if t>SWAP_TICK:
            changed,swapped=model.step(public[t],swapped,.25)
            ix=torch.nonzero(target_ticks==t).flatten()
            normal_score[ix]=current[ix,jobs[ix]]
            swap_score[ix]=changed[ix,jobs[ix]]
    err=float((swap_score-normal_score[donor]).abs().max())
    assert torch.allclose(swap_score,normal_score[donor],atol=2e-5,rtol=2e-5),err
    groups={}
    for i,r in enumerate(rows):
        target=r["target"]
        a=int(normal_score[i].argmax()); content=int(normal_score[i,3:].argmax())
        sc=int(swap_score[i,3:].argmax())
        r.update(original_content=content,original_action=a,swapped_content=sc,donor_target=rows[int(donor[i])]["target"])
        masks=["all",f"family{r['family']}"]
        if r["family"]==3 and r["raw"] in (1,3): masks.append("family3_exclusive")
        for mask in masks:
            c=groups.setdefault(mask,Counter())
            c.update(n=1, content_correct=int(content==target), wants_finish=int(a>=3), wants_correct_finish=int(a==3+target),
                swap_recipient_correct=int(sc==target),swap_donor_correct=int(sc==r["donor_target"]))
    return dict(anchors=len(anchors),groups=groups,rows=rows,max_donor_score_error=err)


def run(seed,stage,model,initial,model_path):
    torch.set_num_threads(1)
    plan=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    assert sha(source(seed))==plan["inputs"][str(source(seed))]
    background=torch.load(source(seed),map_location="cpu",weights_only=False)
    result=measure(model.eval(),initial,background)
    result.update(seed=seed,stage=stage,model_sha=sha(model_path),source_sha=sha(source(seed)),
                  protocol_sha=sha(OUT/"protocol.json"),script_sha=sha(__file__))
    with (OUT/f"{stage}_s{seed}.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,separators=(",",":"))
    print(seed,stage,result["groups"]["all"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["declare","graph"])
    parser.add_argument("--seed",type=int,choices=range(3))
    args=parser.parse_args()
    if args.command=="declare":
        declare()
    else:
        from experiments.wait_content.train import load_trained
        model,saved=load_trained(args.seed,"end_only")
        # The public reference history starts from the old 147456 world/graph h.
        bg=torch.load(source(args.seed),map_location="cpu",weights_only=False)
        run(args.seed,"previous_graph",model,bg["initial_h"],
            ROOT/f"data/wait-content/train-v1/seed_{args.seed:03d}/end_only/checkpoint.pt")
