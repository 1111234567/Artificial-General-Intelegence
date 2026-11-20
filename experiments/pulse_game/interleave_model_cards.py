"""Run complementary capabilities on the very same frozen graphs and export portable inference copies."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_capacity as cap, interleave_capacity_report as ci
from . import interleave_stream as stream, interleave_induction as induction
from . import interleave_quest_train as quest, interleave_quest_eval as qeval
from . import local_trace_train as old
from .interleave_interference import DATA,EVIDENCE


def families(cohort):
    confirm=cohort=="confirmation"
    return {
        "uniform memory":stream.Source("objective",DATA/("retention-objective-confirmation" if confirm else "retention-objective"),"uniform",8000),
        "cold joint":stream.Source("joint",DATA/("cold-replication" if confirm else "cold"),"local_active",8000),
        "context then joint":stream.Source("joint",DATA/("joint-continuation-replication" if confirm else "joint-continuation")/"full-direct","local_active",8000),
        "uniform then joint":stream.Source("joint",DATA/("joint-continuation-replication" if confirm else "joint-continuation")/"uniform","local_active",8000),
        "serial mean":stream.Source("joint",DATA/("terminal-quest-replication" if confirm else "terminal-quest"),"mean",8000),
        "serial conjunction":stream.Source("joint",DATA/("terminal-quest-replication" if confirm else "terminal-quest"),"conjunction",8000),
    }


@torch.no_grad()
def row(graph,rule,seed,worlds):
    indicators={};raw={}
    tape=cap.make_tape("odor",seed,worlds,16)
    z=cap.evaluate(graph,rule,tape,(0,1,4,16),snapshot_worlds=0)
    for k in (0,1,4,16):
        cut=z["capacities"].tolist().index(k)
        good=(z["side"][0,cut,:k+1]==z["right_value"][:,:k+1,-1].T)
        indicators[f"A_after_{k}"]=good[0].reshape(worlds,4).mean(1)
        if k:
            indicators[f"A_and_recent_after_{k}"]=(good[0]&good[-1]).reshape(worlds,4).mean(1)
            indicators[f"all_{k+1}"]=good.all(0).reshape(worlds,4).mean(1)
        raw[f"capacity/{k}"]=good
    induction_tape=induction.world(seed+5,0,worlds,"linear",0.,8)
    fresh=induction.evaluate(graph,rule,induction_tape,(8,))
    for mode in ("coherent","unrelated_history"):
        r=fresh[f"{mode}/8"].reshape(worlds,4,-1)
        indicators[f"{mode}/unseen"]=r[:,[1,3],0].mean(1)
        indicators[f"{mode}/exception_after_one"]=r[:,[0,2],2].mean(1)
        indicators[f"{mode}/exception_and_typical"]=r[:,[0,2],4].mean(1)
        raw[f"induction/{mode}"]=r
    c=quest.Config(worlds=worlds,seed=seed+17)
    trial=quest.make_tape([0],c,1,domain=7391511,novel=True,shared=True)
    solved=qeval.native_payload(quest.play(graph,rule,trial,c.max_wait),trial)
    uniform=qeval.native_payload(qeval.native_reference(trial,"uniform"),trial)
    law=solved["law"][0,0]
    select=law[:,0]!=law[:,1]
    for label in qeval.NATIVE_LABELS:
        v=solved["success/"+label][0]
        indicators["serial_conflict/"+label]=v[:,select].mean(1)
        indicators["serial_uniform_conflict/"+label]=uniform["success/"+label][0][:,select].mean(1)
        raw["serial/"+label]=v
    return indicators,raw


def run(out,cohort,worlds,seed):
    torch.set_num_threads(1)
    births=list(range(731,737)) if cohort=="original" else list(range(811,817))
    models=families(cohort)
    dest=out/cohort;dest.mkdir(parents=True,exist_ok=True)
    report=dict(births=births,worlds=worlds,seed=seed,
                note="Each row uses a single frozen inherited graph and rule across every assay, with no outer optimizer "
                     "in any deployment. The native serial assay samples three successive motor actions on the fixed neural "
                     "contact clock with pre-integrated travel endpoints and no intervening taste. The object capacity queries are independent forks. Shared cohorts and world "
                     "geometries are paired before resampling. No means across tasks are used to select a graph.",
                models={},hashes={})
    intervals=ci.Crossed(len(births),worlds,seed+19,4000)
    families_data={}
    for label,spec in models.items():
        rootrows=[]
        for b in births:
            g,r=spec.load(b)
            metrics,raw=row(g,r,seed,worlds)
            key=label.replace(" ","-")
            path=dest/f"{key}-{b}.npz";np.savez_compressed(path,**raw,**{f"metric/{k}":v for k,v in metrics.items()})
            report["hashes"][path.name]=sha256(path.read_bytes()).hexdigest()
            # Save only inference tensors in a checked, small format readable by the existing graph loader.
            export=dest/"learners"
            old.save_checkpoint(export,"mixed",key,b,spec.step,g,r,0)
            restored,rule=old.load_checkpoint(export,"mixed",key,b,spec.step)
            if not all(torch.equal(v,rule.state_dict()[k]) for k,v in r.state_dict().items()):
                raise ValueError("Export changed the inherited update rule")
            if not all(torch.equal(getattr(g,k),getattr(restored,k)) for k in cap.GRAPH_FIELDS):
                raise ValueError("Export changed a graph parameter")
            p=old.checkpoint_path(export,"mixed",key,b,spec.step)
            report["hashes"][p.relative_to(dest).as_posix()]=sha256(p.read_bytes()).hexdigest()
            rootrows.append(metrics)
        bank={k:np.stack([v[k] for v in rootrows]) for k in rootrows[0]}
        families_data[label]=bank
        rel=spec.path.relative_to(DATA)
        report["models"][label]=dict(folder=rel.as_posix(),arm=spec.arm,step=spec.step,
                                      scores={k:intervals(v) for k,v in bank.items()})
        print(cohort,label,flush=True)
    report["contrasts"]={}
    for a,b in (("context then joint","uniform then joint"),("serial conjunction","serial mean"),
                ("context then joint","cold joint")):
        report["contrasts"][a+" minus "+b]={k:intervals(v-families_data[b][k]) for k,v in families_data[a].items()}
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    p=EVIDENCE/f"model-cards-{cohort}.json"
    p.write_text(json.dumps(report,indent=1)+"\n",encoding="utf8")


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--cohort",choices=("original","confirmation"),required=True)
    p.add_argument("--out",type=Path,default=DATA/"model-cards")
    p.add_argument("--worlds",type=int,default=128)
    a=p.parse_args()
    run(a.out,a.cohort,a.worlds,3917911 if a.cohort=="original" else 3917929)
