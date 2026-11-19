"""Can observations from long-experience games calibrate reading as well as writing, at fixed storage?"""
import argparse
from itertools import product
import json
import time
import numpy as np
import torch
from . import recurring_memory as plastic, recurring_train as previous, recurring_world as task
from . import recurring_audit as earlier, symmetry_policy_train as base, symmetry_audit as intact
from . import symmetry_world as game,research_audit as audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


SCHED={
 "old_immediate":None,
 "old_interleaved":0,
 "repeated_anchor":dict(gap=32,distinct=6,query="anchor",recurring=True),
 "rare":dict(gap=40,distinct=8,query="rare",recurring=True,noise=.13),
 "many":dict(gap=48,distinct=16,query="random",recurring=True,spread=.13),
 "novel":dict(gap=32,distinct=12,query="random",recurring=False,noise=.10),
 "older":dict(gap=64,distinct=6,query="anchor",recurring=True),
 "more_noise":dict(gap=40,distinct=8,query="random",recurring=True,noise=.26,spread=.16),
 "novel_anchor":dict(gap=40,distinct=8,query="anchor",recurring=False),
 "similar_conflicts":dict(gap=40,distinct=12,query="random",recurring=True,conflicts=True,spread=.15),
}


def world_for(seeds,condition,nonce):
    opts=SCHED[condition]
    if not isinstance(opts,dict):
        w=intact.joint(seeds,pairs=12,probes=3,span=4,noise=.04,jitter=.08,matched=True)
        return w if opts is None else game.with_distractors(w,4,nonce=nonce)
    return task.make(seeds,**opts,nonce=nonce)[0]


def policies(body,genes,points):
    cells=np.broadcast_to(body,(len(points),len(body))).copy()
    cells[:,7]=points[:,0];cells[:,9]=points[:,1]
    w=np.broadcast_to(plastic.traits(genes),(len(points),7))
    return np.column_stack((cells,w)).astype(np.float32)


def search(par,folder):
    params,body,_,source=previous.parent(par)
    original=OUT/"recurring-write-v1"/(par+"-writer2")/"frozen.json"
    f=json.loads(original.read_text());genes=np.asarray(f["genes"])
    points=np.array([[body[7],body[9]],*product([0.,.012,.025,.05,.09,.14,.19,.26],
                    [.65,.72,.78,.83,.875,.91,.95])],np.float32)
    raw=policies(body,genes,points)
    trainkeys=("old_immediate","old_interleaved","repeated_anchor","rare","many","novel")
    trainrows={"FIFO":[],"consolidate":[]}
    seeds=base.evalseeds(17536019,192)
    for cond in trainkeys:
        w=world_for(seeds,cond,1736311);co=plastic.Cohort(params,w)
        for mode,tag in ((0,"FIFO"),(2,"consolidate")):
            kern=plastic.kernel(raw.shape[1],params["capacity"],mode)
            trainrows[tag].append(co.run_decoded(raw,kern))
    picks={}
    for tag,sets in trainrows.items():
        results=np.stack(sets);score=base.utility(results).mean(0)
        j=int(score.argmax())
        picks[tag]=dict(index=j,age=float(points[j,0]),null=float(points[j,1]),mean_score=float(score[j]),
                        all_cases=results[:,j].tolist(),native=results[:,0].tolist())
    freeze=dict(parent=par,write=genes.tolist(),development_worlds=seeds,conditions=trainkeys,
                all_candidates=points.tolist(),winners=picks,parent_writer_sha=audit.fingerprint(original),
                source=source["settings"]["parent_sha"])
    _write(folder/(par+"-frozen.json"),freeze)
    return freeze


def confirm(par,freeze,seeds,dest):
    params,body,birth,src=previous.parent(par)
    d=freeze["winners"];g=np.array(freeze["write"])
    result={}
    for cond in SCHED:
        w=world_for(seeds,cond,68019437);co=plastic.Cohort(params,w);result[cond]={}
        for label,mode,point in (
            ("parent_FIFO",0,[body[7],body[9]]),("write_only",2,[body[7],body[9]]),
            ("read_only",0,[d["FIFO"]["age"],d["FIFO"]["null"]]),
            ("read_and_write",2,[d["consolidate"]["age"],d["consolidate"]["null"]]),
            ("averaging_only",3,[d["consolidate"]["age"],d["consolidate"]["null"]]),
            ("latest_taste_only",4,[d["consolidate"]["age"],d["consolidate"]["null"]]),
        ):
            ph=policies(body,g,np.asarray([point]))
            kern=plastic.kernel(ph.shape[1],params["capacity"],mode)
            r,m=co.run_decoded(ph,kern,individual=True)[0]
            result[cond][label]=earlier.summarize(w,r,m)
    _write(dest/(par+".json"),dict(parent=par,roots=seeds,winners=d,cases=result,
           candidate_scores="actual net outcomes at free physical choices",generative_nonce=68019437))
    print(par,d["FIFO"]["age"],d["consolidate"]["age"],flush=True)


def main():
    torch.set_num_threads(1);folder=OUT/"recurring-read-write-v2";folder.mkdir(exist_ok=True)
    dest=audit.OUT/"recurring-read-write-sealed";dest.mkdir(exist_ok=True)
    roots=[f.parent.name for f in sorted((OUT/"symmetry-policy-v1").glob("*--full-*/frozen.json"))]
    if len(roots)!=3:raise RuntimeError("Need all independent parents")
    seeds=base.evalseeds(203893711,1024)
    for name in roots:
        path=folder/(name+"-frozen.json")
        frozen=json.loads(path.read_text()) if path.exists() else search(name,folder)
        if not (dest/(name+".json")).is_file():confirm(name,frozen,seeds,dest)
    _write(dest/"complete.json",dict(parents=roots,worlds=len(seeds)))


if __name__=="__main__":main()
