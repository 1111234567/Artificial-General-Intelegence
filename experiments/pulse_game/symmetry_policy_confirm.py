"""Final untouched physical worlds for reward-tuned policies, with independent NumPy rollouts."""
import argparse
import json
import time
import numpy as np
import torch

from . import mixture_flash as flash, symmetry_grounded as parent, symmetry_memory_deploy as native
from . import symmetry_policy as tested, symmetry_policy_train as train, symmetry_audit as controls
from . import symmetry_world as game, symmetry_memory_audit as metrics, research_audit as audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


SEED=768398311
CONDITIONS={
 "immediate":(12,4,0,.04,.08,True),"delay2":(12,4,2,.04,.08,True),"delay4":(12,4,4,.04,.08,True),
 "delay8":(12,4,8,.04,.08,True),"delay12":(12,4,12,.04,.08,True),"over_capacity":(12,4,24,.04,.08,True),
 "other_habitat_gap":(12,4,8,.04,.08,False),"long_A_interval":(16,16,2,.04,.14,True),
 "more_noise":(16,8,2,.20,.18,True),"two_examples":(2,2,0,.04,.08,True),
 "identical_A_history":(12,4,0,.04,.08,True)}


def episode(seeds,setting,mask):
    p,span,gap,noise,jitter,same=setting
    base=controls.joint(seeds,pairs=p,probes=4,span=span,noise=noise,jitter=jitter,matched=True,mask=mask)
    return game.with_distractors(base,gap,same_habitat=same)


def extract(w,r,master):
    data=metrics.rows(w,r,master)
    first=np.flatnonzero(w.query[0])[0]
    data["anchor"]=w.anchor_sign[:len(w.family)//2,first].astype(float)
    return {k:v.tolist() for k,v in data.items()}


def run(path,seeds,folder):
    fit=json.loads(path.read_text());settings=fit["settings"];mode=settings["mode"]
    checkpoint=OUT/"symmetry-grounded-v1"/(settings["parent"]+".pt")
    if audit.fingerprint(checkpoint)!=settings["parent_sha"]:raise RuntimeError("Frozen parent changed")
    p=parent.load(checkpoint);candidate=np.asarray(fit["best"]["genome"])[None]
    native_params=tested.materialize(p,candidate,mode)
    table={};compatibility={}
    for cond,config in CONDITIONS.items():
        w=episode(seeds,config,cond=="identical_A_history")
        model=tested.Cohort(p,w)
        variants={"selected":(candidate,0),"parent":(np.zeros_like(candidate),0)}
        if cond in ("immediate","delay4","over_capacity"):
            variants.update(transplanted=(candidate,1),clear_rule=(candidate,2),transplant_events=(candidate,3),
                            no_write=(candidate,4),no_anchor=(candidate,5),invert_rule=(candidate,6))
        table[cond]={}
        for key,(v,flag) in variants.items():
            r,m=model.run(v,mode,individual=True,intervention=flag)[0]
            table[cond][key]=extract(w,r,m)
            rows=table[cond][key]
            if cond=="identical_A_history" and (np.any(rows["both"]) or not np.array_equal(rows["first_side_even"],rows["first_side_odd"])):
                raise RuntimeError("Hidden goal reached policy before first decision")
        if cond in ("immediate","delay4"):
            sample=seeds[:48];held=episode(sample,config,False)
            sim=tested.Cohort(p,held);r,g=sim.run(candidate,mode,individual=True)[0]
            cpu=native.Forager(native_params,sample+sample,boundary=held.exposed)
            ref=flash.rollout(cpu,held.menu)
            compatibility[cond]=dict(choice_match=float(np.mean(r.contact_side==ref.contact_side)),
                 tick_match=float(np.mean(r.contact_tick==ref.contact_tick)),reward_match=float(np.mean(r.reward==ref.reward)),
                 master_max_error=float(np.max(np.abs(g-np.stack(cpu.record,1)))),worlds=len(sample),events=int(ref.reward.size))
    _write(folder/(path.parent.name+".json"),dict(settings=settings,seeds=seeds,configurations=CONDITIONS,
           cases=table,parity=compatibility,gen=fit["best"]["gen"],phenotype_hash=audit.fingerprint(path),
           pair_choice_protocol="both_first_contacts_required"))
    print("confirmed",path.parent.name,compatibility,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--folder",default="symmetry-policy-v1")
    p.add_argument("--worlds",type=int,default=1024);args=p.parse_args();torch.set_num_threads(1)
    root=OUT/args.folder;dest=audit.OUT/(args.folder+"-sealed");dest.mkdir(exist_ok=True)
    seeds=train.evalseeds(SEED,args.worlds)
    if len(set(seeds))!=len(seeds):raise RuntimeError("Duplicate life")
    expected=6
    while True:
        paths=sorted(root.glob("*/frozen.json"))
        for path in paths:
            if not (dest/(path.parent.name+".json")).is_file():run(path,seeds,dest)
        if len(paths)==expected:break
        time.sleep(8)
    _write(dest/"complete.json",dict(worlds=len(seeds),models=expected))


if __name__=="__main__":main()
