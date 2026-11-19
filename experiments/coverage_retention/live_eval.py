"""Return A in the same world and hidden state after control/B-only training."""
from datetime import datetime,timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import argparse
import os
import subprocess
import sys
import numpy as np
import torch

from experiments.action_split import evaluate as base
from experiments.action_split.data import sha,read,write_new
from experiments.experience_transfer.train_transfer import mapping_scope,mapping_table,SEEDS
from experiments.lifelong.world import ContextWorld
from experiments.remaining_time.model import RemainingTimeGRUAgent
from experiments.remaining_time.evaluate import time_statistics

ROOT=base.ROOT
OUT=ROOT/"data/coverage-retention/live-v1"
POLICIES=("control","withdraw")
WARM,STEPS=512,4096


def checkpoint(seed,policy):
    from experiments.coverage_retention.train import checkpoint_path
    return checkpoint_path(seed,policy,256)


def world_path(seed,policy):
    from experiments.coverage_retention.data import states_path
    return states_path(seed,policy)


def stem(seed,policy):return OUT/f"seed_{seed:03d}"/f"{policy}_long"


def declare():
    sources={}
    for path in (Path(__file__),Path(__file__).with_name("eval_test.py"),Path(__file__).with_name("eval_audit.py"),
                 Path(__file__).with_name("analysis-plan.md"),ROOT/"experiments/action_split/evaluate.py",
                 ROOT/"experiments/remaining_time/model.py",ROOT/"experiments/remaining_time/evaluate.py",
                 ROOT/"experiments/lifelong/world.py",ROOT/"experiments/continuity/world.py"):
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    write_new(OUT/"protocol.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources,
        seeds=list(range(3)),policies=POLICIES,conditions={"long":[8.,16.]},warm=WARM,test=STEPS,dt=.25,
        initialization="Exact checkpointmodel/hidden and actual branchfinal world afterteacheract. Restore futurefamilyAB only through nativeAPI; no h/clock/RNG/request/receipt/scheduler reset. No optimizerupdate.",
        action="Native11argmax, timepredictionread-only. Allown warm andtest, no teacheractions.",
        denominator="Same completepostwarm neweligiblemature requests as freshscorer, localissued/matureindices withglobalphysicalclock/requestIDs. Inheritedrequests neverpromoted to testdenominator.",
        pairing="control/withdraw trainingphysicalstream ismatched. After512 ownwarm, futureAB events andmaturednewcontextidentities mustmatch. Do not comparefullhistoryeventhashes, which includepre-withdrawalcontext.",
        scope="Finite oldA return without weightlearning andwithout state reset; cannotcomparethe differentcases directly to freshworldpercentages as a pairedstateintervention.",
        diagnostic_bucket="Legacy overdue_beyond_training bucket is literal remaining<-.25 only; some such labels were seen under midpointteacher."))


def freeze():
    from experiments.coverage_retention import train
    assert (train.OUT/"globalfreeze.json").exists()
    plan=read(OUT/"protocol.json")
    for path,digest in plan["sources"].items():assert sha(ROOT/path)==digest,path
    files={str(path.relative_to(ROOT)):sha(path) for s in range(3) for p in POLICIES for path in (checkpoint(s,p),world_path(s,p))}
    write_new(OUT/"freeze.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(OUT/"protocol.json"),training_freeze_sha256=sha(train.OUT/"globalfreeze.json"),files=files))


def restore(seed,policy):
    """Copy numerical endpoint first; the only world edit is future-family policy."""
    cp=torch.load(checkpoint(seed,policy),map_location="cpu",weights_only=False)
    states=torch.load(world_path(seed,policy),map_location="cpu",weights_only=False)
    initial=deepcopy(states["final"])
    model=RemainingTimeGRUAgent(**cp["model_config"])
    model.load_state_dict(cp["model"]);model.eval().requires_grad_(False)
    h=cp["hidden"].clone()
    world=ContextWorld.from_state_dict(initial,device="cpu")
    world.set_family_policy((SEEDS[seed]["history_cue"],SEEDS[seed]["B_cue"]))
    return model,h,world,initial


@torch.inference_mode()
def run(seed,policy):
    torch.set_num_threads(1)
    plan,frozen=read(OUT/"protocol.json"),read(OUT/"freeze.json")
    assert sha(OUT/"protocol.json")==frozen["protocol_sha256"]
    for path,digest in plan["sources"].items():assert sha(ROOT/path)==digest,path
    for path in (checkpoint(seed,policy),world_path(seed,policy)):
        assert sha(path)==frozen["files"][str(path.relative_to(ROOT))]
    values={k:[] for k in (*base.FACT_FIELDS,"obs","actions","receipts","now","world_tick")}
    predictions=[];logit_rows=[]
    with mapping_scope(mapping_table(seed,"related")):
        model,h,world,initial=restore(seed,policy)
        initial_h=h.clone();evaluation_initial=world.state_dict()
        global_start_tick=world.tick+1;global_start_now=world.now+world.dt
        assert initial["acted"] and world.tick==66047 and h.shape==(32,192)
        for _ in range(WARM+STEPS):
            obs=world.observe();logits,h=model.step(obs,h,world.dt)
            actions=logits.argmax(-1)
            predictions.append(model.time_head(h).numpy().copy());logit_rows.append(logits.numpy().copy())
            values["obs"].append(obs.numpy().copy());values["actions"].append(actions.numpy().copy())
            values["now"].append(world.now);values["world_tick"].append(world.tick)
            for key,attr in base.FACT_FIELDS.items():values[key].append(getattr(world,attr).numpy().copy())
            world.act(actions);values["receipts"].append(world.last_result.numpy().copy())
        final=world.state_dict();event_hash=world.event_hash
    facts={k:np.stack(v) for k,v in values.items()}
    facts["actor_ids"]=np.asarray(world.actor_ids);facts["final_h"]=h.numpy().copy()
    report=base.summarize_facts(facts,seed,warm=WARM)
    path=stem(seed,policy);path.parent.mkdir(parents=True,exist_ok=True)
    bundle=path.with_name(path.name+"_state.pt")
    with bundle.open("xb") as f:
        torch.save(dict(initial_world=initial,evaluation_initial_world=evaluation_initial,initial_h=initial_h,
                        final_world=final,final_h=h.clone()),f)
    with path.with_suffix(".npz").open("xb") as f:np.savez_compressed(f,**facts)
    report.update(seed=seed,policy=policy,condition="long",mode="live",
        checkpoint_path=str(checkpoint(seed,policy).relative_to(ROOT)),checkpoint_sha256=sha(checkpoint(seed,policy)),
        source_native_world_path=str(world_path(seed,policy).relative_to(ROOT)),source_native_world_sha256=sha(world_path(seed,policy)),source_native_world_stage="final",
        initial_state_path=str(bundle.relative_to(ROOT)),initial_state_sha256=sha(bundle),
        global_start_tick=global_start_tick,global_start_now=global_start_now,
        world_config=dict(initial["spec"],regime=initial["regime"]),
        external_hash=event_hash,protocol_sha256=sha(OUT/"protocol.json"),freeze_sha256=sha(OUT/"freeze.json"),facts_sha256=sha(path.with_suffix(".npz")))
    write_new(path.with_suffix(".json"),report)
    prediction,logits=np.stack(predictions),np.stack(logit_rows)
    summary=time_statistics(facts,prediction,logits,report,WARM)
    side=path.with_name(path.name+"_time")
    with side.with_suffix(".npz").open("xb") as f:np.savez_compressed(f,time_prediction=prediction,native_logits=logits)
    summary.update(seed=seed,policy=policy,condition="long",mode="live",facts_path=str(path.with_suffix(".npz").relative_to(ROOT)),
        facts_sha256=report["facts_sha256"],checkpoint_path=report["checkpoint_path"],checkpoint_sha256=report["checkpoint_sha256"],
        sidecar_sha256=sha(side.with_suffix(".npz")),protocol_sha256=report["protocol_sha256"],freeze_sha256=report["freeze_sha256"])
    write_new(side.with_suffix(".json"),summary)


def launch(workers=3):
    logs=OUT/"logs";logs.mkdir(parents=True,exist_ok=True)
    def job(key):
        seed,policy=key
        with (logs/f"{seed}_{policy}.log").open("x",encoding="utf8") as log:
            p=subprocess.run([sys.executable,"-m","experiments.coverage_retention.live_eval","run","--seed",str(seed),"--policy",policy],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        return dict(seed=seed,policy=policy,exit_code=p.returncode)
    with ThreadPoolExecutor(max_workers=workers) as pool:runs=list(pool.map(job,[(s,p) for s in range(3) for p in POLICIES]))
    write_new(OUT/"workers.json",dict(runs=runs));assert all(r["exit_code"]==0 for r in runs),runs
    files={};sidecars={}
    shared=("event_kind","event_id","event_eligible","event_raw","event_version","event_duration","event_window","event_family","mature","mature_id","now","world_tick")
    for seed in range(3):
        rows=[read(stem(seed,p).with_suffix(".json")) for p in POLICIES]
        ids=[{(r["actor"],r["job"],r["rid"],r["family"],r["raw"]) for r in q["requests"]} for q in rows]
        assert ids[0]==ids[1]
        with np.load(stem(seed,"control").with_suffix(".npz")) as a,np.load(stem(seed,"withdraw").with_suffix(".npz")) as b:
            for key in shared:np.testing.assert_array_equal(a[key][WARM:],b[key][WARM:])
        for policy in POLICIES:
            path=stem(seed,policy).with_suffix(".json");files[str(path.relative_to(ROOT))]=sha(path)
            side=path.with_name(path.stem+"_time.json");sidecars[str(side.relative_to(ROOT))]=sha(side)
    write_new(OUT/"completion.json",dict(complete=True,runs=6,sidecars=6,mode="live",no_state_reset=True,
        matched_postwarm_external_requests=True,files=files,time_files=sidecars))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("declare","freeze","run","all"))
    p.add_argument("--seed",type=int,choices=range(3));p.add_argument("--policy",choices=POLICIES);p.add_argument("--workers",type=int,default=3)
    a=p.parse_args()
    if a.command=="declare":declare()
    elif a.command=="freeze":freeze()
    elif a.command=="run":run(a.seed,a.policy)
    else:launch(a.workers)
