"""Prospective fresh/live evaluation of replay and its no-review suffix."""
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

from experiments.remaining_time import evaluate as previous
from experiments.remaining_time.model import RemainingTimeGRUAgent
from experiments.action_split.data import sha,read,write_new
from experiments.coverage_retention import error_signature as signatures
from experiments.experience_transfer.train_transfer import mapping_scope,mapping_table,SEEDS
from experiments.lifelong.world import ContextWorld

ROOT=previous.ROOT
BASE=ROOT/"data/episodic-review"
FRESH=BASE/"eval-v1"
LIVE=BASE/"live-v1"
ARMS=("none","replay_A","replay_B")
TRAINED=tuple(a+str(s) for s in (256,320) for a in ARMS)
POLICIES=("baseline",*TRAINED)
CONDITIONS={"long":(8.,16.),"longer":(16.,32.)}
WARM,STEPS=512,4096


def parts(policy):
    if policy=="baseline":return None,None
    arm,stage=policy[:-3],int(policy[-3:])
    assert arm in ARMS and stage in (256,320)
    return arm,stage


def checkpoint(seed,policy):
    from experiments.episodic_review import train,retention_train
    if policy=="baseline":return train.source_checkpoint(seed)
    arm,stage=parts(policy)
    return train.checkpoint_path(seed,arm,256) if stage==256 else retention_train.checkpoint_path(seed,arm,64)


def world_path(seed,policy):
    from experiments.episodic_review import train,retention_data
    _,stage=parts(policy)
    return train.states_path(seed) if stage==256 else retention_data.states_path(seed)


def config(seed,condition):
    spec=SEEDS[seed]
    return dict(seed=856000+seed,batch_size=32,n_jobs=2,active_jobs=2,
        actor_ids=tuple(range(101000,101032)),family_allowed=(spec["history_cue"],spec["B_cue"]),
        dt=.25,duration_range=CONDITIONS[condition],gap_range=(16.,32.))


def folder(live=False):return LIVE if live else FRESH
def stem(seed,policy,condition,live=False):return folder(live)/f"seed_{seed:03d}"/f"{policy}_{condition}"


def source_files():
    return [Path(__file__),Path(__file__).with_name("eval_test.py"),
        Path(__file__).with_name("eval_audit.py"),
        Path(__file__).with_name("analysis-plan.md"),Path(__file__).with_name("retention-plan.md"),
        ROOT/"experiments/remaining_time/evaluate.py",ROOT/"experiments/remaining_time/model.py",
        ROOT/"experiments/action_split/evaluate.py",ROOT/"experiments/coverage_retention/error_signature.py",
        ROOT/"experiments/lifelong/world.py",ROOT/"experiments/continuity/world.py"]


def declare(live=False):
    out=folder(live)
    write_new(out/"protocol.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),mode="live" if live else "fresh",
        sources={p.relative_to(ROOT).as_posix():sha(p) for p in source_files()},
        seeds=list(range(3)),policies=TRAINED if live else POLICIES,
        conditions={"long":[8.,16.]} if live else CONDITIONS,warm=WARM,test=STEPS,dt=.25,
        primary="Fixedstudy256: replay_A versus equallymasked replay_B, withnone andsourcebaseline references. Nativewholewindow strictAconflict/Bconflict/shared plusidle/timingfailures.",
        retention="Study320 means64furtherB-onlyupdates withno replay, fixedbeforeanyheldoutscore. Report256and320, neverpickbest.",
        initialization=("Eachstage restores its actualB-onlyworldfinalandownh; futurefamilyABonly, no h/clock/receipt/request/RNG reset; evaluate18continuousreturns." if live else
            "Freshworldseed856000+seed actors101000..101031, all7policies samecases percondition, h0 atindependentorigin thencontinuous;42runs."),
        endpoint_worlds="study256 usesoldwithdrawfinaltick66047, study320 usessuffixfinal74239; livestageshavedifferentcases andarenotpairedacrossstages.",
        action="Originalnative11argmaxonly. Auxiliarytimeprediction/logits archivedandnevercontrolaction.",
        denominator="Everypostwarm actualNEW/REPLACE eligiblemature request. Include noBEGIN/no-window/noFINISH. Inheritedpendingrequests separatelyvalidated, notnewtestdenominator.",
        rule_signature="PREDECLARED read-only classification of firstactualFINISH inlegalwindow: currentrule, literalotherruleforconflicts, otherpayload,wrongversion,noattempt. Usesalready-audited priorclassifier; no outputsselectdata.",
        classifier_implementation=dict(path="experiments/coverage_retention/error_signature.py",sha256=sha(signatures.__file__),
            parent_parser_amendment_sha256=sha(signatures.OUT/"parser_amendment_001.json")),
        limits="Validtimerowsdependonactualhistory; legacyoverdue_beyond_training isonly remaining<-.25 andwaspartlycoveredbymidpointteacher. Fixed2durations, no extrapolationclaimfromreplay."))


def freeze(live=False):
    from experiments.episodic_review import train,retention_train
    first,second=train.OUT/"globalfreeze.json",retention_train.OUT/"globalfreeze.json"
    assert first.exists() and second.exists()
    out=folder(live);plan=read(out/"protocol.json")
    for name,digest in plan["sources"].items():assert sha(ROOT/name)==digest,name
    policies=TRAINED if live else POLICIES
    files={str(checkpoint(s,p).relative_to(ROOT)):sha(checkpoint(s,p)) for s in range(3) for p in policies}
    if live:
        for s in range(3):
            for p in policies:files[str(world_path(s,p).relative_to(ROOT))]=sha(world_path(s,p))
    write_new(out/"freeze.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(out/"protocol.json"),primary_freeze_sha256=sha(first),suffix_freeze_sha256=sha(second),
        checkpoints={str(checkpoint(s,p).relative_to(ROOT)):sha(checkpoint(s,p)) for s in range(3) for p in policies},files=files))


def checked(seed,policy,live):
    out=folder(live);plan,frozen=read(out/"protocol.json"),read(out/"freeze.json")
    assert sha(out/"protocol.json")==frozen["protocol_sha256"]
    for name,digest in plan["sources"].items():assert sha(ROOT/name)==digest,name
    for p in ((checkpoint(seed,policy),world_path(seed,policy)) if live else (checkpoint(seed,policy),)):
        assert sha(p)==frozen["files"][str(p.relative_to(ROOT))]


def classify(seed,policy,condition,live=False):
    path=stem(seed,policy,condition,live)
    report=read(path.with_suffix(".json"))
    spec=dict(seed=seed,mode="live" if live else "fresh",policy=policy,stem=str(path.relative_to(ROOT)),
        checkpoint_path=report["checkpoint_path"],checkpoint_sha256=report["checkpoint_sha256"],facts_sha256=report["facts_sha256"])
    result=signatures.classify_run(spec)
    result["parent_implementation_provenance"]=dict(protocol_sha256=result["protocol_sha256"],parser_amendment_sha256=result.pop("parser_amendment_sha256"))
    result.update(format="episodic-review-prospective-rule-signature-v1",posthoc=False,condition=condition,
        protocol_sha256=sha(folder(live)/"protocol.json"))
    write_new(path.with_name(path.name+"_rule.json"),result)


def run_fresh(seed,policy,condition):
    previous.OUT=FRESH;previous.POLICIES=POLICIES;previous.checkpoint=checkpoint;previous.config=config
    previous.base.CONDITIONS=CONDITIONS
    previous.run(seed,policy,condition)
    classify(seed,policy,condition)


@torch.inference_mode()
def run_live(seed,policy):
    assert policy in TRAINED
    checked(seed,policy,True);torch.set_num_threads(1)
    cp=torch.load(checkpoint(seed,policy),map_location="cpu",weights_only=False)
    worlds=torch.load(world_path(seed,policy),map_location="cpu",weights_only=False)
    initial=deepcopy(worlds["final"])
    model=RemainingTimeGRUAgent(**cp["model_config"])
    model.load_state_dict(cp["model"]);model.eval().requires_grad_(False)
    h=cp["hidden"].clone();initial_h=h.clone()
    values={k:[] for k in (*previous.base.FACT_FIELDS,"obs","actions","receipts","now","world_tick")}
    predictions=[];logit_rows=[]
    with mapping_scope(mapping_table(seed,"related")):
        world=ContextWorld.from_state_dict(initial,device="cpu")
        _,stage=parts(policy)
        assert world.tick==(66047 if stage==256 else 74239) and world._acted
        world.set_family_policy((SEEDS[seed]["history_cue"],SEEDS[seed]["B_cue"]))
        evaluation_initial=world.state_dict();global_start_tick=world.tick+1
        for _ in range(WARM+STEPS):
            obs=world.observe();logits,h=model.step(obs,h,world.dt)
            actions=logits.argmax(-1)
            predictions.append(model.time_head(h).numpy().copy());logit_rows.append(logits.numpy().copy())
            values["obs"].append(obs.numpy().copy());values["actions"].append(actions.numpy().copy())
            values["now"].append(world.now);values["world_tick"].append(world.tick)
            for key,attr in previous.base.FACT_FIELDS.items():values[key].append(getattr(world,attr).numpy().copy())
            world.act(actions);values["receipts"].append(world.last_result.numpy().copy())
        final=world.state_dict();event_hash=world.event_hash
    facts={k:np.stack(v) for k,v in values.items()}
    facts["actor_ids"]=np.asarray(world.actor_ids);facts["final_h"]=h.numpy().copy()
    report=previous.base.summarize_facts(facts,seed,warm=WARM)
    path=stem(seed,policy,"long",True);path.parent.mkdir(parents=True,exist_ok=True)
    bundle=path.with_name(path.name+"_state.pt")
    with bundle.open("xb") as f:torch.save(dict(initial_world=initial,evaluation_initial_world=evaluation_initial,
        initial_h=initial_h,final_world=final,final_h=h.clone()),f)
    with path.with_suffix(".npz").open("xb") as f:np.savez_compressed(f,**facts)
    report.update(seed=seed,policy=policy,condition="long",mode="live",study_step=stage,
        checkpoint_path=str(checkpoint(seed,policy).relative_to(ROOT)),checkpoint_sha256=sha(checkpoint(seed,policy)),
        source_native_world_path=str(world_path(seed,policy).relative_to(ROOT)),source_native_world_sha256=sha(world_path(seed,policy)),source_native_world_stage="final",
        initial_state_path=str(bundle.relative_to(ROOT)),initial_state_sha256=sha(bundle),global_start_tick=global_start_tick,
        global_start_now=global_start_tick*.25,world_config=dict(initial["spec"],regime=initial["regime"]),external_hash=event_hash,
        protocol_sha256=sha(LIVE/"protocol.json"),freeze_sha256=sha(LIVE/"freeze.json"),facts_sha256=sha(path.with_suffix(".npz")))
    write_new(path.with_suffix(".json"),report)
    pred,logits=np.stack(predictions),np.stack(logit_rows)
    aux=previous.time_statistics(facts,pred,logits,report,WARM)
    side=path.with_name(path.name+"_time")
    with side.with_suffix(".npz").open("xb") as f:np.savez_compressed(f,time_prediction=pred,native_logits=logits)
    aux.update(seed=seed,policy=policy,condition="long",mode="live",facts_path=str(path.with_suffix(".npz").relative_to(ROOT)),
        facts_sha256=report["facts_sha256"],checkpoint_path=report["checkpoint_path"],checkpoint_sha256=report["checkpoint_sha256"],
        sidecar_sha256=sha(side.with_suffix(".npz")),protocol_sha256=report["protocol_sha256"],freeze_sha256=report["freeze_sha256"])
    write_new(side.with_suffix(".json"),aux)
    classify(seed,policy,"long",True)


def launch(live=False,workers=6):
    out=folder(live);policies=TRAINED if live else POLICIES;conditions=("long",) if live else tuple(CONDITIONS)
    logs=out/"logs";logs.mkdir(parents=True,exist_ok=True)
    def job(key):
        s,p,c=key
        command=[sys.executable,"-m","experiments.episodic_review.evaluate","run","--seed",str(s),"--policy",p,"--condition",c]
        if live:command.append("--live")
        with (logs/f"{s}_{p}_{c}.log").open("x",encoding="utf8") as f:
            completed=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        return dict(seed=s,policy=p,condition=c,exit_code=completed.returncode)
    with ThreadPoolExecutor(max_workers=workers) as pool:runs=list(pool.map(job,[(s,p,c) for s in range(3) for p in policies for c in conditions]))
    write_new(out/"workers.json",dict(runs=runs));assert all(r["exit_code"]==0 for r in runs),runs
    files={};time_files={};rule_files={}
    for seed in range(3):
        for condition in conditions:
            grouping=([p for p in policies if p.endswith(str(stage))] for stage in (256,320)) if live else (policies,)
            for group in grouping:
                rows=[read(stem(seed,p,condition,live).with_suffix(".json")) for p in group]
                assert len({r["external_hash"] for r in rows})==1
                identities=[{(r["actor"],r["job"],r["rid"],r["family"],r["raw"]) for r in x["requests"]} for x in rows]
                assert all(x==identities[0] for x in identities)
            for policy in policies:
                path=stem(seed,policy,condition,live).with_suffix(".json")
                files[str(path.relative_to(ROOT))]=sha(path)
                for suffix,target in (("_time",time_files),("_rule",rule_files)):
                    side=path.with_name(path.stem+suffix+".json");target[str(side.relative_to(ROOT))]=sha(side)
    write_new(out/"completion.json",dict(complete=True,mode="live" if live else "fresh",runs=len(runs),sidecars=len(runs),
        rules=len(runs),same_external_requests_within_stage=True,files=files,time_files=time_files,rule_files=rule_files))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("declare","freeze","run","all"))
    p.add_argument("--live",action="store_true");p.add_argument("--seed",type=int,choices=range(3))
    p.add_argument("--policy",choices=POLICIES);p.add_argument("--condition",choices=CONDITIONS);p.add_argument("--workers",type=int,default=6)
    a=p.parse_args()
    if a.command=="declare":declare(a.live)
    elif a.command=="freeze":freeze(a.live)
    elif a.command=="run":
        if a.live:run_live(a.seed,a.policy)
        else:run_fresh(a.seed,a.policy,a.condition)
    else:launch(a.live,a.workers)
