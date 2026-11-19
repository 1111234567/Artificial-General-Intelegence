"""Prospective fixed64 B-only updates after each frozen episodic endpoint.

The primary study is immutable. This separate continuation has no replay,
reinitialization, warm pass, reset, or h reconstruction.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import numpy as np
import torch

from experiments.action_split.train import deterministic, runtime, sha, write_new
from experiments.continuity.run import cpu_tree, state_digest
from experiments.coverage_retention import train as frozen
from experiments.episodic_review import train as primary
from experiments.remaining_time.model import RemainingTimeGRUAgent

OUT = ROOT / "data/episodic-review/retention-train-v1"
PLAN = OUT / "protocol.json"
FORMAT = "episodic-review-prospective64-B-only-retention-v1"
ARMS, SEEDS = primary.ARMS, primary.SEEDS
WARM, TRAIN_STEPS, BLOCK, BATCH, UPDATES = 0, 8192, 128, 32, 64
SNAPSHOTS = (0,64)
SOURCE_CURSOR, SOURCE_OPTIMIZER_STEP = 66048,512
assert_exact, optimizer_steps = frozen.assert_exact,frozen.optimizer_steps
HEAD_FORMAT, OPTIMIZER, STREAM_FIELDS = frozen.HEAD_FORMAT,frozen.OPTIMIZER,frozen.STREAM_FIELDS


def data_module():
    from experiments.episodic_review import retention_data
    return retention_data


def source_checkpoint(seed,arm):
    return primary.checkpoint_path(seed,arm,256)


def checkpoint_path(seed,arm,step=64):
    assert int(seed) in SEEDS and arm in ARMS and int(step) in SNAPSHOTS
    return OUT / f"seed_{int(seed):03d}" / arm / f"step{int(step):03d}.pt"


def states_path(seed):
    return data_module().states_path(seed)


def load_stream(seed):
    return data_module().load_stream(seed)


def load_source(seed,arm):
    cp = torch.load(source_checkpoint(seed,arm),map_location="cpu",weights_only=False)
    assert cp["format"] == primary.FORMAT and cp["replay_arm"] == arm
    assert cp["step"] == 256 and cp["stream_cursor"] == 32768
    assert cp["cumulative_stream_cursor"] == SOURCE_CURSOR
    assert optimizer_steps(cp["optimizer"]) == [SOURCE_OPTIMIZER_STEP]
    assert cp["hidden"].shape == (32,192)
    return cp


class Learner(frozen.Learner):
    """Use the frozen native action update with complete primary source state."""
    def __init__(self,model,stream,arm,*,source,block=BLOCK,updates=UPDATES,**kwargs):
        assert arm in ARMS
        adapted = deepcopy(source)
        adapted.update(work=deepcopy(source["cumulative_work"]),
            stream_cursor=source["cumulative_stream_cursor"])
        super().__init__(model,stream,"withdraw",source=adapted,block=block,updates=updates,**kwargs)
        self.primary_arm = arm
        self.source_checkpoint = str(source_checkpoint(self.seed,arm))
        self.primary_work = deepcopy(source["work"])
        self.source_replay_work = deepcopy(source.get("replay_work",source.get("source_replay_work",{})))

    def payload(self):
        cp = super().payload()
        cp.update(format=FORMAT,experiment_arm=self.primary_arm,primary_arm=self.primary_arm,
            actual_world_arm="withdraw",replay=False,primary_step=256,study_step=256+self.step,
            primary_work=deepcopy(self.primary_work),source_replay_work=deepcopy(self.source_replay_work),
            source_world_path=str(primary.states_path(self.seed)),source_world_stage="final",
            native_world_path=str(states_path(self.seed)),
            native_world_stage="initial" if self.step==0 else ("final" if self.step==self.updates else None),
            initialization="full primary256 model+Adam512+h+RNG copied; fixed64 B-only updates; no replay, warm, reset or h reconstruction",
            carried_hidden_semantics="source and suffix h are pre-update numerical live B state; carried unchanged across updates")
        return cp

    def save(self,name):
        assert self.folder is not None
        path = self.folder / (name+".pt")
        if path.exists():
            raise FileExistsError(path)
        cp = self.payload()
        torch.save(cp,path)
        write_new(path.with_suffix(".json"),dict(path=str(path),sha256=sha(path),
            step=self.step,study_step=256+self.step,seed=self.seed,arm="action",objective="action",
            experiment_arm=self.primary_arm,stream_cursor=self.cursor,work=self.work,
            source_work=self.source_work,cumulative_work=self.cumulative_work(),
            source_optimizer_step=self.source_optimizer_step,
            cumulative_optimizer_step=self.source_optimizer_step+self.step,
            optimizer_state_steps=optimizer_steps(self.optimizer),
            hidden_sha256=state_digest(cp["hidden"]),
            initial_hidden_sha256=self.source_state_hashes["hidden"],
            core_sha256=state_digest(cpu_tree(self.model.core.state_dict())),
            native_head_sha256=state_digest(cpu_tree(self.model.head.state_dict())),
            native_world_path=cp["native_world_path"],native_world_stage=cp["native_world_stage"],
            native_world_global_cursor=cp["native_world_global_cursor"],protocol_hash=self.protocol_hash))

    @classmethod
    def restore(cls,cp,stream,*,device="cpu",folder=None):
        if isinstance(cp,(str,Path)):
            cp = torch.load(cp,map_location="cpu",weights_only=False)
        assert cp["format"] == FORMAT
        model = RemainingTimeGRUAgent(**cp["model_config"])
        model.load_state_dict(cp["model"])
        source = deepcopy(cp)
        source.update(cumulative_work=deepcopy(cp["source_work"]),
            cumulative_stream_cursor=cp["source_stream_cursor"],
            protocol_hash=cp["source_protocol_hash"],work=deepcopy(cp["primary_work"]))
        obj = cls(model.to(device),stream,cp["primary_arm"],source=source,
            seed=cp["seed"],block=cp["settings"]["block"],updates=cp["settings"]["updates"],
            folder=folder,protocol_hash=cp["protocol_hash"],bindings=cp["bindings"],source_hashes=cp["source_hashes"])
        obj.source_optimizer_step = cp["source_optimizer_step"]
        obj.source_state_hashes = deepcopy(cp["source_state_hashes"])
        obj.cursor,obj.step = cp["stream_cursor"],cp["step"]
        obj.initial_core_hash,obj.initial_head_hash = cp["initial_core_hash"],cp["initial_head_hash"]
        obj.work = deepcopy(cp["work"])
        obj.restore_rng(cp)
        assert optimizer_steps(obj.optimizer) == [obj.source_optimizer_step+obj.step]
        return obj


def learner_from_source(seed,arm,stream,*,device="cuda",**kwargs):
    source = load_source(seed,arm)
    model = RemainingTimeGRUAgent(**source["model_config"])
    model.load_state_dict(source["model"])
    return Learner(model.to(device),stream,arm,source=source,seed=seed,**kwargs)


def sources():
    return list(dict.fromkeys([Path(__file__).resolve(),Path(__file__).with_name("retention_train_test.py"),
        Path(__file__).with_name("retention_audit.py"),*primary.sources(),*data_module().sources()]))


def declare(extra_sources=()):
    if PLAN.exists():
        raise FileExistsError(PLAN)
    deterministic(0)
    assert torch.cuda.is_available()
    assert Path(sys.executable).resolve() == (ROOT / ".venv/Scripts/python.exe").resolve()
    parent = primary.checked()
    freeze_path = primary.OUT / "globalfreeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    assert freeze["endpoint_count"]==9 and freeze["protocol_hash"]==parent["protocol_hash"]
    data = data_module()
    data_plan = data.checked()
    completion_path = data.OUT / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    assert completion["complete"] and completion["protocol_hash"]==data_plan["protocol_hash"]
    primary_audit_paths = [ROOT / "data/episodic-review/train-audit-v1/audit_plan.json",
        ROOT / "data/episodic-review/train-audit-v1/audit.json"]
    primary_audit = json.loads(primary_audit_paths[1].read_text(encoding="utf-8"))
    assert primary_audit["passed"] and primary_audit["training_protocol_hash"]==parent["protocol_hash"]
    assert primary_audit["freeze_hash"]==freeze["freeze_hash"]
    assert primary_audit["audit_plan_sha256"]==sha(primary_audit_paths[0])
    bindings,expected = {},{}
    for seed in SEEDS:
        stream = load_stream(seed)
        frozen.validate_stream(stream,WARM,BLOCK,UPDATES)
        expected[str(seed)] = frozen.summarize_rows(stream,0,TRAIN_STEPS)
        bindings[str(seed)] = {}
        for arm in ARMS:
            source = load_source(seed,arm)
            endpoint = freeze["endpoints"][f"{seed}:{arm}"]
            assert sha(source_checkpoint(seed,arm))==endpoint["sha256"]
            paths = [primary.PLAN,freeze_path,*primary_audit_paths,completion_path,
                data.PLAN,*map(Path,data.input_paths(seed)),
                source_checkpoint(seed,arm),source_checkpoint(seed,arm).with_suffix(".json"),
                primary.states_path(seed),source_checkpoint(seed,arm).parent / "completed.json"]
            bindings[str(seed)][arm] = {str(path):sha(path) for path in dict.fromkeys(paths)}
    plan = dict(format=FORMAT,created_utc=datetime.now(timezone.utc).isoformat(),
        seeds=list(SEEDS),arms=list(ARMS),snapshots=list(SNAPSHOTS),audit_steps=[1,2],
        runtime=runtime(),source_sha256={str(path):sha(path) for path in
            dict.fromkeys([*sources(),*map(Path,extra_sources)])},input_sha256=bindings,
        primary_protocol_hash=parent["protocol_hash"],primary_protocol_sha256=sha(primary.PLAN),
        primary_freeze_hash=freeze["freeze_hash"],primary_freeze_sha256=sha(freeze_path),
        primary_audit_sha256=sha(primary_audit_paths[1]),
        data_protocol_hash=data_plan.get("protocol_hash"),data_protocol_sha256=sha(data.PLAN),
        source_policy="each frozen primary256 exact full model+Adam512+h+RNG; original main study untouched",
        source_stream_cursor=SOURCE_CURSOR,source_optimizer_step=SOURCE_OPTIMIZER_STEP,
        optimizer=dict(kind="Adam",**OPTIMIZER,reset=False,source_step=512,final_step=576,
            clipping="separate policy/core+native head norm1 and detached-feature time-head norm1"),
        stream=dict(warm_ticks=0,training_ticks=8192,block=128,batch=32,updates=64,
            global_start=66048,global_stop=74240,actual_world="same native B-only continuation for all3 arms, no boundary changes"),
        implementation="frozen coverage_retention.Learner.update verbatim; no replay; carry every full-block pre-update numerical h",
        checkpoint="local000/064 = study256/320 full model+Adam+h+RNG+counters; actual001/002 before/after+preclip/clipped gradients",
        expected_live_rows=expected,
        no_adaptation="prospectively fixed64 updates for all9 before any new heldout scoring; no changes to primary256; all9 suffix endpoints and18 snapshots frozen together")
    assert plan["runtime"]==parent["runtime"]
    plan["optimizer"]["betas"] = list(OPTIMIZER["betas"])
    plan["protocol_hash"] = state_digest(plan)
    write_new(PLAN,plan)
    print(json.dumps(dict(event="declared",path=str(PLAN),protocol_hash=plan["protocol_hash"])),flush=True)
    return plan


def checked():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert state_digest({k:v for k,v in plan.items() if k!="protocol_hash"})==plan["protocol_hash"]
    assert sha(primary.PLAN)==plan["primary_protocol_sha256"]
    assert sha(primary.OUT / "globalfreeze.json")==plan["primary_freeze_sha256"]
    assert sha(data_module().PLAN)==plan["data_protocol_sha256"]
    for path,digest in plan["source_sha256"].items():
        assert sha(path)==digest,"frozen source changed: "+path
    for arms in plan["input_sha256"].values():
        for paths in arms.values():
            for path,digest in paths.items():
                assert sha(path)==digest,"frozen input changed: "+path
    return plan


def run_arm(seed,arm,plan,stream):
    folder = checkpoint_path(seed,arm).parent
    learner = learner_from_source(seed,arm,stream,folder=folder,protocol_hash=plan["protocol_hash"],
        bindings=plan["input_sha256"][str(seed)][arm],source_hashes=plan["source_sha256"])
    learner.save("step000")
    began = time.perf_counter()
    while learner.step<UPDATES:
        learner.update(save_audit=learner.step<2)
    learner.save("step064")
    assert learner.cursor==8192 and optimizer_steps(learner.optimizer)==[576]
    assert all(learner.work[key]==value for key,value in plan["expected_live_rows"][str(seed)].items())
    cp = checkpoint_path(seed,arm)
    result = dict(seed=seed,arm=arm,passed=True,updates=64,study_step=320,optimizer_step=576,
        final_checkpoint=str(cp),final_checkpoint_sha256=sha(cp),work=learner.work,
        source_work=learner.source_work,cumulative_work=learner.cumulative_work(),
        wall_seconds=time.perf_counter()-began,protocol_hash=plan["protocol_hash"])
    write_new(folder / "completed.json",result)
    print(json.dumps(dict(event="completed",seed=seed,arm=arm,study_step=320)),flush=True)
    del learner
    torch.cuda.empty_cache()
    return result


def run(seed):
    plan = checked()
    deterministic(938000+seed)
    assert runtime()==plan["runtime"]
    stream = load_stream(seed)
    results = [run_arm(seed,arm,plan,stream) for arm in ARMS]
    write_new(OUT / f"seed_{seed:03d}_completed.json",dict(seed=seed,runs=results))


def freeze_endpoints():
    plan = checked()
    endpoints = {}
    for seed in SEEDS:
        for arm in ARMS:
            cp = checkpoint_path(seed,arm)
            report = json.loads((cp.parent / "completed.json").read_text(encoding="utf-8"))
            assert report["passed"] and sha(cp)==report["final_checkpoint_sha256"]
            snapshots = {}
            for step in SNAPSHOTS:
                path = checkpoint_path(seed,arm,step)
                meta = path.with_suffix(".json")
                j = json.loads(meta.read_text(encoding="utf-8"))
                assert sha(path)==j["sha256"] and j["protocol_hash"]==plan["protocol_hash"]
                snapshots[f"step{step:03d}"] = dict(path=str(path),sha256=sha(path),metadata_path=str(meta),metadata_sha256=sha(meta))
            endpoints[f"{seed}:{arm}"] = dict(path=str(cp),sha256=sha(cp),seed=seed,arm=arm,snapshots=snapshots,
                work=report["work"],cumulative_work=report["cumulative_work"])
    result = dict(format="episodic-review-retention-all9-frozen-v1",created_utc=datetime.now(timezone.utc).isoformat(),
        protocol_hash=plan["protocol_hash"],primary_freeze_hash=plan["primary_freeze_hash"],
        endpoints=endpoints,endpoint_count=9,snapshot_count=18,study_step=320,heldout_scoring_started=False)
    result["freeze_hash"] = state_digest(result)
    write_new(OUT / "globalfreeze.json",result)
    print(json.dumps(dict(event="globally_frozen",path=str(OUT / "globalfreeze.json"),freeze_hash=result["freeze_hash"])),flush=True)
    return result


def all_seeds():
    checked()
    pending,active,results = list(SEEDS),[],[]
    while pending or active:
        while pending and len(active)<2:
            seed = pending.pop(0)
            log = (OUT / f"seed_{seed:03d}.log").open("x",encoding="utf-8")
            process = subprocess.Popen([sys.executable,"-u","-m","experiments.episodic_review.retention_train","run","--seed",str(seed)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy(),creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            active.append((seed,process,log))
        for seed,process,log in list(active):
            status = process.poll()
            if status is not None:
                active.remove((seed,process,log)); log.close()
                results.append(dict(seed=seed,exit_code=status))
        if active:
            time.sleep(.5)
    write_new(OUT / "exit_codes.json",results)
    if any(row["exit_code"] for row in results):
        raise RuntimeError(results)
    return freeze_endpoints()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("declare","run","all","freeze"))
    parser.add_argument("--seed",type=int,choices=SEEDS)
    parser.add_argument("--extra-source",action="append",default=[])
    args = parser.parse_args()
    if args.command=="declare":
        declare(args.extra_source)
    elif args.command=="all":
        all_seeds()
    elif args.command=="freeze":
        freeze_endpoints()
    else:
        if args.seed is None:
            parser.error("run requires --seed")
        run(args.seed)


if __name__=="__main__":
    main()
