"""Additional frozen probes for supported-domain and equal-horizon memory tests."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import torch
from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import atomic_json,state_digest
from experiments.credit.run import file_sha256
from experiments.loops.audit_run import verify_files
from experiments.lifelong import study


def specs():
    rows=[]
    for tick,families,conditions in ((131072,range(4),("short","long_both")),
            (163840,range(4),("short",)),(164864,(0,),("short","long_both")),
            (167936,(0,),("short","long_both"))):
        for family in families:
            for condition in conditions:rows.append(dict(tick=tick,family=family,condition=condition,steps=2048,purpose="aligned_memory"))
    return rows


def prepare(parent_protocol,folder,raw):
    parent_protocol,folder,raw=Path(parent_protocol).resolve(),Path(folder).resolve(),Path(raw).resolve()
    path=folder/"coverage-probe-supplement-v1.json"
    if path.exists():raise FileExistsError(path)
    parent=json.loads(parent_protocol.read_text(encoding="utf-8"));verify_files(parent["sources"])
    sources={str(Path(__file__).resolve()):file_sha256(__file__)}
    test=Path(__file__).with_name("test_supplement.py");sources[str(test)]=file_sha256(test)
    plan=dict(format="lifelong-coverage-supplement-v1",created_utc=datetime.now(timezone.utc).isoformat(),
        parent_protocol=str(parent_protocol),parent_protocol_sha256=file_sha256(parent_protocol),sources=sources,
        evaluation_specs=specs(),raw_root=str(raw),training_changed=False,
        reason="Before examining formal evaluation scores, a measurement audit found that narrow training lacked within-support short-condition longitudinal probes, and original temporal probes had different horizons. Add only frozen evaluation on already-declared snapshots; every new fork has2048ticks. Some original training branches had already completed. Original46forks/branch remain preserved.",
        interpretation="Combine these probes with existing2048tick source/final probes. Mainend131072, before-return163840, return+1024/+4096, final+8192 can be compared at equal horizon in short and long_both; task0 retention is primary and other families are controls.")
    assert len(plan["evaluation_specs"])==16
    atomic_json(path,plan)
    files={**parent["sources"],**sources,str(parent_protocol):file_sha256(parent_protocol),str(path):file_sha256(path)}
    jobs=[]
    for item in parent["branches"]:
        jobs.append(dict(id=item["id"],command=[sys.executable,"-m","experiments.lifelong.supplement","evaluate",
            "--protocol",str(path),"--branch-id",item["id"],"--checkpoint",str(Path(item["output"])/"latest.pt"),
            "--output",str(raw/"evaluation"/f"{item['id']}.json")],
            outputs=[str(raw/"evaluation"/f"{item['id']}.json")],log=str(raw/"logs"/f"{item['id']}.log")))
    for kind,entries in (("training",[]),("baseline",[]),("evaluation",jobs)):
        atomic_json(folder/f"coverage-{kind}-commands-v1.json",dict(cwd=str(study.ROOT),file_hashes=files,jobs=entries))
    print("Frozen480 additional matched-horizon forks; training and original1380forks unchanged.")


def read_plan(path,identifier):
    sha=file_sha256(path);plan=json.loads(Path(path).read_text(encoding="utf-8"))
    if plan["format"]!="lifelong-coverage-supplement-v1":raise ValueError("unknown supplement")
    verify_files(plan["sources"])
    if file_sha256(plan["parent_protocol"])!=plan["parent_protocol_sha256"]:raise ValueError("parent protocol changed")
    parent,item,parent_sha=study.read_plan(plan["parent_protocol"],identifier)
    return plan,parent,item,sha,parent_sha


def evaluate(path,identifier,checkpoint,output):
    from experiments.lifelong.run import NativeTrainer
    from experiments.lifelong.evaluate import evaluate as fork
    plan,parent,item,sha,parent_sha=read_plan(path,identifier)
    out,output=Path(item["output"]),Path(output)
    if Path(checkpoint).resolve()!=(out/"latest.pt").resolve() or output.exists():raise ValueError("wrong native source or existing supplement")
    done=study.completion(parent,item,parent_sha);states={};records=[]
    with _preserve_process_state():
        for tick in sorted({r["tick"] for r in plan["evaluation_specs"]}):
            cp=out/"checkpoints"/f"tick_{tick:06d}.pt";entry=done["checkpoints"][str(tick)]
            if file_sha256(cp)!=entry["sha256"]:raise ValueError("native supplement checkpoint changed")
            trainer=NativeTrainer.from_checkpoint(cp,device=parent["evaluation_device"]);study.validate(trainer,parent,item,tick)
            before=state_digest(trainer.semantic_state());normalized=study.normalized_hash(trainer,parent["training_device"])
            if normalized!=entry["semantic_hash"]:raise ValueError("native supplement semantic mismatch")
            states[str(tick)]=dict(checkpoint=str(cp),checkpoint_sha256=entry["sha256"],state_hash=before,
                training_device_normalized_state_hash=normalized,family_recency=trainer.report()["family_recency"])
            for spec in (r for r in plan["evaluation_specs"] if r["tick"]==tick):
                result=fork(trainer.model,trainer.world,trainer.h,family=spec["family"],condition=spec["condition"],
                    steps=spec["steps"],dt=parent["dt"],device=parent["evaluation_device"])
                result["probe"]=spec;records.append(result)
            if state_digest(trainer.semantic_state())!=before or file_sha256(cp)!=entry["sha256"]:raise AssertionError("supplement mutated source")
            del trainer
    if file_sha256(path)!=sha:raise ValueError("supplement protocol changed")
    read_plan(path,identifier)
    atomic_json(output,dict(format="lifelong-coverage-evaluation-v1",branch=item,parent_protocol_sha256=parent_sha,
        protocol_sha256=sha,source_unchanged=True,states=states,records=records))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("mode",choices=("prepare","evaluate"))
    for name in ("protocol","parent-protocol","output","raw","checkpoint"):parser.add_argument("--"+name,type=Path)
    parser.add_argument("--branch-id");args=parser.parse_args();torch.set_num_threads(1)
    if args.mode=="prepare":prepare(args.parent_protocol,args.output,args.raw)
    else:evaluate(args.protocol,args.branch_id,args.checkpoint,args.output)
