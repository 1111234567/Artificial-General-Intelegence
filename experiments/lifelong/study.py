"""Finite native task interleaving, review, and long-time coverage study."""
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
from experiments.lifelong.schedule import CHECKPOINT_TICKS

ROOT=Path(__file__).resolve().parents[2]


def evaluation_specs():
    rows=[]
    def add(tick,families,conditions,steps,purpose):
        for family in families:
            for condition in conditions:rows.append(dict(tick=tick,family=family,condition=condition,steps=steps,purpose=purpose))
    add(1024,range(4),["short"],1024,"early_main")
    for tick in (32768,65536,98304,131072):add(tick,range(4),["long_both"],1024,"main_retention")
    add(163840,range(4),["long_both"],2048,"before_return")
    for tick in (164864,167936):add(tick,[0],["long_both"],1024,"return_learning")
    add(172032,range(4),["short","long_wait","long_idle","long_both","far"],2048,"final")
    return rows


def prepare(folder,raw):
    folder,raw=Path(folder).resolve(),Path(raw).resolve();path=folder/"study-protocol-v1.json"
    if path.exists():raise FileExistsError(path)
    paths=list((ROOT/"experiments/lifelong").glob("*.py"))
    # Analysis/plot scripts may evolve independently of frozen experimental execution.
    paths=[p for p in paths if p.name not in {"report.py","test_report.py","plot_results.py"}]
    paths += [ROOT/"experiments"/name for name in ("continuity/model.py","continuity/world.py","continuity/run.py","continuity/evaluate.py","integration/diagnose.py","integration/run.py","factors/structure.py","rewiring/plasticity.py","credit/run.py","credit/local.py","credit/batch.py","credit/windows_runner.py","loops/audit_run.py","loops/structure.py","loops/interventions.py")]
    files={str(p):file_sha256(p) for p in paths}
    variants=[dict(architecture="fixed_shrink",schedule=schedule,learning=learning,coverage=coverage)
        for schedule in ("blocked","mixed") for learning in ("recent_repeat","history_replay") for coverage in ("narrow","wide")]
    variants += [dict(architecture=architecture,schedule="mixed",learning="history_replay",coverage="wide") for architecture in ("grow_shrink","open_shrink")]
    branches=[]
    for seed in (0,1,2):
        for variant in variants:
            label="__".join(variant[k] for k in ("architecture","schedule","learning","coverage"))
            branches.append(dict(id=f"seed_{seed:03d}__{label}",seed=seed,**variant,output=str(raw/"training"/f"seed_{seed:03d}"/label)))
    attachment=Path(r"C:\Users\Jerry\.codex\attachments\8dccaaef-d964-42c2-b355-70d6ef60c6f3\已粘贴的文本.txt")
    plan=dict(format="lifelong-study-protocol-v1",created_utc=datetime.now(timezone.utc).isoformat(),sources=files,
        branches=branches,variants=variants,seeds=[0,1,2],target_ticks=172032,main_ticks=131072,withdrawal_ticks=32768,
        return_ticks=8192,chunk_ticks=128,batch_size=32,width=128,initial_fan_in=32,dt=.25,
        training_device="cuda",evaluation_device="cpu",checkpoint_ticks=list(CHECKPOINT_TICKS),
        evaluation_specs=evaluation_specs(),replay_capacity=32,replay_block_ticks=256,replay_warmup_ticks=128,
        attachment=dict(path=str(attachment),sha256=file_sha256(attachment)),
        family_mappings=[[0,1,2,3],[1,0,2,3],[0,1,3,2],[1,2,3,0]],
        timing_coverage=dict(narrow=dict(duration=[1.,4.],gap=[.5,2.]),wide=dict(duration=[1.,16.],gap=[.5,32.])),
        primary="Frozen-weight function retention before the withheld family receives any new live feedback, followed by matched return learning; strict, ordinary and idle errors all retained.",
        learning="Live128 and auxiliary suffix128 are computed before one shared Adam update; equal normalized gradient mix. Recent reconstruction and historical replay both use128 detached warmup, except matching startup fallback. Admission is nonoverlapping256, after update; retention and sampling RNGs separate.",
        native="All policies operate from random birth. Same initial sparse W/function per seed; fixed/grow/open rules differ from first update, no pretrained graft. Initial fan-in32 scaled by sqrt(128/32); hidden zeros only at birth.",
        limits=["Four observable contexts change payload rules with shared/conflicting cases; the control goals still share the same request protocol, not unrelated real-world domains.",
            "Main schedule budgets match live ticks, not exact per-family request counts; actual commands and supervised positions are measured.",
            "Wider time support changes event density as well as coverage under fixed tick budgets.",
            "History replay processes original own-trajectory observations and labels; it does not recreate a fresh closed-loop world or overwrite the live hidden state.",
            "All branches have the same dense tensor allocation and one Adam step per live128; auxiliary work is matched, not free.",
            "Open/grow capacity trajectories can diverge; native structural contrasts test complete update policies, not pure placement at identical live edge counts.",
            "Native growth retains the previously specified shadow and shared-age Adam handoff; no extra age fix is silently added.",
            "Recurring task rules and interval control do not yet establish hierarchical composition, general intelligence or unrestricted lifelong learning."])
    assert len(branches)==30 and len(plan["evaluation_specs"])==46
    atomic_json(path,plan);frozen={**files,str(path):file_sha256(path)}
    training=[];evaluation=[]
    for item in branches:
        out=Path(item["output"]);tail=["--protocol",str(path),"--branch-id",item["id"]]
        prefix=[sys.executable,"-m","experiments.lifelong.study"]
        training.append(dict(id=item["id"],command=prefix+["train"]+tail,
            outputs=[str(out/"latest.pt"),str(out/"launch-complete.json")],log=str(raw/"logs"/f"train_{item['id']}.log")))
        evaluation.append(dict(id=item["id"],command=prefix+["evaluate"]+tail+["--checkpoint",str(out/"latest.pt"),"--output",str(raw/"evaluation"/f"{item['id']}.json")],
            outputs=[str(raw/"evaluation"/f"{item['id']}.json")],log=str(raw/"logs"/f"evaluate_{item['id']}.log")))
    for name,jobs in (("training",training),("evaluation",evaluation),("baseline",[])):
        atomic_json(folder/f"{name}-commands-v1.json",dict(cwd=str(ROOT),file_hashes=frozen,jobs=jobs))
    print("Frozen30 native lifetimes,390 full checkpoints,1380 frozen forks.")


def read_plan(path,identifier):
    sha=file_sha256(path);plan=json.loads(Path(path).read_text(encoding="utf-8"))
    if plan["format"]!="lifelong-study-protocol-v1":raise ValueError("unknown protocol")
    verify_files(plan["sources"])
    return plan,next(x for x in plan["branches"] if x["id"]==identifier),sha


def normalized_hash(trainer,device):
    value=trainer.semantic_state();value["config"]["device"]=device
    return state_digest(value)


def validate(trainer,plan,item,tick=None):
    for key in ("seed","architecture","schedule","learning","coverage"):
        if getattr(trainer.config,key)!=item[key]:raise ValueError(f"foreign native branch {key}")
    for key in ("width","batch_size","chunk_ticks","initial_fan_in","main_ticks","withdrawal_ticks","return_ticks"):
        if getattr(trainer.config,key)!=plan[key]:raise ValueError(f"native budget mismatch {key}")
    if trainer.hidden_resets or trainer.ticks%plan["chunk_ticks"] or not 0<=trainer.ticks<=plan["target_ticks"] or (tick is not None and trainer.ticks!=tick):raise ValueError("invalid native progress")


def verify_end(path,identifier,sha):
    if file_sha256(path)!=sha:raise ValueError("protocol changed")
    read_plan(path,identifier)


def completion(plan,item,sha):
    out=Path(item["output"]);value=json.loads((out/"launch-complete.json").read_text())
    if (value.get("format")!="lifelong-launch-complete-v1" or value.get("branch")!=item or value.get("protocol_sha256")!=sha
            or value.get("stage_ticks")!=plan["target_ticks"] or value.get("target")!=plan["target_ticks"]
            or value.get("checkpoint_sha256")!=file_sha256(out/"latest.pt")):
        raise ValueError("invalid native completion")
    if set(value["checkpoints"])!={str(t) for t in plan["checkpoint_ticks"]}:raise ValueError("incomplete native snapshot set")
    return value


def train(path,identifier):
    from experiments.lifelong.run import NativeTrainer,main as train_main
    plan,item,sha=read_plan(path,identifier);out=Path(item["output"])
    if (out/"launch-complete.json").exists():
        done=completion(plan,item,sha)
        for tick,entry in done["checkpoints"].items():
            cp=out/"checkpoints"/f"tick_{int(tick):06d}.pt"
            if file_sha256(cp)!=entry["sha256"]:raise ValueError("changed native snapshot")
        with _preserve_process_state():
            final=NativeTrainer.from_checkpoint(out/"latest.pt",device=plan["evaluation_device"])
            validate(final,plan,item,plan["target_ticks"])
            if normalized_hash(final,plan["training_device"])!=done["checkpoints"][str(plan["target_ticks"])]["semantic_hash"]:
                raise ValueError("completed latest state differs from final snapshot")
        verify_end(path,identifier,sha);return
    start=out.with_name(out.name+".launch-start.json");binding=dict(format="lifelong-launch-start-v1",branch=item,protocol_sha256=sha)
    if start.exists():
        if json.loads(start.read_text())!=binding:raise ValueError("foreign native start")
    elif (out/"latest.pt").exists():raise ValueError("unbound native checkpoint")
    else:atomic_json(start,binding)
    command=["--output",str(out),"--device",plan["training_device"]]
    if (out/"latest.pt").exists():
        with _preserve_process_state():
            partial=NativeTrainer.from_checkpoint(out/"latest.pt",device=plan["evaluation_device"]);validate(partial,plan,item)
        del partial;command += ["--resume",str(out/"latest.pt")]
    else:
        for key in ("seed","architecture","schedule","learning","coverage"):command += ["--"+key,str(item[key])]
    train_main(command)
    entries={}
    with _preserve_process_state():
        for tick in plan["checkpoint_ticks"]:
            cp=out/"checkpoints"/f"tick_{tick:06d}.pt"
            trainer=NativeTrainer.from_checkpoint(cp,device=plan["evaluation_device"]);validate(trainer,plan,item,tick)
            entries[str(tick)]=dict(path=str(cp),sha256=file_sha256(cp),semantic_hash=normalized_hash(trainer,plan["training_device"]))
            del trainer
    verify_end(path,identifier,sha)
    atomic_json(out/"launch-complete.json",dict(format="lifelong-launch-complete-v1",branch=item,protocol_sha256=sha,
        target=plan["target_ticks"],stage_ticks=plan["target_ticks"],checkpoint_sha256=file_sha256(out/"latest.pt"),checkpoints=entries,hidden_resets=0))


def evaluate(path,identifier,checkpoint,output):
    from experiments.lifelong.run import NativeTrainer
    from experiments.lifelong.evaluate import evaluate as fork
    plan,item,sha=read_plan(path,identifier);out=Path(item["output"]);output=Path(output)
    if Path(checkpoint).resolve()!=(out/"latest.pt").resolve() or output.exists():raise ValueError("wrong ready file or existing output")
    done=completion(plan,item,sha);records=[];states={}
    with _preserve_process_state():
        for tick in sorted({s["tick"] for s in plan["evaluation_specs"]}):
            entry=done["checkpoints"][str(tick)];cp=out/"checkpoints"/f"tick_{tick:06d}.pt"
            if file_sha256(cp)!=entry["sha256"]:raise ValueError("native snapshot bytes changed")
            trainer=NativeTrainer.from_checkpoint(cp,device=plan["evaluation_device"]);validate(trainer,plan,item,tick)
            before=state_digest(trainer.semantic_state());normalized=normalized_hash(trainer,plan["training_device"])
            if normalized!=entry["semantic_hash"]:raise ValueError("native snapshot semantic changed")
            states[str(tick)]=dict(checkpoint=str(cp),checkpoint_sha256=entry["sha256"],state_hash=before,training_device_normalized_state_hash=normalized,family_recency=trainer.report()["family_recency"])
            for spec in (s for s in plan["evaluation_specs"] if s["tick"]==tick):
                record=fork(trainer.model,trainer.world,trainer.h,family=spec["family"],condition=spec["condition"],steps=spec["steps"],dt=plan["dt"],device=plan["evaluation_device"])
                record["probe"]=spec;records.append(record)
            if state_digest(trainer.semantic_state())!=before:raise AssertionError("frozen native evaluation changed training state")
            if file_sha256(cp)!=entry["sha256"]:raise ValueError("native snapshot changed during evaluation")
            del trainer
    verify_end(path,identifier,sha)
    atomic_json(output,dict(format="lifelong-evaluation-v1",branch=item,protocol_sha256=sha,source_unchanged=True,states=states,records=records))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("mode",choices=("prepare","train","evaluate"))
    for name in ("protocol","output","raw","checkpoint"):parser.add_argument("--"+name,type=Path)
    parser.add_argument("--branch-id");args=parser.parse_args();torch.set_num_threads(1)
    if args.mode=="prepare":prepare(args.output,args.raw)
    elif args.mode=="train":train(args.protocol,args.branch_id)
    else:evaluate(args.protocol,args.branch_id,args.checkpoint,args.output)
