"""Frozen shrink/support/turnover x idle-penalty factorial continuation study."""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys
import torch
from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import atomic_json, state_digest
from experiments.credit.run import file_sha256
from experiments.integration.diagnose import diagnose_fork
from experiments.loops.audit_run import verify_files

ROOT = Path(__file__).resolve().parents[2]
MODES = ("dense", "dense_shrink", "fixed", "fixed_shrink", "retire", "grow")
LAMBDAS = (("zero", 0.), ("small", .1), ("large", 1.))
CONDITIONS = ["iid", "long_wait", "longer", "sparse", "long_sparse"]


def prepare(folder, raw):
    folder, raw = Path(folder).resolve(), Path(raw).resolve()
    protocol = folder / "study-protocol-v1.json"
    if protocol.exists(): raise FileExistsError(protocol)
    groups = {
        "factors": ["__init__.py", "structure.py", "run.py", "test_structure.py", "test_run.py", "study.py", "test_study.py", "queue.py", "test_queue.py"],
        "rewiring": ["plasticity.py", "run.py", "test_plasticity.py", "test_run.py"],
        "continuity": ["model.py", "world.py", "run.py", "evaluate.py"],
        "credit": ["run.py", "local.py"], "integration": ["run.py", "diagnose.py"],
        "loops": ["audit_run.py", "structure.py", "interventions.py"],
    }
    paths = [ROOT/"experiments"/group/name for group,names in groups.items() for name in names]
    sources = {str(path):file_sha256(path) for path in paths}
    origins, branches = [], []
    for seed in (0,1,2):
        old = ROOT/f"data/rewiring/main-v1/training/seed_{seed:03d}/retire_high"
        origin = {"id":f"seed_{seed:03d}", "seed":seed, "checkpoint":str(old/"checkpoint_B.pt"),
                  "checkpoint_sha256":file_sha256(old/"checkpoint_B.pt"),
                  "historical_A2":str(old/"checkpoint_A2.pt"), "historical_A2_sha256":file_sha256(old/"checkpoint_A2.pt")}
        origins.append(origin)
        for mode in MODES:
            for label, strength in LAMBDAS:
                name = f"{mode}_idle_{label}"
                branches.append({"id":f"seed_{seed:03d}_{name}","seed":seed,"mode":mode,"idle_label":label,
                    "idle_lambda":strength,"shrink_step":2e-4,"source":origin["checkpoint"],
                    "source_sha256":origin["checkpoint_sha256"],"origin_id":origin["id"],
                    "output":str(raw/"training"/f"seed_{seed:03d}"/name)})
    attachment = Path(r"C:\Users\Jerry\.codex\attachments\327b2baf-5605-4c59-867c-8ddabb6f3a3d\已粘贴的文本.txt")
    plan = {"format":"factors-study-protocol-v1","created_utc":datetime.now(timezone.utc).isoformat(),
        "sources":sources,"origins":origins,"branches":branches,"modes":list(MODES),
        "idle_lambdas":[v for _,v in LAMBDAS],"target_ticks":65536,"chunk_ticks":64,
        "source_stage_ticks":65536,"training_device":"cuda","evaluation_device":"cpu",
        "conditions":CONDITIONS,"evaluation_steps":2048,"dt":.25,
        "return_regime":{"duration":[1.,4.],"gap":[.5,2.],"name":"rewiring-A2"},
        "review_attachment":{"path":str(attachment),"sha256":file_sha256(attachment)},
        "objective":"Separate soft shrinkage, allowed sparse support, permanent retirement and regrowth; test idle-action penalty tradeoffs.",
        "loss":"Original weighted CE (WAIT=.1) plus idle_lambda times mean negative log p(WAIT) over enabled inactive slots captured after observe and before act; zero term for no idle opportunities.",
        "comparisons":[["dense_shrink","dense"],["fixed_shrink","fixed"],["fixed","dense"],
                       ["fixed_shrink","dense_shrink"],["retire","fixed_shrink"],["grow","retire"]],
        "limits":["All branches inherit one sparse B history per seed; this is continuation from that history, not a from-scratch architectural comparison.",
            "Fixed allowed support remains eligible even if its current weight equals zero; actual nonzero support differs from allowed support.",
            "Regrowth versus retirement also changes active capacity, birth amplitudes and optimizer handoff; this does not isolate a pure turnover-only effect.",
            "Idle mask is privileged training feedback like the existing teacher; it is not a new model input or an evaluation action gate.",
            "An increased idle penalty may trade false actions against necessary actions; completion, missed work and idle errors are all reported.",
            "The original global Adam age and candidate normalization are retained in the factorial study; local numerical diagnostics do not prove their causal responsibility.",
            "No strength is selected from evaluation results; all eighteen conditions are reported across three trained seeds."]}
    atomic_json(protocol, plan)
    frozen = {**sources,str(protocol):file_sha256(protocol)}
    training, evaluation, baseline = [], [], []
    prefix = [sys.executable,"-m","experiments.factors.study"]
    for item in branches:
        out = Path(item["output"])
        tail = ["--protocol",str(protocol),"--branch-id",item["id"]]
        training.append({"id":item["id"],"command":prefix+["train"]+tail,
            "outputs":[str(out/"checkpoint.pt"),str(out/"launch-complete.json")],
            "log":str(raw/"logs"/f"train_{item['id']}.log")})
        evaluation.append({"id":item["id"],"command":prefix+["evaluate"]+tail+
            ["--checkpoint",str(out/"checkpoint.pt"),"--output",str(raw/"evaluation"/f"{item['id']}.json")],
            "outputs":[str(raw/"evaluation"/f"{item['id']}.json")],
            "log":str(raw/"logs"/f"evaluate_{item['id']}.log")})
    for item in origins:
        baseline.append({"id":item["id"],"command":prefix+["baseline","--protocol",str(protocol),
            "--origin-id",item["id"],"--output",str(raw/"baseline"/f"{item['id']}.json")],
            "outputs":[str(raw/"baseline"/f"{item['id']}.json")],
            "log":str(raw/"logs"/f"baseline_{item['id']}.log")})
    for name, jobs in (("training",training),("evaluation",evaluation),("baseline",baseline)):
        atomic_json(folder/f"{name}-commands-v1.json",{"cwd":str(ROOT),"file_hashes":frozen,"jobs":jobs})
    print("Frozen 54 learning branches, 15 shared-source forks and 270 endpoint forks.")


def read_plan(path, branch_id=None, origin_id=None):
    sha = file_sha256(path)
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if plan["format"] != "factors-study-protocol-v1": raise ValueError("unknown protocol")
    verify_files(plan["sources"])
    item = next(row for row in plan["branches" if branch_id else "origins"] if row["id"] == (branch_id or origin_id))
    verify_files({item["source"]:item["source_sha256"]} if branch_id else {item["checkpoint"]:item["checkpoint_sha256"]})
    return plan,item,sha


def validate(trainer, plan, item, complete=False):
    for key in ("mode","idle_lambda","shrink_step"):
        if getattr(trainer.factor_config,key) != item[key]: raise ValueError(f"foreign factor {key}")
    origin = trainer.factor_origin
    if (Path(origin["checkpoint"]).resolve() != Path(item["source"]).resolve()
            or origin["checkpoint_sha256"] != item["source_sha256"] or trainer.config.seed != item["seed"]):
        raise ValueError("foreign factor source/seed")
    if (trainer.hidden_resets or trainer.credit_config.chunk_ticks != plan["chunk_ticks"]
            or trainer.factor_ticks % plan["chunk_ticks"] or not 0 <= trainer.factor_ticks <= plan["target_ticks"]
            or trainer.factor_start_stage_tick != plan["source_stage_ticks"]
            or trainer.stage_ticks != trainer.factor_start_stage_tick+trainer.factor_ticks
            or (complete and trainer.factor_ticks != plan["target_ticks"])):
        raise ValueError("invalid factor progress")
    regime = plan["return_regime"]
    if (tuple(trainer.world.regime.duration_range) != tuple(regime["duration"])
            or tuple(trainer.world.regime.gap_range) != tuple(regime["gap"])
            or trainer.world.regime.name != regime["name"] or len(trainer.regime_changes) != 2
            or tuple(trainer.config.train_dts) != (.25,)):
        raise ValueError("factor live regime differs")


def normalized_hash(trainer, device):
    value = trainer.semantic_state()
    value["credit"]["continuation"]["config"]["device"] = device
    return state_digest(value)


def verify_end(path, sha, **identity):
    if file_sha256(path) != sha: raise ValueError("protocol changed")
    read_plan(path, **identity)


def marker(plan,item,sha):
    out = Path(item["output"])
    value = json.loads((out/"launch-complete.json").read_text(encoding="utf-8"))
    if (value.get("format") != "factors-launch-complete-v1" or value.get("branch") != item
            or value.get("protocol_sha256") != sha or value.get("target") != plan["target_ticks"]
            or value.get("stage_ticks") != plan["target_ticks"] or value.get("hidden_resets") != 0
            or value.get("checkpoint_sha256") != file_sha256(out/"checkpoint.pt")):
        raise ValueError("foreign or incomplete factor completion")
    return value


def train(path, branch_id):
    from experiments.factors.run import FactorTrainer, main as train_main
    plan,item,sha = read_plan(path,branch_id=branch_id)
    out = Path(item["output"])
    if (out/"launch-complete.json").exists():
        done = marker(plan,item,sha)
        with _preserve_process_state():
            trainer = FactorTrainer.from_checkpoint(out/"checkpoint.pt",device=plan["evaluation_device"])
            validate(trainer,plan,item,complete=True)
            if normalized_hash(trainer,plan["training_device"]) != done["semantic_hash"]: raise ValueError("completion semantic mismatch")
        verify_end(path,sha,branch_id=branch_id)
        return
    start = out.with_name(out.name+".launch-start.json")
    binding = {"format":"factors-launch-start-v1","branch":item,"protocol_sha256":sha}
    if start.exists():
        if json.loads(start.read_text(encoding="utf-8")) != binding: raise ValueError("foreign partial binding")
    elif (out/"checkpoint.pt").exists(): raise ValueError("partial checkpoint unbound")
    else: atomic_json(start,binding)
    command = ["--source",item["source"],"--mode",item["mode"],"--idle-lambda",str(item["idle_lambda"]),
        "--output",str(out),"--target-ticks",str(plan["target_ticks"]),"--device",plan["training_device"]]
    if (out/"checkpoint.pt").exists():
        with _preserve_process_state():
            partial = FactorTrainer.from_checkpoint(out/"checkpoint.pt",device=plan["training_device"])
            validate(partial,plan,item)
        del partial
        command.append("--resume")
    train_main(command)
    with _preserve_process_state():
        trainer = FactorTrainer.from_checkpoint(out/"checkpoint.pt",device=plan["training_device"])
        validate(trainer,plan,item,complete=True)
        semantic = normalized_hash(trainer,plan["training_device"])
    verify_end(path,sha,branch_id=branch_id)
    atomic_json(out/"launch-complete.json",{"format":"factors-launch-complete-v1","branch":item,"protocol_sha256":sha,
        "target":plan["target_ticks"],"stage_ticks":plan["target_ticks"],"checkpoint_sha256":file_sha256(out/"checkpoint.pt"),
        "semantic_hash":semantic,"hidden_resets":0})


def evaluate(path,branch_id,checkpoint,output):
    from experiments.factors.run import FactorTrainer
    plan,item,sha = read_plan(path,branch_id=branch_id)
    cp,output = Path(checkpoint).resolve(),Path(output)
    if cp != (Path(item["output"])/"checkpoint.pt").resolve() or output.exists(): raise ValueError("wrong checkpoint or existing evaluation")
    done = marker(plan,item,sha)
    with _preserve_process_state():
        trainer = FactorTrainer.from_checkpoint(cp,device=plan["evaluation_device"])
        validate(trainer,plan,item,complete=True)
        before = state_digest(trainer.semantic_state())
        normalized = normalized_hash(trainer,plan["training_device"])
        if normalized != done["semantic_hash"]: raise ValueError("evaluation semantic mismatch")
        records = [diagnose_fork(trainer.model,trainer.world,trainer.h,condition=c,steps=plan["evaluation_steps"],
            dt=plan["dt"],device=plan["evaluation_device"]) for c in plan["conditions"]]
        if state_digest(trainer.semantic_state()) != before: raise AssertionError("evaluation changed source")
    verify_end(path,sha,branch_id=branch_id)
    marker(plan,item,sha)
    atomic_json(output,{"format":"factors-evaluation-v1","branch":item,"protocol_sha256":sha,"source_unchanged":True,
        "checkpoint":str(cp),"checkpoint_sha256":done["checkpoint_sha256"],"state_hash":before,
        "training_device_normalized_state_hash":normalized,"records":records})


def baseline(path,origin_id,output):
    from experiments.rewiring.run import RewireTrainer
    plan,item,sha = read_plan(path,origin_id=origin_id)
    if Path(output).exists(): raise FileExistsError(output)
    with _preserve_process_state():
        trainer = RewireTrainer.from_checkpoint(item["checkpoint"],device=plan["evaluation_device"])
        before = state_digest(trainer.semantic_state())
        records = [diagnose_fork(trainer.model,trainer.world,trainer.h,condition=c,steps=plan["evaluation_steps"],
            dt=plan["dt"],device=plan["evaluation_device"]) for c in plan["conditions"]]
        if state_digest(trainer.semantic_state()) != before: raise AssertionError("baseline changed source")
    verify_end(path,sha,origin_id=origin_id)
    atomic_json(Path(output),{"format":"factors-baseline-v1","origin":item,"protocol_sha256":sha,
        "state_hash":before,"source_unchanged":True,"records":records})


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("prepare","train","evaluate","baseline"))
    for name in ("protocol","output","raw","checkpoint"): parser.add_argument("--"+name,type=Path)
    parser.add_argument("--branch-id")
    parser.add_argument("--origin-id")
    args=parser.parse_args();torch.set_num_threads(1)
    if args.mode=="prepare":prepare(args.output,args.raw)
    elif args.mode=="train":train(args.protocol,args.branch_id)
    elif args.mode=="evaluate":evaluate(args.protocol,args.branch_id,args.checkpoint,args.output)
    else:baseline(args.protocol,args.origin_id,args.output)
