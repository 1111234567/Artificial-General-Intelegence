"""Complete the public/state BEGIN by native/learned FINISH frozen policy comparison."""
import argparse
from collections import Counter
from pathlib import Path

from experiments.action_return import readout as begin, finish_readout as finish
import torch


OUT = begin.ROOT/"data/action-return/crossed-readout-v1"
PLAN = OUT/"protocol.json"


def declare():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    finish.checked()
    inputs = [begin.PLAN,finish.PLAN]
    for seed in range(3):
        a=begin.OUT/f"seed_{seed:03d}"
        f=finish.ROOT/f"seed_{seed:03d}"
        inputs.extend((a/"fits.pt", a/"result.json", f/"model.pt", f/"result.json"))
    OUT.mkdir(exist_ok=True,parents=True)
    begin.old.write(PLAN,dict(design="After observing the two BEGIN arms and the state-BEGIN conditional FINISH arm, complete the remaining frozen cell before its evaluation. Public current observation BEGIN gate, identical 24-state conditional FINISH head. No refitting or threshold selection.",
        source_sha256=begin.old.sha(__file__), inputs={str(p):begin.old.sha(p) for p in inputs},
        seeds=[0,1,2],families=[0,1,2,3],conditions=["short","long_both"],steps=2048,
        primary="actors4..7; same future primary request identities in all four policies; report ordinary/strict, actual idle actions, accepted and legal FINISH counts and shadow FINISH responses."))


def check():
    p=begin.old.read(PLAN)
    if begin.old.sha(__file__)!=p["source_sha256"] or any(begin.old.sha(f)!=h for f,h in p["inputs"].items()):
        raise ValueError("frozen factorial comparison inputs changed")
    return p


def run(seed):
    plan=check(); _,parent=finish.checked()
    a=begin.OUT/f"seed_{seed:03d}"
    f=finish.ROOT/f"seed_{seed:03d}"
    fit_payload=torch.load(a/"fits.pt",map_location="cpu",weights_only=False)
    state=torch.load(f/"model.pt",map_location="cpu",weights_only=False)
    endpoint=parent["sources"][str(seed)]["endpoint"]
    trainer=begin.native.NativeTrainer.from_checkpoint(endpoint["path"],device="cpu")
    before=begin.state_digest(trainer.semantic_state())
    assert before==endpoint["semantic_hash"]
    head=begin.BeginController(trainer.model,"public_obs56",fit_payload["fits"]["public_obs56"],state["begin_head.indices"].tolist())
    fixed=[{key:state[f"j{j}_{key}"] for key in ("mean","std","coef","bias")} for j in range(2)]
    model=finish.FinishController(head,fixed)
    control=begin.old.read(a/"result.json")
    same_finish=begin.old.read(f/"result.json")
    assert begin.old.sha(f/"model.pt")==same_finish["model_sha256"] and begin.old.sha(a/"fits.pt")==control["fits_sha256"]
    requests=[req for group in fit_payload["training_request_sets"].values() for req in group]
    output=OUT/f"seed_{seed:03d}.json"
    if output.exists():
        raise FileExistsError(output)
    records=[]
    for cond in plan["conditions"]:
        for fam in plan["families"]:
            value=finish.evaluate_conditional(model,trainer,fam,cond,requests)
            for comp in (next(r for r in control["records"] if (r["arm"],r["condition"],r["family"])==("public_obs56",cond,fam)),
                         next(r for r in same_finish["records"] if (r["condition"],r["family"])==(cond,fam))):
                for key in ("start_hidden_hash","start_world_state_hash","end_event_hash","end_scheduler_hash"):
                    if comp["record"][key]!=value["record"][key]: raise AssertionError(key)
                if comp["shadow"]["heldout"]["primary_request_set_sha256"]!=value["shadow"]["heldout"]["primary_request_set_sha256"]:
                    raise AssertionError("different held-out primary requests")
            value.update(condition=cond,family=fam)
            records.append(value)
            print(seed,cond,fam,flush=True)
    if before!=begin.state_digest(trainer.semantic_state()):
        raise ValueError("source endpoint changed")
    begin.old.write(output,dict(protocol_sha256=begin.old.sha(PLAN), seed=seed, records=records))


def summarize():
    p=check();rows=[];inputs={}
    for seed in p["seeds"]:
        a=begin.old.read(begin.OUT/f"seed_{seed:03d}"/"result.json")["records"]
        b=begin.old.read(finish.ROOT/f"seed_{seed:03d}"/"result.json")["records"]
        path=OUT/f"seed_{seed:03d}.json";c=begin.old.read(path)
        assert c["protocol_sha256"]==begin.old.sha(PLAN)
        inputs[str(path)]=begin.old.sha(path)
        for cond in p["conditions"]:
            for gate,ending,records in (("non_input24","native",[r for r in a if r["arm"]=="non_input24"]),
                    ("public_obs56","native",[r for r in a if r["arm"]=="public_obs56"]),
                    ("non_input24","non_input24",b),("public_obs56","non_input24",c["records"])):
                chosen=[r for r in records if r["condition"]==cond]
                assert sorted(r["family"] for r in chosen)==p["families"]
                total=Counter()
                for r in chosen: total.update(r["shadow"]["heldout"]["counts"])
                rows.append(dict(seed=seed,condition=cond,begin=gate,finish=ending,counts=dict(total)))
    begin.old.write(OUT/"summary.json",dict(protocol_sha256=begin.old.sha(PLAN),additional_forks=24,inputs=inputs,heldout=rows))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("declare","run","summarize"));parser.add_argument("--seed",type=int,choices=range(3))
    args=parser.parse_args()
    if args.action=="declare": declare()
    elif args.action=="run": run(args.seed)
    else: summarize()
