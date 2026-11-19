"""Replay all saved main executions through an independent world instance.

This checks saved public observations and teacher labels against actual world
causality, and reports surviving old requests after withdrawal. It does not
feed private auditing fields to the policy or claim to audit every historical
argmax from unsaved per-round policy weights.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import json

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import torch
from experiments.learn_remember.gru_train import OUT,load_checkpoint,sha,write_json
from experiments.lifelong.world import NEW,REPLACE
from experiments.continuity.run import state_digest


def audit(seed):
    torch.set_num_threads(1)
    folder=OUT/f"seed_{seed:03d}"
    model,world,hidden,cp=load_checkpoint(folder/"random.pt")
    cfg=cp["config"]
    totals=dict(ticks=0,public_observations_exact=0,oracle_labels_exact=0,executed_actions=0)
    old=dict(public_commands=0,nonwait_labels=0,active_tick_slots=0,unique_labeled_requests=set(),
             unique_public_commands=set(),teacher_nonwait_labels=0)
    checkpoints={}
    for stage,rounds,families,duration,gap in (
        ("short",cfg["short_rounds"],(3,),(1.,4.),(.5,2.)),
        ("wide",cfg["wide_rounds"],(0,1,2,3),(1.,16.),(.5,32.)),
        ("withdrawal",cfg["withdrawal_rounds"],(0,1,2),(1.,16.),(.5,32.))):
        world.set_family_policy(families)
        world.set_regime(duration_range=duration,gap_range=gap,name=f"gru-v1/{stage}")
        for index in range(rounds):
            path=folder/f"trajectory_{stage}_{index:03d}.pt"
            data=torch.load(path,map_location="cpu",weights_only=False)
            for tick in range(len(data["obs"])):
                obs=world.observe()
                labels=world.oracle("midpoint")
                assert torch.equal(obs,data["obs"][tick]),(stage,index,tick,"observation")
                assert torch.equal(labels,data["labels"][tick]),(stage,index,tick,"labels")
                teachers=data["teacher_lanes"]
                assert torch.equal(data["actions"][tick][teachers],labels[teachers]),"teacher lane mismatch"
                if stage=="withdrawal":
                    mask=world.current_family==3
                    command=(world.event_family==3)&((world.event_kind==NEW)|(world.event_kind==REPLACE))
                    nonwait=mask&(labels!=0)
                    old["public_commands"]+=int(command.sum())
                    old["nonwait_labels"]+=int(nonwait.sum())
                    old["active_tick_slots"]+=int((mask&world.active).sum())
                    old["teacher_nonwait_labels"]+=int((nonwait&teachers[:,None]).sum())
                    for lane,job in nonwait.nonzero().tolist():
                        old["unique_labeled_requests"].add((lane,job,int(world.request_id[lane,job])))
                    for lane,job in command.nonzero().tolist():
                        old["unique_public_commands"].add((lane,job,int(world.request_id[lane,job])))
                world.act(data["actions"][tick])
                totals["ticks"]+=1
                totals["public_observations_exact"]+=1
                totals["oracle_labels_exact"]+=1
                totals["executed_actions"]+=data["actions"][tick].numel()
            name=stage+"_mid" if index+1==rounds//2 else stage if index+1==rounds else None
            if name:
                _,_,_,saved=load_checkpoint(folder/f"{name}.pt")
                actual=state_digest(world.state_dict())
                assert actual==state_digest(saved["world"]),(name,"world state")
                checkpoints[name]=dict(world_state_exact=True,sha256=sha(folder/f"{name}.pt"))
    old["unique_labeled_requests"]=[list(x) for x in sorted(old["unique_labeled_requests"])]
    old["unique_public_commands"]=[list(x) for x in sorted(old["unique_public_commands"])]
    old["nonwait_label_gradient_repetitions"]=old["nonwait_labels"]*cfg["epochs"]
    report=dict(seed=seed,totals=totals,checkpoints=checkpoints,withdrawal_residual_family3=old,
        no_prior_family3_history_read_by_main_fit=True,
        source=dict(path=str(Path(__file__)),sha256=sha(__file__)),
        limit="Checks world causality, public observations, labels, teacher-executed lanes and exact checkpoint worlds. Per-round historical model weights were not all saved; this audit does not independently rederive every historical own-action argmax. Frozen endpoints are separately tested by an independent evaluator.")
    write_json(folder/"execution_audit.json",report)
    print(json.dumps(dict(seed=seed,done=True,totals=totals,residual=old)),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--seed",type=int,required=True)
    audit(parser.parse_args().seed)
