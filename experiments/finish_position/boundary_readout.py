"""Read frozen actions at matched pre-ready, first-legal and original teacher ticks."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from experiments.experience_transfer.train_transfer import SEEDS, mapping_scope, mapping_table
from experiments.learn_remember.gru_train import GRUAgent
from experiments.lifelong.evaluate import CONDITIONS
from experiments.lifelong.world import ContextWorld
from experiments.finish_position.switch_test import BASE, ARMS, checkpoint, sha


OUT = BASE/"boundary-v1"
WARM, STEPS = 512, 2048


def declare():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/"protocol.json").open("x", encoding="utf8") as stream:
        json.dump(dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha=sha(__file__),
            seed_offset=630180,actors=list(range(84300,84316)),warm=WARM,test=STEPS,stage="B008",
            design="Same 56-dimensional public history for both frozen models. Real midpoint teacher executes in an independent long mixed two-slot world, including cancellations/replacements. No tested model acts in this auxiliary measurement.",
            cases="New eligible mature conflicting A and B requests after warmup that have last actual processing WAIT, first legal tick and original teacher FINISH; retain full denominator and excluded counts.",
            outcomes="At the same three request times compare eleven-way intents and correct version+payload; estimate how often last WAIT and first legal correct FINISH are both chosen. Also count any premature FINISH intention during processing. This is not closed-loop completion."),stream,indent=2)


def checked():
    assert json.loads((OUT/"protocol.json").read_text(encoding="utf8"))["source_sha"] == sha(__file__)


@torch.inference_mode()
def collect(seed):
    checked()
    torch.set_num_threads(1)
    spec=SEEDS[seed]
    data={k:[] for k in ("obs","teacher","now","rid","active","started","ready","deadline","version","payload","raw","family","eligible","mature","mature_id")}
    with mapping_scope(mapping_table(seed,"related")):
        world=ContextWorld(seed=630180+seed,batch_size=16,n_jobs=2,active_jobs=2,
            actor_ids=tuple(range(84300,84316)),family_allowed=(spec["history_cue"],spec["B_cue"]),dt=.25,**CONDITIONS["long_both"])
        for _ in range(WARM+STEPS):
            obs=world.observe()
            action=world.oracle("midpoint")
            entries=dict(obs=obs,teacher=action,rid=world.request_id,active=world.active,started=world.started,
                ready=world.ready_time,deadline=world.deadline,version=world.version,payload=world.payload,
                raw=world.raw_payload,family=world.current_family,eligible=world.current_eligible,
                mature=world.matured_mask,mature_id=world.matured_request_id)
            for k,v in entries.items():
                data[k].append(v.numpy().copy())
            data["now"].append(world.now)
            world.act(action)
    facts={k:np.stack(vals) for k,vals in data.items()}
    path=OUT/f"facts_{seed}.npz"
    if path.exists(): raise FileExistsError(path)
    np.savez_compressed(path,**facts)


def samples(facts,seed):
    spec=SEEDS[seed]; rows=[]
    for t,lane,job in zip(*np.where(facts["mature"])):
        rid=int(facts["mature_id"][t,lane,job])
        if rid//2<WARM:
            continue
        steps=np.flatnonzero(facts["rid"][:,lane,job]==rid)
        cue=steps[0]
        raw=int(facts["raw"][cue,lane,job])
        if raw not in spec["conflict_raw"]:
            continue
        which=facts["family"][cue,lane,job]
        assert which in (spec["B_cue"],spec["history_cue"])
        family="B" if which==spec["B_cue"] else "A"
        sl=(steps,lane,job)
        active=facts["active"][sl] & facts["started"][sl]
        ready=facts["ready"][sl]; now=facts["now"][steps]
        processing=steps[active & (now<ready-1e-6)]
        legal=steps[active & (now>=ready-1e-6) & (now<=facts["deadline"][sl]+1e-6)]
        finishes=steps[facts["teacher"][sl]>=3]
        common=dict(rid=rid,lane=int(lane),job=int(job),family=family,raw=raw)
        if not (len(processing) and len(legal) and len(finishes)):
            rows.append(dict(**common,complete=False))
            continue
        assert len(finishes)==1 and processing[-1]+1==legal[0]
        target=int(facts["teacher"][finishes[0],lane,job])
        assert target==3+4*int(facts["version"][cue,lane,job])+spec[family][raw]
        rows.append(dict(**common,complete=True,target=target,processing=processing.tolist(),
            ticks=[int(processing[-1]),int(legal[0]),int(finishes[0])]))
    return rows


@torch.inference_mode()
def score(seed,arm):
    checked();torch.set_num_threads(1)
    path=OUT/f"facts_{seed}.npz"
    with np.load(path) as z:
        facts={k:z[k] for k in z.files}
    cp=checkpoint(seed,arm); state=torch.load(cp,weights_only=False,map_location="cpu")
    model=GRUAgent(**state["model_config"]);model.load_state_dict(state["model"]);model.eval()
    obs=torch.from_numpy(facts["obs"]);h=model.initial_state(16)
    scores=[]
    for x in obs:
        logits,h=model.step(x,h,.25);scores.append(logits.numpy())
    logits=np.stack(scores)
    counts={c:Counter() for c in ("A","B")}
    result=samples(facts,seed)
    for r in result:
        c=counts[r["family"]];c["matured"]+=1
        if not r["complete"]:
            continue
        c["n"]+=1; lane,job=r["lane"],r["job"]
        values=logits[r["ticks"],lane,job]
        act=values.argmax(-1)
        c["processing_any_finish"]+=int((logits[r["processing"],lane,job].argmax(-1)>=3).any())
        c["wait_then_first_correct"]+=int((act[0]==0) and (act[1]==r["target"]))
        for i,name in enumerate(("last_wait","first_legal","midpoint_teacher")):
            c[name+"_wait"]+=int(act[i]==0)
            c[name+"_finish"]+=int(act[i]>=3)
            c[name+"_correct"]+=int(act[i]==r["target"])
            c[name+"_conditional8"]+=int(values[i,3:].argmax()+3==r["target"])
        r["actions"]=act.tolist()
    value=dict(seed=seed,arm=arm,facts_sha=sha(path),model_sha=sha(cp),counts=counts,rows=result)
    with (OUT/f"{arm}_{seed}.json").open("x",encoding="utf8") as f: json.dump(value,f,separators=(",",":"))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("declare","collect","score"))
    p.add_argument("--seed",type=int,choices=range(3));p.add_argument("--arm",choices=ARMS);args=p.parse_args()
    if args.command=="declare":declare()
    elif args.command=="collect":collect(args.seed)
    else:score(args.seed,args.arm)
