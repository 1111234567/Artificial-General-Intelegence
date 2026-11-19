"""Score B content and old-answer intrusions on one identical demonstration per seed."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json

import torch
from experiments.learn_remember.gru_train import GRUAgent, sha
from experiments.lifelong import world as context
from experiments.lifelong.evaluate import CONDITIONS
from unittest.mock import patch


ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"data/experience-transfer"
OUT=BASE/"content-v2"
ARMS=("related","disjoint","direct")
STAGES=("B000","Bfirst002","Bfirst008","B001","B002","B004","B008")
# Explicit, pre-agreed rows; independent checks compare actual targets with the raw facts.
MAPS=(
    ((0,1,2,3),(2,1,0,3),(0,3,2,1),1),
    ((0,1,2,3),(0,3,2,1),(2,1,0,3),2),
    ((0,3,1,2),(0,3,2,1),(3,0,1,2),0),
)


def declare():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"protocol.json").open("x",encoding="utf-8") as f:
        json.dump(dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
            implementation_note="v1 used ContextWorld's four-slot default and produced 112 inputs. Scoring failed at the GRU input check before any policy result. v2 specifies the intended two-slot,56-input world with otherwise unchanged scoring; v1 sources, protocol and streams are retained.",
            dimensions="3 seeds, pure B long_both, 16 heldout actors, 256 tick warmup then 2048 observed ticks. No training.",
            history="Teacher actually chooses midpoint actions. Scores of the three models do not enter this world. Every new request scored once at first teacher FINISH, before execution. Teacher correct-version BEGIN receipt is available and common to all policies.",
            metrics="Full11 desired FINISH and exact action, conditional8 content, version-known4 payload, and margin/correct B versus common wrong previous payload on changed symbols. Group by raw A==B vs A!=B for all arms.",
            declaration="Secondary diagnostic declared before any trained-B measurements. Main claims depend on actual free actions.",
            maps=MAPS,stages=STAGES,arms=ARMS),f,indent=2)


def make_stream(seed):
    torch.set_num_threads(1)
    plan=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    assert plan["source_sha256"]==sha(__file__)
    a,c,b,cue=MAPS[seed]
    table=list(context.FAMILY_MAPPINGS)
    table[cue]=b
    x=[]; rows=[]; seen=set()
    with patch.object(context,"FAMILY_MAPPINGS",tuple(table)):
        w=context.ContextWorld(seed=981730+seed,batch_size=16,n_jobs=2,active_jobs=2,actor_ids=tuple(range(45000,45016)),
                               future_family=cue,dt=.25,**CONDITIONS["long_both"])
        for t in range(2304):
            obs=w.observe()
            y=w.oracle("midpoint")
            x.append(obs.clone())
            for i,j in (y>=3).nonzero().tolist():
                rid=int(w.request_id[i,j]); raw=int(w.raw_payload[i,j])
                if rid//2<256 or not bool(w.current_eligible[i,j]):
                    continue
                key=(i,j,rid)
                assert key not in seen and int(w.payload[i,j])==b[raw]
                seen.add(key)
                rows.append(dict(t=t,lane=i,job=j,rid=rid,raw=raw,target=int(y[i,j]),prior_a=a[raw],prior_c=c[raw]))
            w.act(y)
    path=OUT/f"stream_{seed}.pt"
    if path.exists(): raise FileExistsError(path)
    stream=torch.stack(x)
    assert stream.shape==(2304,16,56)
    torch.save(dict(obs=stream,rows=rows),path)
    return path


@torch.inference_mode()
def score(seed):
    torch.set_num_threads(1)
    plan=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    assert plan["source_sha256"]==sha(__file__)
    path=OUT/f"stream_{seed}.pt"; data=torch.load(path,weights_only=False,map_location="cpu")
    out=[]; artifacts={}
    for arm in ARMS:
        for stage in STAGES:
            p=BASE/f"training-v1/seed_{seed:03d}/{arm}/{stage}.pt"
            cp=torch.load(p,weights_only=False,map_location="cpu")
            model=GRUAgent(**cp["model_config"])
            model.load_state_dict(cp["model"]); model.eval()
            # CPU sequential GRU; no new initial state is inserted at request boundaries.
            h=model.initial_state(16)
            logits=[]
            for part in data["obs"].split(128):
                y,h=model(part,h)
                logits.append(y)
            logits=torch.cat(logits)
            counts={name:Counter() for name in ("overlap","conflict")}
            for row in data["rows"]:
                target=row["target"]
                v, payload=divmod(target-3,4)
                group="overlap" if row["prior_a"]==payload else "conflict"
                vals=logits[row["t"],row["lane"],row["job"]]
                native=int(vals.argmax()); within=vals[3+4*v:7+4*v]
                co=counts[group]
                co.update(n=1,exact_eight=int(vals[3:].argmax())==target-3,
                    payload_given_version=int(within.argmax())==payload,
                    chose_finish=native>=3,chose_correct_action=native==target)
                if group=="conflict":
                    assert row["prior_a"]==row["prior_c"]!=payload
                    old=row["prior_a"]
                    co["old_payload_given_version"]+=int(int(within.argmax())==old)
                    co["new_outscores_old"]+=int(bool(within[payload]>within[old]))
            artifacts[str(p.relative_to(ROOT))]=sha(p)
            out.append(dict(seed=seed,arm=arm,stage=stage,counts=counts))
    with (OUT/f"seed_{seed}.json").open("x",encoding="utf-8") as f:
        json.dump(dict(stream_sha=sha(path),sources=artifacts,records=out),f,indent=2)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=("declare","stream","score"));parser.add_argument("--seed",type=int,choices=range(3))
    args=parser.parse_args()
    if args.command=="declare":declare()
    elif args.command=="stream":make_stream(args.seed)
    else:score(args.seed)
