"""The unchanged held-out two-request test, with only the deployment checkpoint varied."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from unittest.mock import patch

import torch

from experiments.review_selection import pairs, pair_audit


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/wait-position"
OUT = BASE / "pairs-v1"
ARMS = ("midpoint", "random", "hard")


def model_path(seed, arm):
    return BASE/f"train-v1/seed_{seed:03d}/{arm}/B008.pt"


def declare():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), Path(pairs.__file__), Path(pair_audit.__file__), pairs.OUT/"protocol.json"]
    paths += [pairs.OUT/f"seed_{seed:03d}_uniform.json" for seed in range(3)]
    with (OUT/"protocol.json").open("x", encoding="utf-8") as f:
        json.dump(dict(created_utc=datetime.now(timezone.utc).isoformat(), arms=ARMS, seeds=list(range(3)),
            files={str(path.relative_to(ROOT)):pairs.sha(path) for path in paths},
            test="Use the previous 512+4096 tick two-slot external schedule, 16 unseen actors, AA/AB/BA/BB. Execute frozen model argmax on every tick, without teaching or h resets. No cancellations/replacements in this supplement.",
            primary="Both halves strictly completed in mature same-slot same-raw AB/BA conflict pairs. Counts for A then B and B then A, pair ordinary outcomes, shared and repeat controls, and all idle actions are saved.",
            checks="Match external events, exact pair identities, and each denominator with prior uniform. Reconstruct accepted actions and error-free completions from recorded actions, times, targets, versions and receipts."), f, indent=2)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_plan():
    plan=read(OUT/"protocol.json")
    for name,expected in plan["files"].items():
        assert pairs.sha(ROOT/name)==expected, name


def run(seed,arm):
    torch.set_num_threads(1)
    check_plan()
    out=OUT/f"seed_{seed:03d}_{arm}.json"
    if out.exists():
        raise FileExistsError(out)
    with patch.object(pairs,"path_for",model_path):
        r,facts=pairs.rollout(seed,arm)
    source=pairs.OUT/f"seed_{seed:03d}_uniform.json"
    orig=read(source)
    assert r["events_sha"]==orig["events_sha"]
    assert {k:v["pairs"] for k,v in r["pair_counts"].items()}=={k:v["pairs"] for k,v in orig["pair_counts"].items()}
    # Identical recorded test stimuli do not imply identical closed-loop receipts.
    assert [(x["request"],x["issued"],x["seq"],x["raw"],x["context"]) for x in r["requests"]]==[
        (x["request"],x["issued"],x["seq"],x["raw"],x["context"]) for x in orig["requests"]]
    import numpy as np
    np.savez_compressed(out.with_suffix(".npz"),**facts)
    r.update(reference_sha=pairs.sha(source), checkpoint_sha=pairs.sha(model_path(seed,arm)),
             facts_sha=pairs.sha(out.with_suffix(".npz")),protocol_sha=pairs.sha(OUT/"protocol.json"))
    out.write_text(json.dumps(r,separators=(",",":")),encoding="utf-8")
    print(seed,arm,r["pair_counts"]["conflict_switch"],flush=True)


def summarize():
    check_plan()
    rows=[];checks=[]
    for seed in range(3):
        matched=None
        old=read(pairs.OUT/f"seed_{seed:03d}_uniform.json")
        for arm in ARMS:
            with patch.object(pairs,"OUT",OUT),patch.object(pairs,"path_for",model_path):
                audit,ids=pair_audit.check(seed,arm)
            if matched is None:
                matched=ids
            assert matched==ids
            r=read(OUT/f"seed_{seed:03d}_{arm}.json")
            if arm=="midpoint" and r["pair_counts"]==old["pair_counts"]:
                audit["previous_uniform_counts_exact"]=True
            rows.append(dict(seed=seed,arm=arm,counts=r["pair_counts"],idle=r["idle_actions"],idle_n=r["idle_opportunities"]))
            checks.append(audit)
    aggregate={}
    for arm in ARMS:
        vals=[r["counts"]["conflict_switch"] for r in rows if r["arm"]==arm]
        aggregate[arm]=dict(pairs=sum(r["pairs"] for r in vals),strict=sum(r["both_strict"] for r in vals),
            ordinary=sum(r["both_ordinary"] for r in vals),strict_mean=mean(100*r["both_strict"]/r["pairs"] for r in vals))
    (OUT/"summary.json").write_text(json.dumps(dict(audit="passed",rows=rows,checks=checks,aggregate=aggregate),
        indent=2),encoding="utf-8")
    print(aggregate)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("declare","run","summarize"))
    parser.add_argument("--seed",type=int,choices=range(3));parser.add_argument("--arm",choices=ARMS)
    args=parser.parse_args()
    if args.command=="declare":declare()
    elif args.command=="run":run(args.seed,args.arm)
    else:summarize()
