"""Derive each acceptance, clean completion and eligibility from saved pair facts."""
from collections import defaultdict, Counter
import json
import numpy as np

from experiments.review_selection import pairs as p


def check(seed,arm):
    path=p.OUT/f"seed_{seed:03d}_{arm}.json"
    r=json.loads(path.read_text(encoding="utf-8"))
    assert p.sha(path.with_suffix(".npz"))==r["facts_sha"]
    assert p.sha(p.path_for(seed,arm))==r["checkpoint_sha"]
    with np.load(path.with_suffix(".npz")) as z:
        act,feedback,active,ids=[z[k] for k in ("actions","receipts","active","ids")]
    assert act.shape==(p.WARM+p.TEST,16,2)
    valid=np.zeros_like(active)
    modeled=np.zeros_like(active)
    per_slot=defaultdict(list)
    for row in r["requests"]:
        per_slot[row["lane"],row["job"]].append(row)
    metrics=defaultdict(Counter)
    signatures=[]
    for (lane,job),requests in per_slot.items():
        for idx,row in enumerate(requests):
            assert row["seq"]==idx
            start=row["issued"]
            stop=requests[idx+1]["issued"] if idx+1<len(requests) else len(act)
            t=np.arange(start,stop)
            choice=act[start:stop,lane,job]
            assert (ids[start:stop,lane,job]==row["request"]).all()
            in_time=t*.25<=row["deadline"]+1e-6
            eligible=in_time.copy()
            begins=np.flatnonzero(in_time & (choice==row["version"]+1))
            finish=None
            if len(begins):
                first=int(begins[0])
                valid[start+first,lane,job]=True
                legal=in_time & (t*.25 >= t[first]*.25+row["duration"]-1e-6)
                finishes=np.flatnonzero(legal & (choice==3+4*row["version"]+row["target"]))
                if len(finishes):
                    finish=int(finishes[0])
                    valid[start+finish,lane,job]=True
                    eligible[finish+1:]=False
            modeled[start:stop,lane,job]=eligible
            mature=(len(act)-1)*.25>row["deadline"]+1e-6
            assert row["matured"]==mature
            if mature:
                done=finish is not None
                clean=done and not ((choice!=0)&eligible&~valid[start:stop,lane,job]).any()
                assert row["done"]==done and row["strict"]==clean
            signatures.append((lane,job,row["request"],row["raw"],row["context"],start))
        for j in range(0,len(requests)-1,2):
            first,second=requests[j:j+2]
            assert first["raw"]==second["raw"] and first["conflict"]==second["conflict"]
            if first["issued"]<p.WARM or not second["matured"]:
                continue
            assert first["matured"]
            part="conflict" if first["conflict"] else "shared"
            order=first["context"]+second["context"]
            for name in (order,"repeat" if order[0]==order[1] else "switch"):
                c=metrics[f"{part}_{name}"]
                c.update(pairs=1,both_strict=int(first["strict"] and second["strict"]),
                    both_ordinary=int(first["done"] and second["done"]),
                    first_strict=int(first["strict"]),second_strict=int(second["strict"]))
    assert np.array_equal(modeled,active)
    expected=np.where(valid,1,np.where(act==0,0,-1))
    assert np.array_equal(expected,feedback)
    assert metrics==r["pair_counts"]
    assert int((~active[p.WARM:]).sum())==r["idle_opportunities"]
    assert int(((~active[p.WARM:]) & (act[p.WARM:]!=0)).sum())==r["idle_actions"]
    return dict(path=path.name,sha256=p.sha(path),events=r["events_sha"],checked_ticks=len(act)),signatures


if __name__=="__main__":
    checks=[]
    for seed in range(3):
        base=None
        for arm in p.ARMS:
            r,ids=check(seed,arm)
            if base is None:base=ids
            assert base==ids
            checks.append(r)
    (p.OUT/"audit.json").write_text(json.dumps(dict(audit="passed",checks=checks),indent=2),encoding="utf-8")
