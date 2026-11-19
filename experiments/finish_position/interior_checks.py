"""Apply the fixed pair/teacher-history tests to the single interior-time supplement."""
import argparse
from datetime import datetime, timezone
import json
import os
from unittest.mock import patch

from experiments.finish_position import boundary_readout as h, switch_test as s


OUT=s.BASE/"interior-checks-v1"


def model_path(seed,arm):
    assert arm=="interior"
    return s.BASE/f"interior-v1/seed_{seed:03d}/interior/B008.pt"


def declare():
    for name,source in (("pairs",s.OUT),("history",h.OUT)):
        folder=OUT/name
        folder.mkdir(parents=True,exist_ok=True)
        original=source/"protocol.json"
        plan=s.previous.read(original)
        plan.update(arms=["interior"],parent_sha=s.sha(original),created_utc=datetime.now(timezone.utc).isoformat(),
            supplementary_code=s.sha(__file__),why="Main endpoints traded B legal misses versus A premature action and paired reliability. One previously unmeasured floor-halfway position, selected after observing the main endpoints.")
        with (folder/"protocol.json").open("x",encoding="utf-8") as f:json.dump(plan,f,indent=2)
        if name=="history":
            for seed in range(3):
                src=source/f"facts_{seed}.npz"; dest=folder/src.name
                os.link(src,dest)
                assert s.sha(src)==s.sha(dest)


def run(seed):
    with (patch.object(s,"OUT",OUT/"pairs"), patch.object(s.previous,"OUT",OUT/"pairs"),
          patch.object(s.previous,"model_path",model_path)):
        assert s.previous.read(s.OUT/"protocol.json")["supplementary_code"]==s.sha(__file__)
        s.execute(seed,"interior")
    with patch.object(h,"OUT",OUT/"history"),patch.object(h,"checkpoint",model_path):
        h.score(seed,"interior")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("command",choices=("declare","run","summarize"));p.add_argument("--seed",type=int)
    a=p.parse_args()
    if a.command=="declare":declare()
    elif a.command=="run":
        run(a.seed)
    else:
        with patch.object(s.previous,"model_path",model_path),patch.object(s.previous,"OUT",OUT/"pairs"),patch.object(s.previous,"ARMS",("interior",)):
            s.previous.summarize()
