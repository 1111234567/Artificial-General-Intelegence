"""Apply the already defined request-pair and fixed-history tests to fixed-late WAIT."""
import argparse
from datetime import datetime,timezone
import json
from unittest.mock import patch

from experiments.wait_position import switch_pairs as sp, fixed_history as fh
from experiments.review_selection.pairs import sha


OUT=sp.BASE/"last-checks-v1"


def checkpoint(seed,arm):
    assert arm=="last"
    return sp.BASE/f"last-v1/seed_{seed:03d}/last/B008.pt"


def declare():
    OUT.mkdir(parents=True,exist_ok=True)
    for dirname,source in (("pairs",sp.OUT),("history",fh.OUT)):
        folder=OUT/dirname
        folder.mkdir(exist_ok=True)
        old=source/"protocol.json"
        plan=sp.read(old)
        plan.update(arms=["last"], parent_sha=sha(old), supplementary_code=sha(__file__),
            supplement="Fixed final eligible processing WAIT, chosen after the initial hard-position distribution showed frequent selection of this boundary. Original tests/scoring/budgets remain as declared.",
            frozen_utc=datetime.now(timezone.utc).isoformat())
        with (folder/"protocol.json").open("x",encoding="utf-8") as f:json.dump(plan,f,indent=2)


def run(seed):
    with patch.object(sp,"model_path",checkpoint),patch.object(sp,"ARMS",("last",)):
        with patch.object(sp,"OUT",OUT/"pairs"):
            assert sp.read(sp.OUT/"protocol.json")["supplementary_code"]==sha(__file__)
            sp.run(seed,"last")
        with patch.object(fh,"OUT",OUT/"history"):
            assert sp.read(fh.OUT/"protocol.json")["supplementary_code"]==sha(__file__)
            fh.run(seed)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("command",choices=("declare","run","summarize"))
    p.add_argument("--seed",type=int,choices=range(3));a=p.parse_args()
    if a.command=="declare":declare()
    elif a.command=="run":run(a.seed)
    else:
        with patch.object(sp,"model_path",checkpoint),patch.object(sp,"ARMS",("last",)),patch.object(sp,"OUT",OUT/"pairs"):
            sp.summarize()
