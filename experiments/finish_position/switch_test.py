"""Free-action A/B pair test on identical previously validated external cases."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import patch

from experiments.review_selection.pairs import sha
from experiments.wait_position import switch_pairs as previous


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/finish-position"
OUT = BASE / "pairs-v1"
ARMS = ("midpoint", "first")


def checkpoint(seed, arm):
    return BASE / f"train-v1/seed_{seed:03d}/{arm}/B008.pt"


def declare():
    OUT.mkdir(parents=True, exist_ok=True)
    parent = previous.OUT / "protocol.json"
    plan = previous.read(parent)
    refs = [previous.BASE / f"last-checks-v1/pairs/seed_{s:03d}_last.json" for s in range(3)]
    plan["files"].update({str(path.relative_to(ROOT)): sha(path) for path in refs + [Path(__file__), parent]})
    plan.update(arms=list(ARMS), created_utc=datetime.now(timezone.utc).isoformat(),
        intervention="Identical final pre-ready WAIT, original teacher midpoint or first legal FINISH target during old A rehearsal. The B teacher and all test actions are unchanged.",
        benchmark="Use the same 317 completed-exposure pairs and identical 512 warmup plus 4096 free ticks per seed as the frozen parent. Compare both-strict and both-eventual separately; decompose AB and BA; preserve AA/BB.")
    with (OUT/"protocol.json").open("x", encoding="utf8") as stream:
        json.dump(plan, stream, indent=2)


def execute(seed, arm):
    previous.run(seed, arm)
    cur = previous.read(OUT/f"seed_{seed:03d}_{arm}.json")
    old = previous.read(previous.BASE/f"last-checks-v1/pairs/seed_{seed:03d}_last.json")
    assert cur["events_sha"] == old["events_sha"]
    assert {k:c["pairs"] for k,c in cur["pair_counts"].items()} == {k:c["pairs"] for k,c in old["pair_counts"].items()}
    if arm == "midpoint":
        assert cur["pair_counts"] == old["pair_counts"]
        assert cur["idle_opportunities"] == old["idle_opportunities"] and cur["idle_actions"] == old["idle_actions"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare","run","summarize"))
    parser.add_argument("--seed", type=int, choices=range(3))
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.command == "declare":
        declare()
    else:
        with patch.object(previous,"model_path",checkpoint), patch.object(previous,"OUT",OUT), patch.object(previous,"ARMS",ARMS):
            if args.command == "run":
                execute(args.seed, args.arm)
            else:
                previous.summarize()
