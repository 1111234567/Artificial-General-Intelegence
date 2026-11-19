"""Evaluate a frozen meta-learned controller only via the game's public observation API."""

import argparse
import json
from pathlib import Path

import torch

from experiments.seed_evolution.large_runner import _write
from . import meta_model, research_audit as audit, world
from .sensation import BriefCue


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint",type=Path)
    parser.add_argument("--sha256",required=True)
    parser.add_argument("--split",choices=("confirmation","replication","transfer","hard_task","diagnostic"),default="confirmation")
    parser.add_argument("--name",required=True)
    parser.add_argument("--cue-ticks",type=int,default=16)
    args=parser.parse_args()
    torch.set_num_threads(1)
    if audit.fingerprint(args.checkpoint)!=args.sha256:
        raise ValueError("Checkpoint differs from the frozen candidate")
    model=meta_model.load_policy(args.checkpoint,"cpu")
    tape=world.make_tape(audit.split(args.split))
    def factory(ticks=args.cue_ticks,**kw):
        return BriefCue(meta_model.RolloutAgent(model,**kw),ticks)
    conditions=audit.normal_and_controls(factory,tape)
    for name,kw in (("no_fast_write",{"write_enabled":False}),("no_fast_read",{"read_enabled":False}),
                    ("no_slow_recurrence",{"recurrence_enabled":False})):
        conditions[name]=audit.life_rows(tape,world.rollout(factory(**kw),tape))
    for k in (0,1,2,4,16):
        if k!=args.cue_ticks:
            conditions[f"cue_for_{k}_ticks"]=audit.life_rows(tape,world.rollout(factory(ticks=k),tape))
    moved=world.swap_sides(world.make_tape(tape.seeds,distance_jitter=.22,wind_scale=.12),first_trial=32)
    conditions["motion_perturbed"]=audit.life_rows(moved,world.rollout(factory(),moved))
    path=audit.write_audit(args.name,{"checkpoint":str(args.checkpoint),"sha256":args.sha256,"split":args.split,
                                      "cue_ticks":args.cue_ticks,"policy_source":audit.fingerprint(meta_model.__file__)},
                           tape,conditions)
    result=json.loads(path.read_text(encoding="utf-8"))
    print(path, {k:round(v["late_accuracy"]["mean"],4) for k,v in result["summary"].items()},flush=True)


if __name__=="__main__":
    main()
