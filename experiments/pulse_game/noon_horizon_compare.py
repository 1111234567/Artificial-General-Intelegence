"""Separate a changed routing policy from changes to its underlying physical body."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from . import noon_horizon as env, noon_horizon_train as train


def measure(records, worlds):
    names = tuple(records)
    n = len(records[names[0]]["action"])
    row = np.arange(n)
    result = {}
    for body, raw in records.items():
        for key in ("candidate", "no_taste", "queries"):
            if not np.array_equal(raw[key], records[names[0]][key]):
                raise ValueError("A cross-body route comparison needs the same public history and omitted identity")
        for experience, cue in (("short",0),("long",1)):
            k = 1 if cue == 0 else raw["events"].shape[-2]
            options = raw["events"][:,cue,:,:k,1].astype(float).sum(-1)
            result[f"{experience}/{body}/uniform"] = options.mean(-1)
            for route, chooser in records.items():
                selected = (chooser["action"][:,cue]>0).astype(int)
                result[f"{experience}/{body}/route_from_{route}"] = options[row,selected]
    for key in tuple(result):
        if key.startswith("short/"):
            rest = key.removeprefix("short/")
            result[f"mix/{rest}"] = (result[key]+result[f"long/{rest}"])/2
    for experience in ("short","long","mix"):
        for body in names[1:]:
            ancestor = result[f"{experience}/frozen/route_from_frozen"]
            fitted = result[f"{experience}/{body}/route_from_{body}"]
            frozen_route_in_fit = result[f"{experience}/{body}/route_from_frozen"]
            new_route_in_frozen = result[f"{experience}/frozen/route_from_{body}"]
            result[f"{experience}/{body}/native_minus_frozen_native"] = fitted-ancestor
            result[f"{experience}/{body}/same_fit_state_route_gain"] = fitted-frozen_route_in_fit
            result[f"{experience}/{body}/same_frozen_state_route_gain"] = new_route_in_frozen-ancestor
            result[f"{experience}/{body}/uniform_body_change"] = (result[f"{experience}/{body}/uniform"]-
                                                                  result[f"{experience}/frozen/uniform"])
            result[f"{experience}/{body}/same_frozen_policy_body_change"] = frozen_route_in_fit-ancestor
    return {name: v.reshape(worlds,4).mean(1) for name,v in result.items()}


def run(cohort, stage):
    info = train.load_plan()
    seed = info["holdout_seeds"][cohort]
    worlds = info["holdout_worlds"]
    births = env.COHORTS[cohort]
    root = env.DATA/"evaluation"/cohort
    keys = ("candidate", "no_taste", "queries", "action", "events")
    looked_at = []
    total = []
    for birth in births:
        banks = {}
        for condition, step in (("frozen",0), *((arm,stage) for arm in train.ARMS)):
            file = root/f"{worlds}-{step}"/f"{condition}-{birth}.npz"
            with np.load(file,allow_pickle=False) as raw:
                banks[condition] = {key: raw[key] for key in keys}
            looked_at.append((file.relative_to(env.world.ROOT).as_posix(),env.digest(file)))
        total.append(measure(banks,worlds))
    rows = {key: np.stack([entry[key] for entry in total]) for key in total[0]}
    result = dict(cohort=cohort,births=list(births),stage=stage,worlds=worlds,seed=seed,
                  plan_sha256=env.digest(train.PLAN),source_sha256=env.digest(Path(__file__)),files=dict(looked_at),
                  metrics=env.intervals.PairedCrossed(len(births),worlds,seed+331).summary(rows))
    env.storage._write_json(env.EVIDENCE/f"horizon-body-route-{cohort}-{stage}.json",result)


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--cohort",choices=env.COHORTS,required=True)
    cli.add_argument("--stage",type=int,required=True)
    run(**vars(cli.parse_args()))
