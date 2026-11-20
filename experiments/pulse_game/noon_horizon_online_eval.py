"""Compare independently trained one-path and matched paired route heads on new physical lives."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from . import noon_horizon_online as fit, noon_horizon as env
from . import noon_horizon_anchor as anchor, noon_horizon_anchor_preview as preview
from . import noon_horizon_eval as stats


BASELINES=("preview_native","quiet_native","always_known")
STREAMS=(1,2)


def head_weights(cohort,birth):
    plan,parents=fit.settings()
    identities={}
    weights={}
    index=list(env.COHORTS[cohort]).index(birth)
    parent=env.checkpoint(cohort,birth).relative_to(env.world.ROOT).as_posix()
    for stream in STREAMS:
        for strategy in fit.ARMS:
            place=fit.ROOT/"training"/cohort/f"stream-{stream}"/strategy
            path=place/"route-residual.npz"
            with np.load(path,allow_pickle=False) as saved:
                if int(saved["step"])!=plan["updates"] or str(saved["plan_sha256"])!=env.digest(fit.PLAN):
                    raise ValueError("Registered training endpoint missing")
                if str(saved["parent_hashes"][index])!=parents[parent]:
                    raise ValueError("A route was loaded against the wrong physical body")
                if str(saved["paid_ledger_sha256"])!=env.digest(place/"paid-training-ledger.npz"):
                    raise ValueError("Trained route and actual paid paths differ")
                coefficient=torch.from_numpy(saved["parameters"][index])
                if coefficient.shape!=(fit.WIDTH,) or not coefficient.isfinite().all():
                    raise ValueError("Route residual has the wrong coefficient count")
            label=f"{strategy}_{stream}"
            weights[label]=coefficient
            identities[path.relative_to(env.world.ROOT).as_posix()]=env.digest(path)
    return weights,identities


def reduce(raw,worlds):
    arms=(*BASELINES,*(f"{a}_{s}" for a in fit.ARMS for s in STREAMS))
    rows={}
    for label in arms:
        chosen={k:raw[k] for k in ("candidate","no_taste")}
        chosen.update({k:raw[f"{label}/{k}"] for k in ("action","motor_logit")})
        chosen["events"]=raw["quiet_events" if label=="quiet_native" else "events"]
        table=stats.measure(chosen,worlds)
        true=raw[f"{label}/direct_events"][...,1].sum(-1).reshape(worlds,4,2).mean(1)
        for h,which in enumerate(("short","long")):
            np.testing.assert_array_equal(table[f"{which}/native/total"],true[:,h])
        rows[label]=table
    for arm in fit.ARMS:
        rows[arm]={key:(rows[f"{arm}_1"][key]+rows[f"{arm}_2"][key])/2 for key in rows[f"{arm}_1"]}
    output={f"{arm}/{key}":v for arm,matrix in rows.items() for key,v in matrix.items()}
    for l in ("online", "online_1", "online_2", "paired", "paired_1", "paired_2"):
        relatives=BASELINES
        if l.startswith('online'):
            relatives=(*relatives,l.replace('online','paired'))
        for r in relatives:
            for key,data in rows[l].items():
                output[f"{l}_minus_{r}/{key}"]=data-rows[r][key]
    return output


def run(cohort):
    plan,parents=fit.settings()
    worlds=plan["holdout_worlds"]
    seed=plan["holdout_seeds"][cohort]
    roots=env.COHORTS[cohort]
    directory=fit.ROOT/"evaluation"/cohort; directory.mkdir(parents=True,exist_ok=True)
    hashes={}
    code={Path(p).name:env.digest(p) for p in (__file__,fit.__file__,preview.__file__,anchor.__file__,stats.__file__,env.__file__)}
    plan_hash=env.digest(fit.PLAN)
    rows=[]
    for birth in roots:
        heads,files=head_weights(cohort,birth)
        body=env.checkpoint(cohort,birth)
        key=body.relative_to(env.world.ROOT).as_posix()
        if env.digest(body)!=parents[key]: raise ValueError("Physical body differs from training")
        signature=repr((code,plan_hash,files,parents[key],worlds,seed))
        dest=directory/f"{birth}.npz"
        if not dest.exists():
            graph,rule=env.load(cohort,birth)
            data=anchor.replay(graph,rule,seed,worlds,lambda *args:preview.tests(*args,heads))
            env.storage._write_raw(dest,**data,signature=signature)
        with np.load(dest,allow_pickle=False) as archive:
            if str(archive["signature"])!=signature:raise ValueError("Mismatched held-out paths")
            rows.append(reduce(archive,worlds))
        hashes.update(files)
        hashes[dest.relative_to(env.world.ROOT).as_posix()]=env.digest(dest)
        hashes[key]=parents[key]
        print(cohort,birth,flush=True)
    table={key:np.stack([row[key] for row in rows]) for key in rows[0]}
    result=dict(cohort=cohort,births=list(roots),worlds=worlds,seed=seed,streams=STREAMS,plan_sha256=plan_hash,
                code=code,files=hashes,metrics=env.intervals.PairedCrossed(len(roots),worlds,seed+331).summary(table))
    env.storage._write_json(env.EVIDENCE/f"horizon-online-{cohort}.json",result)


if __name__=="__main__":
    cli=ArgumentParser(description=__doc__)
    cli.add_argument("--cohort",choices=env.COHORTS,required=True)
    args=cli.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run(args.cohort)
