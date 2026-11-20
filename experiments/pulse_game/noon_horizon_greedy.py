"""Analyze only the native route discretization using sealed real path replays."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from . import noon_horizon as env, noon_horizon_eval as stats


FIELDS=("short/select_unknown_actual","long/select_unknown_actual","switch/unknown_actual_long_minus_short",
        "short/native/total","long/native/total","mix/native/total",
        "short/native_minus_always_known/total","long/native_minus_always_known/total","mix/native_minus_always_known/total",
        "short/native_minus_uniform/total","long/native_minus_uniform/total","mix/native_minus_uniform/total",
        "long/native_minus_always_known/review","long/native_minus_always_known/paid_accuracy",
        "mix/native_minus_opposite_cue_route/total")


def run():
    stage=12000
    report={}
    inputs={}
    seal=env.EVIDENCE/"horizon-seal.json"
    sealed=json.loads(seal.read_text(encoding="utf8"))
    for cohort,births in env.COHORTS.items():
        archived=env.EVIDENCE/f"horizon-result-{cohort}-{stage}.json"
        if env.digest(archived)!=sealed["result_files"][archived.relative_to(env.world.ROOT).as_posix()]:
            raise ValueError("Original evaluation changed")
        detail=json.loads(archived.read_text(encoding="utf8"))
        worlds,seed=detail["worlds"],detail["seed"]
        path=env.DATA/"evaluation"/cohort
        rows=[]
        for birth in births:
            both=[]
            for at,name in ((0,"frozen"),(stage,"informative")):
                file=path/f"{worlds}-{at}"/f"{name}-{birth}.npz"
                relative=file.relative_to(env.world.ROOT).as_posix()
                check=env.digest(file)
                if sealed["eval_inputs"][relative]!=check:raise ValueError("Original route replay changed")
                inputs[relative]=check
                with np.load(file,allow_pickle=False) as raw:
                    observed={key:raw[key] for key in ("candidate","no_taste","motor_logit","events","action")}
                deterministic={**observed,"action":np.where(observed["motor_logit"]>0,1,-1)}
                both.append((stats.measure(observed,worlds),stats.measure(deterministic,worlds)))
            base,info=both
            variants={"frozen_sampled":base[0],"frozen_greedy":base[1],"trained_sampled":info[0],"trained_greedy":info[1]}
            for label,a,b in (("greedy_minus_sampled",info[1],info[0]),("greedy_minus_frozen_sampled",info[1],base[0])):
                variants[label]={key:a[key]-b[key] for key in FIELDS}
            rows.append({f"{name}/{key}":v[key] for name,v in variants.items() for key in FIELDS})
        matrix={key:np.stack([r[key] for r in rows]) for key in rows[0]}
        report[cohort]=env.intervals.PairedCrossed(len(births),worlds,seed+331).summary(matrix)
    result=dict(analysis="Post hoc native route argmax only. The review and later paid choices retain every actual archived action and draw.",
                stage=stage,bodies={k:list(v) for k,v in env.COHORTS.items()},archive_seal_sha256=env.digest(seal),
                source_sha256=env.digest(Path(__file__)),inputs=inputs,metrics=report)
    env.storage._write_json(env.EVIDENCE/"horizon-greedy-native.json",result)


if __name__=="__main__":
    cli=ArgumentParser(description=__doc__)
    cli.parse_args()
    run()
