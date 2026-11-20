"""Evaluate independent planning-horizon lives and their real sibling routes."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from . import noon_horizon as env, noon_horizon_train as train


def interval_axes(dictionary, worlds):
    return {key: data.reshape(worlds, 4).mean(1) for key, data in dictionary.items()}


def measure(raw, worlds):
    n = len(raw["candidate"])
    rows = np.arange(n)
    choice = (raw["action"] > 0).astype(int)
    prob = 1 / (1 + np.exp(-raw["motor_logit"]))
    missing = raw["no_taste"]
    missing_chance = np.where(missing[:, None] != 0, prob, 1-prob)
    select_missing = choice == missing[:, None]
    earned = raw["events"][..., 1].astype(float)
    if not np.isin(earned, (-1, 1)).all():
        raise ValueError("Every replayed physical opportunity must pay a real signed resource")
    if not (np.all(missing[rows ^ 1] == 1 - missing)
            and np.array_equal(raw["candidate"], raw["candidate"][rows ^ 1])):
        raise ValueError("The paired histories did not preserve the public candidate menu")

    result = {"switch/unknown_actual_long_minus_short": select_missing[:, 1].astype(float)-select_missing[:, 0],
              "switch/unknown_probability_long_minus_short": missing_chance[:, 1]-missing_chance[:, 0],
              "switch/known_to_unknown": (~select_missing[:,0]) & select_missing[:,1],
              "switch/unknown_to_known": select_missing[:,0] & (~select_missing[:,1]),
              "switch/side_preference_long_minus_short": choice[:,1].astype(float)-choice[:,0],
              "switch/route_changed": choice[:,0] != choice[:,1]}
    for h, name in enumerate(("short", "long")):
        k = 1 if h == 0 else earned.shape[-1]
        chosen = choice[:, h]
        other_cue = choice[:, 1-h]
        tests = {"native": chosen, "always_known": 1-missing, "always_unknown": missing,
                 "opposite": 1-chosen, "opposite_history": chosen[rows ^ 1], "opposite_cue_route": other_cue}
        events = earned[:, h]
        reread = {"total": events[..., :k].sum(-1), "review": events[..., 0]}
        if h:
            reread["paid_accuracy"] = (events[..., 1:] > 0).mean(-1)
            reread["paid_resource"] = events[..., 1:].sum(-1)
        result[f"{name}/select_unknown_actual"] = select_missing[:, h]
        result[f"{name}/select_unknown_probability"] = missing_chance[:, h]
        result[f"{name}/right_actual"] = chosen
        result[f"{name}/switch_role_changed_across_omission"] = chosen != chosen[rows ^ 1]
        for score, matrix in reread.items():
            result[f"{name}/uniform/{score}"] = matrix.mean(1)
            for policy, selected in tests.items():
                result[f"{name}/{policy}/{score}"] = matrix[rows, selected]
            for reference in ("uniform", "opposite", "opposite_history", "opposite_cue_route", "always_known", "always_unknown"):
                result[f"{name}/native_minus_{reference}/{score}"] = (result[f"{name}/native/{score}"]
                                                                      - result[f"{name}/{reference}/{score}"])
            result[f"{name}/unknown_minus_known/{score}"] = matrix[rows,missing]-matrix[rows,1-missing]
        result[f"{name}/native_wrong_cue/total"] = earned[rows,1-h,other_cue,:k].sum(-1)
        result[f"{name}/native_minus_wrong_cue/total"] = (result[f"{name}/native/total"]
                                                          - result[f"{name}/native_wrong_cue/total"])
        result[f"{name}/same_route_cue_effect/total"] = (result[f"{name}/native/total"]
                                                         - earned[rows,1-h,chosen,:k].sum(-1))

    for reference in ("native", "native_minus_uniform", "native_minus_opposite", "native_minus_opposite_history",
                      "native_minus_opposite_cue_route", "native_minus_wrong_cue", "native_minus_always_known",
                      "native_minus_always_unknown", "always_known", "always_unknown", "uniform"):
        result[f"mix/{reference}/total"] = (result[f"short/{reference}/total"]+result[f"long/{reference}/total"])/2
    return interval_axes(result, worlds)


@torch.no_grad()
def verify_physical(graph, rule, tape, missing, future):
    state, study = env.initial(graph,rule,tape,missing)
    rows = torch.arange(tape.lanes)
    if not (torch.equal(study[rows,missing,1], torch.zeros_like(study[:,0,1]))
            and (study[rows,1-missing,1].abs()==1).all()):
        raise ValueError("Invalid physical zero-taste or observed support history")
    cue = torch.ones_like(tape.decision_draw)
    decision = env.signal(rule.expand(tape.lanes),state,tape,cue)
    short, _ = env.branches(graph,rule,decision.state,tape,future,0)
    long, _ = env.branches(graph,rule,decision.state,tape,future,future.target.shape[1])
    assert torch.equal(short[:,:,0],long[:,:,0]) and torch.count_nonzero(short[:,:,1:]) == 0
    pairs = future.target.reshape(tape.lanes,-1,2)
    assert (pairs.sum(-1)==1).all()


def run(cohort, stage, worlds=None):
    plan = train.load_plan()
    standard = plan["holdout_worlds"]
    worlds = standard if worlds is None else worlds
    if stage not in plan["checkpoints"]:
        raise ValueError("Evaluate an ex ante checkpoint")
    seed = plan["holdout_seeds"][cohort] + (0 if worlds == standard else 4193)
    length = plan["paid_tail_length"]
    births = env.COHORTS[cohort]
    tape, missing = env.lesson.prepare(seed, (), 0, worlds, 2, omissions=True, heldout=True)
    future = env.opportunities(seed, (), 0, worlds, length)
    reports = {}
    hashes = {}
    code = {Path(p).name: env.digest(p) for p in (__file__,env.__file__,train.__file__,env.lesson.__file__,
                                                env.world.__file__,env.net.__file__,env.net.native.__file__)}
    stem = str((code,env.digest(train.PLAN),stage,seed,worlds))
    dest = env.DATA/"evaluation"/cohort/f"{worlds}-{stage}"
    dest.mkdir(parents=True,exist_ok=True)
    arms = ("frozen",) if stage == 0 else train.ARMS
    for arm in arms:
        grouped = []
        for birth in births:
            spec = (cohort,birth) if arm=="frozen" else (cohort,birth,stage,arm,env.DATA/"fit")
            checkpoint = env.checkpoint(*spec)
            sha = env.digest(checkpoint)
            signature = stem+sha
            file = dest/f"{arm}-{birth}.npz"
            if not file.exists():
                graph,rule = env.load(*spec)
                null = torch.arange(16)
                verify_physical(graph,rule,env.lesson.expand(tape,null),missing[null],future.subset(null))
                sets = []
                for j in range(0,tape.lanes,192):
                    ii = torch.arange(j,min(j+192,tape.lanes))
                    sets.append(env.evaluate(graph,rule,env.lesson.expand(tape,ii),missing[ii],future.subset(ii)))
                raw = {k:np.concatenate([x[k] for x in sets]) for k in sets[0]}
                env.storage._write_raw(file,**raw,signature=signature)
            with np.load(file,allow_pickle=False) as raw:
                if str(raw["signature"])!=signature:
                    raise ValueError(f"Unexpected evaluation identity {file}")
                grouped.append(measure(raw,worlds))
            hashes[str(checkpoint.relative_to(env.world.ROOT))]=sha
            hashes[str(file.relative_to(env.world.ROOT))]=env.digest(file)
            print(cohort,stage,arm,birth,flush=True)
        reports[arm] = {key:np.stack([root[key] for root in grouped]) for key in grouped[0]}
    if stage:
        reports["informative_minus_uninformative"] = {key:reports["informative"][key]-reports["uninformative"][key]
                                                       for key in reports["informative"]}
    resample = env.intervals.PairedCrossed(len(births),worlds,seed+331)
    result = dict(cohort=cohort,births=list(births),stage=stage,seed=seed,worlds=worlds,paid_tail_length=length,
                  paired_private_histories=4,plan_sha256=env.digest(train.PLAN),code=code,files=hashes,
                  summary={arm:resample.summary(scores) for arm,scores in reports.items()})
    tag = "" if worlds == standard else f"-development-{worlds}"
    env.storage._write_json(env.EVIDENCE/f"horizon-result-{cohort}-{stage}{tag}.json",result)


if __name__ == "__main__":
    cli=ArgumentParser(description=__doc__)
    cli.add_argument("--cohort",choices=env.COHORTS,required=True)
    cli.add_argument("--stages",type=int,nargs="+",required=True)
    cli.add_argument("--worlds",type=int)
    args=cli.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for st in args.stages:
        run(args.cohort,st,args.worlds)
