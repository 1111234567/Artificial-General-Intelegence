"""Causal single-receipt fork with an unpractised cue and a physical, unreinforced retest."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.seed_evolution.large_runner import _write
from . import research_audit as audit, world


TEACH = 16
LAGS = (0,1,8)


def lesson(tape):
    """Two equally unfamiliar raw cues; interference never contains either."""
    x = tape.cue[:, TEACH].copy()
    y = (x+1)%4
    cue = (x[:,None] + 2 + tape.cue%2)%4
    cue[:,TEACH] = x
    expected=tape.good_map[np.arange(tape.n)[:,None],tape.ctx,cue]
    return replace(tape,cue=cue,good_id=expected),x,y


def question(tape,ctx,cue,left_id):
    index = np.arange(tape.n)
    # Same geometry, motion perturbation and timing for every question and branch.
    return replace(tape,ctx=ctx[:,None],cue=cue[:,None],left_id=left_id[:,None],
                   good_id=tape.good_map[index,ctx,cue][:,None],
                   distances=tape.distances[:,TEACH:TEACH+1],wind=tape.wind[:,TEACH:TEACH+1])


def test_question(policy,tape):
    env=world.PulseWorld(tape,feedback="none")
    obs=env.observe()
    while obs is not None:
        obs=env.step(policy.act(obs))
    return env.result().contact_id[:,0]


def one_batch(factory,seeds,cue_ticks=16):
    from .sensation import BriefCue
    tape,x,y=lesson(world.make_tape(seeds,trials=TEACH+9))
    def subject():
        base=factory(seeds)
        return BriefCue(base,cue_ticks) if cue_ticks!=16 else base
    reference=world.rollout(subject(),tape,record=True)
    previous=reference.contact_id[:,TEACH]
    eligible=previous>=0
    expected=tape.good_id[:,TEACH]
    context=tape.ctx[:,TEACH]
    layout=tape.left_id[:,TEACH]
    private_truth=previous==expected
    questions={name:question(tape,c,k,side) for name,(c,k,side) in {
        "same":(context,x,1-layout), "other_context":(1-context,x,1-layout),
        "other_cue":(context,y,1-layout),"same_layout":(context,x,layout)}.items()}
    if not np.all(questions["same"].good_id[:,0] != questions["other_context"].good_id[:,0]):
        raise ValueError("Need conflicting A/B targets")
    prefix=subject()
    prefix.reset(tape.n)
    for t in range(TEACH):
        for tick in range(world.TICKS):
            obs=reference.obs[:,t,tick]
            if not np.array_equal(prefix.act(obs),reference.pulses[:,t,tick]):
                raise ValueError("Recreated policy is not deterministic for the paired prefix")
    result={}
    for sign in (1,-1):
        branch=deepcopy(prefix)
        triggered=np.zeros(tape.n,dtype=bool)
        for tick in range(world.TICKS):
            obs=reference.obs[:,TEACH,tick].copy()
            pulse=obs[:,world.CONTACT_ID].any(1)
            if np.any(triggered & pulse):
                raise ValueError("Duplicate receipt")
            triggered|=pulse
            obs[pulse,world.REWARD_POS]=sign>0
            obs[pulse,world.REWARD_NEG]=sign<0
            actions=branch.act(obs)
            if not np.array_equal(actions[~triggered],reference.pulses[~triggered,TEACH,tick]):
                raise ValueError("Arms differ before the intervened receipt")
        if not np.array_equal(triggered,eligible):
            raise ValueError("Some real contacts did not get a receipt")
        for lag in range(9):
            if lag in LAGS:
                for name,q in questions.items():
                    result[sign,lag,name]=test_question(deepcopy(branch),q)
            if lag<8:
                for tick in range(world.TICKS):
                    branch.act(reference.obs[:,TEACH+1+lag,tick])
    rows=[]
    for actor in range(tape.n):
        rows.append({"seed":seeds[actor],"eligible":bool(eligible[actor]),"previous":int(previous[actor]),
                     "true_positive":bool(private_truth[actor]), "targets":{k:int(q.good_id[actor,0]) for k,q in questions.items()},
                     "answers":{str(lag):{name:[int(result[1,lag,name][actor]),int(result[-1,lag,name][actor])]
                                          for name in questions} for lag in LAGS}})
    return rows


def report(rows):
    eligible=[row for row in rows if row["eligible"]]
    if not eligible:
        return {"eligible":0,"opportunities":len(rows)}
    def stats(values):
        return audit.estimates(values)
    lags={}
    for lag in LAGS:
        values={}
        for row in eligible:
            i=0 if row["true_positive"] else 1
            answers=row["answers"][str(lag)]
            row_effects={}
            for key,(positive,negative) in answers.items():
                row_effects[key+"_repeat_effect"]=int(positive==row["previous"])-int(negative==row["previous"])
                row_effects[key+"_true_gain"]=int(answers[key][i]==row["targets"][key])-int(answers[key][1-i]==row["targets"][key])
                row_effects[key+"_both_covered"]=int(positive>=0 and negative>=0)
            for label,j in (("true",i),("flipped",1-i)):
                row_effects["joint_"+label]=int(all(answers[k][j]==row["targets"][k] for k in ("same","other_context")))
            row_effects["joint_gain"]=row_effects["joint_true"]-row_effects["joint_flipped"]
            row_effects["context_specificity"]=row_effects["same_repeat_effect"]-row_effects["other_context_repeat_effect"]
            row_effects["cue_specificity"]=row_effects["same_repeat_effect"]-row_effects["other_cue_repeat_effect"]
            for key,v in row_effects.items():
                values.setdefault(key,[]).append(v)
        lags[str(lag)]={k:stats(v) for k,v in values.items()}
    return {"eligible":len(eligible),"opportunities":len(rows),"lags":lags}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject",choices=["spike","meta"],required=True)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--hash",default="")
    p.add_argument("--name",required=True)
    p.add_argument("--network-seed",type=int,default=3)
    p.add_argument("--split",default="replication")
    p.add_argument("--worlds",type=int,default=128)
    p.add_argument("--cue-ticks",type=int,default=16)
    args=p.parse_args()
    seeds=audit.split(args.split)[:args.worlds]
    if args.subject=="spike":
        from . import spike_policy as spike
        cfg=spike.SpikeConfig(use_residual=True)
        def factory(s):
            return spike.BatchSpikingPolicy(cfg,network_seed=args.network_seed,action_seeds=[
                np.random.SeedSequence([x,0x4C49464C]) for x in s])
        source=audit.fingerprint(spike.__file__)
    else:
        import torch
        from . import meta_model
        torch.set_num_threads(1)
        if audit.fingerprint(args.checkpoint)!=args.hash:
            raise ValueError("Incorrect frozen checkpoint")
        model=meta_model.load_policy(args.checkpoint,"cpu")
        def factory(_):
            return meta_model.RolloutAgent(model)
        source=args.hash
    rows=[]
    for i in range(0,len(seeds),32):
        rows.extend(one_batch(factory,seeds[i:i+32],args.cue_ticks))
    out={**report(rows),"policy_sha256":source,"cue_ticks":args.cue_ticks,"split":args.split,
         "network_seed":args.network_seed,"rows":rows,"source_sha256":audit.fingerprint(__file__)}
    _write(audit.OUT/(args.name+".json"),out)
    print(args.name,out["eligible"], {k:round(out["lags"]["8"][k]["mean"],3)
                                    for k in ("same_true_gain","joint_true","joint_gain","context_specificity")},flush=True)


if __name__=="__main__":
    main()
