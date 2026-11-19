"""Independent paired validation of the frozen four-choice ring forager."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from experiments.seed_evolution.large_runner import _write
from . import research_audit as audit, ring_spikes as snn, ring_world as world


def life_metrics(t,r):
    pair_n,pair_d=audit.pair_counts(t.ctx,t.cue,r.first_correct)
    return {k:v.tolist() for k,v in {
        "early":r.first_correct[:,:8].mean(1),"all":r.first_correct.mean(1),
        "late":r.first_correct[:,-32:].mean(1),"late_solved":r.solved[:,-32:].mean(1),
        "late_bites":r.bites[:,-32:].mean(1),"coverage":(r.first>=0).mean(1),
        "gain":r.first_correct[:,-32:].mean(1)-r.first_correct[:,:8].mean(1),
        "net_reward":(r.positive-r.negative).mean(1),
        "first_latency_or_timeout":np.where(r.first_tick>=0,r.first_tick,world.TICKS).mean(1),
        "pair_rate":pair_n/pair_d}.items()}


class TestReadout:
    def __init__(self,policy,competition):
        self.policy,self.competition=policy,competition

    def reset(self,n):
        self.policy.reset(n)

    def act(self,obs):
        if np.all(obs[:,world.NEW]) and self.policy.age==63:
            self.policy.cfg=replace(self.policy.cfg,competition=self.competition,learning_rate=0.)
        return self.policy.act(obs)


class AtBite:
    def __init__(self,policy,signs):
        self.policy,self.signs=policy,signs

    def reset(self,n):
        self.policy.reset(n);self.j=-1

    def act(self,obs):
        if np.all(obs[:,world.NEW]):self.j+=1
        event=(obs[:,world.POSITIVE]+obs[:,world.NEGATIVE])>0
        frame=obs.copy()
        frame[:,world.POSITIVE]=event & (self.signs[:,self.j]>0)
        frame[:,world.NEGATIVE]=event & (self.signs[:,self.j]<0)
        return self.policy.act(frame)


def evaluate(t,cfg,seed,alternatives):
    def policy(c=cfg,**kw):return snn.Forager(c,seed,t.seeds,**kw)
    primary=world.rollout(policy(),t)
    donor=np.arange(t.n)^1
    donor_sign=primary.positive[donor]-primary.negative[donor]
    runs={"true":primary,
          "no_feedback":world.rollout(policy(),t,feedback="none"),
          "time_yoke":world.rollout(policy(),t,feedback="yoked",yoke=primary.waveform[donor]),
          "contact_yoke":world.rollout(AtBite(policy(),donor_sign),t),
          "no_writes":world.rollout(policy(replace(cfg,learning_rate=0.)),t),
          "recurrence_cut":world.rollout(policy(no_recurrence=True),t),
          "multiple_bites":world.rollout(policy(),t,retry=True),
          "cue_one_tick":world.rollout(policy(),t,cue_ticks=1),
          "freeze_at_64":world.rollout(TestReadout(policy(),cfg.competition),t),
          "freeze_without_competition":world.rollout(TestReadout(policy(),0.),t)}
    for label,c in alternatives.items():runs[label]=world.rollout(policy(c),t)
    for label in ("freeze_at_64","freeze_without_competition"):
        if not (np.array_equal(runs[label].first[:,:64],primary.first[:,:64])
                and np.array_equal(runs[label].first_tick[:,:64],primary.first_tick[:,:64])):
            raise ValueError("The common training histories diverged before the readout fork")
    # Completely resample private outcomes at each trial: feedback has no reusable information.
    random_goals=np.stack([np.random.default_rng(np.random.SeedSequence([s,90817])).integers(4,size=t.trials,
                          dtype=np.int8) for s in t.seeds])
    random_t=replace(t,goal=random_goals)
    data={k:life_metrics(t,v) for k,v in runs.items()}
    data["unpredictable"]=life_metrics(random_t,world.rollout(policy(),random_t))
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen",type=Path,default=Path("data/pulse-game/ring-value-search-v1/frozen.json"))
    p.add_argument("--hash",required=True)
    p.add_argument("--split",default="hard_task")
    p.add_argument("--worlds",type=int,default=128)
    p.add_argument("--seeds",type=int,nargs="+",default=[3,19,37,53])
    p.add_argument("--out",type=Path,default=audit.OUT/"ring-confirmed.json")
    args=p.parse_args()
    if audit.fingerprint(args.frozen)!=args.hash:raise ValueError("Candidate not frozen")
    selected=json.loads(args.frozen.read_text())
    if audit.fingerprint(snn.__file__)!=selected["policy_sha256"] or audit.fingerprint(world.__file__)!=selected["world_sha256"]:
        raise ValueError("Implementation no longer matches selected candidate")
    cfg=snn.Config(**selected["best"]["config"])
    worlds=audit.split(args.split)[:args.worlds]
    t=world.make_tape(worlds)
    alts={}
    for method in ("delta","rls"):
        for relative in (True,False):
            fits=[v for v in selected["finalists"] if v["config"]["method"]==method and bool(v["config"]["competition"])==relative]
            q=max(fits,key=lambda x:x["validation"]["late"])["config"]
            if q!=selected["best"]["config"]:
                alts[method+("_competition" if relative else "_absolute")]=snn.Config(**q)
    report={"seeds":worlds,"hash":args.hash,"subject":selected["best"]["config"],"trials":t.trials,
            "baselines":{},"graphs":{},"source_sha256":audit.fingerprint(__file__)}
    for name in ("first","oracle","eliminate"):
        for retry in (True,False):
            res=world.rollout(world.PublicForager(name,t.goal if name=="oracle" else None),t,retry=retry)
            report["baselines"][name+("_retry" if retry else "_once")]={k:audit.estimates(v) for k,v in life_metrics(t,res).items()}
    for net in args.seeds:
        data=evaluate(t,cfg,net,alts)
        summ={label:{k:audit.estimates(v,clusters=2) for k,v in d.items()} for label,d in data.items()}
        pairs={label:audit.estimates(np.subtract(data["true"]["late"],d["late"]),clusters=2)
               for label,d in data.items() if label!="true"}
        readonly=audit.estimates(np.subtract(data["freeze_at_64"]["late"],
                                             data["freeze_without_competition"]["late"]),clusters=2)
        report["graphs"][str(net)]={"summary":summ,"first_late_true_minus":pairs,"frozen_readout_effect":readonly,
                                    "per_world":data}
        _write(args.out,report)
        print("ring",net,{l:round(v["late"]["mean"],4) for l,v in summ.items()},flush=True)


if __name__=="__main__":
    main()
