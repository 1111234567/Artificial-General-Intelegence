"""Clamp only the teaching motion and test the agent's own first choice in both private worlds."""
import json
import numpy as np
from . import mixture_flash as flash, mixture_metric, symmetry_audit as cases, symmetry_world as game
from . import recurring_train as selection, symmetry_policy as policy, symmetry_memory_deploy as native
from . import research_audit as audit, world
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


class CommonMotion:
    def __init__(self,brain,seeds,tr,cut):
        self.actor=brain;self.explorer=mixture_metric.Explorer(seeds,tr);self.cut=cut;self.j=-1
    def reset(self,n):self.actor.reset(n);self.explorer.reset(n);self.j=-1
    def act(self,obs):
        if obs[:,world.TRIAL_START].all():self.j+=1
        actual=self.actor.act(obs);common=self.explorer.act(obs)
        if self.j<self.cut:return common
        return actual


def main():
    seeds=np.random.default_rng(88500411).integers(10_000_000,470_000_000,256).tolist()
    out={};n=len(seeds)
    w=cases.joint(seeds,pairs=12,probes=1,span=4,noise=.04,jitter=.08,matched=True)
    for path in sorted((OUT/"symmetry-policy-v1").glob("*--full-*/frozen.json")):
        name=path.parent.name;p,physical,v,source=selection.parent(name);params=policy.materialize(p,v,"full")
        out[name]={}
        for clamped in (False,True):
            brain=native.Forager(params,seeds+seeds,boundary=w.exposed)
            agent=CommonMotion(brain,seeds+seeds,w.menu.body.trials,w.exposed) if clamped else brain
            r=flash.rollout(agent,w.menu)
            y=r.observed_reward[:,:w.exposed];same=(y[:n]==y[n:])
            if clamped and not same.all():raise ValueError("Matched physical receipts failed")
            a,b=np.split(game.metrics(w,r)["first"],2)
            out[name]["matched_motion" if clamped else "native_motion"]=dict(
                  first=((a+b)*.5).tolist(),both=(a*b).tolist(),event_sign_agreement=float((y[:n].sum(2)==y[n:].sum(2)).mean()),
                  all_tick_values_agree=float(same.mean()),whole_event_waveform_agreement=float(same.all(2).mean()),
                  entire_teaching_history_identical=float(same.all((1,2)).mean()))
        print(name,{k:(np.mean(d["first"]),d["whole_event_waveform_agreement"]) for k,d in out[name].items()},flush=True)
    _write(audit.OUT/"matched-actual-teaching-motion.json",dict(seeds=seeds,rows=out))


if __name__=="__main__":main()
