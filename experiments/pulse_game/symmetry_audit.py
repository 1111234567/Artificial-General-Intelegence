"""Public exemplar bound and causal probes of which learning rule a life has selected."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import time
import numpy as np
import torch

from . import mixture_flash as flash, mixture_world as foods, symmetry_world as game, symmetry_train as training
from . import symmetry_deploy as neurons, research_audit as audit, world
from .attractor_search import OUT
from .paired_choice import first_contact_pairs
from experiments.seed_evolution.large_runner import _write


class PublicMemory:
    """Knows the possible odor transformation. Its sign must be inferred from observed tastes."""
    def __init__(self,seeds,fixed=None):
        self.seeds=seeds;self.fixed=fixed
    def reset(self,n):
        self.history=[[] for _ in range(n)];self.gates=np.zeros(n,np.float32)
        self.last=np.zeros((n,2,4),np.float32);self.hab=np.zeros(n,int);self.age=-1
        self.noise=[np.random.default_rng(np.random.SeedSequence([s,930135])) for s in self.seeds]
        self.momentum=np.zeros(n,int);self.got=np.zeros(n,bool);self.have_ctx=np.zeros(n,bool)
    def act(self,obs):
        new=obs[:,world.TRIAL_START].all()
        if new:
            self.age+=1;self.got[:]=False;self.have_ctx[:]=False
            self.momentum=np.array([r.choice([-1,1]) for r in self.noise])
        ctx=obs[:,foods.CTX];show=ctx.any(1)
        self.hab[show]=ctx[show].argmax(1);self.have_ctx|=show
        x=2*obs[:,foods.LEFT.start:foods.RIGHT.stop].reshape(len(obs),2,4)-1
        sense=x.any((1,2));self.last[sense]=x[sense];self.got|=sense
        received=obs[:,world.REWARD_POS]-obs[:,world.REWARD_NEG]
        for i in np.flatnonzero((received!=0)&obs[:,world.CONTACT_ID].any(1)):
            side=obs[i,world.CONTACT_ID].argmax()
            smell=self.last[i,side];hab=self.hab[i]
            matches=[(float(-np.dot(smell,a)),s) for a,s,c in self.history[i] if c==hab]
            if matches and max(matches)[0]>.80:
                self.gates[i]+=max(matches)[1]*received[i]
            self.history[i].append((smell.copy(),received[i],hab))
        for i in np.flatnonzero((sense|show)&self.got&self.have_ctx):
            memory=[(a,s) for a,s,c in self.history[i] if c==self.hab[i]]
            if not memory:continue
            recent,y=memory[-1]
            related=(-self.last[i]@recent).argmax()
            which=np.sign(self.gates[i]) if self.fixed is None else self.fixed
            if which:
                pick=related if y*which>0 else 1-related
                self.momentum[i]=2*pick-1
        return np.eye(2,dtype=np.float32)[(self.momentum>0).astype(int)]*obs[:,world.ACTIVE,None]


class Intervention:
    def __init__(self,agent,kind,cutoff):
        self.agent=agent;self.kind=kind;self.at=cutoff;self.t=-1
    def reset(self,n):self.t=-1;self.agent.reset(n)
    def act(self,obs):
        if obs[:,world.TRIAL_START].all():
            self.t+=1
            if self.t==self.at:
                n=len(obs)//2
                if self.kind=="swap_confidence":self.agent.logit[:]=np.roll(self.agent.logit,n,axis=0)
                elif self.kind=="erase_confidence":self.agent.logit.fill(0.)
                elif self.kind=="swap_weights":self.agent.fast[:]=np.roll(self.agent.fast,n,axis=0)
        if self.kind=="no_anchor" and self.t==self.at:
            obs=obs.copy();obs[:,world.REWARD_POS:world.REWARD_NEG+1]=0.
        return self.agent.act(obs)


def joint(seeds,*,pairs,probes,span,noise,jitter,matched,mask=False):
    n=len(seeds)
    w=game.make(seeds+seeds,pairs=pairs,probes=probes,interleave=span,noise=noise,jitter=jitter,
                family=np.r_[np.ones(n,int),-np.ones(n,int)],balanced_stream=matched)
    cut=w.exposed
    if not (np.array_equal(w.menu.chemicals[:n,cut:],w.menu.chemicals[n:,cut:]) and
            np.array_equal(w.menu.realized[:n,cut],w.menu.realized[n:,cut]) and
            np.array_equal(w.menu.probability[:n,cut+1::2],1-w.menu.probability[n:,cut+1::2])):
        raise ValueError("After teaching, branches must share B observations with opposite B goals")
    if matched and not np.array_equal(w.menu.realized[:n,:cut],w.menu.realized[n:,:cut]):
        raise ValueError("Teaching signs differ despite the stream-matched protocol")
    if mask:
        # Preserve exactly the same A sign stream; also present identical A
        # odor history in both branches. The first B action then cannot reveal
        # the hidden family. All goals stay private until after that action.
        w.menu.chemicals[n:,:cut]=w.menu.chemicals[:n,:cut]
        w.menu.realized[n:,:cut]=w.menu.realized[:n,:cut]
        w.menu.probability[n:,:cut]=w.menu.probability[:n,:cut]
    return w


def control(model,path,change,seeds,w):
    if model.startswith("public"):
        fixed=None if model=="public" else (1 if model=="public_same" else -1)
        return Intervention(PublicMemory(seeds+seeds,fixed),change,w.exposed)
    cut="cross" if change=="cut_cross" else "all" if change=="cut_all" else None
    return Intervention(neurons.Forager(training.load(path,cut),seeds+seeds,change=change),change,w.exposed)


def test(spec):
    model,path,seeds,settings,branches,dest=spec
    torch.set_num_threads(1)
    conf_name,params=settings
    name=f"{model}-{conf_name}"
    saved=dest/(name+".json")
    if saved.is_file():return
    prefix=joint(seeds,**params)
    out={}
    n=len(seeds)
    for tag in branches:
        w=joint(seeds,**params,mask=True) if tag=="matched_history" else prefix
        agent=control(model,path,tag,seeds,w)
        result=flash.rollout(agent,w.menu)
        met=game.metrics(w,result)
        def split(key):return met[key][:n],met[key][n:]
        a,b=split("first")
        pair=first_contact_pairs(w,result)
        if tag=="matched_history" and (not np.array_equal(pair["first_side_even"],pair["first_side_odd"]) or (a*b).any()):
            raise RuntimeError("First-choice causal null violated")
        values=dict(first=(a+b)/2,even=a,odd=b,both=a*b,**pair,
                    all=(met["all"][:n]+met["all"][n:])*.5,
                    contact=(met["contact"][:n]+met["contact"][n:])*.5)
        if isinstance(agent.agent,neurons.Forager):
            # Expert numbering can flip across independent network births.
            gates=agent.agent.records[w.exposed]
            values.update(evidence_even=gates[:n,0],evidence_odd=gates[n:,0])
        out[tag]={k:v.tolist() for k,v in values.items()}
    _write(saved,dict(model=model,protocol=conf_name,seeds=seeds,configuration=params,
                        checkpoint=audit.fingerprint(path) if path else None,rows=out,
                        pair_choice_protocol="both_first_contacts_required"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder",default="symmetry-circuit-v2");p.add_argument("--workers",type=int,default=3)
    p.add_argument("--worlds",type=int,default=256);p.add_argument("--steps",type=int,default=2200)
    p.add_argument("--models",nargs="+");p.add_argument("--tag",default="native")
    args=p.parse_args();torch.set_num_threads(1)
    root=OUT/args.folder;dest=audit.OUT/(args.folder+"-"+args.tag);dest.mkdir(exist_ok=True)
    seeds=[int(np.random.SeedSequence([s,89453171]).generate_state(1)[0]) for s in audit.split("transfer")[:args.worlds]]
    expected=args.models or [f"rotations-{m}-e{1 if m=='single' else 2}-s{s}" for m in ("evidence","uniform","single")
                             for s in (6371,2377,9091)]
    if args.models is None:expected+=["public","public_same","public_opposite"]
    atlas=[("paired",dict(pairs=12,probes=4,span=4,noise=.04,jitter=.08,matched=True)),
           ("remote",dict(pairs=12,probes=4,span=12,noise=.04,jitter=.15,matched=True)),
           ("few",dict(pairs=4,probes=4,span=4,noise=.04,jitter=.08,matched=True)),
           ("one_back",dict(pairs=12,probes=4,span=1,noise=.04,jitter=.08,matched=False))]
    done=set()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while len(done)<len(expected):
            ready=[]
            for name in expected:
                if name in done:continue
                hist=root/(name+"-history.json")
                if name.startswith("public") or (hist.is_file() and json.loads(hist.read_text())[-1]["step"]>=args.steps):ready.append(name)
            work=[]
            for name in ready:
                path=None if name.startswith("public") else root/(name+".pt")
                for setting in atlas:
                    branches=["true"]
                    if setting[0]=="paired":
                        branches+=["matched_history","no_anchor"]
                        if "evidence" in name:branches+=["uniform","invert_evidence","swap_confidence","erase_confidence","swap_weights","no_write","cut_cross","cut_all"]
                    work.append((name,path,seeds,setting,branches,dest))
            for f in as_completed([pool.submit(test,job) for job in work]):f.result()
            done.update(ready)
            if ready:print("checked",*ready,flush=True)
            if len(done)<len(expected):time.sleep(10)
    _write(dest/"done.json",dict(models=sorted(done),worlds=len(seeds)))


if __name__=="__main__":main()
