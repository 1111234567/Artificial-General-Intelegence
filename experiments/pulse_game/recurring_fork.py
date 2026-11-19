"""Restore one component of the very same pre-query body after a long stream of real tastes."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import time
import numpy as np
from . import recurring_memory as memory, recurring_train as parent, recurring_world as task
from . import mixture_flash as flash, mixture_world as food, symmetry_policy as controller
from . import symmetry_policy_train as seedsrc, symmetry_world as game, research_audit as audit, world
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


class Monitor(memory.Forager):
    def __init__(self,p,g,seeds,mode,w,intervention):
        super().__init__(p,g,seeds,mode)
        self.change=intervention;self.teach=w.exposed
        self.query=int(np.flatnonzero(w.query[0])[0]);self.related=w.related_side[:,self.query]
        self.family=w.family.astype(float);self.sign=w.anchor_sign[:,self.query]
        self.d={}
    def reset(self,n):
        super().reset(n)
        self.here=np.arange(n);self.saved_slot=np.full(n,-1,int)
        self.anchor_k=np.zeros((n,4),np.float32);self.anchor_sign=np.zeros(n,np.float32)
        self.age0=np.zeros(n,int);self.evidence=None;self.preclock=0
    def store(self,ids,sd,sgn):
        super().store(ids,sd,sgn);slot=self.last_slot[ids];was=self.saved_slot[ids]
        if self.round==self.teach:
            self.saved_slot[ids]=slot
            self.anchor_k[ids]=self.previous_write[ids,sd]
            self.anchor_sign[ids]=sgn;self.age0[ids]=self.num_seen[ids]-1
        elif self.teach<self.round<self.query:
            deleted=(was==slot)&~self.last_merged[ids]
            self.saved_slot[ids[deleted]]=-1
    def act(self,obs):
        new=obs[:,world.TRIAL_START].all()
        if new:
            self.preclock=0
            if self.round+1==self.teach:self.evidence=self.master.copy()
            if self.round+1==self.query:
                idx=self.saved_slot.clip(min=0);ret=self.saved_slot>=0
                self.d.update(anchor_present=ret.astype(float),rule_direction=(self.master*self.family>0).astype(float),
                              prior_rule=(self.evidence*self.family>0).astype(float),
                              anchor_sign_intact=(ret & (self.signs[self.here,idx]*self.anchor_sign>0)).astype(float))
                if self.change in ("restore_rule","rule_and_age","all_three"):self.master[:]=self.evidence
                if self.change in ("no_age","rule_and_age","anchor_and_age","all_three"):
                    self.model={**self.model,"age_cost":0.}
                if self.change in ("restore_anchor","anchor_and_age","all_three","distractor_control"):
                    # Same slot budget. A factual anchor intervention restores the
                    # actual recorded address, sign and original timestamp. The
                    # sham takes the neighboring independent life's actual anchor.
                    slot=np.where(ret,self.saved_slot,self.recency.argmin(1))
                    donor=self.here^1 if self.change=="distractor_control" else self.here
                    self.addresses[self.here,slot]=self.anchor_k[donor]
                    self.signs[self.here,slot]=self.anchor_sign[donor]
                    self.recency[self.here,slot]=self.age0
                    self.count=np.maximum(self.count,slot+1)
        pulses=super().act(obs)
        if self.round==self.query and self.preclock==4:
            gain=np.tanh(self.model["scalars"][4]*self.master)
            r=self.previous_value
            self.d.update(related_recall=r[self.here,self.related].astype(float),
                          comparison_right=((r[self.here,self.related]-r[self.here,1-self.related])*gain*
                                            self.sign*self.family>0).astype(float))
        self.preclock+=1
        return pulses


def one(spec):
    name,args,ss,mode=spec;folder=audit.OUT/"recurring-identical-prefix";folder.mkdir(exist_ok=True)
    out=folder/(name+f"-w{mode}-d{args['distinct']}-r{int(args.get('recurring',True))}.json")
    if out.exists():return
    params,body,birth,src=parent.parent(name)
    w,_=task.make(ss,**args,nonce=981224)
    f=json.loads((OUT/"recurring-write-v1"/(name+f"-writer{mode}")/"frozen.json").read_text())
    phen=controller.materialize(params,birth,src["settings"]["mode"])
    raw={};prefix=None
    for tag in ("true","no_age","restore_rule","restore_anchor","rule_and_age","anchor_and_age","all_three","distractor_control"):
        a=Monitor(phen,f["genes"],ss+ss,mode,w,tag);res=flash.rollout(a,w.menu)
        before=res.contact_tick[:,:a.query]
        if prefix is None:prefix=(before.copy(),res.contact_side[:,:a.query].copy(),res.reward[:,:a.query].copy())
        if not all(np.array_equal(x,y) for x,y in zip(prefix,(before,res.contact_side[:,:a.query],res.reward[:,:a.query]),strict=True)):
            raise ValueError("Pre-query history changed across interventions")
        met=game.metrics(w,res);data={**a.d,"success":met["first"],"family":w.family,"actual_anchor_sign":a.anchor_sign}
        raw[tag]={k:np.asarray(v).reshape(2,len(ss)).tolist() for k,v in data.items()}
    _write(out,dict(seed=ss,parent=name,writer=mode,protocol=args,query=int(a.query),cases=raw));print("checked",out.stem,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--workers",type=int,default=3)
    args=p.parse_args()
    ss=seedsrc.evalseeds(76942113,144)
    names=("public-learned-s7243--full-e8719","public-learned-s4187--full-e9173","public-learned-s9679--full-e5879")
    combos=[(n,dict(gap=32,distinct=k,recurring=r,query="anchor",probes=1),ss,m)
            for n in names for k,r in ((6,True),(16,True),(8,False)) for m in (1,2)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for job in as_completed([pool.submit(one,j) for j in combos]):job.result()


if __name__=="__main__":main()
