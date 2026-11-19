"""Take the four-alternative LIF association rule back to the sealed original request game.

The private map and target enter only the original environment and scoring code.
The candidate sees its two original 28-channel ports, including a one-tick
cue and the environmental signal reporting the previous action's outcome.
"""

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np
import torch

from experiments.seed_evolution.large_runner import _write
from experiments.undirected import local_rule_evaluator as original_eval,local_rule_world as original_world
from experiments.undirected import uncapped_private_symbol_world as private
from experiments.undirected.evolved_plasticity_world import NEW,READY,CONTEXT,RAW,FIRST_FINISH,_receipt_table
from experiments.undirected.uncapped_reward_precision import _ACK,_REJ
from experiments.undirected.uncapped_answer_feedback import _RecordPrimaryActions
from . import ring_spikes,ring_world,research_audit


@dataclass(frozen=True)
class Options:
    key:str="onset"
    actuator:str="once"
    feedback:str="true"
    sensory_decay:float=.87
    adaptation:float=0.
    fatigue_decay:float=.92
    ready_gain:float=.92
    value_gain:float=.36
    motor_leak:float=.55
    allow_retry:bool=False
    no_recurrence:bool=False


class BridgeAgent:
    def __init__(self,config,options,seeds,net):
        self.cfg=config;self.opt=options
        self.p=ring_spikes.Forager(replace(config,receptor_decay=options.sensory_decay),
                                   net,seeds,no_recurrence=options.no_recurrence)
        self.p.reset(len(seeds));p=self.p
        self.v=np.zeros((len(seeds),4),np.float32)
        self.motor=np.zeros_like(self.v)
        self.fatigue=np.zeros_like(p.h)
        self.tag=np.zeros_like(p.last_key)
        self.tried=np.zeros(len(seeds),bool)
        self.rngs=[np.random.default_rng(np.random.SeedSequence([s,0x1A592])) for s in seeds]
        self.rand=np.empty((0,len(seeds),4),np.float32);self.tick=0
        self.n=len(seeds)

    def _tie(self):
        i=self.tick%64
        if not i:self.rand=np.stack([r.random((64,4),dtype=np.float32) for r in self.rngs],1)
        return self.rand[i]

    def act(self,observation):
        raw=observation[0].numpy();p=self.p;opt=self.opt
        new=raw[:,NEW]!=0;ready=raw[:,READY]!=0
        sensory=np.zeros((self.n,ring_world.OBS),np.float32)
        if opt.feedback!="none":
            if opt.feedback=="true":
                sensory[:,ring_world.POSITIVE]=raw[:,_ACK];sensory[:,ring_world.NEGATIVE]=raw[:,_REJ]
            elif opt.feedback=="invert":
                sensory[:,ring_world.POSITIVE]=raw[:,_REJ];sensory[:,ring_world.NEGATIVE]=raw[:,_ACK]
            else:raise ValueError("unknown signed outcome")
        p._learn(sensory)
        sensory[:,:2]=raw[:,CONTEXT.start:CONTEXT.start+2];sensory[:,2:6]=raw[:,RAW]
        p.sense*=opt.sensory_decay
        np.maximum(p.sense,sensory[:,:6],out=p.sense)
        # The recurrent state is continuous across all requests. Reading an
        # onset sample is an explicit timing prior and can be disabled.
        p.h=np.maximum(p.cfg.leak*p.h+p.sense@p.win+p.s@p.rec+p.bias,-2.)
        p.s=(p.h>=(p.thresh+self.fatigue)).astype(np.float32);p.h*=1.-p.s
        self.fatigue*=opt.fatigue_decay;self.fatigue+=opt.adaptation*p.s
        p.z*=p.cfg.history;p.z+=(1-p.cfg.history)*p.s
        p.mean+=.008*(p.z-p.mean)
        if opt.key=="onset":
            activity=p.s+p.h
            self.tag[new]=(activity[new]-.18)@p.project
            key=self.tag
        elif opt.key=="continuous":
            key=(p.z-p.mean)@p.project
        else:raise ValueError("unknown persistence rule")
        key=key/np.sqrt(.01+(key*key).sum(1,keepdims=True))
        values=np.einsum("bfk,bk->bf",p.w,key)
        random=self._tie()
        trial_values=values+1e-5*random
        self.tried[new]=False
        if opt.actuator=="once":
            fired=ready & ~self.tried
            choice=trial_values.argmax(1)
        elif opt.actuator=="lif":
            self.v*=opt.motor_leak
            self.v+=.04+opt.ready_gain*ready[:,None]+opt.value_gain*np.tanh(values)+.035*(random-.5)
            np.maximum(self.v,-1.,out=self.v)
            choice=self.v.argmax(1)
            fired=self.v[np.arange(self.n),choice]>=1.
            # A winner's spike recruits lateral inhibition at the motor port.
            self.v[fired,:]-=.68
            self.v[np.flatnonzero(fired),choice[fired]]=0.
        else:raise ValueError("unknown action readout")
        if opt.allow_retry and opt.actuator=="once":
            fired|=ready & (raw[:,_REJ]!=0)
        self.tried|=fired
        self.motor.fill(0);self.motor[np.flatnonzero(fired),choice[fired]]=1
        p.last_odor[:]=self.motor*(ready[:,None] if opt.actuator=="lif" else 1)
        p.last_key[:]=key;p.last_pred=values[np.arange(self.n),choice]
        self.tick+=1
        result=np.zeros((1,self.n,2),np.int64);result[0,fired,0]=FIRST_FINISH+choice[fired]
        return torch.from_numpy(result)


def batch(config,opt,net,seeds):
    tapes=[original_world.schedule(seed=s,actors=8,visits=16,device="cpu") for s in seeds]
    dims=dict(public=2,stage=2,raw=1,context=1,target=1,maps=0)
    tape=replace(tapes[0],**{k:torch.cat([getattr(t,k) for t in tapes],dim=d) for k,d in dims.items()})
    actor_seeds=[(s*8+i) for s in seeds for i in range(8)]
    policy=BridgeAgent(config,opt,actor_seeds,net)
    observer=_RecordPrimaryActions(policy,tape,1)
    base=private.rollout_with_table(_receipt_table("cpu").half())(observer,tape,1)
    ret=[]
    for i,seed in enumerate(seeds):
        t=tapes[i];actions=observer.actions[:,:,:,8*i:8*i+8]
        metrics=original_eval.trace_statistics(actions,t,original_world.partners(t.raw,t.context))
        q={k:v.item() for k,v in metrics.items()}
        q["selection_score"]=q["strict"]+.1*(q["completion"]-q["strict"])+(q["pairs_both_strict"]-.1*q["post_answer_actions"])/512
        q["seed"]=seed
        ret.append(q)
    if any(abs(np.mean([v[k] for v in ret])-base[k].item())>1e-7 for k in ("strict","completion")):
        raise ValueError("Primary trace and original world metrics disagree")
    return ret


def main():
    from dataclasses import asdict
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step",choices=["discover","confirm"],required=True)
    ap.add_argument("--ring-file",type=Path,default=Path("data/pulse-game/ring-value-search-v1/frozen.json"))
    args=ap.parse_args()
    torch.set_num_threads(1)
    cfg=ring_spikes.Config(**json.loads(args.ring_file.read_text())["best"]["config"])
    out=research_audit.OUT
    tag=out/"legacy-bridge-selected.json"
    if args.step=="discover":
        choices=[Options(key=key,actuator=actuator,sensory_decay=decay,ready_gain=rg)
                 for key in ("onset","continuous") for actuator in ("once","lif")
                 for decay in (0.,.65,.86,.93,.98) for rg in ((.92,1.10) if actuator=="lif" else (.92,))]
        seeds=[106_105_000+i for i in range(6)]
        rows=[]
        for opt in choices:
            metrics=batch(cfg,opt,13,seeds)
            rows.append({"options":asdict(opt),"strict":float(np.mean([r["strict"] for r in metrics])),
                         "selection_score":float(np.mean([r["selection_score"] for r in metrics]))})
        finalists={}
        for key in ("onset","continuous"):
            for actuator in ("once","lif"):
                eligible=[r for r in rows if r["options"]["key"]==key and r["options"]["actuator"]==actuator]
                finalists[key+"_"+actuator]=max(eligible,key=lambda r:r["selection_score"])
        _write(tag,{"ring_sha256":research_audit.fingerprint(args.ring_file),
                    "seeds":seeds,"net":13,"all":rows,"selected":finalists,
                    "source_sha256":research_audit.fingerprint(__file__)})
        print(finalists,flush=True)
        return
    settings=json.loads(tag.read_text())
    if settings["source_sha256"]!=research_audit.fingerprint(__file__):raise ValueError("source changed")
    orig=Path("data/undirected/2026-11-19-local-rule-v1/audits/0128.json")
    seeds=json.loads(orig.read_text())["seeds"]
    report={"protocol":str(orig),"seeds":seeds,"development":settings["selected"],"arms":{}}
    for name,row in settings["selected"].items():
        selected=Options(**row["options"])
        variants={"true":(cfg,selected),"hidden":(cfg,replace(selected,feedback="none")),
                  "flipped":(cfg,replace(selected,feedback="invert")),
                  "freeze_weights":(replace(cfg,learning_rate=0),selected),
                  "cut_recurrence":(cfg,replace(selected,no_recurrence=True))}
        for condition,(c,opt) in variants.items():
            rows=[]
            for net in (3,19,37,53):
                for i in range(0,len(seeds),8):
                    rows.extend([{**r,"network_seed":net} for r in batch(c,opt,net,seeds[i:i+8])])
            report["arms"].setdefault(name,{})[condition]=rows
            _write(out/"legacy-bridge-confirmed.json",report)
            print(name,condition,np.mean([r["strict"] for r in rows]),np.mean([r["pair_strict_rate"] for r in rows]),
                  flush=True)


if __name__=="__main__":
    main()
