"""Audit receipt frequency, source trace and action coverage on the original sealed task."""

from dataclasses import asdict,replace
import json
from pathlib import Path

import numpy as np
import torch

from experiments.seed_evolution.large_runner import _write
from experiments.undirected import local_rule_evaluator as evaluator,local_rule_world as worlds
from experiments.undirected import uncapped_private_symbol_world as private
from experiments.undirected.evolved_plasticity_world import NEW,READY,CONTEXT,RAW,FIRST_FINISH,_receipt_table
from experiments.undirected.uncapped_answer_feedback import _RecordPrimaryActions
from experiments.undirected.uncapped_reward_precision import _ACK,_REJ
from . import research_audit as audit, ring_spikes
from .legacy_bridge import BridgeAgent, Options


class Intervention:
    """Intervene on the incoming sign/eligibility; physical action and retirement stay native."""

    def __init__(self,agent,mode,seeds):
        self.agent,self.mode=agent,mode
        self.trial=np.full(len(seeds),-1,np.int32)
        self.tried=np.zeros(len(seeds),bool)
        self.last_is_first=np.zeros(len(seeds),bool)
        self.fake=np.stack([np.random.default_rng(np.random.SeedSequence([s,0x928E1])).integers(
            4,size=(2,4)) for s in seeds])
        self.last_expected=np.zeros(len(seeds),np.int8)
        self.last_choice=np.full(len(seeds),-1,np.int8)
        self.context=np.zeros(len(seeds),np.int8);self.cue=self.context.copy()

    def act(self,obs):
        raw=obs[0].numpy();feedback=(raw[:,_ACK]+raw[:,_REJ])!=0
        if self.mode=="first_only":
            self.agent.p.last_odor[~self.last_is_first]=0
        elif self.mode=="wrong_address":
            self.agent.p.last_key=np.roll(self.agent.p.last_key,7,axis=1)
        if self.mode in ("shadow","positive_only","negative_only","silent_after32"):
            obs=obs.clone()
            if self.mode=="shadow":
                good=self.last_choice==self.last_expected
                obs[0,:,_ACK]=torch.from_numpy((feedback&good).astype(np.float32))
                obs[0,:,_REJ]=torch.from_numpy((feedback&~good).astype(np.float32))
            elif self.mode=="positive_only":obs[0,:,_REJ]=0
            elif self.mode=="negative_only":obs[0,:,_ACK]=0
            else:
                rows=torch.from_numpy(self.trial>=32)
                obs[0,rows,_ACK]=0;obs[0,rows,_REJ]=0
        new=raw[:,NEW]!=0
        self.trial[new]+=1;self.tried[new]=False
        self.context[new]=np.argmax(raw[new,CONTEXT.start:CONTEXT.start+2],axis=1)
        self.cue[new]=np.argmax(raw[new,RAW],axis=1)
        actions=self.agent.act(obs); choice=actions[0,:,0].numpy()-FIRST_FINISH
        valid=(choice>=0)&(raw[:,READY]!=0)
        self.last_is_first=valid&~self.tried
        self.tried|=valid
        self.last_choice=choice.copy()
        self.last_expected=self.fake[np.arange(len(choice)),self.context,self.cue]
        return actions


class PublicMemory:
    """Public-only table upper reference for a given reply budget."""

    def __init__(self,n,repeat):
        self.repeat=repeat;self.val=np.ones((n,2,4,4),bool);self.index=np.arange(n)
        self.ctx=np.zeros(n,np.int8);self.cue=self.ctx.copy();self.last=np.full(n,-1,np.int8)
        self.tried=np.zeros(n,bool);self.count=np.zeros(n,np.int64)

    def act(self,obs):
        f=obs[0].numpy();valid=self.last>=0
        yes=valid & (f[:,_ACK]!=0);no=valid & (f[:,_REJ]!=0)
        self.val[self.index[yes],self.ctx[yes],self.cue[yes]]=False
        self.val[self.index[yes],self.ctx[yes],self.cue[yes],self.last[yes]]=True
        self.val[self.index[no],self.ctx[no],self.cue[no],self.last[no]]=False
        self.count+=yes|no
        new=f[:,NEW]!=0
        self.ctx[new]=f[new,CONTEXT.start:CONTEXT.start+2].argmax(1)
        self.cue[new]=f[new,RAW].argmax(1);self.tried[new]=False
        choose=(f[:,READY]!=0) & (~self.tried|self.repeat)
        values=self.val[self.index,self.ctx,self.cue]
        self.last=np.where(choose,values.argmax(1),-1)
        self.tried|=choose
        out=np.zeros((1,len(f),2),np.int64);out[0,choose,0]=self.last[choose]+FIRST_FINISH
        return torch.from_numpy(out)


def run_batch(cfg,opt,seeds,net,mode):
    tapes=[worlds.schedule(seed=s,actors=8,visits=16,device="cpu") for s in seeds]
    dims=dict(public=2,stage=2,raw=1,context=1,target=1,maps=0)
    tape=replace(tapes[0],**{k:torch.cat([getattr(t,k) for t in tapes],dim=d) for k,d in dims.items()})
    lives=[s*8+i for s in seeds for i in range(8)]
    if mode.startswith("table"):
        agent=PublicMemory(len(lives),repeat=(mode=="table_retry"))
    else:
        agent=Intervention(BridgeAgent(cfg,opt,lives,net),mode,lives)
    recorder=_RecordPrimaryActions(agent,tape,1)
    metrics=private.rollout_with_table(_receipt_table("cpu").half())(recorder,tape,1)
    writes=agent.count if mode.startswith("table") else agent.agent.p.writes
    rows=[]
    for i,s in enumerate(seeds):
        t=tapes[i];trace=recorder.actions[:,:,:,8*i:8*i+8]
        q=evaluator.trace_statistics(trace,t,worlds.partners(t.raw,t.context))
        out={k:v.item() for k,v in q.items()}
        raw=trace[:,:,0,:].numpy()
        issued=(raw>=FIRST_FINISH)&(t.stage.numpy()==2)
        first=issued.argmax(1);actions=np.take_along_axis(raw,first[:,None,:],axis=1)[:,0]
        hit=issued.any(1)&(((actions-FIRST_FINISH)%4)==((t.target.numpy()-FIRST_FINISH)%4))
        out.update(seed=s,net=net,early_first=float(hit[:8].mean()),late_first=float(hit[32:].mean()),
                   writes=int(writes[i*8:i*8+8].sum()))
        rows.append(out)
    assert abs(np.mean([r["strict"] for r in rows])-metrics["strict"].item())<1e-7
    return rows


def main():
    torch.set_num_threads(1)
    cfg=ring_spikes.Config(**json.loads(Path("data/pulse-game/ring-value-search-v1/frozen.json").read_text())["best"]["config"])
    settings=json.loads((audit.OUT/"legacy-bridge-selected.json").read_text())
    opt=Options(**settings["selected"]["continuous_lif"]["options"])
    old=json.loads(Path("data/undirected/2026-11-19-local-rule-v1/audits/0128.json").read_text())["seeds"]
    future=audit.split("hard_task")[:32]
    variants={"reference":(cfg,opt,"true"),"only_first_receipt":(cfg,opt,"first_only"),
              "write_to_wrong_features":(cfg,opt,"wrong_address"),"other_private_mapping":(cfg,opt,"shadow"),
              "positive_only":(cfg,opt,"positive_only"),"negative_only":(cfg,opt,"negative_only"),
              "no_outcomes_after_32":(cfg,opt,"silent_after32"),"no_long_receptor_trace":(cfg,replace(opt,sensory_decay=0),"true"),
              "plain_delta":(replace(cfg,method="delta"),opt,"true"),"diagonal":(replace(cfg,method="diagonal"),opt,"true"),
              "table_one_choice":(cfg,opt,"table_once"),"table_can_retry":(cfg,opt,"table_retry")}
    report={"shared_options":asdict(opt),"source":audit.fingerprint(__file__),"sets":{}}
    for split,seeds in (("original",old),("new",future)):
        data=report["sets"][split]={}
        for label,(c,o,mode) in variants.items():
            rows=[]
            for net in ((3,) if mode.startswith("table") else (3,53)):
                for i in range(0,len(seeds),8):rows.extend(run_batch(c,o,seeds[i:i+8],net,mode))
            data[label]=rows
            print(split,label,{k:round(float(np.mean([r[k] for r in rows])),4) for k in (
                "strict","pair_strict_rate","early_first","late_first","writes")},flush=True)
            _write(audit.OUT/"legacy-bridge-mechanism.json",report)


if __name__=="__main__":
    main()
