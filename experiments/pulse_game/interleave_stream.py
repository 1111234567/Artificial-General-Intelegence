"""Evaluate retention and fresh learning throughout 256 physically continuous contacts."""

from argparse import ArgumentParser
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_age_train as age, interleave_train as joint, interleave_readback as rb
from . import interleave_capacity as capacity, local_trace_train as previous
from . import local_trace_graph as local, revisit_choice_graph as legacy
from .interleave_interference import DATA, EVIDENCE, ROOT


BLOCKS = (0,1,2,4,8,16,32,64)
MODES = ("carry","reset_each_block","no_writes")
READS = ("anchor","first_b","both","new_item","anchor_after_new","new_no_write")


@dataclass(frozen=True)
class Source:
    family: str
    path: Path
    arm: str
    step: int

    def load(self,birth):
        if self.family == "joint":
            return joint.load_checkpoint(self.path,self.arm,birth,self.step)
        if self.family == "lifetime":
            return age.load_checkpoint(self.path,self.arm,birth,self.step)
        if self.family == "readback":
            return rb.load_checkpoint(self.path,self.arm,birth,self.step)
        if self.family == "objective":
            # The focused comparator saves ordinary graph + rule files in the legacy npz layout.
            return previous.load_checkpoint(self.path,"odor",self.arm,birth,self.step)
        return previous.load_checkpoint(self.path,self.family,self.arm,birth,self.step)


def tape_for_worlds(birth,worlds,blocks,seed):
    selected = []
    available = 0
    step = 0
    while available < worlds:
        cfg=age.Config(pairs=2*worlds,max_blocks=8)
        arrays=age.single_tape(cfg,birth,step,blocks,domain=seed)
        first_odor=arrays[6][:,0,0]==age.ODOR
        selected.append(tuple(a[first_odor] for a in arrays))
        available+=int(first_odor.sum())
        step+=1
    return age.Tape(*(torch.from_numpy(np.concatenate(parts,axis=0)[:worlds]
                                      .reshape(-1,*parts[0].shape[2:]))
                      for parts in zip(*selected,strict=True)))


def start(g,r,count):
    return legacy.start(g,count) if r is None else local.start(g,count)


def wait(r,state,ticks):
    return legacy.wait(state,ticks) if r is None else local.wait(r.base if isinstance(r,rb.View) else r,state,ticks)


def action(r,state,left,right,draw):
    return legacy.act(state,left,right,draw) if r is None else local.act(r.base if isinstance(r,rb.View) else r,state,left,right,draw)


def contact(r,state,left,right,side_factor,fixed,draw,mode,permutation=None):
    a=action(r,state,left,right,draw[:,0])
    earned=fixed+side_factor*a.side
    if r is None:
        exp=legacy.taste(a,left,right,earned)
        proposals=legacy.propose(exp,draw[:,1:5].reshape(-1,2,2))
        # The caller supplies a dedicated sixth uniform for the legacy selector.
        decision=legacy.choose(proposals,draw[:,5])
        out=legacy.commit(exp,proposals,decision,intervention="disabled" if mode=="sham" else "active")
    elif isinstance(r,rb.View):
        exp=local.observe(r.base,a,left,right,earned)
        out=rb.commit(r,exp,draw[:,1:5],permutation,intervention=mode)
    else:
        exp=local.observe(r,a,left,right,earned)
        out=local.commit(r,exp,draw[:,1:5],intervention=mode)
    return out.state, a.side, earned


class ColdProbes:
    """All probes fork the snapshot. No probe modifies the stream."""

    def __init__(self,tape,seed,birth):
        n=len(tape.factor)
        rng=np.random.default_rng([seed,birth,9712])
        # Common sensory/action randomness in the two opposing whole-stream histories.
        def public(x):
            return torch.from_numpy(np.repeat(x,2,axis=0)).float()
        ids=(tape.obj[:,:2]==0).to(torch.int64).argmax(1)
        rows=torch.arange(n)
        self.menus=[]
        for index in (ids,1-ids):
            sign=public(age.signs(rng,n//2))
            positive=tape.right[rows,index]*sign[:,None]
            goal=tape.factor[rows,index]*sign
            draw=public(rng.random(n//2,dtype=np.float32))
            self.menus.append((-positive,positive,goal,draw))
        x=public(age.directions(rng,n//2)).repeat_interleave(2,0)
        sign=public(age.signs(rng,(n//2,2))).repeat_interleave(2,0)
        self.novel_study=x*sign[:,0,None]
        self.novel_return=x*sign[:,1,None]
        law=torch.tensor([-1.,1.]).repeat(n)
        self.learn_factor=sign[:,0]*law
        self.return_factor=sign[:,1]*law
        self.uniforms=public(rng.random((n//2,7),dtype=np.float32)).repeat_interleave(2,0)
        self.permutation=torch.from_numpy(rng.random((n//2,local.NODES)).argsort(1)).repeat_interleave(4,0)
        self.rows=rows.repeat_interleave(2)

    def read(self,r,r2,state):
        readings=[]
        for left,right,goal,u in self.menus:
            a=action(r,state,left,right,u)
            readings.append(a.side==goal)
        readings.append(readings[0]&readings[1])
        expanded=state.subset(self.rows)
        novel=[]
        for mode in ("active","sham"):
            after,_,_=contact(r2,expanded,-self.novel_study,self.novel_study,self.learn_factor,
                              torch.zeros_like(self.learn_factor),self.uniforms[:,:6],mode,self.permutation)
            new_state=wait(r2,after,4)
            a=action(r2,new_state,-self.novel_return,self.novel_return,self.uniforms[:,6])
            novel.append((a.side==self.return_factor).reshape(-1,2))
            if mode=="active":
                left,right,goal,u=(v[self.rows] for v in self.menus[0])
                a=action(r2,new_state,left,right,u)
                anchor_after=(a.side==goal).reshape(-1,2)
        return torch.stack([*(v[:,None].expand(-1,2) for v in readings),novel[0],anchor_after,novel[1]])


@torch.no_grad()
def evaluate(g,rule,tape,seed,birth,blocks,modes):
    n,t=tape.factor.shape
    coeff=None if rule is None else rule.expand(n)
    expanded=None if rule is None else rule.expand(2*n)
    probe=ColdProbes(tape,seed,birth)
    rng=np.random.default_rng([seed,birth,77341])
    extra=np.repeat(rng.random((n//2,t,1),dtype=np.float32),2,axis=0)
    uniforms=torch.cat((tape.draws,torch.from_numpy(extra)),dim=2)
    permutation=None
    if isinstance(coeff,rb.View) and coeff.arm=="permuted":
        prng=np.random.default_rng([seed,birth,731157])
        permutation=torch.from_numpy(prng.random((n//2,t,local.NODES)).argsort(2)).repeat_interleave(2,0)
    stages=list(blocks)
    out={}
    for mode in modes:
        state=start(g,rule,n)
        spends,choices,at_stage,drift=[],[],[],[]
        for contact_id in range(t+1):
            if contact_id%4==0 and contact_id//4 in stages:
                at_stage.append(probe.read(coeff,expanded,state).numpy())
                W=state.graph.weights
                mask=state.graph.present
                drift.append(torch.stack(((W-g.weights).flatten(1).abs().mean(1),
                                         (state.graph.bias-g.bias).abs().mean(1),
                                         (mask!=g.present).flatten(1).sum(1).float()),1).numpy())
            if contact_id==t:
                break
            if mode=="reset_each_block" and contact_id%4==0:
                state=start(g,rule,n)
            if contact_id:
                state=wait(coeff,state,tape.wait[:,contact_id])
            state,side,earned=contact(coeff,state,tape.left[:,contact_id],tape.right[:,contact_id],
                                      tape.factor[:,contact_id],tape.fixed[:,contact_id],
                                      uniforms[:,contact_id],"sham" if mode=="no_writes" else "active",
                                      permutation[:,contact_id] if permutation is not None else None)
            spends.append((earned>0).numpy())
            choices.append(side.to(torch.int8).numpy())
        out[mode+"/credit"]=np.stack(spends,axis=1)
        out[mode+"/side"]=np.stack(choices,axis=1)
        out[mode+"/read"]=np.stack(at_stage)
        out[mode+"/parameter_distance"]=np.stack(drift)
    out.update(kind=tape.kind.numpy().astype(np.int8),obj=tape.obj.numpy().astype(np.int16),
               law=tape.laws.numpy().astype(np.int8))
    return out


def rates(raw,blocks,modes):
    result={}
    for mode in modes:
        read=raw[mode+"/read"]
        good=raw[mode+"/credit"]
        stages=[]
        for i,n in enumerate(blocks):
            cell={k:float(read[i,j].mean()) for j,k in enumerate(READS)}
            if n:
                lo=4*blocks[i-1];hi=4*n
                quiz=np.arange(lo,hi)%4>=2
                kind=raw["kind"][:,lo:hi][:,quiz]
                law=raw["law"][:,lo:hi][:,quiz]
                scored=good[:,lo:hi][:,quiz]
                cell["quiz"]=float(scored.mean())
                cell["both_block_quizzes"]=float(scored.reshape(len(good),-1,2).all(-1).mean())
                for code,key in ((age.ODOR,"odor"),(age.CONTEXT,"context"),(age.FEEDER,"feeder")):
                    sel=kind==code
                    measured=scored[sel]
                    cell[key]=float(measured.mean()) if len(measured) else None
                    by_law=[scored[sel & (law==s)] for s in (-1,1)]
                    cell[key+"_weaker_law"]=min(float(v.mean()) for v in by_law) if all(len(v) for v in by_law) else None
                first_b_unreviewed=(raw["obj"][:,4:hi]==1).sum(1)==0
                cell["first_b_unreviewed_n"]=int(first_b_unreviewed.sum())
                cell["first_b_unreviewed"]=float(read[i,1,first_b_unreviewed].mean()) if first_b_unreviewed.any() else None
            stages.append(cell)
        result[mode]=stages
    return result


def run(source,label,births,worlds,seed,blocks=BLOCKS,modes=MODES):
    torch.set_num_threads(1)
    directory=DATA/"stream"/label
    directory.mkdir(parents=True,exist_ok=True)
    report=dict(births=births,seed=seed,independent_worlds_per_birth=worlds,blocks=blocks,
                contacts_per_block=4,readout_keys=READS,source=dict(family=source.family,arm=source.arm,step=source.step),
                notes="First block always physical odor; later blocks mix anchored odor, bilateral context conflict and "
                      "mark tasks. Same stable worlds throughout, new and occasionally repeated B objects. "
                      "Snapshot probes are read-only independent forks, new-item signs crossed within each full history.",
                roots={},sha256={})
    for birth in births:
        g,r=source.load(birth)
        tape=tape_for_worlds(birth,worlds,max(blocks),seed)
        raw=evaluate(g,r,tape,seed,birth,blocks,modes)
        p=directory/f"{birth}.npz"
        np.savez_compressed(p,**raw)
        report["roots"][birth]=rates(raw,blocks,modes)
        report["sha256"][p.name]=sha256(p.read_bytes()).hexdigest()
        print(label,birth,flush=True)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/f"stream-{label}.json").write_text(json.dumps(report,indent=1,allow_nan=False)+"\n",encoding="utf8")


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--label",required=True)
    p.add_argument("--family",choices=("joint","lifetime","objective","readback","odor","feeder"),required=True)
    p.add_argument("--run",type=Path,default=ROOT/"data/local-trace-20261119/local")
    p.add_argument("--arm",default="active")
    p.add_argument("--step",type=int,default=8000)
    p.add_argument("--worlds",type=int,default=64)
    p.add_argument("--seed",type=int,default=638309)
    p.add_argument("--births",nargs="+",type=int,default=list(previous.BIRTHS))
    p.add_argument("--blocks",nargs="+",type=int,default=BLOCKS)
    p.add_argument("--modes",nargs="+",choices=MODES,default=MODES)
    a=p.parse_args()
    run(Source(a.family,a.run,a.arm,a.step),a.label,a.births,a.worlds,a.seed,a.blocks,a.modes)
