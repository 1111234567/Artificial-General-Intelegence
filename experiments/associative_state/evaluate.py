"""Public-receipt association updates followed by native eleven-action control."""
from __future__ import annotations
import os
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var,"1")
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.associative_state.model import AssociationMemory
from experiments.associative_state.bridge import PublicReceiptBinder, RandomMappingWorld, translate
from experiments.associative_state.train import OUT, SETTINGS, ARMS, sha, write_json, check_protocol
from experiments.learn_remember.gru_train import GRUAgent
from experiments.experience_transfer.train_transfer import SEEDS, mapping_table
from experiments.continuity.world import NEW, REPLACE

CONDITIONS=("immediate","delay","adapt","theta_only","clear","no_read","no_write","revise")


def mappings(seed,batch):
    rng=np.random.default_rng(881000+seed)
    a=np.argsort(rng.random((batch,4)),axis=-1)
    b=a.copy();b[:,0],b[:,1]=a[:,1],a[:,0]
    changed=a.copy();changed[:,2],changed[:,3]=a[:,3],a[:,2]
    return torch.from_numpy(np.stack((a,b),1)),torch.from_numpy(np.stack((changed,b),1))


def load(seed,arm):
    cp=torch.load(OUT/f"seed_{seed:03d}"/arm/"final.pt",map_location="cpu",weights_only=False)
    model=AssociationMemory(**cp["metadata"]["model_config"])
    model.load_state_dict(cp["model"]);model.eval()
    protocol=check_protocol();origin=protocol["origins"][str(seed)]
    assert sha(ROOT/origin["path"])==origin["sha256"]
    old=torch.load(ROOT/origin["path"],map_location="cpu",weights_only=False)
    executor=GRUAgent(**old["model_config"])
    executor.load_state_dict({k:v for k,v in old["model"].items() if k.startswith(("core.","head."))})
    executor.eval().requires_grad_(False)
    return model,executor


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def params_hash(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Subject:
    def __init__(self,seed,arm):
        self.seed,self.arm=seed,arm
        self.model,self.executor=load(seed,arm)
        self.contexts=(SEEDS[seed]["history_cue"],SEEDS[seed]["B_cue"])
        self.original=torch.tensor(mapping_table(seed,"related"),dtype=torch.long)
        self.table,self.revised=mappings(seed,SETTINGS["actors"])
        self.world=RandomMappingWorld(seed=981000+seed,batch_size=SETTINGS["actors"],n_jobs=2,active_jobs=2,
            actor_ids=tuple(range(201000,201000+SETTINGS["actors"])),dt=.25,
            duration_range=(8.,16.),gap_range=(16.,32.),family_allowed=(self.contexts[0],),
            mappings=self.table,contexts=self.contexts)
        self.binder=PublicReceiptBinder(SETTINGS["actors"],n_jobs=2,contexts=self.contexts)
        self.memory=self.model.initial_state(SETTINGS["actors"])
        self.hidden=self.executor.initial_state(SETTINGS["actors"])
        self.coverage=torch.zeros(SETTINGS["actors"],8,dtype=torch.long)
        self.work=dict(ticks=0,public_examples=0,parameter_updates=0,gradient_examples=0)

    def snapshot(self):
        return dict(model=deepcopy(self.model.state_dict()),memory=self.memory.clone(),hidden=self.hidden.clone(),
                    binder=self.binder.state_dict(),world=self.world.state_dict(),coverage=self.coverage.clone(),
                    work=deepcopy(self.work),parameter_hash=params_hash(self.model),memory_hash=tensor_hash(self.memory))

    def observe_and_update(self,obs,write=True,optimizer=None):
        """The only write source is this parser of public observations."""
        evidence=self.binder.consume(obs)
        losses=[];n=0
        for keys,values,mask in evidence:
            count=int(mask.sum())
            if not count:continue
            lanes=mask.nonzero(as_tuple=True)[0]
            self.coverage[lanes,keys[mask]]+=1
            if optimizer is not None:
                scores=self.model.read(self.memory[mask].detach(),keys[mask])
                losses.append(F.cross_entropy(scores,values[mask],reduction="sum"))
                n+=count
            if write:
                with torch.no_grad():self.memory=self.model.write(self.memory,keys,values,mask).detach()
            self.work["public_examples"]+=count
        if losses:
            loss=torch.stack(losses).sum()/n
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),5.)
            optimizer.step()
            self.work["parameter_updates"]+=1;self.work["gradient_examples"]+=n
        return evidence

    def predictions(self,obs,no_read=False):
        features=obs.reshape(SETTINGS["actors"],2,28)
        family=features[...,24:28].argmax(-1)
        raw=features[...,5:9].argmax(-1)
        which=(family==self.contexts[1]).long()
        keys=4*which+raw
        if no_read:return torch.zeros_like(keys)
        with torch.no_grad():
            return torch.stack([self.model.read(self.memory,keys[:,j]).argmax(-1) for j in range(2)],1)

    def advance(self,teacher=False,write=True,no_read=False,optimizer=None):
        obs=self.world.observe()
        self.observe_and_update(obs,write=write,optimizer=optimizer)
        pred=self.predictions(obs,no_read=no_read)
        mapped=translate(obs,pred,self.original)
        with torch.no_grad():logits,self.hidden=self.executor.step(mapped,self.hidden)
        actions=self.world.oracle("midpoint") if teacher else logits.argmax(-1)
        # Return before act, allowing the evaluator to archive actual opportunities.
        return obs,pred,actions,logits

    def support(self,ticks,family,write=True,learn=False,archive=False):
        self.world.set_family_policy((self.contexts[family],))
        optimizer=torch.optim.Adam(self.model.parameters(),lr=SETTINGS["online_lr"]) if learn else None
        records=[];digest=hashlib.sha256();start=self.snapshot()
        for _ in range(ticks):
            obs,pred,actions,_=self.advance(teacher=True,write=write,optimizer=optimizer)
            digest.update(obs.numpy().tobytes());digest.update(actions.numpy().tobytes())
            if archive:records.append(obs.clone())
            self.world.act(actions);self.work["ticks"]+=1
        return dict(start=start,final=self.snapshot(),teacher_stream_sha256=digest.hexdigest(),
                    observations=torch.stack(records) if records else None,
                    optimizer=None if optimizer is None else optimizer.state_dict())


def probe(subject,folder,name):
    start=subject.snapshot();world=subject.world
    world.set_family_policy(subject.contexts)
    book={};finished=[];idle=0;idle_actions=0;translated=0;translation_correct=0
    arrays={k:[] for k in ("obs","actions","prediction","event_kind","event_rid","event_eligible","event_family",
        "raw","target","mature_mask","mature_rid","mature_done","mature_strict","active")}
    for index in range(SETTINGS["eval_ticks"]):
        obs,pred,actions,_=subject.advance(write=False,no_read=name=="no_read")
        # Mature emissions refer to earlier requests, before this tick's cues.
        for lane,job in world.matured_mask.nonzero().tolist():
            key=(lane,job,int(world.matured_request_id[lane,job]))
            if key in book:
                row=book.pop(key);row.update(mature_tick=world.tick,
                    ordinary=bool(world.matured_completed[lane,job]),strict=bool(world.matured_clean_completed[lane,job]))
                finished.append(row)
        command=(world.event_kind==NEW)|(world.event_kind==REPLACE)
        for lane,job in command.nonzero().tolist():
            family=int(world.event_family[lane,job]);fam=subject.contexts.index(family)
            raw=int(world.raw_event_payload[lane,job]);correct=int(world.event_payload[lane,job])
            is_correct=int(pred[lane,job])==correct
            translated+=1;translation_correct+=is_correct
            if bool(world.event_eligible[lane,job]):
                rid=int(world.event_request_id[lane,job])
                book[lane,job,rid]=dict(lane=lane,job=job,rid=rid,issued_tick=world.tick,family="AB"[fam],raw=raw,target=correct,
                    conflict=bool(subject.world.private_mappings[lane,0,raw]!=subject.world.private_mappings[lane,1,raw]),
                    translation_correct=is_correct)
        idle_mask=~world.active
        idle+=int(idle_mask.sum());idle_actions+=int(((actions!=0)&idle_mask).sum())
        values=dict(obs=obs,actions=actions,prediction=pred,event_kind=world.event_kind,event_rid=world.event_request_id,
            event_eligible=world.event_eligible,event_family=world.event_family,raw=world.raw_event_payload,target=world.event_payload,
            mature_mask=world.matured_mask,mature_rid=world.matured_request_id,mature_done=world.matured_completed,
            mature_strict=world.matured_clean_completed,active=world.active)
        for key,value in values.items():arrays[key].append(value.detach().numpy().copy())
        world.act(actions);subject.work["ticks"]+=1
    groups={}
    for label,selected in (("all",finished),("A",[r for r in finished if r["family"]=="A"]),
        ("B",[r for r in finished if r["family"]=="B"]),
        ("A_conflict",[r for r in finished if r["family"]=="A" and r["conflict"]]),
        ("B_conflict",[r for r in finished if r["family"]=="B" and r["conflict"]]),
        ("A_shared",[r for r in finished if r["family"]=="A" and not r["conflict"]]),
        ("B_shared",[r for r in finished if r["family"]=="B" and not r["conflict"]])):
        groups[label]=dict(n=len(selected),strict=sum(r["strict"] for r in selected),ordinary=sum(r["ordinary"] for r in selected),
                           translation_correct=sum(r["translation_correct"] for r in selected))
    final=subject.snapshot()
    assert torch.equal(start["memory"],final["memory"])
    assert start["parameter_hash"]==final["parameter_hash"]
    report=dict(seed=subject.seed,arm=subject.arm,condition=name,groups=groups,requests=finished,
        right_censored=list(book.values()),idle_opportunities=idle,idle_actions=idle_actions,
        command_translations=translated,correct_translations=translation_correct,
        start_tick=start["world"]["tick"] if "tick" in start["world"] else None,end_tick=world.tick,
        memory_norm=float(subject.memory.norm(dim=-1).mean()),coverage_min=int(start["coverage"].min()),
        start_work=start["work"],final_work=final["work"],parameters=subject.model.parameter_report(),
        memory_readonly=True,parameters_frozen_during_probe=True)
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/name
    np.savez_compressed(path.with_suffix(".npz"),**{k:np.stack(v) for k,v in arrays.items()})
    torch.save(dict(start=start,final=final),path.with_suffix(".pt"))
    report["arrays_sha256"]=sha(path.with_suffix(".npz"));report["states_sha256"]=sha(path.with_suffix(".pt"))
    write_json(path.with_suffix(".json"),report)
    print(json.dumps(dict(event="probe",seed=subject.seed,arm=subject.arm,condition=name,groups=groups,
                         idle=[idle_actions,idle],norm=report["memory_norm"])),flush=True)
    return report


def run(seed,arm):
    torch.set_num_threads(1);timer=time.perf_counter()
    folder=OUT/f"seed_{seed:03d}"/arm/"evaluation"
    folder.mkdir(parents=True,exist_ok=False)
    subject=Subject(seed,arm)
    first=subject.support(SETTINGS["support_ticks"],0,archive=True)
    second=subject.support(SETTINGS["support_ticks"],1,archive=True)
    common=subject.snapshot()
    torch.save(dict(A=first,B=second),folder/"support.pt")
    results=[]
    for condition in CONDITIONS:
        branch=deepcopy(subject)
        if condition in ("delay","adapt","theta_only"):
            segment=branch.support(SETTINGS["delay_ticks"],1,write=condition!="theta_only",learn=condition!="delay")
            if condition=="theta_only":assert torch.equal(segment["start"]["memory"],segment["final"]["memory"])
            torch.save(segment,folder/f"{condition}_support.pt")
        elif condition=="clear":branch.memory.zero_()
        elif condition=="no_write":
            branch.memory.zero_();branch.hidden=branch.executor.initial_state(SETTINGS["actors"])
            branch.binder=PublicReceiptBinder(SETTINGS["actors"],2,contexts=branch.contexts)
            branch.coverage.zero_();branch.work=dict(ticks=0,public_examples=0,parameter_updates=0,gradient_examples=0)
            # Same actual teacher observations, with this arm's own h and no M writes.
            for obs in torch.cat((first["observations"],second["observations"]),0):
                branch.observe_and_update(obs,write=False)
                predicted=branch.predictions(obs)
                with torch.no_grad():_,branch.hidden=branch.executor.step(translate(obs,predicted,branch.original),branch.hidden)
                branch.work["ticks"]+=1
        elif condition=="revise":
            branch.world.set_mappings(branch.revised)
            segment=branch.support(SETTINGS["support_ticks"],0)
            torch.save(segment,folder/"revise_support.pt")
        results.append(probe(branch,folder,condition))
    # No branch can mutate the common numerical checkpoint.
    assert torch.equal(subject.memory,common["memory"]) and params_hash(subject.model)==common["parameter_hash"]
    write_json(folder/"completion.json",dict(seed=seed,arm=arm,conditions=list(CONDITIONS),seconds=time.perf_counter()-timer,
        support_sha256=sha(folder/"support.pt"),pretraining_sha256=sha(folder.parent/"final.pt"),
        reports={c:sha(folder/f"{c}.json") for c in CONDITIONS}))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--seed",type=int,required=True);p.add_argument("--arm",choices=ARMS,required=True)
    args=p.parse_args();run(args.seed,args.arm)
