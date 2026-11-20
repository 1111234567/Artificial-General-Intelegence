"""Confirm length-dependent nonlinear triangle learning using six new inherited graphs."""

from argparse import ArgumentParser
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import interleave_triangle_train as tri
from .interleave_interference import DATA,EVIDENCE


ROOTS=tuple(range(811,817))
STAGES=(0,8000,16000,24000,32000)
TRAIN_SEED=8784361
TEST_SEED=9209741


def fit(out,warm,group,stop):
    torch.set_num_threads(1)
    graph,rule=tri._parents(warm,group)
    optimizer=torch.optim.Adam((graph.weights,graph.bias,*rule.parameters()),lr=.003)
    folder=out/("group-"+"-".join(map(str,group)));folder.mkdir(exist_ok=True,parents=True)
    file=folder/"resume.pt";step=0;log=[]
    if file.is_file():
        d=torch.load(file,weights_only=True,map_location="cpu")
        if d["births"]!=list(group):raise ValueError("Wrong triangle replication group")
        with torch.no_grad():
            graph.weights.copy_(d["graph"]["weights"]);graph.bias.copy_(d["graph"]["bias"])
        rule.load_state_dict(d["rule"]);optimizer.load_state_dict(d["optimizer"])
        step,log=d["step"],d["trace"].tolist()
    def save():
        artifact=dict(births=list(group),step=step,graph={k:getattr(graph,k).detach() for k in tri.revisit.capacity.GRAPH_FIELDS},
                      rule=rule.state_dict(),optimizer=optimizer.state_dict(),trace=torch.as_tensor(log))
        tri.revisit._atomic(file,lambda f:torch.save(artifact,f))
        if step in STAGES:
            for i,b in enumerate(group):tri.old_fit.save_checkpoint(out,"odor","direct",b,step,graph,rule,i)
    if not file.exists():save()
    first=step;started=perf_counter()
    for step in range(first+1,stop+1):
        batch=[tri.make_tape(TRAIN_SEED,b,4,-.5,update=step) for b in group]
        tape={k:torch.cat([s[k] for s in batch]) for k in batch[0]}
        metrics=tri._step(graph,rule,optimizer,tape,4)
        log.append(metrics[:,[0,1,2,6,7]].numpy().tolist())
        if step%500==0 or step==stop:save()
        if step%1000==0 or step in (1,stop):
            print(group,step,np.array(log[-100:])[:,:,:3].mean((0,1)).round(3),
                  round((perf_counter()-started)/(step-first),3),flush=True)


def permuted_tape(tape,order):
    new=dict(tape);old=tape["order"];idx=torch.arange(len(old))[:,None]
    # The physical draw, orientation and elapsed gap travel with the object when
    # presentation order changes. All three terminal probes are independent forks.
    by_object=old.argsort(1)
    new_order=torch.tensor(order)[None].expand_as(old)
    source=by_object[idx,new_order]
    for key in ("study_flip","study_u","wait"):new[key]=tape[key][idx,source]
    new["order"]=new_order
    return new


@torch.no_grad()
def eval_stage(out,source,arm,births,step,worlds,seed,label,all_orders):
    torch.set_num_threads(1)
    dest=out/"order-validation"/label;dest.mkdir(exist_ok=True,parents=True)
    report=dict(births=births,step=step,worlds=worlds,seed=seed,all_orders=all_orders,results={},sha256={})
    for b in births:
        graph,rule=tri.old_fit.load_checkpoint(source,"odor",arm,b,step)
        base=tri.make_tape(seed,b,worlds,-.5)
        ords=list(permutations(range(3))) if all_orders else [None]
        raw={};passes=[];allmaps=[];mixed=[];actual=[]
        for j,order in enumerate(ords):
            tape=permuted_tape(base,order) if order is not None else base
            r=tri.trials(graph,rule,tape,"active")
            x=r["logit"].reshape(worlds,8,3,2);law=tape["law"].numpy().reshape(worlds,8,3)
            m=.5*(x[...,0]-x[...,1])*law
            ok=(m>.05).all(2)
            passes.append(ok[:,[0,7]].all(1));allmaps.append(ok.all(1))
            mixed.append(ok[:,1:7]);actual.append((r["selected"].reshape(worlds,8,3,2)==law[...,None]).all(2))
            raw[f"active_logit_{j}"]=r["logit"]; raw[f"active_selected_{j}"]=r["selected"]
        for mode in ("sham","false_sign","shuffled_objects"):
            t=tri.trials(graph,rule,base,mode)
            raw[f"{mode}_logit"]=t["logit"]
        wins=np.stack(passes);complete=np.stack(allmaps)
        report["results"][b]=dict(paired_uniform_success_per_order=float(wins.mean()),
            paired_uniform_all_orders=float(wins.all(0).mean()),paired_uniform_any_order=float(wins.any(0).mean()),
            all_eight_maps_per_order=float(complete.mean()),all_eight_maps_all_orders=float(complete.all(0).mean()),
            mixed_maps_all_three=float(np.stack(mixed).mean()),realized_all_three=float(np.stack(actual).mean()),
            controls={name:float((.5*(raw[f"{name}_logit"].reshape(worlds,8,3,2)[...,0]
                                     -raw[f"{name}_logit"].reshape(worlds,8,3,2)[...,1])
                                  *law>.05)[:,[0,7]].all((1,2)).mean()) for name in ("sham","false_sign","shuffled_objects")})
        p=dest/f"{b}-{step}.npz";np.savez_compressed(p,**raw,laws=law,paired=wins,all_maps=complete)
        report["sha256"][p.name]=sha256(p.read_bytes()).hexdigest();print("verified",label,b,step,flush=True)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/f"triangle-orders-{label}-{step}.json").write_text(json.dumps(report,indent=1)+"\n",encoding="utf8")


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("fit","eval"),required=True)
    p.add_argument("--out",type=Path,default=DATA/"triangle-confirmation")
    p.add_argument("--warm",type=Path,default=DATA/"retention-objective-confirmation")
    p.add_argument("--births",nargs="+",type=int,default=ROOTS)
    p.add_argument("--step",type=int,default=32000);p.add_argument("--worlds",type=int,default=96)
    p.add_argument("--source",type=Path);p.add_argument("--arm",default="direct")
    p.add_argument("--seed",type=int,default=TEST_SEED);p.add_argument("--label",default="independent")
    p.add_argument("--all-orders",action="store_true")
    a=p.parse_args()
    if a.mode=="eval":
        eval_stage(a.out,a.source or a.out,a.arm,a.births,a.step,a.worlds,a.seed,a.label,a.all_orders);return
    a.out.mkdir(parents=True,exist_ok=True)
    plan=dict(independent_roots=ROOTS,training_seed=TRAIN_SEED,validation_seed=TEST_SEED,
              selected="Unchanged uniformly weighted trajectory through 32000, absolute endpoint fixed before seeing "
                       "the final original 32k or any new-birth data. At 16k the original strict gate still improved. "
                       "All eight histories, pair of frustrated maps, and six distinct study orders evaluated separately.",
              parent_sha256={str(b):sha256(tri.old_fit.checkpoint_path(a.warm,"odor","uniform",b,8000).read_bytes()).hexdigest()
                             for b in ROOTS},code_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    manifest=a.out/"run.json"
    if manifest.exists() and json.loads(manifest.read_text())!=json.loads(json.dumps(plan)):raise ValueError("Protocol mismatch")
    tri.revisit._write_json(manifest,plan)
    fit(a.out,a.warm,tuple(a.births),a.step)


if __name__=="__main__":main()
