"""Continue exact-triangle learning, separating additional time from different world weighting."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import interleave_age_train as age, interleave_train as common, interleave_triangle_train as tri
from .interleave_interference import DATA,EVIDENCE


MODES=("same","equal_classes")


def weighted_step(graph,rule,optimizer,tape,worlds):
    state=tri.native.start(graph,worlds*8)
    view=rule.expand(worlds*8);rows=torch.arange(len(tape["law"]));score=0
    for t in range(3):
        act,edit,_,_=tri._contact(view,state,tape,rows,t,"active")
        score=score+act.motor_log_prob+edit.edit_log_prob
        state=tri.native.wait(view,edit.state,tape["wait"][:,t],max_ticks=8)
    item,flip=tape["query"],tape["query_flip"]
    left,right=tri.binding.present(tape,item,flip)
    a=tri.native.act(view,state,left,right,tape["query_u"])
    earned=(a.side*flip*tape["law"][rows,item]).detach().reshape(-1,worlds,8)
    group_mean=earned.mean(2,keepdim=True)
    baseline=(group_mean.sum(1,keepdim=True)-group_mean)/(worlds-1)
    weight=earned.new_tensor([2.,2/3,2/3,2/3,2/3,2/3,2/3,2.])
    optimizer.zero_grad(set_to_none=True)
    loss=-((score+a.motor_log_prob).reshape_as(earned)*weight*(earned-baseline)).mean((1,2)).sum()
    loss.backward()
    norm=tri.revisit._gradients((graph.weights,graph.bias,*rule.parameters()),2.)
    if norm is None:raise FloatingPointError("Non-finite weighted triangle gradient")
    optimizer.step()
    with torch.no_grad():
        good=(earned>0).float()
        return torch.stack((good.mean((1,2)),good[:,:,[0,7]].mean((1,2)),good[:,:,1:7].mean((1,2)),norm),1)


def group_checkpoint(path,mode,births,step):
    return path/mode/("-".join(map(str,births)))/f"step-{step:06d}.pt"


def fit(out,parent,warm,births,mode,stop):
    torch.set_num_threads(1)
    init=tri.revisit._group_dir(parent,"direct",births)/"optimizer-008000.pt"
    born=torch.load(init,weights_only=True,map_location="cpu")
    graph,rule=tri._parents(warm,births)
    optimizer=torch.optim.Adam((graph.weights,graph.bias,*rule.parameters()),lr=.003)
    folder=group_checkpoint(out,mode,births,stop).parent
    folder.mkdir(parents=True,exist_ok=True)
    files=sorted(folder.glob("step-*.pt"))
    old=torch.load(files[-1],map_location="cpu",weights_only=True) if files else born
    if old["births"]!=list(births) or (files and old["arm"]!=mode):raise ValueError("Incorrect triangle trajectory")
    with torch.no_grad():
        for k in ("weights","bias","present"):getattr(graph,k).copy_(old["graph"][k])
    rule.load_state_dict(old["rule"]);optimizer.load_state_dict(old["optimizer"])
    step=old["step"];trace=old["history"].tolist() if files else []
    start=step;timer=perf_counter()
    def save():
        d=dict(arm=mode,step=step,births=list(births),history=torch.as_tensor(trace),
               graph={k:getattr(graph,k).detach() for k in ("weights","bias","present")},
               rule=rule.state_dict(),optimizer=optimizer.state_dict())
        common._atomic_save(group_checkpoint(out,mode,births,step),lambda f:torch.save(d,f))
        if step in (8000,12000,16000,24000,32000,stop):
            for i,b in enumerate(births):
                tri.old_fit.save_checkpoint(out,"odor",mode,b,step,graph,rule,i)
    if not files:save()
    for step in range(start+1,stop+1):
        rows=[tri.make_tape(tri.TRAIN_SEED,b,4,-.5,update=step) for b in births]
        t={k:torch.cat([d[k] for d in rows]) for k in rows[0]}
        if mode=="same":
            metrics=tri._step(graph,rule,optimizer,t,4)[:,[0,1,2,6]]
        else:
            metrics=weighted_step(graph,rule,optimizer,t,4)
        trace.append(metrics.detach().tolist())
        if step%1000==0 or step==stop:save()
        if step%500==0 or step==stop:
            print(mode,births,step,np.mean(trace[-100:],axis=(0,1)).round(3),
                  round((perf_counter()-timer)/max(1,step-start),3),flush=True)


@torch.no_grad()
def evaluate(path,births,arms,stages,worlds,seed,out):
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=True)
    report=dict(births=births,arms=arms,stages=stages,worlds_per_birth=worlds,seed=seed,results={},sha256={})
    for step in stages:
        cells={}
        for mode in arms:
            results={}
            for b in births:
                g,r=tri.old_fit.load_checkpoint(path,"odor",mode,b,step)
                observed={};raw={}
                for rho in (-.5,-.45,0.):
                    tape=tri.make_tape(seed,b,worlds,rho)
                    for branch in ("active","sham","false_sign","shuffled_objects"):
                        z=tri.trials(g,r,tape,branch);a=tri._analyze(tape,z)
                        key=f"{rho}/{branch}"
                        observed[key]={k:a[k] for k in ("all_sampled","uniform_all_greedy","other_all_greedy",
                                                       "both_uniform_histories_pass","geometries","r2_lagged_linear",
                                                       "own_first_order_energy","weakest_object_law")}
                        raw[key+"/side"]=z["selected"];raw[key+"/logit"]=z["logit"]
                file=out/f"{mode}-{b}-{step}.npz";np.savez_compressed(file,**raw)
                report["sha256"][file.name]=sha256(file.read_bytes()).hexdigest()
                results[b]=observed
            cells[mode]=results
        report["results"][step]=cells
        print("evaluated",step,flush=True)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/f"triangle-extension-{stages[-1]}.json").write_text(json.dumps(report,indent=1)+"\n",encoding="utf8")


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("fit","eval"),required=True)
    p.add_argument("--out",type=Path,default=DATA/"triangle-extension")
    p.add_argument("--parent",type=Path,default=DATA/"triangle-objective")
    p.add_argument("--warm",type=Path,default=DATA/"retention-objective")
    p.add_argument("--stop",type=int,default=32000)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--worlds",type=int,default=192)
    p.add_argument("--eval-seed",type=int,default=7183351)
    p.add_argument("--stages",type=int,nargs="+",default=[8000,16000,24000,32000])
    a=p.parse_args()
    births=list(range(731,737));groups=[births[:3],births[3:]]
    if a.mode=="eval":
        evaluate(a.out,births,MODES,a.stages,a.worlds,a.eval_seed,a.out/"evaluation");return
    if a.stop<=8000:raise ValueError("Extension starts at the frozen 8000-step optimizer")
    a.out.mkdir(parents=True,exist_ok=True)
    plan=dict(births=births,arms=MODES,parent_step=8000,final_planned=32000,
              group_parents={str(group):sha256((tri.revisit._group_dir(a.parent,"direct",group)/"optimizer-008000.pt")
                                               .read_bytes()).hexdigest() for group in groups},
              software_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
              criterion="After observing the direct 8000-step 4.86% strict paired success, extend on the exact "
                        "physical triangle. Both arms retain ordinary Adam moments and matched subsequent worlds. "
                        "same keeps the preceding objective exactly; equal_classes gives the +++/--- cases combined "
                        "half the sampling mass in its reward loss and the six other histories the other half. This is "
                        "an experimenter-set weighting, never a correctness label supplied to the graph. Validation "
                        "keeps all assignments equally represented and uses independent rotations.")
    manifest=a.out/"run.json"
    if manifest.exists() and json.loads(manifest.read_text())!=json.loads(json.dumps(plan)):raise ValueError("Protocol changed")
    common._atomic_save(manifest,lambda f:f.write_text(json.dumps(plan,indent=2)+"\n",encoding="utf8"))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for endpoint in sorted({t for t in (12000,16000,24000) if t<=a.stop}|{a.stop}):
            for future in [pool.submit(fit,a.out,a.parent,a.warm,group,arm,endpoint)
                           for group in groups for arm in MODES]:
                future.result()


if __name__=="__main__":main()
