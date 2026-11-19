"""Full-trajectory action imitation; actual-outcome PG is a separate stage."""
import os
for _var in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ.setdefault(_var,"1")
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
import argparse
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from experiments.learned_binding.common import *
from experiments.learned_binding.data import generate


def loss_for(logits,actions):
    weights=logits.new_ones(11);weights[0]=SETTINGS["wait_weight"]
    return F.cross_entropy(logits.reshape(-1,11),actions.reshape(-1),weight=weights)


def make_data(job):
    seed,index=job;torch.set_num_threads(1)
    check_protocol();wide=index>=8
    folder=OUT/f"seed_{seed:03d}"/"data";folder.mkdir(parents=True,exist_ok=True)
    path=folder/f"stream_{index:02d}.pt"
    if path.exists():raise FileExistsError(path)
    data=generate(seed,SETTINGS["train_batch"],SETTINGS["train_ticks"],
        duration_range=(8.,16.),gap_range=(16.,32.) if wide else (1.,4.),
        stream_id=index,random_seed=47000,split="train")
    torch.save(data,path)
    return dict(seed=seed,stream=index,path=str(path.relative_to(ROOT)),sha256=sha(path))


def prepare(workers=12):
    path=OUT/"data-freeze.json"
    if path.exists():raise FileExistsError(path)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(make_data,(s,i)) for s in range(3) for i in range(16)]
        rows=[]
        for done in as_completed(jobs):
            row=done.result();rows.append(row);print(json.dumps(dict(event="data",seed=row["seed"],stream=row["stream"])),flush=True)
    write_json(path,dict(streams=rows))


def credit_check(model,obs,actions):
    """Measure gradients at old successful receipts from later action losses."""
    model.zero_grad(set_to_none=True)
    logits,state,diag=model(obs,model.initial_state(obs.shape[1]),return_diagnostics=True)
    # A fixed last-quarter action loss; early gradients cannot be direct label loss.
    stop=len(obs)*3//4
    loss=loss_for(logits[stop:],actions[stop:])
    grads=torch.autograd.grad(loss,[diag[n] for n in ("q","k","value","gate")],allow_unused=False)
    pub=obs.reshape(len(obs),obs.shape[1],2,28)
    success=(pub[...,14:22].sum(-1)>.5)&(pub[...,22]>.5)&(pub[...,23]<.5)
    old=success.clone();old[stop:]=False
    result={}
    for name,grad in zip(("q","k","value","gate"),grads):
        selected=grad[old]
        result[name]=dict(early_success_rows=int(old.sum()),maxabs=float(selected.abs().max()) if selected.numel() else 0.,
                         nonzero_scalars=int((selected!=0).sum()),finite=bool(torch.isfinite(grad).all()))
    return dict(loss_tail_starts_at=stop,tail_loss=float(loss.detach()),gradients=result,
                executor_gradients_absent=all(p.grad is None for p in model.executor.parameters()))


def train(seed,device="cpu",pilot=False,updates=32):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not pilot:check_protocol()
    model=new_native(seed,device);model.train();initial_executor=parameter_hash(model.executor)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=SETTINGS["ce_lr"])
    if pilot:
        data=[generate(seed,batch=16,ticks=512,stream_id=i,random_seed=990001,split="train",
            duration_range=(8.,16.),gap_range=(1.,4.)) for i in range(4)]
        count=updates
    else:
        bindings=json.loads((OUT/"data-freeze.json").read_text(encoding="utf-8"))["streams"]
        own=sorted([r for r in bindings if r["seed"]==seed],key=lambda r:r["stream"])
        for r in own:assert sha(ROOT/r["path"])==r["sha256"]
        data=[torch.load(ROOT/r["path"],map_location="cpu",weights_only=False) for r in own]
        count=SETTINGS["ce_updates"]
    folder=OUT/f"seed_{seed:03d}"/"pretrain"
    if not pilot:folder.mkdir(parents=True,exist_ok=False)
    log=[];timer=time.perf_counter()
    for step in range(count):
        index=step%len(data) if pilot else (step%8 if step<SETTINGS["dense_updates"] else (step-SETTINGS["dense_updates"])%16)
        obs=data[index]["obs"].to(device);actions=data[index]["actions"].to(device)
        logits,state=model(obs,model.initial_state(obs.shape[1]))
        loss=loss_for(logits,actions)
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        grad=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.))
        optimizer.step()
        if step==0 or (step+1)%64==0 or step+1==count:
            finish=actions>=3
            row=dict(update=step+1,stream=index,loss=float(loss.detach()),grad_norm=grad,
                finish_accuracy=float((logits.argmax(-1)[finish]==actions[finish]).float().mean()),
                seconds=time.perf_counter()-timer)
            log.append(row);print(json.dumps(dict(event="CE",seed=seed,**row)),flush=True)
    assert parameter_hash(model.executor)==initial_executor
    if not pilot:
        # One fixed long stream, no fitting or selection based on this diagnostic.
        audit=credit_check(model,data[8]["obs"].to(device),data[8]["actions"].to(device))
        cp=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},model_config=model.config(),
            optimizer=optimizer.state_dict(),updates=count,seed=seed,settings=SETTINGS,
            executor_hash=initial_executor,parameter_report=model.parameter_report(),log=log,credit_check=audit)
        torch.save(cp,folder/"final.pt")
        write_json(folder/"training.json",dict(seed=seed,settings=SETTINGS,log=log,credit_check=audit,
            parameter_report=model.parameter_report(),seconds=time.perf_counter()-timer,device=device,
            checkpoint_sha256=sha(folder/"final.pt"),executor_unchanged=True))
    return model


def train_one(job):
    seed,device=job;path=OUT/"logs"/f"pretrain_{seed}.log";path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as f,redirect_stdout(f),redirect_stderr(f):train(seed,device)
    return seed


def train_all(device="cpu"):
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs=[(s,("cuda" if s==0 else "cpu") if device=="mixed" else device) for s in range(3)]
        for done in as_completed([pool.submit(train_one,job) for job in jobs]):print(json.dumps(dict(trained_seed=done.result())),flush=True)
    write_json(OUT/"training-freeze.json",dict(all_complete_before_scoring=True,
        checkpoints={str(native_checkpoint(s).relative_to(ROOT)):sha(native_checkpoint(s)) for s in range(3)}))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("command",choices=("data","train","pilot"));p.add_argument("--device",default="cpu")
    p.add_argument("--workers",type=int,default=12);p.add_argument("--updates",type=int,default=32)
    a=p.parse_args()
    if a.command=="data":prepare(a.workers)
    elif a.command=="train":train_all(a.device)
    else:train(0,a.device,pilot=True,updates=a.updates)
