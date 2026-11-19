"""Fully addressed graph with continuous activity and matched bootstrap experience budgets."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import self_param_writer as fw, ring_logic_study as io


EVIDENCE=io.REPO/"docs/evidence/self-edit-dose-2026-11-19/carried"
REG=io.read(EVIDENCE/"protocol.json")
DATA=io.REPO/"data/self-edit-dose-20261119/carried"


@dataclass
class Life:
    odor: np.ndarray
    sign: np.ndarray
    cue: np.ndarray
    swap: np.ndarray
    outcomes: np.ndarray

    def subset(self,idx):
        return Life(*(x[idx] for x in (self.odor,self.sign,self.cue,self.swap,self.outcomes)))


def make(seed,count,length,channels=1,noise=0.,reverse=False,twins=False):
    rng=np.random.default_rng(seed)
    n=count//2 if twins else count
    odor=rng.normal(size=(n,channels,3));odor=1.35*odor/np.linalg.norm(odor,axis=-1,keepdims=True)
    sign=rng.choice((-1,1),size=(n,channels))
    cue=np.argsort(rng.random((n,length//channels,channels)),axis=-1).reshape(n,length)
    swap=rng.choice((-1,1),size=(n,length))
    slips=np.where(rng.random((n,length))<noise,-1,1).astype(np.int8)
    truth=np.take_along_axis(sign,cue,axis=1)*swap
    if reverse:truth[:,length//2:]*=-1
    outcomes=truth*slips
    if twins:
        odor,sign,cue,swap,truth,outcomes=(np.repeat(a,2,0) for a in (odor,sign,cue,swap,truth,outcomes))
        for a in (sign,truth,outcomes):a[1::2]*=-1
    return Life(odor.astype(np.float32),sign,cue,swap,outcomes),truth.astype(np.int8)


def inspection(graph,post,address):
    edge,receiver,sender=fw.decode(address,graph.width)
    present,value,b=fw.read_address(graph,address,"own")
    rows=torch.arange(len(address))
    stimulus=torch.zeros_like(graph.bias)
    stimulus[:,0]=present
    stimulus[:,1]=value
    stimulus[:,2]=b
    stimulus[rows,receiver]+=0.4
    stimulus[rows,sender]+=0.4*edge.to(stimulus.dtype)
    return fw.advance(graph,post,stimulus,2)


def interact(initial,tape,draws,*,stochastic,disabled=False):
    count,length=tape.cue.shape
    graph=initial.repeat(count)
    h=torch.zeros_like(graph.bias)
    row=np.arange(count)
    motors=[];addresses=[];operations=[];rewards=[];sides=[];commands=[];status=[]
    for t in range(length):
        left=torch.as_tensor((tape.odor[row,tape.cue[:,t]]*tape.swap[:,t,None]).astype(np.float32))
        right=-left
        h=fw.advance(graph,h,fw.sensory(graph,left,right),5)
        logit=2.5*(h[:,11]-h[:,10])
        go=draws[:,t,2]<logit.sigmoid().detach() if stochastic else logit>0
        side=go.to(torch.float32)*2-1
        # Only the sign delivered at the contacted side is passed back to the graph.
        taste=torch.as_tensor(tape.outcomes[:,t],dtype=torch.float32)*side
        pre=fw.advance(graph,h,fw.sensory(graph,left,right,side),3)
        post=fw.advance(graph,pre,fw.sensory(graph,left,right,side,taste),3)
        addr,lp_a=fw.sample(fw.address_logits(graph,pre,post),draws[:,t,0])
        logits,payload=fw.operation_logits(graph,post,addr,view="own")
        op,lp_o=fw.sample(logits,draws[:,t,1])
        # Carry the actual recurrent readback state into the next contact.
        h=inspection(graph,post,addr)
        graph,rc=fw.commit(graph,fw.Command(addr,op,payload),keep_connected=True,
                           intervention="disabled" if disabled else "active",value_limit=None)
        motors.append(F.logsigmoid(side*logit));addresses.append(lp_a);operations.append(lp_o)
        rewards.append(taste);sides.append(side.detach());commands.append(torch.stack((addr,op),dim=-1))
        status.append(rc.status)
    return tuple(torch.stack(a,dim=1) for a in (motors,addresses,operations,rewards,sides,commands,status)),graph,h


def paths(seed,diversity,horizon):
    return DATA/f"{seed}-{diversity}-{horizon}.npz"


def train(job):
    torch.set_num_threads(1)
    seed,diversity,horizon=job
    initial=fw.spawn(seed,REG["writer_nodes"])
    birth=fw.Graph(nn.Parameter(initial.weights),nn.Parameter(initial.bias),initial.present)
    optimizer=torch.optim.Adam([birth.weights,birth.bias],lr=.009)
    batch,T,steps=REG["batch"],REG["contacts"],REG["updates"]
    large=REG["history_banks"]["fresh"]
    bank,_=make([seed,73111],large,T)
    n=REG["history_banks"][diversity]
    rng=np.random.default_rng([seed,73112])
    orders=np.concatenate([rng.permutation(n) for _ in range(large//n)]).reshape(steps,batch)
    noise=np.random.default_rng([seed,73113])
    maximum=REG["command_credit"][horizon]
    motor_base=torch.zeros(T);write_base=torch.zeros(T)
    trace=[];started=time.perf_counter()
    for ids in orders:
        u=torch.from_numpy(noise.random((batch,T,3),dtype=np.float32))
        (lm,la,lo,r,*_),_,_=interact(birth,bank.subset(ids),u,stochastic=True)
        loss=fw.policy_loss(lm,la,lo,r,motor_base,write_base,write_horizon=maximum)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_([birth.weights,birth.bias],2)
        optimizer.step()
        with torch.no_grad():
            motor_base.lerp_(fw.reward_to_go(r).mean(0),.1)
            write_base.lerp_(fw.reward_to_go(r,maximum).mean(0),.1)
        trace.append(r.mean(0).numpy())
    p=paths(seed,diversity,horizon);p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p,w=birth.weights.detach().numpy(),b=birth.bias.detach().numpy(),mask=birth.present.numpy(),
                        seen=n,contacts=steps*batch*T,curve=np.asarray(trace),seconds=time.perf_counter()-started)
    print(f"completed birth={seed} diversity={diversity} write-credit={maximum}",flush=True)


@torch.no_grad()
def evaluate(job):
    torch.set_num_threads(1)
    seed,diversity,horizon=job
    p=paths(seed,diversity,horizon)
    with np.load(p,allow_pickle=False) as z:
        birth=fw.Graph(*(torch.from_numpy(z[k]) for k in ("w","b","mask")))
    outcomes={}
    conditions=(("short",8,1,0.,False),("clean",64,2,0.,False),("noise",64,2,.2,False),("change",64,2,.2,True))
    for name,length,channels,noise,reverse in conditions:
        tape,truth=make([seed,73114,channels],96,length,channels,noise,reverse,twins=True)
        draws=torch.from_numpy(np.random.default_rng([seed,73115,length]).random((96,length,3),dtype=np.float32))
        for disabled in (False,True):
            (lm,la,lo,r,side,commands,status),final,h=interact(birth,tape,draws,stochastic=False,disabled=disabled)
            arm="disabled" if disabled else "live"
            outcomes[name+"/"+arm]=np.stack((side.numpy(),truth,r.numpy()),axis=-1).astype(np.int8)
            outcomes[name+"/"+arm+"/commands"]=commands.numpy().astype(np.int16)
            outcomes[name+"/"+arm+"/status"]=status.numpy().astype(np.int8)
            if not torch.isfinite(final.weights).all() or not torch.isfinite(final.bias).all():raise RuntimeError("Nonfinite")
            if torch.any(final.weights[~final.present]!=0):raise RuntimeError("Dormant weight leak")
            if name=="short" and not np.all(side.numpy()[::2,0]==side.numpy()[1::2,0]):
                raise RuntimeError("First choice leaked private orientation")
    np.savez_compressed(p.with_suffix(".scores.npz"),**outcomes)


def source_files():
    return {p.name:io.sha(p) for p in (Path(__file__),Path(fw.__file__))}


def summarize(jobs):
    scores={};results={}
    for seed,d,h in jobs:
        with np.load(paths(seed,d,h).with_suffix(".scores.npz"),allow_pickle=False) as z:
            for name in ("short","clean","noise","change"):
                for arm in ("live","disabled"):
                    a=z[name+"/"+arm];good=a[...,0]==a[...,1]
                    val=good[:,-(4 if name=="short" else 8):].mean(1).reshape(-1,2).mean(1)
                    scores.setdefault((name,arm,d,h),[]).append(val)
    rng=np.random.default_rng(73118)
    g=len(REG["births"]);n=48
    bg=rng.multinomial(g,np.ones(g)/g,20000)/g
    # Each training root has separate validation worlds; sample nested world pairs.
    def interval(x,tail):
        per_seed=[]
        for row in x:
            wg=rng.multinomial(n,np.ones(n)/n,20000)/n
            per_seed.append(wg@row)
        bootstrap=(bg*np.stack(per_seed,axis=1)).sum(1)
        return dict(mean=float(x.mean()),ci=np.quantile(bootstrap,[tail,1-tail]).tolist(),by_root=x.mean(1).tolist())
    for env in ("short","clean","noise","change"):
        cells={f"{d}/{h}":np.asarray(scores[env,"live",d,h]) for d in REG["history_banks"] for h in REG["command_credit"]}
        L,S=cells["fresh/all"],cells["fresh/next"]
        l,s=cells["replayed/all"],cells["replayed/next"]
        contrast={"diversity":(L+S-l-s)/2,"discrete_credit":(L+l-S-s)/2,"interaction":L-S-l+s}
        results[env]=dict(accuracy={k:float(v.mean()) for k,v in cells.items()},
                          disabled={f"{d}/{h}":float(np.mean(scores[env,"disabled",d,h])) for d in REG["history_banks"] for h in REG["command_credit"]},
                          effects={k:interval(v,.025/9 if env!="clean" else .025) for k,v in contrast.items()})
    zip_path=EVIDENCE/"traces-and-checkpoints.zip"
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as z:
        for job in jobs:
            p=paths(*job)
            for f in (p,p.with_suffix(".scores.npz")):z.write(f,f.name)
    io.write(EVIDENCE/"summary.json",dict(results=results,archive=io.sha(zip_path),sources=source_files(),
             source_of_registration=io.sha(EVIDENCE/"protocol.json"),contacts_per_fit=REG["updates"]*REG["batch"]*REG["contacts"]))


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--workers",type=int,default=5);args=p.parse_args()
    jobs=[(s,d,h) for s in REG["births"] for d in REG["history_banks"] for h in REG["command_credit"]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:list(pool.map(train,jobs))
    io.write(EVIDENCE/"frozen.json",dict(sources=source_files(),checkpoints={paths(*j).name:io.sha(paths(*j)) for j in jobs}))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:list(pool.map(evaluate,jobs))
    summarize(jobs)
