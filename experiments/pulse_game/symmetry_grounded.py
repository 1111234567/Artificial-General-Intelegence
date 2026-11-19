"""Ground a sensory address in public flashes, then infer repeated odor transforms from coactivation."""
import argparse
from dataclasses import asdict
import os
import time
import numpy as np
import torch
from torch import nn

from . import mixture_circuit_train as helpers, symmetry_memory as core, symmetry_memory_train as opt
from . import symmetry_train as source
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


class Grounded(core.Memory):
    def __init__(self,cfg):
        super().__init__(cfg)
        for name in ("a","b","read","write"):delattr(self,name)
        self.actual_odor=nn.Linear(cfg.nodes,4)
        self.transform=nn.Parameter(torch.eye(4))

    def heads(self,states):
        decoded=self.actual_odor(states)
        return nn.functional.normalize(decoded,dim=-1),nn.functional.normalize(decoded@self.transform,dim=-1)

    def export(self,cut=None):
        arr=lambda x:x.detach().cpu().numpy().copy()
        layer=lambda m:(arr(m.weight.T),arr(m.bias))
        return dict(matrix=arr(self.sensor.matrix(cut)),sensors=arr(self.sensor.sensors),
                    ground=layer(self.actual_odor),transform=arr(self.transform),
                    scalars=[float(v.detach()) for v in self.scalars()],latent=4,capacity=self.cfg.capacity,
                    change=self.cfg.change,rule=([layer(self.rule[0]),layer(self.rule[2])] if self.cfg.change=="learned" else None))


def measured_transform(public,capacity,*,shuffle=False):
    # All pairs in the recent public history contribute. No matched index,
    # hidden family, chosen preference, future sign or goal is inspected.
    x=public["odor"]
    side=public["side"].astype(int)
    rows=np.arange(len(x))[:,None];t=np.arange(x.shape[1])[None,:]
    taste=x[rows,t,side].astype(np.float64)
    taste/=np.maximum(np.linalg.norm(taste,axis=2,keepdims=True),1e-9)
    past=np.roll(taste,1,axis=0) if shuffle else taste
    cov=np.zeros((4,4))
    for lag in range(1,min(capacity,x.shape[1]-1)+1):
        lhs=past[:,:-lag].reshape(-1,4)
        rhs=taste[:,lag:].reshape(-1,4)
        cov+=lhs.T@rhs/len(lhs)
    left,singular,right=np.linalg.svd(cov)
    return (left@right).astype(np.float32),singular.tolist(),cov.tolist()


def public_fit(net,data,steps,device,seed):
    rng=np.random.default_rng(seed)
    optim=torch.optim.Adam(list(net.sensor.parameters())+list(net.actual_odor.parameters()),lr=.004)
    start=time.perf_counter()
    for i in range(steps):
        b=helpers.batch(data,torch.as_tensor(rng.integers(len(data["side"]),size=72),device=device))
        times=helpers.early_times(b,float(rng.choice([.0,.5,1.,1.])))
        st=net.sensor.states(b,times)
        error=(net.actual_odor(st)-b["odor"]).square().mean()
        optim.zero_grad(set_to_none=True);error.backward();optim.step()
    return float(error.detach()),time.perf_counter()-start


def load(path,cut=None):
    d=torch.load(path,map_location="cpu",weights_only=True)
    net=Grounded(core.Config(**d["spec"]));net.load_state_dict(d["state"])
    return net.export(cut)


def run(folder,bank,data,val,seed,change,initial,steps,batch,device):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    cfg=core.Config(change=change,sensory_loss=0.)
    model=Grounded(cfg).to(device)
    warm_path=folder/f"public-receptor-s{seed}.pt"
    if warm_path.exists():
        warm=torch.load(warm_path,map_location=device,weights_only=True);model.load_state_dict(warm,strict=False)
    else:
        mse,spent=public_fit(model,data,360,device,seed)
        torch.save({k:v for k,v in model.state_dict().items() if k.startswith(("sensor.","actual_odor."))},warm_path)
        print("Reconstructed actually observed public flash",seed,mse,round(spent,1),flush=True)
    measured,s,cov=measured_transform(bank,cfg.capacity,shuffle=initial=="unrelated")
    if initial=="random":
        measured=np.linalg.qr(rng.normal(size=(4,4)))[0].astype(np.float32)
    with torch.no_grad():model.transform.copy_(torch.from_numpy(measured).to(device))
    name=f"{initial}-{change}-s{seed}"
    _write(folder/(name+"-init.json"),dict(singular=s,covariance=cov,initial_transform=measured.tolist(),
            sampled_public_lives=len(bank["side"]),sensory_receptor_checkpoint=warm_path.name))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0018,weight_decay=.0002)
    start=time.perf_counter();hist=[];best=float("inf");path=folder/(name+".pt")
    for j in range(steps+1):
        if j:
            b=helpers.batch(data,torch.as_tensor(rng.integers(len(data["side"]),size=batch),device=device))
            physical=opt.loss(model,b,float(rng.choice([0,.5,.75,1.,1.])),0.)[1]
            optimizer.zero_grad(set_to_none=True);physical.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.5);optimizer.step()
        if j%100==0 or j==steps:
            with torch.no_grad():
                vals=[]
                for ix in np.array_split(np.arange(len(val["side"])),8):
                    v=helpers.batch(val,torch.as_tensor(ix,device=device))
                    vals.append([float(opt.loss(model,v,f,0.)[1]) for f in (.33,1.)])
            score=float(np.mean(vals))
            hist.append(dict(step=j,validation=score,elapsed=time.perf_counter()-start))
            record=dict(spec=asdict(cfg),state=model.state_dict(),seed=seed,step=j,score=score,hist=hist,initial=initial)
            if j==0:torch.save(record,folder/(name+"-before.pt"))
            if score<best:
                best=score;tmp=path.with_suffix(".partial.pt");torch.save(record,tmp);os.replace(tmp,path)
            _write(folder/(name+"-history.json"),hist)
            print(name,j,round(score,4),round(best,4),round(time.perf_counter()-start,1),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder",default="symmetry-grounded-v1");p.add_argument("--data",default="symmetry-circuit-v2")
    p.add_argument("--changes",nargs="+",default=["delta","learned","fixed"])
    p.add_argument("--initial",nargs="+",default=["public","unrelated","random"])
    p.add_argument("--seeds",nargs="+",type=int,default=[7243,4187,9679]);p.add_argument("--steps",type=int,default=750)
    p.add_argument("--batch",type=int,default=72);p.add_argument("--device",default="cuda")
    args=p.parse_args();torch.set_num_threads(1)
    folder=OUT/args.folder;folder.mkdir(exist_ok=True)
    raw=source.data(OUT/args.data,5190761,6144,True)
    data=helpers.torch_data(raw,args.device)
    val=helpers.torch_data(source.data(OUT/args.data,3741649,768,True),args.device)
    for seed in args.seeds:
        for initial in args.initial:
            for change in args.changes:
                run(folder,raw,data,val,seed,change,initial,args.steps,args.batch,args.device)
    _write(folder/"complete.json",dict(seeds=args.seeds,changes=args.changes,initial=args.initial,steps=args.steps))


if __name__=="__main__":main()
