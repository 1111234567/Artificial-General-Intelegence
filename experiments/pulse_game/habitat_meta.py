"""Learn where a higher-order reward association should be written from real tastes."""
import argparse
from dataclasses import asdict,dataclass
import os
import time
import numpy as np
import torch
from torch import nn
from . import habitat_world as game, symmetry_world, symmetry_grounded as ground, symmetry_memory as event
from . import mixture_circuit_train as helpers, mixture_metric, research_audit as audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


@dataclass(frozen=True)
class Config:
    routing:str="learned"
    slots:int=6
    nodes:int=24
    width:int=40
    retention:float=1.
    updater:str="delta"


class MetaSynapses(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        self.first=nn.Linear(cfg.nodes,cfg.width)
        self.second=nn.Linear(cfg.width,cfg.slots)
        self.rate=nn.Parameter(torch.tensor(.25))
        self.scale=nn.Parameter(torch.tensor(.45))
        if cfg.updater=="rls":self.log_ridge=nn.Parameter(torch.tensor(-1.4))
        elif cfg.updater!="delta":raise ValueError(cfg.updater)
        if cfg.routing not in ("single","public","shared_public","learned","random"):raise ValueError(cfg.routing)
        if cfg.routing=="shared_public":self.log_shared=nn.Parameter(torch.tensor(-.35))
        if cfg.routing in ("random","single","public","shared_public"):
            for m in (self.first,self.second):m.requires_grad_(False)

    def code(self,state,ctx):
        if self.cfg.routing=="single":return torch.ones_like(ctx[...,:1])
        if self.cfg.routing=="public":return nn.functional.normalize(ctx,dim=-1)
        if self.cfg.routing=="shared_public":
            shared=torch.ones_like(ctx[...,:1])*self.log_shared.exp()
            return nn.functional.normalize(torch.cat((ctx,shared),-1),dim=-1)
        h=self.first(state).tanh()
        return nn.functional.normalize(self.second(h),dim=-1,eps=1e-7)

    def forward(self,s,ce,read_s,recalled_at,recalled_now,rewards):
        x=self.code(s,ce)
        q=self.code(read_s,ce)
        feature=x*recalled_at[:,:,None]
        eta=.025+1.5*self.rate.sigmoid()
        t=torch.arange(s.shape[1],device=s.device)
        lag=t[:,None]-t[None,:]-1
        decay=self.cfg.retention**lag.clamp(min=0)*(lag>=0)
        if self.cfg.updater=="rls":
            eye=torch.eye(s.shape[1],device=s.device,dtype=s.dtype)
            ridge=.015+self.log_ridge.exp()
            lower=torch.linalg.cholesky(feature@feature.transpose(1,2)+ridge*eye)
            innovations=torch.linalg.solve_triangular(lower,rewards.unsqueeze(2),upper=False).squeeze(2)
            # Every leading Cholesky block is the exact factor of its prefix.
            # Reject the current and all future innovations when forming a prediction.
            cross=torch.linalg.solve_triangular(lower,feature@q.transpose(1,2),upper=False).transpose(1,2)
            pre=(cross.tril(-1)*innovations[:,None,:]).sum(2)
        else:
            gram=(feature@feature.transpose(1,2))*eta*decay
            err=torch.linalg.solve_triangular(gram+torch.eye(s.shape[1],device=s.device),rewards.unsqueeze(2),
                                              upper=False,unitriangular=True)
            pre=((q@feature.transpose(1,2)*eta*decay)@err).squeeze(2)
        amp=.40+4.*self.scale.sigmoid()
        return (pre*amp).tanh()*recalled_now

    def export(self):
        def layer(a):return a.weight.T.detach().cpu().numpy().copy(),a.bias.detach().cpu().numpy().copy()
        return dict(routing=self.cfg.routing,first=layer(self.first),second=layer(self.second),
                    eta=float((.025+1.5*self.rate.sigmoid()).detach()),amp=float((.40+4.*self.scale.sigmoid()).detach()),
                    retention=self.cfg.retention,updater=self.cfg.updater,
                    ridge=float((.015+self.log_ridge.exp()).detach()) if self.cfg.updater=="rls" else None,
                    shared=float(self.log_shared.exp().detach()) if self.cfg.routing=="shared_public" else 0.,
                    slots={"single":1,"public":2,"shared_public":3}.get(self.cfg.routing,self.cfg.slots))


def raw(folder,seed,n):
    path=folder/f"sampled-{seed}-{n}.npz"
    if path.exists():
        with np.load(path) as a:return {k:a[k] for k in a.files}
    bag={};rng=np.random.default_rng(seed)
    for off in range(0,n,96):
        ss=rng.integers(10_000_000,470_000_000,min(96,n-off)).tolist()
        # Equal total durations; variable private task lengths and teaching order.
        a=int(rng.choice([6,8,10,12,14]));b=20-a
        w=game.make(ss,pairs=(a,b),span=int(rng.choice([2,4,6])),visits=4,noise=float(rng.uniform(.0,.13)),
                    jitter=float(rng.uniform(.04,.15)),appearance=rng.choice(["discrete","smooth"]))
        explorer=mixture_metric.Explorer(ss,w.menu.body.trials);res=game.rollout(explorer,w,record=True)
        # The helper uses only observed pulses, touched side/time and actual taste.
        public=symmetry_world.public_rows(w,res)
        for k,v in public.items():bag.setdefault(k,[]).append(v)
    bank={k:np.concatenate(v) for k,v in bag.items()}
    tmp=path.with_suffix(".partial.npz");np.savez_compressed(tmp,**bank);os.replace(tmp,path)
    return bank


def receptor(path):
    saved=torch.load(path,map_location="cpu",weights_only=True)
    network=ground.Grounded(event.Config(**saved["spec"]));network.load_state_dict(saved["state"])
    return network


def precompute(net,raw,device):
    # These states are action-independent conditional on the recorded trajectory.
    # Only actually chosen bites participate in memory retrievability.
    data=helpers.torch_data(raw,device)
    net=net.to(device)
    bag={}
    with torch.no_grad():
        for ids in np.array_split(np.arange(len(data["side"])),max(1,(len(data["side"])+63)//64)):
            b=helpers.batch(data,torch.as_tensor(ids,device=device))
            sel=b["side"].long()[:,:,None,None].expand(-1,-1,1,net.cfg.nodes)
            t=b["touch"].long();s=net.sensor.states(b,t)
            q,k=net.heads(s.gather(2,sel).squeeze(2))
            y=b["reward"].float()
            remembered,_,_=net.retrieved(q,k,y)
            chunk=dict(s=s.mean(2),ctx=b["ctx"],r=remembered,y=y,
                       free=(b["odor"].diff(dim=2).abs().sum((2,3))>.01))
            for phase,f in enumerate((0.,.33,.66,1.)):
                state=net.sensor.states(b,helpers.early_times(b,f))
                query,_=net.heads(state.gather(2,sel).squeeze(2))
                retrieved,_,_=net.retrieved(query,k,y)
                chunk[f"s{phase}"]=state.mean(2);chunk[f"r{phase}"]=retrieved
            for k,v in chunk.items():bag.setdefault(k,[]).append(v)
    return {k:torch.cat(parts) for k,parts in bag.items()}


def objective(net,b,phase):
    pred=net(b["s"],b["ctx"],b[f"s{phase}"],b["r"],b[f"r{phase}"],b["y"])
    err=(pred-b["y"]).square()
    return .76*err[b["free"]].mean()+.24*err[~b["free"]].mean()


def load(path):
    saved=torch.load(path,map_location="cpu",weights_only=True)
    net=MetaSynapses(Config(**saved["config"]));net.load_state_dict(saved["state"])
    return net.export()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder",default="habitat-meta-v1");p.add_argument("--lives",type=int,default=4608)
    p.add_argument("--seeds",nargs="+",type=int,default=[2791,4679,9341]);p.add_argument("--steps",type=int,default=1400)
    p.add_argument("--routes",nargs="+",default=["learned","single","public","random"])
    p.add_argument("--device",default="cuda");p.add_argument("--updater",choices=["delta","rls"],default="delta")
    p.add_argument("--data-folder")
    args=p.parse_args();torch.set_num_threads(1)
    folder=OUT/args.folder;folder.mkdir(exist_ok=True)
    # Frozen sensory receptor chosen from the public-only initialization,
    # before any reward-dependent fine tuning on the one-habitat game.
    source=OUT/"symmetry-grounded-v1/public-delta-s7243-before.pt"
    sensor=receptor(source)
    data_folder=OUT/args.data_folder if args.data_folder else folder
    data=precompute(sensor,raw(data_folder,8491531,args.lives),args.device)
    valid=precompute(sensor,raw(data_folder,5947713,768),args.device)
    print("folded fixed public trajectories",len(data["s"]),len(valid["s"]),flush=True)
    for seed in args.seeds:
        for r in args.routes:
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            cfg=Config(routing=r,updater=args.updater)
            model=MetaSynapses(cfg).to(args.device)
            opt=torch.optim.AdamW(model.parameters(),lr=.0027,weight_decay=.0002)
            name=f"{r}-s{seed}";path=folder/(name+".pt");history=[];best=float("inf");start=time.perf_counter()
            for i in range(args.steps+1):
                if i:
                    b=helpers.batch(data,torch.as_tensor(rng.integers(len(data["s"]),size=128),device=args.device))
                    loss=objective(model,b,int(rng.integers(4)))
                    opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.5);opt.step()
                if i%100==0 or i==args.steps:
                    with torch.no_grad():
                        vals=[float(objective(model,helpers.batch(valid,torch.as_tensor(ix,device=args.device)),phase))
                              for ix in np.array_split(np.arange(len(valid["s"])),6) for phase in (1,3)]
                    v=float(np.mean(vals))
                    history.append(dict(step=i,score=v,time=time.perf_counter()-start))
                    if v<best:
                        best=v;tmp=path.with_suffix(".partial.pt")
                        torch.save(dict(config=asdict(cfg),state=model.state_dict(),step=i,validation=v,seed=seed,
                                        public_receptor=source.as_posix(),receptor_sha=audit.fingerprint(source)),tmp)
                        os.replace(tmp,path)
                    _write(folder/(name+"-history.json"),history)
                    print(name,i,round(v,4),round(best,4),round(time.perf_counter()-start,1),flush=True)
    _write(folder/"complete.json",dict(seeds=args.seeds,forms=args.routes,steps=args.steps))


if __name__=="__main__":main()


