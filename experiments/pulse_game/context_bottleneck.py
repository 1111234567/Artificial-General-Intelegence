"""Factor a public context sensory bottleneck and prolonged conflicting experience at matched physical budgets."""
from dataclasses import asdict
import argparse
import json
import shutil
import time
from types import SimpleNamespace
import numpy as np
import torch

from . import unified_evolution as evolution,unified_train as train,habitat_meta as neural,unified_physical as body
from . import unified_audit as baseline,unified_world as game,unified_confirm as previous
from . import habitat_world as original,habitat_deploy as brain,research_audit as audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


def context_decoder(data,val):
    # The target is the real, two-current context flash from this physical
    # request. All sampled times are after it was observed and before contact.
    def gather(b):
        x=torch.cat([b[f"s{i}"].reshape(-1,b["s"].shape[-1]) for i in range(4)])
        y=b["ctx"].reshape(-1,2).repeat(4,1)
        return x,y
    x,y=gather(data);x=x.double();y=y.double()
    mx,my=x.mean(0),y.mean(0)
    xc=x-mx;yc=y-my
    w=torch.linalg.solve(xc.T@xc+.00001*len(x)*torch.eye(x.shape[1],device=x.device,dtype=x.dtype),xc.T@yc)
    b=my-mx@w
    vx,vy=gather(val);pred=vx@w.float()+b.float()
    return w.float(),b.float(),dict(heldout_public_mse=float((pred-vy).square().mean()),
                                   copy_global_mean=float((vy-my.float()).square().mean()))


def reduced(data,w,b):
    return {k:(v@w+b if k=="s" or (len(k)==2 and k[0]=="s" and k[1].isdigit()) else v) for k,v in data.items()}


def prepare_heads(folder,seeds):
    seeds=list(seeds)
    if all((folder/f"bottleneck-{s}.pt").exists() for s in seeds):return
    source=OUT/"symmetry-grounded-v1/public-delta-s7243-before.pt";sensor=neural.receptor(source)
    original_data=OUT/"unified-transfer-v1"
    data=train.precompute(sensor,train.raw(original_data,12492371,6400),"cuda")
    val=train.precompute(sensor,train.raw(original_data,31190571,960),"cuda")
    w,b,check=context_decoder(data,val);data=reduced(data,w,b);val=reduced(val,w,b)
    _write(folder/"public_context_fit.json",dict(result=check,weight=w.cpu().tolist(),bias=b.cpu().tolist(),
           source=audit.fingerprint(source),objective="reconstruct actually observed public currents"))
    temp=folder/"two_dimensional";temp.mkdir(exist_ok=True)
    for s in seeds:
        cfg=neural.Config(nodes=2,routing="learned",updater="rls")
        train.train(data,val,s,cfg,SimpleNamespace(steps=2200,device="cuda"),temp,source)
        saved=torch.load(temp/f"rls-learned-s{s}.pt",map_location="cpu",weights_only=True)
        state=saved["state"]
        fac=state["first.weight"]
        folded={**state,"first.weight":fac@w.cpu().T,"first.bias":state["first.bias"]+fac@b.cpu()}
        # Identical deployment interface and timing; no typed context channel
        # reaches the online address through a shortcut.
        torch.save({**saved,"config":asdict(neural.Config(routing="learned",updater="rls")),
                    "state":folded,"public_context_fit":check},folder/f"bottleneck-{s}.pt")
        before=neural.MetaSynapses(cfg);before.load_state_dict(state)
        after=neural.MetaSynapses(neural.Config(routing="learned",updater="rls"));after.load_state_dict(folded)
        y=torch.randn(47,24);ctx=torch.randn(47,2)
        torch.testing.assert_close(before.code(y@w.cpu()+b.cpu(),ctx),after.code(y,ctx),rtol=1e-4,atol=1e-5)


def extended(seed,g,lives):
    short,spec,roots=evolution.env_at(seed,g,lives)
    if g%2:return short,spec,roots
    rng=np.random.default_rng(np.random.SeedSequence([seed,g,119823]))
    length=int(rng.choice([32,48,64,80]))
    spec={**spec,"depths":(0,4,12,length),"span":2}
    long_total=2*(spec["initial"]+length+len(spec["depths"])+1)
    target=lives*short.arena.menu.body.trials
    n=max(8,int(round(target/long_total)))
    roots=roots[:n]
    out=game.make(roots,**spec)
    assert out.arena.menu.body.trials==long_total
    return out,{**spec,"short_equivalent_requests":target,"actual_requests":len(roots)*long_total},roots


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--gens",type=int,default=128)
    ap.add_argument("--width",type=int,default=768);args=ap.parse_args();torch.set_num_threads(1)
    root=OUT/"old-habitat-repair-v1";root.mkdir(exist_ok=True);settings=SimpleNamespace(gens=args.gens,width=args.width,lives=80)
    ids=(1373,8539,6547);prepare_heads(root,ids)
    for s,es in zip(ids,(39461,81439,95617),strict=True):
        f,_,motion=baseline.resources(OUT/"unified-transfer-v1",f"rls-learned-s{s}")
        for kind in ("full_state","bottleneck"):
            path=(root/f"bottleneck-{s}.pt" if kind=="bottleneck" else OUT/f"unified-transfer-v1/rls-learned-s{s}.pt")
            route=neural.load(path)
            for name,stream in (("ordinary",None),("prolonged",extended)):
                label=f"{kind}-{name}-s{s}"
                evolution.optimize(root,label,es,"full",settings,frozen_inputs=(f,route,motion),model_path=path,environment_stream=stream)
    _write(root/"complete.json",dict(seeds=list(ids),gen=args.gens,width=args.width))


if __name__=="__main__":main()
