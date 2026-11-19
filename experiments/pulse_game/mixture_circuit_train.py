"""Train a private-taste predictor from physically sampled public rewards and frozen prefixes."""
import argparse
from dataclasses import asdict
import os
import time
import numpy as np
import torch

from . import mixture_circuit as circuit, mixture_flash as flash, research_audit as audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


def torch_data(bank,device):
    return {k:torch.as_tensor(v,device=device) for k,v in bank.items()}


def batch(data,rows):
    return {k:v[rows] for k,v in data.items()}


def early_times(tape,frac):
    begin=torch.maximum(tape["ot"],tape["ct"]).long()
    span=(tape["touch"].long()-begin).clamp(min=0)
    return begin+(frac*span).round().long()


def objective(model,part,h,fraction):
    output=model.value(part,h,early_times(part,fraction))
    return (output-part["reward"][:,h:]).square().mean()


def load(path,cut=None):
    saved=torch.load(path,map_location="cpu",weights_only=True)
    model=circuit.TasteCircuit(circuit.Config(**saved["config"]))
    model.load_state_dict(saved["state"])
    return model.export(cut)


def train(cfg,seed,data,val,args,folder):
    torch.manual_seed(seed)
    name=f"{cfg.graph}-{cfg.learner}-s{seed}"
    model=circuit.TasteCircuit(cfg).to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=.0026,weight_decay=.00025)
    rng=np.random.default_rng(seed)
    path=folder/(name+".pt")
    checkpoint=folder/(name+"-progress.pt")
    history=[]
    best=float("inf")
    step0=0
    def state(step):return dict(config=asdict(cfg),state=model.state_dict(),seed=seed,step=step,history=history,
                                score=float(best),training_lives=len(data["side"]),objective="sampled physical taste prediction")
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location=args.device,weights_only=False)
        model.load_state_dict(saved["state"]);opt.load_state_dict(saved["opt"])
        rng.bit_generator.state=saved["rng"]
        history=saved["history"];best=saved["score"];step0=saved["step"]
    start=time.perf_counter()
    for step in range(step0+1,args.updates+1):
        ids=torch.as_tensor(rng.integers(len(data["side"]),size=args.batch),device=args.device)
        part=batch(data,ids)
        h=int(rng.choice([8,16,32,48,64]))
        fraction=float(rng.choice([.0,.33,.67,1.,1.]))
        loss=objective(model,part,h,fraction)
        opt.zero_grad(set_to_none=True);loss.backward()
        grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.5))
        if not np.isfinite(grad):raise ValueError((step,loss,grad))
        opt.step()
        if step==1 or step%100==0 or step==args.updates:
            with torch.no_grad():
                # Interleave all private environment families in each slice to
                # avoid overweighting one of them in checkpoint selection.
                vals=[]
                for ids_np in np.array_split(np.arange(len(val["side"])),12):
                    v=batch(val,torch.as_tensor(ids_np,device=args.device))
                    vals.append([float(objective(model,v,h,frac)) for h,frac in ((16,.33),(48,1.))])
                score=float(np.mean(vals))
            history.append(dict(step=step,validation=score,train=float(loss),seconds=time.perf_counter()-start))
            if score<best:
                best=score
                tmp=path.with_suffix(".partial.pt");torch.save(state(step),tmp);os.replace(tmp,path)
            tmp=checkpoint.with_suffix(".partial.pt")
            torch.save({**state(step),"opt":opt.state_dict(),"rng":rng.bit_generator.state},tmp);os.replace(tmp,checkpoint)
            _write(folder/(name+"-history.json"),history)
            print(name,step,"new sign error",round(score,5),"best",round(best,5),
                  "seconds",round(time.perf_counter()-start,1),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder",default="mixture-circuit-v1")
    p.add_argument("--graphs",nargs="+",default=["rotations","diagonal","stateless"])
    p.add_argument("--learners",nargs="+",default=["local","ridge","delta"])
    p.add_argument("--seeds",nargs="+",type=int,default=[3187,6947])
    p.add_argument("--device",default="cuda")
    p.add_argument("--batch",type=int,default=48);p.add_argument("--updates",type=int,default=900)
    p.add_argument("--lives",type=int,default=3072)
    args=p.parse_args()
    torch.set_num_threads(1)
    folder=OUT/args.folder;folder.mkdir(exist_ok=True)
    data=torch_data(flash.public_examples(folder,8617399,args.lives,80),args.device)
    val=torch_data(flash.public_examples(folder,9615741,384,80),args.device)
    for seed in args.seeds:
        for g in args.graphs:
            for l in args.learners:
                cfg=circuit.Config(graph=g,learner=l)
                train(cfg,seed,data,val,args,folder)
    _write(folder/"finished.json",dict(seeds=args.seeds,graphs=args.graphs,learners=args.learners,
           steps=args.updates,lives=args.lives,encoder_sha=audit.fingerprint(circuit.__file__)))


if __name__=="__main__":main()
