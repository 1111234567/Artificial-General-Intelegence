"""Paired forward+backward timing of an algebraically equivalent causal training implementation."""

from pathlib import Path
import json
import statistics
import time

import numpy as np
import torch

from experiments.seed_evolution.large_runner import _write
from . import ring_metric as model, research_audit as audit


def main():
    torch.set_num_threads(1)
    folder=Path("data/pulse-game/ring-meta-visible-balanced")
    checkpoint=torch.load(folder/"best.pt",map_location="cpu",weights_only=True)
    with np.load(folder/"initial-experiences.npz") as file:
        arrays=[file[k][:96].copy() for k in ("x","y","valid")]
    report={"shape":list(arrays[0].shape),"unmasked_events":int(arrays[2].sum()),"devices":{}}
    for device in ("cpu","cuda"):
        net=model.Metric(model.Settings(**checkpoint["settings"])).to(device)
        net.load_state_dict(checkpoint["model"])
        x,y,valid=[torch.from_numpy(v).to(device) for v in arrays]
        def one(parallel):
            net.zero_grad(set_to_none=True)
            predicted=net.predict_events(x,y,valid,parallel=parallel)
            loss=((y-predicted).square()*valid).sum()/valid.sum()
            loss.backward()
            if device=="cuda":torch.cuda.synchronize()
            return predicted.detach()
        a=one(False).cpu()
        old=[p.grad.cpu().clone() for p in net.parameters() if p.grad is not None]
        b=one(True).cpu()
        new=[p.grad.cpu().clone() for p in net.parameters() if p.grad is not None]
        error=float((a-b).abs().max());gerr=max(float((u-v).abs().max()) for u,v in zip(old,new,strict=True))
        elapsed={False:[],True:[]}
        for i in range(16):
            for flag in ((False,True) if i%2==0 else (True,False)):
                start=time.perf_counter();one(flag);elapsed[flag].append(time.perf_counter()-start)
        ratios=np.array(elapsed[False])/np.array(elapsed[True])
        selection=np.random.default_rng(67).integers(16,size=(6000,16))
        out={"max_value_abs_error":error,"max_gradient_abs_error":gerr,
             "sequential_ms":1000*statistics.median(elapsed[False]),"parallel_ms":1000*statistics.median(elapsed[True]),
             "speedup":float(np.median(ratios)),"ci95":np.quantile(np.median(ratios[selection],axis=1),[.025,.975]).tolist(),
             "raw_seconds":{str(k):v for k,v in elapsed.items()}}
        report["devices"][device]=out; print(device,{k:v for k,v in out.items() if k!="raw_seconds"},flush=True)
    _write(audit.OUT/"causal-solve-performance.json",report)


if __name__=="__main__":
    main()
