"""Does the historical per-write numeric bound itself explain loss of new learning?"""

from contextlib import ExitStack
from hashlib import sha256
import json
from unittest.mock import patch

import numpy as np
import torch

from . import interleave_plasticity as plastic, interleave_triangle_angular as nested
from . import local_trace_graph as net, local_trace_train as trained
from .interleave_interference import DATA,EVIDENCE,ROOT


RANGES={"original":(1.15,1.15),"double":(2.3,2.3),"fourfold":(4.6,4.6),
        "free_weights":(float("inf"),1.15),"free_bias":(1.15,float("inf")),
        "free_both":(float("inf"),float("inf")),"birth_envelope":(1.15,1.15)}


def run(label,source,arm,births,worlds,seed,archive):
    dest=DATA/"write-range"/label
    dest.mkdir(parents=True,exist_ok=True)
    hashes={};bank={k:[] for k in RANGES};born={}
    base_health=plastic.health
    for birth in births:
        g,r=trained.load_checkpoint(source,"odor",arm,birth,8000)
        w=g.weights[g.present]
        born[birth]=dict(edges=int(w.numel()),max_weight=float(w.abs().max()),
                         fraction_weights_above_native=float((w.abs()>1.15001).float().mean()))
        def envelope(old,delta,limit):
            initial=g.weights.abs() if old.ndim==3 else g.bias.abs()
            radius=initial.clamp_min(limit)
            return (old+delta).clamp(min=torch.minimum(old,-radius),max=torch.maximum(old,radius))
        def health(state,start):
            result=base_health(state,start)
            active=state.graph.present
            w=state.graph.weights.abs()
            result["above_historical_range"]=((w>1.15001)&active).sum((1,2))/active.sum((1,2)).clamp_min(1)
            result["maximum_weight"]=torch.where(active,w,0.).amax((1,2))
            return result
        for mode,(max_w,max_b) in RANGES.items():
            # These overrides apply only inside this frozen single-process replay.
            # At no point is a training process or an on-disk checkpoint changed.
            with ExitStack() as scope:
                scope.enter_context(patch.object(net,"WEIGHT_LIMIT",max_w))
                scope.enter_context(patch.object(net,"BIAS_LIMIT",max_b))
                scope.enter_context(patch.object(plastic,"BRANCHES",("intact",)))
                scope.enter_context(patch.object(plastic,"health",health))
                if mode=="birth_envelope":scope.enter_context(patch.object(net,"_bounded_update",envelope))
                z=plastic.evaluate(g,r,seed,birth,worlds,"novel",plastic.AGES)
            if archive and mode=="original":
                with np.load(DATA/"plasticity"/archive/f"novel-{birth}.npz") as prior:
                    for item in ("new_side","new_correct","old_correct","fresh_delta_l1"):
                        np.testing.assert_array_equal(z[item][:,0],prior[item][:,0])
            x=z["new_correct"][:,0]
            a=z["old_correct"][:,0]
            stats=np.stack((x.reshape(-1,worlds,2).mean(2),a.reshape(-1,worlds,2).mean(2),
                            (x&a).reshape(-1,worlds,2).mean(2),
                            z["above_historical_range"],z["maximum_weight"]),axis=-1)
            bank[mode].append(stats)
            f=dest/f"{birth}-{mode}.npz"
            np.savez_compressed(f,**z)
            hashes[f.name]=sha256(f.read_bytes()).hexdigest()
        print(label,birth,flush=True)
    values={k:np.stack(v) for k,v in bank.items()}
    measures=("new_X","old_after_X","both","above_native_range","maximum_weight")
    results={}
    for mode,table in list(values.items()):
        if mode!="original":values[mode+" minus original"]=table-values["original"]
    for mode,table in values.items():
        estimates=nested.intervals(table.transpose(0,2,1,3).reshape(len(births),worlds,-1))
        results[mode]={}
        for i,age in enumerate(plastic.AGES):
            cells={}
            for j,key in enumerate(measures):
                column=i*len(measures)+j
                cells[key]=dict(mean=estimates["mean"][column],ci95=[v[column] for v in estimates["ci95"]],
                                by_birth=[v[column] for v in estimates["by_birth"]])
            results[mode][str(age)]=cells
    return dict(births=list(births),worlds_per_birth=worlds,seed=seed,initial=born,scores=results,raw_sha256=hashes)


if __name__=="__main__":
    torch.set_num_threads(1)
    original=list(range(731,737));new=list(range(811,817))
    report=dict(
        notes="Same events, motor and structural draws, and no extra attraction to birth. Only numeric feasibility "
              "is changed in deployment. The birth-envelope arm permits each address to move back to its own "
              "inherited magnitude once it has shrunk below its birth value. Normal growth and retirement remain "
              "active. Independent histories vary across each training birth. Compare by paired world and body.",
        prior=run("prior",ROOT/"data/local-trace-20261119/local","active",original,96,584321,"prior-odor"),
        uniform=run("uniform",DATA/"retention-objective","uniform",original,64,584321,"objective-uniform-8000"),
        confirmation=run("confirmation",DATA/"retention-objective-confirmation","uniform",new,96,933611,None),
    )
    (EVIDENCE/"native-write-range.json").write_text(json.dumps(report,indent=1)+"\n",encoding="utf8")
