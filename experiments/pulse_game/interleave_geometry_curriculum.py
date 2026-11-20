"""Keep the controller, action objective and budget fixed while varying three-object study geometries."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import interleave_confirm_triangle as replication, interleave_triangle_train as tri
from . import interleave_checkpoint_publish as publish
from .interleave_interference import DATA, EVIDENCE


STAGES = (0, 8000, 32000, 64000)
ARMS = ("orthogonal", "interleaved")
PLAN = EVIDENCE / "geometry-curriculum-plan.json"


def geometry(arm, seed, step):
    if arm == "orthogonal":
        return arm
    if arm != "interleaved" or step < 1:
        raise ValueError("Invalid interleaving schedule")
    block, slot = divmod(step-1, 4)
    order = np.random.default_rng([817519, seed, block]).permutation(4)
    return ("hard", "hard", "orthogonal", "random")[order[slot]]


def tape(seed, birth, step, layout, worlds=4):
    trial = tri.make_tape(seed, birth, worlds, -.5 if layout=="hard" else 0., update=step)
    if layout=="random":
        rng=np.random.default_rng([4480027,seed,birth,step])
        cues=rng.normal(size=(worlds,3,3))
        cues=(1.35*cues/np.linalg.norm(cues,axis=2,keepdims=True)).astype(np.float32)
        trial["right"]=torch.from_numpy(np.repeat(cues,len(tri.LABELS),axis=0))
        trial["left"]=-trial["right"]
    return trial


def run(cohort, group, arm):
    torch.set_num_threads(1)
    original=cohort=="original"
    first=731 if original else 811
    births=list(range(first,first+3)) if group=="low" else list(range(first+3,first+6))
    seed=tri.TRAIN_SEED if original else replication.TRAIN_SEED
    source=DATA/("retention-objective" if original else "retention-objective-confirmation")
    root=DATA/"geometry-curriculum"/cohort
    folder=root/arm/group
    folder.mkdir(parents=True,exist_ok=True)
    out=folder/"resume.pt"
    g,r=tri._parents(source,births)
    opt=torch.optim.Adam((g.weights,g.bias,*r.parameters()),lr=.003)
    manifest=dict(cohort=cohort,arm=arm,group=births,seed=seed,worlds=4,stages=STAGES,
                  source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  plan=sha256(PLAN.read_bytes()).hexdigest(),parents={str(b):sha256(tri.old_fit.checkpoint_path(
                      source,"odor","uniform",b,8000).read_bytes()).hexdigest() for b in births})
    mark=folder/"plan.json"
    publish.locked_plan(mark,manifest,Path(__file__).stem,"source_sha256")
    step=0;trace=[];counts={k:0 for k in ("hard","random","orthogonal")}
    if out.exists():
        d=torch.load(out,map_location="cpu",weights_only=True)
        if d["manifest"]!=sha256(mark.read_bytes()).hexdigest():
            raise ValueError("Resume and protocol hashes differ")
        with torch.no_grad():
            for k in tri.revisit.capacity.GRAPH_FIELDS:
                getattr(g,k).copy_(d["graph"][k])
        r.load_state_dict(d["rule"]);opt.load_state_dict(d["optimizer"])
        step=d["step"];trace=d["trace"].tolist();counts=d["counts"]
    def save():
        obj=dict(step=step,counts=counts,trace=torch.tensor(trace),optimizer=opt.state_dict(),
                 manifest=sha256(mark.read_bytes()).hexdigest(),
                 graph={k:getattr(g,k).detach() for k in tri.revisit.capacity.GRAPH_FIELDS},rule=r.state_dict())
        if step in STAGES:
            for i,b in enumerate(births):
                tri.old_fit.save_checkpoint(root,"odor",arm,b,step,g,r,i)
        tri.revisit._atomic(out,lambda f:torch.save(obj,f))
    if not out.exists() or publish.needs_export(root,"odor",arm,births,step,STAGES):save()
    initial=step;start=perf_counter()
    for step in range(initial+1,STAGES[-1]+1):
        layout=geometry(arm,seed,step)
        parts=[tape(seed,b,step,layout) for b in births]
        merged={k:torch.cat([p[k] for p in parts]) for k in parts[0]}
        result=tri._step(g,r,opt,merged,4)
        counts[layout]+=1
        trace.append(result[:,[0,1,2,6,7]].tolist())
        if step%1000==0:
            save()
            print(cohort,group,arm,step,counts,round((perf_counter()-start)/(step-initial),3),flush=True)
    if arm=="interleaved" and counts!={"hard":32000,"random":16000,"orthogonal":16000}:
        raise ValueError("The balanced geometry schedule diverged")


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--cohort",choices=("original","independent"),required=True)
    p.add_argument("--group",choices=("low","high"),required=True)
    p.add_argument("--arm",choices=ARMS,required=True)
    a=p.parse_args()
    run(a.cohort,a.group,a.arm)
