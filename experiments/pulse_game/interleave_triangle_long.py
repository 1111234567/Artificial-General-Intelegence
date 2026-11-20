"""Follow the unchanged hard-triangle optimizer to a fixed 64k endpoint in both cohorts."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json
import time

import numpy as np
import torch

from . import interleave_triangle_train as tri, interleave_triangle_extension as extended
from . import interleave_confirm_triangle as replication
from .interleave_interference import DATA


def run(cohort,group,follow):
    torch.set_num_threads(1)
    first=cohort=="original"
    roots=list(range(731,737)) if first else list(range(811,817))
    births=roots[:3] if group=="low" else roots[3:]
    folder=DATA/f"triangle-long-{cohort}"
    warm=DATA/("retention-objective" if first else "retention-objective-confirmation")
    if first:
        parent=extended.group_checkpoint(DATA/"triangle-extension","same",births,32000)
        reference=DATA/"triangle-extension";parent_arm="same";world_seed=tri.TRAIN_SEED
    else:
        parent=DATA/"triangle-confirmation"/("group-"+"-".join(map(str,births)))/"resume.pt"
        reference=DATA/"triangle-confirmation";parent_arm="direct";world_seed=replication.TRAIN_SEED
    required=[tri.revisit._checkpoint(reference,parent_arm,b,32000) for b in births]
    if follow:
        while not (parent.exists() and all(p.exists() for p in required)):time.sleep(15)
    if not all(p.exists() for p in required):raise FileNotFoundError("Parent endpoint incomplete")
    start_state=torch.load(parent,weights_only=True,map_location="cpu")
    if start_state["step"]!=32000 or list(start_state["births"])!=births:
        raise ValueError("Need the exact 32k Adam state")
    g,r=tri._parents(warm,births);opt=torch.optim.Adam((g.weights,g.bias,*r.parameters()),lr=.003)
    subdir=folder/("group-"+"-".join(map(str,births)));subdir.mkdir(parents=True,exist_ok=True)
    manifest=dict(cohort=cohort,births=births,training_seed=world_seed,from_step=32000,to_step=64000,
                  optimizer_parent_sha256=sha256(parent.read_bytes()).hexdigest(),
                  parent_files={str(b):sha256(p.read_bytes()).hexdigest() for b,p in zip(births,required,strict=True)},
                  source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  note="At 32k the original, ordinary uniform reward arm still rose versus 16k. Extend it unchanged, "
                       "and specify the same fixed independent 64k test before the independent 32k data are inspected. "
                       "No new nonlinear operation or reward weighting; continue Adam including its moments.")
    meta=subdir/"plan.json"
    if meta.exists() and json.loads(meta.read_text())!=manifest:raise ValueError("Changed extension protocol")
    tri.revisit._write_json(meta,manifest)
    resume=subdir/"resume.pt"
    d=torch.load(resume,weights_only=True,map_location="cpu") if resume.exists() else start_state
    with torch.no_grad():
        for k in ("weights","bias","present"):getattr(g,k).copy_(d["graph"][k])
    r.load_state_dict(d["rule"]);opt.load_state_dict(d["optimizer"])
    step=d["step"];log=d["extension_trace"].tolist() if resume.exists() else []
    if not all(int(o["step"])==step for o in opt.state_dict()["state"].values()):
        raise ValueError("Adam step and graph stage differ")
    def save():
        payload=dict(births=births,step=step,graph={k:getattr(g,k).detach() for k in tri.revisit.capacity.GRAPH_FIELDS},
                     rule=r.state_dict(),optimizer=opt.state_dict(),extension_trace=torch.as_tensor(log))
        tri.revisit._atomic(resume,lambda f:torch.save(payload,f))
        if step in (32000,40000,48000,56000,64000):
            for i,b in enumerate(births):tri.old_fit.save_checkpoint(folder,"odor","direct",b,step,g,r,i)
    if not resume.exists():
        save()
        for b in births:
            before,br=tri.old_fit.load_checkpoint(reference,"odor",parent_arm,b,32000)
            after,ar=tri.old_fit.load_checkpoint(folder,"odor","direct",b,32000)
            if not all(torch.equal(getattr(before,k),getattr(after,k)) for k in tri.revisit.capacity.GRAPH_FIELDS):
                raise AssertionError("Restart graph changed")
            if not all(torch.equal(v,ar.state_dict()[k]) for k,v in br.state_dict().items()):
                raise AssertionError("Restart rule changed")
    initial=step;started=time.perf_counter()
    for step in range(initial+1,64001):
        parts=[tri.make_tape(world_seed,b,4,-.5,update=step) for b in births]
        tape={k:torch.cat([sample[k] for sample in parts]) for k in parts[0]}
        v=tri._step(g,r,opt,tape,4)[:,[0,1,2,6,7]]
        log.append(v.tolist())
        if step%1000==0:
            save()
            print(cohort,births,step,np.mean(log[-100:],axis=(0,1)).round(3),
                  round((time.perf_counter()-started)/(step-initial),3),flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--cohort",choices=("original","independent"),required=True)
    p.add_argument("--group",choices=("low","high"),required=True)
    p.add_argument("--follow",action="store_true")
    a=p.parse_args();run(a.cohort,a.group,a.follow)
