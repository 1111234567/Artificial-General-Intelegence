"""Use the plain GRU's exact common external evaluation worlds for the tied model."""
import argparse
from collections import Counter
import json
import time

import numpy as np
import torch

from experiments.learn_remember import evaluate_gru as ev, memory_assay as assay, symmetric_gru as sym
from experiments.learn_remember.gru_train import load_checkpoint
from experiments.lifelong.world import ContextWorld


def run(seed,stage):
    folder=sym.OUT/f"seed_{seed:03d}"
    p=folder/(stage+".pt")
    meta=json.loads((folder/(stage+".symmetry.json")).read_text(encoding="utf-8"))
    assert ev.sha(p)==meta["checkpoint_sha256"]
    model,_,_,_=load_checkpoint(p)
    model.eval()
    hh=model.core.weight_hh_l0.detach().reshape(3,model.width,model.width)
    assert torch.equal(hh,hh.transpose(-1,-2))
    plan=ev.checked()
    common=ev.common_stream(seed,plan)
    world=ContextWorld.from_state_dict(common["world"],device="cpu")
    hidden=ev.warmup(model,common)
    records=[]
    for condition in plan["conditions"]:
        for family in plan["families"]:
            scores,frames,_=ev.rollout(model,world,hidden,family=family,condition=condition,steps=plan["steps"])
            baseline=ev.read(ev.OUT/f"seed_{seed:03d}/random_common_{condition}_family_{family}.json")
            for key in ("start_world_hash","end_event_hash","end_scheduler_hash","primary_identity_hash"):
                assert scores[key]==baseline[key], (family,condition,key)
            assert np.array_equal(frames["logits"].argmax(-1),frames["actions"])
            facts=folder/f"{stage}_{condition}_f{family}.facts.npz"
            np.savez_compressed(facts,**{k:v for k,v in frames.items() if k not in ("obs","logits")})
            scores.update(facts_sha=ev.sha(facts),matched_reference_sha=ev.sha(ev.OUT/f"seed_{seed:03d}/random_common_{condition}_family_{family}.json"))
            records.append(scores)
    result=dict(seed=seed,stage=stage,model_sha=ev.sha(p),script_sha=ev.sha(__file__),
        evaluation_sha=ev.sha(ev.__file__),source_common_sha=ev.sha(ev.OUT/f"common_seed_{seed:03d}.pt"),records=records)
    with (folder/f"behavior_{stage}.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,separators=(",",":"))
    assay.run(seed,"sym_"+stage,model,model.initial_state(8),p)


if __name__=="__main__":
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--seed",required=True,type=int,choices=range(3))
    a.add_argument("--wait",action="store_true")
    args=a.parse_args()
    torch.set_num_threads(1)
    for stage in ("wide","withdrawal"):
        folder=sym.OUT/f"seed_{args.seed:03d}"
        cap=time.monotonic()+3600
        while args.wait and not (folder/(stage+".symmetry.json")).is_file() and time.monotonic()<cap:
            time.sleep(5)
        run(args.seed,stage)
