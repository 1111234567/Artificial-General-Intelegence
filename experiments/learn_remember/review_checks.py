"""Test review endpoints on the same public cases and the fixed memory assay."""
import argparse
import json
import time

import torch

from experiments.learn_remember import evaluate_symmetric, gru_train, memory_assay as m, symmetric_gru


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",required=True,type=int,choices=range(3))
    p.add_argument("--symmetric",action="store_true")
    args=p.parse_args()
    torch.set_num_threads(1)
    folder=(symmetric_gru.OUT if args.symmetric else gru_train.OUT)/f"seed_{args.seed:03d}"
    marker=folder/("review.symmetry.json" if args.symmetric else "review.json")
    cap=time.monotonic()+3600
    while not marker.exists() and time.monotonic()<cap:
        time.sleep(5)
    cp=folder/"review.pt"
    record=json.loads(marker.read_text(encoding="utf-8"))
    assert m.sha(cp)==record.get("checkpoint_sha256", record.get("sha256"))
    if args.symmetric:
        evaluate_symmetric.run(args.seed,"review")
    else:
        model,_,_,_=gru_train.load_checkpoint(cp)
        m.run(args.seed,"review",model,model.initial_state(8),cp)
