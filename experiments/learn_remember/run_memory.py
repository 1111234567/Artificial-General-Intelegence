"""Apply the same preregistered matched-memory test as GRU stages arrive."""
import argparse
from pathlib import Path
import time

import torch

from experiments.learn_remember import memory_assay as assay
from experiments.learn_remember.gru_train import load_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,choices=range(3),required=True)
    parser.add_argument("--root",type=Path,default=assay.BASE/"gru-v1")
    parser.add_argument("--wait",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(1)
    for stage in ("random","short","wide","withdrawal_mid","withdrawal"):
        if (assay.OUT/f"{stage}_s{args.seed}.json").exists():
            continue
        checkpoint=args.root/f"seed_{args.seed:03d}/{stage}.pt"
        manifest=checkpoint.with_suffix(".json")
        limit=time.monotonic()+3600
        # The sidecar is emitted only after torch.save has closed and been hashed.
        while args.wait and not manifest.exists() and time.monotonic()<limit:
            time.sleep(5)
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        model,_,_,_=load_checkpoint(checkpoint)
        assay.run(args.seed,stage,model,model.initial_state(8),checkpoint)


if __name__=="__main__":
    main()
