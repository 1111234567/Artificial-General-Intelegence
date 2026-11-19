"""Execute matched recurrent/intact and zero-recurrence searches, one GPU job at a time."""

import argparse
import json
import sys
import time

import torch

from experiments.seed_evolution.large_runner import _write
from . import attractor_search as search


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--population",type=int,default=2048)
    ap.add_argument("--generations",type=int,default=160)
    ap.add_argument("--actors",type=int,default=48)
    ap.add_argument("--batch",type=int,default=32)
    ap.add_argument("--seeds",type=int,nargs="+",default=[149221,893611,437918])
    ap.add_argument("--conditions",nargs="+",default=["mixed","serial_noise"])
    ap.add_argument("--tag",default="overnight")
    args=ap.parse_args()
    start=time.perf_counter();done=[]
    logfile=search.OUT/("attractor-"+args.tag+"-status.json")
    for seed in args.seeds:
        for task in args.conditions:
            for cut in (False,True):
                name=f"attractor-{args.tag}-{task}-{'cut' if cut else 'full'}-{seed}"
                path=search.OUT/name/"search.json"
                _write(logfile,{"active":name,"finished":done,"elapsed":time.perf_counter()-start})
                if not path.exists() or len(json.loads(path.read_text())["history"])<args.generations:
                    sys.argv=[__file__,"--name",name,"--population",str(args.population),"--generations",str(args.generations),
                              "--actors",str(args.actors),"--batch",str(args.batch),"--difficulty",task,"--seed",str(seed),
                              "--validate-every","8"]+(["--cut"] if cut else [])
                    search.main()
                result=json.loads(path.read_text());done.append({"name":name,"best":result["best"],
                                                               "generations":len(result["history"])})
                torch.cuda.empty_cache()
    _write(logfile,{"active":None,"finished":done,"elapsed":time.perf_counter()-start})


if __name__=="__main__":
    main()
