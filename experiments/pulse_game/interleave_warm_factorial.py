"""Repeat the matched recency/dose separations from a competent multi-object initialization."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json

import torch

from . import interleave_revisit_factorial as study, local_trace_train as old
from .interleave_interference import DATA,EVIDENCE


def prepare(export,source,births):
    export.mkdir(parents=True,exist_ok=True)
    manifest=dict(meaning="Loader adapter: these exact graphs and rules came from the 8000-step uniform-query "
                  "multi-object fit (not the prior single-object fit). The old loader expects the storage role "
                  "odor/active. The graph, mask, bias and every inherited rule coefficient are verified equal.",
                  births=births,donors={},exports={})
    for b in births:
        g,r=old.load_checkpoint(source,"odor","uniform",b,8000)
        raw=old.checkpoint_path(source,"odor","uniform",b,8000)
        encoded=old.checkpoint_path(export,"odor","active",b,8000)
        if not encoded.is_file():
            old.save_checkpoint(export,"odor","active",b,8000,g,r,0)
        actual,actual_rule=old.load_checkpoint(export,"odor","active",b,8000)
        if not all(torch.equal(getattr(g,k),getattr(actual,k)) for k in ("weights","bias","present")):
            raise AssertionError("Warm graph export changed graph values")
        if not all(torch.equal(v,actual_rule.state_dict()[k]) for k,v in r.state_dict().items()):
            raise AssertionError("Warm export changed write rule")
        manifest["donors"][str(b)]=sha256(raw.read_bytes()).hexdigest()
        manifest["exports"][str(b)]=sha256(encoded.read_bytes()).hexdigest()
    study.revisit._write_json(export/"export.json",manifest)


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--cohort",choices=("original","confirmation"),required=True)
    p.add_argument("--workers",type=int,default=2)
    args=p.parse_args()
    original=args.cohort=="original"
    roots=list(range(731,737) if original else range(811,817))
    source=DATA/("retention-objective" if original else "retention-objective-confirmation")
    out=DATA/("factorial-retrained-"+args.cohort)
    export=out/"uniform-parent"
    prepare(export,source,roots)
    config=study.Config(world_seed=431987 if original else 431988)
    study.run(out,export,EVIDENCE,"factorial-retrained-"+args.cohort,config,roots,64,
              9285911 if original else 9285912,args.workers,1)


if __name__=="__main__":main()
