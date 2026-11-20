"""Score the same nonlinear-memory training trajectory on the ordinary held-out model card."""

from argparse import ArgumentParser
from hashlib import sha256
import json

import numpy as np
import torch

from . import interleave_capacity_report as statistics, interleave_model_cards as cards
from . import local_trace_train as old
from .interleave_interference import DATA, EVIDENCE


def run(cohort):
    torch.set_num_threads(1)
    independent = cohort=="confirmation"
    births = list(range(811,817)) if independent else list(range(731,737))
    triangle_cohort = "independent" if independent else "original"
    common_root = DATA / ("triangle-confirmation" if independent else "triangle-objective")
    long_root = DATA / ("triangle-long-"+triangle_cohort)
    folder=DATA/"triangle-transfer"/cohort
    folder.mkdir(parents=True,exist_ok=True)
    parent = json.loads((EVIDENCE/f"model-cards-{cohort}.json").read_text())
    worlds, seed = parent["worlds"], parent["seed"]
    bootstrap=statistics.Crossed(len(births),worlds,seed+19,4000)
    arrays, scores, paired, hashes = {}, {}, {}, {}
    for step,path in ((0,common_root),(8000,common_root),(32000,long_root),(64000,long_root)):
        rows = []
        for birth in births:
            g,r=old.load_checkpoint(path,"odor","direct",birth,step)
            values,raw=cards.row(g,r,seed,worlds)
            if step==0:
                archived=DATA/"model-cards"/cohort/f"uniform-memory-{birth}.npz"
                if sha256(archived.read_bytes()).hexdigest()!=parent["hashes"][archived.name]:
                    raise ValueError("Parent card changed")
                with np.load(archived) as z:
                    for k,v in values.items():
                        np.testing.assert_array_equal(v,z["metric/"+k])
            file=folder/f"{birth}-{step}.npz"
            np.savez_compressed(file,**raw,**{"metric/"+k:v for k,v in values.items()})
            hashes[file.name]=sha256(file.read_bytes()).hexdigest()
            checkpoint=old.checkpoint_path(path,"odor","direct",birth,step)
            hashes[checkpoint.relative_to(DATA).as_posix()]=sha256(checkpoint.read_bytes()).hexdigest()
            rows.append(values)
        arrays[step]={k:np.stack([row[k] for row in rows]) for k in rows[0]}
        scores[step]={k:bootstrap(v) for k,v in arrays[step].items()}
        if step:
            paired[str(step)+" minus 0"]={k:bootstrap(v-arrays[0][k]) for k,v in arrays[step].items()}
        print(cohort,step,flush=True)
    report=dict(births=births,worlds=worlds,seed=seed,scores=scores,paired=paired,raw_sha256=hashes,
                note="Uniform objective parents specialized on three jointly learned zero-sum cues for the indicated "
                     "number of outer updates. All other worlds are different from triangle training. Each frozen "
                     "graph is used independently across the full common card with no new outer training. The "
                     "stage-zero actions must reproduce the archived uniform memory cards. World and birth "
                     "bootstrap is paired; six physical terminal orders on the triangle itself are reported separately.")
    (EVIDENCE/f"triangle-transfer-{cohort}.json").write_text(json.dumps(report,indent=1)+"\n",encoding="utf8")


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--cohort",choices=("original","confirmation"),required=True)
    run(parser.parse_args().cohort)
