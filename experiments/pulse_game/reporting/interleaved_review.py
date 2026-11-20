"""Reconstruct the return on one real review selected from the graph's motor uncertainty."""

from hashlib import sha256
import json
import numpy as np

from ..interleave_interference import DATA,EVIDENCE
from ..interleave_capacity_report import Crossed


def run():
    summary={}
    for label in ("uniform","uniform-independent"):
        meta=json.loads((EVIDENCE/f"review-choice-{label}.json").read_text())
        n=meta["worlds_per_split"];births=meta["births"]
        ci=Crossed(len(births),n,796779,5000)
        group={}
        for count in (4,8):
            measures={}
            for birth in births:
                file=f"{birth}-{count}.npz"
                path=DATA/"review-choice"/label/file
                if sha256(path.read_bytes()).hexdigest()!=meta["sha256"][file]:
                    raise ValueError("Review trace does not match its verified source")
                with np.load(path) as raw:
                    outcomes=raw["test/outcomes"][0,:,:,:,0]
                    row=np.arange(len(outcomes))
                    values={}
                    for mode in ("uniform","least_confident","most_confident"):
                        chosen=raw["selection/"+mode]
                        correct=outcomes[row,chosen]
                        values[mode+"/all"]=correct.all(1).reshape(n,4).mean(1)
                        values[mode+"/mean"]=correct.mean(1).reshape(n,4).mean(1)
                        if not np.isclose(values[mode+"/all"].mean(),meta["roots"][str(birth)][str(count)]["methods"][mode]["all"]):
                            raise ValueError("Incorrect private-law grouping")
                    for metric in ("all","mean"):
                        values["least minus uniform/"+metric]=values["least_confident/"+metric]-values["uniform/"+metric]
                    for k,v in values.items():measures.setdefault(k,[]).append(v)
            group[count]={k:ci(np.stack(v)) for k,v in measures.items()}
        summary[label]=dict(births=births,worlds=n,results=group)
    (EVIDENCE/"review-choice-paired.json").write_text(json.dumps(summary,indent=1)+"\n",encoding="utf8")


if __name__=="__main__":
    run()
