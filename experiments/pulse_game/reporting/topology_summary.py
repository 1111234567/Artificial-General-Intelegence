"""Compare training compensation to surgery on an already learned recurrent topology."""
import json
from pathlib import Path
import numpy as np
from .. import replicate_statistics,research_audit as audit
from ..attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


def main():
    folder=audit.OUT/"matched-sensory-topology";files=sorted(folder.glob("birth-*.json"))
    if not (folder/"complete.json").is_file() or len(files)!=6:raise RuntimeError("Wait for all six preregistered training pairs")
    bank={}
    for file in files:
        d=json.loads(file.read_text())
        if not d["paired_search_worlds"]:raise ValueError(file.name)
        for scenario,cases in d["rows"].items():
            baseline=cases["intact"]
            for arm,current in cases.items():
                for k in ("B_after_0","B_after_12","B_after_48","return_A","both"):
                    if k not in current:continue
                    a=np.asarray(current[k]);reference=np.asarray(baseline[k])
                    for label,idx in (("shared",[0,3]),("conflict",[1,2])):
                        bank.setdefault("/".join((scenario,arm,label,k)),[]).append(a[idx].mean(0))
                        if arm!="intact":
                            bank.setdefault("/".join((scenario,arm+"_minus_intact",label,k)),[]).append((a-reference)[idx].mean(0))
    result={k:replicate_statistics.crossed(v) for k,v in bank.items()}
    p=OUT/"morning-20261119/topology-comparison.json"
    _write(p,dict(scope="paired full and diagonal development including projection fit, reward training and matched evolutionary worlds",
            births=6,world_roots=256,rows=result))
    print("six matched training births verified")


if __name__=="__main__":main()
