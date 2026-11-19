"""Paired-world summaries for delayed credit and the independently posed object-learning task."""
import json
from pathlib import Path
import numpy as np
from .. import research_audit as audit
from ..attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


def avg(vectors):return audit.estimates(np.mean(np.stack(vectors),axis=0))


def summary():
    folder=audit.OUT
    out={}
    for branch,sub,families in (("delays","asynchronous-credit-sealed",True),("deferral","deliberate-credit-sealed",True),
              ("objects","object-learning-set",False),("switch","object-content-sealed",False)):
        files=sorted(p for p in (folder/sub).glob("*.json") if p.stem!="complete")
        groups={}
        for p in files:
            d=json.loads(p.read_text());case=d.get("condition",d.get("scenario",d.get("regime",d.get("experiment","single"))))
            groups.setdefault(case,[]).append(d)
        book={}
        for regime,ds in groups.items():
            table={}
            for arm,v in ds[0]["rows"].items():
                table[arm]={}
                for k in ("correct_binding","packets","anchor_available","extra_commit_wait","B_after_0","B_after_12","return_A","both",
                    "trial_1","trial_2","trial_6","all","early_second","late_second","initial_A","initial_B","initial_both",
                    "trial1","trial2","late_trial2","object_second","object_late_second","back_to_rules","return_to_relations","return_to_A","return_to_B"):
                    if k not in v:continue
                    arr=np.stack([np.asarray(d["rows"][arm][k]) for d in ds])
                    if arr.ndim==3:
                        if arr.shape[1]!=4:raise ValueError((regime,k,arr.shape))
                        values=arr[:,[1,2]].mean(1)
                    elif arr.ndim==2:values=arr
                    else:continue
                    table[arm][k]=avg(values)
                    lookup={"delays":"fifo","deferral":"immediate_same_rule","objects":"native","switch":"native"}[branch]
                    if lookup in ds[0]["rows"] and arm!=lookup:
                        raw=np.stack([np.asarray(d["rows"][lookup][k]) for d in ds])
                        if raw.ndim==3:raw=raw[:,[1,2]].mean(1)
                        table[arm][k]["difference_vs_"+lookup]=avg(values-raw)
            book[regime]=table
        out[branch]=book
    return out


if __name__=="__main__":
    _write(OUT/"morning-20261119/credit-and-transfer.json",summary())
