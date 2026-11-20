"""Verify the primary neural graph checkpoints and rebuild its sensory-free behavioral endpoint."""

import json

import numpy as np
import torch

from .. import native_rehearsal_eval as ev
from .. import native_rehearsal_fit as fit
from .. import native_rehearsal_graph as live


def main():
    torch.set_num_threads(1)
    results=json.loads((fit.EVIDENCE/"behavior.json").read_text(encoding="utf8"))
    files={};heads={};answers=0;first_update_checks=0
    for k in fit.KINDS:
        for birth in fit.BIRTHS:
            model,path=ev.load(k,birth)
            blob=torch.load(path,weights_only=True,map_location="cpu")
            signature=blob["signature"]
            if not (signature["plan"]==fit.digest(fit.EVIDENCE/"protocol.md") and
                    signature["source"]==fit.digest(fit.__file__) and blob["ledger"].shape==(fit.STEPS,6) and
                    torch.isfinite(blob["ledger"]).all().item()):
                raise ValueError(f"Unsealed fit at {path}")
            if {int(state["step"]) for state in blob["optimizer"]["state"].values()}!={fit.STEPS}:
                raise ValueError("Native optimization count mismatch")
            first=live.NativeGraph(birth,blob["layout"])
            opt=torch.optim.Adam(first.parameters(),lr=.003)
            sample=fit.update(first,opt,k,birth,1,fit.BATCH)
            np.testing.assert_allclose(sample,blob["ledger"][0].numpy(),rtol=1e-6,atol=1e-6)
            first_update_checks+=1
            heads[f"{k}/{birth}"]=dict(sha256=fit.digest(path),birth=birth,steps=fit.STEPS,
                                       edges=int(model.mask.sum()),width=model.width)
            for seed in ev.SEEDS:
                p=fit.DATA/"eval"/str(seed)/f"{k}-{birth}.npz"
                fingerprint=fit.digest(p)
                if fingerprint!=results[str(seed)]["hashes"][p.name]:raise ValueError(p)
                files[f"{seed}/{p.name}"]=fingerprint
                with np.load(p,allow_pickle=False) as raw:
                    for mode in ev.MODES:
                        a=raw[mode+"/action"]
                        np.testing.assert_array_equal(a*raw["law"]*raw["orientation"],raw[mode+"/pay"])
                        if mode=="normal":
                            if a.shape!=(ev.ROOTS*16,2):raise ValueError("Unbalanced factual archive")
                            answers+=a.size
                            near=np.isin(raw["angle"],(15,30))
                            conflict=raw["law"][:,0]!=raw["law"][:,1]
                            both=(raw[mode+"/pay"][near&conflict]>0).all(1).mean()
                            endpoint=results[str(seed)]["results"]["close"][k+"/normal"]["conflict/both"]["by_birth"]
                            np.testing.assert_allclose(both,endpoint[fit.BIRTHS.index(birth)],rtol=0,atol=1e-7)
    fit.storage._write_json(fit.EVIDENCE/"primary-audit.json",dict(
        primary_fits=heads,full_budget_routes=fit.STEPS*fit.BATCH*len(heads),primary_raw=files,
        factual_heldout_actions=answers,first_updates_recreated=first_update_checks,
        modalities_verified=list(ev.MODES),source_sha256={m:fit.digest(p) for m,p in (
            ("graph",live.__file__),("fit",fit.__file__),("eval",ev.__file__),("audit",__file__))}))
    print(len(heads),"fits;",len(files),"physical files;",answers,"factual test actions")


if __name__=="__main__":
    main()
