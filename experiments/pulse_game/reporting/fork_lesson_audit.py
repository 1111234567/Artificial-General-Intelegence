"""Rebuild the forced hypothetical-choice comparison from the saved physical evaluation routes."""

import json

import numpy as np
import torch

from .. import fork_lesson_fit as base
from .. import fork_lesson_sender as sender
from .. import fork_lesson_whatif_eval as test
from .. import fork_lesson_whatif_fit as fit
from .. import fork_lesson_world as world


def main():
    torch.set_num_threads(1)
    evidence=base.EVIDENCE
    report=json.loads((evidence/"whatif-eval-results.json").read_text(encoding="utf8"))
    sources={};models={};points=0;routes=0
    for seed in test.SEEDS:
        scores=report["reports"][str(seed)]["scores"]
        scene=test.scenes(seed)
        oracle=world.oracle(scene).numpy()
        for kind in test.KINDS:
            for n,b in enumerate(base.BIRTHS):
                f=test.DATA/str(seed)/f"{kind}-{b}.npz";sha=base.digest(f)
                assert sha==report["reports"][str(seed)]["raw_sha256"][f.name]
                sources[f.relative_to(base.ROOT).as_posix()]=sha
                model,_,model_sha=fit.load_final(kind,b)
                models[kind+"/"+str(b)]=model_sha
                with np.load(f,allow_pickle=False) as z:
                    assert str(z["body"])==model_sha and str(z["plan"])==base.digest(test.PLAN)
                    np.testing.assert_array_equal(z["door_returns"],oracle)
                    strict=oracle[:,1]!=oracle[:,0]
                    for name in test.MODES:
                        a=z[name+"/action"];q=z[name+"/forecast"]
                        np.testing.assert_array_equal(z[name+"/actual_pay"],np.take_along_axis(oracle,a,axis=1))
                        v=(oracle[strict,1]*(q[strict,1]-q[strict,0]))
                        rank=np.where(v>0,1,np.where(v<0,0,.5))
                        np.testing.assert_allclose(rank.mean(),scores[kind+"/"+name]["strict_rank"]["by_birth"][n])
                        routes+=a.size
                    points+=len(strict)
    change={}
    for k in sender.KINDS:
        for b in base.BIRTHS:
            parent,identity=sender.load_parent(k,b)
            for arm in sender.ARMS:
                f=sender.path(k,b,arm);x=torch.load(f,weights_only=True,map_location="cpu")
                assert x["parent"]==identity
                mismatches=[nm for nm,row in parent.state_dict().items() if not torch.equal(row,x["body"][nm])]
                assert mismatches==["return_weight"]
                diff=(parent.return_weight!=x["body"]["return_weight"])
                torch.testing.assert_close(diff,parent.mask[20:,:20],atol=0,rtol=0)
                change[f"{k}/{b}/{arm}"]=int(diff.sum())
    out=dict(whatif_models=models,whatif_physical_files=sources,paired_histories=points,
             action_entries_reconstructed=routes,sender_exclusively_changed_axons=change,
             original_fit_sha=base.digest(evidence/"fit-audit.json"),audit_source=base.digest(__file__),
             scope="Each pair of forced query reports comes from the same pretrial state; "
                   "per-template physical examination is separate from the strict 40-node follow-up.")
    base.storage._write_json(evidence/"whatif-eval-audit.json",out)
    print("verified",len(sources),"test files,",points,"paired histories and",routes,"route entries")


if __name__=="__main__":
    main()
