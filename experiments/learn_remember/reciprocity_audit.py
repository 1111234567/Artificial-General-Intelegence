"""Reconstruct strict completions from actual receipts and verify saved shared pairs."""
import json

import numpy as np
import torch

from experiments.learn_remember import symmetric_gru as sym
from experiments.learn_remember.gru_train import load_checkpoint, sha
from experiments.lifelong.world import NEW, REPLACE


def positions(mask,rids):
    t,l,j=np.nonzero(mask)
    return {(int(lane),int(job),int(rid)) for lane,job,rid in zip(l,j,rids[t,l,j])}


def entirely(stages):
    checks=[]
    torch.set_num_threads(1)
    for seed in range(3):
        folder=sym.OUT/f"seed_{seed:03d}"
        for stage in stages:
            p=folder/(stage+".pt")
            model,_,_,cp=load_checkpoint(p)
            model=sym.constrain(model)
            optimizer=torch.optim.Adam(model.parameters(),lr=cp["config"]["lr"])
            optimizer.load_state_dict(cp["optimizer"])
            packed=model.core.parametrizations.weight_hh_l0.original
            assert packed.shape==(3,192*193//2)
            for param,state in optimizer.state.items():
                assert param.shape==state["exp_avg"].shape==state["exp_avg_sq"].shape
            dense=cp["model"]["core.weight_hh_l0"].view(3,192,192)
            assert torch.equal(dense,dense.transpose(-1,-2)) and torch.equal(model.core.weight_hh_l0,dense.reshape(-1,192))
            assert sum(t.numel() for t in model.parameters())==93238
            record=json.loads((folder/f"behavior_{stage}.json").read_text(encoding="utf-8"))
            assert record["model_sha"]==sha(p)
            nsuccess=0
            for r in record["records"]:
                path=folder/f"{stage}_{r['condition']}_f{r['family']}.facts.npz"
                assert sha(path)==r["facts_sha"]
                with np.load(path) as data:
                    # Only new, eligible commands whose request deadline actually matured.
                    ids=data["request_id"]; outcome=data["result"]; acts=data["actions"]
                    issued=positions(np.isin(data["event_kind"],[NEW,REPLACE]) & data["event_eligible"] &
                                     (data["current_regime_id"]==r["evaluation_regime_id"]), ids)
                    mature=positions(data["matured_mask"],data["matured_request_id"]) & issued
                    success=positions((outcome==1)&(acts>=3),ids) & mature
                    errors=positions((outcome==-1)&data["active"],ids)
                    strict=success-errors
                    assert len(mature)==r["counts"]["matured"]
                    assert len(success)==r["counts"]["completed"]
                    assert len(strict)==r["counts"]["clean_completed"]
                    actual_idle=~data["active"]
                    assert actual_idle.sum()==r["counts"]["idle_opportunities"]
                    assert (actual_idle & (acts!=0)).sum()==r["counts"]["idle_actions"]
                    nsuccess+=len(strict)
            checks.append(dict(seed=seed,stage=stage,eight_matched_forks=len(record["records"]),
                unique_recurrent_parameters=packed.numel(),strict_completions_reconstructed=nsuccess,
                checkpoint_sha=sha(p),validated_behavior_sha=sha(folder/f"behavior_{stage}.json")))
    return checks


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--stages",nargs="+",choices=("wide","withdrawal","review"),default=("wide","withdrawal","review"))
    stages=p.parse_args().stages
    name="independent-audit.json" if set(stages)=={"wide","withdrawal","review"} else "partial-fact-audit.json"
    (sym.OUT/name).write_text(json.dumps(dict(audit="passed",checks=entirely(stages)),indent=2),encoding="utf-8")
