"""Check the teacher-only diagnostic against stepwise GRU inference."""
import argparse
import json

import numpy as np
import torch

from experiments.experience_transfer import content as c


@torch.inference_mode()
def check(seed):
    torch.set_num_threads(1)
    archive=c.OUT/f"stream_{seed}.pt"
    d=torch.load(archive,weights_only=False,map_location="cpu")
    src=c.OUT/f"seed_{seed}.json"
    result=json.loads(src.read_text(encoding="utf-8"))
    assert result["stream_sha"]==c.sha(archive)
    train=json.loads((c.BASE/"training-v1/protocol.json").read_text(encoding="utf-8"))
    spec=train["seeds"][str(seed)]
    a,old,b,cue=c.MAPS[seed]
    assert (list(a),list(old),list(b),cue)==(spec["A"],spec["C"],spec["B"],spec["B_cue"])
    key=("t","lane","job")
    ix=tuple(np.array([r[k] for r in d["rows"]]) for k in key)
    target=np.array([r["target"] for r in d["rows"]])
    raw=np.array([r["raw"] for r in d["rows"]])
    v,p=(target-3)//4,(target-3)%4
    assert np.array_equal(p,np.array(b)[raw])
    mask=np.array(a)[raw]==p
    checks=[]
    for arm in ("related","disjoint"):
        for stage in ("B000","B008"):
            path=c.BASE/f"training-v1/seed_{seed:03d}/{arm}/{stage}.pt"
            assert result["sources"][str(path.relative_to(c.ROOT))]==c.sha(path)
            cp=torch.load(path,weights_only=False,map_location="cpu")
            model=c.GRUAgent(**cp["model_config"])
            model.load_state_dict(cp["model"]);model.eval()
            h=model.initial_state(16)
            values=[]
            for obs in d["obs"]:
                scores,h=model.step(obs,h,.25)
                values.append(scores.numpy())
            pred=np.stack(values)[ix]
            choices=pred.argmax(-1)
            cond=pred[:,3:].argmax(-1)
            payload_logits=pred[np.arange(len(p))[:,None],3+4*v[:,None]+np.arange(4)]
            summary=next(r["counts"] for r in result["records"] if (r["arm"],r["stage"])==(arm,stage))
            for group,m in (("overlap",mask),("conflict",~mask)):
                expected=dict(n=m.sum(),exact_eight=(cond[m]==target[m]-3).sum(),
                    payload_given_version=(payload_logits[m].argmax(1)==p[m]).sum(),chose_finish=(choices[m]>=3).sum(),
                    chose_correct_action=(choices[m]==target[m]).sum())
                if group=="conflict":
                    prior=np.array(old)[raw[m]]
                    pm=payload_logits[m]; select=np.arange(len(pm))
                    expected.update(old_payload_given_version=(pm.argmax(1)==prior).sum(),
                                    new_outscores_old=(pm[select,p[m]]>pm[select,prior]).sum())
                assert summary[group]==expected,(arm,stage,group,summary[group],expected)
            checks.append(dict(arm=arm,stage=stage,rows=len(p)))
    (c.OUT/f"stepwise-check-{seed}.json").write_text(json.dumps(dict(source_sha=c.sha(src),checks=checks),indent=2),encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--seed",type=int,required=True,choices=range(3));args=p.parse_args()
    check(args.seed)
