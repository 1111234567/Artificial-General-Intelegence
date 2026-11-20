"""Matched downstream behavior of hard-only, easy-only and interleaved-geometry graduates."""

from argparse import ArgumentParser
from hashlib import sha256
from itertools import permutations
from pathlib import Path
import json

import numpy as np
import torch

from . import interleave_capacity_report as ci, interleave_confirm_triangle as perm
from . import interleave_model_cards as cards, interleave_triangle_angular as angular
from . import interleave_triangle_train as tri
from .interleave_interference import DATA, EVIDENCE


ARMS = ("hard", "orthogonal", "interleaved")
SHAPES = ("hard", "orthogonal", "random")
MEASURES = ("both_hard_maps_per_order", "both_hard_maps_all_orders", "eight_maps_all_orders",
            "three_actual", "hard_three_actual", "mixed_three_actual")
CARD_KEYS = ("all_2","all_5","A_after_16","A_and_recent_after_16",
             "coherent/unseen","coherent/exception_after_one","serial_conflict/both_modalities")


def sources(cohort, arm, step):
    if arm=="hard":
        if step<32000:
            parent="triangle-objective" if cohort=="original" else "triangle-confirmation"
        else:
            parent="triangle-long-"+cohort
        return DATA/parent,"direct"
    return DATA/"geometry-curriculum"/cohort,arm


def test_tape(birth, worlds, shape):
    t=tri.make_tape(angular.SEED,birth,worlds,-.5 if shape=="hard" else 0.)
    if shape=="random":
        rng=np.random.default_rng([911399,angular.SEED,birth])
        x=rng.normal(size=(worlds,3,3))
        x=(1.35*x/np.linalg.norm(x,axis=2,keepdims=True)).astype(np.float32)
        t["right"]=torch.from_numpy(np.repeat(x,8,axis=0))
        t["left"]=-t["right"]
    return t


@torch.no_grad()
def triad(g,r,tape,archive=None):
    worlds=len(tape["law"])//8
    law=tape["law"].numpy().reshape(worlds,8,3)
    margin_ok,actions=[],[]
    for j,order in enumerate(permutations(range(3))):
        result=tri.trials(g,r,perm.permuted_tape(tape,order),"active")
        if archive is not None:
            np.testing.assert_array_equal(result["logit"],archive[f"active_logit_{j}"])
            np.testing.assert_array_equal(result["selected"],archive[f"active_selected_{j}"])
        logits=result["logit"].reshape(worlds,8,3,2)
        margin=(logits[...,0]-logits[...,1])*law/2
        margin_ok.append((margin>tri.MARGIN_THRESHOLD).all(2))
        actions.append(result["selected"].reshape(worlds,8,3,2))
    passes=np.stack(margin_ok)
    sides=np.stack(actions)
    correct=(sides==law[None,...,None]).all(3)
    hard=passes[:,:,[0,7]].all(2)
    return np.stack((hard.mean(0),hard.all(0),passes.all((0,2)),
                     correct.mean((0,2,3)),correct[:,:,[0,7]].mean((0,2,3)),
                     correct[:,:,1:7].mean((0,2,3))),axis=-1),sides,passes


def stage(cohort, step):
    torch.set_num_threads(1)
    original=cohort=="original"
    births=list(range(731,737)) if original else list(range(811,817))
    group="original" if original else "confirmation"
    common_card=json.loads((EVIDENCE/f"model-cards-{group}.json").read_text())
    transfer=json.loads((EVIDENCE/f"triangle-transfer-{group}.json").read_text())["raw_sha256"]
    evaluation_digest=sha256(Path(__file__).read_bytes()).hexdigest()
    n,card_seed=common_card["worlds"],common_card["seed"]
    folder=DATA/"geometry-comparison"/cohort/str(step)
    folder.mkdir(exist_ok=True,parents=True)
    tables,hashes={},{}
    for arm in ARMS:
        path,tag=sources(cohort,arm,step)
        groups={k:[] for k in (*SHAPES,"card")}
        for b in births:
            checkpoint=tri.old_fit.checkpoint_path(path,"odor",tag,b,step)
            checkpoint_key=checkpoint.relative_to(DATA).as_posix()
            checkpoint_digest=sha256(checkpoint.read_bytes()).hexdigest()
            hashes[checkpoint_key]=checkpoint_digest
            output=folder/f"{arm}-{b}.npz"
            legacy=False
            if output.exists():
                with np.load(output) as saved:
                    legacy="checkpoint_sha256" not in saved.files
                    if not legacy and (str(saved["checkpoint_sha256"])!=checkpoint_digest
                                       or str(saved["evaluator_sha256"])!=evaluation_digest):
                        raise ValueError(f"Cached array belongs to a different checkpoint or evaluation: {output}")
            if not output.exists() or legacy:
                g,r=tri.old_fit.load_checkpoint(path,"odor",tag,b,step)
                if arm=="hard":
                    prior_file=DATA/"triangle-transfer"/group/f"{b}-{step}.npz"
                    if sha256(prior_file.read_bytes()).hexdigest()!=transfer[prior_file.name] or transfer[checkpoint_key]!=checkpoint_digest:
                        raise ValueError("The native model card and the graduated graph are not the sealed pair")
                    with np.load(prior_file) as prior:
                        card={k:prior["metric/"+k] for k in CARD_KEYS}
                else:
                    card,_=cards.row(g,r,card_seed,n)
                raw={"card":np.stack([card[k] for k in CARD_KEYS],axis=1)}
                for shape in SHAPES:
                    tape=test_tape(b,n,shape)
                    if step>=32000 and arm=="hard" and shape=="hard":
                        file=DATA/"long-order-verification/order-validation"/("long-"+cohort)/f"{b}-{step}.npz"
                        with np.load(file) as archive:
                            metrics,sides,passes=triad(g,r,tape,archive)
                    else:
                        metrics,sides,passes=triad(g,r,tape)
                    raw[shape]=metrics
                    raw[shape+"/side"]=sides
                    raw[shape+"/margin_pass"]=passes
                    raw[shape+"/law"]=tape["law"].numpy().reshape(n,8,3)
                if legacy:
                    with np.load(output) as previous:
                        for k in raw:np.testing.assert_array_equal(raw[k],previous[k])
                raw.update(checkpoint_sha256=checkpoint_digest,evaluator_sha256=evaluation_digest)
                tri.revisit._write_raw(output,**raw)
            hashes[output.relative_to(DATA).as_posix()]=sha256(output.read_bytes()).hexdigest()
            with np.load(output) as saved:
                for k in groups:groups[k].append(saved[k])
        tables[arm]={k:np.stack(v) for k,v in groups.items()}
        print(cohort,step,arm,flush=True)
    compared=dict(tables)
    for a,b in (("interleaved","hard"),("interleaved","orthogonal"),("hard","orthogonal")):
        compared[a+" minus "+b]={k:v-tables[b][k] for k,v in tables[a].items()}
    if step==64000:
        early=DATA/"geometry-comparison"/cohort/"32000"
        early_hard={k:[] for k in tables["hard"]}
        for b in births:
            with np.load(early/f"hard-{b}.npz") as z:
                for k in early_hard:early_hard[k].append(z[k])
        compared["interleaved64k minus hard32k"]={k:v-np.stack(early_hard[k]) for k,v in tables["interleaved"].items()}
    bootstrap=ci.Crossed(len(births),n,card_seed+19,4000)
    scores={}
    for label,sections in compared.items():
        scores[label]={}
        for section,a in sections.items():
            names=CARD_KEYS if section=="card" else MEASURES
            method=bootstrap if section=="card" else angular.intervals
            scores[label][section]={measure:method(a[:,:,j]) for j,measure in enumerate(names)}
    result=dict(cohort=cohort,step=step,births=births,worlds=n,raw_sha256=hashes,scores=scores,
                notes="Exact same graph at a training stage across every downstream column. Triple-shape metrics "
                      "average six study orders and eight law assignments inside each geometry. The first two "
                      "margin metrics require both of the +++ and --- histories to succeed with all three "
                      "orientation-correct scores >.05. Three-actual is the real sampled result from sibling "
                      "probes. Triangle geometries vary by training birth (nested resampling); the separate common "
                      "card geometries are shared by training births (crossed resampling).")
    (EVIDENCE/f"geometry-comparison-{cohort}-{step}.json").write_text(json.dumps(result,indent=1)+"\n",encoding="utf8")


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--cohort",choices=("original","independent"),required=True)
    p.add_argument("--step",choices=(8000,32000,64000),type=int,required=True)
    a=p.parse_args()
    stage(a.cohort,a.step)
