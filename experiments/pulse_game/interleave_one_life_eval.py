"""Test actual two-action earnings and six independently read memories on sealed joint lives."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import torch

from . import interleave_one_life as study, interleave_cross_skill_eval as six
from . import interleave_geometry_eval as geometric, interleave_triangle_angular as nested
from . import interleave_capacity_report as ci, interleave_model_cards as cards
from .interleave_interference import DATA,EVIDENCE


SUBSETS=("all_maps","context_conflict","hard_and_context_conflict")
PAID=("two_paid","triangle_paid","context_or_mark_paid")


def masks():
    conflict=six.LAWS[:,3]!=six.LAWS[:,4]
    return (np.ones(64,dtype=bool),conflict,conflict&np.all(six.LAWS[:,:3]==six.LAWS[:,:1],axis=1))


@torch.no_grad()
def consecutive(g,r,seed,worlds):
    result={}
    for order in study.ARM:
        ticket=study.ticket(seed,0,131,worlds,order,domain=1665931,all_laws=True)
        collected=[];history=[]
        for i in range(0,worlds,12):
            chunk={k:v[i*64:(i+12)*64] for k,v in ticket.items()}
            _,pay,choice,support=study.perform(g,r,chunk)
            idx=chunk["query"]
            law=chunk["law"].gather(1,idx)
            np.testing.assert_array_equal(pay.numpy(),(choice*law*chunk["qsign"]).numpy())
            collected.append(torch.stack((choice,pay),dim=-1).to(torch.int8).numpy().reshape(-1,64,2,2))
            history.append(support.to(torch.int8).numpy().reshape(-1,64,7,2))
        data=np.concatenate(collected)
        query=ticket["query"].numpy().reshape(worlds,64,2)
        result[order+"/paid"]=data
        result[order+"/support"]=np.concatenate(history)
        result[order+"/query"]=query
        u=ticket["qu"][:,:,0].numpy()
        random_side=np.where(u<.5,1,-1)
        random_pay=random_side*ticket["law"].gather(1,ticket["query"]).numpy()*ticket["qsign"].numpy()
        for policy,ok in (("active",data[:,:,:,1]>0),("uniform",random_pay.reshape(worlds,64,2)>0)):
            a=(ok&(query<3)).any(2);b=(ok&(query>=3)).any(2)
            for label,keep in zip(SUBSETS,masks(),strict=True):
                values=np.stack(((a&b)[:,keep],a[:,keep],b[:,keep]),axis=-1)
                result[f"{order}/{policy}/{label}"]=values.mean(1)
    return result


def run(cohort,step,split):
    torch.set_num_threads(1)
    first=cohort=="original";births=list(range(731,737)) if first else list(range(811,817))
    group="original" if first else "confirmation"
    seed=(622819 if first else 622829) if split=="shared" else (614913 if first else 614929)
    worlds=six.N_WORLD
    # Keep one parent validation cache per split, and bind all traces to the generating weights.
    root=DATA/"one-life-eval"/cohort/split
    protocol=sha256(Path(__file__).read_bytes()).hexdigest()
    tape=six.make(seed,worlds)
    truth=six.LAWS[None,:,:]*tape["query_orientation"][:,None,:]
    card=read(EVIDENCE/f"model-cards-{group}.json")
    series={};hashes={}
    for arm in ("parent",*study.ARM):
        if arm=="parent":source,role=geometric.sources(cohort,"hard",64000);at=64000
        else:source,role,at=study.DEST/cohort,arm,step
        rows={};secondary={}
        for b in births:
            checkpoint=study.tri.old_fit.checkpoint_path(source,"odor",role,b,at)
            identity=sha256(checkpoint.read_bytes()).hexdigest()
            hashes[checkpoint.relative_to(DATA).as_posix()]=identity
            f=root/f"{arm}-{b}-{at}.npz"
            if not f.exists():
                g,r=study.tri.old_fit.load_checkpoint(source,"odor",role,b,at)
                raw=dict(truth=truth.astype(np.int8),checkpoint_sha256=identity,evaluator_sha256=protocol)
                for i,order in enumerate(six.ORDERS):
                    for mode in six.MODES:
                        chosen=[];contacts=[]
                        for j in range(0,worlds,12):
                            chunk={k:v[j:j+12] for k,v in tape.items()}
                            out,hist=six.simulate(g,r,chunk,i,mode)
                            chosen.append(out);contacts.append(hist)
                        raw[f"{order}/{mode}/side"]=np.concatenate(chosen)
                        raw[f"{order}/{mode}/contact"]=np.concatenate(contacts)
                if split=="shared" and arm=="parent":
                    with np.load(DATA/"cross-skill-eval"/cohort/"0"/f"parent-{b}.npz") as old:
                        if str(old["checkpoint_sha256"])!=identity:raise ValueError("Wrong archived parent")
                        for k,v in raw.items():
                            if k not in ("checkpoint_sha256","evaluator_sha256"):
                                np.testing.assert_array_equal(v,old[k])
                raw.update({"reward/"+k:v for k,v in consecutive(g,r,seed+71,worlds).items()})
                if split=="shared":
                    other,_=cards.row(g,r,card["seed"],card["worlds"])
                    raw["card"]=np.stack([other[k] for k in geometric.CARD_KEYS],axis=1)
                    hard,actions,passes=geometric.triad(g,r,geometric.test_tape(b,128,"hard"))
                    raw.update(hard_standalone=hard,hard_actions=actions,hard_margin_pass=passes)
                study.tri.revisit._write_raw(f,**raw)
            hashes[f.relative_to(DATA).as_posix()]=sha256(f.read_bytes()).hexdigest()
            with np.load(f) as raw:
                if str(raw["checkpoint_sha256"])!=identity or str(raw["evaluator_sha256"])!=protocol:
                    raise ValueError("Unexpected evaluation or checkpoint identity")
                six.collect(raw,truth,SUBSETS,rows)
                for policy in ("active","uniform"):
                    for cond in SUBSETS:
                        scores=[raw[f"reward/{order}/{policy}/{cond}"] for order in study.ARM]
                        secondary.setdefault(f"reward/{policy}/{cond}",[]).append(np.mean(scores,axis=0))
                        for order,values in zip(study.ARM,scores,strict=True):
                            secondary.setdefault(f"reward/{policy}/{cond}/{order}",[]).append(values)
                if split=="shared":
                    for k in ("card","hard_standalone"):secondary.setdefault(k,[]).append(raw[k])
            print(cohort,step,split,arm,b,flush=True)
        series[arm]={k:np.stack(v) for k,v in {**rows,**secondary}.items()}
    for a,b in (("interspersed","blocked"),("interspersed","parent"),("blocked","parent")):
        series[a+" minus "+b]={k:v-series[b][k] for k,v in series[a].items()}
    common=ci.Crossed(len(births),worlds,seed+1,4000)
    card_ci=ci.Crossed(len(births),card["worlds"],card["seed"]+19,4000)
    results={}
    for arm,parts in series.items():
        results[arm]={}
        for key,value in parts.items():
            names=PAID if key.startswith("reward/") else six.METRICS
            method=common
            if key=="card":names=geometric.CARD_KEYS;method=card_ci
            if key=="hard_standalone":names=geometric.MEASURES;method=nested.intervals
            results[arm][key]={m:method(value[:,:,i]) for i,m in enumerate(names)}
    out=dict(births=births,step=step,split=split,seed=seed,worlds=worlds,hashes=hashes,results=results,
             notes="Fitted only on the earned conjunction of two real terminal actions after joint histories. "
                   "Six-item cold probes are from one pre-probe state, not six sequential remunerated actions. "
                   "The separate reward/ panels execute both terminal actions continuously and withhold their "
                   "taste as in training. One pair of questions (one per domain) and one of its two orders are "
                   "sampled per independent geometry, then held fixed for all 64 laws. "
                   "All world statistics keep the full 64 private assignments crossed "
                   "within geometry. The hardest subset has two homogeneous triangle signs, two opposing "
                   "context assignments and two feeder signs, giving eight of the 64 assignments.")
    (EVIDENCE/f"one-life-{cohort}-{split}-{step}.json").write_text(json.dumps(out,indent=1)+"\n",encoding="utf8")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--cohort",choices=("original","independent"),required=True)
    p.add_argument("--step",type=int,choices=(4000,8000),required=True)
    p.add_argument("--split",choices=("shared","disjoint"),required=True)
    args=p.parse_args()
    run(args.cohort,args.step,args.split)
