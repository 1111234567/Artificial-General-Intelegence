"""Can an optimized closed graph learn a private mixed change beyond one polarity bit?"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import zipfile

import numpy as np

from . import live_module_edits_study as life, ring_logic_study as io, ring_logic_world as sensory


EVIDENCE = io.REPO/"docs/evidence/open-graph-learning-2026-11-19/remap"
DATA = io.REPO/"data/pulse-game/mixed-remap-20261119"
REG = io.read(EVIDENCE/"protocol.json")


def design(roots, rows):
    tape, obs, lane, cell, flip, placebo = life.tapes(roots,rows,reversal=False)
    # Successive blocks each contain every cell exactly once in a fresh order.
    change = flip - 20
    original = sensory.truth()
    switched = np.empty((len(original),len(roots),len(sensory.PAIRS)),dtype=bool)
    for rule,table in enumerate(original):
        for idx,root in enumerate(roots):
            rng = np.random.default_rng([root, REG["table_rng"], rule])
            selected = np.concatenate([rng.choice(np.flatnonzero(table==v),5,replace=False)
                                       for v in (False,True)])
            switched[rule,idx] = False
            switched[rule,idx,selected] = True
    return tape, obs, lane, cell, change, placebo, switched


def scoring_tape(parts,rows,twin):
    tape, obs, lane, cell, change, placebo, switched = parts
    rule = np.repeat([r[3] for r in rows],len(cell))
    swaps = switched[rule[:,None],lane[:,None],cell[lane]] ^ bool(twin)
    clock = np.arange(cell.shape[1])
    during = (clock>=change[lane,None]) & (clock<change[lane,None]+60)
    right = sensory.truth()[rule[:,None],cell[lane]] ^ (during & swaps)
    masked = replace(tape, good_id=np.where(right,1-tape.left_id,tape.left_id))
    return masked,obs,lane,cell,change,placebo


def summarize_arm(physical,parts,rows,twin):
    _,_,_,cells,change,_,switched = parts
    groups=physical.shape[0]
    result={}
    correct=physical[...,2]>0
    for name,offset in (("pre",-20),("B1",0),("B2",20),("B3",40),("A_return",60)):
        idx=change[:,None]+offset+np.arange(20)
        positions=np.broadcast_to(idx,(*correct.shape[:-1],20))
        decisions=np.take_along_axis(correct,positions,axis=-1)
        chosen=np.take_along_axis(physical[...,0],positions,axis=-1)
        ids=np.take_along_axis(cells,idx,axis=-1)
        changed=np.take_along_axis(switched^bool(twin),np.broadcast_to(ids,switched.shape),axis=-1)
        if not np.all(changed.sum(-1)==10):raise ValueError("Unbalanced cells")
        result[name]=decisions.mean(-1)
        result[name+"_changed"]=(decisions*changed).sum(-1)/10
        result[name+"_unchanged"]=(decisions*(~changed)).sum(-1)/10
        result[name+"_coverage"]=(chosen>=0).mean(-1)
    if result["B1"].shape!=(groups,len(sensory.RULES),len(cells)):raise ValueError("Invalid life ordering")
    return result


def matched_twins(sides,parts):
    cells,change=parts[3:5]
    yes0,yes1=(x[...,2]>0 for x in sides)
    sel=np.broadcast_to(change[:,None],(*yes0.shape[:-1],1))
    s0,s1=(np.take_along_axis(x[...,0],sel,axis=-1) for x in sides)
    if not np.array_equal(s0,s1):raise ValueError("Different action before first B feedback")
    a,b=(np.take_along_axis(x,sel,axis=-1) for x in (yes0,yes1))
    if not np.array_equal(a.astype(int)+b,(s0>=0).astype(int)):
        raise ValueError("Zero-information first B score must equal half contact coverage")
    prefix=np.arange(life.TRIALS)[None,:]<change[:,None]
    if not np.all((sides[0]==sides[1]) | ~prefix[None,None,:,:,None]):
        raise ValueError("Public action histories differ before remapping")
    p=change[:,None]+40+np.arange(len(sensory.PAIRS))
    ix=np.broadcast_to(p,(*yes0.shape[:-1],20))
    both=np.take_along_axis(yes0&yes1,ix,axis=-1)
    return both.mean(-1)


def _pack(path,vals):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".partial.npz")
    np.savez_compressed(tmp,**vals)
    tmp.replace(path)


def run(arm):
    from . import open_life_study as learner
    cfg=read_config()
    roots=range(REG["world_start"],REG["world_start"]+REG["world_count"])
    target=DATA/f"{arm}.npz"
    head=life.heads(tuple(REG["groups"]))
    parts=design(roots,head[2])
    ids=list(roots)
    outputs=[]
    vals=dict(cells=parts[3],switch=parts[4],mask=parts[6],roots=ids)
    for twin in (0,1):
        tape=scoring_tape(parts,head[2],twin)
        res=learner.run(head,ids,tape,"legacy" if arm=="legacy_joint" else arm,cfg)
        phys=res["physical"]
        outputs.append(phys)
        vals[f"physical_{twin}"]=phys
        for k,v in summarize_arm(phys,parts,head[2],twin).items(): vals[f"{twin}/{k}"]=v
    vals["both_B3"]=matched_twins(outputs,parts)
    _pack(target,vals)
    print("finished remap",arm,flush=True)


def read_config():
    from . import open_life_study as learner
    return learner.sealed_recipe(learner.OUT)


def report():
    arms=REG["conditions"]
    roots=REG["world_count"];groups=len(REG["groups"])
    rng=np.random.default_rng(438317)
    bg=rng.multinomial(groups,np.ones(groups)/groups,20000)/groups
    wg=rng.multinomial(roots,np.ones(roots)/roots,20000)/roots
    def estimate(x,tail=.025):
        world=x.mean(axis=1)
        quantile=np.quantile(((bg@world)*wg).sum(axis=-1),[tail,1-tail])
        return dict(mean=float(x.mean()),ci95=quantile.tolist())
    matrices={}
    for arm in arms:
        with np.load(DATA/f"{arm}.npz",allow_pickle=False) as z:
            matrices[arm]={k[2:]:(z[k]+z["1/"+k[2:]])/2 for k in z.files if k.startswith("0/")}
            matrices[arm]["paired_both_B3"]=z["both_B3"]
    curves={arm:{k:estimate(v) for k,v in bank.items()} for arm,bank in matrices.items()}
    full=matrices["adaptive"]
    effect={"B3_minus_chance":full["B3"]-.5,
            "B3_minus_scalar":full["B3"]-matrices["head_only"]["B3"],
            "B3_minus_placebo":full["B3"]-matrices["placebo"]["B3"]}
    primary={k:estimate(x,.025/3) for k,x in effect.items()}
    secondary={
        "B3_minus_B1":estimate(full["B3"]-full["B1"]),
        "B3_minus_previous_joint":estimate(full["B3"]-matrices["legacy_joint"]["B3"]),
        "A_return_minus_pre":estimate(full["A_return"]-full["pre"]),
        "B3_changed_minus_unchanged":estimate(full["B3_changed"]-full["B3_unchanged"]),
    }
    archive=EVIDENCE/"remap-raw.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for arm in arms:z.write(DATA/f"{arm}.npz",f"{arm}.npz")
    io.write(EVIDENCE/"summary.json",dict(curves=curves, primary=primary, secondary=secondary,
             compound_test_pass=all(v["ci95"][0]>0 for v in primary.values()),
             raw_sha256=io.sha(archive), source_sha256=io.sha(Path(__file__)),configuration=read_config()))


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("run","report"));p.add_argument("--workers",type=int,default=2)
    args=p.parse_args()
    if args.stage=="run":
        read_config()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:list(pool.map(run,REG["conditions"]))
    report()
