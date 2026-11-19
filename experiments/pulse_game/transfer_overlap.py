"""Evaluate a frozen trust policy while varying only the overlap of the two sensory planes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from . import relation_world as world
from .relation_analysis import Intervals
from .relation_study import ROOT, read, write, digest


EVIDENCE = ROOT / "docs/evidence/adaptive-trust-2026-11-19"
REG = read(EVIDENCE / "overlap-protocol.json")
DATA = ROOT / "data/pulse-game/adaptive-trust-20261119/overlap"
NONCE = 65172039


def roots():
    lo, hi = REG["world_range"]
    return (lo+np.random.default_rng(REG["seed"]).choice(hi-lo, REG["worlds"], replace=False)).tolist()


def planes(seed, nonce, degrees):
    base = np.linalg.qr(world._rng(seed, nonce, 631).normal(size=(4, 2)))[0]
    spare = world._rng(seed, nonce, 981).normal(size=(4, 2))
    complement = np.linalg.qr(spare-base@(base.T@spare))[0]
    angle = np.deg2rad(degrees)
    return base, np.cos(angle)*base + np.sin(angle)*complement


def prior_at_angle(protocol, degrees):
    """Same B object and all A anchor/sign RNG draws as in the ordinary relation game."""
    if not 0 <= degrees <= 90: raise ValueError("Principal angles outside [0, 90]")
    root_indices = np.unique(protocol.row_root, return_index=True)[1]
    tags = np.stack((protocol.history["related"].appearance[root_indices,0],
                     protocol.teaching.appearance[root_indices,0]), axis=1)
    stages = []
    for baseline, stream, physical_stream, query in (
        (protocol.history["related"], 101, 20, False),
        (protocol.probes["old_related"], 104, 23, True),
    ):
        n_pairs = baseline.menu.body.trials // 3
        design = []
        for seed in protocol.roots:
            _, basis = planes(seed, protocol.nonce, degrees)
            transform = np.eye(4)-2*basis@basis.T
            rng = world._rng(seed, protocol.nonce, stream)
            anchors = world._unit(rng.normal(size=(n_pairs,4)))
            signs = np.repeat(rng.choice((-1,1),size=n_pairs), 2)
            generator = world._queries if query else world._teaching
            design.append(generator(anchors,transform,signs,0.,rng))
        stages.append(world._stage(protocol.roots,protocol.seeds,protocol.row_root,tags,protocol.parity,
                                   design,0,query,protocol.nonce,physical_stream))
    return replace(protocol, history={"related":stages[0]}, probes={**protocol.probes,"old_related":stages[1]})


def sign_twins(protocol):
    lookup = {(root,int(a),int(b)):i for i,(root,(a,b)) in enumerate(zip(protocol.seeds,protocol.parity,strict=True))}
    return np.array([lookup[(s,int(a),-int(b))] for s,(a,b) in zip(protocol.seeds,protocol.parity,strict=True)])


def sibling_schedule(trace, start, donors):
    plan = {}
    missing = 0
    for step,(weight,_,valid) in trace.items():
        if step < start:
            continue
        matched = valid & valid[donors]
        plan[step] = np.where(matched,weight[donors],np.where(valid,weight,0.)).astype(np.float32)
        missing += int(np.count_nonzero(valid & ~valid[donors]))
    return plan, missing


def execute(protocol,seed,mode):
    from .trust_study import factory
    actor = factory(protocol.seeds,seed,mode)
    actor.reset(len(protocol.seeds))
    history = world._play(actor,protocol.history["related"])
    post_a = deepcopy(actor) if mode=="learned" else None
    start = protocol.history["related"].menu.body.trials
    intact,_,_ = world._series(protocol,actor,"related",history)
    outcomes = {"intact":intact}
    donors = sign_twins(protocol)
    missing = 0
    if post_a is not None:
        post_a.schedule, missing = sibling_schedule(actor.gate_trace,start,donors)
        yoked,_,_ = world._series(protocol,post_a,"related",history)
        outcomes["sign_yoked"] = yoked
    numeric = {}
    for variant,doses in outcomes.items():
        for dose,metrics in doses.items():
            for key in ("new","old","joint","b_true_receipts","history_receipts","new_contact","new_silent_receipts"):
                numeric[f"{variant}/{dose}/{key}"] = metrics[key]
    for d in protocol.doses[1:]:
        step=start+3*d//2-1
        default=(np.zeros(len(donors)),None,np.zeros(len(donors),bool))
        weight,_,valid = actor.gate_trace.get(step,default)
        both = valid & valid[donors]
        prefix=f"gate/{d}/"
        numeric[prefix+"weight_sum"] = np.where(valid,weight,0.)
        numeric[prefix+"count"] = valid.astype(int)
        numeric[prefix+"twin_abs_delta_sum"] = np.where(both,np.abs(weight-weight[donors]),0.)
        numeric[prefix+"twin_count"] = both.astype(int)
    return numeric, missing


def stamp():
    code = Path(__file__).parent
    required = [Path(__file__),EVIDENCE/"overlap-protocol.json",EVIDENCE/"checkpoint.json",EVIDENCE/"seal.json",
                *(code/f for f in ("trust_study.py","trust_gate.py","relation_world.py","relation_learner.py"))]
    files = {p.relative_to(ROOT).as_posix():digest(p) for p in required}
    key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    return files, DATA/key[:16]


def run_one(job):
    seed,mode=job
    _,folder=stamp()
    path=folder/f"{seed}-{mode}.npz"
    if path.exists(): return
    bank={}
    missing=0
    ids=roots()
    for left in range(0,len(ids),32):
        base=world.make(ids[left:left+32],partition="heldout",history_pairs=12,probes=REG["probes_per_dose"],
                        doses=tuple(REG["doses"]),nonce=NONCE)
        for angle in REG["angles_degrees"]:
            rows,n_missing=execute(prior_at_angle(base,angle),seed,mode)
            missing+=n_missing
            for key,v in rows.items():
                v = np.asarray(v)
                if v.shape!=(len(base.seeds),) or not np.isfinite(v).all(): raise ValueError(key)
                bank.setdefault(f"{angle}/{key}",[]).append(v)
    folder.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix(".partial.npz")
    np.savez_compressed(partial,roots=np.asarray(ids),fallback_writes=np.array(missing),
                        **{k:np.concatenate(v) for k,v in bank.items()})
    partial.replace(path)
    print(f"evaluated graded overlap, birth {seed} {mode}",flush=True)


def report():
    source,folder=stamp()
    estimates, contrasts, gate, archive = {}, {}, {}, {}
    total_fallback=0
    ci=Intervals(len(REG["births"]),REG["worlds"])
    saved={}
    for mode in REG["modes"]:
        bank={}
        for seed in REG["births"]:
            path=folder/f"{seed}-{mode}.npz"
            archive[path.name]=digest(path)
            with np.load(path,allow_pickle=False) as raw:
                if list(raw["roots"]) != roots(): raise ValueError("Wrong physical roots")
                total_fallback+=int(raw["fallback_writes"])
                for key in raw.files:
                    if key not in ("roots","fallback_writes"):
                        bank.setdefault(key,[]).append(raw[key].reshape(-1,4).mean(axis=1))
        values={k:np.stack(v) for k,v in bank.items()}
        saved[mode]=values
        for angle in REG["angles_degrees"]:
            for variant in (("intact","sign_yoked") if mode=="learned" else ("intact",)):
                p=f"{angle}/{variant}/"
                if not np.allclose(values[p+"0/new"],.5*values[p+"0/new_contact"]):
                    raise ValueError("Zero-dose parity leak")
                early=(values[p+"2/new"]+values[p+"4/new"])/2
                estimates[f"{mode}/{angle}/{variant}/early"]=ci.estimate(early)
                for d in REG["doses"]:
                    for k in ("new","old","joint","b_true_receipts","new_contact"):
                        estimates[f"{mode}/{angle}/{variant}/{d}/{k}"]=ci.estimate(values[p+f"{d}/{k}"])
            for d in REG["doses"][1:]:
                p=f"{angle}/gate/{d}/"
                gate[f"{mode}/{angle}/{d}"]={
                    "mean_weight_given_write":float(values[p+"weight_sum"].sum()/values[p+"count"].sum()),
                    "mean_absolute_change_if_B_sign_flips":float(values[p+"twin_abs_delta_sum"].sum()/values[p+"twin_count"].sum())}
    for angle in REG["angles_degrees"]:
        p=f"{angle}/intact/"
        def early(mode,variant):
            pref=f"{angle}/{variant}/"
            return (saved[mode][pref+"2/new"]+saved[mode][pref+"4/new"])/2
        for other in ("always","constant","local"):
            contrasts[f"{angle}/early/learned_minus_{other}"]=ci.estimate(early("learned","intact")-early(other,"intact"))
        contrasts[f"{angle}/early/learned_minus_sign_yoked"]=ci.estimate(early("learned","intact")-early("learned","sign_yoked"))
    write(EVIDENCE/"overlap-summary.json",dict(protocol=REG,source=source,raw_sha256=archive,
          root_seeds=roots(),estimates=estimates,contrasts=contrasts,gate=gate,missing_donor_writes=total_fallback))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("evaluate","report"))
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    from .trust_study import seal
    seal()
    if args.stage=="report":report(); return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run_one,[(s,m) for m in REG["modes"] for s in REG["births"]]))
    report()


if __name__=="__main__":
    main()
