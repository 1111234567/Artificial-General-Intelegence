"""Can a mature port graph retain its nonlinear skill with a sparse set of pair coefficients?"""
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from . import dendrite_ports_check as check, graph_interactions as task, ring_logic_study as io


EVIDENCE=io.REPO/"docs/evidence/open-graph-learning-2026-11-19/dendrites"
DATA=io.REPO/"data/pulse-game/dendrite-ports-20261119"


def controls(q,births,worlds):
    top=np.zeros_like(q);random=np.zeros_like(q)
    for lane in range(len(q)):
        rng=np.random.default_rng([int(births[lane]),int(worlds[lane]),294507])
        for receiver,x in enumerate(q[lane]):
            keys=np.flatnonzero(x)
            if not len(keys):continue
            k=(len(keys)+3)//4
            ranked=keys[np.argsort(-np.abs(x[keys]),kind="stable")[:k]]
            shuffled=rng.choice(keys,k,replace=False)
            top[lane,receiver,ranked]=x[ranked]
            random[lane,receiver,shuffled]=x[shuffled]
    assert np.array_equal((top!=0).sum(-1),(random!=0).sum(-1))
    return top,random


def main():
    summary=io.read(DATA/"summary.json")
    rng=np.random.default_rng(390821);n,m=len(check.BIRTH_TEST),len(check.WORLD_TEST)
    draws=rng.integers(n,size=(20000,n,1)),rng.integers(m,size=(20000,1,m))
    cases={};archive=EVIDENCE/"quarter-probes.zip"
    with ZipFile(archive,"w",ZIP_DEFLATED) as out:
        for mode in check.MODES:
            data=check.load(DATA,summary,mode)
            last=check.TRIALS+check.EXTRA
            long,_,_,_=task.make_menu(data["world"],mode,last+check.EXTRA)
            idx=np.s_[:,last:]
            parts={k:getattr(long.body,k)[idx] for k in ("ctx","cue","left_id","good_id","distances","wind")}
            menu=replace(long,body=replace(long.body,**parts),chemicals=long.chemicals[idx],
                         probability=long.probability[idx],realized=long.realized[idx])
            original=data["product/final_pair"]
            top,random=controls(original,data["birth"],data["world"])
            stats={};accuracies={}
            for arm,q in (("intact",original),("largest",top),("random",random)):
                branch=data | {"product/final_pair":q}
                result=check.probe(branch,menu,summary["recipe"],cut=False)
                accuracies[arm]=result["correct"].mean(1)
                stats[arm]=dict(accuracy=float(accuracies[arm].mean()),coverage=float((result["contact"]>=0).mean()),
                                active_pair_coefficients_per_graph=float((q!=0).sum(axis=(1,2)).mean()))
                buf=BytesIO();np.savez(buf,contact=result["contact"],tick=result["tick"],truth=result["truth"],
                                      nonzero=(q!=0).sum(axis=(1,2)),birth=data["birth"],world=data["world"])
                out.writestr(f"{mode}/{arm}.npz",buf.getvalue())
            stats["largest_minus_intact"]=check.interval(accuracies["largest"]-accuracies["intact"],.025/4,draws)
            stats["largest_minus_random"]=check.interval(accuracies["largest"]-accuracies["random"],.025/4,draws)
            cases[mode]=stats
    io.write(EVIDENCE/"quarter-summary.json",dict(results=cases,source_sha256=io.sha(Path(__file__)),
             original=io.sha(DATA/"summary.json"), archive_sha256=io.sha(archive),
             protocol_sha256=io.sha(EVIDENCE/"quarter-protocol.json")))
    print(cases)


if __name__=="__main__": main()
