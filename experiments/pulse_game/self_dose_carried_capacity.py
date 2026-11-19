"""Repeat the same continuous-state full-credit cell in the smaller writable graph."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch
import zipfile

import numpy as np

from . import self_dose_continuous as base


def narrow(seed):
    job=(seed,"fresh","all")
    with patch.object(base,"DATA",base.DATA/"narrow"),patch.dict(base.REG,{"writer_nodes":4}):
        base.train(job)
        base.evaluate(job)


def report():
    roots=base.REG["births"]
    result={}
    rng=np.random.default_rng(713024)
    n=48
    birth_draws=rng.multinomial(len(roots),np.ones(len(roots))/len(roots),20000)/len(roots)
    for env in ("short","clean","noise","change"):
        cells={4:[],8:[]}
        for seed in roots:
            for width,folder in ((4,base.DATA/"narrow"),(8,base.DATA)):
                with np.load(folder/f"{seed}-fresh-all.scores.npz",allow_pickle=False) as file:
                    arr=file[env+"/live"]
                    good=arr[...,0]==arr[...,1]
                    cells[width].append(good[:,-(4 if env=="short" else 8):].mean(1).reshape(-1,2).mean(1))
        diff=np.array(cells[8])-np.array(cells[4])
        worlds=np.stack([(rng.multinomial(n,np.ones(n)/n,20000)/n)@row for row in diff],axis=1)
        draws=(birth_draws*worlds).sum(1)
        tail=.025/3 if env!="clean" else .025
        result[env]=dict(narrow=float(np.mean(cells[4])),wide=float(np.mean(cells[8])),
                         difference=float(diff.mean()),ci=np.quantile(draws,[tail,1-tail]).tolist(),
                         by_birth=diff.mean(1).tolist())
    archive=base.EVIDENCE/"narrow-traces-and-checkpoints.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted((base.DATA/"narrow").glob("*.npz")):z.write(p,p.name)
    base.io.write(base.EVIDENCE/"capacity.json",dict(cells=result,
         plan_sha256=base.io.sha(base.EVIDENCE/"capacity-plan.json"),archive_sha256=base.io.sha(archive)))


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__);parser.add_argument("--workers",type=int,default=4)
    options=parser.parse_args()
    with ProcessPoolExecutor(max_workers=options.workers) as pool:list(pool.map(narrow,base.REG["births"]))
    report()
