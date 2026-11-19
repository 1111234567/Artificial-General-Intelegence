"""Parallel CPU scheduling; finish and freeze all training before scoring."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
import argparse
import json
from pathlib import Path
import torch
from experiments.associative_state.train import OUT, ARMS, train, sha, write_json


def train_one(job):
    seed,arm=job
    log=OUT/"logs"/f"train_{seed}_{arm}.log";log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("w",encoding="utf-8") as f,redirect_stdout(f),redirect_stderr(f):train(seed,arm)
    return dict(stage="train",seed=seed,arm=arm,log=str(log))


def eval_one(job):
    from experiments.associative_state.evaluate import run
    seed,arm=job
    log=OUT/"logs"/f"eval_{seed}_{arm}.log"
    with log.open("w",encoding="utf-8") as f,redirect_stdout(f),redirect_stderr(f):run(seed,arm)
    return dict(stage="eval",seed=seed,arm=arm,log=str(log))


def freeze():
    files={}
    for seed in range(3):
        rows=[json.loads((OUT/f"seed_{seed:03d}"/a/"training.json").read_text(encoding="utf-8")) for a in ARMS]
        assert len({r["train_sample_sha256"] for r in rows})==1
        left,right=[torch.load(OUT/f"seed_{seed:03d}"/a/"initial.pt",weights_only=True) for a in ARMS[:2]]
        assert left.keys()==right.keys() and all(torch.equal(left[k],right[k]) for k in left)
        for arm in ARMS:
            p=OUT/f"seed_{seed:03d}"/arm/"final.pt";files[str(p.relative_to(OUT))]=sha(p)
    write_json(OUT/"training-freeze.json",dict(all_training_complete_before_scoring=True,
        paired_train_samples=True,paired_add_delta_initial_parameters=True,checkpoints=files))


def launch(stage,workers):
    if stage=="eval":
        frozen=json.loads((OUT/"training-freeze.json").read_text(encoding="utf-8"))
        for name,h in frozen["checkpoints"].items():assert sha(OUT/name)==h
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(train_one if stage=="train" else eval_one,(s,a)) for s in range(3) for a in ARMS]
        for future in as_completed(futures):print(json.dumps(future.result()),flush=True)
    if stage=="train":freeze()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("train","eval"));p.add_argument("--workers",type=int,default=9)
    a=p.parse_args();launch(a.stage,a.workers)
