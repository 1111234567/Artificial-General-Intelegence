from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ.setdefault(_k,"1")
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
import argparse
import hashlib
import json
import random
import time
import torch
from experiments.seed_evolution import core
from experiments.learned_binding import common

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"data/seed-evolution/2026-11-11-v1"
CONFIG=dict(repeats=3,generations=4,population=4,parents=2,sigma=.15,inner_steps=64,
    final_steps=512,source_seeds=[0,1,2],dev_streams=[1100,1101],test_streams=[2100,2101],
    actors=16,support_ticks=2048,probe_ticks=2048,random_seed=78111,b_ticks=4096,
    search_device="cpu",final_device="cpu",dtype="float32",threads_per_worker=1)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):common.write_json(p,x)
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def folder(rep,cid):return OUT/"search"/f"rep_{rep}"/cid
def model_hash(state):
    h=hashlib.sha256()
    for n,t in state.items():h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def declare():
    p=OUT/"protocol.json"
    if p.exists():raise FileExistsError(p)
    files=[Path(__file__),Path(core.__file__),Path(__file__).with_name("analysis-plan.md")]
    for name in ["model.py","read_first.py","frozen_trace.py","pretrain.py","data.py","interaction.py","common.py"]:
        files.append(ROOT/"experiments/learned_binding"/name)
    files += [ROOT/"experiments/associative_state/bridge.py",ROOT/"experiments/lifelong/world.py",ROOT/"experiments/continuity/world.py",ROOT/"experiments/experience_transfer/train_transfer.py",ROOT/"experiments/learn_remember/gru_train.py"]
    origins={str(s):dict(path=str(common.source_checkpoint(s).relative_to(ROOT)),sha256=sha(common.source_checkpoint(s))) for s in range(3)}
    write(p,dict(client_date="2026-11-11",config=CONFIG,source_hashes={str(f.relative_to(ROOT)):sha(f) for f in files},origins=origins,
        data_freeze_ref=dict(path=str(core.DATA_FREEZE.relative_to(ROOT)),sha256=sha(core.DATA_FREEZE)),
        unique_candidates=84,budget_per_arm_per_repeat=16,shared_first_generation=True,
        audit_candidates=[dict(rep=r,id=cid) for r in range(3) for cid in ["shared_g00_i00","evolution_g03_i00","random_g03_i00"]]))


def check():
    p=read(OUT/"protocol.json");assert p["config"]==CONFIG
    for n,h in p["source_hashes"].items():assert sha(ROOT/n)==h,n
    for x in [*p["origins"].values(),p["data_freeze_ref"]]:assert sha(ROOT/x["path"])==x["sha256"]
    return p


def evaluate_candidate(desc):
    torch.set_num_threads(1);check();rep=desc["rep"];path=folder(rep,desc["id"]);path.mkdir(parents=True,exist_ok=False)
    if desc["parent_id"] is None:genome=core.new_genome(rep,desc["init_seed"])
    else:
        parent=torch.load(folder(rep,desc["parent_id"])/"initial.pt",weights_only=False)
        genome=core.mutate_genome(parent,desc["mutation_seed"],sigma=CONFIG["sigma"])
    torch.save(genome,path/"initial.pt");before=core.genome_hash(genome)
    trained=core.inner_train(rep,genome,steps=64,checkpoints=(0,64),device="cpu")
    assert core.genome_hash(genome)==before
    cp=trained["checkpoints"][64];torch.save(cp,path/"inner64.pt")
    score=core.score_model(rep,cp["model"],stream_ids=tuple(CONFIG["dev_streams"]),support_ticks=2048,probe_ticks=2048,actors=16,random_seed=78111)
    write(path/"score.json",score);write(path/"training.json",dict(metadata=trained["metadata"],log=trained["log"]))
    record=dict(**desc,genome_hash=before,trained_hash=model_hash(cp["model"]),fitness=score["fitness"],
        artifacts={n:sha(path/n) for n in ["initial.pt","inner64.pt","score.json","training.json"]})
    write(path/"candidate.json",record)
    return record


def ranked(ids,records):return sorted(ids,key=lambda x:(-records[x]["fitness"],x))


def search(workers=12):
    check()
    if (OUT/"search.json").exists():raise FileExistsError("search already complete")
    records={r:{} for r in range(3)};pools={(r,a):[] for r in range(3) for a in ("evolution","random")};history=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for g in range(4):
            jobs=[];generation=[]
            for r in range(3):
                for arm in (["shared"] if g==0 else ["evolution","random"]):
                    top=[] if g==0 else ranked(pools[r,arm],records[r])[:2]
                    parent_seed=650000+r*100+g
                    rng=random.Random(parent_seed)
                    ids=[]
                    for i in range(4):
                        cid=f"{arm}_g{g:02d}_i{i:02d}";ids.append(cid)
                        parent=rng.choice(top) if arm=="evolution" else None
                        desc=dict(rep=r,source_seed=r,arm=arm,generation=g,index=i,id=cid,parent_id=parent,
                            init_seed=(710000+r*10000+g*100+i if parent is None else None),
                            mutation_seed=(810000+r*10000+g*100+i if parent else None),sigma=.15,
                            parent_selection_seed=parent_seed if arm=="evolution" else None)
                        jobs.append(pool.submit(evaluate_candidate,desc))
                    generation.append(dict(rep=r,arm=arm,generation=g,top2_before=top,new_ids=ids))
            for future in as_completed(jobs):
                record=future.result();records[record["rep"]][record["id"]]=record
                print(json.dumps(dict(event="candidate",rep=record["rep"],id=record["id"],fitness=record["fitness"])),flush=True)
            for entry in generation:
                r=entry["rep"];arms=("evolution","random") if g==0 else (entry["arm"],)
                for arm in arms:
                    pools[r,arm].extend(entry["new_ids"]);best=ranked(pools[r,arm],records[r])[0]
                    row=dict(**{**entry,"arm":arm},evaluated_ids=list(pools[r,arm]),best_id=best,best_fitness=records[r][best]["fitness"])
                    history.append(row)
            write(OUT/f"generation_{g}.json",dict(generation=g,history=history,records=records))
    winners=[dict(rep=r,arm=a,id=ranked(pools[r,a],records[r])[0]) for r in range(3) for a in ("evolution","random")]
    write(OUT/"search.json",dict(history=history,records=records,winners=winners,unique_candidates=sum(map(len,records.values()))))


def train_final(winner):
    r,a,cid=winner["rep"],winner["arm"],winner["id"];path=OUT/"final"/f"rep_{r}"/a;path.mkdir(parents=True,exist_ok=False)
    genome=torch.load(folder(r,cid)/"initial.pt",weights_only=False)
    trained=core.inner_train(r,genome,steps=512,checkpoints=(0,1,64,511,512),device="cpu")
    previous=torch.load(folder(r,cid)/"inner64.pt",weights_only=False)
    common.assert_tree_equal(previous,trained["checkpoints"][64],"search/final64")
    torch.save(trained,path/"training.pt");write(path/"training.json",dict(winner=winner,metadata=trained["metadata"],log=trained["log"],sha256=sha(path/"training.pt")))
    return dict(winner=winner,path=str((path/"training.pt").relative_to(ROOT)),sha256=sha(path/"training.pt"))


def finals(workers=6):
    check();winners=read(OUT/"search.json")["winners"]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows=list(pool.map(train_final,winners))
    write(OUT/"final-freeze.json",dict(all_final_training_complete_before_test=True,models=rows))


def test_final(row):
    torch.set_num_threads(1);winner=row["winner"];r,a=winner["rep"],winner["arm"]
    path=ROOT/row["path"];assert sha(path)==row["sha256"];trained=torch.load(path,weights_only=False)
    dest=path.parent/"evaluations";dest.mkdir(exist_ok=False);items=[]
    for step,condition in [(0,"own"),(64,"own"),(512,"own"),(512,"teacher"),(512,"no_read"),(512,"Bdelay")]:
        result=core.score_model(r,trained["checkpoints"][step]["model"],stream_ids=(2100,2101),actors=16,support_ticks=2048,probe_ticks=2048,random_seed=78111,
            teacher_support=condition=="teacher",read_enabled=condition!="no_read",b_ticks=4096 if condition=="Bdelay" else 0,archive=True)
        archives=result.pop("archives");stem=dest/f"{step:03d}_{condition}"
        torch.save(archives,stem.with_suffix(".pt"));result.update(winner=winner,step=step,condition=condition,archives_sha256=sha(stem.with_suffix(".pt")))
        write(stem.with_suffix(".json"),result);items.append(dict(name=stem.name,fitness=result["fitness"],sha256=sha(stem.with_suffix(".json"))))
    write(dest/"completion.json",dict(winner=winner,conditions=items));return dict(winner=winner,conditions=items)


def tests(workers=6):
    check();frozen=read(OUT/"final-freeze.json");assert frozen["all_final_training_complete_before_test"]
    with ProcessPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(test_final,frozen["models"]))
    write(OUT/"test-completion.json",dict(models=rows))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("command",choices=("declare","search","finals","test"));p.add_argument("--workers",type=int)
    a=p.parse_args()
    if a.command=="declare":declare()
    elif a.command=="search":search(a.workers or 12)
    elif a.command=="finals":finals(a.workers or 6)
    else:tests(a.workers or 6)
