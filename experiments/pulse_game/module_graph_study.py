"""Compare local cycles joined by random bridges to matched single-ring and free births."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from . import free_graph_model, free_graph_study as main_study, module_graphs, plastic_topology_model
from . import ring_logic_model as recipe, ring_logic_study as study, ring_logic_world as env


ARMS=("single64","modules64","free64","single_min","modules_min")
SUBSETS={"calibration":(68730000,48), "standard":(68731000,128), "long":(68732000,128)}
BASE=study.REPO/"data/pulse-game/module-graph-20261119"
EVIDENCE=study.REPO/"docs/evidence/free-graph-2026-11-19/modules"
STEPS=(240,1200)
KEYS=recipe.PARAMETERS


def arcs(seed,arm):
    if arm=="single64":return free_graph_model.graph_for(seed,"free_chords")
    if arm=="free64":return free_graph_model.graph_for(seed,"fully_free")
    if arm=="single_min":return module_graphs.single_sparse(seed)
    return module_graphs.graph_for(seed,64 if arm=="modules64" else "minimum")


def population():
    return main_study.births()


def make_bank(rows):
    masks=np.zeros((len(rows),recipe.WIDTH,recipe.WIDTH),dtype=bool)
    for i,(_,_,seed,_,arm) in enumerate(rows):
        s,t=arcs(seed,arm).T
        masks[i,t,s]=True
    return plastic_topology_model.Bank([r[2] for r in rows],masks)


def episode(stage):
    first,count=SUBSETS[stage]
    return main_study.scoring_data(range(first,first+count),"long" if stage=="long" else "standard")


def fingerprints():
    code=Path(__file__).parent
    files=["module_graphs.py","module_graph_study.py","plastic_topology_model.py","free_graph_model.py",
           "free_graph_study.py","ring_logic_model.py","ring_logic_world.py","ring_logic_study.py","world.py"]
    return {f:study.sha(code/f) for f in files}


def register():
    path=BASE/"registered.json"
    if path.exists():
        if study.read(path)["sources"]!=fingerprints():raise RuntimeError("Module registration source differs")
        return
    comparison=main_study.roots("train")
    worlds=[root for first,count in SUBSETS.values() for root in range(first,first+count)]
    assert len(set(worlds))==len(worlds) and not set(worlds).intersection(comparison)
    samples=population().tolist()
    graphs={}
    for s in np.ravel(samples):
        graphs[str(s)]={name:module_graphs.sample(int(s),count).record()
                        for name,count in (("modules64",64),("modules_min","minimum"))}
    study.write(BASE/"graphs.json",graphs)
    study.write(path, dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()), sources=fingerprints(),
        arms=ARMS, births=samples, graph_sha256=study.sha(BASE/"graphs.json"),
        construction="Sample uniform permutations without fixed points until at least two cycles occur. No fixed ring count or sizes. Each node has exactly one incoming/outgoing internal edge. For 64 edges sample 32 additional distinct ordered cross-ring arcs uniformly conditioned on each ring having incoming AND outgoing links. No additional global SCC rejection; explicitly audit global reachability. For minimum globally closed control, connect rings in one random cyclic ordering with one independently sampled sending/receiving node on each bridge.",
        budget="32 nodes, 449 trained coefficients, one per-receiver recurrent gain normalized by sqrt(current indegree). Single and modular minimum controls have identical total 32+number-of-rings edges per seed. Two matched random starts per group, same sampled signed physical receipts and 20 ordered table cells. Adam lr .024; CPU float32; independent 240- and 1200-step calibration selects among two restarts per arm/rule. No structural changes during this experiment.",
        training_roots=[comparison.start,len(comparison)], subsets=SUBSETS, budgets=STEPS,
        estimands="At 1200 standard, modular64 minus single64; modular64 minus free64; modular_min minus single_min, for each of 3 rules. Paired crossed bootstrap of 16 groups and 128 world roots, Bonferroni over 9 contrasts. Each root includes all 20 cells. Timeout is an error. Long-gap and short-budget descriptive; also report fraction of scenes with all 20 actions correct.",
        scoring="A common, randomly generated sensory prefix is executed continuously into the scored query. Movement choices are passed through the original physical body. Held-out worlds change geometry/noise/timing within profile, not the truth tables."))


@cache
def training():
    return main_study.training_data(torch.device("cpu")),episode("calibration")


def train(group):
    torch.set_num_threads(1)
    if (BASE/"frozen.json").exists():raise RuntimeError("Training is closed")
    if (BASE/f"checkpoints/group-{group:02}.json").exists():return
    begin=time.perf_counter()
    data,cal=training()
    rows=[(group,r,int(s),j,arm) for r,s in enumerate(population()[group]) for j in range(len(env.RULES)) for arm in ARMS]
    bank=make_bank(rows);optimizer=torch.optim.Adam(bank.parameters(),lr=.024)
    torch.manual_seed(5837401+group)
    labels=torch.tensor([r[3] for r in rows]); groups=torch.arange(2)
    reports=[]
    for step in range(1,max(STEPS)+1):
        loss=recipe.train_step(bank,*data,labels,groups,env.TIMING["standard"][0],1,optimizer)
        if step in STEPS:
            if not torch.isfinite(loss) or int(optimizer.state[bank.drive]["step"])!=step:raise RuntimeError("Invalid fit")
            f=BASE/f"checkpoints/group-{group:02}-{step}.npz"
            main_study.write_arrays(f,{k:p.detach().numpy() for k,p in bank.named_parameters()})
            y=study.score(bank,rows,cal)
            values=(y[...,2]>0).mean(axis=(1,2)).reshape(2,len(env.RULES),len(ARMS))
            reports.append(dict(updates=step,file=f.name,sha256=study.sha(f),
                                validation=values.tolist(),selected=values.argmax(axis=0).tolist()))
    study.write(BASE/f"checkpoints/group-{group:02}.json",dict(rows=rows,records=reports,seconds=time.perf_counter()-begin))
    print(f"fitted modular group {group} in {time.perf_counter()-begin:.1f}s",flush=True)


def seal():
    reg=study.read(BASE/"registered.json")
    if reg["sources"]!=fingerprints() or reg["graph_sha256"]!=study.sha(BASE/"graphs.json"):raise RuntimeError("Registration changed")
    records=[study.read(BASE/f"checkpoints/group-{g:02}.json") for g in range(len(population()))]
    for group in records:
        for r in group["records"]:
            if r["sha256"]!=study.sha(BASE/"checkpoints"/r["file"]):raise RuntimeError("Weights changed")
    info=dict(registration=study.sha(BASE/"registered.json"), groups=records)
    path=BASE/"frozen.json"
    if path.exists() and study.read(path)!=info:raise RuntimeError("Selection changed")
    if not path.exists():study.write(path,info)
    return info


@cache
def confirmation():
    return {s:episode(s) for s in ("standard","long")}


def score(group):
    torch.set_num_threads(1)
    path=BASE/f"scores/group-{group:02}.npz"
    if path.exists():return
    record=study.read(BASE/"frozen.json")["groups"][group]
    rows=record["rows"];arrays={}
    for r in record["records"]:
        j,k=np.indices((len(env.RULES),len(ARMS)))
        sel=np.ravel_multi_index((np.asarray(r["selected"]),j,k),(2,len(env.RULES),len(ARMS))).ravel()
        selected=[rows[i] for i in sel]
        bank=make_bank(selected)
        with np.load(BASE/"checkpoints"/r["file"],allow_pickle=False) as data:
            bank.load_state_dict({k:torch.from_numpy(data[k][sel]) for k in KEYS})
        for prof,ctx in confirmation().items():
            y=study.score(bank,selected,ctx)
            x=y.reshape(len(env.RULES),len(ARMS),-1,len(env.PAIRS),3).astype(np.int8)
            inferred=(x[...,0]==env.truth()[:,None,None,:]) & (x[...,0]>=0)
            if not np.array_equal(inferred,x[...,2]>0):raise RuntimeError("Event mismatch")
            arrays[f"{r['updates']}/{prof}"]=x
    main_study.write_arrays(path,arrays)


def summarize():
    seal()
    paths=sorted((BASE/"scores").glob("group-*.npz"))
    if len(paths)!=len(population()):raise RuntimeError("Missing selected groups")
    outputs={}
    for p in paths:
        with np.load(p,allow_pickle=False) as raw:
            for k in raw.files:outputs.setdefault(k,[]).append(raw[k])
    rng=np.random.default_rng(683175)
    groups=len(paths); n=SUBSETS["standard"][1]
    bg=rng.multinomial(groups,np.ones(groups)/groups,20000)/groups
    wr=rng.multinomial(n,np.ones(n)/n,20000)/n
    estimates={}
    for k,blocks in outputs.items():
        full=np.stack(blocks)
        correct=full[...,2]>0; means=correct.mean(axis=-1)
        tab={}
        for j,rule in enumerate(env.RULES):
            arms={}
            for a,name in enumerate(ARMS):
                x=correct[:,j,a];side=full[:,j,a,:,:,0]
                arms[name]=dict(accuracy=float(x.mean()),timeouts=int((side<0).sum()),all20=float(x.all(-1).mean()),
                                per_group=x.mean(axis=(1,2)).tolist())
            comparisons={}
            for left,right in (("modules64","single64"),("modules64","free64"),("modules_min","single_min")):
                delta=means[:,j,ARMS.index(left)]-means[:,j,ARMS.index(right)]
                samples=((bg@delta)*wr).sum(axis=1)
                tail=.025/9 if k=="1200/standard" else .025
                comparisons[f"{left}-{right}"]=dict(estimate=float(delta.mean()),ci=np.quantile(samples,[tail,1-tail]).tolist())
            tab[rule]=dict(arms=arms,contrasts=comparisons)
        estimates[k]=tab
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    bundle=EVIDENCE/"raw.zip"
    with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as out:
        for f in sorted(BASE.rglob("*")):
            if f.is_file():out.write(f,f.relative_to(BASE).as_posix())
    study.write(EVIDENCE/"summary.json",dict(results=estimates, raw_sha256=study.sha(bundle),
                registered_sha256=study.sha(BASE/"registered.json"),frozen_sha256=study.sha(BASE/"frozen.json")))


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("register","train","confirm","report","all"))
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    register()
    if args.phase=="register":return
    if args.phase in ("all","train"):
        with ProcessPoolExecutor(max_workers=args.workers) as pool:list(pool.map(train,range(len(population()))))
        seal()
    if args.phase in ("all","confirm"):
        seal()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:list(pool.map(score,range(len(population()))))
    if args.phase in ("all","report"):summarize()


if __name__=="__main__":main()
