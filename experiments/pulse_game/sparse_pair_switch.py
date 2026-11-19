"""Can receipt-grown native pairs adapt to a change in which chemical components interact?"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import zipfile

import numpy as np

from . import graph_interactions as task, ring_logic_study as io, world
from .unified_live_graph import Recipe, strongly_connected
from .unified_live_graph_study import Body


EVIDENCE = io.REPO / "docs/evidence/sparse-native-2026-11-19/switch"
MAIN = EVIDENCE.parent / "main"
REG = io.read(EVIDENCE/"protocol.json")
DATA = io.REPO / "data/pulse-game/sparse-switch-20261119"


def setup():
    return io.read(MAIN/"development.json")["selected"]["setup"]


def graph(births,roots,mode):
    if mode=="alternative":
        from . import sparse_alt_study as alt
        choice=io.read(io.REPO/"data/pulse-game/sparse-native-20261119-alternative/development.json")
        policy=alt.ProbePolicy(**choice["runs"][choice["winner"]]["policy"])
        return alt.brain_for("corroborating",births,roots,policy)
    from .sparse_pairs_model import DenseWGraph, SparseDendriteGraph, PairPolicy
    conf=setup()
    shared=dict(pair_rate=conf["pair_rate"],pair_decay=conf["pair_decay"])
    base=Recipe(**conf["recipe"])
    if mode=="dense":return DenseWGraph(births,base,product=True,**shared)
    return SparseDendriteGraph(births,base,world_seeds=roots,policy=PairPolicy(**conf["policy"]),**shared)


def meal(roots,stage,*,probe=False,negative=False):
    length=REG["silent_trials"] if probe else REG["training_trials_per_stage"]
    stream=2*stage+int(probe)
    draws=[int(np.random.SeedSequence([int(s),stream,8137503]).generate_state(1)[0]) for s in roots]
    seeds=tuple(dict.fromkeys(roots))
    unique_indices=[roots.index(s) for s in seeds]
    cache,_,noise,_=task.make_menu([draws[i] for i in unique_indices],"stable",length)
    root_index={s:i for i,s in enumerate(seeds)}
    ix=np.array([root_index[s] for s in roots])
    weights=[]
    for s in seeds:
        rng=np.random.default_rng([int(s),int(stage==1),8137504])
        w=rng.normal(size=2 if stage==1 else 4)
        weights.append(w/np.linalg.norm(w))
    weights=np.array(weights)
    pairs=np.array([[0,1],[2,3]]) if stage==1 else task.CROSS
    features=task.interactions(cache.chemicals*2-1,pairs)
    utilities=np.einsum("btij,bj->bti",features,weights)
    p=(1/(1+np.exp(-task.TASTE_GAIN*utilities))).astype(np.float32)
    uniforms=np.array([np.random.default_rng([int(s),stream,534508]).random((length,2)) for s in seeds])
    tastes=np.where(uniforms<p,1,-1).astype(np.int8)
    if negative and stage==1: p=1-p;tastes=-tastes
    attrs={k:getattr(cache.body,k)[ix] for k in ("ctx","cue","left_id","good_id","good_map","distances","wind")}
    body=replace(cache.body,seeds=tuple(draws),**attrs)
    return replace(cache,body=body,chemicals=cache.chemicals[ix],realized=tastes[ix],
                   probability=p[ix],weights=weights[ix]), noise[ix]


def portion(menu,sl):
    names=("ctx","cue","left_id","good_id","wind","distances")
    body=replace(menu.body,**{name:getattr(menu.body,name)[:,sl] for name in names})
    return replace(menu,body=body,chemicals=menu.chemicals[:,sl],realized=menu.realized[:,sl],
                   probability=menu.probability[:,sl])


def counts(brain):
    if hasattr(brain,"active_counts"):return brain.active_counts().copy()
    if hasattr(brain,"live"):return np.bincount(brain.live[0],minlength=len(brain.state))
    return (brain.pair!=0).sum(axis=(1,2))


def play(brain,menu,noise,offset,*,silent=False):
    if silent: brain=deepcopy(brain)
    before=counts(brain)
    body=Body(menu,None,menu.body.trials if silent else 0)
    obs=body.observe();events=[];used=[]
    for i in range(menu.body.trials):
        brain.start(np.zeros(body.tape.n,np.float32) if silent else brain.recipe.exploration*noise[:,i])
        for _ in range(world.TICKS): obs=body.step(brain.advance(obs))
        _,_,edits,_=brain.update(offset+i)
        events.append(edits)
        used.append(counts(brain))
    res=body.result()
    shown=res.observed_reward.sum(axis=-1)
    events=np.stack(events,axis=1)
    if silent and (shown.any() or (events>=0).any() or not np.array_equal(before,counts(brain))):
        raise RuntimeError("Silent probe learned")
    codes=np.stack((res.contact_side,res.contact_tick,res.reward,menu.probability.argmax(2),shown),axis=-1)
    return codes.astype(np.int8),events,np.stack(used,axis=1).astype(np.int16)


def simulate(birth_ids,mode):
    roots=REG["roots"]*len(birth_ids)
    births=np.repeat(birth_ids,len(REG["roots"]))
    brain=graph(births,roots,mode)
    length=REG["training_trials_per_stage"]
    source=meal(roots,0)
    initial,_,used_a=play(brain,*source,0)
    a_probe=play(brain,*meal(roots,0,probe=True),length,silent=True)[0]
    results={"A/training":initial,"A/count":used_a,"A/probe":a_probe}
    for twin in (0,1):
        current=deepcopy(brain)
        if mode=="no_new_in_B":current.proposals_enabled=False
        menu,noise=meal(roots,1,negative=twin)
        b_probe=meal(roots,1,negative=twin,probe=True)
        start=0
        for dose in REG["B_doses"]:
            if start<dose:
                cut=slice(start,dose)
                obs,_,use=play(current,portion(menu,cut),noise[:,cut],length+start)
                results[f"{twin}/B/training/{dose}"]=obs
                results[f"{twin}/B/count/{dose}"]=use
            results[f"{twin}/B/{dose}"]=play(current,*b_probe,length+dose,silent=True)[0]
            start=dose
        if mode=="no_new_in_B": current.proposals_enabled=True
        obs,_,use=play(current,*meal(roots,2),2*length)
        results[f"{twin}/return/training"]=obs
        results[f"{twin}/return/count"]=use
        results[f"{twin}/return/probe"]=play(current,*meal(roots,2,probe=True),3*length,silent=True)[0]
        if not all(strongly_connected(mask&(w!=0)) for mask,w in zip(current.mask,current.weight,strict=True)):
            raise RuntimeError("Disconnected physical graph")
    # Zero B evidence: identical action and geometry, with exactly complementary B truth.
    np.testing.assert_array_equal(results["0/B/0"][:,:,:2],results["1/B/0"][:,:,:2])
    np.testing.assert_array_equal(results["0/B/0"][:,:,3],1-results["1/B/0"][:,:,3])
    return {k:v.reshape((len(birth_ids),len(REG["roots"]))+v.shape[1:]) for k,v in results.items()}


def run_chunk(job):
    mode,idx=job
    path=DATA/f"{mode}-{idx}.npz"
    result=simulate(REG["births"][idx:idx+2],mode)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".partial.npz")
    np.savez_compressed(tmp,**result);tmp.replace(path)
    print("completed",mode,idx,flush=True)


def accuracy(z):
    return ((z[...,0]==z[...,3])&(z[...,0]>=0)).mean(axis=-1)


def report():
    estimates={};curve={};archive=EVIDENCE/"raw.zip"
    rng=np.random.default_rng(673681)
    n,m=len(REG["births"]),len(REG["roots"])
    b=rng.multinomial(n,np.full(n,1/n),20000)/n
    w=rng.multinomial(m,np.full(m,1/m),20000)/m
    def ci(x,tail=.025):
        draws=((b@x)*w).sum(1)
        return dict(mean=float(x.mean()),ci95=np.quantile(draws,[tail,1-tail]).tolist())
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for mode in REG["modes"]:
            chunks=[]
            for i in range(0,n,2):
                p=DATA/f"{mode}-{i}.npz"
                with np.load(p,allow_pickle=False) as f:chunks.append({k:f[k] for k in f.files})
                z.write(p,p.name)
            data={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
            means={}
            for d in REG["B_doses"]:
                means[f"B{d}"]=(accuracy(data[f"0/B/{d}"])+accuracy(data[f"1/B/{d}"]))/2
            means["A"]=accuracy(data["A/probe"])
            means["A_return"]=(accuracy(data["0/return/probe"])+accuracy(data["1/return/probe"]))/2
            counts_b=np.stack([data[f"{t}/B/count/640"][...,-1] for t in (0,1)])
            estimates[mode]=dict(probes={k:ci(x) for k,x in means.items()},pairs_end_B=float(counts_b.mean()),
                pairs_max_B=int(max(np.max(data[f"{t}/B/count/{d}"]) for t in (0,1) for d in (80,240,640))),
                B_coverage=float(np.mean([np.mean(data[f"{t}/B/640"][...,0]>=0) for t in (0,1)])))
            curve[mode]=means["B640"]
    primary={k:ci(curve["guided"]-v,.025/3) for k,v in (("minus_chance",.5),("minus_dense",curve["dense"]),("minus_no_new",curve["no_new_in_B"]))}
    io.write(EVIDENCE/"summary.json",dict(results=estimates,primary=primary,recipe=setup(),
        raw_sha256=io.sha(archive),protocol_sha256=io.sha(EVIDENCE/"protocol.json"), code_sha256=io.sha(Path(__file__))))


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("run","report"));p.add_argument("--workers",type=int,default=4)
    a=p.parse_args()
    if a.phase=="run":
        setup()
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            list(pool.map(run_chunk,[(mode,i) for i in range(0,len(REG["births"]),2) for mode in REG["modes"]]))
    report()
