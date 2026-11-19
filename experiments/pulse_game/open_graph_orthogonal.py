"""Fork the *actual recurrent graph* across complementary, new preference directions."""
from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import zipfile

import numpy as np

from . import mixture_world as food, ring_logic_study as io, unified_live_graph as net
from . import unified_live_graph_study as prior, world


EVIDENCE=io.REPO/"docs/evidence/open-graph-learning-2026-11-19/orthogonal"
REG=io.read(EVIDENCE/"protocol.json")


def private_vectors(root):
    rng=np.random.default_rng([root, 5913157])
    a,b=rng.normal(size=(2,food.DIM))
    a/=np.linalg.norm(a)
    b-=a*np.dot(a,b)
    b/=np.linalg.norm(b)
    return a,b


def make_menu(seeds,stage,length):
    indices=np.asarray(seeds)
    # Same fresh physical world on both twins; no public discontinuity labels.
    physical=[int(np.random.SeedSequence([int(s),stage,3269]).generate_state(1)[0]) for s in seeds]
    m=food.make(physical,length,rule="linear")
    directions=np.stack([private_vectors(s)[int(stage in (2,3))] for s in seeds])
    x=m.chemicals*2-1
    prob=1/(1+np.exp(-4.8*np.einsum("btfi,bi->btf",x,directions)))
    uniform=[]; noise=[]; sham=[]
    for s in seeds:
        rng=np.random.default_rng([s,stage,56245])
        uniform.append(rng.random((length,2)))
        r=rng.random(length).clip(1e-5,1-1e-5)
        noise.append(np.log(r/(1-r)))
        sham.append(rng.choice([-1,1],length))
    signs=np.where(np.asarray(uniform)<prob,1,-1).astype(np.int8)
    if stage in (2,3):
        half=len(seeds)//2
        if not np.array_equal(indices[:half],indices[half:]):raise ValueError("Twin ordering")
        prob[half:]=1-prob[:half];signs[half:]=-signs[:half]
    return replace(m, probability=prob, realized=signs),np.asarray(noise,np.float32),np.asarray(sham,np.int8)


def live(brain,menu,noise,sham,visible):
    body=prior.Body(menu,sham,0 if visible else menu.body.trials)
    start_mask=brain.mask.copy()
    edits=[]
    observations=body.observe()
    for request in range(menu.body.trials):
        brain.start(brain.recipe.exploration*noise[:,request] if visible else np.zeros(len(noise),np.float32))
        for _ in range(world.TICKS):
            observations=body.step(brain.advance(observations))
        _,_,events,_=brain.update(request)
        edits.append(events)
    result=body.result();events=np.stack(edits,axis=1)
    if not visible and (np.any(result.observed_reward) or (events>=0).any() or not np.array_equal(start_mask,brain.mask)):
        raise RuntimeError("Silent probes changed graph")
    truth=menu.probability.argmax(axis=2)
    return np.stack((result.contact_side,result.contact_tick,result.reward,truth),-1),events


def simulate(birth,arm,recipe):
    n=len(REG["roots"]); seeds=REG["roots"]*2
    brain=net.Graph([birth]*len(seeds),recipe)
    twin=np.roll(np.arange(2*n),n)
    out={}
    for stage in range(3):
        label=REG["stages"][stage]
        train=REG["questions_per_stage"]-REG["silent_tail"]
        menu,noise,sham=make_menu(seeds,stage*2,train)
        physical,events=live(brain,menu,noise,sham if arm=="placebo" else None,True)
        out[label+"/teach"]=physical;out[label+"/edits"]=events
        if stage==0:
            assert np.array_equal(brain.state,brain.state[twin]) and np.array_equal(brain.weight,brain.weight[twin])
        if stage==1:
            assert np.array_equal(physical[:n,0,:2],physical[n:,0,:2])
            assert np.array_equal(physical[:n,0,3],1-physical[n:,0,3])
        probes,nnoise,_=make_menu(seeds,stage*2+1,REG["silent_tail"])
        if stage==1 and arm=="joint":
            for what in ("synapses","activity"):
                alternate=deepcopy(brain)
                if what=="synapses":
                    alternate.weight=brain.weight[twin].copy()
                    alternate.mask=brain.mask[twin].copy()
                    alternate.generation=brain.generation[twin].copy()
                    alternate.generation_writes=brain.generation_writes[twin].copy()
                else:alternate.state=brain.state[twin].copy()
                out[label+"/"+what]=live(alternate,probes,nnoise,None,False)[0]
            out["B/end_mask"]=brain.mask.copy();out["B/end_weight"]=brain.weight.copy();out["B/end_state"]=brain.state.copy()
        # All probes branch from the actual post-teaching state. No probe stimuli
        # or terminal motor state enter the next teaching phase.
        out[label+"/silent"]=live(deepcopy(brain),probes,nnoise,None,False)[0]
        if not all(net.strongly_connected(m&(w!=0)) for m,w in zip(brain.mask,brain.weight,strict=True)):
            raise RuntimeError("Disconnected native graph")
    return out


def rate(a):
    # births, twins, roots, foods, fields
    return ((a[...,0]==a[...,3])&(a[...,0]>=0)).mean(axis=(1,3))


def run():
    dev=prior.read(io.REPO/"data/pulse-game/unified-live-20261119/development.json")
    recipe=net.Recipe(**dev["selected"]["recipe"])
    births=REG["births"]; n=len(REG["roots"])
    bank={}
    for arm in ("joint","placebo"):
        for birth in births:
            out=simulate(birth,arm,recipe)
            for key,x in out.items():
                bank.setdefault(arm+"/"+key,[]).append(x.reshape((2,n)+x.shape[1:]))
    bank={k:np.stack(v) for k,v in bank.items()}
    avg={k:rate(x) for k,x in bank.items() if k.rsplit("/",1)[-1] in ("silent","synapses","activity")}
    d=avg["joint/B/silent"]
    b=len(births);rng=np.random.default_rng(86530)
    gb=rng.multinomial(b,np.full(b,1/b),20000)/b
    wb=rng.multinomial(n,np.full(n,1/n),20000)/n
    def ci(x,tail=.025):
        sam=((gb@x)*wb).sum(1)
        return dict(mean=float(x.mean()),ci95=np.quantile(sam,[tail,1-tail]).tolist())
    primary={k:ci(x,.025/3) for k,x in (("B_minus_chance",d-.5),("B_minus_sham",d-avg["placebo/B/silent"]),
                                       ("B_minus_swap_synapses",d-avg["joint/B/synapses"]))}
    physical=bank["joint/B/silent"]
    hits=(physical[...,0]==physical[...,3])&(physical[...,0]>=0)
    archive=EVIDENCE/"physical.zip"
    import io as streams
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for k,x in bank.items():
            buf=streams.BytesIO();np.save(buf,x,allow_pickle=False);z.writestr(k+".npy",buf.getvalue())
    io.write(EVIDENCE/"summary.json",dict(recipe=vars(recipe),rates={k:ci(x) for k,x in avg.items()},primary=primary,
             both_complementary_B_correct=ci((hits[:,0]&hits[:,1]).mean(-1)), all_pass=all(m["ci95"][0]>0 for m in primary.values()),
             B_minus_swap_activity=ci(d-avg["joint/B/activity"]),births=births,roots=REG["roots"],
             edits={k:(x>=0).sum(axis=tuple(range(x.ndim-1))).tolist() for k,x in bank.items() if k.endswith("/edits")},
             archive_sha256=io.sha(archive),source_sha256=io.sha(Path(__file__)),protocol_sha256=io.sha(EVIDENCE/"protocol.json")))


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.parse_args()
    run()
