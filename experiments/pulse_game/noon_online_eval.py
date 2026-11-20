"""Deploy reward-learned review readouts on new geometries, bodies and memory loads."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import torch

from . import noon_online_readout as fit, noon_readback as bank, noon_readback_fit as initial
from .interleave_capacity_report import Crossed


def collect(cohort,family,count,worlds=384):
    torch.set_num_threads(1)
    seed=9651577 if cohort=="original" else 9651581
    ids=list(bank.world.COHORTS[cohort]);t,first,assigned=bank.tape(seed,worlds,count)
    folder=fit.OUT/"heldout"/cohort/family/str(count);folder.mkdir(exist_ok=True,parents=True)
    files={};checks={}
    for b in ids:
        if family=="uniform":root,tag,stage=bank.world.parent_path(cohort),"uniform",8000
        else:root,tag,stage=bank.world.PARENT_DATA/"one-life-repair"/cohort,"mixed",2000
        cp=bank.old.checkpoint_path(root,"odor",tag,b,stage);check=sha256(cp.read_bytes()).hexdigest()
        dest=folder/f"{b}.npz"
        if not dest.exists():
            graph,rule=bank.old.load_checkpoint(root,"odor",tag,b,stage);items=[]
            for j in range(0,worlds,32):
                ix=torch.arange(8*j,min(t.lanes,8*(j+32)))
                items.append(bank.trials(graph,rule,bank.lesson.expand(t,ix),first[ix],assigned[ix],j==0))
            raw={k:np.concatenate([d[k] for d in items]) for k in items[0]}
            bank.lesson.storage._write_raw(dest,**raw,checkpoint=check,seed=seed,worlds=worlds,
                                          generator=sha256(Path(bank.__file__).read_bytes()).hexdigest())
        with np.load(dest) as z:
            if str(z["checkpoint"])!=check or int(z["seed"])!=seed or int(z["worlds"])!=worlds:raise ValueError("Incorrect online test bank")
        files[b]=sha256(dest.read_bytes()).hexdigest();checks[b]=check
    bank.lesson.storage._write_json(folder/"index.json",dict(births=ids,seed=seed,worlds=worlds,hashes=files,checkpoints=checks))
    print(cohort,family,count,flush=True)


def policies(step):
    out={};hashes={}
    for arm in fit.ARMS:
        out[arm]=[]
        for seed in fit.SEEDS:
            f=fit.OUT/str(seed)/arm/f"step-{step}.pt"
            obj=torch.load(f,map_location="cpu",weights_only=True)
            if obj["step"]!=step:raise ValueError("Wrong learner stage")
            model=initial.Readout("logged");model.load_state_dict(obj["weights"])
            out[arm].append(model);hashes[f.relative_to(bank.world.DATA).as_posix()]=sha256(f.read_bytes()).hexdigest()
    old=json.loads((bank.DATA/"readout.json").read_text())
    for k in ("logged","paired"):out["offline_"+k]=[initial.restored(old,k)]
    return out,hashes


def world_metrics(raw,prob):
    r=np.arange(len(prob));take=(raw["u"]<prob).astype(int)
    # A side shuffle across private valuations in the same geometric world keeps
    # the total number of physically executed left and right reviews exactly equal.
    donor=np.repeat(take.reshape(-1,8).mean(1),8)
    cues=raw["candidates"]
    good=raw["correct"]
    two=np.take_along_axis(good,np.broadcast_to(cues[:,None,:],(len(r),2,2)),axis=2).all(2)
    back={}
    for k,v in (("mean",good.mean(2)),("all",good.all(2)),("both",two),("pay",raw["paid"])):
        left=v[:,0].astype(float);dy=v[:,1].astype(float)-left
        picked=left+take*dy
        back[k]=picked
        back["minus_random/"+k]=(take-.5)*dy
        back["expected_minus_random/"+k]=(prob-.5)*dy
        back["own_minus_yoked/"+k]=(take-donor)*dy
    back["buy_right"]=take
    return {k:v.reshape(-1,8).mean(1) for k,v in back.items()}


def interval(table,ci,seed=182883):
    n,b,w=table.shape
    rng=np.random.default_rng(seed)
    fit_weights=rng.multinomial(n,np.full(n,1/n),len(ci.bw))/n
    samples=np.stack([((ci.br@v)*ci.bw).sum(1) for v in table],axis=1)
    return dict(mean=float(table.mean()),ci95=np.quantile((samples*fit_weights).sum(1),[.025,.975]).tolist(),
                by_training_seed=table.mean((1,2)).tolist(),by_body=table.mean((0,2)).tolist())


@torch.no_grad()
def run(cohort,family,count,step=4000):
    torch.set_num_threads(1);models,hashes=policies(step)
    src=fit.OUT/"heldout"/cohort/family/str(count);meta=json.loads((src/"index.json").read_text())
    acc={}
    t,first,_=bank.tape(meta["seed"],meta["worlds"],count)
    test_slice=torch.arange(8*4)
    for b in meta["births"]:
        f=src/f"{b}.npz"
        if sha256(f.read_bytes()).hexdigest()!=meta["hashes"][str(b)]:raise ValueError("Held-out motor trace mismatch")
        with np.load(f) as z:d={k:z[k] for k in ("u","z","menu","correct","paid","candidates")}
        z=torch.from_numpy(d["z"]).double();menu=torch.from_numpy(d["menu"]).double()
        weak=(z[:,:,0]-z[:,:,1]).abs().numpy()
        probas={a:[m(z,menu).numpy() for m in mm] for a,mm in models.items()}
        probas["native"]=[menu.sigmoid().numpy()]
        probas["weak"]=[np.where(weak[:,1]==weak[:,0],.5,(weak[:,1]<weak[:,0]).astype(float))]
        if family=="uniform":graph,rule=bank.world.load_parent(cohort,b)
        else:graph,rule=bank.old.load_checkpoint(bank.world.PARENT_DATA/"one-life-repair"/cohort,"odor","mixed",b,2000)
        tiny=bank.lesson.expand(t,test_slice);decision,signals=fit.observe(graph,rule,tiny,first[test_slice])
        torch.testing.assert_close(signals,torch.from_numpy(d["z"][test_slice]),rtol=0,atol=0)
        for a,pp in probas.items():
            rows=[]
            for i,prob in enumerate(pp):
                if a in fit.ARMS:
                    side=(torch.from_numpy(d["u"][test_slice]<prob[test_slice]).float()*2-1)[:,None]
                    realized=fit.execute(graph,rule,tiny,decision,side)[:,0].numpy()
                    assigned=(side.numpy().ravel()>0).astype(int)
                    np.testing.assert_array_equal(realized,d["paid"][test_slice.numpy(),assigned])
                rows.append(world_metrics(d,prob))
            for k in rows[0]:acc.setdefault(a,{}).setdefault(k,[]).append(np.stack([v[k] for v in rows]))
    series={a:{k:np.stack(v,1) for k,v in m.items()} for a,m in acc.items()}
    for a,b in (("live","forced_pair"),("sampled_pair","forced_pair"),("live","weak"),("forced_pair","weak"),
                ("live","native"),("live","offline_logged")):
        series[a+" minus "+b]={k:v-series[b][k] for k,v in series[a].items()}
    ci=Crossed(len(meta["births"]),meta["worlds"],meta["seed"]+count*11,3500)
    result=dict(cohort=cohort,family=family,count=count,step=step,worlds=meta["worlds"],seed=meta["seed"],
                raw=meta["hashes"],learners=hashes,training_seeds=list(fit.SEEDS),births=meta["births"],
                results={a:{k:interval(v,ci) for k,v in m.items()} for a,m in series.items()},
                notes="Sampling units: independent optimization streams, test bodies and shared test geometries. "
                      "All privately different histories in a scene remain paired. For all three learned policies "
                      "each fit seed and test body was checked against direct one-route execution "
                      "on the first four test geometries. A randomized selector has the mean of the two actual "
                      "continuation scores. No independent-cohort weights enter training of the small readout.")
    bank.lesson.storage._write_json(bank.world.EVIDENCE/f"online-{cohort}-{family}-{count}-{step}.json",result)
    print(cohort,family,count,step,flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--cohort",choices=bank.world.COHORTS,required=True)
    p.add_argument("--mode",choices=("collect","score"),required=True);p.add_argument("--step",type=int,default=4000)
    a=p.parse_args()
    for n in (4,2,8):
        for family in ("uniform","repair"):
            if a.mode=="collect":collect(a.cohort,family,n)
            else:run(a.cohort,family,n,a.step)
