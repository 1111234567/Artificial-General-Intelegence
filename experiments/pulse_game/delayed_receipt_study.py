"""Split feedback availability from credit assignment on the unchanged physical choices."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import argparse
import json
import numpy as np
import torch
from . import delayed_receipts as delivery,independent_births as ancestry,unified_world as game,unified_physical as physical
from . import unified_audit as clinic,adaptive_local as local,habitat_world as environment,research_audit as audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


def run(job):
    birth,setting,mode,weights,seeds,nonce=job;torch.set_num_threads(1)
    root=OUT/"independent-lifelong-births-v1";alpha=json.loads((root/"registered.json").read_text())["reopening_rule"]
    folder=root/f"birth-{birth}";f,r,m=ancestry.resources(folder)
    fit=json.loads(next(folder.glob("*--full-*/frozen.json")).read_text())
    f,r,m=physical.materialize(f,r,m,np.asarray(fit["best"]["genome"]),"full")
    n=len(seeds);fam=np.repeat([[1,1],[1,-1],[-1,1],[-1,-1]],n,axis=0);ss=seeds*4
    e=game.make(ss,force=fam,nonce=nonce,**clinic.SETTINGS["standard"])
    a=local.LocalAdaptive(f,{**r,"release":alpha},ss,motor=m)
    ch=delivery.Channel(**setting)
    res,extra=delivery.run(e,a,ss,ch,mode,weights)
    if ch.lag==0 and ch.wait==0 and mode=="oracle":
        other=local.LocalAdaptive(f,{**r,"release":alpha},ss,motor=m)
        ref=environment.rollout(other,e.arena,silent_probes=False)
        if not (np.array_equal(res.contact_tick,ref.contact_tick) and np.array_equal(res.reward,ref.reward)):
            raise AssertionError("A zero-delay adapter changed the physical trace")
    goals=game.measures(e,res)
    report={k:v.reshape(4,n).tolist() for k,v in goals.items()}
    for k,v in extra.items():report[k]=v.reshape(4,n).tolist() if isinstance(v,np.ndarray) else v
    return report


def select(pool,seeds,roots,folder):
    rng=np.random.default_rng(17873)
    weights=np.r_[[[8.,1.,.10],[3.,1.,.4],[8.,1.,0.],[1.,0.,0.],[0.,1.,.2]],rng.uniform([.5,-.2,-.4],[12.,2.,1.1],size=(11,3))]
    small=roots[:24]
    setting=asdict(delivery.Channel(lag=40,ordered=False,echo_dims=2,echo_noise=.2,wait=52))
    jobs=[pool.submit(run,(seeds[0],setting,"match",w,small,3359713)) for w in weights]
    values=[]
    for job in jobs:
        m=job.result();values.append(.5*np.mean(m["both"])+.25*np.mean(m["B_after_12"])+.25*np.mean(m["return_A"]))
    win=int(np.argmax(values));f=dict(input="actual chosen food's positive probe outcomes",development=small,candidates=weights.tolist(),
                          scores=values,index=win,weights=weights[win].tolist())
    _write(folder/"selected.json",f);return f


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--workers",type=int,default=5);args=p.parse_args()
    base=OUT/"asynchronous-credit-v1";base.mkdir(exist_ok=True);dest=audit.OUT/"asynchronous-credit-sealed";dest.mkdir(exist_ok=True)
    reg=json.loads((OUT/"independent-lifelong-births-v1/registered.json").read_text());births=reg["fixed_birth_seeds"][:3]
    search=np.random.default_rng(48930971).integers(7_000_000,475_000_000,48).tolist()
    roots=np.random.default_rng(57231911).integers(7_000_000,475_000_000,112).tolist()
    settings={
      "immediate":delivery.Channel(lag=0,wait=0),
      "short":delivery.Channel(lag=6,wait=0),
      "ordered_wait":delivery.Channel(lag=40,ordered=True,wait=52),
      "reordered_wait":delivery.Channel(lag=40,ordered=False,wait=52),
      "reordered_no_wait":delivery.Channel(lag=40,ordered=False,wait=0),
      "degraded_echo":delivery.Channel(lag=40,ordered=False,echo_dims=1,echo_noise=.38,wait=52),
      "no_echo":delivery.Channel(lag=40,ordered=False,echo_dims=0,wait=52)}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        data=json.loads((base/"selected.json").read_text()) if (base/"selected.json").is_file() else select(pool,births,search,base)
        for birth in births:
            for label,setting in settings.items():
                path=dest/f"{birth}-{label}.json"
                if path.exists():continue
                arms=["time_local","fifo","fifo_side","match","oracle"]
                scores={k:pool.submit(run,(birth,asdict(setting),k,data["weights"],roots,34761973)) for k in arms}
                rows={k:f.result() for k,f in scores.items()}
                _write(path,dict(birth=birth,condition=label,settings=asdict(setting),weights=data["weights"],seeds=roots,rows=rows))
                print(birth,label,{k:(round(float(np.mean(v["correct_binding"])),3),round(float(np.asarray(v["return_A"])[[1,2]].mean()),3))
                       for k,v in rows.items()},flush=True)
    _write(dest/"complete.json",dict(births=births,roots=len(roots)))


if __name__=="__main__":main()
