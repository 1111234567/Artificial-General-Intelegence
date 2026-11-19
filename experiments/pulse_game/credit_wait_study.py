"""Do confidence in a predicted taste or a second actual receipt improve the decision to write?"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict,replace
import argparse
import json
import numpy as np
from . import credit_wait as local,delayed_receipts as environment,delayed_receipt_study as compare,research_audit as audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


def measure(result):
    return .5*np.mean(result["both"])+.25*np.mean(result["B_after_12"])+.25*np.mean(result["return_A"])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--workers",type=int,default=5);args=p.parse_args()
    base=OUT/"deliberate-credit-v1";base.mkdir(exist_ok=True);dest=audit.OUT/"deliberate-credit-sealed";dest.mkdir(exist_ok=True)
    reg=json.loads((OUT/"independent-lifelong-births-v1/registered.json").read_text());births=reg["fixed_birth_seeds"][:3]
    selected=json.loads((OUT/"asynchronous-credit-v1/selected.json").read_text())["weights"]
    rng=np.random.default_rng(40937621)
    candidates=[local.Rule(wait=0,margin=0,prediction=0.),local.Rule(),local.Rule(wait=16,margin=1.5,prediction=.8)]
    for _ in range(21):
        candidates.append(local.Rule(wait=int(rng.choice([2,6,10,16])),margin=float(rng.uniform(.15,1.45)),
                 prediction=float(rng.uniform(-.8,2.4)),echo_gain=float(rng.uniform(1.,9.)),prior_gap=float(rng.uniform(0,.85))))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        path=base/"selected.json"
        if not path.exists():
            train=np.random.default_rng(67199415).integers(7_000_000,470_000_000,28).tolist()
            results=[]
            for c in (environment.Channel(lag=40,wait=60,echo_dims=2,echo_noise=.17),
                      environment.Channel(lag=40,wait=60,echo_dims=1,echo_noise=.35)):
                fut=[pool.submit(compare.run,(births[0],asdict(c),o,None,train,39173163)) for o in candidates]
                results.append([measure(f.result()) for f in fut])
            idx=np.argsort(-np.asarray(results).mean(0),kind="stable")[:5]
            val=np.random.default_rng(11276819).integers(7_000_000,470_000_000,48).tolist()
            fut=[pool.submit(compare.run,(births[1],asdict(environment.Channel(lag=40,wait=60,echo_dims=2,
                    echo_noise=.22)),candidates[k],None,val,57976113)) for k in idx]
            vv=[measure(f.result()) for f in fut];winner=int(idx[np.argmax(vv)])
            _write(path,dict(criterion="actual outcomes at the free physical choices only",
                    candidates=[asdict(v) for v in candidates],train=results,shortlist=idx.tolist(),validation=vv,
                    winner=asdict(candidates[winner])))
        best=local.Rule(**json.loads(path.read_text())["winner"])
        root=np.random.default_rng(39751681).integers(7_000_000,470_000_000,112).tolist()
        protocols={"two_coordinates":environment.Channel(lag=40,wait=60,echo_dims=2,echo_noise=.17),
                   "one_noisy_coordinate":environment.Channel(lag=40,wait=60,echo_dims=1,echo_noise=.36),
                   "no_coordinates":environment.Channel(lag=40,wait=60,echo_dims=0),
                   "no_extra_time":environment.Channel(lag=40,wait=0,echo_dims=2,echo_noise=.17)}
        for birth in births:
            for scenario,ch in protocols.items():
                file=dest/f"{birth}-{scenario}.json"
                if file.exists():continue
                arms={"trained":best,"immediate_same_rule":replace(best,wait=0),
                      "no_old_prediction":replace(best,prediction=0.),"no_joint_assignment":replace(best,joint=False),
                      "simple_immediate":local.Rule(wait=0,margin=0.,prediction=0.),
                      "old_match":"match","oracle":"oracle"}
                work={k:pool.submit(compare.run,(birth,asdict(ch),v,selected,root,1246579)) for k,v in arms.items()}
                obs={k:f.result() for k,f in work.items()}
                _write(file,dict(birth=birth,scenario=scenario,seeds=root,settings=asdict(ch),rule=asdict(best),rows=obs))
                print(birth,scenario,{k:round(np.mean(v["correct_binding"]),3) for k,v in obs.items()},flush=True)
    _write(dest/"complete.json",dict(births=births,worlds=len(root)))


if __name__=="__main__":main()
