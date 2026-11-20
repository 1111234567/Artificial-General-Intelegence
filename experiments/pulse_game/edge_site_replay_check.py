"""Test an old-blind strategy calibrated to equal new-item accuracy on separate worlds."""

import argparse
import json

import numpy as np
import torch

from . import edge_site_replay as ref
from . import noon_market_stats as stats


SEED=258391
WORLDS=256
OUT=ref.DATA.parent/"matched-b"
MODES=("replay","mixed","random","b_only","native","four","eight")
DIFFS=[("replay","mixed"),("replay","native"),("four","random"),("eight","random"),("eight","native")]
CAL=ref.EVIDENCE/"replay-results.json"
PLAN=ref.EVIDENCE/"matched-b-plan.md"


def calibration():
    original=json.loads(CAL.read_text(encoding="utf8"))["original"]["scores"]
    rates=[original[p]["primary/4/B"]["mean"] for p in ("replay","b_only","random")]
    target,b_only,random=rates
    if not random<target<b_only:
        raise ValueError("A common B-performance mixture does not exist")
    return (target-random)/(b_only-random)


def event_fields(worlds,angle,groups):
    rng=np.random.default_rng([SEED,angle,5609])
    control=np.random.default_rng([SEED,angle,8731])
    def common(x):
        return torch.from_numpy(np.repeat(x,4,axis=0))
    options=[common(rng.integers(4,size=(worlds,groups))) for _ in range(8)]
    coin=common(control.random(worlds))<calibration()
    orient=common(control.choice(np.array([-1.,1.],np.float32),(worlds,2)))
    first=common(control.integers(2,size=worlds))
    draw=common(control.random((worlds,2),dtype=np.float32))
    return options,coin,(orient,first,draw)


def few_options(prefix,receipts,options):
    pred=torch.stack([ref.expected_scores(ref.sites.apply(prefix,index),receipts) for index in options],dim=1)
    objective=pred[:,:,0]*pred[:,:,1]
    objective=objective.masked_fill(pred[:,:,1]<pred[:,0:1,1]-.01,-torch.inf)
    indices=torch.stack(options,dim=1)
    lane=torch.arange(len(indices))
    return {key:indices[lane,objective[:,:n].argmax(1)] for key,n in (("four",4),("eight",8))}


@torch.no_grad()
def run(birth):
    torch.set_num_threads(1)
    g,r=ref.sites.load_parent(birth)
    panels={}
    for angle in (15,30):
        world=ref.interference.worlds(SEED,0,WORLDS,angle)
        prefix=ref.sites.prepare(g,r,world)
        options,coin,final=event_fields(WORLDS,angle,prefix.candidates.address.shape[1])
        receipts=((world.a*world.orders[:,0,None],prefix.action_a.side*prefix.experience_a.taste),
                  (world.b*world.orders[:,1,None],prefix.action_b.side*prefix.experience_b.taste))
        picks=ref.choose(prefix,receipts,options[0])
        picks["mixed"]=torch.where(coin[:,None],picks["b_only"],picks["random"])
        picks.update(few_options(prefix,receipts,options))
        target,first=ref.test_world(world,final)
        truth=torch.stack((world.law_a,world.law_b),1)*final[0]
        panels[f"{angle}/truth"]=truth.to(torch.int8).numpy()
        panels[f"{angle}/agree"]=(world.law_a==world.law_b).numpy()
        panels[f"{angle}/mixed_coin"]=coin.numpy()
        for mode in MODES:
            state=ref.sites.apply(prefix,mode="native") if mode=="native" else ref.sites.apply(prefix,picks[mode])
            actual=ref.sites.probe(prefix.rule,state,target,wait=4,order=first)
            if not torch.equal(actual.correct,actual.side==truth):raise ValueError("Incorrect actual score")
            panels[f"{angle}/{mode}"]=actual.side.to(torch.int8).numpy()
            net=(prefix.native_b.state.graph.weights-state.graph.weights).flatten(1)
            panels[f"{angle}/{mode}/dose"]=torch.stack(
                (net.clamp_min(0).sum(1),(-net).clamp_min(0).sum(1),net.ne(0).sum(1)),1).numpy()
    ref.storage._write_raw(OUT/f"{birth}.npz",**panels,seed=SEED,calibration=ref.digest(CAL),plan=ref.digest(PLAN),
                           source=ref.digest(__file__),search=ref.digest(ref.__file__),probability=calibration())


def summarize():
    result={"mixture_probability":calibration(),"calibration_sha256":ref.digest(CAL),"worlds":WORLDS,
            "seed":SEED,"plan_sha256":ref.digest(PLAN),"cohorts":{}}
    for cohort in ("original","independent"):
        births=ref.parents.births(cohort);data={p:{} for p in MODES};hashes={}
        maxdose=0.
        for birth in births:
            path=OUT/f"{birth}.npz";hashes[str(birth)]=ref.digest(path)
            with np.load(path,allow_pickle=False) as z:
                if (str(z["plan"])!=ref.digest(PLAN) or str(z["source"])!=ref.digest(__file__)
                    or str(z["calibration"])!=ref.digest(CAL) or float(z["probability"])!=calibration()):
                    raise ValueError(f"Mismatched inputs at {birth}")
                for mode in MODES:
                    pieces={}
                    for angle in (15,30):
                        choice=z[f"{angle}/{mode}"];truth=z[f"{angle}/truth"]
                        if mode!="native":
                            dose=z[f"{angle}/{mode}/dose"];rand=z[f"{angle}/random/dose"]
                            np.testing.assert_array_equal(dose[:,2],rand[:,2])
                            maxdose=max(maxdose,float(np.max(np.abs(dose[:,:2]-rand[:,:2]))))
                        if mode=="mixed":
                            donor=np.where(z[f"{angle}/mixed_coin"][:,None],z[f"{angle}/b_only"],z[f"{angle}/random"])
                            np.testing.assert_array_equal(choice,donor)
                        for label,mask in (("conflict",~z[f"{angle}/agree"]),("agree",z[f"{angle}/agree"])):
                            a,b=(choice[mask]==truth[mask]).T
                            for name,observed in (("A",a),("B",b),("both",a&b)):
                                pieces.setdefault(f"{label}/{name}",[]).append(observed.reshape(WORLDS,2).mean(1))
                    for key,by_angle in pieces.items():
                        data[mode].setdefault(key,[]).append(np.mean(by_angle,axis=0))
        rows={p:{k:np.stack(v) for k,v in series.items()} for p,series in data.items()}
        scores=stats.summarize(rows,WORLDS,births,SEED+811,DIFFS)
        b=scores["replay minus mixed"]["conflict/B"]["ci95"]
        criteria=dict(B_equivalent=b[0]>-.05 and b[1]<.05,
                      A_improves=scores["replay minus mixed"]["conflict/A"]["ci95"][0]>0,
                      both_improve=scores["replay minus mixed"]["conflict/both"]["ci95"][0]>0)
        result["cohorts"][cohort]=dict(scores=scores,births=births,hashes=hashes,max_dose_error=maxdose,criteria=criteria)
    ref.storage._write_json(ref.EVIDENCE/"matched-b-results.json",result)
    print("B-only mixture probability",calibration())
    for cohort,group in result["cohorts"].items():
        print(cohort,group["criteria"])
        for mode in ("replay minus mixed","four minus random","eight minus random","eight minus native"):
            print(mode,[(k,round(100*v["mean"],2),np.round(np.array(v["ci95"])*100,2).tolist())
                        for k,v in group["scores"][mode].items() if k.startswith("conflict")])


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--birth",type=int)
    args=ap.parse_args()
    if args.birth is not None:run(args.birth)
    else:summarize()
