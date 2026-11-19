"""Can one learner switch object relationships and retain the old rule with no flagged task ID?"""
from dataclasses import asdict,replace
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import numpy as np
import torch
from . import content_routing as routing,object_transfer as base,object_transfer_study as original,object_world
from . import habitat_world,independent_births as births,unified_physical as phy,lifetime_retention_fork as recall
from . import research_audit as audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


def one(job):
    seed,roots,nonce,config,which,length=job;torch.set_num_threads(1)
    folder=OUT/f"independent-lifelong-births-v1/birth-{seed}"
    reg=json.loads((folder.parent/"registered.json").read_text())
    f,r,m=births.resources(folder);fit=json.loads(next(folder.glob("*--full-*/frozen.json")).read_text())
    f,r,m=phy.materialize(f,r,m,np.array(fit["best"]["genome"]),"full");r={**r,"release":reg["reopening_rule"]}
    prime=habitat_world.make(roots,pairs=(12,12),visits=2,span=4,noise=.05,appearance="smooth",novel_seed=nonce)
    ow,ln=object_world.make(roots,problems=length,variable=True,noise=.04,nonce=nonce+1)
    ow.tags[:]=prime.tags;ow.appearance[:]=prime.tags[:,0,None,:]
    final=recall.probe([s+9539 for s in roots],prime.tags,prime.relation,rounds=8)
    if config is not None:a=routing.Agent(f,r,roots,motion=m,config=routing.Settings(**config))
    else:a=base.Agent(f,r,roots,motion=m,kind=which)
    a.reset(len(roots))
    obs=original.live(a,prime,True);before=habitat_world.metrics(prime,obs)
    real=original.live(a,ow,False);objects=object_world.metrics(ow,ln,real)
    last=original.live(a,final,True);q=final.probe;hits=(last.contact_side==final.truths)&(last.contact_side>=0)
    # The new object game uses the exact same A habitat sensation. Test-time
    # pulses, physical contacts and actual signed samples continue without resets.
    ctx=final.menu.body.ctx[q].reshape(len(roots),-1);hit=hits[q].reshape(len(roots),-1)
    h0=hit[ctx==0].reshape(len(roots),-1);h1=hit[ctx==1].reshape(len(roots),-1)
    return dict(initial_A=before["A"].tolist(),initial_B=before["B"].tolist(),initial_both=before["both"].tolist(),
              trial1=objects["trial_1"],trial2=objects["trial_2"],late_trial2=objects["late_second"],
              return_to_relations=hit.mean(1).tolist(),return_to_A=h0.mean(1).tolist(),return_to_B=h1.mean(1).tolist(),
              requests=a.round+1,visit_length=ln.tolist(),end_gate=a.weights_at_start[-1].tolist() if a.weights_at_start else [])


def main():
    torch.set_num_threads(1);folder=OUT/"object-content-v1";folder.mkdir(exist_ok=True)
    dest=audit.OUT/"object-content-sealed";dest.mkdir(exist_ok=True)
    reg=json.loads((OUT/"independent-lifelong-births-v1/registered.json").read_text());models=reg["fixed_birth_seeds"][:3]
    rng=np.random.default_rng(48917911)
    options=[routing.Settings(),routing.Settings(evidence=0.,protect=False)]
    options.extend(routing.Settings(evidence=float(rng.choice([2.,4.,7.,10.,14.])),prediction=float(rng.uniform(.2,2.5)),
                        global_rate=float(rng.uniform(.25,1.05)),prior_limit=float(rng.uniform(.8,3.8)),hold=float(rng.uniform(.72,.995)))
                   for _ in range(12))
    train=np.random.default_rng(65413991).integers(7_000_000,465_000_000,40).tolist()
    val=np.random.default_rng(81973351).integers(7_000_000,465_000_000,64).tolist()
    roots=np.random.default_rng(24795361).integers(7_000_000,465_000_000,160).tolist()
    metric=lambda v:.25*np.mean(v["initial_both"])+.35*np.mean(v["trial2"])+.40*np.mean(v["return_to_relations"])
    with ProcessPoolExecutor(max_workers=4) as pool:
        tag=folder/"frozen.json"
        if not tag.is_file():
            pending=[pool.submit(one,(models[0],train,4815179,asdict(o),None,20)) for o in options]
            results=[metric(x.result()) for x in pending]
            top=np.argsort(-np.asarray(results),kind="stable")[:4]
            pending=[pool.submit(one,(models[1],val,5798135,asdict(options[j]),None,24)) for j in top]
            score=[metric(x.result()) for x in pending];i=int(top[np.argmax(score)])
            _write(tag,dict(params=asdict(options[i]),candidates=[asdict(o) for o in options],train=results,
                              shortlist=top.tolist(),val=score,criterion="observed physical returns across all three stages"))
        chosen=routing.Settings(**json.loads(tag.read_text())["params"])
        arms={"both":chosen,"read_only":replace(chosen,protect=False),"write_only":replace(chosen,read=False),
              "switch_off":replace(chosen,read=False,protect=False),"native":None,"adaptive":None,"identity":None,"equal":None}
        for seed in models:
            file=dest/f"{seed}.json"
            if file.exists():continue
            jobs={a:pool.submit(one,(seed,roots,83649013,asdict(o) if o else None,a,40)) for a,o in arms.items()}
            result={a:fu.result() for a,fu in jobs.items()}
            _write(file,dict(seed=seed,roots=roots,weights=asdict(chosen),rows=result))
            print(seed,{k:(round(np.mean(d["trial2"]),3),round(np.mean(d["return_to_relations"]),3)) for k,d in result.items()},flush=True)
    _write(dest/"complete.json",dict(models=models,worlds=roots))


if __name__=="__main__":main()
