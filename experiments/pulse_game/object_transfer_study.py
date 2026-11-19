"""Independent learning-set transfer: two new objects, shuffled sides, one privately rewarding."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import numpy as np
import torch
from . import object_world as objects,object_transfer as policies,independent_births as births
from . import unified_physical,habitat_world,unified_world as rules,lifetime_retention_fork as replay
from . import research_audit as audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


def live(actor,w,silent):
    sim=habitat_world.Body(w,silent_probes=silent);obs=sim.observe()
    while obs is not None:obs=sim.step(actor.act(obs))
    return sim.result()


def test(job):
    seed,regime,ids,dest,alpha=job
    path=dest/f"{seed}-{regime}.json"
    if path.exists():return
    torch.set_num_threads(1)
    folder=OUT/f"independent-lifelong-births-v1/birth-{seed}";f,r,m=births.resources(folder)
    fit=json.loads(next(folder.glob("*--full-*/frozen.json")).read_text())
    front,route,motion=unified_physical.materialize(f,r,m,np.asarray(fit["best"]["genome"]),"full")
    route={**route,"release":alpha}
    silence=regime=="old_rule";n=len(ids)
    if silence:
        family=np.repeat([[1,1],[1,-1],[-1,1],[-1,-1]],n,axis=0);ids=ids*4
        prior=rules.make(ids,force=family,nonce=9875131)
        w=prior.arena
    else:w,length=objects.make(ids,problems=40,noise=.12 if regime=="noisy" else 0.,
                            variable=regime=="variable_duration",nonce=5149901)
    arms={}
    for kind in ("native","identity","equal","adaptive","reverse_gate"):
        a=policies.Agent(front,route,ids,motion=motion,kind=kind);a.reset(len(ids))
        out=live(a,w,silence)
        if silence:
            result={k:v.reshape(4,n).tolist() for k,v in rules.measures(prior,out).items()}
        else:
            result=objects.metrics(w,length,out)
        if a.weights_at_start:
            gate=np.stack(a.weights_at_start,1)
            result["identity_weight_early"]=gate[:,:6].mean(1).tolist()
            result["identity_weight_late"]=gate[:,-12:].mean(1).tolist()
        arms[kind]=result
    _write(path,dict(birth=seed,regime=regime,roots=ids[:n],rows=arms,hold=.96,rate=.9))
    print(seed,regime,flush=True)


def chain(job):
    seed,ss,dest,alpha=job;torch.set_num_threads(1)
    path=dest/f"{seed}-cross_game.json"
    if path.exists():return
    folder=OUT/f"independent-lifelong-births-v1/birth-{seed}";f,r,m=births.resources(folder)
    fit=json.loads(next(folder.glob("*--full-*/frozen.json")).read_text())
    f,r,m=unified_physical.materialize(f,r,m,np.asarray(fit["best"]["genome"]),"full")
    r={**r,"release":alpha}
    prime=habitat_world.make(ss,pairs=(12,12),span=4,visits=2,noise=.05,appearance="smooth",novel_seed=595651)
    ow,length=objects.make(ss,problems=32,noise=0.,nonce=119353)
    ow.tags[:]=prime.tags;ow.appearance[:]=prime.tags[:,0,None,:]
    target=replay.probe([s+29511 for s in ss],prime.tags,prime.relation,rounds=8)
    rows={}
    for kind in ("native","identity","equal","adaptive","reverse_gate"):
        a=policies.Agent(f,r,ss,motion=m,kind=kind);a.reset(len(ss))
        initial=live(a,prime,True)
        first=habitat_world.metrics(prime,initial)
        object_out=live(a,ow,False);second=objects.metrics(ow,length,object_out)
        final=live(a,target,True);q=target.probe
        correct=((final.contact_side==target.truths)&(final.contact_side>=0))[q].reshape(len(ss),-1)
        rows[kind]=dict(initial_A=first["A"].tolist(),initial_B=first["B"].tolist(),object_second=second["trial_2"],
                        object_late_second=second["late_second"],back_to_rules=correct.mean(1).tolist(),
                        total_requests=a.round+1)
        if a.weights_at_start:
            g=np.stack(a.weights_at_start,1)
            s0=prime.menu.body.trials;s1=s0+ow.menu.body.trials
            rows[kind].update(gate_end_rules=g[:,s0-6:s0].mean(1).tolist(),
                 gate_end_objects=g[:,s1-6:s1].mean(1).tolist(),gate_after_return=g[:,s1:].mean(1).tolist())
    _write(path,dict(birth=seed,roots=ss,experiment="single continuously carried brain across all three scenes",rows=rows))
    print(seed,"cross_game",flush=True)


def main():
    folder=audit.OUT/"object-learning-set";folder.mkdir(exist_ok=True)
    root=OUT/"independent-lifelong-births-v1";reg=json.loads((root/"registered.json").read_text());alpha=reg["reopening_rule"]
    seeds=np.random.default_rng(25364119).integers(6_000_000,445_000_000,128).tolist()
    selected=reg["fixed_birth_seeds"][:3]
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(test,(s,c,seeds,folder,alpha)) for s in selected for c in ("new_objects","variable_duration","noisy","old_rule")]
        jobs+=[pool.submit(chain,(s,seeds,folder,alpha)) for s in selected]
        for j in as_completed(jobs):j.result()
    _write(folder/"complete.json",dict(births=selected,roots=seeds))


if __name__=="__main__":main()
