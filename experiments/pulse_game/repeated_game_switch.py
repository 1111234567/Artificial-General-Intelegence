"""Repeatedly change game families in one life, with and without spaced actual old-rule receipts."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
import json
import numpy as np
import torch

from . import habitat_world as body,object_world as objects,recall_world,content_routing as adaptive
from . import adaptive_local,independent_births,unified_physical, research_audit as audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


def play(brain,arena,*,silent,blind=False):
    sim=body.Body(arena,silent_probes=silent,feedback="none" if blind else "true")
    obs=sim.observe()
    while obs is not None:obs=sim.step(brain.act(obs))
    return sim.result()


def isolate_teaching(w):
    stop=w.start_test
    tape=w.menu.body
    def trim(a):return a[:,:stop].copy()
    tape=replace(tape,ctx=trim(tape.ctx),cue=trim(tape.cue),left_id=trim(tape.left_id),good_id=trim(tape.good_id),
                 wind=trim(tape.wind),distances=trim(tape.distances))
    menu=replace(w.menu,body=tape,chemicals=trim(w.menu.chemicals),probability=trim(w.menu.probability),realized=trim(w.menu.realized))
    return replace(w,menu=menu,appearance=trim(w.appearance),probe=trim(w.probe),truths=trim(w.truths))


def one(job):
    seed,arm,show,roots,dest=job
    path=dest/f"{seed}-{arm}-{'review' if show else 'silent'}.json"
    if path.exists():return
    torch.set_num_threads(1)
    folder=OUT/f"independent-lifelong-births-v1/birth-{seed}"
    reg=json.loads((folder.parent/"registered.json").read_text())
    f,r,m=independent_births.resources(folder)
    fit=json.loads(next(folder.glob("*--full-*/frozen.json")).read_text())
    f,r,m=unified_physical.materialize(f,r,m,np.asarray(fit["best"]["genome"]),"full")
    r={**r,"release":reg["reopening_rule"]}
    cfg=adaptive.Settings(**json.loads((OUT/"object-content-v1/frozen.json").read_text())["params"])
    n=len(roots)
    if arm=="native":a=adaptive_local.LocalAdaptive(f,r,roots,motor=m)
    else:a=adaptive.Agent(f,r,roots,motion=m,config=replace(cfg,protect=arm!="read_only"))
    a.reset(n)
    original=body.make(roots,pairs=(12,12),span=4,noise=.04,visits=1,appearance="smooth",novel_seed=5471935)
    initial=play(a,original,silent=True)
    rows=[]
    for epoch in range(32):
        sub=[int(np.random.SeedSequence([s,epoch,391779]).generate_state(1)[0]) for s in roots]
        lengths=int(np.random.default_rng(3739+epoch).integers(4,10))
        game,l=objects.make(sub,problems=lengths,noise=.05,variable=True,nonce=731979)
        # Which previously seen habitat hosts the object discrimination game
        # varies independently of every private sign. No new task label appears.
        host=np.array([np.random.default_rng(np.random.SeedSequence([s,epoch,931])).integers(2) for s in roots])
        game.tags[:]=original.tags
        game.appearance[:]=original.tags[np.arange(n),host][:,None,:]
        o=objects.metrics(game,l,play(a,game,silent=False))
        recall=recall_world.probe([s+87 for s in sub],original.tags,original.relation,rounds=4)
        real=play(a,recall,silent=True);q=recall.probe
        hit=((real.contact_side==recall.truths)&(real.contact_side>=0))[q].reshape(n,-1)
        context=recall.menu.body.ctx[q].reshape(n,-1)
        rows.append(dict(epoch=epoch+1,object_trial2=o["trial_2"],old_rule=hit.mean(1).tolist(),
             just_used_habitat=hit[context==host[:,None]].reshape(n,-1).mean(1).tolist(),
             other_habitat=hit[context!=host[:,None]].reshape(n,-1).mean(1).tolist()))
        if (epoch+1)%4==0 and epoch+1<32:
            revision=body.make(sub,pairs=(4,4),span=4,visits=1,noise=.04,relations=original.relation,appearance="smooth",novel_seed=848731)
            revision.tags[:]=original.tags
            revision.appearance[:]=original.tags[np.arange(n)[:,None],revision.menu.body.ctx]
            # Same extra genuine physical episodes in both arms. Only the
            # informative old tastes are or are not sent to the controller.
            play(a,isolate_teaching(revision),silent=True,blind=not show)
    _write(path,dict(birth=seed,arm=arm,spaced_review=show,roots=roots,total_requests=a.round+1,physical_ticks=a.it,
            initial_both=body.metrics(original,initial)["both"].tolist(),rows=rows,sampled_reviews=7*16 if show else 0))
    print(seed,arm,show,a.round+1,round(np.mean([v["old_rule"] for v in rows[-4:]]),3),flush=True)


def main():
    root=OUT/"independent-lifelong-births-v1";reg=json.loads((root/"registered.json").read_text())
    dest=audit.OUT/"repeated-game-switches";dest.mkdir(exist_ok=True)
    seeds=np.random.default_rng(77481931).integers(6_000_000,461_000_000,96).tolist()
    births=reg["fixed_birth_seeds"][:3]
    jobs=[(b,arm,x,seeds,dest) for b in births for arm in ("native","selective","read_only") for x in (False,True)]
    with ProcessPoolExecutor(max_workers=5) as pool:
        for t in as_completed([pool.submit(one,j) for j in jobs]):t.result()
    _write(dest/"complete.json",dict(births=births,roots=seeds,switches=32))


if __name__=="__main__":main()
