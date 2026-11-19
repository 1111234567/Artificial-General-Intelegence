"""What from earlier life is necessary when fresh food may be tasted but no new relation lesson is available?"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import numpy as np

from . import live_continuity as stream, lifelong_agent as controller, longrun_world as longworld
from . import habitat_world as habits,mixture_world as tastes,symmetry_world as signed,world


from .recall_world import probe


def live(actor,simulation):
    e=habits.Body(simulation,silent_probes=True);obs=e.observe()
    while obs is not None:obs=e.step(actor.act(obs))
    return e.result()


def run(job):
    seeds,index=job;n=len(seeds);folder=stream.OUT/"retention-fork";folder.mkdir(parents=True,exist_ok=True)
    path=folder/f"cohort-{index}.json"
    if path.exists():return
    tags,rules,_=stream.private(seeds,800)
    w=longworld.make(seeds,blocks=64,mode="stable",private_rules=rules,public_habitats=tags,nonce=5182731)
    a=controller.new_lives(seeds);a.reset(n);stream.play(a,w)
    future_seeds=[int(np.random.SeedSequence([s,183137]).generate_state(1)[0]) for s in seeds]
    cases={}
    with tempfile.TemporaryDirectory(prefix="agi_fork_") as temp:
        snap=Path(temp)/"factual.npz";controller.save_life(snap,a)
        for mode in ("ongoing","restored","cold","only_higher_rule","only_episode_memory",
                     "clear_sensory","different_life_rule","opposite_hidden_B"):
            cand=a if mode=="ongoing" else controller.restore_life(snap)
            empty=controller.new_lives(seeds);empty.reset(n)
            if mode in ("cold","only_higher_rule","only_episode_memory"):
                keep=(("g","cov") if mode=="only_higher_rule" else ("mem","signed","count") if mode=="only_episode_memory" else ())
                for name in controller.STATE_ARRAYS:
                    if name=="noise" or name in keep:continue
                    old=getattr(empty,name)
                    if old is not None:setattr(cand,name,old.copy())
                # Keep the same future actuator noise, clock and request count
                # in all state ablations, including the cold functional state.
            if mode=="clear_sensory":cand.h.fill(0.)
            if mode=="different_life_rule":
                pair=np.arange(n)^1;cand.g[:]=cand.g[pair];cand.cov[:]=cand.cov[pair]
            new=probe(future_seeds,tags,rules,flip=mode=="opposite_hidden_B")
            out=live(cand,new)
            j=np.arange(1,new.menu.body.trials,2)
            side=out.contact_side[:,j];correct=(side>=0)&(side==new.truths[:,j]);cc=new.menu.body.ctx[:,j]
            ia=(cc==0).argmax(1);ib=(cc==1).argmax(1);ids=np.arange(n)
            cases[mode]=dict(first_A=correct[ids,ia].astype(float).tolist(),first_B=correct[ids,ib].astype(float).tolist(),
                    first_both=(correct[ids,ia]&correct[ids,ib]).astype(float).tolist(),all=correct.mean(1).tolist(),
                    coverage=(side>=0).mean(1).tolist(),choices=side.tolist(),physical_ticks=out.contact_tick.tolist())
            if mode=="restored":
                for k in ("choices","physical_ticks"):
                    if cases["ongoing"][k]!=cases["restored"][k]:raise AssertionError("Full state did not continue identically")
    stream.write(path,dict(roots=seeds,future_seeds=future_seeds,warm_requests=w.arena.menu.body.trials,
                    future_probes=8,new_pair_teaching=0,rows=cases))
    print("scored",index,flush=True)


def main():
    root=stream.OUT/"retention-fork";root.mkdir(parents=True,exist_ok=True)
    seeds=np.random.default_rng(306714511).integers(8_000_000,470_000_000,256).tolist()
    with ProcessPoolExecutor(max_workers=4) as pool:
        future=[pool.submit(run,(seeds[i:i+32],i//32)) for i in range(0,len(seeds),32)]
        for f in as_completed(future):f.result()
    stream.write(root/"complete.json",dict(roots=seeds))


if __name__=="__main__":main()
