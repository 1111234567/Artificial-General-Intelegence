"""Identical held-out Scout courses for independently developed controllers and body delays."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from experiments.seed_evolution.large_runner import _write
from . import research_audit as audit,scout_world as game,world
from . import meta_scout_model,scout_neural,scout_neural_train


class Delay:
    """A real actuator lag. Queue clears only when the world respawns the body."""
    def __init__(self,agent,lag):
        self.agent,self.lag=agent,lag
    def reset(self,n):
        self.agent.reset(n);self.queue=[np.zeros((n,2),bool) for _ in range(self.lag)]
    def act(self,obs):
        if np.all(obs[:,world.TRIAL_START]):
            for item in self.queue:item.fill(False)
        action=self.agent.act(obs)
        if not self.lag:return action
        out=self.queue.pop(0);self.queue.append(action.copy())
        return out


def measures(t,r):
    correct_side=t.left_id!=t.good_id
    opposite=correct_side!=t.scout_side
    must=t.sensitive[np.arange(t.n)[:,None],t.cue]&opposite
    late=np.arange(t.trials)[None,:]>=t.trials-16
    selected=must&late
    den=selected.sum(1)
    return dict(overall=r.correct.mean(1).tolist(),early=r.correct[:,:8].mean(1).tolist(),
                late=r.correct[:,-16:].mean(1).tolist(),coverage=(r.contact_id>=0).mean(1).tolist(),
                must_n=(r.correct&r.reversed&selected).sum(1).tolist(),must_d=den.tolist(),
                mean_duration=np.where(r.contact_tick<0,world.TICKS,r.contact_tick+1).mean(1).tolist())


def endpoint(v):
    s={k:audit.estimates(a,clusters=2) for k,a in v.items() if k not in ("must_n","must_d")}
    n,d=(np.array(v[k]).reshape(-1,2).sum(1) for k in ("must_n","must_d"))
    ids=np.random.default_rng(225).integers(len(n),size=(4096,len(n)))
    s["must_reverse_late"]={"mean":float(n.sum()/d.sum()),"ci95":np.quantile(n[ids].sum(1)/d[ids].sum(1),[.025,.975]).tolist(),
                             "correct":int(n.sum()),"opportunities":int(d.sum())}
    return s


def evaluate(label,factory,tape):
    data={}
    def run(condition,agent,**kw):
        res=game.rollout_scout(agent,tape,**kw)
        data[condition]=measures(tape,res)
        return res
    true=run("true",factory())
    if label not in ("oracle","public_history"):
        run("fast_write_off",factory(write=False))
        run("recurrence_cut",factory(recurrent=False))
        run("outcome_off",factory(),feedback="none")
        idx=np.arange(tape.n)^1
        run("outcome_yoke",factory(),feedback="yoked",yoke=true.observed_reward[idx])
        run("probe_yoke",factory(),sensor="replay",replay=true.observed_context[idx])
    run("probe_off",factory(),sensor="none")
    for lag in (1,2):
        run(f"motor_delay_{lag}",Delay(factory(),lag))
    return {"per_world":data,"summary":{k:endpoint(v) for k,v in data.items()}}


def main():
    torch.set_num_threads(1)
    parent=audit.ROOT/"data/pulse-game"
    files={"reward_structured":parent/"scout-20261119/models/pure-parity-final.pt",
           "raw_imitation":parent/"meta-scout-even-2313/frozen-scout-even-520.pt",
           "raw_self_targets":parent/"meta-scout-self-even-2338/frozen-self-even-early.pt"}
    # All model files were announced frozen before constructing these worlds.
    digest={k:audit.fingerprint(v) for k,v in files.items()}
    seeds=[];base=tuple(audit.split("hard_task"))
    for stream in range(20):
        possible=[int(np.random.SeedSequence([v,0xA5C017,stream]).generate_state(1)[0]) for v in base]
        t=game.make_scout_tape(possible,trials=1)
        seeds.extend(np.array(possible,dtype=np.int64)[t.good_map[:,0].sum(1)%2==1].tolist())
        if len(seeds)>=256:break
    seeds=seeds[:256]
    tape=game.make_scout_tape(seeds,trials=64,distance_jitter=.22,wind_scale=.12)
    tape=game.ScoutTape(**vars(world.swap_sides(tape,first_trial=32)))
    results={"seeds":seeds,"odd_a_maps":True,"frozen":{k:[str(v),digest[k]] for k,v in files.items()},
             "baseline_trials":tape.trials,"models":{}}
    for label in ("oracle","public_history",*files):
        if label=="oracle":
            factory=lambda:game.ScoutOraclePolicy(tape)
        elif label=="public_history":
            factory=lambda:game.ScoutHistoryPolicy(seed=307445)
        elif label=="reward_structured":
            model,_=scout_neural_train.load(files[label])
            class Structured(scout_neural.ScoutNetPolicy):
                def __init__(self,write,recurrent):
                    super().__init__(model,write=write,seed=307445);self.used_edges=recurrent
                def reset(self,n):
                    super().reset(n)
                    if not self.used_edges:self.weights=torch.zeros_like(self.weights)
            factory=lambda write=True,recurrent=True:Structured(write,recurrent)
        else:
            model=meta_scout_model.load_scout_model(files[label],"cpu")
            factory=lambda write=True,recurrent=True:meta_scout_model.ScoutRawAgent(model,write_enabled=write,recurrence_enabled=recurrent)
        result=evaluate(label,factory,tape)
        results["models"][label]=result
        _write(audit.OUT/"scout-common-worlds.json",results)
        print(label,{k:round(v["overall"]["mean"],4) for k,v in result["summary"].items()},flush=True)


if __name__=="__main__":
    main()
