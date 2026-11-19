"""Challenge a fixed learner with untrained mappings, local reversals and brief cues."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from experiments.seed_evolution.large_runner import _write
from . import research_audit as audit, world
from .sensation import BriefCue


def change_rule(tape, style):
    expected=tape.good_id.copy()
    rngs=[np.random.default_rng(np.random.SeedSequence([s,6137])) for s in tape.seeds]
    focal_cue=np.array([rng.integers(4) for rng in rngs],dtype=np.int8)
    focal_context=np.array([rng.integers(2) for rng in rngs],dtype=np.int8)
    changed=tape.cue==focal_cue[:,None]
    if style=="one_context":
        changed &= tape.ctx==focal_context[:,None]
    elif style!="both_contexts":
        raise ValueError("invalid reversal")
    changed[:,:tape.trials//2]=False
    expected[changed]=1-expected[changed]
    return replace(tape,good_id=expected),changed


def independent_rules(tape):
    matrix = tape.good_map.copy()
    matrix[:,1]=np.stack([np.random.default_rng(np.random.SeedSequence([s,8067])).integers(2,size=4)
                          for s in tape.seeds])
    return replace(tape,good_map=matrix,good_id=matrix[np.arange(tape.n)[:,None],tape.ctx,tape.cue])


def rate(rows):
    num=sum(n for n,_ in rows);den=sum(d for _,d in rows)
    if den==0:
        return {"correct":0,"opportunities":0,"pooled":None}
    pairs=np.array(rows,dtype=np.int64)
    draw=np.random.default_rng(317).integers(len(pairs),size=(3000,len(pairs)))
    subs=pairs[draw].sum(1)
    ci=np.quantile(subs[subs[:,1]>0,0]/subs[subs[:,1]>0,1],[.025,.975]).tolist()
    return {"correct":int(num),"opportunities":int(den),"pooled":num/den,"ci95":ci}


def probe_reversal(tape, factory, normal):
    output={}
    for mode in ("both_contexts","one_context"):
        changed,mask=change_rule(tape,mode)
        outcome=world.rollout(factory(),changed)
        half=tape.trials//2
        if not np.array_equal(outcome.contact_id[:,:half],normal.contact_id[:,:half]):
            raise ValueError("Matched worlds diverge before reversal")
        ordinal=np.cumsum(mask,axis=1)
        after=np.arange(tape.trials)[None,:]>=half
        scopes={"first_changed":mask & (ordinal==1),"second_changed":mask & (ordinal==2),
                "third_or_later":mask & (ordinal>=3),"unaffected":after & ~mask}
        items={}
        for name,valid in scopes.items():
            rows=[(int((v & h).sum()),int(v.sum())) for v,h in zip(valid,outcome.correct,strict=True)]
            other=[(int((v & h).sum()),int(v.sum())) for v,h in zip(valid,normal.correct,strict=True)]
            items[name]={"reversed":rate(rows),"stationary":rate(other)}
        output[mode]=items
    return output


def run(tape,factory):
    result=world.rollout(factory(),tape)
    independent=independent_rules(tape)
    second=world.rollout(factory(),independent)
    inv=independent.good_map[:,0]==independent.good_map[:,1]
    same=inv[np.arange(tape.n)[:,None],tape.cue]
    late=np.arange(tape.trials)[None,:]>=tape.trials//2
    report={"true":audit.life_rows(tape,result),"independent":audit.life_rows(independent,second)}
    masks={"independent_same_target":same&late,"independent_opposite_target":~same&late}
    conditional={n:rate([(int((v&hit).sum()),int(v.sum())) for v,hit in zip(mask,second.correct,strict=True)])
                 for n,mask in masks.items()}
    reversals=probe_reversal(tape,factory,result)
    for k in (0,1,2,4):
        report[f"cue_ticks_{k}"]=audit.life_rows(tape,world.rollout(BriefCue(factory(),k),tape))
    moved=world.swap_sides(world.make_tape(tape.seeds,trials=tape.trials,distance_jitter=.22,wind_scale=.12),
                           first_trial=tape.trials//2)
    report["move"]=audit.life_rows(moved,world.rollout(factory(),moved))
    return report,conditional,reversals


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject",choices=("motor","meta","spike"),required=True)
    p.add_argument("--frozen",type=Path,required=True)
    p.add_argument("--hash",required=True)
    p.add_argument("--name",required=True)
    p.add_argument("--nets",nargs="+",type=int,default=[3,19,37,53])
    p.add_argument("--split",default="transfer")
    p.add_argument("--worlds",type=int,default=256)
    p.add_argument("--with-controls",action="store_true")
    args=p.parse_args()
    if audit.fingerprint(args.frozen)!=args.hash:
        raise ValueError("Frozen configuration hash mismatch")
    seeds=audit.split(args.split)[:args.worlds]
    tape=world.make_tape(seeds)
    if args.subject=="motor":
        from . import motor_lif
        frozen=json.loads(args.frozen.read_text(encoding="utf-8"))
        if frozen["source_sha256"]!=audit.fingerprint(motor_lif.__file__):
            raise ValueError("Selected motor implementation changed")
        cfg=motor_lif.MotorConfig(**frozen["selection"]["config"])
    elif args.subject=="meta":
        from . import meta_model
        import torch
        torch.set_num_threads(1)
        m=meta_model.load_policy(args.frozen,"cpu")
    else:
        from . import spike_policy as spike
        cfg=spike.SpikeConfig(use_residual=True)
    overall={"frozen":str(args.frozen),"sha256":args.hash,"split":args.split,"seeds":seeds,"models":{}}
    for net in ([0] if args.subject=="meta" else args.nets):
        if args.subject=="motor":
            def factory(): return motor_lif.MotorLife(cfg,net,seeds)
        elif args.subject=="meta":
            def factory(): return meta_model.RolloutAgent(m)
        else:
            def factory(): return spike.BatchSpikingPolicy(cfg,network_seed=net,action_seeds=[
                np.random.SeedSequence([s,0x4C49464C]) for s in seeds])
        res,cond,rev=run(tape,factory)
        if args.with_controls:
            res.update(audit.normal_and_controls(factory,tape))
            if args.subject=="motor":
                for label,pol in (("no_write",motor_lif.MotorLife(replace(cfg,learning_rate=0.),net,seeds)),
                                  ("reservoir_cut",motor_lif.MotorLife(cfg,net,seeds,mute_recurrence=True)),
                                  ("stop_writing_after_32",motor_lif.MotorLife(cfg,net,seeds,freeze_after=32))):
                    res[label]=audit.life_rows(tape,world.rollout(pol,tape))
        report,contrasts=audit.overview(res)
        overall["models"][str(net)]={"summary":report,"true_minus":contrasts,"by_rule":cond,
                                     "reversal":rev,"per_world":res}
        _write(audit.OUT/(args.name+".json"),overall)
        print(args.name,net,{k:round(v["late_accuracy"]["mean"],4) for k,v in report.items()},flush=True)


if __name__=="__main__":
    main()
