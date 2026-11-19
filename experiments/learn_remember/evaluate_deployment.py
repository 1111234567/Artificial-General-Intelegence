"""Separately declared deployment-history sensitivity checks, never primary scores."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from experiments.learn_remember import evaluate_gru as e
from experiments.learn_remember.gru_train import GRUAgent, load_checkpoint
from experiments.lifelong.world import ContextWorld
from experiments.lifelong.evaluate import CONDITIONS

OUT = e.ROOT / 'data/learn-remember/evaluation-deployment-v1'
PLAN = OUT / 'protocol.json'
ARCHES = ('plain', 'symmetric')
STAGES = ('wide', 'withdrawal', 'review')


def folder(arch, seed):
    return e.ROOT / 'data/learn-remember' / ('gru-v1' if arch == 'plain' else 'symmetric-v1') / f'seed_{seed:03d}'


def prepare():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    sources = [Path(__file__), Path(e.__file__), e.ROOT/'experiments/learn_remember/evaluate_symmetric.py',
               e.ROOT/'experiments/learn_remember/symmetric_gru.py',
               e.ROOT/'experiments/learn_remember/gru_train.py', e.ROOT/'experiments/lifelong/world.py',
               e.ROOT/'experiments/continuity/world.py']
    inputs = {str(folder(a,s)/(stage+'.pt')): e.sha(folder(a,s)/(stage+'.pt'))
              for a in ARCHES for s in range(3) for stage in STAGES}
    plan = dict(format='learn-remember-deployment-history-supplement-v1',
        created_utc=datetime.now(timezone.utc).isoformat(),
        declaration='Declared after original WAIT-history outcomes revealed a symmetric-policy deployment anomaly, before reading any own-warm or native-endpoint supplemental outcomes. Exploratory followup; does not replace the frozen primary comparison.',
        primary_protocol_hash=e.checked()['protocol_hash'], sources={str(p):e.sha(p) for p in sources}, inputs=inputs,
        architectures=list(ARCHES), seeds=[0,1,2], stages=list(STAGES), family=3, condition='long_both', steps=2048,
        tests=['Original WAIT common history reused read-only.',
               'own_warm: identical heldout birth world and1024 identical external scheduling switches; execute frozen model11-way argmax to generate its own receipts/h. No weight update or teacher. Compare action-independent event/scheduler/context hashes to original WAIT history.',
               'endpoint: full saved64-lane training world and hidden state, then native free2048-tick continuation. Report all64 actors. These actors/requests are not heldout and different endpoint ages are not matched cases.'],
        model_checks='Before supplemental outcomes bind checkpoint hashes, verify every loaded tensor equals its serialized source, and for symmetric all three hh matrices exactly equal their transposes. Rebuild shared-pair constrained model from dense checkpoint then deploy again; require exact parameter and public-history action/output equivalence.',
        crossing_trigger='After own_warm vs original WAIT results: if any wide-stage symmetric seed has absolute ordinary or strict difference>=0.10 OR idle-actions-per1000 difference>=100, run both crossed cells for ALL three wide symmetric seeds, family3 long_both2048: WAIT world+own h and own world+WAIT h. No other followup conditions.',
        denominators='Same original matured eligible postfork evaluation-contract denominator and idle enabled preaction denominator. Own-warm vs WAIT common futures and mature identities must match; endpoint denominator kept separate. Report per seed plus mean/sample SD. Retained raw strata remain descriptive.',
        continuity='From birth warmup and throughout free branch world/h persist without resets. Endpoint preserves native world/h. Detached crossings are explicit causal state interventions, never claimed as ordinary continuity.',
        artifacts='Complete public observations, logits, argmax actions, scoring arrays, request rows and initial/end world+h; hashes and replayable scorer evidence retained.')
    plan['protocol_hash'] = e.state_digest(plan)
    e.write(PLAN, plan)
    return plan


def checked():
    plan=e.read(PLAN)
    assert plan['protocol_hash']==e.state_digest({k:v for k,v in plan.items() if k!='protocol_hash'})
    replacements={}
    amendment=OUT/'implementation-amendment.json'
    if amendment.exists():
        change=e.read(amendment)
        assert change['protocol_hash']==plan['protocol_hash']
        assert change['original_sha256']==plan['sources'][change['path']]
        assert e.sha(change['original_snapshot'])==change['original_sha256']
        replacements[change['path']]=change['amended_sha256']
    for path,digest in {**plan['sources'],**plan['inputs']}.items():
        assert e.sha(path)==replacements.get(path,digest),path
    return plan


@torch.inference_mode()
def load_checked(arch,seed,stage):
    path=folder(arch,seed)/(stage+'.pt')
    model,world,h,cp=load_checkpoint(path)
    model.eval()
    for key,value in model.state_dict().items():
        assert torch.equal(value,cp['model'][key]),key
    result=dict(path=str(path),sha256=e.sha(path),all_loaded_tensors_exact=True,
                model_hash=e.state_digest(model.state_dict()),config=model.config())
    if arch=='symmetric':
        from experiments.learn_remember.symmetric_gru import constrain,deploy
        hh=model.core.weight_hh_l0.detach().reshape(3,model.width,model.width)
        assert torch.equal(hh,hh.transpose(-1,-2))
        result['symmetry_max_abs']=float((hh-hh.transpose(-1,-2)).abs().max())
        expanded=GRUAgent(**model.config());expanded.load_state_dict(cp['model'])
        constrained=constrain(expanded)
        rebuilt=deploy(constrained).eval()
        for key,value in model.state_dict().items():
            assert torch.equal(value,rebuilt.state_dict()[key]),key
        common=e.common_stream(seed)
        a=model.initial_state(8);b=rebuilt.initial_state(8)
        for obs in common['observations'][:32]:
            la,a=model.step(obs,a);lb,b=rebuilt.step(obs,b)
            assert torch.equal(la,lb) and torch.equal(a,b)
        result.update(rebuilt_packed_deployment_parameters_exact=True,rebuilt_public32_outputs_h_exact=True)
    return model,world,h,cp,result


@torch.inference_mode()
def own_warm(model,seed):
    common=e.common_stream(seed)
    world=ContextWorld.from_state_dict(common['birth_world'])
    state=model.initial_state(world.batch_size)
    observations=[];actions=[]
    for tick in range(1024):
        if tick%128==0:
            block=tick//128
            world.set_family(block%4)
            name='short' if block<4 else 'long_both'
            world.set_regime(**CONDITIONS[name],name=f'evaluation-common/{block}/{name}',dt=.25)
        obs=world.observe()
        logits,state=model.step(obs,state,world.elapsed)
        act=logits.argmax(-1)
        world.act(act)
        observations.append(obs.clone());actions.append(act.clone())
    end=world.state_dict()
    assert world.event_hash==common['world']['event_hash']
    assert e.state_digest(end['scheduler'])==e.state_digest(common['world']['scheduler'])
    assert e.state_digest(end['context'])==e.state_digest(common['world']['context'])
    return world,state,dict(observations=torch.stack(observations),actions=torch.stack(actions),world=end,h=state,
        external_event_hash=world.event_hash,scheduler_hash=e.state_digest(end['scheduler']),
        context_hash=e.state_digest(end['context']),all_external_events_match_WAIT=True)


def baseline(arch,seed,stage):
    if arch=='plain':
        return e.read(e.OUT/f'seed_{seed:03d}'/f'{stage}_common_long_both_family_3.json')
    value=e.read(folder(arch,seed)/f'behavior_{stage}.json')
    return next(row for row in value['records'] if row['family']==3 and row['condition']=='long_both')


def save_rollout(model,world,h,arch,seed,stage,history,plan):
    stem=OUT/f'seed_{seed:03d}'/f'{arch}_{stage}_{history}'
    if stem.with_suffix('.json').exists():return e.read(stem.with_suffix('.json'))
    scores,frames,states=e.rollout(model,world,h,family=3,condition='long_both',steps=2048)
    if history!='endpoint':
        old=baseline(arch,seed,stage)
        for key in ('primary_identity_hash','end_event_hash','end_scheduler_hash','end_context_hash'):
            assert scores[key]==old[key],(arch,seed,stage,history,key)
    stem.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(stem.with_suffix('.npz'),**frames)
    torch.save(states,stem.with_suffix('.pt'))
    requests=scores.pop('primary_requests');e.write(stem.with_suffix('.requests.json'),requests)
    scores.update(arch=arch,seed=seed,stage=stage,history=history,protocol_hash=plan['protocol_hash'],
        actor_ids=list(world.actor_ids),batch_size=world.batch_size,actor_partition='training' if history=='endpoint' else 'heldout',
        artifacts={str(stem.with_suffix(ext)):e.sha(stem.with_suffix(ext)) for ext in ('.npz','.pt','.requests.json')})
    e.write(stem.with_suffix('.json'),scores)
    print(json.dumps(dict(arch=arch,seed=seed,stage=stage,history=history,metrics=scores['metrics'],counts=scores['counts'])),flush=True)
    return scores


def run(seed):
    torch.set_num_threads(1)
    plan=checked()
    for arch in ARCHES:
        for stage in STAGES:
            model,world,h,cp,check=load_checked(arch,seed,stage)
            e.write(OUT/f'seed_{seed:03d}'/f'{arch}_{stage}_load-check.json',check)
            own,own_h,history=own_warm(model,seed)
            torch.save(history,OUT/f'seed_{seed:03d}'/f'{arch}_{stage}_own-warm.pt')
            save_rollout(model,own,own_h,arch,seed,stage,'own_warm',plan)
            save_rollout(model,world,h,arch,seed,stage,'endpoint',plan)


def crossed():
    torch.set_num_threads(1);plan=checked();trigger=[]
    for seed in range(3):
        own=e.read(OUT/f'seed_{seed:03d}'/'symmetric_wide_own_warm.json')
        old=baseline('symmetric',seed,'wide')
        delta={key:own['metrics'][key]-old['metrics'][key] for key in old['metrics']}
        if abs(delta['ordinary_completion_rate'])>=.10 or abs(delta['strict_completion_rate'])>=.10 or abs(delta['idle_actions_per1000'])>=100:
            trigger.append(dict(seed=seed,delta=delta))
    e.write(OUT/'crossing-decision.json',dict(triggered=bool(trigger),trigger_cases=trigger,all3seeds_if_triggered=True))
    if not trigger:return
    for seed in range(3):
        model,_,_,_,_=load_checked('symmetric',seed,'wide')
        common=e.common_stream(seed)
        wait_world=ContextWorld.from_state_dict(common['world']);wait_h=e.warmup(model,common)
        own=torch.load(OUT/f'seed_{seed:03d}'/'symmetric_wide_own-warm.pt',weights_only=False)
        own_world=ContextWorld.from_state_dict(own['world'])
        save_rollout(model,wait_world,own['h'],'symmetric',seed,'wide','wait_world_own_h',plan)
        save_rollout(model,own_world,wait_h,'symmetric',seed,'wide','own_world_wait_h',plan)


def summarize():
    plan=checked();records=[e.read(p) for p in sorted(OUT.glob('seed_*/*.json')) if not p.name.endswith(('load-check.json','.requests.json'))]
    for arch in ARCHES:
        for seed in range(3):
            for stage in STAGES:
                old=dict(baseline(arch,seed,stage));old.update(arch=arch,seed=seed,stage=stage,history='wait_warm',actor_partition='heldout')
                records.append(old)
    summary=[]
    for arch in ARCHES:
        for stage in STAGES:
            for history in ('wait_warm','own_warm','endpoint','wait_world_own_h','own_world_wait_h'):
                chosen=[r for r in records if (r['arch'],r['stage'],r['history'])==(arch,stage,history)]
                if not chosen:continue
                row=dict(arch=arch,stage=stage,history=history,n_seeds=len(chosen),per_seed=[dict(seed=r['seed'],counts=r['counts'],metrics=r['metrics'],unique_raw_1_3=r['unique_raw_1_3']) for r in chosen])
                for metric in chosen[0]['metrics']:
                    values=[r['metrics'][metric] for r in chosen]
                    row[metric]=dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values) if len(values)>1 else None)
                summary.append(row)
    result=dict(protocol_hash=plan['protocol_hash'],rows=summary,new_forks=len([r for r in records if r['history']!='wait_warm']))
    e.write(OUT/'summary.json',result);return result


def audit():
    plan=checked();paths=sorted(OUT.glob('seed_*/*.json'));forks=0
    from experiments.learn_remember.evaluate_audit import vector_counts
    for path in paths:
        if path.name.endswith(('load-check.json','.requests.json')):continue
        row=e.read(path)
        for artifact,digest in row['artifacts'].items():assert e.sha(artifact)==digest
        states=torch.load(path.with_suffix('.pt'),weights_only=False)
        with np.load(path.with_suffix('.npz')) as archive:
            frames={key:archive[key] for key in archive.files}
        assert np.array_equal(frames['logits'].argmax(-1),frames['actions'])
        independent=vector_counts(frames,states['initial_world'],row['evaluation_regime_id'],row['fork_tick'])
        for key,value in independent.items():assert row['counts'].get(key,0)==value
        for key in ('initial_world','end_world'):
            assert e.state_digest(states[key])==row['start_world_hash' if key=='initial_world' else 'end_world_hash']
        assert e.state_digest(states['initial_h'])==row['start_h_hash']
        assert e.state_digest(states['end_h'])==row['end_h_hash']
        forks+=1
    result=dict(protocol_hash=plan['protocol_hash'],forks=forks,verified=True,source_sha256=e.sha(__file__))
    e.write(OUT/'audit.json',result);print(json.dumps(result));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','crossed','summarize','audit']);parser.add_argument('--seed',type=int,default=0);args=parser.parse_args()
    if args.command=='run':run(args.seed)
    elif args.command=='prepare':prepare()
    elif args.command=='crossed':crossed()
    elif args.command=='audit':audit()
    else:summarize()
