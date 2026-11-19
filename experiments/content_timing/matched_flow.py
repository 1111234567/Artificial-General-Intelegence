"""Prospectively fixed matched content/duration forks on real continuous anchors.

Only this new experiment writes under data/content-timing/matched-v1. Frozen
historical scripts and fitted controllers are imported without modification.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import torch
from experiments.action_return import readout as b, finish_readout as f, refit_policy as r
from experiments.lifelong.world import FAMILY_MAPPINGS, NEW, REPLACE
from experiments.lifelong.evaluate import CONDITIONS

OUT = b.ROOT / 'data/content-timing/matched-v1'
PLAN = OUT / 'protocol.json'
FORMAT = 'matched-real-anchor-content-duration-v1'
STAGES = ('cue', 'midpoint', 'common_age2', 'first_legal', 'teacher_first', 'common_short_teacher')
DURATIONS = (2.0, 8.0)
QUOTA = 3
MAX_BACKGROUND = 4096
DT = .25
EPS = 1e-6


def write(path, value):
    b.old.write(Path(path), value)


def read(path):
    return b.old.read(Path(path))


def sha(path):
    return b.old.sha(Path(path))


def freeze():
    if PLAN.exists():
        raise FileExistsError('immutable protocol already exists')
    parent = read(r.PLAN)
    if parent['protocol_hash'] != b.state_digest({k: v for k, v in parent.items() if k != 'protocol_hash'}):
        raise ValueError('parent protocol digest')
    sources = {}
    for seed in range(3):
        folder = r.OUT / f'seed_{seed:03d}'
        result = read(folder / 'result.json')
        if result['result_hash'] != b.state_digest({k: v for k, v in result.items() if k != 'result_hash'}):
            raise ValueError('parent result digest')
        model = folder / 'model_own_policy.pt'
        endpoint = parent['seeds'][str(seed)]['source']['endpoint']
        if sha(model) != result['model_sha256']['own_policy'] or sha(endpoint['path']) != endpoint['sha256']:
            raise ValueError('wrong frozen source bytes')
        sources[str(seed)] = dict(model=str(model), model_sha256=sha(model), endpoint=endpoint,
            parent_result=str(folder/'result.json'), parent_result_sha256=sha(folder/'result.json'))
    files = [Path(__file__).resolve(), Path(b.__file__).resolve(), Path(f.__file__).resolve(),
        Path(r.__file__).resolve(), b.ROOT/'experiments/lifelong/world.py',
        b.ROOT/'experiments/continuity/world.py', b.ROOT/'experiments/undirected/model.py',
        b.ROOT/'experiments/undirected/run.py']
    plan = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
        status='fixed_before_any_new_background_or_treatment_result',
        selection_history='Prior refit-policy-v1 aggregate results were known. This protocol fixes one new matched experiment with no threshold search and no new readout fitting.',
        seeds=[0,1,2], families=[0,1,2,3], actors=[4,5,6,7], jobs=[0,1],
        dimensions=dict(raw_payload=[0,1,2,3], duration=list(DURATIONS), real_version=[0,1]),
        sources=sources, parent_protocol=dict(path=str(r.PLAN),sha256=sha(r.PLAN)),
        source_files={str(p):sha(p) for p in files},
        background=dict(source='Full actual147456 endpoint world and inherited h; frozen own_policy model. Execute its actual eleven-action argmax at every tick in all8 actors.',
            condition='long_both', dt=DT, family='Change future family only; inherited pending contracts remain intact.',
            max_ticks=MAX_BACKGROUND, quota_per_seed_family_lane_job_real_version=QUOTA,
            selection='Earliest eligible real NEW/REPLACE under the new regime and requested family in each lane/job/real-version stratum. No response score is used. Require the next scheduled target NEW/REPLACE/CANCEL to be strictly after cue+8+original window. Continue through the entire needed source-observation suffix.',
            anchor='Save full real post-observe/pre-act world, exact original observation, continuous pre-cue h and source post-cue h. Real version and kind are never relabeled or altered.'),
        treatment=dict(arms='4 raw payloads x durations2,8 at each identical actual anchor; preserve real family, version, event kind, request identity, window and scheduler.',
            world_changes=['raw_payload','raw_event_payload','payload','event_payload','duration','event_duration','deadline','_score_deadline'],
            pulse='Only target job raw-payload one-hot and duration in the real cue observation are replaced. Mapped target follows the original family mapping. No hidden state, weight, previous receipt, context or version is replaced.',
            hash='Original event_hash remains the authentic scheduler prefix digest; intervention is separately fingerprinted and the complete mutated world hash is saved. It must not be mistaken for an unmodified scheduler event.',
            actions='Target executes its correct-version BEGIN at cue, then WAIT throughout. All non-target actor/job actions are locked to the recorded original own-policy background actions at the same absolute tick. Model argmax is measured but not executed in these forks.',
            receipts='Retain actual unmodified world-generated receipts. Fixed real version means identical BEGIN action/accepted result in all8 variants; WAIT yields no subsequent receipt. No FINISH is executed, so payload labels never enter feedback. Version is present in BEGIN receipt and is not isolated as a pure memory test.',
            equality='Assert every non-target public observation equals source background exactly, and every complete observation across all8 treatments is identical after cue. At cue only target payload/duration columns may differ.',
            legal='Use actual active+started and ready_time<=now<=deadline, after an actually accepted locked BEGIN; first observed legal time is not teacher midpoint.'),
        observation=dict(stages=list(STAGES), timing='All scores/features sampled post-model-step/pre-action. midpoint is duration/2 since actual locked BEGIN; common_age2 is common absolute cue-age2. teacher_first is first observed midpoint-oracle FINISH time; common_short_teacher is short-arm teacher_first absolute age in both duration arms.',
            stage_duplicates='short first_legal=common_age2 and short teacher_first=common_short_teacher reuse the same state with different stage labels. Do not count these rows as independent samples.',
            outputs='Unchanged actual11 logits, actual8 FINISH logits, uncentered original conditional affine offsets8, descending FINISH rank8, original FP64 non_input24 states; teacher and private-world labels are scorer-only.'),
        analysis=dict(primary='Within-anchor pairwise correct-versus-alternative target score differential, symmetric paired ordering and argmax distinction at equal duration/stage; per-seed descriptive summaries across four families.',
            timing='At common_age2 compare short legal vs long processing; at common_short_teacher compare short teacher-FINISH vs long processing. Separately report WAIT/BEGIN/FINISH argmax and FINISH-minus-best-nonFINISH margin.',
            heldout='Actors4..7 were held out from external readout fitting only; backbone saw all8. Independent probes must split whole anchor families of variants together and use actor4,5 train vs6,7 test then reverse; no score-selected model or hyperparameter changes here.',
            independence='Three seeds are the independent replicated units. Anchors, jobs, versions, treatments and stages are repeated nested measurements.'),
        schema=dict(file='seed_000/records.pt (similarly001,002)', rows='List of metadata dictionaries; all tensors use the exact same row order.',
            tensors=dict(x24='float64[N,24]',scores11='float64[N,11]',finish_scores8='float64[N,8]',
                finish_offsets8='float64[N,8]',rank8='int64[N,8]; descending original FINISH scores; code4*version+mapped_payload'),
            metadata=['row_id','anchor_id','seed','family','actor_lane','actor_id','job','case_id','request_id','cue_kind','version',
                'source_raw_payload','raw_payload','mapped_payload','target8','duration_arm','duration','stage','age','tick','now','issued_time',
                'active','started','ready_time','deadline','legal','previous_observation_time','previous_observation_legal',
                'model_action','conditional_choice8','content_correct','version_correct','payload_correct','teacher_action','teacher_finish_expected',
                'finish_vs_nonfinish_margin','target_external_event_free','source_fingerprint','treatment_fingerprint']),
        limits=['Controlled real-world forks under locked operations, not autonomous policy behavior or completion evidence.',
            'Duration changes preserve the original next-external-event scheduler; selection restricts analysis to an uninterrupted treatment window, so this is not a natural resampled duration regime.',
            'Version is observed and fixed within an anchor, and its BEGIN receipt is retained. Raw payload is the isolated within-anchor content manipulation.',
            'Only three seeds; paired discrimination and decodability do not establish learned causal use or native recurrent learning.'])
    plan['protocol_hash'] = b.state_digest(plan)
    OUT.mkdir(parents=True,exist_ok=True)
    write(PLAN,plan)
    return plan


def checked():
    plan=read(PLAN)
    if plan['protocol_hash'] != b.state_digest({k:v for k,v in plan.items() if k!='protocol_hash'}):
        raise ValueError('protocol digest')
    for path,expected in plan['source_files'].items():
        if sha(path)!=expected:
            raise ValueError('frozen source changed '+path)
    if sha(plan['parent_protocol']['path'])!=plan['parent_protocol']['sha256']:
        raise ValueError('parent protocol changed')
    return plan


def load_model(seed,plan):
    source=plan['sources'][str(seed)]
    for path,expected in ((source['model'],source['model_sha256']),
                          (source['endpoint']['path'],source['endpoint']['sha256']),
                          (source['parent_result'],source['parent_result_sha256'])):
        if sha(path)!=expected:
            raise ValueError('source bytes changed')
    trainer=b.native.NativeTrainer.from_checkpoint(source['endpoint']['path'],device='cpu')
    if b.state_digest(trainer.semantic_state())!=source['endpoint']['semantic_hash']:
        raise ValueError('complete endpoint semantics differ')
    saved=torch.load(source['model'],weights_only=False,map_location='cpu')
    begin=[{k:saved[f'begin_head.j{j}_{k}'] for k in ('mean','std','coeff','bias')} for j in range(2)]
    finish=[{k:saved[f'j{j}_{k}'] for k in ('mean','std','coef','bias')} for j in range(2)]
    model=f.FinishController(b.BeginController(trainer.model,'non_input24',begin,saved['begin_head.indices'].tolist()),finish)
    model.load_state_dict(saved,strict=True)
    if b.state_digest(model.state_dict())!=b.state_digest(saved):
        raise ValueError('controller reconstruction differs')
    return trainer,model.eval()


def next_external(world,i,j):
    return min(float(world._next_new_time[i,j]),float(world._modify_time[i,j]))


def collect(seed,family,trainer,model):
    world=b.ContextWorld.from_state_dict(trainer.world.state_dict(),device='cpu')
    h=trainer.h.clone()
    world.set_family(family)
    world.set_regime(**CONDITIONS['long_both'],name=f'matched/background/{family}',dt=DT)
    counts=Counter();anchors=[];observations=[];actions=[];end_needed=-1
    initial_world=world.state_dict()
    with torch.no_grad():
        for t in range(MAX_BACKGROUND+44):
            before=h.clone();obs=world.observe().double()
            logits,h=model.step(obs,h,world.elapsed)
            observations.append(obs.clone());actions.append(logits.argmax(-1).clone())
            if t<MAX_BACKGROUND:
                for i in range(4,8):
                    for j in range(2):
                        version=int(world.version[i,j]);key=(i,j,version)
                        command=int(world.event_kind[i,j]) in (NEW,REPLACE)
                        window=float(world.event_window[i,j]);safe=next_external(world,i,j)>world.now+8+window+EPS
                        if command and bool(world.event_eligible[i,j]) and int(world.event_regime_id[i,j])==world.regime_id and int(world.current_family[i,j])==family and safe and counts[key]<QUOTA:
                            state=world.state_dict();identity=f's{seed}-f{family}-a{i}-j{j}-v{version}-q{counts[key]:02d}'
                            fingerprint=b.state_digest(dict(world=state,h_before=before,observation=obs))
                            anchors.append(dict(anchor_id=identity,seed=seed,family=family,actor_lane=i,actor_id=world.actor_ids[i],job=j,
                                version=version,case_id=world.case_ids[i],request_id=int(world.request_id[i,j]),cue_kind=int(world.event_kind[i,j]),
                                source_raw_payload=int(world.raw_event_payload[i,j]),source_duration=float(world.duration[i,j]),window=window,
                                source_index=t,cue_tick=world.tick,cue_time=world.now,next_external_time=next_external(world,i,j),
                                source_fingerprint=fingerprint,world=state,h_before=before.clone(),h_after=h.clone(),obs=obs.clone()))
                            counts[key]+=1;end_needed=max(end_needed,t+math.ceil((8+window)/DT))
            world.act(actions[-1])
            done=len(counts)==16 and min(counts.values())>=QUOTA
            if done and t>=end_needed:
                break
    if len(counts)!=16 or min(counts.values())<QUOTA:
        raise ValueError(f'fixed background quota not attained: seed{seed} family{family} {counts}')
    return dict(anchors=anchors,observations=torch.stack(observations),actions=torch.stack(actions),
        initial_world=initial_world,initial_h=trainer.h.clone(),end_world=world.state_dict(),end_h=h.clone(),
        source_controller_hash=b.state_digest(model.state_dict()),counts={str(k):v for k,v in counts.items()},
        source_rule='actual uninterrupted free eleven-way own_policy argmax from inherited full endpoint',ticks=len(observations))


def change_command(anchor,raw,duration):
    world=b.ContextWorld.from_state_dict(anchor['world'],device='cpu')
    i,j=anchor['actor_lane'],anchor['job'];mapped=FAMILY_MAPPINGS[anchor['family']][raw]
    for name,value in (('raw_payload',raw),('raw_event_payload',raw),('payload',mapped),('event_payload',mapped),
                       ('duration',duration),('event_duration',duration)):
        getattr(world,name)[i,j]=value
    world.deadline[i,j]=world.now+duration+anchor['window']
    world._score_deadline[i,j]=world.deadline[i,j]
    obs=anchor['obs'].clone();offset=j*28
    obs[i,offset+5:offset+9]=0;obs[i,offset+5+raw]=1;obs[i,offset+9]=duration
    treatment_hash=b.state_digest(dict(anchor=anchor['source_fingerprint'],raw_payload=raw,duration=duration,
        world=world.state_dict(),model_observation=obs))
    return world,obs,treatment_hash


def is_legal(world,i,j):
    return bool(world.active[i,j] and world.started[i,j] and world.ready_time[i,j]-EPS<=world.now<=world.deadline[i,j]+EPS)


def teacher_index(duration,window):
    return math.ceil((duration+window/2-EPS)/DT)


def fork_anchor(anchor,background,model,tables):
    i,j=anchor['actor_lane'],anchor['job'];variants=[];worlds=[];obslist=[]
    for duration in DURATIONS:
        for raw in range(4):
            world,obs,fingerprint=change_command(anchor,raw,duration)
            variants.append(dict(raw_payload=raw,duration=duration,treatment_fingerprint=fingerprint))
            worlds.append(world);obslist.append(obs)
    h=anchor['h_before'].repeat(8,1)
    stage_ticks=[]
    for v in variants:
        d=v['duration'];stage_ticks.append(dict(cue=0,midpoint=round((d/2)/DT),common_age2=8,
            first_legal=round(d/DT),teacher_first=teacher_index(d,anchor['window']),
            common_short_teacher=teacher_index(2.,anchor['window'])))
    horizon=max(max(s.values()) for s in stage_ticks)
    prev_legal=[False]*8;first_legal=[None]*8;first_teacher=[None]*8
    initial_scheduler=b.state_digest(anchor['world']['scheduler'])
    treatment_worlds=[]
    for k,world in enumerate(worlds):
        if b.state_digest(world.state_dict()['scheduler'])!=initial_scheduler:
            raise ValueError('scheduler changed during command treatment')
        treatment_worlds.append(dict(treatment_fingerprint=variants[k]['treatment_fingerprint'],
            initial_world_hash=b.state_digest(world.state_dict()),scheduler_hash=initial_scheduler,
            original_event_hash=world.event_hash))
    with torch.no_grad():
        for offset in range(horizon+1):
            if offset:
                obslist=[w.observe().double() for w in worlds]
            obs=torch.stack(obslist)
            source_obs=background['observations'][anchor['source_index']+offset]
            keep=torch.ones_like(source_obs,dtype=torch.bool);keep[i,j*28:(j+1)*28]=False
            for k in range(8):
                if not torch.equal(obs[k][keep],source_obs[keep]):
                    raise ValueError('non-target public background diverged')
                if offset and int(worlds[k].event_kind[i,j])!=0:
                    raise ValueError('target external event inside treatment window')
            if offset:
                if not torch.equal(obs,obs[:1].expand_as(obs)):
                    raise ValueError('post-cue public observations differ across treatments')
            else:
                expected=source_obs.clone();allowed=torch.zeros_like(keep);allowed[i,j*28+5:j*28+10]=True
                if not all(torch.equal(v[~allowed],expected[~allowed]) for v in obs):
                    raise ValueError('unapproved cue observation change')
            logits,h=model.step(obs.flatten(0,1),h,worlds[0].elapsed)
            logits=logits.reshape(8,8,2,11);state=h.reshape(8,8,80)
            for k,(v,world) in enumerate(zip(variants,worlds)):
                legal=is_legal(world,i,j);teacher=int(world.oracle('midpoint')[i,j])
                if legal and first_legal[k] is None:first_legal[k]=offset
                if teacher>=3 and first_teacher[k] is None:first_teacher[k]=offset
                for stage,tick in stage_ticks[k].items():
                    if offset!=tick:continue
                    if stage=='first_legal' and first_legal[k]!=offset:raise ValueError('not actual first legal')
                    if stage=='teacher_first' and first_teacher[k]!=offset:raise ValueError('not actual first teacher FINISH')
                    raw=v['raw_payload'];mapped=FAMILY_MAPPINGS[anchor['family']][raw];target=4*anchor['version']+mapped
                    x=state[k,i].index_select(0,model.begin_head.indices).clone();scores=logits[k,i,j].clone()
                    offsets=(x-getattr(model,f'j{j}_mean'))/getattr(model,f'j{j}_std')
                    offsets=offsets@getattr(model,f'j{j}_coef').T+getattr(model,f'j{j}_bias')
                    rank=torch.argsort(scores[3:],descending=True,stable=True);choice=int(rank[0]);action=int(scores.argmax())
                    row=dict(row_id=len(tables['rows']),anchor_id=anchor['anchor_id'],seed=anchor['seed'],family=anchor['family'],
                        actor_lane=i,actor_id=anchor['actor_id'],job=j,case_id=anchor['case_id'],request_id=anchor['request_id'],
                        cue_kind=anchor['cue_kind'],version=anchor['version'],source_raw_payload=anchor['source_raw_payload'],
                        raw_payload=raw,mapped_payload=mapped,target8=target,duration_arm='short' if v['duration']==2 else 'long',
                        duration=v['duration'],stage=stage,age=offset*DT,tick=world.tick,now=world.now,issued_time=float(world.issued_time[i,j]),
                        active=bool(world.active[i,j]),started=bool(world.started[i,j]),ready_time=float(world.ready_time[i,j]),
                        deadline=float(world.deadline[i,j]),legal=legal,previous_observation_time=world.now-DT,
                        previous_observation_legal=prev_legal[k],model_action=action,conditional_choice8=choice,content_correct=choice==target,
                        version_correct=choice//4==anchor['version'],payload_correct=choice%4==mapped,teacher_action=teacher,
                        teacher_finish_expected=teacher>=3,finish_vs_nonfinish_margin=float(scores[3:].max()-scores[:3].max()),
                        target_external_event_free=True,source_fingerprint=anchor['source_fingerprint'],treatment_fingerprint=v['treatment_fingerprint'])
                    tables['rows'].append(row);tables['x24'].append(x);tables['scores11'].append(scores)
                    tables['finish_scores8'].append(scores[3:]);tables['finish_offsets8'].append(offsets.clone());tables['rank8'].append(rank)
                prev_legal[k]=legal
            locked=background['actions'][anchor['source_index']+offset].clone()
            locked[i,j]=1+anchor['version'] if offset==0 else 0
            for world in worlds:
                world.act(locked)
                if offset==0 and int(world.last_result[i,j])!=1:raise ValueError('locked BEGIN not actually accepted')
    return dict(anchor_id=anchor['anchor_id'],variants=treatment_worlds,public_observation_equality=True,
        non_target_source_observation_equality=True,target_begin_accepted=True,target_external_event_free=True,
        first_legal_ticks=first_legal,first_teacher_ticks=first_teacher,horizon_ticks=horizon,
        all_model_argmax_actions_measurement_only=True)


def run(seed):
    torch.set_num_threads(1);plan=checked();folder=OUT/f'seed_{seed:03d}';folder.mkdir(exist_ok=False)
    started=time.perf_counter();trainer,model=load_model(seed,plan)
    source_hash=b.state_digest(trainer.semantic_state());model_hash=b.state_digest(model.state_dict())
    tables={key:[] for key in ('rows','x24','scores11','finish_scores8','finish_offsets8','rank8')}
    traces=[];backgrounds=[]
    for family in range(4):
        background=collect(seed,family,trainer,model)
        path=folder/f'background_f{family}.pt';torch.save(background,path)
        backgrounds.append(dict(family=family,path=str(path),sha256=sha(path),ticks=background['ticks'],anchors=len(background['anchors']),counts=background['counts']))
        print(seed,'background',family,background['ticks'],len(background['anchors']),flush=True)
        for n,anchor in enumerate(background['anchors']):
            traces.append(fork_anchor(anchor,background,model,tables))
            if (n+1)%16==0:print(seed,'forks',family,n+1,flush=True)
    for key in tables:
        if key!='rows':tables[key]=torch.stack(tables[key])
    tables.update(format=FORMAT,protocol_hash=plan['protocol_hash'],non_input_indices=model.begin_head.indices.tolist(),
        source_controller_hash=model_hash,source_semantic_hash=source_hash)
    path=folder/'records.pt';torch.save(tables,path)
    if source_hash!=b.state_digest(trainer.semantic_state()) or model_hash!=b.state_digest(model.state_dict()):
        raise ValueError('frozen endpoint/controller mutated')
    checked()
    result=dict(format=FORMAT,seed=seed,protocol_hash=plan['protocol_hash'],records_path=str(path),records_sha256=sha(path),
        background=backgrounds,traces=traces,rows=len(tables['rows']),anchors=len(traces),source_unchanged=True,
        source_semantic_hash=source_hash,source_controller_hash=model_hash,seconds=time.perf_counter()-started)
    result['result_hash']=b.state_digest(result);write(folder/'result.json',result)


def launch():
    checked()
    def one(seed):
        with (OUT/f'seed_{seed:03d}.log').open('w') as stream:
            proc=subprocess.Popen([sys.executable,'-m','experiments.content_timing.matched_flow','run','--seed',str(seed)],
                cwd=b.ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1'},stdout=stream,stderr=subprocess.STDOUT)
            return dict(seed=seed,pid=proc.pid,returncode=proc.wait())
    with ThreadPoolExecutor(max_workers=3) as pool:codes=list(pool.map(one,range(3)))
    write(OUT/'exit_codes.json',codes)
    if any(x['returncode'] for x in codes):raise RuntimeError(codes)


def summarize():
    plan=checked();summaries=[];pair_rows=[];timing=[];files={}
    for seed in range(3):
        folder=OUT/f'seed_{seed:03d}';result=read(folder/'result.json')
        if result['result_hash']!=b.state_digest({k:v for k,v in result.items() if k!='result_hash'}):raise ValueError('result digest')
        if sha(result['records_path'])!=result['records_sha256']:raise ValueError('raw changed')
        data=torch.load(result['records_path'],weights_only=False,map_location='cpu');rows=data['rows']
        files[str(folder/'result.json')]=sha(folder/'result.json');files[result['records_path']]=result['records_sha256']
        groups=defaultdict(list)
        for n,row in enumerate(rows):groups[(row['family'],row['duration_arm'],row['stage'])].append(n)
        for (family,duration,stage),ids in groups.items():
            selected=[rows[n] for n in ids];count=len(ids)
            action_counts=Counter('WAIT' if row['model_action']==0 else 'BEGIN' if row['model_action']<3 else 'FINISH' for row in selected)
            summaries.append(dict(seed=seed,family=family,duration_arm=duration,stage=stage,n=count,
                content_correct=sum(r['content_correct'] for r in selected),version_correct=sum(r['version_correct'] for r in selected),
                payload_correct=sum(r['payload_correct'] for r in selected),model_action_counts=dict(action_counts),
                mean_finish_vs_nonfinish_margin=statistics.mean(r['finish_vs_nonfinish_margin'] for r in selected),
                legal=sum(r['legal'] for r in selected),teacher_finish_expected=sum(r['teacher_finish_expected'] for r in selected)))
        by_pair=defaultdict(dict)
        for n,row in enumerate(rows):by_pair[(row['anchor_id'],row['duration_arm'],row['stage'])][row['raw_payload']]=n
        for (anchor,duration,stage),group in by_pair.items():
            for a in range(4):
                for c in range(a+1,4):
                    ia,ic=group[a],group[c];ra,rc=rows[ia],rows[ic];ta,tc=ra['target8'],rc['target8']
                    scores=data['finish_scores8'];ma=float(scores[ia,ta]-scores[ia,tc]);mc=float(scores[ic,ta]-scores[ic,tc])
                    delta=ma-mc
                    pair_rows.append(dict(seed=seed,family=ra['family'],anchor_id=anchor,actor_lane=ra['actor_lane'],job=ra['job'],version=ra['version'],
                        duration_arm=duration,stage=stage,raw_a=a,raw_b=c,row_a=ia,row_b=ic,target_a=ta,target_b=tc,
                        relative_score_a=ma,relative_score_b=mc,matched_difference=delta,symmetric_pair_order_correct=delta>0,
                        both_targets_outrank_alternative=ma>0 and mc<0,conditional_argmax_distinct=ra['conditional_choice8']!=rc['conditional_choice8'],
                        both_argmax_correct=ra['content_correct'] and rc['content_correct']))
        indexed={(r['anchor_id'],r['raw_payload'],r['duration_arm'],r['stage']):r for r in rows}
        for r0 in rows:
            if r0['duration_arm']!='short' or r0['stage'] not in ('common_age2','common_short_teacher'):continue
            r1=indexed[(r0['anchor_id'],r0['raw_payload'],'long',r0['stage'])]
            timing.append(dict(seed=seed,family=r0['family'],anchor_id=r0['anchor_id'],raw_payload=r0['raw_payload'],stage=r0['stage'],
                short_row=r0['row_id'],long_row=r1['row_id'],age=r0['age'],
                short_action=r0['model_action'],long_action=r1['model_action'],short_legal=r0['legal'],long_legal=r1['legal'],
                short_teacher_action=r0['teacher_action'],long_teacher_action=r1['teacher_action'],
                short_minus_long_finish_margin=r0['finish_vs_nonfinish_margin']-r1['finish_vs_nonfinish_margin']))
    pair_summaries=[]
    for seed in range(3):
        for duration in ('short','long'):
            for stage in STAGES:
                selected=[r for r in pair_rows if r['seed']==seed and r['duration_arm']==duration and r['stage']==stage]
                pair_summaries.append(dict(seed=seed,duration_arm=duration,stage=stage,pairs=len(selected),
                    mean_matched_difference=statistics.mean(r['matched_difference'] for r in selected),
                    pair_order_correct=sum(r['symmetric_pair_order_correct'] for r in selected),
                    both_targets_outrank_alternative=sum(r['both_targets_outrank_alternative'] for r in selected),
                    conditional_argmax_distinct=sum(r['conditional_argmax_distinct'] for r in selected),both_argmax_correct=sum(r['both_argmax_correct'] for r in selected)))
    write(OUT/'paired_scores.json',dict(protocol_hash=plan['protocol_hash'],rows=pair_rows))
    write(OUT/'matched_timing.json',dict(protocol_hash=plan['protocol_hash'],rows=timing))
    value=dict(format=FORMAT,protocol_hash=plan['protocol_hash'],rows=summaries,paired=pair_summaries,
        bindings=files,paired_scores_sha256=sha(OUT/'paired_scores.json'),matched_timing_sha256=sha(OUT/'matched_timing.json'))
    write(OUT/'summary.json',value)
    lines=['# Matched real-anchor content/duration experiment','',
        'Locked-BEGIN/WAIT controlled forks; model argmax is measured and never executed in these forks. Free own-policy background is separately archived. Three seeds are the independent units.','',
        'Each seed: 4 families x 4 heldout actor lanes x 2 jobs x 2 real versions x 3 anchors = 192 anchors. Each anchor has 4 raw-payload x 2 duration variants; six stages include duplicated short-arm stage labels.','',
        '|duration|stage|seed|content correct / rows|pair order correct / pairs|distinct conditional argmax / pairs|','|---|---|---:|---:|---:|---:|']
    for duration in ('short','long'):
        for stage in STAGES:
            for seed in range(3):
                ss=[r for r in summaries if r['seed']==seed and r['duration_arm']==duration and r['stage']==stage]
                pp=next(r for r in pair_summaries if r['seed']==seed and r['duration_arm']==duration and r['stage']==stage)
                lines.append(f"|{duration}|{stage}|{seed}|{sum(r['content_correct'] for r in ss)}/{sum(r['n'] for r in ss)}|{pp['pair_order_correct']}/{pp['pairs']}|{pp['conditional_argmax_distinct']}/{pp['pairs']}|")
    lines+=['','Pairs use the symmetric differential [score_a(target_a)-score_a(target_b)] - [score_b(target_a)-score_b(target_b)]. A positive differential can reveal weak content sensitivity even when the final argmax remains constant. These nested pairs are descriptive, not independent replications.','',
        'Actual world-generated BEGIN receipts are preserved and identical across payload/duration treatments within each anchor. Version is fixed and is also repeated in the BEGIN receipt. Scheduler is preserved; anchors require an uninterrupted target window through8+the original window. Only the cue payload/duration differs in public inputs; every post-cue observation across the eight variants and every non-target observation versus the actual source background was checked for exact equality.','',
        'Common_age2 compares short first legal against long processing. Common_short_teacher compares the first short-arm midpoint-teacher FINISH tick against long processing. Teacher labels are read-only. First legal is not an expected-FINISH teacher time. See matched_timing.json for paired action and FINISH margin differences.','',
        'Raw FP64 features and scores are in seed_000/records.pt through seed_002/records.pt. Original source trajectories, true anchor world/h snapshots, source fingerprints and intervention fingerprints are in background_f0.pt through background_f3.pt in each seed folder. Protocol and result hashes bind all sources.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','run','launch','summarize'));parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.command=='freeze':freeze()
    elif args.command=='run':run(args.seed)
    elif args.command=='launch':launch()
    else:summarize()
