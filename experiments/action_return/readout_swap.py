"""Post-result fixed BEGIN-score swap; no fit, threshold or backbone changes."""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from experiments.action_return import readout as r
from experiments.continuity.run import state_digest
from experiments.continuity.evaluate import _preserve_process_state
from experiments.lifelong.evaluate import evaluate
from experiments.lifelong.world import ContextWorld,NEW,REPLACE,NONE

torch.set_num_threads(1)
OUT=ROOT/'data/action-return/readout-swap-v1';PLAN=OUT/'protocol.json';SOURCE=Path(__file__).resolve()
FORMAT='action-return-readout-swap-posthoc-v1'

def prepare():
    main=r.check()
    if PLAN.exists():raise FileExistsError('swap protocol immutable')
    sources={}
    for seed in range(3):
        folder=r.OUT/f'seed_{seed:03d}'
        sources[str(seed)]={k:dict(path=str(folder/name),sha256=r.old.sha(folder/name)) for k,name in [('fit','fits.pt'),('fit_meta','fit.json'),('result','result.json')]}
    p=dict(format=FORMAT,created_utc=datetime.now(timezone.utc).isoformat(),posthoc=True,
        timing='Added after main non_input24 results showed many inappropriate BEGINs; not a prespecified main outcome.',
        source=dict(path=str(SOURCE),sha256=r.old.sha(SOURCE)),main_protocol=dict(path=str(r.PLAN),sha256=r.old.sha(r.PLAN)),sources=sources,
        seeds=[0,1,2],conditions=['short','long_both'],families=[0,1,2,3],steps=2048,arms=['original_non_input24','swap_begin_scores'],
        intervention='Freeze original trained non_input24 controller. Swap only its final BEGIN1/BEGIN2 scores at every normal step, preserving all nonBEGIN scores and native next h. No fitting or tuning.',
        gate='On each actual swap-trajectory obs/h, compare pre-swap and post-swap eleven-way argmax BEGIN-vs-nonBEGIN status and require exact equality. Unique highest BEGIN switches version; exact tied BEGIN scores retain original argmax convention. A tie against WAIT/FINISH retains original gate because both BEGIN indices have the same priority relative to each nonBEGIN index.',
        inference='Only public obs and native h, as in the original controller; no oracle, target gate, reset or forced target action.',
        baseline='Rerun all24 original non_input24 forks with added read-only pulse scorer; complete original evaluator record must exactly equal the saved main result.',
        source_state='Same actual endpoint world/h/model per original fork; subsequent receipts and h may diverge because swapped version choices change actions.',
        measures='Decoder-fitting-heldout lanes4..7: matched matured primary feasible correct BEGIN requests/primary matured requests, idle actions per1000 inactive opportunities, active correct-version and wrong-version BEGIN attempts. Split primary matured feasible correct accepted BEGIN requests by whether the accepted action occurred at that job NEW/REPLACE pulse or NONE; also save all actual accepted event counts separately.',
        matching='Require original/swap same source world/h, final external event/scheduler streams, actor/cases and primary matured request fingerprints for each fork.',
        interpretation='A behavioral counterfactual of swapping a fixed gate classifier version choice on each given h. Subsequent trajectories need not have identical timing or gate choices; not an isolated passive version accuracy or proof of native learned memory.',
        fit_unchanged=main['train'],new_forks=48,unique_intervention_forks=24,repeated_original_forks=24)
    OUT.mkdir(parents=True,exist_ok=True);r.old.write(PLAN,p);return p

def check():
    p=r.old.read(PLAN);main=r.check()
    if p['format']!=FORMAT or r.old.sha(SOURCE)!=p['source']['sha256'] or r.old.sha(r.PLAN)!=p['main_protocol']['sha256']:raise ValueError('swap/main plan or source changed')
    for files in p['sources'].values():
        for item in files.values():
            if r.old.sha(item['path'])!=item['sha256']:raise ValueError('main fit/result changed')
    return p,main

class SwapController(nn.Module):
    def __init__(self,controller):
        super().__init__();self.controller=deepcopy(controller)
        self.width,self.input_size,self.n_jobs,self.n_actions=controller.width,controller.input_size,controller.n_jobs,controller.n_actions
        self.checked=self.unique_begin_reversals=self.tied_begin_selected=0
    def step(self,obs,h,dt,max_step=None):
        before,h=self.controller.step(obs,h,dt,max_step);after=before.clone();after[...,1]=before[...,2];after[...,2]=before[...,1]
        a=before.argmax(-1);b=after.argmax(-1);ga=a.gt(0)&a.lt(3);gb=b.gt(0)&b.lt(3)
        if not torch.equal(ga,gb):raise AssertionError('score swap changed same-h BEGIN gate')
        unique=ga&before[...,1].ne(before[...,2])
        if not torch.equal(b[unique],3-a[unique]):raise AssertionError('unique BEGIN version not reversed')
        self.checked+=a.numel();self.unique_begin_reversals+=int(unique.sum());self.tied_begin_selected+=int((ga&~unique).sum())
        return after,h
    def config(self):return dict(kind='fixed_begin_score_swap',controller=self.controller.config(),same_h_gate_checks=self.checked,
        unique_selected_begin_reversals=self.unique_begin_reversals,tied_selected_begin=self.tied_begin_selected)

class PulseShadow(r.Shadow):
    def __init__(self,source,requests):
        super().__init__(source,requests);self.accepted_details={};self.extra={g:Counter() for g in self.counts}
    def act(self,world,actions):
        labels=world.oracle('midpoint');active=world.active.clone();version=world.version.clone();kind=world.event_kind.clone();begin=actions.gt(0)&actions.lt(3)
        for i,j in world.matured_mask.nonzero().tolist():
            key=self.key(world,i,j,world.matured_request_id[i,j])
            if key in self.seen and key in self.accepted_details:
                group='overlap' if i<4 else 'heldout';pulse=self.accepted_details[key]
                for g in (group,'all'):self.extra[g]['primary_feasible_began_at_cue' if pulse else 'primary_feasible_began_without_pulse']+=1
        for group,sl in (('overlap',slice(0,4)),('heldout',slice(4,8)),('all',slice(0,8))):
            self.extra[group]['active_correct_version_begin_attempts']+=int((active&begin&actions.sub(1).eq(version))[sl].sum())
            self.extra[group]['active_wrong_version_begin_attempts']+=int((active&begin&actions.sub(1).ne(version))[sl].sum())
        result=super().act(world,actions)
        feasible=world.last_result.eq(1)&begin&actions.eq(labels)
        for i,j in feasible.nonzero().tolist():
            event=int(kind[i,j])
            if event not in (NONE,NEW,REPLACE):raise AssertionError('accepted correct BEGIN at unexpected event')
            pulse=event in (NEW,REPLACE);self.accepted_details[self.key(world,i,j)]=pulse;group='overlap' if i<4 else 'heldout'
            for g in (group,'all'):self.extra[g]['accepted_feasible_begin_events_at_cue' if pulse else 'accepted_feasible_begin_events_without_pulse']+=1
        return result
    def result(self):
        result=super().result()
        for group,extra in self.extra.items():result[group]['pulse_counts']=dict(extra)
        return result

def evaluate_pulse(model,trainer,*,family,condition,requests):
    shadow=PulseShadow(trainer.world,requests)
    def observed(world,actions):return shadow.act(world,actions)
    with _preserve_process_state(),patch.object(ContextWorld,'act',observed):record=evaluate(model,trainer.world,trainer.h,family=family,condition=condition,steps=2048,device='cpu')
    extra=shadow.result()
    for group,row in extra.items():
        c=row['counts'];e=row['pulse_counts']
        if c.get('primary_feasible_began',0)!=e.get('primary_feasible_began_at_cue',0)+e.get('primary_feasible_began_without_pulse',0):raise AssertionError('pulse split did not partition matched matured feasible begins')
    if record['primary']['matured']!=extra['all']['counts'].get('primary_matured',0):raise AssertionError('pulse shadow primary mismatch')
    return dict(record=record,shadow=extra)

def run(seed):
    p,main=check();out=OUT/f'seed_{seed:03d}';out.mkdir(exist_ok=False);started=time.perf_counter()
    src=main['sources'][str(seed)];trainer=r.native.NativeTrainer.from_checkpoint(src['endpoint']['path'],device='cpu');initial=state_digest(trainer.semantic_state())
    saved=torch.load(p['sources'][str(seed)]['fit']['path'],map_location='cpu',weights_only=False)
    controller=r.BeginController(trainer.model,'non_input24',saved['fits']['non_input24'],saved['reconstruction']['non_input_indices'])
    requests=[tuple(k) for group in saved['training_request_sets'].values() for k in group];archive=r.old.read(p['sources'][str(seed)]['result']['path'])['records'];records=[]
    for arm in p['arms']:
        model=controller if arm=='original_non_input24' else SwapController(controller)
        for condition in p['conditions']:
            for family in p['families']:
                value=evaluate_pulse(model,trainer,family=family,condition=condition,requests=requests)
                original=next(x for x in archive if x['arm']=='non_input24' and x['condition']==condition and x['family']==family)
                if arm=='original_non_input24':
                    if json.loads(json.dumps(value['record'],allow_nan=False))!=original['record']:raise AssertionError('original repeat changed evaluator result')
                    if {g:{k:v for k,v in value['shadow'][g].items() if k!='pulse_counts'} for g in value['shadow']}!=original['shadow']:raise AssertionError('original repeat changed original shadow')
                for field in ('start_world_state_hash','start_hidden_hash','end_scheduler_hash','end_event_hash','actor_ids','case_ids'):
                    if json.loads(json.dumps(value['record'][field]))!=original['record'][field]:raise AssertionError('swap cases/source mismatch')
                for group in value['shadow']:
                    if value['shadow'][group]['primary_request_set_sha256']!=original['shadow'][group]['primary_request_set_sha256']:raise AssertionError('swap primary cases differ')
                value.update(arm=arm,condition=condition,family=family);records.append(value);r.old.write(out/'partial.json',dict(seed=seed,records=records))
                print(json.dumps(dict(seed=seed,arm=arm,condition=condition,family=family,done=len(records),seconds=time.perf_counter()-started)),flush=True)
    if state_digest(trainer.semantic_state())!=initial:raise AssertionError('swap changed source state')
    check();result=dict(format=FORMAT,seed=seed,protocol_sha256=r.old.sha(PLAN),records=records,original_exact_forks=8,source_unchanged=True,seconds=time.perf_counter()-started)
    r.old.write(out/'result.json',result);return result

def summary():
    p,_=check();rows=[]
    for seed in p['seeds']:
        result=r.old.read(OUT/f'seed_{seed:03d}'/'result.json')
        if len(result['records'])!=16 or result['protocol_sha256']!=r.old.sha(PLAN):raise ValueError('incomplete swap result')
        for arm in p['arms']:
            for condition in p['conditions']:
                for group in ('heldout','overlap','all'):
                    counts=Counter();pulse=Counter()
                    for item in result['records']:
                        if item['arm']==arm and item['condition']==condition:counts.update(item['shadow'][group]['counts']);pulse.update(item['shadow'][group]['pulse_counts'])
                    rows.append(dict(seed=seed,arm=arm,condition=condition,actor_group=group,counts=dict(counts),pulse_counts=dict(pulse),
                        feasible_began_rate=counts['primary_feasible_began']/counts['primary_matured'],idle_actions_per1000=1000*counts['idle_actions']/counts['idle_opportunities']))
    value=dict(format=FORMAT,protocol_sha256=r.old.sha(PLAN),posthoc=True,rows=rows);r.old.write(OUT/'summary.json',value);return value

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','run','summary'));parser.add_argument('--seed',type=int,choices=(0,1,2));args=parser.parse_args()
    value=prepare() if args.command=='prepare' else run(args.seed) if args.command=='run' else summary();print(json.dumps(dict(format=value['format'],output=str(OUT))),flush=True)
