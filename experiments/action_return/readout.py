"""Frozen-state BEGIN controller: prospectively fixed fitting and autonomous forks.

This is an external, supervised action head, not native port learning. Source
models/worlds/evaluators are imported unchanged. The optional scorer patch only
reads the actual world before/after its original act method.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from torch.nn import functional as F
from experiments.coverage_exit import coverage as old
from experiments.continuity.run import state_digest
from experiments.continuity.evaluate import _preserve_process_state, _new_since_fork
from experiments.lifelong.world import ContextWorld, NEW, REPLACE
from experiments.lifelong.evaluate import evaluate, _phases
from experiments.undirected import run as native

torch.set_num_threads(1)
FORMAT='action-return-readout-v1'
OUT=ROOT/'data/action-return/readout-v1'
PLAN=OUT/'protocol.json'
ARMS=('native','non_input24','all80','public_obs56')
SOURCE=Path(__file__).resolve()

def prepare():
    if PLAN.exists(): raise FileExistsError('immutable readout protocol already exists')
    original=old.check_protocol(old.PROTOCOL)
    verified=old.read(old.RAW/'verification.json')
    sources={}
    for seed in range(3):
        folder=old.RAW/'training'/f'seed_{seed:03d}'/'begin_quarter'
        terminal=old.read(folder/'terminal.json')
        cp=Path(terminal['checkpoint'])
        if terminal['tick']!=147456 or old.sha(cp)!=terminal['checkpoint_sha256']: raise ValueError('wrong endpoint')
        audits=sorted((folder/'audit').glob('*.pt.gz'))
        if len(audits)!=128: raise ValueError('missing original records')
        for file in audits:
            if old.sha(file)!=verified['files'][str(file)]: raise ValueError('unverified record')
        sources[str(seed)]=dict(endpoint=dict(path=str(cp),sha256=old.sha(cp),semantic_hash=terminal['semantic_end']),
            start=next(s for s in original['starts'] if s['seed']==seed),
            audits=[dict(path=str(f),sha256=old.sha(f)) for f in audits[:64]],
            native_evaluation=dict(path=str(folder/'evaluation.json'),sha256=old.sha(folder/'evaluation.json')))
    plan=dict(format=FORMAT,created_utc=datetime.now(timezone.utc).isoformat(),
        status='fixed_before_fitting_or_new_autonomous_behavior',sources=sources,
        original_protocol=dict(path=str(old.PROTOCOL),sha256=old.sha(old.PROTOCOL)),
        original_source_pins=original['source_pins'],source=dict(path=str(SOURCE),sha256=old.sha(SOURCE)),
        prompt='Can readable version information in frozen ports state drive autonomous BEGIN? Fit causal small action decoder, no oracle/gating/private features at inference, no reset.',
        seeds=[0,1,2],arms=list(ARMS),conditions=['short','long_both'],families=[0,1,2,3],steps=2048,
        source_rule='Assisted ports/undirected/recorded512/fixed/warmup complete endpoints147456; actual inherited endpoint h and world at every free fork.',
        train=dict(updates='original assisted offsets0..63 inclusive',actors=[0,1,2,3],positions='all live tick/job positions, including WAIT/FINISH',
            features='Endpoint-frozen backbone replay over original public observations and actual-action receipts. Initial h is original CP131072 h; continuously replay all8 actors without resets. These features differ from original historical h under changing weights.',
            labels='original midpoint teacher action; collapse1->BEGIN1,2->BEGIN2,everything else->OTHER',
            weights='original eleven-class weights before collapse: teacher WAIT0=.1, every nonWAIT=1; normalize weighted CE by sum of original row weights',
            fitting='separate seed/job affine reference-class softmax, OTHER logit fixed0; train-only population mean/std; zero-variance columns fixed0; squared L2=.001 on standardized slopes only',
            optimizer=dict(name='LBFGS',lr=1,max_iter=120,max_eval=180,history_size=30,line_search_fn='strong_wolfe',tolerance_grad=1e-10,tolerance_change=1e-12),
            initialization='all decoder coefficients/bias zero; deterministic; no validation or hyperparameter search',
            dimensions=dict(non_input24=24,all80=80,public_obs56=56),
            free_parameters_total_both_jobs=dict(non_input24=100,all80=324,public_obs56=228)),
        policy='For each job, m=max(native WAIT and8 FINISH logits). Replace only BEGINk with m + decoder_logit(BEGINk)-decoder_logit(OTHER). Leave WAIT/FINISH logits unchanged. Eleven-way argmax. Inference features only public current obs or post-step h; no world/scorer labels.',
        test=dict(primary_actor_positions=[4,5,6,7],overlap_diagnostic=[0,1,2,3],
            scope='full8-actor worlds retained; actors4..7 held out from decoder fitting only (all8 were seen by original backbone training); report0..3 as decoder-training-lane overlap; future request identity sets disjoint from decoder training required',
            baseline='rerun unchanged native evaluator with read-only scorer shadow; entire result must exactly equal saved original endpoint evaluation',
            frozen='same actual endpoint model/world/h; no model learning, resets, cue suppression, state substitution, forced target action or oracle gate'),
        metrics=dict(primary='Decoder-fitting-heldout matured primary requests with accepted autonomous BEGIN / matured primary requests, plus feasible-correct accepted BEGIN subset; strict and ordinary completion on same primary request identities',
            timing='all-position teacher-BEGIN sensitivity and begin precision, wrong-version begins, duplicate begins; conditional diagnostic only',
            idle='inactive enabled job opportunities and false BEGIN per1000; all inactive actions separately',
            loop='own-trajectory shadow midpoint FINISH label positions, legal finish opportunities, FINISH attempts, legally timed attempts, accepted FINISH and matured completions',
            case_matching='same primary matured request fingerprints between all four policies; no counterfactual oracle trajectories',
            aggregation='sum four family counts within each seed/condition/group; three independent seeds, descriptive seed SD only'),
        limitations=['Externally supervised affine action controller; no claim native ports learned routing.','All80 and public_obs56 have greater feature dimension and parameter count than non_input24.','Replay training receipts arose under original changing-backbone assisted policy; autonomous evaluation distribution may differ.','No FINISH decoder is trained; OTHER class includes FINISH and native FINISH scores remain available.'])
    OUT.mkdir(parents=True,exist_ok=True);old.write(PLAN,plan);return plan

def check():
    p=old.read(PLAN)
    if p['format']!=FORMAT or old.sha(SOURCE)!=p['source']['sha256']: raise ValueError('readout source/plan changed')
    if old.sha(p['original_protocol']['path'])!=p['original_protocol']['sha256']: raise ValueError('old protocol changed')
    for path,sha in p['original_source_pins'].items():
        if old.sha(path)!=sha: raise ValueError('frozen original source changed')
    for src in p['sources'].values():
        for item in [src['endpoint'],src['start'],src['native_evaluation'],*src['audits']]:
            if old.sha(item['path'])!=item['sha256']: raise ValueError('source input bytes changed')
    return p

def fit(x,original_labels,plan):
    x=x.double();labels=torch.where(original_labels.le(2),original_labels,0).long()
    mean=x.mean(0);scale=(x-mean).square().mean(0).sqrt();keep=scale.ne(0)
    z=(x[:,keep]-mean[keep])/scale[keep]
    coeff=nn.Parameter(torch.zeros((2,z.shape[1]),dtype=torch.float64));bias=nn.Parameter(torch.zeros(2,dtype=torch.float64))
    weights=torch.ones(len(original_labels),dtype=torch.float64);weights[original_labels.eq(0)]=.1
    opts=plan['train']['optimizer'];optimizer=torch.optim.LBFGS([coeff,bias],**{k:v for k,v in opts.items() if k!='name'})
    evaluations=0
    def closure():
        nonlocal evaluations
        optimizer.zero_grad();raw=z@coeff.T+bias
        logits=torch.cat((torch.zeros((len(z),1),dtype=torch.float64),raw),1)
        loss=(weights*F.cross_entropy(logits,labels,reduction='none')).sum()/weights.sum()+.001*coeff.square().sum()
        loss.backward();evaluations+=1;return loss
    initial=float(closure().detach());optimizer.step(closure);final=float(closure().detach())
    full=torch.zeros((2,x.shape[1]),dtype=torch.float64);full[:,keep]=coeff.detach()
    fit=dict(mean=mean,std=torch.where(keep,scale,torch.ones_like(scale)),coeff=full,bias=bias.detach().clone())
    info=dict(positions=len(x),class_counts=torch.bincount(labels,minlength=3).tolist(),original_action_counts=torch.bincount(original_labels,minlength=11).tolist(),
        initial_objective=initial,final_objective=final,closure_evaluations=evaluations,
        optimizer_iterations=int(optimizer.state[coeff].get('n_iter',0)),max_abs_gradient=max(float(v.grad.abs().max()) for v in (coeff,bias) if v.numel()),
        mean=mean.tolist(),std=scale.tolist(),zero_variance_columns=(~keep).nonzero().flatten().tolist(),coeff=full.tolist(),bias=bias.detach().tolist())
    return fit,info

class BeginController(nn.Module):
    def __init__(self,base,arm,fits,non_input):
        super().__init__();self.base=deepcopy(base);self.arm=arm
        self.width,self.input_size,self.n_jobs,self.n_actions=base.width,base.input_size,base.n_jobs,base.n_actions
        self.register_buffer('indices',torch.tensor(non_input,dtype=torch.long))
        for job,item in enumerate(fits):
            for key,value in item.items(): self.register_buffer(f'j{job}_{key}',value.clone())
    def step(self,obs,state,dt,max_step=None):
        native_logits,state=self.base.step(obs,state,dt,max_step)
        x=obs.double() if self.arm=='public_obs56' else state if self.arm=='all80' else state.index_select(-1,self.indices)
        logits=native_logits.clone();m=torch.cat((native_logits[...,:1],native_logits[...,3:]),-1).amax(-1)
        for job in range(self.n_jobs):
            z=(x-getattr(self,f'j{job}_mean'))/getattr(self,f'j{job}_std')
            offsets=z@getattr(self,f'j{job}_coeff').T+getattr(self,f'j{job}_bias')
            logits[...,job,1:3]=m[...,job,None]+offsets
        return logits,state
    def config(self):
        return dict(kind='external_frozen_affine_begin_controller',arm=self.arm,base=self.base.config(),
            preserves_native_nonbegin_scores=True,backbone_state_flow_unchanged=True)

def extract(seed,plan,endpoint):
    src=plan['sources'][str(seed)]
    original=native.NativeTrainer.from_checkpoint(src['start']['path'],device='cpu')
    if state_digest(original.semantic_state())!=src['start']['semantic_hash']: raise ValueError('wrong full initial state')
    world=original.world;h=original.h.clone();frozen_h=h.clone();frozen=deepcopy(endpoint.model)
    non_input=[i for i in range(80) if i not in endpoint.model.input_ports.tolist()]
    rows={job:dict(h=[],obs=[],label=[],request=[]) for job in range(2)}
    cases=list(world.case_ids)
    for offset,item in enumerate(src['audits']):
        audit=torch.load(io.BytesIO(gzip.decompress(Path(item['path']).read_bytes())),map_location='cpu',weights_only=False)
        record=audit['record']
        if not torch.equal(h,record['start_h']): raise AssertionError('actual original start_h differs')
        original.model.load_state_dict({**record['parameters'],**record['buffers']},strict=True)
        with torch.no_grad():
            for t,(obs,elapsed) in enumerate(zip(record['observations'],record['elapsed'])):
                # observe is the sole original scheduler advance; labels follow it.
                replay_obs=world.observe().to(dtype=torch.float64);labels=world.oracle('midpoint')
                if not torch.equal(replay_obs,obs) or not torch.equal(labels,record['targets'][t]): raise AssertionError('historical observations/labels differ')
                logits,h=original.model.step(obs,h,elapsed);_,frozen_h=frozen.step(obs,frozen_h,elapsed)
                if not torch.equal(logits.argmax(-1),audit['actions']['model_actions'][t]): raise AssertionError('historical native actions differ')
                for job in range(2):
                    rows[job]['h'].append(frozen_h[:4].clone());rows[job]['obs'].append(obs[:4].clone());rows[job]['label'].append(labels[:4,job].clone())
                    rows[job]['request'].extend((cases[i],job,int(world.request_id[i,job])) for i in range(4) if int(world.request_id[i,job])>=0)
                world.act(audit['actions']['actual_actions'][t])
        if not torch.equal(h,record['end_h']) or state_digest(world.state_dict())!=audit['trace']['live_world_hash']: raise AssertionError('historical actual endpoint differs')
    for row in rows.values():
        for key in ('h','obs','label'): row[key]=torch.cat(row[key])
        row['request']=sorted(set(row['request']))
    return rows,non_input,dict(blocks=64,actual_h_observations_labels_actions_and_world_endpoints_exact=True,
        frozen_feature_start_h_hash=state_digest(original.h),frozen_feature_end_h_hash=state_digest(frozen_h),
        frozen_feature_rule='one continuous endpoint-frozen replay; original receipts, no hidden resets; not historical actual h',
        train_actors=list(world.actor_ids[:4]),test_actors=list(world.actor_ids[4:]),non_input_indices=non_input)

class Shadow:
    """Read-only actual-trajectory scorer, split by actor lane, never a policy input."""
    def __init__(self,source,train_requests):
        self.fork_tick=source.tick;self.regime=source.regime_id+1;self.cases=source.case_ids
        self.train_requests=set(train_requests);self.seen=set();self.began=set();self.feasible_began=set();self.matured={g:[] for g in ('heldout','overlap','all')}
        self.counts={g:Counter() for g in self.matured};self.original=ContextWorld.act
    def key(self,world,i,j,request=None): return (self.cases[i],j,int(world.request_id[i,j]) if request is None else int(request))
    def act(self,world,actions):
        labels=world.oracle('midpoint');begin=actions.gt(0)&actions.lt(3);finish=actions.ge(3)
        enabled=torch.zeros_like(world.active);enabled[:,:world.active_jobs]=True
        active=world.active&enabled;legal=active&torch.as_tensor(_phases(world)==3)
        # Context current_regime_id survives until next request; matured regime is
        # tracked explicitly below using first-observed request metadata instead.
        for i in range(world.batch_size):
            group='overlap' if i<4 else 'heldout'
            for j in range(world.n_jobs):
                key=self.key(world,i,j)
                if int(world.event_kind[i,j]) in (NEW,REPLACE) and key[2]>=0:
                    if int(world.event_regime_id[i,j])==self.regime:
                        if key in self.train_requests: raise AssertionError('train/test future request leakage')
                        self.seen.add(key)
                if bool(world.matured_mask[i,j]):
                    mk=self.key(world,i,j,world.matured_request_id[i,j])
                    if mk in self.seen:
                        for g in (group,'all'):
                            c=self.counts[g];c['primary_matured']+=1;c['primary_began']+=int(mk in self.began)
                            c['primary_feasible_began']+=int(mk in self.feasible_began)
                            c['primary_completed']+=int(world.matured_completed[i,j]);c['primary_clean_completed']+=int(world.matured_clean_completed[i,j]);self.matured[g].append(mk)
        for group,sl in (('overlap',slice(0,4)),('heldout',slice(4,8)),('all',slice(0,8))):
            c=self.counts[group];sel=lambda v:int(v[sl].sum())
            target_begin=labels.gt(0)&labels.lt(3)
            for name,mask in dict(opportunities=enabled,teacher_begin_positions=enabled&target_begin,
                teacher_finish_labels=enabled&labels.ge(3),begin_attempts=enabled&begin,
                correct_begin_at_teacher_position=enabled&begin&actions.eq(labels)&target_begin,
                any_begin_at_teacher_position=enabled&begin&target_begin,
                begins_at_nonbegin_teacher_positions=enabled&begin&~target_begin,
                wrong_version_begin=active&begin&actions.sub(1).ne(world.version),
                duplicate_begin=active&begin&world.started&actions.sub(1).eq(world.version),
                idle_opportunities=enabled&~active,idle_false_begin=enabled&~active&begin,
                idle_actions=enabled&~active&actions.ne(0),legal_finish_opportunities=legal,
                finish_attempts=enabled&finish,legal_finish_attempts=legal&finish).items(): c[name]+=sel(mask)
        result=self.original(world,actions)
        accepted=world.last_result.eq(1)
        for i,j in (accepted&begin).nonzero().tolist(): self.began.add(self.key(world,i,j))
        for i,j in (accepted&begin&actions.eq(labels)).nonzero().tolist(): self.feasible_began.add(self.key(world,i,j))
        for group,sl in (('overlap',slice(0,4)),('heldout',slice(4,8)),('all',slice(0,8))):
            self.counts[group]['accepted_begin']+=int((accepted&begin)[sl].sum());self.counts[group]['accepted_finish']+=int((accepted&finish)[sl].sum())
        return result
    def result(self):
        result={}
        for group,c in self.counts.items():
            ids=sorted(self.matured[group]);result[group]=dict(counts=dict(c),primary_request_set_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
                primary_request_count=len(ids),no_training_request_overlap=True)
        return result

def evaluate_shadow(model,trainer,*,family,condition,requests):
    shadow=Shadow(trainer.world,requests)
    def observed(world,actions): return shadow.act(world,actions)
    with _preserve_process_state(),patch.object(ContextWorld,'act',observed):
        record=evaluate(model,trainer.world,trainer.h,family=family,condition=condition,steps=2048,device='cpu')
    extra=shadow.result()
    for field,shadow_key in [('matured','primary_matured'),('completed','primary_completed'),('clean_completed','primary_clean_completed')]:
        if record['primary'][field]!=extra['all']['counts'].get(shadow_key,0): raise AssertionError('shadow primary denominator mismatch')
    if record['request_diagnosis']['primary']['began']!=extra['all']['counts'].get('primary_began',0): raise AssertionError('shadow begins mismatch')
    return dict(record=record,shadow=extra)

def run(seed):
    p=check();out=OUT/f'seed_{seed:03d}';out.mkdir(exist_ok=False)
    started=time.perf_counter();src=p['sources'][str(seed)]
    trainer=native.NativeTrainer.from_checkpoint(src['endpoint']['path'],device='cpu')
    initial=state_digest(trainer.semantic_state())
    if initial!=src['endpoint']['semantic_hash'] or trainer.ticks!=147456 or trainer.config.wait_weight!=.1: raise ValueError('endpoint semantics differ')
    data,non_input,reconstruction=extract(seed,p,trainer);fits={};fitinfo={}
    buffer=io.BytesIO();torch.save(dict(groups=data,non_input=non_input,reconstruction=reconstruction),buffer)
    (out/'train_dataset.pt.gz').write_bytes(gzip.compress(buffer.getvalue(),compresslevel=1,mtime=0))
    for arm in ARMS[1:]:
        fits[arm]=[];fitinfo[arm]=[]
        for job in range(2):
            row=data[job];x=row['obs'] if arm=='public_obs56' else row['h'] if arm=='all80' else row['h'][:,non_input]
            value,info=fit(x,row['label'],p);fits[arm].append(value);fitinfo[arm].append(info)
    requests=[k for row in data.values() for k in row['request']]
    torch.save(dict(fits=fits,reconstruction=reconstruction,training_request_sets={j:data[j]['request'] for j in data}),out/'fits.pt')
    old.write(out/'fit.json',dict(seed=seed,protocol_sha256=old.sha(PLAN),fits_sha256=old.sha(out/'fits.pt'),train_dataset_sha256=old.sha(out/'train_dataset.pt.gz'),fit=fitinfo,reconstruction=reconstruction,
        training_positions_per_job={j:len(data[j]['label']) for j in data}))
    del data
    archive=old.read(src['native_evaluation']['path'])['records'];results=[]
    for arm in ARMS:
        model=trainer.model if arm=='native' else BeginController(trainer.model,arm,fits[arm],non_input)
        for condition in p['conditions']:
            for family in p['families']:
                value=evaluate_shadow(model,trainer,family=family,condition=condition,requests=requests)
                if arm=='native':
                    original=next(r for r in archive if r['family']==family and r['condition']==condition)
                    if json.loads(json.dumps(value['record'],allow_nan=False))!=original: raise AssertionError('native shadow rerun differs from original saved evaluator')
                else:
                    base=next(r for r in results if r['arm']=='native' and r['family']==family and r['condition']==condition)
                    for field in ('start_world_state_hash','start_hidden_hash','end_scheduler_hash','end_event_hash','actor_ids','case_ids'):
                        if value['record'][field]!=base['record'][field]: raise AssertionError(f'unmatched fork {field}')
                    for group in value['shadow']:
                        if value['shadow'][group]['primary_request_set_sha256']!=base['shadow'][group]['primary_request_set_sha256']: raise AssertionError('matured cases differ')
                value.update(arm=arm,condition=condition,family=family);results.append(value)
                old.write(out/'partial.json',dict(seed=seed,records=results,seconds=time.perf_counter()-started))
                print(json.dumps(dict(seed=seed,arm=arm,condition=condition,family=family,done=len(results),seconds=time.perf_counter()-started)),flush=True)
    if state_digest(trainer.semantic_state())!=initial: raise AssertionError('source complete state changed')
    check();result=dict(format=FORMAT,seed=seed,protocol_sha256=old.sha(PLAN),source_unchanged=True,native_exact_saved_forks=8,
        fits_sha256=old.sha(out/'fits.pt'),fit_json_sha256=old.sha(out/'fit.json'),records=results,seconds=time.perf_counter()-started)
    old.write(out/'result.json',result);return result

def summarize():
    p=check();rows=[]
    for seed in p['seeds']:
        value=old.read(OUT/f'seed_{seed:03d}'/'result.json')
        if value['protocol_sha256']!=old.sha(PLAN) or len(value['records'])!=32: raise ValueError('incomplete behavior')
        for arm in ARMS:
            for condition in p['conditions']:
                records=[r for r in value['records'] if r['arm']==arm and r['condition']==condition]
                for group in ('heldout','overlap','all'):
                    c=Counter()
                    for r in records:c.update(r['shadow'][group]['counts'])
                    def rate(n,d,m=1): return m*c[n]/c[d] if c[d] else None
                    rows.append(dict(seed=seed,arm=arm,condition=condition,actor_group=group,counts=dict(c),
                        began_matured_request_rate=rate('primary_began','primary_matured'),ordinary_completion_rate=rate('primary_completed','primary_matured'),
                        feasible_began_matured_request_rate=rate('primary_feasible_began','primary_matured'),
                        strict_completion_rate=rate('primary_clean_completed','primary_matured'),idle_false_begin_per1000=rate('idle_false_begin','idle_opportunities',1000),
                        correct_begin_sensitivity=rate('correct_begin_at_teacher_position','teacher_begin_positions'),
                        any_begin_sensitivity=rate('any_begin_at_teacher_position','teacher_begin_positions'),begin_teacher_precision=rate('correct_begin_at_teacher_position','begin_attempts')))
    result=dict(format=FORMAT,protocol_sha256=old.sha(PLAN),rows=rows,
        independent_seeds=3,source_identity='same canonical assisted endpoint per seed, frozen backbone, externally trained action-head intervention',
        verification='exactly8 saved native evaluation forks per seed; all action heads matched same cases, original sources and full source state unchanged')
    old.write(OUT/'summary.json',result)
    import csv
    flat=[{k:v for k,v in r.items() if k!='counts'}|r['counts'] for r in rows]
    with (OUT/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in flat))));writer.writeheader();writer.writerows(flat)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','run','summary'));parser.add_argument('--seed',type=int,choices=(0,1,2));args=parser.parse_args()
    result=prepare() if args.command=='prepare' else run(args.seed) if args.command=='run' else summarize()
    print(json.dumps(dict(format=result['format'],output=str(OUT))),flush=True)
if __name__=='__main__':main()
