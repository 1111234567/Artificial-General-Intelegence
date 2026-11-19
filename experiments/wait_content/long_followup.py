"""One fixed longer future continuation; original2048 remains the primary test.

The protocol is frozen before reading any trained-arm behavioral score. Reuse
the exact8-lane model/warmup/world, execute its actual11-way argmax8192 times,
and count only newly issued, matured requests after the original2048 prefix.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
from experiments.wait_content import train as tr, eval as ev
from experiments.lifelong.world import ContextWorld, NEW, REPLACE
from experiments.lifelong.evaluate import CONDITIONS, TRACE_HASH_VERSION, _phases

ROOT=tr.ROOT
OUT=ROOT/'data/wait-content/long-v1'
PLAN=OUT/'protocol.json'
ARMS=ev.ARMS
PREFIX=2048
STEPS=8192
FORMAT='wait-content-fixed-long-followup-v1'
PRIVATE=('request_id','active','started','version','payload','raw_payload','current_family',
    'duration','deadline','ready_time','issued_time','current_eligible','current_regime_id',
    'event_kind','event_regime_id','event_eligible','matured_mask','matured_request_id',
    'matured_completed','matured_clean_completed')
EXTERNAL=('request_id','version','payload','raw_payload','current_family','duration','deadline',
    'issued_time','current_eligible','current_regime_id','event_kind','event_regime_id',
    'event_eligible','matured_mask','matured_request_id')


def write(path,value):
    ev.write(path,value)


def read(path):
    return ev.read(path)


def sha(path):
    return ev.sha(path)


def digest_arrays(data,names):
    digest=hashlib.sha256()
    for name in names:
        array=np.ascontiguousarray(data[name])
        digest.update(name.encode());digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def action_hash(actions):
    digest=hashlib.sha256(f'{TRACE_HASH_VERSION}/actions'.encode())
    digest.update(np.ascontiguousarray(actions,dtype=np.int64).tobytes())
    return digest.hexdigest()


def behavior_rates(counts):
    measured=('ordinary_completion_rate','strict_completion_rate','feasible_begin_rate',
              'idle_actions_per1000','idle_begin_per1000','idle_finish_per1000',
              'processing_early_finish_rate','legal_finish_attempt_rate','legal_finish_content_accuracy')
    values=ev.rates(counts)
    return {key:values[key] for key in measured}


def prepare():
    if PLAN.exists():
        raise FileExistsError('immutable long followup protocol exists')
    train_plan=tr.checked();eval_plan=ev.checked()
    source_paths=[Path(__file__).resolve(),Path(tr.__file__).resolve(),Path(ev.__file__).resolve(),
        ROOT/'experiments/action_return/finish_readout.py',ROOT/'experiments/action_return/readout.py',
        ROOT/'experiments/lifelong/evaluate.py',ROOT/'experiments/lifelong/world.py',
        ROOT/'experiments/continuity/world.py',ROOT/'experiments/undirected/model.py']
    inputs={str(tr.PLAN):sha(tr.PLAN),str(ev.PLAN):sha(ev.PLAN)}
    if (ev.OUT/'implementation-amendment.json').exists():
        inputs[str(ev.OUT/'implementation-amendment.json')]=sha(ev.OUT/'implementation-amendment.json')
    for seed in range(3):
        for name in ('start.pt','endpoint.pt','manifest.json','requests.json'):
            p=tr.DATA/f'seed_{seed:03d}'/name;inputs[str(p)]=sha(p)
        for arm in ARMS:
            p=ev.OUT/f'seed_{seed:03d}'/f'{arm}_warmup.pt';inputs[str(p)]=sha(p)
            if arm!='untrained_anchor':
                p=tr.OUT/f'seed_{seed:03d}'/arm/'checkpoint.pt';inputs[str(p)]=sha(p)
        for item in tr.source_info(seed).values():
            inputs[item['path']]=sha(item['path'])
    value=dict(format=FORMAT,created_utc=datetime.now(timezone.utc).isoformat(),
        declaration='This final fixed behavioral supplement was chosen before this worker read any trained-arm behavioral scores. Only training losses and budget checks were seen. Parent supplied the anchor low-base-rate fact: one ordinary completion among984 long requests. All36 continuations will run to8192 irrespective of new trained scores; no further training or optional stopping.',
        primary_test='Original2048-tick evaluation remains the primary report. This8192 continuation is a separately labeled low-base-rate sensitivity supplement, not a replacement test or a new seed.',
        source_files={str(p):sha(p) for p in source_paths},input_files=inputs,
        train_protocol_hash=train_plan['protocol_hash'],eval_protocol_hash=eval_plan['protocol_hash'],
        seeds=[0,1,2],arms=list(ARMS),families=[0,1,2,3],condition='long_both',steps=STEPS,prefix=PREFIX,forks=36,
        initialization='Identical real data-v1 endpoint world/scheduler/clock/pending contracts/receipts in every arm. Reuse each model-specific existing formal warmup.pt, binding its bytes, model fingerprint and common start_hidden_hash. Anchor warmed h must equal collector endpoint h. Never use fit-lane h or zero state.',
        trajectory='Exact original evaluator setup: clone full8-lane world, set_family, set_regime long_both with original evaluation name and dt; preserve pending contracts. Every tick observe original public56-vector, unmodified model.step with complete continuous h, execute11-action argmax, receive actual original world act feedback. No resets, private gates, teacher actions or updates. Scalar dt and original8-lane batch retained.',
        prefix_verification='Persist first2048 actual action hash using the original evaluator hash algorithm,2048 endpoint world/scheduler/event/h hashes, and all-group matured identities/completion/strict/began/feasible/idle/accepted/legal counts. After fixed collection, compare every36 prefix with formal eval-v1 counterpart; failed equality invalidates followup until explained.',
        suffix_requests='Offsets0..2047 are prefix. Suffix request cohort requires eligible NEW/REPLACE under evaluation contract, issue_offset>=2048 and actual maturity observed before offset8192. Exclude requests issued before2048 but maturing later, and suffix-issued requests still censored at8192. Do not select on BEGIN, reaching legal window, attempts or outcome. Report excluded crossing/censored counts.',
        suffix_idle='Count all actually inactive enabled opportunities at offsets2048..8191 and actual nonWAIT actions, with separate BEGIN/FINISH; this opportunity denominator depends on policy and is not conditioned on request maturity.',
        metrics='Per seed sum4families: suffix-cohort ordinary/strict completion and accepted feasible BEGIN; all-suffix actual idle numerator/denominator; legal opportunities/correct/wrong attempts both for suffix matured cohort and all suffix actual opportunities. Preserve raw counts and all actor groups; heldout lanes4..7 primary.',
        pairing='Within each seed/family all3arms must share identical external command/maturity streams and suffix matured request identities. Report each seed wait_mixed-minus-end_only and mean/sampleSD across3seeds;36forks are not independent seeds.',
        persistence='Perfork compact npz actual actions/results/private scoring facts including complete external stream, compact request records, prefix check evidence, original source/model fingerprints and own warm h. Scoring performed after actual actions; no scorer values enter model.',
        runtime=dict(workers=3,threads_per_worker=1,dtype='float64',device='cpu',warmup_reuse='one per seed/arm;4families sequentially'),
        limits=['Same future continuations extend the primary test; they are not independent confirmation or extra training.',
                'One model-specific same-public-history warm start and one fixed training budget; no claim of native self-directed learning.',
                'Three paired seeds and rare successes; more requests narrow case dependence but do not create more independent seeds.',
                'Heldout lanes4..7 were present in historical backbone training.'])
    value=ev.normalized(value);value['protocol_hash']=ev.state_digest(value)
    OUT.mkdir(parents=True,exist_ok=True);write(PLAN,value)
    return value


def checked():
    plan=read(PLAN)
    if plan['protocol_hash']!=ev.state_digest({k:v for k,v in plan.items() if k!='protocol_hash'}):
        raise ValueError('long followup protocol hash')
    replacements={}
    amendment_path=OUT/'implementation-amendment.json'
    if amendment_path.exists():
        amendment=read(amendment_path)
        if amendment['protocol_hash']!=plan['protocol_hash']:
            raise ValueError('foreign followup implementation amendment')
        for item in amendment['changed_sources']:
            if item['original_sha256']!=plan['source_files'][item['path']] or sha(item['original_snapshot'])!=item['original_sha256']:
                raise ValueError('original frozen implementation not preserved')
            replacements[item['path']]=item['amended_sha256']
    for group in ('source_files','input_files'):
        for path,expected in plan[group].items():
            expected=replacements.get(path,expected) if group=='source_files' else expected
            if sha(path)!=expected:
                raise ValueError('bound followup source/input changed '+path)
    return plan


def request_rows(data,case_ids,actor_ids,regime):
    """Reconstruct primary issuance/maturity from saved facts, independent of action."""
    command=np.isin(data['event_kind'],[NEW,REPLACE]) & data['event_eligible'] & (data['event_regime_id']==regime)
    rows={}
    for tick,lane,job in zip(*np.where(command)):
        tick,lane,job=map(int,(tick,lane,job))
        rid=int(data['request_id'][tick,lane,job]);key=(lane,job,rid)
        if key in rows:
            raise ValueError('duplicate issued identity')
        rows[key]=dict(case_id=case_ids[lane],actor_id=actor_ids[lane],lane=lane,job=job,request_id=rid,
            issued_offset=int(tick),issued_tick=int(data['tick'][tick]),issued_time=float(data['issued_time'][tick,lane,job]),
            duration=float(data['duration'][tick,lane,job]),deadline=float(data['deadline'][tick,lane,job]),
            family=int(data['current_family'][tick,lane,job]),version=int(data['version'][tick,lane,job]),
            mapped_payload=int(data['payload'][tick,lane,job]),raw_payload=int(data['raw_payload'][tick,lane,job]),
            matured=False,matured_offset=None,completed=None,clean_completed=None,
            accepted_begin_offset=None,feasible_begin=False)
    for tick,lane,job in zip(*np.where(data['matured_mask'])):
        row=rows.get((lane,job,int(data['matured_request_id'][tick,lane,job])))
        if row is None:
            continue
        if row['matured']:
            raise ValueError('identity matured twice')
        row.update(matured=True,matured_offset=int(tick),completed=bool(data['matured_completed'][tick,lane,job]),
                   clean_completed=bool(data['matured_clean_completed'][tick,lane,job]))
    begin=np.isin(data['actions'],[1,2]) & (data['result']==1)
    for tick,lane,job in zip(*np.where(begin)):
        row=rows.get((lane,job,int(data['request_id'][tick,lane,job])))
        if row is None:
            continue
        if row['accepted_begin_offset'] is None:
            row['accepted_begin_offset']=int(tick)
        row['feasible_begin'] |= bool(data['actions'][tick,lane,job]==data['teacher_actions'][tick,lane,job])
    return sorted(rows.values(),key=lambda r:(r['lane'],r['job'],r['request_id']))


def compute(data,rows,start,stop,*,suffix=False):
    output={}
    for group,lanes in (('heldout',range(4,8)),('overlap',range(4)),('all',range(8))):
        selected=[r for r in rows if r['lane'] in lanes and r['matured'] and r['matured_offset']<stop and
                  (r['issued_offset']>=start if suffix else r['matured_offset']>=start)]
        subset=np.zeros_like(data['active'][start:stop])
        for lane in lanes:
            for job in range(2):
                ids=[r['request_id'] for r in selected if r['lane']==lane and r['job']==job]
                subset[:,lane,job]=np.isin(data['request_id'][start:stop,lane,job],ids)
        enabled=data['enabled'][start:stop].copy()
        outside=[lane for lane in range(8) if lane not in lanes];enabled[:,outside]=False
        active=data['active'][start:stop]&enabled
        idle=enabled&~active
        actions=data['actions'][start:stop];result=data['result'][start:stop]
        begin=np.isin(actions,[1,2]);finish=actions>=3
        legal=active&(data['phase'][start:stop]==3)
        processing=active&(data['phase'][start:stop]==2)
        exact=actions-3==4*data['version'][start:stop]+data['payload'][start:stop]
        c=Counter(primary_matured=len(selected),primary_completed=sum(r['completed'] for r in selected),
            primary_clean_completed=sum(r['clean_completed'] for r in selected),
            primary_began=sum(r['accepted_begin_offset'] is not None for r in selected),
            primary_feasible_began=sum(r['feasible_begin'] for r in selected))
        masks=dict(opportunities=enabled,active_opportunities=active,idle_opportunities=idle,
            idle_actions=idle&(actions!=0),idle_begin=idle&begin,idle_finish=idle&finish,
            begin_attempts=enabled&begin,finish_attempts=enabled&finish,
            accepted_begin=enabled&begin&(result==1),accepted_finish=enabled&finish&(result==1),
            legal_finish_opportunities=legal,legal_finish_attempts=legal&finish,
            legal_finish_correct=legal&finish&exact,legal_finish_wrong=legal&finish&~exact,
            processing_opportunities=processing,processing_early_finish=processing&finish,
            processing_wait=processing&(actions==0),
            cohort_legal_finish_opportunities=legal&subset,cohort_legal_finish_attempts=legal&subset&finish,
            cohort_legal_finish_correct=legal&subset&finish&exact,cohort_legal_finish_wrong=legal&subset&finish&~exact,
            cohort_processing_opportunities=processing&subset,cohort_processing_early_finish=processing&subset&finish,
            cohort_accepted_finish=active&subset&finish&(result==1))
        for name,mask in masks.items():c[name]=int(mask.sum())
        identity=[(r['case_id'],r['job'],r['request_id']) for r in selected]
        excluded_cross=[r for r in rows if r['lane'] in lanes and r['issued_offset']<start and r['matured'] and start<=r['matured_offset']<stop]
        excluded_censored=[r for r in rows if r['lane'] in lanes and start<=r['issued_offset']<stop and not r['matured']]
        output[group]=dict(counts=dict(c),primary_request_set_sha256=ev.identity_hash(identity),
            excluded_prefix_crossing_requests=len(excluded_cross) if suffix else 0,
            excluded_suffix_censored_requests=len(excluded_censored),rates=behavior_rates(c))
    return output


@torch.inference_mode()
def continuation(model,world,h,family):
    """Minimal original evaluator trajectory; all scoring is post-trajectory."""
    live=ContextWorld.from_state_dict(world.state_dict(),device='cpu')
    source=live.state_dict();source_hash=ev.state_digest(source)
    state=h.clone();model_hash=ev.state_digest(model.state_dict())
    live.set_family(family)
    live.set_regime(**CONDITIONS['long_both'],name=f'evaluation/{family}/long_both',dt=world.dt)
    regime=live.regime_id
    logs={name:[] for name in PRIVATE}
    logs.update({name:[] for name in ('tick','time','elapsed','phase','enabled','actions','result','teacher_actions')})
    enabled=torch.zeros_like(live.active);enabled[:,:live.active_jobs]=True
    prefix_endpoint=None
    for offset in range(STEPS):
        obs=live.observe().double()
        logits,state=model.step(obs,state,float(live.elapsed))
        actions=logits.argmax(-1)
        for name in PRIVATE:logs[name].append(getattr(live,name).cpu().numpy().copy())
        logs['tick'].append(live.tick);logs['time'].append(live.now);logs['elapsed'].append(float(live.elapsed))
        logs['phase'].append(_phases(live));logs['enabled'].append(enabled.numpy().copy())
        logs['actions'].append(actions.numpy().copy());logs['teacher_actions'].append(live.oracle('midpoint').numpy().copy())
        live.act(actions)
        logs['result'].append(live.last_result.numpy().copy())
        if offset+1==PREFIX:
            saved=live.state_dict()
            prefix_endpoint=dict(end_world_state_hash=ev.state_digest(saved),end_scheduler_hash=ev.state_digest(saved['scheduler']),
                end_event_hash=live.event_hash,end_hidden_hash=ev.state_digest({'hidden':state}),end_tick=live.tick)
    arrays={name:np.asarray(value) for name,value in logs.items()}
    if model_hash!=ev.state_digest(model.state_dict()) or source_hash!=ev.state_digest(world.state_dict()):
        raise ValueError('followup mutated source/model')
    rows=request_rows(arrays,world.case_ids,world.actor_ids,regime)
    prefix=compute(arrays,rows,0,PREFIX)
    suffix=compute(arrays,rows,PREFIX,STEPS,suffix=True)
    saved=live.state_dict()
    record=dict(family=family,condition='long_both',steps=STEPS,prefix_steps=PREFIX,
        fork_tick=source['tick'],fork_time=source['now'],start_world_state_hash=source_hash,
        start_scheduler_hash=ev.state_digest(source['scheduler']),start_event_hash=source['event_hash'],
        start_hidden_hash=ev.state_digest({'hidden':h}),end_world_state_hash=ev.state_digest(saved),
        end_scheduler_hash=ev.state_digest(saved['scheduler']),end_event_hash=live.event_hash,
        end_hidden_hash=ev.state_digest({'hidden':state}),end_tick=live.tick,
        action_hash=action_hash(arrays['actions']),prefix_action_hash=action_hash(arrays['actions'][:PREFIX]),
        prefix_endpoint=prefix_endpoint,prefix=prefix,suffix=suffix,
        external_stream_hash=digest_arrays(arrays,EXTERNAL),prefix_external_stream_hash=digest_arrays({k:v[:PREFIX] for k,v in arrays.items()},EXTERNAL),
        case_ids=list(world.case_ids),actor_ids=list(world.actor_ids),evaluation_regime_id=regime,
        model_hash=model_hash,world_resets=0,hidden_resets=0,parameter_updates=0)
    return record,arrays,rows


def run(seed,arm):
    torch.set_num_threads(1);plan=checked()
    folder=OUT/f'seed_{seed:03d}'/arm;folder.mkdir(parents=True,exist_ok=False)
    model,initial=tr.load_initial(seed) if arm=='untrained_anchor' else tr.load_trained(seed,arm)
    model.eval();model_hash=ev.state_digest(model.state_dict())
    data=tr.DATA/f'seed_{seed:03d}'
    start=torch.load(data/'start.pt',map_location='cpu',weights_only=False)
    endpoint=torch.load(data/'endpoint.pt',map_location='cpu',weights_only=False)
    warmed=torch.load(ev.OUT/f'seed_{seed:03d}'/f'{arm}_warmup.pt',map_location='cpu',weights_only=False)
    if warmed['model_hash']!=model_hash or warmed['protocol_hash']!=plan['eval_protocol_hash']:
        raise ValueError('warmup model/evaluation fingerprint')
    if warmed['warmup']['start_hidden_hash']!=ev.state_digest({'hidden':start['h']}):
        raise ValueError('warmup original inherited h differs')
    h=warmed['h'].clone()
    if arm=='untrained_anchor' and not torch.equal(h,endpoint['h']):
        raise ValueError('anchor warmup h not collector endpoint')
    world=ContextWorld.from_state_dict(endpoint['world'],device='cpu')
    started=time.perf_counter();records=[]
    for family in range(4):
        record,arrays,rows=continuation(model,world,h,family)
        stem=f'family_{family}'
        npz=folder/(stem+'.facts.npz');np.savez_compressed(npz,**arrays)
        req=folder/(stem+'.requests.json.gz');ev.compressed_json(req,rows)
        record.update(arm=arm,seed=seed,facts_artifact=dict(path=str(npz),sha256=sha(npz)),
                      requests_artifact=dict(path=str(req),sha256=sha(req)))
        records.append(record)
        write(folder/'partial.json',dict(seed=seed,arm=arm,records=records,protocol_hash=plan['protocol_hash']))
        print(seed,arm,'family',family,'seconds',round(time.perf_counter()-started,1),flush=True)
        del arrays,rows
    checked()
    result=dict(format=FORMAT,seed=seed,arm=arm,protocol_hash=plan['protocol_hash'],records=records,
                seconds=time.perf_counter()-started,model_hash=model_hash,warmup_sha256=sha(ev.OUT/f'seed_{seed:03d}'/f'{arm}_warmup.pt'))
    result=ev.normalized(result);result['result_hash']=ev.state_digest(result);write(folder/'result.json',result)
    return result


def compare_prefix(records):
    """Read formal behavioral output only after fixed followup collection."""
    results={seed:read(ev.OUT/f'seed_{seed:03d}'/'result.json') for seed in range(3)}
    checks=[]
    for record in records:
        seed=record['seed'];arm=record['arm'];family=record['family']
        candidates=[v for v in results[seed]['records'] if (v['arm'],v['condition'],v['family'])==(arm,'long_both',family)]
        if len(candidates)!=1:raise ValueError('need one original formal prefix')
        original=candidates[0]
        if record['prefix_action_hash']!=original['record']['action_hash']:
            raise ValueError(f'prefix action mismatch {seed} {arm} {family}')
        for key,actual in record['prefix_endpoint'].items():
            if actual!=original['record'][key]:
                raise ValueError(f'prefix endpoint mismatch {seed} {arm} {family} {key}')
        keys=('primary_matured','primary_completed','primary_clean_completed','primary_began','primary_feasible_began',
              'opportunities','active_opportunities','idle_opportunities','idle_actions','idle_begin','idle_finish',
              'begin_attempts','finish_attempts','accepted_begin','accepted_finish','legal_finish_opportunities',
              'legal_finish_attempts','legal_finish_correct','legal_finish_wrong','processing_opportunities',
              'processing_early_finish','processing_wait')
        for group in ev.GROUPS:
            left=Counter(record['prefix'][group]['counts']);right=Counter(original['shadow'][group]['counts'])
            for key in keys:
                if left[key]!=right[key]:
                    raise ValueError(f'prefix count mismatch {seed} {arm} {family} {group} {key}: {left[key]} vs {right[key]}')
            if record['prefix'][group]['primary_request_set_sha256']!=original['shadow'][group]['primary_request_set_sha256']:
                raise ValueError('prefix request identity mismatch')
        checks.append(dict(seed=seed,arm=arm,family=family,action_hash_exact=True,
            full_world_scheduler_event_hidden_exact=True,all_groups_request_completion_idle_legal_counts_exact=True))
    return dict(passed=True,forks=len(checks),records=checks,
                formal_result_inputs={str(ev.OUT/f'seed_{s:03d}'/'result.json'):sha(ev.OUT/f'seed_{s:03d}'/'result.json') for s in range(3)})


def summarize():
    plan=checked();records=[];inputs={}
    for seed in range(3):
        for arm in ARMS:
            path=OUT/f'seed_{seed:03d}'/arm/'result.json';value=read(path);inputs[str(path)]=sha(path)
            if value['protocol_hash']!=plan['protocol_hash'] or value['result_hash']!=ev.state_digest({k:v for k,v in value.items() if k!='result_hash'}):
                raise ValueError('followup result digest')
            records.extend(value['records'])
    prefix=compare_prefix(records);write(OUT/'prefix_verification.json',prefix)
    for seed in range(3):
        for family in range(4):
            same=[r for r in records if r['seed']==seed and r['family']==family]
            if len(same)!=3:raise ValueError('need three arms')
            for value in same[1:]:
                for key in ('external_stream_hash','prefix_external_stream_hash','start_world_state_hash','start_scheduler_hash','end_scheduler_hash','start_event_hash','end_event_hash'):
                    if value[key]!=same[0][key]:raise ValueError('unmatched full external stream '+key)
                for group in ev.GROUPS:
                    if value['suffix'][group]['primary_request_set_sha256']!=same[0]['suffix'][group]['primary_request_set_sha256']:
                        raise ValueError('unmatched suffix request identities')
    per_seed=[];paired=[]
    for seed in range(3):
        arm_rates={}
        for arm in ARMS:
            counts=Counter();cross=censored=0
            for r in records:
                if r['seed']==seed and r['arm']==arm:
                    group=r['suffix']['heldout'];counts.update(group['counts'])
                    cross+=group['excluded_prefix_crossing_requests'];censored+=group['excluded_suffix_censored_requests']
            rates=behavior_rates(counts);arm_rates[arm]=rates
            per_seed.append(dict(seed=seed,arm=arm,counts=dict(counts),rates=rates,
                excluded_prefix_crossing_requests=cross,excluded_suffix_censored_requests=censored))
        a,z=arm_rates['wait_mixed'],arm_rates['end_only']
        paired.append(dict(seed=seed,delta={k:a[k]-z[k] if a[k] is not None and z[k] is not None else None for k in a}))
    aggregate={}
    for key in paired[0]['delta']:
        values=[r['delta'][key] for r in paired if r['delta'][key] is not None]
        aggregate[key]=dict(mean=statistics.mean(values) if values else None,sample_sd=statistics.stdev(values) if len(values)>1 else None,seeds=len(values))
    result=dict(format=FORMAT,protocol_hash=plan['protocol_hash'],prefix_verification_passed=True,
        full_external_streams_matched=True,suffix_requests_matched=True,heldout=per_seed,paired=paired,
        paired_aggregate=aggregate,inputs=inputs,forks=len(records),limits=plan['limits'])
    result=ev.normalized(result);result['summary_hash']=ev.state_digest(result);write(OUT/'summary.json',result)
    return result


def launch():
    checked()
    def one(job):
        seed,arm=job
        with (OUT/f'seed_{seed:03d}_{arm}.log').open('w') as stream:
            process=subprocess.Popen([sys.executable,'-m','experiments.wait_content.long_followup','run','--seed',str(seed),'--arm',arm],
                cwd=ROOT,env=os.environ.copy(),stdout=stream,stderr=subprocess.STDOUT)
            return dict(seed=seed,arm=arm,pid=process.pid,returncode=process.wait())
    with ThreadPoolExecutor(max_workers=3) as pool:
        codes=list(pool.map(one,[(seed,arm) for seed in range(3) for arm in ARMS]))
    write(OUT/'exit_codes.json',codes)
    if any(r['returncode'] for r in codes):raise RuntimeError(codes)
    return summarize()


def self_test():
    # Tests only fixed fabricated facts, before any new behavioral score read.
    shape=(6,8,2);data={k:np.zeros(shape,dtype=np.int64) for k in PRIVATE}
    data.update(actions=np.zeros(shape,dtype=np.int64),result=np.zeros(shape,dtype=np.int64),
        teacher_actions=np.zeros(shape,dtype=np.int64),phase=np.zeros(shape,dtype=np.int64),enabled=np.ones(shape,dtype=bool))
    data['active']=np.zeros(shape,dtype=bool)
    rows=[dict(lane=4,job=0,request_id=1,case_id='c',issued_offset=0,matured=True,matured_offset=4,completed=True,clean_completed=False,accepted_begin_offset=1,feasible_begin=True),
          dict(lane=4,job=0,request_id=2,case_id='c',issued_offset=3,matured=True,matured_offset=5,completed=True,clean_completed=True,accepted_begin_offset=4,feasible_begin=True),
          dict(lane=4,job=1,request_id=3,case_id='c',issued_offset=4,matured=False,matured_offset=None,completed=None,clean_completed=None,accepted_begin_offset=None,feasible_begin=False)]
    suffix=compute(data,rows,3,6,suffix=True)['heldout']
    assert suffix['counts']['primary_matured']==1 and suffix['counts']['primary_clean_completed']==1
    assert suffix['excluded_prefix_crossing_requests']==1 and suffix['excluded_suffix_censored_requests']==1
    assert suffix['counts']['idle_opportunities']==24
    zero=np.zeros((2,8,2),dtype=np.int64)
    expected=hashlib.sha256(f'{TRACE_HASH_VERSION}/actions'.encode()+zero.tobytes()).hexdigest()
    assert action_hash(zero)==expected
    data.update(tick=np.arange(6))
    data['event_kind'][3,4,0]=NEW;data['event_eligible'][3,4,0]=1;data['event_regime_id'][3,4,0]=1
    data['request_id'][3:,4,0]=10;data['matured_mask'][5,4,0]=1;data['matured_request_id'][5,4,0]=10
    reconstructed=request_rows(data,[f'c{i}' for i in range(8)],list(range(8)),1)
    assert reconstructed[0]['matured_offset']==5
    json.dumps(reconstructed,allow_nan=False)
    print('3 fabricated boundary/hash/request-serialization tests passed',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','run','launch','summarize','test'))
    p.add_argument('--seed',type=int,choices=range(3),default=0);p.add_argument('--arm',choices=ARMS,default=ARMS[0])
    a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='run':run(a.seed,a.arm)
    elif a.command=='launch':launch()
    elif a.command=='summarize':summarize()
    else:self_test()


if __name__=='__main__':main()
