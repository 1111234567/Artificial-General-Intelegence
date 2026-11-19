"""Read-only request-level mechanism analysis of existing B008 mixed rollouts."""
from __future__ import annotations
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import torch

SOURCE=ROOT/'data/review-selection/eval-v1'
OUT=ROOT/'data/review-selection/error-analysis-v1'
ARMS=('uniform','conflict','error','no_review')
EPS=1e-6


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def first(index):
    found=np.flatnonzero(index)
    return int(found[0]) if len(found) else None


def first_error_category(flags):
    # Semantic source counters overlap. This fixed priority makes the bins exclusive.
    for name in ('wrong_version','finish_before_BEGIN','duplicate_BEGIN','early_FINISH','late_FINISH',
                 'legal_window_other_context_payload','legal_window_other_wrong_payload','invalid_action','other_rejection'):
        if flags.get(name):return name
    return None


def summarize(rows):
    c=Counter(matured=len(rows),completed=sum(r['completed'] for r in rows),strict=sum(r['strict'] for r in rows),
        started=sum(r['accepted_BEGIN'] for r in rows),reached_legal_window=sum(r['reached_legal_window'] for r in rows),
        completed_not_strict=sum(r['completed'] and not r['strict'] for r in rows))
    outcomes=Counter(r['mechanism'] for r in rows)
    first_bad=Counter(r['first_error']['category'] for r in rows if r['completed'] and not r['strict'])
    overlap=Counter(flag for r in rows if r['completed'] and not r['strict'] for flag,value in r['first_error']['flags'].items() if value)
    legal=[r for r in rows if r['reached_legal_window']]
    first_finished=[r for r in legal if r['first_legal_window_FINISH'] is not None]
    all_first_bad=Counter(r['first_error']['category'] if r['first_error'] else 'none' for r in rows)
    other_error=Counter('wrong_version' if not r['first_legal_window_FINISH']['version_correct'] else 'other_wrong_payload'
        for r in rows if r['mechanism']=='first_legal_FINISH_other_error')
    early=[r for r in rows if r['completed'] and not r['strict'] and r['first_error']['category']=='early_FINISH']
    early_content=Counter('correct_payload' if r['first_error']['action_payload']==r['first_error']['correct_payload'] else
        'other_context_payload' if r['first_error']['action_payload']==r['first_error']['other_context_payload'] else 'other_wrong_payload' for r in early)
    leads=[r['first_error']['ready']-r['first_error']['now'] for r in early]
    return dict(counts=dict(c),mechanism=dict(outcomes),completed_not_strict_first_error=dict(first_bad),
        completed_not_strict_first_error_overlap=dict(overlap),
        all_matured_first_error=dict(all_first_bad),first_legal_FINISH_other_error_breakdown=dict(other_error),
        completed_not_strict_first_early_FINISH_payload=dict(early_content),
        completed_not_strict_first_early_FINISH_lead_seconds=dict(count=len(leads),mean=float(np.mean(leads)) if leads else None,
            median=float(np.median(leads)) if leads else None,min=min(leads) if leads else None,max=max(leads) if leads else None),
        no_legal_window_FINISH_first_error=dict(Counter(r['first_error']['category'] if r['first_error'] else 'none' for r in rows if r['mechanism']=='legal_window_no_FINISH')),
        ordinary_rate=c['completed']/c['matured'] if c['matured'] else None,
        strict_rate=c['strict']/c['matured'] if c['matured'] else None,
        denominators=dict(all_matured=c['matured'],reached_actual_legal_window=len(legal),
            emitted_FINISH_in_actual_legal_window=len(first_finished),completed_not_strict=c['completed_not_strict']),
        first_legal_tick_correct_FINISH_above_other_context=sum(r['first_legal_logits']['correct_minus_other']>0 for r in legal),
        first_legal_tick_actual_actions=dict(Counter(r['first_legal_logits']['actual_action_type'] for r in legal)),
        source_kind=dict(Counter(r['source_kind'] for r in rows)))


def analyze(seed,arm):
    source=SOURCE/f'seed_{seed:03d}'/f'{arm}_B008_mixed.json'
    result=json.loads(source.read_text(encoding='utf-8'))
    stem=Path(result['evidence_stem'])
    paths=[source,stem.with_suffix('.npz'),stem.with_suffix('.pt'),stem.with_suffix('.requests.json')]
    binding={str(p):sha(p) for p in paths}
    for path,expected in result.get('artifacts',{}).items():assert sha(path)==expected
    with np.load(stem.with_suffix('.npz')) as rawframes:frames={k:rawframes[k] for k in rawframes.files}
    states=torch.load(stem.with_suffix('.pt'),map_location='cpu',weights_only=False)
    stored=json.loads(stem.with_suffix('.requests.json').read_text(encoding='utf-8'))
    initial=states['initial_world'];actors=list(initial['spec']['actor_ids']);jobs=initial['spec']['n_jobs']
    assert np.array_equal(frames['actions'],frames['logits'].argmax(-1))
    active=frames['active'];started=frames['started'];a=frames['actions'];now=frames['now'][:,None,None]
    begin=(a==1)|(a==2);finish=(a>=3)&(a<11);same=np.where(begin,a-1,(a-3)//4)==frames['version']
    payload=(a-3)%4
    accepted_begin=begin&active&same&~started
    actual_legal=active&started&(now>=frames['ready_time']-EPS)&(now<=frames['deadline']+EPS)
    accepted_finish=finish&actual_legal&same&(payload==frames['payload'])
    rejected=(a!=0)&~(accepted_begin|accepted_finish)
    assert np.array_equal(frames['result'],(accepted_begin|accepted_finish).astype(int)-rejected.astype(int))
    # Stored artifacts have private pre-action facts, rather than teacher arrays.
    # Reconstruct causal midpoint teacher labels solely for diagnostic evidence.
    feasible=active&~started&(now+frames['duration']<=frames['deadline']+EPS)
    midpoint=actual_legal&((now>=.5*(frames['ready_time']+frames['deadline'])-EPS)|(now+.25>frames['deadline']+EPS))
    teacher=np.where(feasible,1+frames['version'],np.where(midpoint,3+4*frames['version']+frames['payload'],0))
    commands=((frames['event_kind']==1)|(frames['event_kind']==2))&frames['event_eligible']
    commands&=(frames['event_request_id']//jobs>result['fork_tick'])&(frames['event_regime_id']==result['evaluation_regime_id'])
    issued={}
    for t,i,j in np.argwhere(commands):
        key=(int(i),int(j),int(frames['event_request_id'][t,i,j]))
        assert key not in issued
        issued[key]=int(t)
    mature={}
    inherited=0
    for t,i,j in np.argwhere(frames['matured_mask']):
        rid=int(frames['matured_request_id'][t,i,j]);key=(int(i),int(j),rid)
        if key in issued:mature[key]=int(t)
        else:inherited+=1
    stored_mature={(actors.index(r['actor']),r['job'],r['request_id']):r for r in stored['matured']}
    assert set(mature)==set(stored_mature)
    assert len(issued)==result['counts']['primary_issued']
    assert len(mature)==result['counts']['matured']
    assert len(issued)-len(mature)==result['counts']['primary_censored']
    table=result['mapping_table'];hf=result['history_family'];bf=result['B_family']
    conflicts=result['raw_groups']['conflict']
    rows=[]
    for key,hi in sorted(mature.items()):
        i,j,rid=key;lo=issued[key]
        ts=np.arange(lo,hi);belongs=frames['request_id'][ts,i,j]==rid;ts=ts[belongs]
        assert len(ts)>0
        family=int(frames['event_family'][lo,i,j]);raw=int(frames['raw_event_payload'][lo,i,j])
        correct=int(table[family][raw]);other_family=bf if family==hf else hf;other=int(table[other_family][raw])
        fbegin=accepted_begin[ts,i,j];legal=actual_legal[ts,i,j];ffinish=accepted_finish[ts,i,j]
        completed=bool(ffinish.any());bad=rejected[ts,i,j]&active[ts,i,j]
        clean=bool(completed and not bad[:first(ffinish)+1].any())
        sr=stored_mature[key]
        assert completed==sr['completed'] and clean==sr['clean_completed']
        assert family==sr['family'] and raw==sr['raw']
        if raw not in conflicts:continue
        assert correct!=other
        lb=first(fbegin);ll=first(legal);lf=first(legal&finish[ts,i,j]);error=first(bad)
        mechanism='not_started' if lb is None else 'started_but_never_legal' if ll is None else 'legal_window_no_FINISH'
        finish_record=None;logit_record=None
        if lf is not None:
            t=int(ts[lf]);chosen=int(a[t,i,j]);verok=bool(same[t,i,j]);chosen_payload=int(payload[t,i,j])
            mechanism=('first_legal_FINISH_correct' if verok and chosen_payload==correct else
                'first_legal_FINISH_other_context_payload' if verok and chosen_payload==other else 'first_legal_FINISH_other_error')
            finish_record=dict(tick=int(frames['tick'][t]),frame=t,action=chosen,receipt=int(frames['result'][t,i,j]),
                version_correct=verok,payload=chosen_payload,correct_payload=correct,other_context_payload=other,
                now=float(now[t,0,0]),ready=float(frames['ready_time'][t,i,j]),deadline=float(frames['deadline'][t,i,j]),
                reconstructed_midpoint_teacher=int(teacher[t,i,j]))
        if ll is not None:
            t=int(ts[ll]);chosen=int(a[t,i,j]);lg=frames['logits'][t,i,j]
            ca=3+4*int(frames['version'][t,i,j])+correct;oa=3+4*int(frames['version'][t,i,j])+other
            logit_record=dict(frame=t,tick=int(frames['tick'][t]),correct_FINISH_logit=float(lg[ca]),other_context_FINISH_logit=float(lg[oa]),
                correct_minus_other=float(lg[ca]-lg[oa]),actual_action=chosen,
                actual_action_type='WAIT' if chosen==0 else 'BEGIN' if chosen<3 else 'FINISH',
                reconstructed_midpoint_teacher=int(teacher[t,i,j]))
        error_record=None
        if error is not None:
            t=int(ts[error]);chosen=int(a[t,i,j]);isfinish=bool(finish[t,i,j]);isbegin=bool(begin[t,i,j]);verok=bool(same[t,i,j])
            flags=dict(wrong_version=not verok,finish_before_BEGIN=isfinish and verok and not bool(started[t,i,j]),
                duplicate_BEGIN=isbegin and verok and bool(started[t,i,j]),
                early_FINISH=isfinish and verok and bool(started[t,i,j]) and now[t,0,0]<frames['ready_time'][t,i,j]-EPS,
                late_FINISH=isfinish and verok and now[t,0,0]>frames['deadline'][t,i,j]+EPS,
                wrong_payload=isfinish and verok and int(payload[t,i,j])!=correct,
                legal_window_other_context_payload=isfinish and verok and bool(actual_legal[t,i,j]) and int(payload[t,i,j])==other,
                legal_window_other_wrong_payload=isfinish and verok and bool(actual_legal[t,i,j]) and int(payload[t,i,j]) not in (correct,other),
                invalid_action=chosen<0 or chosen>=11,other_rejection=True)
            category=first_error_category(flags)
            flags['other_rejection']=category=='other_rejection'
            error_record=dict(category=category,flags={k:bool(v) for k,v in flags.items()},frame=t,tick=int(frames['tick'][t]),
                action=chosen,receipt=int(frames['result'][t,i,j]),now=float(now[t,0,0]),
                started=bool(started[t,i,j]),ready=float(frames['ready_time'][t,i,j]) if np.isfinite(frames['ready_time'][t,i,j]) else None,
                deadline=float(frames['deadline'][t,i,j]),correct_payload=correct,other_context_payload=other,
                action_payload=int(payload[t,i,j]) if isfinish else None,reconstructed_midpoint_teacher=int(teacher[t,i,j]))
        assert not(completed and not clean) or error_record is not None
        rows.append(dict(seed=seed,arm=arm,family='A' if family==hf else 'B',family_cue=family,raw=raw,
            actor=int(actors[i]),job=j,request_id=rid,source_kind='NEW' if frames['event_kind'][lo,i,j]==1 else 'REPLACE',
            issued_frame=lo,matured_frame=hi,issued_tick=int(frames['tick'][lo]),matured_tick=int(frames['tick'][hi]),
            completed=completed,strict=clean,accepted_BEGIN=lb is not None,reached_legal_window=ll is not None,
            mechanism=mechanism,first_legal_window_FINISH=finish_record,first_legal_logits=logit_record,first_error=error_record))
    byraw=[]
    for family in ('A','B'):
        for raw in conflicts:
            selected=[r for r in rows if r['family']==family and r['raw']==raw]
            cue=hf if family=='A' else bf
            original_counts=result['by_family_raw'][f'{cue}:{raw}']['counts']
            assert len(selected)==original_counts['matured']
            assert sum(r['completed'] for r in selected)==original_counts['completed']
            assert sum(r['strict'] for r in selected)==original_counts['clean_completed']
            byraw.append(dict(seed=seed,arm=arm,family=family,raw=raw,**summarize(selected)))
    identities=sorted((r['actor'],r['job'],r['request_id'],r['family'],r['raw']) for r in rows)
    audit=dict(seed=seed,arm=arm,source_sha256=binding,argmax_exact=True,receipts_semantics_exact=True,
        all_matured_completed_and_strict_reconstructed_exact=True,all_primary_issued=len(issued),all_primary_matured=len(mature),
        all_primary_censored=len(issued)-len(mature),inherited_maturation_pulses_excluded=inherited,
        conflict_matured=len(rows),conflict_cohort_identity_sha256=hashlib.sha256(json.dumps(identities).encode()).hexdigest(),
        family_raw_source_counts_exact=True,no_model_or_world_calls=True)
    write(OUT/f'seed_{seed:03d}'/f'{arm}.json',dict(audit=audit,by_family_raw=byraw,requests=rows))
    return rows,byraw,audit


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),source='Existing B008 random-mixed rollout artifacts only',
        source_code_sha256=sha(__file__),new_training_updates=0,new_behavior_ticks=0,new_world_or_model_calls=0,
        scope='All3 seeds x4 arms xA/B families xeach of2 conflict raw symbols. Request cohort matches existing primary mature scorer exactly.',
        denominator='Newly issued eligible NEW or REPLACE commands after fork_tick in evaluation regime, whose score window matures within existing rollout. Exclude inherited warmup requests and right-censored requests. Within this cohort retain conflict raws only.',
        legal_window='Observed pre-action active and started state with actual accepted-BEGIN ready_time <= now <= deadline, epsilon1e-6. Timing window is judged before action, independently of chosen version/payload.',
        mechanism='Mutually exclusive: never accepted BEGIN; accepted BEGIN but no observed legal window; otherwise first actually emitted FINISH inside any legal window: correct version/payload; correct version but other context mapped payload; other error; or no FINISH in any actual legal window. First wrong FINISH may be followed by eventual ordinary completion.',
        nonstrict='For ordinary-completed but nonstrict requests locate earliest rejected non-WAIT action while active, prior to accepted FINISH. WAIT omissions are not rejection errors under native strict metric. Mutually exclusive priority: wrong version > FINISH before BEGIN > duplicate BEGIN > early FINISH > late FINISH > legal-window other-context payload > legal-window other wrong payload > invalid code > other. Also retain overlapping primitive flags, so early plus wrong-content is visible.',
        teacher='Teacher arrays were not saved in mixed artifacts. Reconstruct causal midpoint oracle labels from saved pre-action state for diagnosis only; no labels feed model or world.',
        interpretation='Actual legal-window actions support a behavioral diagnosis, not proof of an erased or retained internal association. Correct-vs-other FINISH logit comparison at first legal tick is conditional readout, not completion.')
    write(OUT/'protocol.json',protocol)
    rows=[];byraw=[];audits=[]
    for seed in range(3):
        for arm in ARMS:
            rr,bb,aa=analyze(seed,arm);rows+=rr;byraw+=bb;audits.append(aa)
        assert len({r['conflict_cohort_identity_sha256'] for r in audits if r['seed']==seed})==1
    aggregated=[];perseed=[]
    for arm in ARMS:
        for family in ('A','B'):
            aggregated.append(dict(arm=arm,family=family,**summarize([r for r in rows if r['arm']==arm and r['family']==family])))
            for seed in range(3):
                perseed.append(dict(arm=arm,family=family,seed=seed,**summarize([r for r in rows if r['arm']==arm and r['family']==family and r['seed']==seed])))
    write(OUT/'summary.json',dict(passed=True,protocol=protocol,pooled_counts=aggregated,per_seed=perseed,by_family_raw=byraw,audits=audits))
    flat=[]
    for r in byraw:
        flat.append({k:r[k] for k in ('seed','arm','family','raw','ordinary_rate','strict_rate')}|r['counts']|
                    {'mechanism_'+k:v for k,v in r['mechanism'].items()}|
                    {'first_error_'+k:v for k,v in r['completed_not_strict_first_error'].items()})
    fields=list(dict.fromkeys(k for row in flat for k in row))
    with (OUT/'by-family-raw.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(flat)
    lines=['# Existing mixed-rollout request diagnosis','',
        'Pooled integer counts below describe these exact three saved streams. Per-seed and per-raw denominators/rates remain in summary.json and by-family-raw.csv. No new actions, model calls or training were performed.','',
        '| Arm | Family | Matured | Ordinary | Strict | No BEGIN | BEGIN/no legal window | First legal FINISH correct | Other context | Other wrong | No legal FINISH |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in aggregated:
        c=r['counts'];m=r['mechanism']
        lines.append('| '+ ' | '.join(map(str,[r['arm'],r['family'],c['matured'],c['completed'],c['strict'],
            m.get('not_started',0),m.get('started_but_never_legal',0),m.get('first_legal_FINISH_correct',0),
            m.get('first_legal_FINISH_other_context_payload',0),m.get('first_legal_FINISH_other_error',0),m.get('legal_window_no_FINISH',0)]))+' |')
    lines+=['','First active rejection among ordinary-completed but nonstrict requests (exclusive priority in protocol.json):','',
        '| Arm | Family | Completed but nonstrict | First-error counts |','|---|---|---:|---|']
    for r in aggregated:
        lines.append(f"| {r['arm']} | {r['family']} | {r['counts']['completed_not_strict']} | {json.dumps(r['completed_not_strict_first_error'],sort_keys=True)} |")
    ca=next(r for r in aggregated if r['arm']=='conflict' and r['family']=='A')
    lines+=['',f"Conflict-only review, old A: all{ca['counts']['matured']} requests started and reached a legal window; first active rejection counts={json.dumps(ca['all_matured_first_error'],sort_keys=True)}. Among completed-but-nonstrict requests whose first error was early FINISH, payload counts={json.dumps(ca['completed_not_strict_first_early_FINISH_payload'],sort_keys=True)}."]
    lines+=['',protocol['denominator'],'',protocol['legal_window'],'',protocol['nonstrict'],'',protocol['teacher'],'',protocol['interpretation'],'']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(passed=True,path=str(OUT/'summary.json'),pooled_counts=aggregated)),flush=True)


if __name__=='__main__':main()
