"""Independent raw-array request diagnosis; no model, oracle or world calls."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import numpy as np
EPS=1e-6

def first(mask):
    found=np.flatnonzero(mask)
    return int(found[0]) if len(found) else None

def category(flags):
    for name in ('early_FINISH','wrong_version','finish_before_BEGIN','duplicate_BEGIN','late_FINISH',
                 'legal_other_context_payload','legal_other_wrong_payload','other_rejection'):
        if flags.get(name):return name
    return None

def describe(values):
    values=list(values)
    return dict(count=len(values),mean=float(np.mean(values)) if values else None,
        median=float(np.median(values)) if values else None,min=min(values) if values else None,max=max(values) if values else None)

def summarize(rows):
    c=Counter(matured=len(rows),completed=sum(r['completed'] for r in rows),strict=sum(r['strict'] for r in rows),
        started=sum(r['accepted_BEGIN'] for r in rows),reached_legal_window=sum(r['reached_legal_window'] for r in rows),
        completed_not_strict=sum(r['completed'] and not r['strict'] for r in rows),
        first_error_actual_early_FINISH=sum(bool(r['first_error'] and r['first_error']['flags']['early_FINISH']) for r in rows),
        any_actual_early_FINISH=sum(r['first_actual_early_FINISH'] is not None for r in rows),
        first_legal_FINISH_emitted=sum(r['first_legal_window_FINISH'] is not None for r in rows),
        first_legal_FINISH_correct=sum(bool(r['first_legal_window_FINISH'] and r['first_legal_window_FINISH']['correct']) for r in rows),
        first_legal_FINISH_wrong=sum(bool(r['first_legal_window_FINISH'] and not r['first_legal_window_FINISH']['correct']) for r in rows),
        legal_window_no_FINISH=sum(r['reached_legal_window'] and r['first_legal_window_FINISH'] is None for r in rows))
    assert c['first_legal_FINISH_emitted']==c['first_legal_FINISH_correct']+c['first_legal_FINISH_wrong']
    assert c['reached_legal_window']==c['first_legal_FINISH_emitted']+c['legal_window_no_FINISH']
    nonstrict=[r for r in rows if r['completed'] and not r['strict']]
    early=[r for r in rows if r['first_error'] and r['first_error']['flags']['early_FINISH']]
    def rate(n,d):return c[n]/c[d] if c[d] else None
    return dict(counts=dict(c),mechanism=dict(Counter(r['mechanism'] for r in rows)),
        all_matured_first_error=dict(Counter(r['first_error']['category'] if r['first_error'] else 'none' for r in rows)),
        completed_not_strict_first_error=dict(Counter(r['first_error']['category'] for r in nonstrict)),
        first_error_primitive_flags=dict(Counter(k for r in rows if r['first_error'] for k,v in r['first_error']['flags'].items() if v)),
        first_error_actual_early_lead_seconds=describe(r['first_error']['early_lead_seconds'] for r in early),
        any_actual_early_first_lead_seconds=describe(r['first_actual_early_FINISH']['early_lead_seconds'] for r in rows if r['first_actual_early_FINISH']),
        completed_not_strict_first_early_lead_seconds=describe(r['first_error']['early_lead_seconds'] for r in nonstrict if r['first_error']['flags']['early_FINISH']),
        first_legal_window_latency_seconds=describe(r['first_legal_window_FINISH']['now']-r['first_legal_observation']['now'] for r in rows if r['first_legal_window_FINISH']),
        metrics=dict(ordinary_completion_rate=rate('completed','matured'),strict_completion_rate=rate('strict','matured'),
            first_error_actual_early_rate=rate('first_error_actual_early_FINISH','matured'),any_actual_early_rate=rate('any_actual_early_FINISH','matured'),
            first_legal_FINISH_emitted_per_matured=rate('first_legal_FINISH_emitted','matured'),
            first_legal_FINISH_correct_per_matured=rate('first_legal_FINISH_correct','matured'),
            first_legal_FINISH_wrong_per_matured=rate('first_legal_FINISH_wrong','matured'),
            legal_window_no_FINISH_per_matured=rate('legal_window_no_FINISH','matured'),
            first_legal_FINISH_emitted_per_window=rate('first_legal_FINISH_emitted','reached_legal_window'),
            first_legal_FINISH_correct_precision=rate('first_legal_FINISH_correct','first_legal_FINISH_emitted')),
        denominators=dict(all_matured=c['matured'],actual_legal_window=c['reached_legal_window'],
            emitted_legal_FINISH=c['first_legal_FINISH_emitted'],completed_not_strict=c['completed_not_strict']))

def analyze(result,frames,states,stored):
    initial=states['initial_world'];actors=list(initial['spec']['actor_ids']);jobs=initial['spec']['n_jobs']
    active,started,a=frames['active'],frames['started'],frames['actions'];now=frames['now'][:,None,None]
    assert np.array_equal(a,frames['logits'].argmax(-1))
    begin=(a==1)|(a==2);finish=(a>=3)&(a<11);chosen_version=np.where(begin,a-1,(a-3)//4)
    same=chosen_version==frames['version'];chosen_payload=(a-3)%4
    legal=active&started&(now>=frames['ready_time']-EPS)&(now<=frames['deadline']+EPS)
    accepted_begin=begin&active&same&~started
    accepted_finish=finish&legal&same&(chosen_payload==frames['payload'])
    rejected=(a!=0)&~(accepted_begin|accepted_finish)
    actual_early=active&started&finish&(now<frames['ready_time']-EPS)
    assert np.array_equal(frames['result'],(accepted_begin|accepted_finish).astype(int)-rejected.astype(int))
    commands=((frames['event_kind']==1)|(frames['event_kind']==2))&frames['event_eligible']
    commands&=(frames['event_request_id']//jobs>result['fork_tick'])&(frames['event_regime_id']==result['evaluation_regime_id'])
    issued={}
    for t,i,j in np.argwhere(commands):
        key=int(i),int(j),int(frames['event_request_id'][t,i,j]);assert key not in issued;issued[key]=int(t)
    mature={}
    for t,i,j in np.argwhere(frames['matured_mask']):
        key=int(i),int(j),int(frames['matured_request_id'][t,i,j])
        if key in issued:assert key not in mature;mature[key]=int(t)
    stored_mature={(actors.index(r['actor']),r['job'],r['request_id']):r for r in stored['matured']}
    assert set(mature)==set(stored_mature)
    assert len(issued)==result['counts']['primary_issued'] and len(mature)==result['counts']['matured']
    assert len(issued)-len(mature)==result['counts']['primary_censored']
    hf,bf=result['history_family'],result['B_family'];table=result['mapping_table'];conflicts=result['raw_groups']['conflict'];rows=[]
    all_completed=all_strict=0
    for key,hi in sorted(mature.items()):
        i,j,rid=key;lo=issued[key];ts=np.arange(lo,hi);ts=ts[frames['request_id'][ts,i,j]==rid];assert len(ts)
        family=int(frames['event_family'][lo,i,j]);raw=int(frames['raw_event_payload'][lo,i,j]);correct=int(table[family][raw])
        other=int(table[bf if family==hf else hf][raw])
        assert np.all(frames['payload'][ts,i,j]==correct)
        ffinish=accepted_finish[ts,i,j];bad=rejected[ts,i,j]&active[ts,i,j];end=first(ffinish)
        completed=bool(ffinish.any());strict=bool(completed and not bad[:end+1].any())
        sr=stored_mature[key];assert completed==sr['completed'] and strict==sr['clean_completed']
        assert family==sr['family'] and raw==sr['raw'];all_completed+=completed;all_strict+=strict
        if raw not in conflicts:continue
        lb=first(accepted_begin[ts,i,j]);ll=first(legal[ts,i,j]);lf=first(legal[ts,i,j]&finish[ts,i,j]);le=first(actual_early[ts,i,j]);err=first(bad)
        def fact(t):
            return dict(frame=int(t),tick=int(frames['tick'][t]),now=float(frames['now'][t]),
                action=int(a[t,i,j]),receipt=int(frames['result'][t,i,j]),active=bool(active[t,i,j]),started=bool(started[t,i,j]),
                version_correct=bool(same[t,i,j]),payload=int(chosen_payload[t,i,j]) if finish[t,i,j] else None,
                correct_payload=correct,other_context_payload=other,ready=float(frames['ready_time'][t,i,j]) if np.isfinite(frames['ready_time'][t,i,j]) else None,
                deadline=float(frames['deadline'][t,i,j]),early_lead_seconds=float(frames['ready_time'][t,i,j]-frames['now'][t]) if actual_early[t,i,j] else None)
        mechanism='not_started' if lb is None else 'started_but_never_legal' if ll is None else 'legal_window_no_FINISH'
        finish_record=None
        if lf is not None:
            t=int(ts[lf]);finish_record=fact(t);finish_record['correct']=bool(accepted_finish[t,i,j])
            mechanism='first_legal_FINISH_correct' if finish_record['correct'] else 'first_legal_FINISH_other_context_payload' if same[t,i,j] and chosen_payload[t,i,j]==other else 'first_legal_FINISH_other_error'
        error_record=None
        if err is not None:
            t=int(ts[err]);error_record=fact(t)
            flags=dict(early_FINISH=bool(actual_early[t,i,j]),wrong_version=not bool(same[t,i,j]),
                finish_before_BEGIN=bool(finish[t,i,j] and not started[t,i,j]),duplicate_BEGIN=bool(begin[t,i,j] and started[t,i,j]),
                late_FINISH=bool(finish[t,i,j] and frames['now'][t]>frames['deadline'][t,i,j]+EPS),
                wrong_payload=bool(finish[t,i,j] and chosen_payload[t,i,j]!=correct),
                legal_other_context_payload=bool(finish[t,i,j] and legal[t,i,j] and same[t,i,j] and chosen_payload[t,i,j]==other),
                legal_other_wrong_payload=bool(finish[t,i,j] and legal[t,i,j] and same[t,i,j] and chosen_payload[t,i,j] not in (correct,other)),other_rejection=True)
            cat=category(flags);flags['other_rejection']=cat=='other_rejection';error_record.update(category=cat,flags=flags)
        assert not(completed and not strict) or error_record is not None
        rows.append(dict(seed=result['seed'],arm=result['arm'],snapshot=result['snapshot'],family='A' if family==hf else 'B',family_cue=family,raw=raw,
            actor=int(actors[i]),job=j,request_id=rid,source_kind='NEW' if frames['event_kind'][lo,i,j]==1 else 'REPLACE',
            issued_frame=lo,matured_frame=hi,completed=completed,strict=strict,accepted_BEGIN=lb is not None,reached_legal_window=ll is not None,
            mechanism=mechanism,first_legal_window_FINISH=finish_record,first_legal_observation=fact(int(ts[ll])) if ll is not None else None,
            first_actual_early_FINISH=fact(int(ts[le])) if le is not None else None,first_error=error_record))
    assert all_completed==result['counts']['completed'] and all_strict==result['counts']['clean_completed']
    byraw=[];byfamily=[]
    for family,cue in (('A',hf),('B',bf)):
        subset=[r for r in rows if r['family']==family];byfamily.append(dict(family=family,**summarize(subset)))
        for raw in conflicts:
            selected=[r for r in subset if r['raw']==raw];c=result['by_family_raw'].get(f'{cue}:{raw}',{}).get('counts',{})
            assert len(selected)==c.get('matured',0) and sum(r['completed'] for r in selected)==c.get('completed',0) and sum(r['strict'] for r in selected)==c.get('clean_completed',0)
            byraw.append(dict(family=family,raw=raw,**summarize(selected)))
    identities=sorted((r['actor'],r['job'],r['request_id'],r['family'],r['raw']) for r in rows)
    return dict(seed=result['seed'],arm=result['arm'],snapshot=result['snapshot'],protocol_hash=result['protocol_hash'],
        evidence_stem=result['evidence_stem'],requests=rows,by_family=byfamily,by_family_raw=byraw,
        audit=dict(all_primary_issued=len(issued),all_primary_matured=len(mature),all_primary_censored=len(issued)-len(mature),
            all_matured_completion_and_strict_exact=True,all_receipts_semantically_exact=True,native11_argmax_exact=True,
            conflict_cohort_identity_sha256=hashlib.sha256(json.dumps(identities).encode()).hexdigest(),
            early_requires_actual_active_and_started=True,no_world_or_model_calls=True))
