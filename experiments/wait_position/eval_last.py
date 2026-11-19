"""Post-interim static-last supplement; main210 bindings remain immutable."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import time
import numpy as np
import torch
from experiments.wait_position import eval_position as e
ROOT=e.ROOT
OUT=ROOT/'data/wait-position/eval-last-v1'
TRAIN=ROOT/'data/wait-position/last-v1'
PLAN=OUT/'protocol.json'
MAIN_OUT=ROOT/'data/wait-position/eval-v1'
STAGES=('B000','B001','B002','B004','B008')
FORMAT='wait-position-static-last-autonomous-supplement-v1'
original_write=e.write

def scenarios(stage):return ('B','mixed','history') if stage=='B008' else ('B',)

def checked():
    plan=e.read(PLAN)
    assert plan['protocol_hash']==e.state_digest({k:v for k,v in plan.items() if k!='protocol_hash'})
    e.prior.checked()
    main=e.read(MAIN_OUT/'protocol.json')
    assert main['protocol_hash']==plan['main_protocol_hash']
    assert main['protocol_hash']==e.state_digest({k:v for k,v in main.items() if k!='protocol_hash'})
    for path,digest in dict(main['frozen_files'],**plan['frozen_files']).items():assert e.sha(path)==digest,('supplement frozen dependency',path)
    return plan

def prepare():
    main=e.checked();assert not PLAN.exists()
    train=e.read(TRAIN/'protocol.json')
    files=[Path(__file__).resolve(),MAIN_OUT/'protocol.json',TRAIN/'protocol.json']
    plan=dict(format=FORMAT,created_utc=datetime.now(timezone.utc).isoformat(),
        declaration='Post-interim mechanistic supplement, after observing that over92percent of first-round hard choices used the final eligible WAIT tick and after interim B002 timing diagnostics. Not a prospective fourth arm; main210 bindings and frozen protocol remain unchanged. Supplementary scope frozen before reading last autonomous outcomes.',
        main_protocol_hash=main['protocol_hash'],training_protocol_hash=train['protocol_hash'],
        frozen_files={str(p):e.sha(p) for p in files},seeds=[0,1,2],arms=['last'],snapshots=list(STAGES),
        expected_records=21,steps=main['steps'],batch=main['batch'],spec=main['spec'],
        tests='Exactly same verified public WAIT warmup, heldout world, future events and matured eligible requests. B-only B0/1/2/4/8, B8 random-mixed and A-only history. No learning/action gate; native11-action argmax.',
        timing=main['timing_early'],timing_legal=main['timing_legal'],timing_denominator=main['timing_denominator'],
        comparison='Bind original main hard/midpoint/random/uniform/no_review actual checkpoint results at identical seed/snapshot/scenario; require external/primary identity and actual B NEW equality. Equal-seed means and per-seed raw numerators/denominators.',
        audit='Use unchanged main raw-array audit/scorer through explicit in-process destination/arm/scenario adapter. Main audit has210-row completeness constant; adapter replaces only final expected-record count with21 after exact set assertion. All raw receipts, request outcomes, timings, argmax and heldout identity checks remain unchanged.')
    plan['protocol_hash']=e.state_digest(plan);original_write(PLAN,plan);return plan

def write(path,value):
    path=Path(path)
    if path==OUT/'audit.json':
        expected={(s,'last',t,q) for s in range(3) for t in STAGES for q in scenarios(t)}
        rows=e.records();assert len(rows)==21 and {(r['seed'],r['arm'],r['snapshot'],r['scenario']) for r in rows}==expected
        value=dict(value,complete=True,expected_records=21,supplementary_scope=True)
    return original_write(path,value)

def install():
    e.OUT=OUT;e.TRAIN=TRAIN;e.PLAN=PLAN;e.FORMAT=FORMAT;e.ARMS=('last',);e.NEW_ARMS=('last',);e.SNAPSHOTS=STAGES
    e.scenarios=scenarios;e.checked=checked;e.write=write

def report():
    plan=checked();rows=e.records();assert len(rows)==21
    main=e.read(MAIN_OUT/'summary.json');assert main['complete'] and main['records']==210
    flat=[];bindings={};timing=[]
    for row in rows:
        for arm in ('midpoint','hard','random','uniform','no_review'):
            path=MAIN_OUT/f"seed_{row['seed']:03d}"/f"{arm}_{row['snapshot']}_{row['scenario']}.json"
            other=e.read(path);bindings[str(path)]=e.sha(path)
            for key in ('external_digest','primary_identity_hash','start_scheduler_hash','end_scheduler_hash','start_event_hash','end_event_hash','warmup_observation_hash','actual_unique_B_NEW'):
                assert row[key]==other[key],('last/main unmatched',key,path)
        for part,family in (('A',row['history_family']),('B',row['B_family'])):
            for group,raws in dict(row['raw_groups'],**{f'raw_{raw}':[raw] for raw in row['raw_groups']['conflict']}).items():
                c=Counter()
                for raw in raws:c.update(row['by_family_raw'].get(f'{family}:{raw}',{}).get('counts',{}))
                if not c.get('matured'):continue
                flat.append(dict(seed=row['seed'],arm='last',snapshot=row['snapshot'],scenario=row['scenario'],family=part,raw_group=group,raws=raws,
                    counts=dict(c),metrics=e.base.rates(c),actual_unique_B_NEW=row['actual_unique_B_NEW']))
        if row['scenario']=='mixed':
            path=OUT/'timing'/f"seed_{row['seed']:03d}"/'last_B008.json'
            diagnostic=e.read(path)
            main_timing=e.read(MAIN_OUT/'timing'/f"seed_{row['seed']:03d}"/'hard_B008.json')
            assert diagnostic['audit']['conflict_cohort_identity_sha256']==main_timing['audit']['conflict_cohort_identity_sha256']
            timing.append(diagnostic)
    train=e.read(TRAIN/'protocol.json')
    for row in rows:
        assert row['checkpoint_transfer']['arm']=='last'
        assert row['checkpoint_wait_position']['protocol_hash']==train['protocol_hash']
    def describe(values):
        values=[v for v in values if v is not None]
        return dict(mean=statistics.mean(values) if values else None,sample_sd=statistics.stdev(values) if len(values)>1 else None)
    summaries=[]
    for stage in STAGES:
        for scenario in scenarios(stage):
            for family in ('A','B'):
                for group in ('all','shared','conflict'):
                    per_seed=[r for r in flat if (r['snapshot'],r['scenario'],r['family'],r['raw_group'])==(stage,scenario,family,group)]
                    if not per_seed:continue
                    assert len(per_seed)==3
                    summaries.append(dict(snapshot=stage,scenario=scenario,family=family,raw_group=group,per_seed=per_seed,
                        metrics={m:describe(r['metrics'][m] for r in per_seed) for m in e.base.rates({})}))
    value=dict(format=FORMAT,protocol_hash=plan['protocol_hash'],complete=True,records=21,summaries=summaries,timing=timing,
        main_result_bindings=bindings,note='Supplement after interim mechanism diagnostics. Same requests/exposures as main; rates are equal seed means, never pooled across A/B.')
    write(OUT/'summary.json',value)
    csvrows=[{k:v for k,v in x.items() if k not in ('counts','metrics','raws')}|dict(raws=json.dumps(x['raws']))|x['counts']|x['metrics'] for x in flat]
    with (OUT/'curves.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(r.keys() for r in csvrows))));writer.writeheader();writer.writerows(csvrows)
    def pick(stage,scenario,family,group):return next(r for r in summaries if (r['snapshot'],r['scenario'],r['family'],r['raw_group'])==(stage,scenario,family,group))
    def fmt(v):return 'NA' if v is None else f'{100*v:.2f}%'
    lines=['# 静态最后WAIT位置：中途观察后的机制补充','',plan['declaration'],'',
        '21/21独立绑定，使用与主线完全一致的公开暖身、外部未来、成熟请求身份和11动作argmax；主210条绑定没有增加或改写。所有比率按3seed等权。','',
        '|快照|策略|长B普通|长B严格|idle/千|','|---|---|---:|---:|---:|']
    for stage in STAGES:
        rr=[('last',pick(stage,'B','B','all')['metrics'])]
        for arm in ('midpoint','hard','random','uniform','no_review'):
            r=next(r for r in main['summaries'] if (r['arm'],r['snapshot'],r['scenario'],r['family_part'],r['raw_group'])==(arm,stage,'B','B','all'));rr.append((arm,r['metrics']))
        for arm,m in rr:lines.append(f"|{stage}|{arm}|{fmt(m['ordinary_completion_rate']['mean'])}|{fmt(m['strict_completion_rate']['mean'])}|{m['idle_actions_per1000']['mean']:.3f}|")
    lines+=['','|策略|B8混合A冲突普通|A严格|B冲突普通|B严格|','|---|---:|---:|---:|---:|']
    for arm in ('last','midpoint','hard','random','uniform','no_review'):
        vals=[]
        for family,part in (('A','history'),('B','B')):
            m=pick('B008','mixed',family,'conflict')['metrics'] if arm=='last' else next(r for r in main['summaries'] if (r['arm'],r['snapshot'],r['scenario'],r['family_part'],r['raw_group'])==(arm,'B008','mixed',part,'conflict'))['metrics']
            vals +=[fmt(m['ordinary_completion_rate']['mean']),fmt(m['strict_completion_rate']['mean'])]
        lines.append('|'+arm+'|'+'|'.join(vals)+'|')
    lines+=['','|seed|last家族|成熟|普通|严格|首错实际早发|首次legal对|首次legal错|legal窗口未发|提前秒均值|','|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for diagnostic in timing:
        for r in diagnostic['by_family']:
            c=r['counts'];lead=r['first_error_actual_early_lead_seconds']['mean']
            vals=[diagnostic['seed'],r['family'],*[c[k] for k in ('matured','completed','strict','first_error_actual_early_FINISH','first_legal_FINISH_correct','first_legal_FINISH_wrong','legal_window_no_FINISH')], 'NA' if lead is None else f'{lead:.3f}']
            lines.append('|'+'|'.join(map(str,vals))+'|')
    lines+=['',plan['timing'], '',plan['timing_legal'],'']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    write(OUT/'completion.json',dict(format=FORMAT,complete=True,records=21,protocol_hash=plan['protocol_hash'],
        main_protocol_hash=plan['main_protocol_hash'],training_protocol_hash=plan['training_protocol_hash'],
        artifacts={str(OUT/name):e.sha(OUT/name) for name in ('audit.json','summary.json','curves.csv','report.md')}))
    return value

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','watch','finalize'));p.add_argument('--seed',type=int,choices=range(3));args=p.parse_args();torch.set_num_threads(1)
    if args.command=='prepare':value=prepare();print(json.dumps(dict(protocol_hash=value['protocol_hash'],records=21)));return
    install()
    if args.command=='watch':print(json.dumps(e.available(args.seed,True)));return
    wanted=[e.result_path(s,'last',t,q) for s in range(3) for t in STAGES for q in scenarios(t)]
    while not all(path.exists() for path in wanted) or not (MAIN_OUT/'completion.json').exists():time.sleep(5)
    e.audit();report();print(json.dumps(dict(event='last_supplement_eval_complete',records=21)),flush=True)
if __name__=='__main__':main()
