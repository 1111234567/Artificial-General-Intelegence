"""Equal-seed autonomous curves and actual request timing summaries."""
from __future__ import annotations
import csv
import json
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.wait_position import eval_position as e,eval_timing

def describe(values):
    values=[v for v in values if v is not None]
    return dict(mean=statistics.mean(values) if values else None,sample_sd=statistics.stdev(values) if len(values)>1 else None)

def behavior_summary():
    """Reuse the validated aggregation, changing only destination/binding APIs."""
    plan=e.checked();p=e.prior
    keys=('OUT','ARMS','FORMAT','checked','records','write')
    saved={key:getattr(p,key) for key in keys}
    try:
        p.OUT=e.OUT;p.ARMS=e.ARMS;p.FORMAT=e.FORMAT;p.checked=lambda:plan;p.records=e.records;p.write=e.write
        return p.summarize()
    finally:
        for key,value in saved.items():setattr(p,key,value)

def timing_summary():
    plan=e.checked();values=[]
    for seed in range(3):
        for arm in e.ARMS:
            for snapshot in e.MIXED:
                path=e.OUT/'timing'/f'seed_{seed:03d}'/f'{arm}_{snapshot}.json'
                row=e.read(path);assert row['protocol_hash']==plan['protocol_hash'];values.append(row)
    assert len(values)==75
    for seed in range(3):
        cohort={r['audit']['conflict_cohort_identity_sha256'] for r in values if r['seed']==seed}
        assert len(cohort)==1,('changed mature conflict cohort',seed)
    summaries=[];flat=[]
    for arm in e.ARMS:
        for snapshot in e.MIXED:
            for family in ('A','B'):
                per_seed=[]
                for seed in range(3):
                    row=next(r for r in values if (r['seed'],r['arm'],r['snapshot'])==(seed,arm,snapshot))
                    item=next(r for r in row['by_family'] if r['family']==family)
                    per_seed.append(dict(seed=seed,**item))
                    for raw in row['by_family_raw']:
                        if raw['family']!=family:continue
                        flat.append(dict(seed=seed,arm=arm,snapshot=snapshot,family=family,raw=raw['raw'],**raw['counts'],**raw['metrics'],
                            first_error_counts=json.dumps(raw['all_matured_first_error'],sort_keys=True),
                            first_error_early_lead_mean=raw['first_error_actual_early_lead_seconds']['mean']))
                summaries.append(dict(arm=arm,snapshot=snapshot,family=family,per_seed=per_seed,
                    metrics={m:describe(r['metrics'][m] for r in per_seed) for m in per_seed[0]['metrics']},
                    first_error_actual_early_lead_seconds_equal_seed_mean=describe(r['first_error_actual_early_lead_seconds']['mean'] for r in per_seed)))
    value=dict(protocol_hash=plan['protocol_hash'],complete=True,mixed_bindings=len(values),summaries=summaries,
        note='All summaries weight seed rates equally; integer per-seed and raw counts remain visible. Lead means average seed means only among seeds with events; full per-request leads are in timing/seed_NNN. Actual early FINISH requires active AND started AND before-ready, independent of version/content.')
    e.write(e.OUT/'timing-summary.json',value)
    with (e.OUT/'timing-curves.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(x.keys() for x in flat))));writer.writeheader();writer.writerows(flat)
    return value

def fmt(value,percent=False):
    if value is None:return 'NA'
    return f'{100*value:.2f}%' if percent else f'{value:.3f}'

def run():
    plan=e.checked();assert len(e.records())==210
    if (e.OUT/'report-binding.json').exists():
        value=e.read(e.OUT/'report-binding.json')
        for path,digest in value['artifacts'].items():assert e.sha(path)==digest
        return value
    summary=behavior_summary();timing=timing_summary();rows=summary['summaries']
    def pick(scenario,arm,snapshot,part,group):return next((r for r in rows if (r['scenario'],r['arm'],r['snapshot'],r['family_part'],r['raw_group'])==(scenario,arm,snapshot,part,group)),None)
    def metric(row,name):return row['metrics'][name]['mean'] if row else None
    lines=['# WAIT监督位置：相同旧样本下的自主行为', '',f"协议 `{plan['protocol_hash']}`；210/210 checkpoint绑定。",'',
        '3个训练种子的比率等权平均，保留样本标准差和每个种子整数分子/分母。三种新策略是 midpoint、hard（风险位置）与 random；旧uniform和no_review为已验证同条件参考。midpoint若实际权重不等于旧uniform，就评估其实际权重。', '',
        '每个冻结模型只读取相同1024 tick公开WAIT暖身，再在2048 tick独立long_both世界取完整11动作argmax。没有动作闸门、教师输入、参数更新或测试内状态重置。mixed是每次NEW/每个job随机抽A/B，允许并发；它不是严格轮换。', '',
        '实际新B例数来自序列化世界的family_new_counts(B)减原B0，包含无损失边界NEW；重放和梯度位置不增加新例。', '',
        '|快照|seed0 B NEW|seed1 B NEW|seed2 B NEW|', '|---|---:|---:|---:|']
    for stage in e.SNAPSHOTS:
        r=pick('B','midpoint',stage,'B','all');vals={x['seed']:x['actual_unique_B_NEW'] for x in r['per_seed']}
        lines.append('|'+stage+'|'+'|'.join(str(vals[s]) for s in range(3))+'|')
    lines+=['','## 长B早期曲线','','|快照|策略|普通完成|严格完成|idle/千|','|---|---|---:|---:|---:|']
    for stage in e.SNAPSHOTS:
        for arm in e.ARMS:
            r=pick('B',arm,stage,'B','all')
            lines.append(f"|{stage}|{arm}|{fmt(metric(r,'ordinary_completion_rate'),True)}|{fmt(metric(r,'strict_completion_rate'),True)}|{fmt(metric(r,'idle_actions_per1000'))}|")
    lines+=['','## 同一随机混合世界，A/B冲突分别计分','','|快照|策略|A普通|A严格|A idle/千|B普通|B严格|B idle/千|','|---|---|---:|---:|---:|---:|---:|---:|']
    for stage in e.MIXED:
        for arm in e.ARMS:
            vals=[]
            for part in ('history','B'):
                r=pick('mixed',arm,stage,part,'conflict');vals +=[fmt(metric(r,'ordinary_completion_rate'),True),fmt(metric(r,'strict_completion_rate'),True),fmt(metric(r,'idle_actions_per1000'))]
            lines.append('|'+stage+'|'+arm+'|'+'|'.join(vals)+'|')
    lines+=['','## B8逐种子实际冲突请求时机','','“提前”要求原世界行动前active且started且now < ready_time；即使版本或内容也错，仍保留这一事实。首错是该请求处于active时最早被拒的非WAIT。合法窗口按实际BEGIN产生的ready_time到deadline定义。', '',
        '|seed|策略|家族|成熟|普通|严格|首错实际早发|有合法窗口|窗口首次发|首次对|首次错|窗口未发|首错提前秒均值|','|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for seed in range(3):
        for arm in e.ARMS:
            for family in ('A','B'):
                r=next(r for r in timing['summaries'] if (r['arm'],r['snapshot'],r['family'])==(arm,'B008',family))
                item=next(x for x in r['per_seed'] if x['seed']==seed);c=item['counts']
                fields=[seed,arm,family,*[c[k] for k in ('matured','completed','strict','first_error_actual_early_FINISH','reached_legal_window','first_legal_FINISH_emitted','first_legal_FINISH_correct','first_legal_FINISH_wrong','legal_window_no_FINISH')],fmt(item['first_error_actual_early_lead_seconds']['mean'])]
                lines.append('|'+'|'.join(map(str,fields))+'|')
    lines+=['', 'timing-summary.json包含所有B0/1/2/4/8的逐种子计数和等权比率；timing-curves.csv进一步拆两种冲突raw。timing/seed_NNN保存每个成熟冲突请求的实际首错、重叠错误标志、首次合法FINISH与提前秒数。无BEGIN、开始但没有实际合法窗口、到达窗口却没有FINISH分别保留，避免把单纯保守解释为成功。', '',
        '普通完成允许先前拒绝；严格完成要求到成功FINISH为止没有被拒。WAIT造成的错过不算拒绝错误，但会降低完成率。idle分母是实际inactive且enabled的job机会，其raw与家族为槽位保留内容。', '',
        '该结果限于有监督原型、3种子和每种子一个独立外部流。等待时机行为并不能单独证明规则关联已经丢失或保留。', '']
    (e.OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    colors=dict(midpoint='#3768b0',hard='#e58a27',random='#4b9c5e',uniform='#9764b0',no_review='#777777')
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),sharey=True)
    for ax,scenario,part,group,stages,title in [(axes[0],'B','B','all',e.SNAPSHOTS,'Long B'),(axes[1],'mixed','history','conflict',e.MIXED,'Mixed A conflict'),(axes[2],'mixed','B','conflict',e.MIXED,'Mixed B conflict')]:
        for arm in e.ARMS:
            rr=[pick(scenario,arm,t,part,group) for t in stages];xx=list(range(len(stages)))
            yy=[metric(r,'strict_completion_rate') for r in rr];sd=[r['metrics']['strict_completion_rate']['sample_sd'] or 0 for r in rr]
            ax.errorbar(xx,yy,yerr=sd,color=colors[arm],marker='o',label=arm,capsize=2)
            for seed in range(3):ax.plot(xx,[next(p['metrics']['strict_completion_rate'] for p in r['per_seed'] if p['seed']==seed) for r in rr],color=colors[arm],alpha=.16,lw=.7)
        ax.set_title(title);ax.set_ylim(0,1);ax.grid(alpha=.2);ax.set_xticks(range(len(stages)),['0','2u','8u','1r','2r','4r','8r'] if len(stages)==7 else ['0','1r','2r','4r','8r'])
    axes[0].set_ylabel('Strict completion');axes[0].legend(fontsize=8);fig.supxlabel('Frozen checkpoints; u = first-epoch updates, r = full B round');fig.suptitle('Equal seed means +/- sample SD; faint traces = individual seeds');fig.tight_layout(rect=(0,.03,1,.94));fig.savefig(e.OUT/'curves.png',dpi=180);plt.close(fig)
    names=('summary.json','curves.csv','curves.png','timing-summary.json','timing-curves.csv','report.md')
    value=dict(protocol_hash=plan['protocol_hash'],source_sha256=e.sha(__file__),complete=True,records=210,artifacts={str(e.OUT/name):e.sha(e.OUT/name) for name in names})
    e.write(e.OUT/'report-binding.json',value);print(json.dumps(dict(event='wait_position_report',complete=True,records=210)),flush=True);return value

if __name__=='__main__':run()
