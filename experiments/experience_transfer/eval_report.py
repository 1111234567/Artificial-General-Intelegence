"""Compact complete-curve exports from the already frozen evaluation evidence."""
from collections import Counter
import csv
import json
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.experience_transfer import eval_transfer as e


def describe(values):
    values=[float(x) for x in values if x is not None]
    return dict(mean=statistics.mean(values) if values else None,
                sample_sd=statistics.stdev(values) if len(values)>1 else None)


def run():
    plan=e.checked()
    records=[e.read(p) for p in sorted(e.OUT.glob('seed_*/*.json')) if not p.name.endswith('.requests.json')]
    rows=[]
    for r in records:
        for part,fam in [('B',r['B_family']),('history',r['history_family'])]:
            for group in ('all','shared','conflict'):
                value=r['strata'][str(fam)][group]
                if value['counts'].get('matured',0)==0:
                    continue
                rows.append(dict(seed=r['seed'],arm=r['arm'],snapshot=r['snapshot'],scenario=r['scenario'],
                    family_part=part,raw_group=group,**value['counts'],**value['metrics'],
                    **{'work_'+k:v for k,v in r['checkpoint_work'].items()}))
    columns=sorted(set().union(*(r.keys() for r in rows))) if rows else []
    with (e.OUT/'curves.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    curves=[]
    for group in ('all','shared','conflict'):
        for snapshot in e.SNAPSHOTS:
            row=dict(raw_group=group,snapshot=snapshot,arms={})
            for arm in (*e.CURVE_ARMS,'short_origin'):
                selected=[r for r in rows if (r['arm'],r['snapshot'],r['scenario'],r['family_part'],r['raw_group'])==(arm,snapshot,'B','B',group)]
                if not selected:
                    continue
                row['arms'][arm]=dict(per_seed=selected,metrics={key:describe(x[key] for x in selected) for key in e.rates({})})
            curves.append(row)
    old=[]
    for seed in range(3):
        for arm in e.ARMS:
            endpoints=[]
            for snapshot in ('B000','B008'):
                rr=next((r for r in records if (r['seed'],r['arm'],r['snapshot'],r['scenario'])==(seed,arm,snapshot,'history')),None)
                if rr is None:
                    continue
                table=rr['mapping_table'];hf,bf=rr['history_family'],rr['B_family']
                unique=[raw for raw in range(4) if table[hf][raw]!=table[bf][raw]]
                c=Counter()
                for raw in unique:
                    c.update(rr['by_family_raw'][f'{hf}:{raw}']['counts'])
                endpoints.append(dict(snapshot=snapshot,raws=unique,counts=dict(c),metrics=e.rates(c)))
            old.append(dict(seed=seed,arm=arm,endpoints=endpoints))
    mixed=[]
    for r in records:
        if r['scenario']=='mixed':
            mixed.append({k:r[k] for k in ('seed','arm','snapshot','counts','metrics','strata',
                'steps_with_both_contexts_active','actor_ticks_with_two_different_active_contexts')})
    value=dict(protocol_hash=plan['protocol_hash'],forks=len(records),B_curves=curves,
        old_unique_retention=old,mixed_context=mixed,
        note='All raw groups and checkpoints retained. Old-unique raws are independently determined per old rule versus B; A has2 and C has4. Shared/conflict B strata always refer to fixed A/B comparison.')
    e.write(e.OUT/'compact-results.json',value)
    fig,axes=plt.subplots(2,3,figsize=(14,7),sharex=True,sharey='row')
    colors=dict(related='#147d92',disjoint='#b65643',direct='#6e7087')
    x=list(range(len(e.SNAPSHOTS)))
    for col,group in enumerate(('all','shared','conflict')):
        for row_no,metric in enumerate(('strict_completion_rate','ordinary_completion_rate')):
            ax=axes[row_no,col]
            for arm in e.CURVE_ARMS:
                selected=[next((r for r in curves if r['raw_group']==group and r['snapshot']==s),None) for s in e.SNAPSHOTS]
                means=[r['arms'].get(arm,{}).get('metrics',{}).get(metric,{}).get('mean') if r else None for r in selected]
                if any(v is None for v in means):
                    continue
                sd=[r['arms'][arm]['metrics'][metric]['sample_sd'] or 0 for r in selected]
                ax.errorbar(x,means,yerr=sd,color=colors[arm],label=arm,marker='o',capsize=2,lw=1.5)
                for seed in range(3):
                    yy=[next((p[metric] for p in r['arms'][arm]['per_seed'] if p['seed']==seed),None) for r in selected]
                    ax.plot(x,yy,color=colors[arm],alpha=.2,lw=.7)
            ax.grid(alpha=.2);ax.set_ylim(0,1)
            if row_no==0:ax.set_title(group)
            if col==0:ax.set_ylabel('strict completion' if row_no==0 else 'ordinary completion')
            ax.set_xticks(x,['0','2u','8u','1r','2r','4r','8r'])
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Unseen B autonomous curves: mean +/- sample SD; faint lines = seeds')
    fig.supxlabel('Frozen snapshots (u = first-epoch updates, r = complete32-epoch rounds); categorical spacing')
    fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(e.OUT/'B-curves.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(forks=len(records),curve_rows=len(curves),csv_rows=len(rows))))


if __name__=='__main__':
    run()
