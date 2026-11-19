"""Summarize the held-out behavior and counterfactual tests without pooling seeds."""
from collections import Counter
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.learn_remember import memory_assay as m

EVIDENCE=m.ROOT/"docs/evidence/learn-remember-2026-11-03"
STAGES=("wide","withdrawal","review")
POLS=("plain","symmetric")
FIELDS=("matured","completed","clean_completed","idle_actions","idle_opportunities")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_records(kind,seed,stage):
    if kind=="plain":
        files=[m.BASE/f"evaluation-v1/seed_{seed:03d}/{stage}_common_{c}_family_{f}.json"
               for c in ("short","long_both") for f in range(4)]
        return [read(p) for p in files],files
    p=m.BASE/f"symmetric-v1/seed_{seed:03d}/behavior_{stage}.json"
    return read(p)["records"],[p]


def analyze():
    data=[]
    checks={}
    memory=[]
    keys={}
    for kind in POLS:
        for seed in range(3):
            for stage in STAGES:
                records,files=source_records(kind,seed,stage)
                for p in files:
                    checks[str(p.relative_to(m.ROOT))]=m.sha(p)
                assert len(records)==8
                for rec in records:
                    k=seed,rec["condition"],rec["family"]
                    current=(rec["primary_identity_hash"],rec["end_scheduler_hash"],rec["end_event_hash"])
                    assert keys.setdefault(k,current)==current
                for cond in ("short","long_both"):
                    grid={rec["family"]:rec for rec in records if rec["condition"]==cond}
                    for scope,families in (("all",range(4)),("old",[3]),("exclusive",[3]),("practiced",range(3))):
                        count=Counter()
                        for f in families:
                            complete=grid[f]["counts"]
                            if scope=="exclusive":
                                raws=[grid[f]["by_raw"][str(v)] for v in (1,3)]
                                counts={k:sum(t[k] for t in raws) for k in FIELDS[:3]}
                            else:
                                counts={k:complete[k] for k in FIELDS}
                            count.update(counts)
                        data.append(dict(kind=kind,seed=seed,stage=stage,condition=cond,scope=scope,counts=count,
                            strict=100*count["clean_completed"]/count["matured"],
                            ordinary=100*count["completed"]/count["matured"],
                            idle=1000*count["idle_actions"]/count["idle_opportunities"] if scope!="exclusive" else None))
                tag=stage if kind=="plain" else "sym_"+stage
                p=m.OUT/f"{tag}_s{seed}.json"
                d=read(p)
                checks[str(p.relative_to(m.ROOT))]=m.sha(p)
                rr={(r["anchor"],r["family"],r["raw"]):r for r in d["rows"]}
                interest=[r for r in d["rows"] if r["family"]==3 and r["raw"] in (1,3)]
                wrong_context=sum(rr[r["anchor"],f,r["raw"]]["original_content"]==r["target"]
                                  for r in interest for f in range(3))
                memory.append(dict(kind=kind,seed=seed,stage=stage,all=d["groups"]["all"],
                    exclusive=d["groups"]["family3_exclusive"], exclusive_old_answer_if_other_context=wrong_context,
                    other_context_n=3*len(interest), donor_max_error=d["max_donor_score_error"]))
    aggregate=[]
    for k in POLS:
        for c in ("short","long_both"):
            for scope in ("all","old","exclusive","practiced"):
                for stage in STAGES:
                    selected=[r for r in data if (r["kind"],r["condition"],r["scope"],r["stage"])==(k,c,scope,stage)]
                    ref=[r for r in data if (r["kind"],r["condition"],r["scope"],r["stage"])==(k,c,scope,"wide")]
                    row=dict(kind=k,condition=c,scope=scope,stage=stage)
                    for metric in ("strict","ordinary","idle"):
                        if selected[0][metric] is None:
                            continue
                        vals=[r[metric] for r in selected]
                        deltas=[v-r[metric] for v,r in zip(vals,ref,strict=True)]
                        row[metric]=dict(mean=mean(vals),sd=stdev(vals),delta_from_wide=mean(deltas),
                                         paired_sd=stdev(deltas))
                    aggregate.append(row)
    return dict(seed_rows=data,aggregate=aggregate,memory=memory,matched_case_sets=len(keys),inputs_sha256=checks)


def draw(result):
    plt.rcParams.update({"font.family":"Microsoft YaHei","font.size":10,"axes.unicode_minus":False})
    fig,axes=plt.subplots(1,3,figsize=(12.9,4.85))
    fig.subplots_adjust(left=.06,right=.985,top=.77,bottom=.25,wspace=.20)
    fig.text(.04,.93,"继续练习其他规则后，旧规则还能独立完成吗？",weight="bold",fontsize=16)
    fig.text(.04,.86,"长任务，全程无错完成率。点为三个种子的均值，竖线表示种子间范围。",color="#4a5964")
    for ax,scope,title in zip(axes,("old","exclusive","practiced"),("撤出的旧规则","旧规则独有的两个符号","持续练习的其他三种规则")):
        for k,label,col,offset in (("plain","普通 GRU","#294e91",-.08),("symmetric","循环权重双向共享","#bd571b",.08)):
            values=np.array([[next(r["strict"] for r in result["seed_rows"] if
                    (r["kind"],r["stage"],r["seed"],r["condition"],r["scope"])==(k,stage,seed,"long_both",scope))
                    for stage in STAGES] for seed in range(3)])
            x=np.arange(3)+offset
            avg=values.mean(0)
            ax.errorbar(x,avg,yerr=np.stack((avg-values.min(0),values.max(0)-avg)),
                        fmt="o",color=col,capsize=4,lw=1.6,label=label)
        ax.set(title=title,xticks=range(3),xticklabels=["撤出前","不复习","1/8 复习"],ylim=(-2,102),yticks=range(0,101,25))
        ax.grid(axis="y",alpha=.2)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("严格完成率（%）")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=2,frameon=False,bbox_to_anchor=(.5,.045))
    fig.text(.04,.022,"每图的后两项是从同一存档出发的两个分支，训练时长与更新数相同。撤出边界仍保留少量未完成请求。",fontsize=8.5,color="#4a5964")
    fig.savefig(EVIDENCE/"retention.png",dpi=155,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    result=analyze()
    (EVIDENCE/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    draw(result)
