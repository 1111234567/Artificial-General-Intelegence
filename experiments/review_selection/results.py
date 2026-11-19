"""Join directly scored learning curves, mixed contexts, and strict pair completions."""
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.review_selection import pairs as exp


ROOT=exp.ROOT
BASE=ROOT/"data/review-selection"
OUT=ROOT/"docs/evidence/review-selection-2026-11-03"
ARMS=("no_review","uniform","conflict","error")
NAMES=("不复习","均匀","只选冲突","按误差")
COLORS=("#697478","#177b79","#b6531d","#604ca3")
FULL=("B000","B001","B002","B004","B008")
STAGES=("B000","Bfirst002","Bfirst008",*FULL[1:])


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def make():
    source={}
    cases={}
    rows=[]
    auc=[]
    pair_rows=[]
    banks=[]
    for seed in range(3):
        for arm in ARMS:
            for stage in STAGES:
                scenarios=("B","mixed") if stage in FULL else ("B",)
                for scenario in scenarios:
                    path=BASE/f"eval-v1/seed_{seed:03d}/{arm}_{stage}_{scenario}.json"
                    r=read(path)
                    source[str(path.relative_to(ROOT))]=exp.sha(path)
                    key=(seed,scenario)
                    assert cases.setdefault(key,r["primary_identity_hash"])==r["primary_identity_hash"]
                    for family,name in ((r["B_family"],"B"),(r["history_family"],"A")):
                        if scenario=="B" and name=="A":
                            continue
                        for group in ("all","conflict","shared"):
                            c=r["strata"][str(family)][group]["counts"]
                            n=c["matured"]
                            rows.append(dict(seed=seed,arm=arm,stage=stage,scenario=scenario,family=name,group=group,
                                new=r["actual_unique_B_NEW"],correct=c.get("clean_completed",0),
                                ordinary=c.get("completed",0),total=n,strict=100*c.get("clean_completed",0)/n,
                                idle_actions=r["counts"]["idle_actions"],idle_opportunities=r["counts"]["idle_opportunities"]))
            for g in ("all","conflict"):
                selected=[next(r for r in rows if (r["seed"],r["arm"],r["stage"],r["scenario"],r["group"])==
                    (seed,arm,s,"B",g)) for s in FULL[:4]]
                x=[r["new"] for r in selected]; y=[r["strict"] for r in selected]
                assert x[0]==0 and all(l<r for l,r in zip(x,x[1:]))
                auc.append(dict(seed=seed,arm=arm,group=g,x=x,strict=y,
                    early_mean=sum((b-a)*(c+d)/2 for a,b,c,d in zip(x,x[1:],y,y[1:]))/x[-1]))
            pair_path=exp.OUT/f"seed_{seed:03d}_{arm}.json"
            pairs=read(pair_path)
            source[str(pair_path.relative_to(ROOT))]=exp.sha(pair_path)
            pair_rows.append(dict(seed=seed,arm=arm,groups=pairs["pair_counts"],external=pairs["events_sha"],
                                  idle_actions=pairs["idle_actions"],idle_opportunities=pairs["idle_opportunities"]))
            if arm!="no_review":
                sel=BASE/f"train-v1/seed_{seed:03d}/{arm}/selections.jsonl"
                choices=[json.loads(line) for line in sel.read_text(encoding="utf-8").splitlines()]
                source[str(sel.relative_to(ROOT))]=exp.sha(sel)
                assert len(choices)==256 and all(len(r["selected"])==64 for r in choices)
                amounts=np.bincount([i for r in choices for i in r["selected"]],minlength=256)
                banks.append(dict(seed=seed,arm=arm,selected_total=int(amounts.sum()),
                    chosen_conflict=sum(r["conflict_count"] for r in choices),distinct=int((amounts>0).sum()),
                    peak=int(amounts.max()),first_conflict=choices[0]["conflict_count"],
                    mean_selected_components=np.mean([r["selected_component_mean"] for r in choices],axis=0).tolist()))
    assert all(len({r["external"] for r in pair_rows if r["seed"]==seed})==1 for seed in range(3))
    return dict(rows=rows,auc=auc,pairs=pair_rows,selection=banks,source_sha256=source)


def draw(data):
    plt.rcParams.update({"font.family":"Microsoft YaHei","font.size":10,"axes.unicode_minus":False})
    fig,axes=plt.subplots(1,3,figsize=(13.7,4.7))
    fig.subplots_adjust(left=.055,right=.982,top=.76,bottom=.24,wspace=.25)
    fig.text(.035,.94,"用同一份复习预算，能否记住 A 并学会 B？",fontsize=17,weight="bold")
    fig.text(.035,.87,"长任务，只计全程无错。线、菱形和柱为三种子均值，小圆点为各个种子。右图放大了低完成率区间。",color="#465961")
    rows=data["rows"]
    for i,(arm,name,color) in enumerate(zip(ARMS,NAMES,COLORS)):
        def outcome(s,stage,scenario,family):
            return next(r for r in rows if (r["seed"],r["arm"],r["stage"],r["scenario"],r["family"],r["group"])==
                        (s,arm,stage,scenario,family,"conflict"))
        vals=[[outcome(s,stage,"B","B") for stage in FULL] for s in range(3)]
        xs=np.mean([[r["new"] for r in seed] for seed in vals],axis=0)
        ys=np.mean([[r["strict"] for r in seed] for seed in vals],axis=0)
        axes[0].plot(xs/1000,ys,"o-",ms=4,lw=1.8,color=color,label=name)
        mix=[(outcome(s,"B008","mixed","A")["strict"],outcome(s,"B008","mixed","B")["strict"]) for s in range(3)]
        a,b=zip(*mix)
        axes[1].scatter(a,b,color=color,alpha=.5,s=20)
        axes[1].scatter(mean(a),mean(b),color=color,s=95,marker="D",edgecolors="white",linewidth=.8)
        both=[r["groups"]["conflict_switch"] for r in data["pairs"] if r["arm"]==arm]
        ratio=[100*r["both_strict"]/r["pairs"] for r in both]
        axes[2].bar(i,mean(ratio),width=.6,color=color,alpha=.35)
        axes[2].scatter([i-.12,i,i+.12],ratio,color=color,s=26)
    axes[0].set(title="学习 B：冲突符号",xlabel="累计新发 B 请求（千）",ylabel="严格完成率（%）",ylim=(-3,104))
    axes[1].set(title="训练末：A/B 随机交错",xlabel="旧 A 冲突请求严格完成率（%）",
                ylabel="新 B 冲突请求严格完成率（%）",xlim=(-1,15),ylim=(-3,104))
    axes[2].set(title="同槽同符号：切换后两次都对",ylabel="成对严格完成率（%）",ylim=(-.4,12),
                xticks=range(len(NAMES)),xticklabels=NAMES)
    for ax in axes:
        ax.grid(axis="y",alpha=.16);ax.spines[["top","right"]].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=4,frameon=False)
    fig.savefig(OUT/"review-selection.png",dpi=150,facecolor="white");plt.close(fig)


if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    result=make()
    (OUT/"counts-and-curves.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    draw(result)
