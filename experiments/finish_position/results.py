"""Bind the learning, free timing, actual switched pairs, and fixed-history results."""
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.finish_position import switch_test as s


OUT=s.ROOT/"docs/evidence/finish-position-2026-11-03"
STAGES=("B000","B001","B002","B004","B008")
ARMS=("midpoint","interior","first")


def collect():
    refs={}; rows=[]; identities={}
    def read(path):
        refs[str(path.relative_to(s.ROOT))]=s.sha(path)
        return json.loads(path.read_text(encoding="utf-8"))
    for seed in range(3):
        for arm in ARMS:
            eval_folder="eval-interior-v1" if arm=="interior" else "eval-v1"
            ev=s.BASE/eval_folder/f"seed_{seed:03d}"
            snapshots=[]
            for stage in STAGES:
                r=read(ev/f"{arm}_{stage}_B.json")
                assert identities.setdefault((seed,"B"),r["primary_identity_hash"])==r["primary_identity_hash"]
                c=r["counts"]
                snapshots.append(dict(stage=stage,new=r["actual_unique_B_NEW"],total=c["matured"],
                    clean=c["clean_completed"],ordinary=c["completed"]))
            x=[r["new"] for r in snapshots[:4]]
            y=[100*r["clean"]/r["total"] for r in snapshots[:4]]
            assert x[0]==0 and all(l<r for l,r in zip(x,x[1:]))
            area=sum((b-a)*(v+w)/2 for a,b,v,w in zip(x,x[1:],y,y[1:]))/x[-1]
            r=read(ev/f"{arm}_B008_mixed.json")
            assert identities.setdefault((seed,"mixed"),r["primary_identity_hash"])==r["primary_identity_hash"]
            diag=read(s.BASE/eval_folder/f"timing/seed_{seed:03d}/{arm}_B008_mixed.json")
            timing={v["family"]:v["counts"] for v in diag["by_family"]}
            for role,k in (("A","history_family"),("B","B_family")):
                c=r["strata"][str(r[k])]["conflict"]["counts"]
                assert (c["matured"],c["completed"],c["clean_completed"])==tuple(timing[role][i] for i in ("matured","completed","strict"))
            pair_folder=s.BASE/"interior-checks-v1/pairs" if arm=="interior" else s.OUT
            history_folder=s.BASE/"interior-checks-v1/history" if arm=="interior" else s.BASE/"boundary-v1"
            p=read(pair_folder/f"seed_{seed:03d}_{arm}.json")
            assert identities.setdefault((seed,"pairs"),p["events_sha"])==p["events_sha"]
            q=read(history_folder/f"{arm}_{seed}.json"); assert identities.setdefault((seed,"teacher"),q["facts_sha"])==q["facts_sha"]
            rows.append(dict(seed=seed,arm=arm,curve=snapshots,early_B=area,timing=timing,
                idle=r["counts"]["idle_actions"],idle_n=r["counts"]["idle_opportunities"],
                pairs=p["pair_counts"],reference_scores=q["counts"]))
    agg={}
    for arm in ARMS:
        rr=[r for r in rows if r["arm"]==arm]
        entry=dict(early_B=mean(r["early_B"] for r in rr),
                   final_B=mean(100*r["curve"][-1]["clean"]/r["curve"][-1]["total"] for r in rr),
                   idle=mean(1000*r["idle"]/r["idle_n"] for r in rr))
        for role in ("A","B"):
            cs=[r["timing"][role] for r in rr]
            entry[role]={k:sum(c[k] for c in cs) for k in cs[0]}
            entry[role]["strict_mean"]=mean(100*c["strict"]/c["matured"] for c in cs)
        pair=[r["pairs"]["conflict_switch"] for r in rr]
        entry["pairs"]={k:sum(c[k] for c in pair) for k in pair[0]}
        agg[arm]=entry
    return dict(rows=rows,aggregate=agg,source_files=refs)


def draw(d):
    plt.rcParams.update({"font.family":"Microsoft YaHei","font.size":10,"axes.unicode_minus":False})
    fig,axes=plt.subplots(1,3,figsize=(12.9,4.45))
    fig.subplots_adjust(top=.76,bottom=.23,left=.06,right=.98,wspace=.27)
    fig.text(.045,.94,"把结束动作教在窗口靠前的位置，会怎样？",fontsize=16,weight="bold")
    fig.text(.045,.864,"三组都在就绪前最后一步监督 WAIT。线和柱为三个种子的均值，右图圆点是逐种子结果。",color="#425461")
    designs=(("midpoint","原教师时刻","#376e86"),("interior","两者之间","#7162a1"),("first","首次合法","#b45b25"))
    for i,(arm,label,color) in enumerate(designs):
        rows=[r for r in d["rows"] if r["arm"]==arm]
        x=np.mean([[v["new"] for v in r["curve"]] for r in rows],0)/1000
        y=np.mean([[100*v["clean"]/v["total"] for v in r["curve"]] for r in rows],0)
        axes[0].plot(x,y,"o-",color=color,ms=4,lw=1.8,label=label)
        axes[1].bar(np.arange(2)+(i-1)*.245,[d["aggregate"][arm][k]["strict_mean"] for k in ("A","B")],width=.23,color=color)
        pairs=[r["pairs"]["conflict_switch"] for r in rows]
        pct=[100*c["both_strict"]/c["pairs"] for c in pairs]
        axes[2].bar(i,mean(pct),width=.50,color=color,alpha=.37)
        axes[2].scatter(np.arange(3)*.075+i-.075,pct,color=color,s=26)
    axes[0].set(title="新 B 学习曲线",xlabel="累计新 B 请求（千）",ylabel="严格完成率（%）")
    axes[1].set(title="末轮随机交错的冲突请求",xticks=[0,1],xticklabels=["旧 A","新 B"],ylabel="严格完成率（%）")
    axes[2].set(title="同一符号切换情境，两次均无错",xticks=range(3),xticklabels=[v[1] for v in designs],ylabel="成对严格完成率（%）")
    for ax in axes:
        ax.set_ylim(-2,102); ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),ncol=3,loc="lower center",frameon=False)
    fig.savefig(OUT/"finish-position.png",dpi=155,facecolor="white");plt.close(fig)


if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=True); data=collect()
    (OUT/"counts.json").write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf-8")
    draw(data)
