"""Compact, source-bound counts for WAIT supervision, including static-last."""
from collections import Counter
from pathlib import Path
from statistics import mean
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.review_selection.pairs import sha


ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"data/wait-position"
OUT=ROOT/"docs/evidence/wait-position-2026-11-03"
ARMS=("midpoint","random","hard","last")
LABELS=("处理中点","随机位置","最高风险","固定末步")
COLORS=("#566f85","#39847b","#b4521d","#7051a8")
FULL=("B000","B001","B002","B004","B008")


def assemble():
    refs={}
    cases={}
    exposures={}
    details=[]
    def read(path):
        refs[str(path.relative_to(ROOT))]=sha(path)
        return json.loads(path.read_text(encoding="utf8"))
    for seed in range(3):
        for arm in ARMS:
            ev=BASE/("eval-last-v1" if arm=="last" else "eval-v1")
            pair_dir=BASE/("last-checks-v1/pairs" if arm=="last" else "pairs-v1")
            hist_dir=BASE/("last-checks-v1/history" if arm=="last" else "history-v1")
            curves=[]
            for snap in FULL:
                row=read(ev/f"seed_{seed:03d}/{arm}_{snap}_B.json")
                key=seed,"B"
                assert cases.setdefault(key,row["primary_identity_hash"])==row["primary_identity_hash"]
                assert exposures.setdefault((seed,snap),row["actual_unique_B_NEW"])==row["actual_unique_B_NEW"]
                c=row["counts"]
                curves.append(dict(stage=snap,new=row["actual_unique_B_NEW"],matured=c["matured"],
                                   strict=c["clean_completed"],ordinary=c["completed"]))
            x=[v["new"] for v in curves[:4]]
            y=[100*v["strict"]/v["matured"] for v in curves[:4]]
            assert x[0]==0 and all(a<b for a,b in zip(x,x[1:]))
            early=sum((b-a)*(i+j)/2 for a,b,i,j in zip(x,x[1:],y,y[1:]))/x[-1]
            mix=read(ev/f"seed_{seed:03d}/{arm}_B008_mixed.json")
            key=seed,"mixed"
            assert cases.setdefault(key,mix["primary_identity_hash"])==mix["primary_identity_hash"]
            timing=read(ev/f"timing/seed_{seed:03d}/{arm}_B008.json")
            counts={v["family"]:v["counts"] for v in timing["by_family"]}
            for role,idx in (("A",mix["history_family"]),("B",mix["B_family"])):
                t=counts[role]; b=mix["strata"][str(idx)]["conflict"]["counts"]
                assert (t["matured"],t["completed"],t["strict"])==(b["matured"],b["completed"],b["clean_completed"])
            pair=read(pair_dir/f"seed_{seed:03d}_{arm}.json")
            assert cases.setdefault((seed,"pairs"),pair["events_sha"])==pair["events_sha"]
            hist=next(r for r in read(hist_dir/f"seed_{seed:03d}.json") if r["arm"]==arm)
            details.append(dict(seed=seed,arm=arm,curves=curves,early_B=early,old_and_new=counts,
                idle=mix["counts"]["idle_actions"],idle_n=mix["counts"]["idle_opportunities"],
                pairs=pair["pair_counts"],fixed_history=hist["metrics"]))
    summaries={}
    for arm in ARMS:
        rows=[r for r in details if r["arm"]==arm]
        a,b=([r["old_and_new"][role] for r in rows] for role in ("A","B"))
        pairs=[r["pairs"]["conflict_switch"] for r in rows]
        summaries[arm]=dict(early_B=mean(r["early_B"] for r in rows),
            B_end=mean(100*r["curves"][-1]["strict"]/r["curves"][-1]["matured"] for r in rows),
            mixed_A=mean(100*c["strict"]/c["matured"] for c in a),
            mixed_B=mean(100*c["strict"]/c["matured"] for c in b),
            idle_per1000=mean(1000*r["idle"]/r["idle_n"] for r in rows),
            total_A=dict(sum((Counter(c) for c in a),Counter())),total_B=dict(sum((Counter(c) for c in b),Counter())),
            switch_counts=dict(sum((Counter(c) for c in pairs),Counter())))
    return dict(details=details,summaries=summaries,source_sha256=refs)


def draw(result):
    plt.rcParams.update({"font.family":"Microsoft YaHei","font.size":10,"axes.unicode_minus":False})
    fig,axes=plt.subplots(1,3,figsize=(14,4.9))
    fig.subplots_adjust(top=.76,bottom=.25,left=.055,right=.99,wspace=.25)
    fig.text(.04,.94,"移动一个 WAIT 标签后，少抢答是否换来了更可靠的切换？",weight="bold",fontsize=16)
    fig.text(.04,.872,"复习的旧请求、启动和结束标签及更新预算相同。前两图为三个种子的均值，右图小点为逐种子结果。",color="#566674")
    for i,(arm,label,color) in enumerate(zip(ARMS,LABELS,COLORS)):
        rows=[r for r in result["details"] if r["arm"]==arm]
        x=np.mean([[v["new"] for v in r["curves"]] for r in rows],0)/1000
        y=np.mean([[100*v["strict"]/v["matured"] for v in r["curves"]] for r in rows],0)
        axes[0].plot(x,y,"o-",ms=4,color=color,label=label)
        metrics=result["summaries"][arm]
        axes[1].bar(np.array([0,1])+(i-1.5)*.17,[metrics["mixed_A"],metrics["mixed_B"]],width=.155,color=color)
        ratios=[100*r["pairs"]["conflict_switch"]["both_strict"]/r["pairs"]["conflict_switch"]["pairs"] for r in rows]
        axes[2].bar(i,mean(ratios),width=.6,color=color,alpha=.35)
        axes[2].scatter(np.arange(3)*.08+i-.08,ratios,color=color,s=26)
    axes[0].set(title="独立学习新 B",xlabel="累计新 B 请求（千）",ylabel="全程无错完成率（%）")
    axes[1].set(title="随机交错的冲突请求",xticks=[0,1],xticklabels=["旧 A","新 B"],ylabel="全程无错完成率（%）")
    axes[2].set(title="同槽同符号，换情境后两次都对",xticks=range(4),xticklabels=LABELS,ylabel="成对严格完成率（%）")
    for ax in axes:
        ax.set_ylim(-2,101);ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",alpha=.18)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=4,frameon=False)
    fig.savefig(OUT/"wait-position.png",dpi=150,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    d=assemble()
    (OUT/"counts.json").write_text(json.dumps(d,indent=2,allow_nan=False),encoding="utf8")
    draw(d)
