"""Paired response curves on the difficult symbols, including the reversed target."""
import json
from pathlib import Path
from statistics import mean,stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.learn_remember.gru_train import sha


ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"data/experience-transfer"
EVIDENCE=ROOT/"docs/evidence/experience-transfer-2026-11-03"
SNAPS=("B000","Bfirst002","Bfirst008","B001","B002","B004","B008")
FULL=("B000","B001","B002","B004","B008")
TARGETS=("B","Bprime")
ARMS=("related","disjoint","direct")


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def locate(target,seed,arm,snap):
    if target=="B":
        return BASE/f"eval-v1/seed_{seed:03d}/{arm}_{snap}_B.json"
    folder=BASE/"reciprocal-v1"
    candidates=[p for p in folder.rglob(f"{arm}_{snap}_B*.json") if f"seed_{seed:03d}" in p.parts
                and not p.name.endswith(".requests.json")]
    if len(candidates)!=1:
        raise FileNotFoundError((seed,arm,snap,candidates))
    return candidates[0]


def assemble():
    rows={}
    sources={}
    cases={}
    retention=[]
    for target in TARGETS:
        for seed in range(3):
            source_training="training-v1" if target=="B" else "reciprocal-v1"
            boundary=BASE/source_training/f"seed_{seed:03d}"/"B_boundary.pt"
            starts=torch.load(boundary,map_location="cpu",weights_only=False)
            sources[str(boundary.relative_to(ROOT))]=sha(boundary)
            for arm in ARMS if target=="B" else ARMS[:2]:
                for snap in SNAPS if target=="B" else FULL:
                    p=locate(target,seed,arm,snap)
                    r=read(p)
                    digest=r["primary_identity_hash"]
                    assert digest==cases.setdefault((target,seed),digest)
                    bf=r["B_family"]
                    group=r["strata"][str(bf)]
                    work=r["checkpoint_work"]
                    before_training=int(starts[arm]["world"]["context"]["tensors"]["family_new_counts"][...,bf].sum())
                    sources[str(p.relative_to(ROOT))]=sha(p)
                    for part in ("all","shared","conflict"):
                        c=group[part]["counts"]
                        rows[target,seed,arm,snap,part]=dict(
                            target=target,seed=seed,history=arm,snapshot=snap,group=part,n=c["matured"],
                            correct=c.get("clean_completed",0),ordinary=c.get("completed",0),
                            strict=100*c.get("clean_completed",0)/c["matured"],
                            completed=100*c.get("completed",0)/c["matured"],
                            new_commands=work["B_unique_new_commands"],
                            all_new_commands=work["B_unique_new_commands"]+(before_training if snap!="B000" else 0),
                            distinct_label_positions=work["B_unique_label_positions"],
                            gradient_label_positions=work["B_gradient_label_positions"])
                if target=="B" and arm!="direct":
                    for snap in ("B000","B008"):
                        for scenario in ("history","mixed"):
                            p=BASE/f"eval-v1/seed_{seed:03d}/{arm}_{snap}_{scenario}.json"
                            r=read(p)
                            sources[str(p.relative_to(ROOT))]=sha(p)
                            families=[("old",r["history_family"])]
                            if scenario=="mixed":
                                families.append(("new",r["B_family"]))
                            for role,family in families:
                                for group in ("all","shared","conflict"):
                                    count=r["strata"][str(family)][group]["counts"]
                                    retention.append(dict(seed=seed,history=arm,snapshot=snap,scenario=scenario,
                                        rule=role,group=group,n=count["matured"],strict_correct=count.get("clean_completed",0),
                                        ordinary=count.get("completed",0)))
    paired=[]
    for seed in range(3):
        for part in ("all","shared","conflict"):
            estimates=[]
            for target in TARGETS:
                aligned,unaligned=ARMS[:2] if target=="B" else tuple(reversed(ARMS[:2]))
                x=[]; ds=[]
                for snap in FULL[:4]:
                    a=rows[target,seed,aligned,snap,part]; c=rows[target,seed,unaligned,snap,part]
                    assert a["n"]==c["n"] and a["new_commands"]==c["new_commands"]
                    x.append(a["new_commands"]); ds.append(a["strict"]-c["strict"])
                assert x[0]==0 and all(u<v for u,v in zip(x,x[1:]))
                area=sum(.5*(ds[k+1]+ds[k])*(x[k+1]-x[k]) for k in range(len(x)-1))/(x[-1]-x[0])
                estimates.append(dict(target=target,seed=seed,group=part,initial_pp=ds[0],
                    early_auc_pp=area,early_gain_auc_pp=area-ds[0]))
            paired.extend(estimates)
            paired.append(dict(target="counterbalanced",seed=seed,group=part,
                **{k:mean(v[k] for v in estimates) for k in ("initial_pp","early_auc_pp","early_gain_auc_pp")}))
    return dict(rows=list(rows.values()),paired=paired,retention=retention,input_sha256=sources)


def plot(d):
    plt.rcParams.update({"font.family":"Microsoft YaHei","font.size":10,"axes.unicode_minus":False})
    fig,axes=plt.subplots(1,3,figsize=(13.7,4.8))
    fig.subplots_adjust(left=.062,right=.98,top=.75,bottom=.24,wspace=.22)
    fig.text(.04,.936,"相同的新经历，旧规则帮助了哪一部分？",fontsize=17,weight="bold")
    fig.text(.04,.86,"独立长任务的严格完成率。曲线是三种子均值，色带是范围。同一面板内各臂看到相同的新请求。",color="#455761")
    for ax,(target,group,title) in zip(axes,[("B","shared","B · 有一段前史已会的符号"),
            ("B","conflict","B · 两组都需改答案的符号"),("Bprime","conflict","B′ · 同一批需改答案的符号")]):
        for arm in ARMS if target=="B" else ARMS[:2]:
            match=arm==("related" if target=="B" else "disjoint")
            color,label=("#276ca0","部分重合前史") if match else ("#bd571b","完全不重合前史")
            if arm=="direct": color,label="#647267","前史不更新"
            scores=[]; exposures=[]
            for seed in range(3):
                rs=[next(r for r in d["rows"] if (r["target"],r["seed"],r["history"],r["snapshot"],r["group"])
                    ==(target,seed,arm,snap,group)) for snap in FULL]
                scores.append([r["strict"] for r in rs]);exposures.append([r["new_commands"] for r in rs])
            x=np.mean(exposures,axis=0)/1000
            y=np.array(scores)
            ax.plot(x,y.mean(0),marker="o",ms=4,color=color,ls=":" if arm=="direct" else "-",label=label)
            if arm!="direct":ax.fill_between(x,y.min(0),y.max(0),alpha=.09,color=color)
        ax.set_title(title,fontsize=11);ax.set_ylim(-2,102)
        ax.set_xlabel("训练块内累计新发命令（千）");ax.grid(axis="y",alpha=.2)
        ax.spines[["right","top"]].set_visible(False)
    axes[0].set_ylabel("全程无错完成率（%）")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=3,frameon=False)
    fig.savefig(EVIDENCE/"transfer.png",dpi=150,facecolor="white");plt.close(fig)


if __name__=="__main__":
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    result=assemble()
    (EVIDENCE/"paired-curves.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    plot(result)
