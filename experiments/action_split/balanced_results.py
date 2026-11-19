"""Fresh-case numerical report for the single mode-normalization supplement."""
from pathlib import Path
import json
import numpy as np
from experiments.action_split.data import sha
from experiments.action_split.balanced_eval import POLICIES

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"data/action-split"
OUT=ROOT/"docs/evidence/action-split-2026-11-05"


def collect():
    pins={}
    def read(path):
        pins[path.relative_to(ROOT).as_posix()]=sha(path)
        return json.loads(path.read_text(encoding="utf8"))
    assert read(BASE/"balanced-eval-v1/completion.json")["complete"]
    rows=[]; raw={}
    for seed in range(3):
        for condition in ("long","longer"):
            for policy in POLICIES:
                value=read(BASE/f"balanced-eval-v1/seed_{seed:03d}/{policy}_{condition}.json")
                raw[(seed,condition,policy)]=value
                for group,g in value["groups"].items():
                    rows.append(dict(seed=seed,condition=condition,policy=policy,group=group,**g,
                        strict_rate=g["strict"]/g["matured"],ordinary_rate=g["ordinary"]/g["matured"],
                        idle_actions=value["idle_actions"],idle_opportunities=value["idle_opportunities"],
                        idle_per_1000=1000*value["idle_actions"]/value["idle_opportunities"]))
    aggregate=[];pairs=[]
    for condition in ("long","longer"):
        for policy in POLICIES:
            for group in raw[(0,condition,policy)]["groups"]:
                selected=[r for r in rows if (r["condition"],r["policy"],r["group"])==(condition,policy,group)]
                assert len(selected)==3
                a=dict(condition=condition,policy=policy,group=group)
                for k in ("strict_rate","ordinary_rate","idle_per_1000"):
                    v=[r[k] for r in sorted(selected,key=lambda x:x["seed"])]
                    a[k]=dict(mean=float(np.mean(v)),seed_values=v)
                for k in ("matured","strict","ordinary","no_begin","started_no_window","legal_window","legal_no_finish","any_early","first_error_early","first_legal_correct","first_legal_wrong"):
                    a[k]=sum(r[k] for r in selected)
                aggregate.append(a)
        for seed in range(3):
            for arm in ("frozen","full"):
                a,b=(raw[(seed,condition,p+"_"+arm)] for p in ("base","balanced"))
                key=lambda r:(r["actor"],r["job"],r["rid"])
                other={key(r):r for r in b["requests"]}
                assert set(other)=={key(r) for r in a["requests"]}
                counts=dict(both=0,original_loss_only=0,balanced_loss_only=0,neither=0)
                for r in a["requests"]:
                    x,y=r["strict"],other[key(r)]["strict"]
                    counts["both" if x and y else "original_loss_only" if x else "balanced_loss_only" if y else "neither"]+=1
                pairs.append(dict(seed=seed,condition=condition,arm=arm,**counts))
    for path in (BASE/"balanced-v1/protocol.json",BASE/"balanced-v1/globalfreeze.json",
                 BASE/"balanced-eval-v1/protocol.json",BASE/"balanced-eval-v1/freeze.json",
                 BASE/"diagnostic-v1/summary.json",BASE/"diagnostic-v1/legal_rows_audit.json",
                 BASE/"balanced-audit-v1/audit.json"):
        read(path)
    for name in ("metrics.json","replay.json"):
        path=BASE/"balanced-eval-audit-v1"/name
        if path.exists():read(path)
    pins[Path(__file__).relative_to(ROOT).as_posix()]=sha(__file__)
    out=dict(description="One post-failure objective change. All5policies measured on same fresh external requests. Percentages equal modelseed means; counts pooled; requests are not independent training replicates.",
        rows=rows,aggregate=aggregate,paired_strict=pairs,source_files=pins)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"balanced-results.json").write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")
    return out


def plot(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"font.size":10})
    colors=("#8b949d","#8db8db","#f0be8d","#246da7","#cc6a2c")
    labels=("Original policy","Original loss, fixed core","Original loss, full","Balanced loss, fixed core","Balanced loss, full")
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout="constrained")
    for row,condition in enumerate(("long","longer")):
        for col,metric in enumerate(("strict_rate","ordinary_rate")):
            ax=axes[row,col]
            for i,(policy,color,label) in enumerate(zip(POLICIES,colors,labels)):
                x=np.arange(3)+(i-2)*.16
                values=[]
                for group in ("all","family_A_conflict","family_B_conflict"):
                    v=[100*r[metric] for r in data["rows"] if (r["condition"],r["policy"],r["group"])==(condition,policy,group)]
                    values.append(v)
                ax.bar(x,[np.mean(v) for v in values],.145,color=color,label=label)
                for at,v in zip(x,values):ax.scatter(at+np.array([-.02,0,.02]),v,s=10,c="#26323c",zorder=3)
            ax.set_xticks(range(3),["All requests","Old A conflicts","New B conflicts"])
            ax.set_ylabel("Percent");ax.set_ylim(0,105);ax.set_axisbelow(True);ax.grid(axis="y",alpha=.2)
            ax.set_title(("Duration 8–16" if condition=="long" else "Held-out duration 16–32")+" / "+("strict" if col==0 else "eventual")+" completion")
    handles,names=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,names,loc="outside lower center",ncol=3,frameon=False,fontsize=9)
    fig.suptitle("One post-failure control: equal loss per action mode\nFresh shared cases; dots show 3 model seeds. Strict completion excludes earlier request errors.",fontsize=14)
    fig.savefig(OUT/"balanced-action-split.png",dpi=170)
    plt.close(fig)


if __name__=="__main__":plot(collect())
