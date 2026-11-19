"""Extract actual request counts and matched-core comparisons without training."""
from pathlib import Path
import json
import numpy as np
from experiments.action_split.data import sha

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"data/action-split"
OUT=ROOT/"docs/evidence/action-split-2026-11-05"
POLICIES=("original","frozen064","full064","frozen256","full256")


def collect():
    inputs={}
    def read(path):
        inputs[path.relative_to(ROOT).as_posix()]=sha(path)
        return json.loads(path.read_text(encoding="utf8"))
    done=read(BASE/"eval-v1/completion.json")
    assert done["complete"]
    rows=[]; records={}; comparisons=[]
    for s in range(3):
        for condition in ("long","longer"):
            for policy in POLICIES:
                data=read(BASE/f"eval-v1/seed_{s:03d}/{policy}_{condition}.json")
                records[(s,condition,policy)]=data
                for group,stats in data["groups"].items():
                    n=stats["matured"]
                    rows.append(dict(seed=s,condition=condition,policy=policy,group=group,**stats,
                        strict_rate=stats["strict"]/n,ordinary_rate=stats["ordinary"]/n,
                        idle_actions=data["idle_actions"],idle_opportunities=data["idle_opportunities"],
                        idle_per_1000=1000*data["idle_actions"]/data["idle_opportunities"],
                        coverage=data["coverage"]))
            for step in (64,256):
                f,full=[records[(s,condition,a+f"{step:03d}")] for a in ("frozen","full")]
                key=lambda r:(r["actor"],r["job"],r["rid"])
                lookup={key(r):r for r in full["requests"]}
                assert set(lookup)=={key(r) for r in f["requests"]}
                for metric in ("strict","ordinary"):
                    counts=dict(both=0,frozen_only=0,full_only=0,neither=0)
                    for r in f["requests"]:
                        a,b=r[metric],lookup[key(r)][metric]
                        counts["both" if a and b else "frozen_only" if a else "full_only" if b else "neither"]+=1
                    comparisons.append(dict(seed=s,condition=condition,step=step,metric=metric,**counts))
    summary=[]
    for condition in ("long","longer"):
        for policy in POLICIES:
            for group in records[(0,condition,policy)]["groups"]:
                subset=[r for r in rows if (r["condition"],r["policy"],r["group"])==(condition,policy,group)]
                assert len(subset)==3
                item=dict(condition=condition,policy=policy,group=group,seeds=[0,1,2])
                for metric in ("strict_rate","ordinary_rate","idle_per_1000"):
                    values=[r[metric] for r in sorted(subset,key=lambda x:x["seed"])]
                    item[metric]=dict(mean=float(np.mean(values)),seed_values=values)
                for metric in ("matured","ordinary","strict","started","legal_window","no_begin","started_no_window",
                               "legal_no_finish","any_early","first_error_early","first_legal_correct","first_legal_wrong","first_legal_none"):
                    item[metric]=sum(r[metric] for r in subset)
                summary.append(item)
    budgets=[]
    for seed in range(3):
        from experiments.action_split.train import checkpoint_path
        for arm in ("frozen","full"):
            cp=checkpoint_path(seed,arm,256)
            meta=read(cp.with_suffix(".json"))
            budgets.append(dict(seed=seed,arm=arm,metadata=meta))
    for path in (BASE/"train-v1/protocol.json",BASE/"train-v1/globalfreeze.json",
                 BASE/"streams-v1/protocol.json",BASE/"eval-v1/protocol.json",
                 BASE/"eval-v1/freeze.json",BASE/"train-audit-v1/audit.json",
                 BASE/"eval-audit-v1/metrics.json",BASE/"eval-audit-v1/replay.json"):
        read(path)
    for seed in range(3):
        read(BASE/f"streams-v1/seed_{seed:03d}/raw_audit.json")
    inputs[Path(__file__).relative_to(ROOT).as_posix()]=sha(__file__)
    result=dict(description="Means give equal weight to three model seeds. Request counts are pooled. Original retained its pretrained head and received no new training; frozen/full share zero new heads and equal supervised updates.",
        rows=rows,summary=summary,matched_comparisons=comparisons,budgets=budgets,source_files=inputs)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")
    return result


def plot(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"font.size":10})
    colors={"original":"#8a939c","frozen256":"#2f78b6","full256":"#dd7b3b"}
    labels={"original":"Original policy (no new training)","frozen256":"Train new heads only","full256":"Train heads + recurrent core"}
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout="constrained")
    def points(condition,policy,group,metric):
        return [r[metric] for r in result["rows"] if (r["condition"],r["policy"],r["group"])==(condition,policy,group)]
    groups=("all","family_A_conflict","family_B_conflict")
    for ax,condition,title in ((axes[0,0],"long","Strict completion, duration 8–16"),(axes[0,1],"longer","Strict completion, held-out duration 16–32")):
        for i,policy in enumerate(colors):
            x=np.arange(3)+(i-1)*.24
            vals=[np.asarray(points(condition,policy,g,"strict_rate"))*100 for g in groups]
            ax.bar(x,[v.mean() for v in vals],.22,color=colors[policy],label=labels[policy])
            for at,v in zip(x,vals):ax.scatter(at+np.array([-.025,0,.025]),v,c="#1e2933",s=15,zorder=3)
        ax.set_xticks(range(3),["All requests","Old A conflicts","New B conflicts"])
        ax.set_ylabel("Percent");ax.set_ylim(0,100 if condition=="long" else 35)
        ax.set_title(title);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    ax=axes[1,0]
    for arm in ("frozen","full"):
        means=[]
        for step in (64,256):
            vals=np.asarray(points("long",arm+f"{step:03d}","all","strict_rate"))*100
            means.append(vals.mean())
            ax.scatter([step]*3,vals,c=colors[arm+"256"],s=20)
        ax.plot((64,256),means,marker="o",color=colors[arm+"256"],label=labels[arm+"256"])
    ax.set_title("Fixed checkpoints, no best-model selection")
    ax.set_xlabel("Updates");ax.set_ylabel("Strict completion, percent");ax.set_ylim(0,40)
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    ax=axes[1,1]
    metrics=("no_begin","started_no_window","first_error_early","legal_no_finish")
    for i,policy in enumerate(colors):
        x=np.arange(4)+(i-1)*.24
        selected=[r for r in result["rows"] if (r["condition"],r["policy"],r["group"])==("long",policy,"all")]
        vals=[100*np.mean([r[k]/r["matured"] for r in selected]) for k in metrics]
        ax.bar(x,vals,.22,color=colors[policy],label=labels[policy])
    ax.set_xticks(range(4),["No BEGIN","No legal window","First error early","No legal FINISH"])
    ax.set_title("Where requests fail, duration 8–16")
    ax.set_ylabel("Percent of all matured requests");ax.set_ylim(0,105)
    ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    handles,names=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,names,loc="outside lower center",ncol=3,frameon=False)
    fig.suptitle("Separate time/content losses: fixed-core versus full training\nBars and lines: equal means over 3 seeds. Both new policy heads started at zero.",fontsize=14)
    fig.savefig(OUT/"action-split.png",dpi=170)
    plt.close(fig)


if __name__=="__main__":plot(collect())
