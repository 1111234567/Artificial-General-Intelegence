"""Portable seed-level evidence for preserved-policy remaining-time learning."""
from pathlib import Path
import json
import numpy as np
from experiments.action_split.data import sha

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"data/remaining-time"
OUT=ROOT/"docs/evidence/remaining-time-2026-11-06"
ARMS=("probe","time","action","joint")
POLICIES=("original",*ARMS)


def collect():
    pins={}
    def read(path):
        pins[path.relative_to(ROOT).as_posix()]=sha(path)
        return json.loads(path.read_text(encoding="utf8"))
    assert read(BASE/"eval-v1/completion.json")["probe_original_all_raw_arrays_exact"]
    assert read(BASE/"common-v1/completion.json")["complete"]
    behavior=[];time=[];comparisons=[]
    for seed in range(3):
        for condition in ("long","longer"):
            raw={}
            for policy in POLICIES:
                r=read(BASE/f"eval-v1/seed_{seed:03d}/{policy}_{condition}.json")
                raw[policy]=r
                groups={name:dict(g,strict_rate=g["strict"]/g["matured"],ordinary_rate=g["ordinary"]/g["matured"])
                        for name,g in r["groups"].items()}
                behavior.append(dict(seed=seed,condition=condition,policy=policy,groups=groups,coverage=r["coverage"],
                    idle_actions=r["idle_actions"],idle_opportunities=r["idle_opportunities"],
                    idle_per_1000=1000*r["idle_actions"]/r["idle_opportunities"]))
                if policy=="original":continue
                for history,folder,suffix in (("own","eval-v1","_time"),("teacher","common-v1","")):
                    path=BASE/f"{folder}/seed_{seed:03d}/{policy}_{condition}{suffix}.json"
                    q=read(path)
                    time.append(dict(seed=seed,condition=condition,policy=policy,history=history,
                        groups=q["groups"],boundary=q["boundary"],
                        mature_request_denominator=q["mature_request_denominator"],paired_requests=q["paired_requests"]))
                    pred=path.with_suffix(".npz");pins[pred.relative_to(ROOT).as_posix()]=sha(pred)
            for lhs,rhs in (("time","probe"),("joint","action")):
                key=lambda x:(x["actor"],x["job"],x["rid"])
                other={key(r):r for r in raw[rhs]["requests"]}
                assert set(other)=={key(r) for r in raw[lhs]["requests"]}
                counts=dict(both=0,left_only=0,right_only=0,neither=0)
                for r in raw[lhs]["requests"]:
                    a,b=r["strict"],other[key(r)]["strict"]
                    counts["both" if a and b else "left_only" if a else "right_only" if b else "neither"]+=1
                comparisons.append(dict(seed=seed,condition=condition,left=lhs,right=rhs,**counts))
    summaries=[];time_summaries=[]
    for condition in ("long","longer"):
        for policy in POLICIES:
            b=[r for r in behavior if (r["condition"],r["policy"])==(condition,policy)]
            s=dict(condition=condition,policy=policy)
            for group in ("all","family_A_conflict","family_B_conflict"):
                s[group]={metric:dict(mean=float(np.mean([r["groups"][group][metric] for r in b])),
                    seed_values=[r["groups"][group][metric] for r in b]) for metric in ("strict_rate","ordinary_rate")}
                for k in ("matured","strict","ordinary","first_error_early","any_early","legal_no_finish","no_begin","started_no_window"):
                    s[group][k]=sum(r["groups"][group][k] for r in b)
            s["idle_per_1000"]=dict(mean=float(np.mean([r["idle_per_1000"] for r in b])),seed_values=[r["idle_per_1000"] for r in b])
            summaries.append(s)
            if policy=="original":continue
            for history in ("own","teacher"):
                rows=[r for r in time if (r["condition"],r["policy"],r["history"])==(condition,policy,history)]
                result=dict(condition=condition,policy=policy,history=history,groups={},boundary={})
                for group in rows[0]["groups"]:
                    values=[r["groups"][group] for r in rows]
                    g=dict(rows=sum(v["rows"] for v in values))
                    for metric in ("mae_physical","rmse_physical","bias_physical","mae_ticks"):
                        vals=[v[metric] for v in values]
                        g[metric]=dict(mean=float(np.mean(vals)) if all(v is not None for v in vals) else None,seed_values=vals)
                    for metric in ("conditional_content_correct","predicted_ready","actual_ready","sign_correct"):
                        g[metric]=sum(v[metric] for v in values)
                    result["groups"][group]=g
                for group in rows[0]["boundary"]:
                    values=[r["boundary"][group] for r in rows]
                    g={k:sum(v[k] for v in values) for k in values[0]}
                    for metric in ("both_sign_correct","first_content_correct","direction_correct"):
                        vals=[v[metric]/v["pairs"] if v["pairs"] else None for v in values]
                        g[metric+"_rate"]=dict(mean=float(np.mean(vals)) if all(v is not None for v in vals) else None,seed_values=vals)
                    result["boundary"][group]=g
                time_summaries.append(result)
    for name in ("train-v1/protocol.json","train-v1/globalfreeze.json","train-audit-v1/audit.json",
                 "targets-v1/protocol.json","targets-v1/completion.json","eval-v1/protocol.json","eval-v1/freeze.json",
                 "eval-audit-v1/replay.json","eval-audit-v1/metrics.json","eval-audit-v1/auxiliary_metrics.json",
                 "eval-audit-v1/common_history_metrics.json","common-v1/protocol.json"):
        read(BASE/name)
    pins[Path(__file__).relative_to(ROOT).as_posix()]=sha(__file__)
    result=dict(description="Percentages and scalar errors are equal means of 3 model seeds; counts are pooled. Original and probe native trajectories exactly match. Time sidecars never control actions. Teacher and own histories remain separate.",
        behavior=behavior,summary=summaries,time=time,time_summary=time_summaries,matched_strict_comparisons=comparisons,source_files=pins)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")
    return result


def plot(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"font.size":10})
    colors=("#8a949e","#d98638","#367cba","#458e73")
    labels=("Time readout only","Time -> core","Action only","Action + time")
    fig,axes=plt.subplots(2,2,figsize=(12.5,9),layout="constrained")
    def b(condition,arm):return next(x for x in result["summary"] if (x["condition"],x["policy"])==(condition,arm))
    def t(arm):return next(x for x in result["time_summary"] if (x["condition"],x["policy"],x["history"])==("long",arm,"teacher"))
    for ax,condition in ((axes[0,0],"long"),(axes[1,1],"longer")):
        vals=[np.asarray(b(condition,a)["all"]["strict_rate"]["seed_values"])*100 for a in ARMS]
        ax.bar(range(4),[v.mean() for v in vals],color=colors)
        for i,v in enumerate(vals):ax.scatter(i+np.array([-.05,0,.05]),v,c="#23313a",s=20,zorder=3)
        ax.set_xticks(range(4),labels,rotation=10)
        ax.set_ylabel("Strict completion, percent");ax.set_ylim(0,100 if condition=="long" else 20)
        ax.set_title("Native free actions / duration "+("8–16" if condition=="long" else "16–32"))
    ax=axes[0,1]
    for j,(arm,color,label) in enumerate(zip(ARMS,colors,labels)):
        values=[t(arm)["groups"][g]["mae_physical"]["seed_values"] for g in ("all_valid","near_boundary")]
        x=np.arange(2)+(j-1.5)*.18
        ax.bar(x,[np.mean(v) for v in values],.17,color=color,label=label)
        for at,v in zip(x,values):ax.scatter(at+np.array([-.025,0,.025]),v,c="#23313a",s=14,zorder=3)
    ax.set_xticks(range(2),["All active started rows","Within 0.25 of readiness"])
    ax.set_ylabel("Time MAE, physical units (lower is better)")
    ax.set_title("Identical held-out teacher histories / duration 8–16")
    ax=axes[1,0]
    values=[np.asarray(t(a)["boundary"]["all"]["first_content_correct_rate"]["seed_values"])*100 for a in ARMS]
    ax.bar(range(4),[v.mean() for v in values],color=colors)
    for at,v in enumerate(values):ax.scatter(at+np.array([-.05,0,.05]),v,c="#23313a",s=20,zorder=3)
    ax.set_xticks(range(4),labels,rotation=10);ax.set_ylim(0,105)
    ax.set_ylabel("Correct version + content, percent")
    ax.set_title("Native action head at first legal tick / same teacher inputs")
    for ax in axes.flat:ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    fig.legend([Patch(color=c) for c in colors],labels,loc="outside lower center",ncol=4,frameon=False)
    fig.suptitle("Preserve the original policy, then learn remaining time\nThree model seeds; readout-only policy is bitwise identical to the original. Auxiliary output never chooses actions.",fontsize=13)
    fig.savefig(OUT/"remaining-time.png",dpi=170)
    plt.close(fig)


if __name__=="__main__":plot(collect())
