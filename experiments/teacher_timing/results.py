"""Portable paired behavior, teacher coverage and common-history evidence."""
from pathlib import Path
import json
import numpy as np
from experiments.action_split.data import sha

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"data/teacher-timing"
OUT=ROOT/"docs/evidence/teacher-timing-2026-11-06"
POLICIES=("original","earliest_action","earliest_joint","midpoint_action","midpoint_joint")


def collect():
    pins={}
    def read(path):
        pins[path.relative_to(ROOT).as_posix()]=sha(path)
        return json.loads(path.read_text(encoding="utf8"))
    assert read(BASE/"eval-v1/completion.json")["complete"]
    assert read(BASE/"common-v1/completion.json")["complete"]
    behavior=[];sidecars=[];common=[];paired=[];raw={}
    for seed in range(3):
        for condition in ("long","longer"):
            for policy in POLICIES:
                r=read(BASE/f"eval-v1/seed_{seed:03d}/{policy}_{condition}.json")
                raw[(seed,condition,policy)]=r
                groups={g:dict(v,strict_rate=v["strict"]/v["matured"],ordinary_rate=v["ordinary"]/v["matured"])
                        for g,v in r["groups"].items()}
                behavior.append(dict(seed=seed,condition=condition,policy=policy,groups=groups,coverage=r["coverage"],
                    idle_actions=r["idle_actions"],idle_opportunities=r["idle_opportunities"],
                    idle_per_1000=1000*r["idle_actions"]/r["idle_opportunities"]))
                if policy!="original":
                    path=BASE/f"eval-v1/seed_{seed:03d}/{policy}_{condition}_time.json"
                    q=read(path)
                    sidecars.append(dict(seed=seed,condition=condition,policy=policy,groups=q["groups"],boundary=q["boundary"],
                        mature_request_denominator=q["mature_request_denominator"],paired_requests=q["paired_requests"]))
                    pred=path.with_suffix(".npz");pins[pred.relative_to(ROOT).as_posix()]=sha(pred)
            contrasts=(("midpoint_action","earliest_action"),("midpoint_joint","earliest_joint"),
                       ("midpoint_joint","midpoint_action"),("earliest_joint","earliest_action"))
            for left,right in contrasts:
                a,b=raw[(seed,condition,left)],raw[(seed,condition,right)]
                key=lambda r:(r["actor"],r["job"],r["rid"])
                lookup={key(r):r for r in b["requests"]}
                assert set(lookup)=={key(r) for r in a["requests"]}
                for metric in ("strict","ordinary"):
                    counts=dict(both=0,left_only=0,right_only=0,neither=0)
                    for r in a["requests"]:
                        x,y=r[metric],lookup[key(r)][metric]
                        counts["both" if x and y else "left_only" if x else "right_only" if y else "neither"]+=1
                    paired.append(dict(seed=seed,condition=condition,left=left,right=right,metric=metric,**counts))
        for history in ("earliest","midpoint"):
            for policy in POLICIES[1:]:
                path=BASE/f"common-v1/seed_{seed:03d}/history_{history}__{policy}.json"
                q=read(path)
                common.append(dict(seed=seed,history=history,policy=policy,groups=q["groups"],boundary=q["boundary"],
                    mature_request_denominator=q["mature_request_denominator"],paired_requests=q["paired_requests"]))
                pred=path.with_suffix(".npz");pins[pred.relative_to(ROOT).as_posix()]=sha(pred)
    summaries=[]
    for condition in ("long","longer"):
        for policy in POLICIES:
            rows=[r for r in behavior if(r["condition"],r["policy"])==(condition,policy)]
            entry=dict(condition=condition,policy=policy)
            for group in rows[0]["groups"]:
                v=[r["groups"][group] for r in rows]
                g={k:sum(x[k] for x in v) for k in v[0] if not k.endswith("_rate")}
                for metric in ("strict_rate","ordinary_rate"):
                    values=[x[metric] for x in v]
                    g[metric]=dict(mean=float(np.mean(values)),seed_values=values)
                entry[group]=g
            values=[r["idle_per_1000"] for r in rows]
            entry["idle_per_1000"]=dict(mean=float(np.mean(values)),seed_values=values)
            summaries.append(entry)
    common_summary=[]
    for history in ("earliest","midpoint"):
        for policy in POLICIES[1:]:
            rows=[r for r in common if(r["history"],r["policy"])==(history,policy)]
            c=dict(history=history,policy=policy)
            for group in ("all_valid","near_boundary","legal","processing"):
                values=[r["groups"][group]["mae_physical"] for r in rows]
                c[group+"_mae"]=dict(mean=float(np.mean(values)),seed_values=values)
            for metric in ("both_sign_correct","first_content_correct","direction_correct"):
                values=[r["boundary"]["all"][metric]/r["boundary"]["all"]["pairs"] for r in rows]
                c[metric+"_rate"]=dict(mean=float(np.mean(values)),seed_values=values)
                c[metric]=sum(r["boundary"]["all"][metric] for r in rows)
            c["pairs"]=sum(r["boundary"]["all"]["pairs"] for r in rows)
            common_summary.append(c)
    training_pairs=[read(BASE/f"streams-v1/seed_{s:03d}/pair_comparison.json") for s in range(3)]
    training_boundaries=[]
    for pair in training_pairs:
        counts={}
        for teacher in ("earliest","midpoint"):
            full=pair["finished_request_intersection"]+len(pair["FINISH_request_horizon_only_differences"][teacher])
            post=pair["label_budgets"][teacher]["action_FINISH_rows"]
            counts[teacher]=dict(full_stream_FINISH=full,warm_FINISH=full-post,training_FINISH=post)
        training_boundaries.append(dict(seed=pair["seed"],**counts))
    for name in ("train-v1/protocol.json","train-v1/globalfreeze.json","train-v1/earliest_regression.json",
                 "streams-v1/protocol.json","streams-v1/audit_completion.json",
                 "train-audit-v1/audit.json","eval-v1/protocol.json","eval-v1/freeze.json",
                 "eval-audit-v1/metrics.json","eval-audit-v1/auxiliary_metrics.json","eval-audit-v1/replay.json",
                 "common-v1/protocol.json","common-audit-v1/metrics.json"):
        read(BASE/name)
    phase=None
    path=BASE/"phase-readout-v1/results.json"
    if path.exists():
        phase=read(path)
        read(BASE/"phase-readout-v1/protocol.json")
        read(BASE/"phase-audit-v1/audit.json")
    pins[Path(__file__).relative_to(ROOT).as_posix()]=sha(__file__)
    result=dict(description="All rates are equal means over three model seeds; counts pooled. Teacher-regime comparison changes labels and actual history. Fixed action-row/update budgets, auxiliary row counts differ. Earliest controls bitwise reproduce prior checkpoints.",
        behavior=behavior,summary=summaries,paired=paired,own_time=sidecars,
        common_time=common,common_summary=common_summary,training_pairs=training_pairs,training_boundaries=training_boundaries,
        posthoc_phase=phase,source_files=pins)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")
    return result


def plot(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"font.size":10})
    colors=("#969da4","#a2bfdb","#eab890","#337ab5","#308974")
    names=("Original policy","Earliest / action","Earliest / joint","Midpoint / action","Midpoint / joint")
    def summary(policy,condition="long"):
        return next(r for r in result["summary"] if(r["policy"],r["condition"])==(policy,condition))
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout="constrained")
    for ax,metric,title in ((axes[0,0],"strict_rate","Strict completion / duration 8–16"),
                           (axes[0,1],"ordinary_rate","Eventual completion / duration 8–16")):
        for j,(policy,color) in enumerate(zip(POLICIES,colors)):
            x=np.arange(3)+(j-2)*.16
            values=[100*np.asarray(summary(policy)[g][metric]["seed_values"]) for g in ("all","family_A_conflict","family_B_conflict")]
            ax.bar(x,[v.mean() for v in values],.15,color=color)
            for at,v in zip(x,values):ax.scatter(at+np.array([-.022,0,.022]),v,c="#25323c",s=13,zorder=3)
        ax.set_xticks(range(3),["All requests","Old A conflicts","New B conflicts"])
        ax.set_ylim(0,105);ax.set_title(title);ax.set_ylabel("Percent")
    ax=axes[1,0]
    for j,(policy,color) in enumerate(zip(POLICIES,colors)):
        rows=[r for r in result["behavior"] if(r["policy"],r["condition"])==(policy,"long")]
        x=np.arange(2)+(j-2)*.16
        values=[100*np.mean([r["groups"]["all"][k]/r["groups"]["all"]["matured"] for r in rows]) for k in ("first_error_early","legal_no_finish")]
        ax.bar(x,values,.15,color=color)
    ax.set_xticks(range(2),["First error: premature FINISH","No FINISH in legal window"])
    ax.set_title("Timing failures / duration 8–16");ax.set_ylabel("Percent of matured requests");ax.set_ylim(0,65)
    ax=axes[1,1]
    values=[100*np.asarray(summary(p,"longer")["all"]["strict_rate"]["seed_values"]) for p in POLICIES]
    ax.bar(range(5),[v.mean() for v in values],color=colors)
    for i,v in enumerate(values):ax.scatter(i+np.array([-.045,0,.045]),v,c="#25323c",s=18,zorder=3)
    ax.set_xticks(range(5),["Original","Early\naction","Early\njoint","Middle\naction","Middle\njoint"])
    ax.set_ylim(0,22);ax.set_ylabel("Strict completion, percent");ax.set_title("Held-out duration 16–32 / unchanged gap")
    for ax in axes.flat:ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    fig.legend([Patch(color=c) for c in colors],names,loc="outside lower center",ncol=3,frameon=False)
    fig.suptitle("Actual teacher timing with the same initial policy and update budget\nBars: equal means over 3 seeds. Dots: model seeds. Models act freely at evaluation.",fontsize=14)
    fig.savefig(OUT/"teacher-timing.png",dpi=170)
    plt.close(fig)


if __name__=="__main__":plot(collect())
