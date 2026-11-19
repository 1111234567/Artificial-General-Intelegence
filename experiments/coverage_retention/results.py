"""Portable exposure, coverage, retention and live-continuation evidence."""
from pathlib import Path
import json
import numpy as np
from experiments.action_split.data import sha

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"data/coverage-retention"
OUT=ROOT/"docs/evidence/coverage-retention-2026-11-07"


def collect():
    pins={}
    def read(path):
        pins[path.relative_to(ROOT).as_posix()]=sha(path)
        return json.loads(path.read_text(encoding="utf8"))
    behavior=[];time=[];raw={};paired=[];summaries=[]
    definitions=(("fresh","eval-v1",("baseline","control256","wide256","withdraw256"),("long","longer","far")),
                 ("live","live-v1",("control","withdraw"),("long",)))
    for mode,folder,policies,conditions in definitions:
        assert read(BASE/f"{folder}/completion.json")["complete"]
        for seed in range(3):
            for condition in conditions:
                for policy in policies:
                    path=BASE/f"{folder}/seed_{seed:03d}/{policy}_{condition}.json"
                    value=read(path);raw[(mode,seed,condition,policy)]=value
                    groups=dict(value["groups"])
                    for fam in ("A","B"):
                        a,b=groups["family_"+fam],groups["family_"+fam+"_conflict"]
                        groups["family_"+fam+"_shared"]={k:a[k]-b[k] for k in a}
                    groups={g:dict(v,strict_rate=v["strict"]/v["matured"],ordinary_rate=v["ordinary"]/v["matured"])
                            for g,v in groups.items()}
                    behavior.append(dict(mode=mode,seed=seed,condition=condition,policy=policy,groups=groups,
                        idle_actions=value["idle_actions"],idle_opportunities=value["idle_opportunities"],
                        idle_per_1000=1000*value["idle_actions"]/value["idle_opportunities"],coverage=value["coverage"]))
                    aux=path.with_name(path.stem+"_time.json");q=read(aux)
                    time.append(dict(mode=mode,seed=seed,condition=condition,policy=policy,groups=q["groups"],boundary=q["boundary"],
                        mature_request_denominator=q["mature_request_denominator"],paired_requests=q["paired_requests"]))
                    pred=aux.with_suffix(".npz");pins[pred.relative_to(ROOT).as_posix()]=sha(pred)
                contrasts=(("wide256","control256"),("withdraw256","control256"),("withdraw256","baseline"),("control256","baseline")) if mode=="fresh" else (("withdraw","control"),)
                for lhs,rhs in contrasts:
                    a,b=raw[(mode,seed,condition,lhs)],raw[(mode,seed,condition,rhs)]
                    key=lambda r:(r["actor"],r["job"],r["rid"])
                    lookup={key(r):r for r in b["requests"]}
                    assert set(lookup)=={key(r) for r in a["requests"]}
                    for group,filtering in (("all",lambda r:True),("A_conflict",lambda r:r["family"]=="A" and r["conflict"]),("B_conflict",lambda r:r["family"]=="B" and r["conflict"])):
                        counts=dict(both=0,left_only=0,right_only=0,neither=0)
                        for r in a["requests"]:
                            if not filtering(r):continue
                            x,y=r["strict"],lookup[key(r)]["strict"]
                            counts["both" if x and y else "left_only" if x else "right_only" if y else "neither"]+=1
                        paired.append(dict(mode=mode,seed=seed,condition=condition,left=lhs,right=rhs,group=group,**counts))
        for condition in conditions:
            for policy in policies:
                rows=[r for r in behavior if(r["mode"],r["condition"],r["policy"])==(mode,condition,policy)]
                result=dict(mode=mode,condition=condition,policy=policy)
                for group in rows[0]["groups"]:
                    entries=[r["groups"][group] for r in rows]
                    g={k:sum(v[k] for v in entries) for k in entries[0] if not k.endswith("_rate")}
                    for metric in ("strict_rate","ordinary_rate"):
                        values=[v[metric] for v in entries];g[metric]=dict(mean=float(np.mean(values)),seed_values=values)
                    result[group]=g
                values=[r["idle_per_1000"] for r in rows]
                result["idle_per_1000"]=dict(mean=float(np.mean(values)),seed_values=values)
                summaries.append(result)
    exposures=[]
    for seed in range(3):
        for arm in ("control","wide","withdraw"):
            value=read(BASE/f"streams-v1/seed_{seed:03d}/{arm}/stream.json")
            exposures.append(dict(seed=seed,arm=arm,counts=value["counts_training"],inherited_A_tail=value["A_inherited_tail"],
                source_cursor=value["source_cursor"],new_warm_ticks=value["warm"]))
    for name in ("streams-v1/protocol.json","streams-v1/completion.json","streams-v1/audit_completion.json",
                 "streams-v1/inherited_A_tail_audit.json","train-v1/protocol.json","train-v1/globalfreeze.json",
                 "train-audit-v1/audit.json","eval-v1/protocol.json","live-v1/protocol.json",
                 "eval-audit-v1/metrics.json","eval-audit-v1/replay.json","live-audit-v1/metrics.json","live-audit-v1/replay.json"):
        read(BASE/name)
    signatures=None
    if (BASE/"error-signature-v1/summary.json").exists():
        signatures=read(BASE/"error-signature-v1/summary.json")
        read(BASE/"error-signature-v1/protocol.json")
        read(BASE/"error-signature-v1/parser_amendment_001.json")
        read(BASE/"error-signature-audit-v1/audit.json")
    pins[Path(__file__).relative_to(ROOT).as_posix()]=sha(__file__)
    out=dict(description="Rates equal3modelseed means; counts pooled. Fresh and live are distinct case sets and cannot be paired across modes. Wide16..32 is trainedcoverage;32..64 remains unseen. Full trainingstate and liveworld/h were preserved.",
        behavior=behavior,summary=summaries,time=time,paired_strict=paired,training_exposure=exposures,
        posthoc_error_signatures=signatures,source_files=pins)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"results.json").write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")
    return out


def plot(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"font.size":10})
    policies=("baseline","control256","wide256","withdraw256")
    colors=("#939ca4","#377eb8","#d28a35","#a85a75")
    labels=("Source baseline","Continue A/B, 8–16","Continue A/B, 8–32","Continue B only, 8–16")
    def point(mode,policy,condition,group,metric="strict_rate"):
        row=next(r for r in result["summary"] if(r["mode"],r["policy"],r["condition"])==(mode,policy,condition))
        return np.asarray(row[group][metric]["seed_values"])*100
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout="constrained")
    ax=axes[0,0]
    for j,(policy,color) in enumerate(zip(policies,colors)):
        values=[point("fresh",policy,c,"all") for c in ("long","longer","far")]
        x=np.arange(3)+(j-1.5)*.19
        ax.bar(x,[v.mean() for v in values],.18,color=color)
        for at,v in zip(x,values):ax.scatter(at+np.array([-.025,0,.025]),v,c="#27323c",s=14,zorder=3)
    ax.set_xticks(range(3),["8–16","16–32","32–64"]);ax.set_xlabel("Evaluation duration")
    ax.set_title("Fresh worlds: trained coverage versus new durations")
    ax.set_ylabel("Strict completion, percent");ax.set_ylim(0,105)
    ax=axes[0,1]
    for j,index in enumerate((0,1,3)):
        values=[point("fresh",policies[index],"long",g) for g in ("family_A_conflict","family_B_conflict","shared")]
        x=np.arange(3)+(j-1)*.23
        ax.bar(x,[v.mean() for v in values],.22,color=colors[index])
        for at,v in zip(x,values):ax.scatter(at+np.array([-.03,0,.03]),v,c="#27323c",s=14,zorder=3)
    ax.set_xticks(range(3),["Old A conflicts","B conflicts","Shared answers"])
    ax.set_title("Fresh worlds, 8–16: old-rule retention")
    ax.set_ylabel("Strict completion, percent");ax.set_ylim(0,105)
    ax=axes[1,0]
    for j,(policy,color) in enumerate((("control",colors[1]),("withdraw",colors[3]))):
        values=[point("live",policy,"long",g) for g in ("all","family_A_conflict","family_B_conflict")]
        x=np.arange(3)+(j-.5)*.26
        ax.bar(x,[v.mean() for v in values],.25,color=color)
        for at,v in zip(x,values):ax.scatter(at+np.array([-.035,0,.035]),v,c="#27323c",s=14,zorder=3)
    ax.set_xticks(range(3),["All requests","Old A conflicts","B conflicts"])
    ax.set_title("A returns in the same world and h; weights remain frozen")
    ax.set_ylabel("Strict completion, percent");ax.set_ylim(0,105)
    ax=axes[1,1]
    for j,index in enumerate((0,1,2)):
        rows=[r for r in result["behavior"] if(r["mode"],r["policy"],r["condition"])==("fresh",policies[index],"longer")]
        values=[100*np.mean([r["groups"]["all"][k]/r["groups"]["all"]["matured"] for r in rows]) for k in ("first_error_early","legal_no_finish")]
        ax.bar(np.arange(2)+(j-1)*.23,values,.22,color=colors[index])
    ax.set_xticks(range(2),["First error: premature FINISH","No FINISH in legal window"])
    ax.set_title("16–32: remaining timing failures");ax.set_ylim(0,105);ax.set_ylabel("Percent of matured requests")
    for ax in axes.flat:ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    fig.legend([Patch(color=c) for c in colors],labels,loc="outside lower center",ncol=2,frameon=False)
    fig.suptitle("Continue the full learned state: duration coverage and old-rule withdrawal\nEqual update/input budgets; dots show 3 model seeds. Live and fresh cases are separate.",fontsize=14)
    fig.savefig(OUT/"coverage-retention.png",dpi=170);plt.close(fig)


if __name__=="__main__":plot(collect())
