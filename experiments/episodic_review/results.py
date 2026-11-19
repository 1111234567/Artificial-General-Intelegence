"""Portable fixed-stage replay and no-review retention evidence."""
from pathlib import Path
import json
import numpy as np
from experiments.action_split.data import sha
from experiments.episodic_review.evaluate import ARMS,POLICIES,TRAINED

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"data/episodic-review"
OUT=ROOT/"docs/evidence/episodic-review-2026-11-07"


def collect():
    pins={}
    def read(path):
        pins[path.relative_to(ROOT).as_posix()]=sha(path)
        return json.loads(path.read_text(encoding="utf8"))
    behavior=[];rules=[];times=[];pairs=[];summaries=[];raw={}
    scopes=(("fresh","eval-v1",POLICIES,("long","longer")),("live","live-v1",TRAINED,("long",)))
    for mode,folder,policies,conditions in scopes:
        assert read(BASE/f"{folder}/completion.json")["complete"]
        for seed in range(3):
            for condition in conditions:
                for policy in policies:
                    path=BASE/f"{folder}/seed_{seed:03d}/{policy}_{condition}.json"
                    r=read(path);raw[(mode,seed,condition,policy)]=r
                    groups=dict(r["groups"])
                    for family in ("A","B"):
                        full,conflict=groups["family_"+family],groups["family_"+family+"_conflict"]
                        groups["family_"+family+"_shared"]={k:full[k]-conflict[k] for k in full}
                    groups={name:dict(g,strict_rate=g["strict"]/g["matured"],ordinary_rate=g["ordinary"]/g["matured"]) for name,g in groups.items()}
                    behavior.append(dict(mode=mode,seed=seed,condition=condition,policy=policy,groups=groups,
                        idle_actions=r["idle_actions"],idle_opportunities=r["idle_opportunities"],
                        idle_per_1000=1000*r["idle_actions"]/r["idle_opportunities"],coverage=r["coverage"]))
                    rule=read(path.with_name(path.stem+"_rule.json"))
                    rules.append(dict(mode=mode,seed=seed,condition=condition,policy=policy,groups=rule["groups"]))
                    aux_path=path.with_name(path.stem+"_time.json");aux=read(aux_path)
                    times.append(dict(mode=mode,seed=seed,condition=condition,policy=policy,groups=aux["groups"],boundary=aux["boundary"],
                        mature_request_denominator=aux["mature_request_denominator"],paired_requests=aux["paired_requests"]))
                    pred=aux_path.with_suffix(".npz");pins[pred.relative_to(ROOT).as_posix()]=sha(pred)
                contrasts=[]
                for stage in (256,320):
                    contrasts.extend((("replay_A"+str(stage),"replay_B"+str(stage)),("replay_A"+str(stage),"none"+str(stage)),("replay_B"+str(stage),"none"+str(stage))))
                if mode=="fresh":
                    contrasts.extend((arm+"320",arm+"256") for arm in ARMS)
                for left,right in contrasts:
                    a,b=raw[(mode,seed,condition,left)],raw[(mode,seed,condition,right)]
                    key=lambda r:(r["actor"],r["job"],r["rid"])
                    lookup={key(r):r for r in b["requests"]}
                    assert set(lookup)=={key(r) for r in a["requests"]}
                    for group,condition_fn in (("all",lambda r:True),("A_conflict",lambda r:r["family"]=="A" and r["conflict"]),("B_conflict",lambda r:r["family"]=="B" and r["conflict"])):
                        counts=dict(both=0,left_only=0,right_only=0,neither=0)
                        for r in a["requests"]:
                            if not condition_fn(r):continue
                            x,y=r["strict"],lookup[key(r)]["strict"]
                            counts["both" if x and y else "left_only" if x else "right_only" if y else "neither"]+=1
                        pairs.append(dict(mode=mode,seed=seed,condition=condition,left=left,right=right,group=group,**counts))
        for condition in conditions:
            for policy in policies:
                selected=[r for r in behavior if(r["mode"],r["condition"],r["policy"])==(mode,condition,policy)]
                summary=dict(mode=mode,condition=condition,policy=policy)
                for group in selected[0]["groups"]:
                    values=[r["groups"][group] for r in selected]
                    out={k:sum(v[k] for v in values) for k in values[0] if not k.endswith("_rate")}
                    for metric in ("strict_rate","ordinary_rate"):
                        v=[row[metric] for row in values];out[metric]=dict(mean=float(np.mean(v)),seed_values=v)
                    summary[group]=out
                v=[r["idle_per_1000"] for r in selected];summary["idle_per_1000"]=dict(mean=float(np.mean(v)),seed_values=v)
                rrows=[r for r in rules if(r["mode"],r["condition"],r["policy"])==(mode,condition,policy)]
                summary["rules"]={}
                for group in rrows[0]["groups"]:
                    values=[r["groups"][group] for r in rrows]
                    counts={k:sum(v["categories"][k] for v in values) for k in values[0]["categories"]}
                    rates={k:dict(mean=float(np.mean([v["category_rates"][k] for v in values])),seed_values=[v["category_rates"][k] for v in values]) for k in counts}
                    summary["rules"][group]=dict(matured=sum(v["matured"] for v in values),categories=counts,rates=rates)
                summaries.append(summary)
    budgets=[]
    for seed in range(3):
        for arm in ARMS:
            primary=read(BASE/f"train-v1/seed_{seed:03d}/{arm}/completed.json")
            suffix=read(BASE/f"retention-train-v1/seed_{seed:03d}/{arm}/completed.json")
            budgets.append(dict(seed=seed,arm=arm,primary=primary,suffix=suffix))
    for name in ("bank-v1/protocol.json","bank-v1/completion.json","train-v1/protocol.json","train-v1/globalfreeze.json",
                 "train-audit-v1/audit.json","retention-streams-v1/protocol.json","retention-streams-v1/completion.json",
                 "retention-train-v1/protocol.json","retention-train-v1/globalfreeze.json","retention-audit-v1/audit.json",
                 "eval-v1/protocol.json","eval-v1/freeze.json","live-v1/protocol.json","live-v1/freeze.json",
                 "eval-audit-v1/metrics.json","eval-audit-v1/replay.json"):
        read(BASE/name)
    runtime_notes={}
    runtime_path=BASE/"runtime-notes/process-priority.json"
    if runtime_path.exists():runtime_notes["priority"]=read(runtime_path)
    pins[Path(__file__).relative_to(ROOT).as_posix()]=sha(__file__)
    result=dict(description="Equal3seed mean rates, pooled counts. Mainstudy256 andfixed64-updateB-onlysuffixstudy320 bothreported. Fresh cases paired acrossstages; live cases paired onlywithinstage. PrimaryA-vsBreview masks/labelschedule matched, notFLOPs-equaltonone.",
        behavior=behavior,summary=summaries,rule_counts=rules,time=times,paired_strict=pairs,budgets=budgets,runtime_notes=runtime_notes,source_files=pins)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")
    return result


def plot(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"font.size":10})
    colors=("#929ba3","#327cb4","#ca8a3d")
    labels=("No review","Review old A","Review old B")
    def row(mode,policy,condition="long"):
        return next(x for x in result["summary"] if(x["mode"],x["policy"],x["condition"])==(mode,policy,condition))
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout="constrained")
    groups=("family_A_conflict","family_B_conflict","all")
    for col,stage in enumerate((256,320)):
        ax=axes[0,col]
        for j,(arm,color) in enumerate(zip(ARMS,colors)):
            values=[100*np.asarray(row("fresh",arm+str(stage))[g]["strict_rate"]["seed_values"]) for g in groups]
            x=np.arange(3)+(j-1)*.23
            ax.bar(x,[v.mean() for v in values],.22,color=color)
            for at,v in zip(x,values):ax.scatter(at+np.array([-.025,0,.025]),v,c="#24323b",s=15,zorder=3)
        ax.set_xticks(range(3),["Old A conflicts","B conflicts","All requests"])
        for index,group in enumerate(groups):
            baseline=100*row("fresh","baseline")[group]["strict_rate"]["mean"]
            ax.plot([index-.39,index+.39],[baseline,baseline],color="#30383e",linestyle=":",linewidth=1.5)
        ax.set_ylim(0,105);ax.set_ylabel("Strict completion, percent")
        ax.set_title("Fresh cases / "+("after 256 updates" if stage==256 else "after 64 more B-only updates"))
    ax=axes[1,0]
    for j,(arm,color) in enumerate(zip(ARMS,colors)):
        values=[100*np.asarray(row("live",arm+str(s))["family_A_conflict"]["strict_rate"]["seed_values"]) for s in (256,320)]
        x=np.arange(2)+(j-1)*.23
        ax.bar(x,[v.mean() for v in values],.22,color=color)
        for at,v in zip(x,values):ax.scatter(at+np.array([-.025,0,.025]),v,c="#24323b",s=15,zorder=3)
    ax.set_xticks(range(2),["Main endpoint","After B-only suffix"])
    ax.set_title("A returns in the same world and h")
    ax.set_ylim(0,105);ax.set_ylabel("Old A conflict strict completion, percent")
    ax=axes[1,1]
    for j,(arm,color) in enumerate(zip(ARMS,colors)):
        values=[100*np.asarray(row("fresh",arm+str(s))["rules"]["A_conflict"]["rates"]["other_rule_substitution"]["seed_values"]) for s in (256,320)]
        x=np.arange(2)+(j-1)*.23
        ax.bar(x,[v.mean() for v in values],.22,color=color)
        for at,v in zip(x,values):ax.scatter(at+np.array([-.025,0,.025]),v,c="#24323b",s=15,zorder=3)
    ax.set_xticks(range(2),["Main endpoint","After B-only suffix"])
    ax.set_title("Old A answered by B rule / first attempt in time window")
    ax.set_ylim(0,105);ax.set_ylabel("Percent of all A conflict requests")
    for ax in axes.flat:ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    fig.legend([*[Patch(color=c) for c in colors],Line2D([0],[0],color="#30383e",linestyle=":")],
        [*labels,"Source baseline (top panels)"],loc="outside lower center",ncol=4,frameon=False)
    fig.suptitle("Fixed old-experience rehearsal and a prospective no-review suffix\nBars: equal means over 3 seeds. Dots: individual seeds. World and live hidden state remain continuous.",fontsize=14)
    fig.savefig(OUT/"episodic-review.png",dpi=170);plt.close(fig)


if __name__=="__main__":plot(collect())
