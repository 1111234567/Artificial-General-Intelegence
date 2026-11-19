"""Aggregate all fixed endpoints; no model selection or extra training."""
from pathlib import Path
import csv
import json
import shutil
import numpy as np
import torch
from experiments.associative_state.train import ROOT, OUT, ARMS, sha, write_json
from experiments.associative_state.evaluate import CONDITIONS
from experiments.associative_state.model import AssociationMemory

EVIDENCE=ROOT/"docs/evidence/associative-state-2026-11-07"


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def rate(values,numerator,denominator):
    rates=[v[numerator]/v[denominator] for v in values if v[denominator]]
    return dict(mean=float(np.mean(rates)) if rates else None,
                by_seed=[v[numerator]/v[denominator] if v[denominator] else None for v in values],
                numerator=sum(v[numerator] for v in values),denominator=sum(v[denominator] for v in values))


def aggregate():
    torch.set_num_threads(1)
    audit=read(OUT/"audit.json")
    assert audit["status"]=="passed"
    groups={};records=[];bindings={};revision_splits=[]
    for arm in ARMS:
        for condition in CONDITIONS:
            reports=[]
            for seed in range(3):
                base=OUT/f"seed_{seed:03d}"/arm/"evaluation"
                assert (base/"completion.json").exists()
                path=base/f"{condition}.json";r=read(path);reports.append(r)
                bindings[str(path.relative_to(ROOT))]=sha(path)
                records.append({k:v for k,v in r.items() if k not in ("requests","right_censored")})
                if condition=="revise":
                    for changed in (True,False):
                        selected=[x for x in r["requests"] if x["family"]=="A" and ((x["raw"] in (2,3))==changed)]
                        revision_splits.append(dict(seed=seed,arm=arm,changed=changed,n=len(selected),
                            strict=sum(x["strict"] for x in selected),ordinary=sum(x["ordinary"] for x in selected),
                            translation_correct=sum(x["translation_correct"] for x in selected)))
            row={}
            for group in reports[0]["groups"]:
                vals=[r["groups"][group] for r in reports]
                row[group]={metric:rate(vals,metric,"n") for metric in ("strict","ordinary","translation_correct")}
            row["idle"]=rate(reports,"idle_actions","idle_opportunities")
            row["memory_norm_by_seed"]=[r["memory_norm"] for r in reports]
            row["coverage_min_by_seed"]=[r["coverage_min"] for r in reports]
            groups[f"{arm}/{condition}"]=row
    # Post-hoc direct read of all eight addresses, with the unmodified native reader.
    # This is explicitly diagnostic and cannot change model/checkpoint selection.
    address_checks=[]
    for seed in range(3):
        for arm in ARMS:
            meta=read(OUT/f"seed_{seed:03d}"/arm/"training.json")
            model=AssociationMemory(**meta["model_config"])
            for condition in CONDITIONS:
                cp=torch.load(OUT/f"seed_{seed:03d}"/arm/"evaluation"/f"{condition}.pt",map_location="cpu",weights_only=False)["start"]
                model.load_state_dict(cp["model"])
                truth=cp["world"]["random_mapping"]["mappings"].reshape(-1,8)
                with torch.no_grad():pred=torch.stack([model.read(cp["memory"],torch.full((truth.shape[0],),k,dtype=torch.long)).argmax(-1) for k in range(8)],1)
                address_checks.append(dict(seed=seed,arm=arm,condition=condition,A_correct=int((pred[:,:4]==truth[:,:4]).sum()),
                    B_correct=int((pred[:,4:]==truth[:,4:]).sum()),n_per_family=truth.shape[0]*4))
    result=dict(dataset=str(OUT.relative_to(ROOT)),scope="Frozen executor with supervised symbol adapter; not native autonomous binding or a MemT reproduction",
        aggregate="Equal mean across three seeds; pooled counts listed separately",groups=groups,per_run=records,
        posthoc_direct_address_diagnostic=address_checks,
        posthoc_revision_changed_unchanged=revision_splits,
        revision_split_note="Post-hoc descriptive split of already audited A requests; changed raw 2/3 was fixed by the original mapping revision. No new fitting, scoring, or selection.",
        direct_address_note="Post-hoc native reader over all eight keys. For no_read this diagnostic deliberately calls the reader that the behavioral ablation bypassed; it measures stored/readable information, not that arm's actual translations.",
        inputs=bindings,protocol_sha256=sha(OUT/"protocol.json"),audit_sha256=sha(OUT/"audit.json"),
        parameters={a:read(OUT/"seed_000"/a/"training.json")["parameters"] for a in ARMS})
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    write_json(EVIDENCE/"results.json",result)
    for name in ("protocol.json","training-freeze.json","audit.json"):
        shutil.copyfile(OUT/name,EVIDENCE/name)
    with (EVIDENCE/"summary.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.writer(f);writer.writerow(["arm","condition","A_conflict_strict_pct","B_conflict_strict_pct","all_strict_pct","all_ordinary_pct","idle_per_1000"])
        for key,g in groups.items():
            a,c=key.split("/");writer.writerow([a,c,*[100*g[v]["strict"]["mean"] for v in ("A_conflict","B_conflict","all")],100*g["all"]["ordinary"]["mean"],1000*g["idle"]["mean"]])
    draw(groups)
    print((EVIDENCE/"summary.csv").read_text(encoding="utf-8"))


def draw(groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout="constrained")
    colors={"add":"#b56b31","delta":"#217e76","recurrent":"#6269ab"}
    names={"add":"Coactivation","delta":"Delta","recurrent":"GRU state"}
    conditions=("immediate","delay","adapt","theta_only")
    for arm in ARMS:
        y=[100*groups[f"{arm}/{c}"]["A_conflict"]["strict"]["mean"] for c in conditions]
        axes[0].plot(range(4),y,"o-",label=names[arm],color=colors[arm])
    axes[0].set(xticks=range(4),xticklabels=["After A+B","B writes","B writes\n+ learning","B learning\nstate fixed"],ylim=(0,100),ylabel="Strict completion (%)",title="Old A: separate evaluation branches")
    for x,arm in enumerate(ARMS):
        values=groups[f"{arm}/revise"]["A"]["strict"]
        axes[1].bar(x,100*values["mean"],color=colors[arm],width=.6)
        axes[1].scatter([x]*3,np.array(values["by_seed"])*100,color="#25313b",s=15,zorder=3)
    axes[1].set(xticks=range(3),xticklabels=[names[a] for a in ARMS],ylim=(0,100),title="A after two associations change")
    for arm in ARMS:
        values=[1000*groups[f"{arm}/{c}"]["idle"]["mean"] for c in ("immediate","delay","adapt","revise")]
        axes[2].plot(range(4),values,"o-",color=colors[arm],label=names[arm])
    axes[2].set(xticks=range(4),xticklabels=["After A+B","B writes","B learning","A revised"],ylabel="Actions / 1,000 idle opportunities",title="Idle behavior",ylim=(0,None))
    axes[0].legend(frameon=False,fontsize=9)
    for ax in axes:ax.grid(axis="y",alpha=.18)
    fig.suptitle("64-state association adapters + the same frozen action policy",fontsize=13)
    fig.savefig(EVIDENCE/"associative-state.png",dpi=170)
    plt.close(fig)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--variant",choices=("v1","shared-v2"),default="v1")
    args=parser.parse_args()
    if args.variant=="shared-v2":
        from experiments.associative_state.shared import activate,SharedAssociationMemory
        activate()
        from experiments.associative_state import train as configured
        OUT=configured.OUT
        AssociationMemory=SharedAssociationMemory
        EVIDENCE=EVIDENCE/"shared-v2"
    aggregate()
