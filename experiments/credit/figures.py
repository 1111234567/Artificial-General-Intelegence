"""Standalone scientific figures from audited, seed-level aggregates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def probe_figure(source, folder):
    with Path(source).open(encoding="utf-8",newline="") as stream:
        rows=list(csv.DictReader(stream))
    panels=(("bp64","long_sparse","BP64 / waits 4–8, sparse input"),
            ("bp64","longer","BP64 / waits 8–16"),
            ("coact_mix","longer","Previous coactivity / waits 8–16"))
    views=(("raw_h","accuracy","Independent raw h","#2475ae"),
           ("tanh_h","accuracy","Independent tanh(h)","#8c63b7"),
           ("tanh_h","native_accuracy","Original action head","#454f5c"),
           ("obs96","accuracy","Current input only","#d9a52c"))
    phases=("cue","waiting","legal")
    evidence=[]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,3,figsize=(14,5.2),sharey=True)
    for axis,(arm,condition,title) in zip(axes,panels):
        for vi,(view,metric,label,color) in enumerate(views):
            means,deviations=[],[]
            for phase in phases:
                selected=[r for r in rows if r["arm"]==arm and r["scenario"]==f"within/{condition}"
                          and r["feature"]==view and r["target"]=="joint" and r["control"]=="none"
                          and r["phase"]==phase and r["slice"]=="all" and r["metric"]==metric]
                if len(selected)!=1 or int(selected[0]["n"])!=3:
                    raise ValueError(f"missing complete plot cell: {arm}/{condition}/{view}/{phase}/{metric}")
                row=selected[0]
                means.append(100*float(row["mean"]))
                deviations.append(100*float(row["sample_sd"]))
                evidence.append(row)
            axis.bar(np.arange(3)+(vi-1.5)*.19,means,width=.18,yerr=deviations,capsize=2,
                     label=label,color=color,linewidth=0,error_kw={"elinewidth":1})
        axis.axhline(12.5,color="#999999",linestyle="--",linewidth=.8)
        axis.set_xticks(range(3),["Cue","Waiting","Legal action"])
        axis.set_title(title,fontsize=11,pad=12)
        axis.set_ylim(0,105)
        axis.set_axisbelow(True)
        axis.grid(axis="y",alpha=.16)
    axes[0].set_ylabel("Version + payload accuracy (%)")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc="upper center",bbox_to_anchor=(.5,.91),frameon=False)
    fig.suptitle("Immediate content is accessible; far waits remain difficult",fontsize=15,x=.5,y=.98)
    fig.text(.06,.035,"Held-out actors; 3 training seeds; mean ± sample SD. Dashed line: 12.5% chance.\n"
             "Within-condition supervised probes. Phase samples are different populations, not paired retention rates.",fontsize=9,color="#4d5968")
    fig.subplots_adjust(left=.06,right=.985,bottom=.19,top=.77,wspace=.12)
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/"independent-readout.png"
    fig.savefig(path,dpi=160)
    fig.savefig(folder/"independent-readout.svg")
    plt.close(fig)
    (folder/"independent-readout.data.json").write_text(json.dumps(dict(source=str(Path(source).resolve()),
        source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),cells=evidence),indent=2),encoding="utf-8")
    return str(path)


def tangent_figure(source,folder):
    data=json.loads(Path(source).read_text(encoding="utf-8"))
    rows=data["by_horizon"]
    x=np.array([r["horizon_ticks"] for r in rows])
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),sharey=True)
    selected=[]
    for ax,suffix,title in zip(axes,("hidden_l2","tanh_feature_l2"),("Hidden state sensitivity","Readout-feature sensitivity")):
        for kind,label,color in (("exact","Complete Heun tangent","#2475ae"),("diagonal","Diagonal approximation","#c2574c")):
            values=[r["metrics"][f"{kind}_{suffix}"] for r in rows]
            median=np.array([v["median"] for v in values])
            low=np.array([v["q25"] for v in values])
            high=np.array([v["q75"] for v in values])
            ax.plot(x,median,"o-",label=label,color=color,markersize=4)
            ax.fill_between(x,low,high,color=color,alpha=.15)
            selected.append(dict(metric=f"{kind}_{suffix}",values=values))
        ax.set_yscale("log")
        ax.set_title(title,fontsize=11)
        ax.set_xlabel("Elapsed ticks (1 tick = 0.25 time units)")
        ax.grid(alpha=.2)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("L2 norm of propagated unit perturbation")
    axes[0].legend(frameon=False,fontsize=9)
    fig.suptitle("Dropping recurrent paths sharply shortens local sensitivity",fontsize=13)
    fig.text(.07,.035,"Independent pilot seed 101; fixed weights and recorded inputs; float64 reference.\n"
             "Median and interquartile range across 128 direction trajectories. This is not a content-memory score.",fontsize=8.7,color="#4d5968")
    fig.subplots_adjust(left=.08,right=.98,top=.81,bottom=.24,wspace=.14)
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/"tangent-paths.png"
    fig.savefig(path,dpi=160)
    fig.savefig(folder/"tangent-paths.svg")
    plt.close(fig)
    (folder/"tangent-paths.data.json").write_text(json.dumps(dict(source=str(Path(source).resolve()),
        source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),ticks=x.tolist(),cells=selected),indent=2),encoding="utf-8")
    return str(path)


def study_figures(source,folder):
    data=json.loads(Path(source).read_text(encoding="utf-8"))
    if data["audit"]["status"]!="passed" or data["counts"]["training_runs"]!=33:
        raise ValueError("figures require the complete audited primary study")
    modes=("frozen","readout_only","bp64","bp_replay","diag_sym","diag_sym_replay",
           "diag_random","diag_random_replay","diag_reset64","diag_instant","diag_shuffle")
    labels=("Frozen","Readout only","BPTT 64","BPTT + replay","Diagonal credit","Diagonal + replay",
            "Random feedback","Random feedback + replay","Trace reset / 64 ticks","Trace reset / tick","Shuffled feedback")
    conditions=("iid","long_wait","longer","sparse","long_sparse")
    condition_labels=("Old IID\nwaits 1–4","Long waits\n4–8","Far waits\n8–16","Sparse input\nwaits 1–4","New sparse\nwaits 4–8")
    values=np.zeros((len(modes),len(conditions)))
    selected=[]
    for i,mode in enumerate(modes):
        for j,condition in enumerate(conditions):
            rows=[r for r in data["main_summary"] if r["mode"]==mode and r["condition"]==condition and r["metric"]=="strict_completion"]
            if len(rows)!=1 or rows[0]["n"]!=3:
                raise ValueError("missing independent seeds in primary figure")
            values[i,j]=100*rows[0]["mean"]
            selected.append(rows[0])
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(10,7.7))
    mapped=ax.imshow(values,cmap="YlGnBu",vmin=0,vmax=100,aspect="auto")
    ax.set_xticks(range(5),condition_labels)
    ax.set_yticks(range(11),labels)
    ax.tick_params(length=0,pad=8)
    for i in range(11):
        for j in range(5):
            ax.text(j,i,f"{values[i,j]:.1f}",ha="center",va="center",color="white" if values[i,j]>54 else "#202b37",fontsize=11)
    fig.colorbar(mapped,ax=ax,shrink=.7,label="Strict autonomous completion (%)",pad=.03)
    ax.set_title("Forward credit and replay in a continuous controller",pad=18,fontsize=14)
    fig.text(.03,.025,"Three seeds; means shown. Fixed final budgets, complete persistent state.\n"
             "See CSV/report for sample SD and paired contrasts. Oracle forks are excluded.",fontsize=9,color="#4d5968")
    fig.subplots_adjust(left=.28,right=.96,bottom=.15,top=.91)
    paths=[folder/"primary-completion.png"]
    fig.savefig(paths[-1],dpi=160)
    fig.savefig(folder/"primary-completion.svg")
    plt.close(fig)
    (folder/"primary-completion.data.json").write_text(json.dumps(dict(source=str(Path(source).resolve()),
        source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),cells=selected),indent=2),encoding="utf-8")
    if data["cycles"]["status"]=="complete":
        cycle_modes=("readout_only","bp64","bp_replay","diag_sym","diag_sym_replay")
        cycle_labels=("Readout only","BPTT","BPTT + replay","Diagonal","Diagonal + replay")
        colors=("#777777","#287cb6","#36a17a","#c65b53","#9365aa")
        fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
        cells=[]
        for ax,condition,title in zip(axes,("iid","long_sparse"),("Old A environment","New B environment")):
            for mode,label,color in zip(cycle_modes,cycle_labels,colors):
                y,err=[],[]
                for segment in range(8):
                    rows=[r for r in data["cycles"]["summary"] if r["mode"]==mode and r["condition"]==condition
                          and r["metric"]=="strict_completion" and r["offset"]==4096 and r["segment"]==segment]
                    if len(rows)!=1 or rows[0]["n"]!=3:
                        raise ValueError("missing cycle seeds")
                    cells.append(rows[0])
                    y.append(100*rows[0]["mean"])
                    err.append(100*rows[0]["sample_sd"])
                ax.errorbar(np.arange(8),y,yerr=err,fmt="o-",capsize=2,color=color,label=label,markersize=4,linewidth=1.6)
            for segment in range(0,8,2):
                ax.axvspan(segment-.5,segment+.5,color="#d7e5f0",alpha=.3,zorder=0)
            ax.set_xticks(np.arange(8),[f"{name}{i//2+1}" for i,name in enumerate(["A","B"]*4)])
            ax.set_ylim(0,103)
            ax.set_title(title)
            ax.set_xlabel("Training segment (4096 ticks each)")
            ax.grid(axis="y",alpha=.15)
            ax.spines[["top","right"]].set_visible(False)
        axes[0].set_ylabel("Strict autonomous completion (%)")
        handles,legend_labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,legend_labels,ncol=5,loc="upper center",bbox_to_anchor=(.5,.91),frameon=False)
        fig.suptitle("Repeated A/B experience: retention and reacquisition",fontsize=14,y=.98)
        cycle_device=data["cycles"].get("evaluation_device","cuda").upper()
        fig.text(.07,.035,"Segment endpoints; mean ± sample SD across 3 seeds. Full state and optimizer continue.\n"
                 f"All within-cycle evaluations use {cycle_device}. Familiar environments; not a new-task plasticity test.",fontsize=9,color="#4d5968")
        fig.subplots_adjust(left=.07,right=.98,bottom=.2,top=.74,wspace=.1)
        paths.append(folder/"revisitation.png")
        fig.savefig(paths[-1],dpi=160)
        fig.savefig(folder/"revisitation.svg")
        plt.close(fig)
        (folder/"revisitation.data.json").write_text(json.dumps(dict(source=str(Path(source).resolve()),
            source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),cells=cells),indent=2),encoding="utf-8")
    return list(map(str,paths))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-summary",type=Path)
    parser.add_argument("--tangent-summary",type=Path)
    parser.add_argument("--study-summary",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if not args.probe_summary and not args.tangent_summary and not args.study_summary:
        parser.error("at least one source is required")
    if args.probe_summary:
        print(probe_figure(args.probe_summary,args.output))
    if args.tangent_summary:
        print(tangent_figure(args.tangent_summary,args.output))
    if args.study_summary:
        print(study_figures(args.study_summary,args.output))
