"""Standalone figures from the fully audited structural/idle factorial study."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODES=("dense","dense_shrink","fixed","fixed_shrink","retire","grow")
LABELS=("All allowed","All + shrink","Fixed support","Fixed + shrink","Retire","Regrow")
STRENGTHS=(0.,.1,1.)
COLORS=("#63738a","#426893","#b27b31","#c0572d","#268d83","#985a94")


def save(fig,folder,name):
    for ext in ("png","svg"):fig.savefig(folder/f"{name}.{ext}",dpi=180)
    plt.close(fig)


def draw(report,folder):
    if not report["complete"] or report["audit"]["status"]!="passed":raise ValueError("validated report required")
    folder.mkdir(parents=True,exist_ok=True)
    index={(r["mode"],r["idle_lambda"],r["condition"],r["metric"]):r for r in report["main_summary"]}
    fig,grid=plt.subplots(2,2,figsize=(11,8.3));axes=grid.flatten()
    specs=[("iid","strict_completion",100,"Returning A: clean completion (%)","viridis",100),
           ("long_sparse","strict_completion",100,"Sparse B: clean completion (%)","viridis",100),
           ("long_sparse","completion",100,"Sparse B: any completion (%)","viridis",100),
           ("long_sparse","inactive_per_1000_idle",1,"Sparse B: errors / 1,000 idle slots","magma_r",None)]
    for ax,(condition,metric,scale,title,cmap,maximum) in zip(axes,specs):
        values=np.array([[index[(mode,strength,condition,metric)]["mean"]*scale for strength in STRENGTHS] for mode in MODES])
        im=ax.imshow(values,vmin=0,vmax=maximum,cmap=cmap,aspect="auto")
        for i in range(6):
            for j in range(3):
                rgba=im.cmap(im.norm(values[i,j]));light=.299*rgba[0]+.587*rgba[1]+.114*rgba[2]
                ax.text(j,i,f"{values[i,j]:.1f}",ha="center",va="center",color="black" if light>.55 else "white",fontsize=11)
        ax.set_xticks(range(3),["0","0.1","1"]);ax.set_yticks(range(6),LABELS)
        ax.set_xlabel("Additional idle penalty coefficient")
        ax.set_title(title,fontsize=10);fig.colorbar(im,ax=ax,shrink=.8,pad=.025)
    fig.suptitle("Soft shrink, allowed support, and idle supervision",fontsize=14)
    fig.text(.5,.015,"Each cell is the mean of three trained seeds. Higher completion and lower idle error are better. All branches share the same sparse B source per seed.",ha="center",fontsize=8.8)
    fig.tight_layout(rect=(0,.05,1,.94));save(fig,folder,"factorial-outcomes")

    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,metric,title in zip(axes,("strict_completion","completion"),("Sparse B: clean completion","Sparse B: any completion")):
        condition="long_sparse"
        for mode,label,color in zip(MODES,LABELS,COLORS):
            xs=[index[(mode,s,condition,"inactive_per_1000_idle")]["mean"] for s in STRENGTHS]
            ys=[100*index[(mode,s,condition,metric)]["mean"] for s in STRENGTHS]
            ax.plot(xs,ys,color=color,lw=1.2,label=label)
            for s,x,y,mark in zip(STRENGTHS,xs,ys,("o","^","s")):
                a=index[(mode,s,condition,"inactive_per_1000_idle")];b=index[(mode,s,condition,metric)]
                ax.errorbar(x,y,xerr=a["sample_sd"],yerr=100*b["sample_sd"],fmt=mark,color=color,ms=6,capsize=2,lw=.7)
        ax.set_xlabel("Wrong actions / 1,000 idle opportunities")
        ax.set_ylabel("Completion (%)");ax.set_ylim(0,100);ax.set_xlim(left=0)
        ax.set_title(title);ax.grid(alpha=.15);ax.spines[["top","right"]].set_visible(False)
    handles,names=axes[0].get_legend_handles_labels()
    fig.legend(handles,names,frameon=False,ncols=3,fontsize=8,loc="upper center",bbox_to_anchor=(.5,.94))
    fig.suptitle("Task completion and idle-control tradeoffs",fontsize=13,y=.995)
    fig.text(.5,.015,"Circle / triangle / square: penalty 0 / 0.1 / 1. Error bars: sample SD across three trained seeds, not confidence intervals.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.83));save(fig,folder,"completion-idle-tradeoff")

    history={(r["mode"],r["idle_lambda"],r["condition"],r["cause"],r["time_since_terminal_bin"],r["metric"]):r for r in report["idle_by_history_summary"]}
    bins=("[0,1)","[1,2)","[2,4)","[4,8)","[8,16)","[16,32)","[32,inf)")
    fig,axes=plt.subplots(2,3,figsize=(12,6.5),sharex=True)
    for col,cause in enumerate(("completion","cancellation","expiry")):
        for strength,color in zip(STRENGTHS,("#64738a","#d19237","#198e88")):
            for row,metric,scale in ((0,"idle_action_rate",1000),(1,"mean_finish_minus_wait_logit",1)):
                stats=[history.get(("retire",strength,"long_sparse",cause,b,metric)) for b in bins]
                ys=[np.nan if r is None or r["mean"] is None else r["mean"]*scale for r in stats]
                sd=[np.nan if r is None or r["sample_sd"] is None else r["sample_sd"]*scale for r in stats]
                axes[row,col].errorbar(range(len(bins)),ys,yerr=sd,fmt="o-",lw=1.2,capsize=2,color=color,label=f"penalty {strength:g}")
        axes[0,col].set_title(f"After {cause}")
        axes[0,col].set_ylim(bottom=0)
        for row in (0,1):
            ax=axes[row,col];ax.grid(axis="y",alpha=.15);ax.spines[["top","right"]].set_visible(False)
        axes[1,col].axhline(0,color="#333333",ls="--",lw=.8)
        axes[1,col].set_xticks(range(len(bins)),("0–1","1–2","2–4","4–8","8–16","16–32","32+"),rotation=35)
        axes[1,col].set_xlabel("Environment time since terminal event")
    axes[0,0].set_ylabel("Wrong actions / 1,000 idle slots")
    axes[1,0].set_ylabel("Max FINISH logit minus WAIT")
    axes[0,0].legend(frameon=False,fontsize=9)
    fig.suptitle("Retire control: idle age, terminal cause and action preference in sparse B",fontsize=12)
    fig.text(.5,.012,"Conditional opportunities after each cause; means and sample SD across seeds. Empty cells remain absent; long-age cells can have fewer observed seeds.",ha="center",fontsize=8.5)
    fig.tight_layout(rect=(0,.045,1,.94));save(fig,folder,"idle-age-diagnosis")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();draw(json.loads(args.report.read_text(encoding="utf-8")),args.output)
