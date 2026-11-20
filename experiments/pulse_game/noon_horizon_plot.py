"""Plot the causal calibration and independently observed route switches."""

from argparse import ArgumentParser
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import noon_horizon as env, noon_horizon_train as fit


def load(filename):
    return json.loads((env.EVIDENCE/filename).read_text(encoding="utf8"))


def line(axis, x, readings, color, label, *, scale=1, style="-"):
    means = np.asarray([z["mean"] for z in readings]) * scale
    bands = np.asarray([z["ci95"] for z in readings]) * scale
    axis.fill_between(x,bands[:,0],bands[:,1],color=color,alpha=.14,linewidth=0)
    axis.plot(x,means,style,marker="o",color=color,label=label,linewidth=1.8,markersize=4)


def plot(stage):
    plan = fit.load_plan()
    fig, ax = plt.subplots(1,3,figsize=(12.2,4.35),layout="constrained",width_ratios=(1.15,1,1))
    palette = ("#15688A", "#A3462E")
    ax[0].axhline(0,color="#7C858C",linewidth=.75)
    ax[0].axvline(plan["paid_tail_length"],color="#848F96",linewidth=.9,linestyle=":")
    lengths = list(env.CAL_LENGTHS)
    for j, cohort in enumerate(env.COHORTS):
        scores = load(f"horizon-calibration-{cohort}.json")["scores"]
        reading = [scores[f"{'plus' if h else 'minus'}/{h}/unknown_minus_known"] for h in lengths]
        line(ax[0],lengths,reading,palette[j],cohort.capitalize())
    ax[0].set_xticks(lengths)
    ax[0].set(xlabel="Paid choices after the return",ylabel="Unknown route minus known route (resource)",title="Causal return before training")
    ax[0].legend(loc="lower right",fontsize=8,frameon=False)
    points = [k for k in plan["checkpoints"] if k <= stage]
    for axis, cohort in zip(ax[1:],env.COHORTS,strict=True):
        results = [load(f"horizon-result-{cohort}-{step}.json")["summary"] for step in points]
        x = np.asarray(points)/1000
        axis.axhline(50,color="#8a9297",linewidth=.75)
        for mode,color in zip(("short","long"),palette,strict=True):
            field = f"{mode}/select_unknown_actual"
            cue_scores = [data.get("informative",data.get("frozen"))[field] for data in results]
            blind = [data.get("uninformative",data.get("frozen"))[field] for data in results]
            line(axis,x,cue_scores,color,mode.capitalize()+" · informative",scale=100)
            axis.plot(x,[100*z["mean"] for z in blind],"--",linewidth=1,color=color,alpha=.5,
                      label=mode.capitalize()+" · uninformative")
        axis.set_xticks(x)
        axis.set_ylim(0,100)
        axis.set(xlabel="Adam updates (thousands)",ylabel="Chose the previously tasteless item (%)",title=cohort.capitalize()+" cohort")
    ax[2].legend(loc="best",fontsize=8,frameon=False)
    for a in ax:
        a.spines[["top","right"]].set_visible(False)
        a.grid(axis="y",color="#D8DFE3",linewidth=.55,alpha=.75)
        a.set_axisbelow(True)
    fig.savefig(env.EVIDENCE/"horizon-progress.png",dpi=180,facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--stage",type=int,required=True)
    plot(cli.parse_args().stage)
