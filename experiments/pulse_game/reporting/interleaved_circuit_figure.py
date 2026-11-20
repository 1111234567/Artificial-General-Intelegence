"""Show the two independently trained cohorts' physical angular response and within-choice dynamics."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..interleave_interference import EVIDENCE


def run():
    samples = {name:json.loads((EVIDENCE/f"triangle-angular-{name}.json").read_text())
               for name in ("original","independent")}
    plt.rcParams["font.sans-serif"]=["Microsoft YaHei","DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"]=False
    fig,(spatial,temporal)=plt.subplots(1,2,figsize=(10.5,4.5),gridspec_kw={"width_ratios":[1.6,1.]},
                                        layout="constrained")
    color={"original":"#0b736a","independent":"#9e4f19"}
    name={"original":"原组","independent":"独立新出生"}
    angle=np.arange(0,365,5)
    for cohort in samples:
        curves=samples[cohort]["results"]
        for step,alpha in (("0",.42),("64000",1.)):
            s=curves[step]["final_curve"]
            y=np.r_[s["mean"],s["mean"][0]]
            label=name[cohort]+("，训练前" if step=="0" else "，64,000 步")
            spatial.plot(angle,y,color=color[cohort],alpha=alpha,linewidth=1.3 if step=="0" else 2.2,label=label)
        point=curves["64000"]["tick_third"]
        low,high=point["ci95"]
        temporal.plot(range(1,6),point["mean"],color=color[cohort],marker="o",label=name[cohort])
        temporal.fill_between(range(1,6),low,high,color=color[cohort],alpha=.14)
    spatial.scatter([0,120,240],[0]*3,marker="^",s=45,color="#394954",zorder=4,label="三种学习气味")
    for ax in (spatial,temporal):
        ax.spines[["top","right"]].set_visible(False)
        ax.axhline(0,color="#899498",lw=.7)
        ax.grid(axis="y",alpha=.16)
    temporal.legend(frameon=False,fontsize=8.5)
    handles,labels=spatial.get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncols=3,frameon=False,fontsize=8.5)
    spatial.set(xlabel="探测气味在学习平面内的角度（度）",ylabel="相反学习经历造成的方向选择差（logit）",
                title="同一张图怎样读出三种气味",xticks=[0,60,120,180,240,300,360],xlim=(-5,365))
    temporal.set(xlabel="做出一次行动时的内部更新步",ylabel="与三种已学方向对齐的三阶分量",
                 title="反应链中的三阶分量",xticks=range(1,6),xlim=(.85,5.15))
    fig.savefig(EVIDENCE/"triangle-angular.png",dpi=170)
    plt.close(fig)


if __name__=="__main__":
    run()
