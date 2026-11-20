"""Compare frozen, physically realized choices and the primary's forced-test forecasts."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EVIDENCE=Path(__file__).resolve().parents[3]/"docs/evidence/fork-lesson-20261120"


def main():
    main_result=json.loads((EVIDENCE/"eval-results.json").read_text(encoding="utf8"))["worlds"]
    trial=json.loads((EVIDENCE/"native-request-results.json").read_text(encoding="utf8"))["worlds"]
    plt.rcParams.update({"font.family":["Microsoft YaHei","DejaVu Sans"],"axes.unicode_minus":False,
                         "axes.spines.top":False,"axes.spines.right":False,"font.size":10})
    fig,axes=plt.subplots(1,2,figsize=(12,4.35),constrained_layout=True)
    datasets=[[],[]]
    for w in main_result.values():
        rates=w["arms"]["whole"]["close_private_and_payoff_opposite"]["rates"]
        datasets[0].append([rates[k]["paid_positive"] for k in ("natural","cut_out","cut_back","donor_back")])
    for w in trial.values():
        rates=w["subsets"]["strict"]["rates"]
        datasets[1].append([rates[g][mode]["rank_accuracy"] for g,mode in
                            (("binding","natural"),("binding","blank"),("misbound","natural"),("binding","cut_return"))])
    for ax,data,title,ylabels,xlabel in zip(axes,datasets,
            ("真实选门：40 节点整体预测训练","内部试探：40 节点，主区单神经元预报"),
            (("正常运行","截断去程","截断回程","换入另一历史的回程"),
             ("按本人意图试探","空试，不递送脉冲","学习时试探错配","截断试探回程")),
            ("两门相反、困难条件下实际选优率（%）","同一身体分别内试两门，主区正确排序率（%）"),strict=True):
        ax.set_title(title,fontsize=13,pad=9)
        for rep,conditions in enumerate(data):
            for row,item in enumerate(conditions):
                mean=100*item["mean"];lo,hi=[100*v for v in item["ci95"]]
                ax.errorbar(mean,3-row+(0.11 if rep==0 else -.11),xerr=[[mean-lo],[hi-mean]],fmt="o",capsize=3,
                            color=("#137466","#a95631")[rep],label=f"留出{rep+1}" if row==0 else None)
        ax.axvline(50,color="#99a2ad",linewidth=1)
        ax.grid(axis="x",alpha=.15);ax.set(yticks=[3,2,1,0],yticklabels=ylabels,xlim=(10,102),ylim=(-.55,3.5),xlabel=xlabel)
        ax.legend(loc="lower right",frameon=False,fontsize=8)
    fig.savefig(EVIDENCE/"forecast-and-inquiry.png",dpi=160)
    plt.close(fig)


if __name__=="__main__":
    main()
