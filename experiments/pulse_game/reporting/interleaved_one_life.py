"""Render the two paid actions and six cold memory probes separately."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .interleaved_curricula import percent,effect,COHORTS
from ..interleave_interference import EVIDENCE


ARMS={"parent":"训练前","blocked":"分块体验","interspersed":"交错体验"}


def main():
    plt.rcParams["font.sans-serif"]=["Microsoft YaHei","DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"]=False
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.4),sharey=True,layout="constrained")
    text=["# 一条生命同时学习六项私有规律","",
          "两种训练日程均从相同的 64k 三角形模型出发。七次接触学习三种气味、同一气味的 A/B 场景价值，以及一种给料器标记。每条生命末尾从两域各抽一题，题序随机，两次实际行动都成功才取得外层训练回报。第一道终局题接触后没有正负味觉输入，图仍根据感觉和选侧正常运行并可写入，然后进行第二次选择。两组均训练 8,000 步，部署时没有外层优化器。","",
          "分块组随机决定哪一域在前。交错组在四次场景或给料器接触之间插入三个气味，按这套时序以场景或标记起止。两组各对象的随机数和总训练时钟相同，各个对象距末次行动的间隔随顺序而变，实际选侧与味觉也能不同。","",
          "## 新留出世界上的实际行动","",
          "表中的串答列记录同一生命中先后作出的两个动作。六项全对列在另一套世界，从同一末状态分别探测六个答案。每出生、每组 96 个世界，每个世界交叉 64 种私有赋值，其中 32 种场景冲突，8 种同时属于困难三角和场景冲突。串答每世界固定一对题及题序，未穷举 18 种可能。百分比是六个出生的平均。","",
          "| 出生 | 训练日程 | 串答两项均获益，全映射 | 串答两项均获益，场景冲突 | 六项全对，全映射 | 六项全对，场景冲突 | 六项全对，困难三角且场景冲突 |",
          "|---|---|---:|---:|---:|---:|---:|"]
    differences=[];controls=[];standalone=[]
    for axis,(c,name) in zip(axes,COHORTS.items(),strict=True):
        splits={key:json.loads((EVIDENCE/f"one-life-{c}-{key}-8000.json").read_text())["results"]
                for key in ("disjoint","shared")}
        dis=splits["disjoint"]
        for a,label in ARMS.items():
            row=dis[a]
            vals=(row["reward/active/all_maps"]["two_paid"],row["reward/active/context_conflict"]["two_paid"],
                  row["active/all_maps/mean_order"]["all_six"],row["active/context_conflict/mean_order"]["all_six"],
                  row["active/hard_and_context_conflict/mean_order"]["all_six"])
            text.append("| "+" | ".join((name,label,*map(percent,vals)))+" |")
        for split,bank in splits.items():
            for comp,cn in (("blocked minus parent","分块减亲本"),("interspersed minus parent","交错减亲本"),
                            ("interspersed minus blocked","交错减分块")):
                z=bank[comp]
                vals=(z["reward/active/context_conflict"]["two_paid"],z["active/context_conflict/mean_order"]["triangle_three"],
                      z["active/context_conflict/mean_order"]["context_both_mark"],z["active/context_conflict/mean_order"]["all_six"])
                differences.append("| "+" | ".join((name,"新世界" if split=="disjoint" else "共同留出库",
                                                        cn,*map(effect,vals)))+" |")
        for i,(key,cn,color) in enumerate((("triangle_three","三种气味全对","#226d88"),
                                           ("context_both_mark","A、B 和标记全对","#bf6a30"),
                                           ("all_six","六项全对","#376e47"))):
            rows=[dis[a]["active/context_conflict/mean_order"][key] for a in ARMS]
            y=100*np.asarray([z["mean"] for z in rows])
            limits=100*np.asarray([z["ci95"] for z in rows]).T
            axis.errorbar(np.arange(3)+(i-1)*.13,y,yerr=np.stack((y-limits[0],limits[1]-y)),fmt="o",color=color,
                          ms=6,capsize=3,label=cn)
        axis.set_title(name);axis.set_xticks(range(3),ARMS.values());axis.set_xlim(-.4,2.4)
        axis.spines[["top","right"]].set_visible(False);axis.grid(axis="y",alpha=.18)
        for a in ("blocked","interspersed"):
            z=dis[a];quiet=z["sham/context_conflict/mean_order"]["all_six"]
            chance=z["uniform/context_conflict/mean_order"]["all_six"]
            controls.append(f"{name}{ARMS[a]}，接触全程禁写后，场景冲突的六项全对为 {percent(quiet)}，"
                            f"同随机数均匀选侧为 {percent(chance)}。")
        for a in ARMS:
            b=splits["shared"][a]
            standalone.append("| "+" | ".join((name,ARMS[a],percent(b["hard_standalone"]["hard_three_actual"]),
                               percent(b["card"]["all_2"]),percent(b["card"]["all_5"]),
                               percent(b["card"]["coherent/unseen"])))+" |")
    axes[0].set_ylabel("场景 A 与 B 价值冲突时的正确率（%）")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncols=3,frameon=False)
    fig.savefig(EVIDENCE/"one-life.png",dpi=170);plt.close(fig)
    text+=["","![新世界中两类记忆及其联合成绩](one-life.png)","",
           "| 出生 | 世界 | 训练日程对比 | 连续双答全获益 | 三种气味全对 | A/B/标记全对 | 六项全对 |",
           "|---|---|---|---:|---:|---:|---:|",*differences,"",
           "以上均为场景价值冲突子集的配对差，单位为百分点。区间同时重采样训练出生及其共享测试几何，保留同一世界的配对规律。","",
           *controls,"","## 另外的新生命还能完成哪些任务","",
           "| 出生 | 训练日程 | 等边三角困难赋值三项实际全对 | 两个普通对象全对 | 五个普通对象全对 | 隐藏线性规律的新对象首次答对 |",
           "|---|---|---:|---:|---:|---:|",*standalone,"",
           "这些单项测试分别从相同的继承模型开始，各自保留内部连续状态。新世界的两动作及六项测验来自原生感觉和选侧时钟。此前报告的真实身体逐拍落地有自己的分母和对照。[原组新世界](one-life-original-disjoint-8000.json) · [独立组新世界](one-life-independent-disjoint-8000.json) · [原组共同留出库](one-life-original-shared-8000.json) · [独立组共同留出库](one-life-independent-shared-8000.json)"]
    (EVIDENCE/"one-life-results.md").write_text("\n".join(text)+"\n",encoding="utf8")


if __name__=="__main__":
    main()
