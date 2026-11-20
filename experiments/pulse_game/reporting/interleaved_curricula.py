"""Render fixed-budget comparisons from sealed behavioral results."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..interleave_interference import EVIDENCE


COHORTS={"original":"原六个出生","independent":"独立六个出生"}
GEOMETRIES={"hard":"只练等边三角","orthogonal":"只练正交气味","interleaved":"三种几何交错"}
ORDERS={"parent":"继续训练前","old_then_new":"先旧后新","new_then_old":"先新后旧","alternating":"逐对交错"}


def read(file):
    return json.loads((EVIDENCE/file).read_text(encoding="utf8"))


def percent(record):
    return f"{100*record['mean']:.2f}%"


def effect(record):
    low,high=record["ci95"]
    return f"{100*record['mean']:+.2f} [{100*low:+.2f}, {100*high:+.2f}]"


def main():
    plt.rcParams["font.sans-serif"]=["Microsoft YaHei","DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"]=False
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.3),sharex=True,sharey=True,layout="constrained")
    colors={"hard":"#a74d2a","orthogonal":"#516b93","interleaved":"#137868"}
    text=["# 同一张图在两种课程中的表现","",
          "## 跨几何的学习","",
          "每组先学习三个对象，末题由实际选侧获得资源。三组使用相同出生、布局随机数、味觉规律、优化器和 64,000 次更新，改变气味间的几何关系。交错组一半为等边三角，四分之一为正交气味，四分之一为随机独立方向。下表使用既有留出库，每个出生 128 个世界。普通三项和困难三项都按真实采样动作计分。严格通过率另要求两个相反困难历史的三个反对称余量都大于 0.05。","",
          "| 出生 | 64k 课程 | 困难两张表严格通过 | 困难三项实际全对 | 普通随机三项全对 | 普通五项全对 |",
          "|---|---|---:|---:|---:|---:|"]
    differences=[]
    for ax,(cohort,description) in zip(axes,COHORTS.items(),strict=True):
        archived={step:read(f"geometry-comparison-{cohort}-{step}.json")["scores"] for step in (8000,32000,64000)}
        for arm,label in GEOMETRIES.items():
            history=[archived[i][arm] for i in archived]
            x=[100*item["random"]["three_actual"]["mean"] for item in history]
            y=[100*item["hard"]["hard_three_actual"]["mean"] for item in history]
            ax.plot(x,y,color=colors[arm],lw=1.6,label=label)
            ax.scatter(x,y,c=colors[arm],s=(17,35,65))
            last=history[-1]
            vals=(last["hard"]["both_hard_maps_per_order"],last["hard"]["hard_three_actual"],
                  last["random"]["three_actual"],last["card"]["all_5"])
            text.append("| "+" | ".join((description,label,*map(percent,vals)))+" |")
        for comparator in ("hard","orthogonal"):
            diff=archived[64000]["interleaved minus "+comparator]
            differences.append(f"| {description} | 交错减{GEOMETRIES[comparator]} | "+
                               " | ".join(effect(v) for v in (diff["hard"]["hard_three_actual"],
                                             diff["random"]["three_actual"],diff["card"]["all_5"]))+" |")
        ax.set_title(description)
        ax.set_xlabel("普通随机三种气味实际全对（%）")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("困难三种气味实际全对（%）")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncols=3,frameon=False)
    fig.savefig(EVIDENCE/"geometry-curriculum.png",dpi=170)
    plt.close(fig)
    text+=["","点从小到大依次为 8k、32k、64k 次更新。两组坐标刻度一致。","",
           "![相同训练预算下三种课程的实际选侧成绩](geometry-curriculum.png)","",
           "| 出生 | 64k 配对差，百分点 | 困难三项实际全对 | 随机三项实际全对 | 普通五项全对 |",
           "|---|---|---:|---:|---:|",*differences,"",
           "交错 64k 与只练等边三角 32k 的困难训练次数都是 32k。使用全局更新编号生成训练世界，两组的困难样本不逐一对应，交错组有额外的 32k 次其他几何更新。逐项区间和这个额外预算对照见[原组](geometry-comparison-original-64000.json)与[独立组](geometry-comparison-independent-64000.json)。","",
           "## 一条连续生命中的六项记忆","",
           "三组从同一批 64k 等边三角模型继续训练，每组使用完全相同的 4,000 次原有三角任务更新和 4,000 次新场景、气味及给料器更新，只改变这两类生命在出生前训练中的次序。下表另外列出继续训练前的亲本。部署时同一张图经历七次接触，再从相同末状态分别采样六个无反馈动作。两个给料器示范的两侧标记相同，其他五次体验由选侧决定味觉；七次接触均可更新活动和图。","",
           "| 出生 | 继续训练方式 | 全部映射的六项全对 | 场景冲突时三角三项 | 场景冲突时 A/B/标记三项 | 场景冲突时六项 | 困难三角且场景冲突时六项 |",
           "|---|---|---:|---:|---:|---:|---:|"]
    diff_rows=[];controls=[]
    for c,name in COHORTS.items():
        source=read(f"cross-skill-{c}-8000.json")
        rows=source["same_life"]
        for arm,label in ORDERS.items():
            z=rows[arm]
            all=z["active/all_maps/mean_order"]
            conflict=z["active/context_conflict/mean_order"]
            hard=z["active/hard_and_context_conflict/mean_order"]
            observed=(all["all_six"],conflict["triangle_three"],conflict["context_both_mark"],
                      conflict["all_six"],hard["all_six"])
            text.append("| "+" | ".join((name,label,*map(percent,observed)))+" |")
        for other in ("old_then_new","new_then_old","parent"):
            vals=rows["alternating minus "+other]["active/context_conflict/mean_order"]
            diff_rows.append("| "+" | ".join((name,"交错减"+ORDERS[other],
                              *(effect(vals[k]) for k in ("triangle_three","context_both_mark","all_six"))))+" |")
        z=rows["alternating"]
        controls.append(f"{name}的交错组在场景冲突时，关闭生内参数写入后三项气味与场景合计六项全对为 "
                        f"{percent(z['sham/context_conflict/mean_order']['all_six'])}，使用同一运动随机数均匀选侧为 "
                        f"{percent(z['uniform/context_conflict/mean_order']['all_six'])}。")
    text+=["","每个出生 96 个世界，三个预习顺序等权。每世界 64 个完整私有赋值，其中 32 个场景冲突，同时满足困难三角与场景冲突的有 8 个。各项差在配对后按训练出生和共享世界重采样。","",
           "| 出生 | 场景冲突下的配对差，百分点 | 三角三项 | A/B/标记三项 | 六项 |",
           "|---|---|---:|---:|---:|",*diff_rows,"","".join(controls),"",
           "一生六项测验与[独立物理运动验证](mixed-results.md)使用不同的身体时钟。出生前练习的先后顺序、部署后七次实际接触的先后顺序也分别记录。[原组完整证据](cross-skill-original-8000.json) · [独立组完整证据](cross-skill-independent-8000.json)"]
    (EVIDENCE/"curricula-and-joint.md").write_text("\n".join(text)+"\n",encoding="utf8")


if __name__=="__main__":
    main()
