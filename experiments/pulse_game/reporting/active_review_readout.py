"""Report the learned readout, its sign symmetry, and calibrated forecasts on actual motor outcomes."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .. import noon_online_readout as online,noon_online_eval as online_eval
from .. import noon_self_assess as forecaster
from ..noon_review_world import EVIDENCE


def pct(v):return f"{100*v['mean']:.2f}%"


def diff(v,scale=100):
    return f"{scale*v['mean']:+.2f} [{scale*v['ci95'][0]:+.2f}, {scale*v['ci95'][1]:+.2f}]"


def read(name):return json.loads((EVIDENCE/name).read_text(encoding="utf8"))


@torch.no_grad()
def symmetry_and_calibration(cohort):
    path=forecaster.OUT/"tests"/cohort/"uniform"/"4"
    meta=json.loads((path/"index.json").read_text())
    summaries={}
    for arm in forecaster.ARMS:
        counts=np.zeros(10);predicted=np.zeros(10);realized=np.zeros(10);max_diff=0.;changes=0;total=0
        ordinary=reflected=0.
        for seed in forecaster.SEEDS:
            m=forecaster.Assessor();m.load_state_dict(torch.load(forecaster.OUT/str(seed)/arm/"step-4000.pt",weights_only=True,map_location="cpu")["model"])
            opposite=forecaster.Assessor();opposite.load_state_dict(m.state_dict())
            opposite.forecast.mul_(-1);opposite.choice[0].mul_(-1)
            for b in meta["births"]:
                with np.load(path/f"{b}.npz") as raw:
                    z=torch.tensor(raw["z"],dtype=torch.double);side=torch.tensor(raw["side"],dtype=torch.double)
                    menu=torch.tensor(raw["native"],dtype=torch.double);u=raw["u"];q=raw["scored"]
                    y=raw["early"][np.arange(len(z)),q]
                ans,_,gate=m(z,side,menu);reverse,_,same=opposite(z,side,menu)
                p=gate.sigmoid().numpy();p2=same.sigmoid().numpy()
                max_diff=max(max_diff,float(np.max(np.abs(p-p2))));changes+=int(np.count_nonzero((u<p)!=(u<p2)))
                v=ans.sigmoid().numpy()[np.arange(len(z)),q];rv=reverse.sigmoid().numpy()[np.arange(len(z)),q]
                np.testing.assert_allclose(v+rv,1,rtol=0,atol=1e-14)
                ordinary+=np.square(v-y).sum();reflected+=np.square(rv-y).sum();total+=len(v)
                bins=np.minimum((v*10).astype(int),9)
                counts+=np.bincount(bins,minlength=10)
                predicted+=np.bincount(bins,weights=v,minlength=10)
                realized+=np.bincount(bins,weights=y,minlength=10)
        if changes or max_diff>1e-12:raise ValueError("A complementary confidence coding changed route behavior")
        summaries[arm]=dict(counts=counts.astype(int).tolist(),prediction=np.divide(predicted,counts,out=np.zeros(10),where=counts>0).tolist(),
                            observed=np.divide(realized,counts,out=np.zeros(10),where=counts>0).tolist(),brier=ordinary/total,
                            reversed_coding_brier=reflected/total,actual_route_changes=changes,max_probability_change=max_diff,
                            records=total)
    return summaries


def figure(stats):
    plt.rcParams["font.sans-serif"]=["Microsoft YaHei","DejaVu Sans"];plt.rcParams["axes.unicode_minus"]=False
    fig,(left,right)=plt.subplots(1,2,figsize=(11,4.6),layout="constrained",gridspec_kw={"width_ratios":[1.05,1.25]})
    palette={"forecast":"#087f72","uncredited":"#a95430","shuffled":"#596b92"}
    labels={"forecast":"本人实际结果校准","uncredited":"仅有复习后的收益","shuffled":"打乱早期结果"}
    left.plot([0,1],[0,1],color="#a3aaa8",lw=.9)
    for arm in forecaster.ARMS:
        x=stats["independent"][arm];n=np.asarray(x["counts"]);chosen=n>=100
        left.plot(np.array(x["prediction"])[chosen],np.array(x["observed"])[chosen],"o-",ms=4.5,color=palette[arm],label=labels[arm])
    left.set(xlim=(0,1),ylim=(0,1),xlabel="身体对已作动作报出的成功概率",ylabel="实际取得正收益的比例",title="能否预测自己刚作的选择")
    left.legend(fontsize=8.5,frameon=False,loc="lower right")
    rr=read("online-independent-uniform-4-4000.json")["results"]
    names=["native","live","sampled_pair","forced_pair","weak"]
    desc=["原生菜单","每生一次真选择","同前态采样两次","两条真分支","外部低置信"]
    for i,name in enumerate(names):
        v=rr[name]["minus_random/all"]
        y=100*v["mean"];lo,hi=np.array(v["ci95"])*100
        right.errorbar(y,i,xerr=[[y-lo],[hi-y]],fmt="o",color="#137b74" if i in (1,2,3) else "#64748b",capsize=3)
    right.axvline(0,color="#9ba6a6",lw=.8)
    right.set_yticks(range(len(desc)),desc);right.invert_yaxis()
    right.set(xlabel="四项实际全对率减同图随机（百分点）",title="同一身体从四项中复习一项")
    for ax in (left,right):
        ax.spines[["top","right"]].set_visible(False);ax.grid(alpha=.14)
    fig.savefig(EVIDENCE/"learned-self-readout.png",dpi=170);plt.close(fig)


def run():
    torch.set_num_threads(1)
    stats={c:symmetry_and_calibration(c) for c in ("original","independent")}
    online.bank.lesson.storage._write_json(EVIDENCE/"self-report-symmetry.json",stats)
    figure(stats)
    lines=["# 从自己的运动反应中学习复习与预测","",
           "两个问题分别计分。一个观察是它是否能估计自己已经作出的动作会不会取得资源，另一个是它利用这种读数选择复习后，未来的实际动作有没有改善。这里显式接通了候选与运动读数，原生图的普通采集和生内写入照常工作。","",
           "## 同一条实际经验能不能教会一个小选择器","",
           "依次给两个候选各看两种摆向，各 5 拍。身体和活动痕迹连续推进，期间没有味觉和参数提交。同一个四隐单元读出把每个候选的两次原生运动 logit 映为一个分数，再由右减左的差与原菜单 drive 一起决定实际去向，共 17 个通过收益学习的系数及四个瞬时运动读数。三种训练均在原有六个身体上各进行三次独立搜索，测试时在另外六个身体上直接使用。每次更新每个训练身体都执行 64 条真正结束的复习轨迹。","",
           "| 训练时取得的实际收益 | 原组四项全对减同图随机，百分点 | 独立组同项，百分点 | 独立组自己的选择减同场景异经历捐赠，百分点 |",
           "|---|---:|---:|---:|"]
    r0=read("online-original-uniform-4-4000.json")["results"];r1=read("online-independent-uniform-4-4000.json")["results"]
    for method,title in (("live","每条生命按当前策略选择一次"),("sampled_pair","同一前态采样两个当前策略动作"),
                         ("forced_pair","同一前态实际走完两个相反动作"),("native","原生采集运动输出"),
                         ("weak","固定选择运动半差绝对值小的一项")):
        lines.append("| "+" | ".join((title,diff(r0[method]["minus_random/all"]),diff(r1[method]["minus_random/all"]),
                                      diff(r1[method]["own_minus_yoked/all"])))+" |")
    lines+=["","随机复习列对同一状态两个真实分支取平均。每次部署只去一次；所有方法花同样的预览时间。四物体中仅有两个可供选择。早先一次 768 几何、每个几何共用一次随机选侧的单侧离线拟合没有成功；改用充分的新鲜世界与各条生命自己的选择后，普通单路线收益已足以产生一个可转移的小优势。[首轮固定数据](readback-test-independent-uniform-4.json) · [新世界单路线](online-independent-uniform-4-4000.json) · [修复身体的转移](online-independent-repair-4-4000.json)","",
           "## 旧回报、当前回报与信号分量","",
           "| 独立身体上的单路线读出 | 四项全对差（百分点） | 当场取得正味的差（百分点） | 当场与一次实际后测的总资源差（每生） | 真写减同味觉禁写的四项全对差（百分点） |",
           "|---|---:|---:|---:|---:|"]
    for family,title in (("uniform","均匀回访专家"),("repair","短长信用修复图")):
        z=read(f"readout-cause-independent-{family}.json")["results"]
        lines.append("| "+" | ".join((title,diff(z["full/active/minus_random/all"]),diff(z["full/active/minus_random/immediate"]),
                                    diff(z["full/active/minus_random/resource"],1),diff(z["full/true_minus_sham/all"])))+" |")
    lines+=["","将学得的打分函数分解为同时反转两个运动 logit 后不变的部分和随之变号的部分。只保留不变部分时，四项实际收益接近完整读出；只保留变号部分，收益大幅减小。这是对新增小读出的功能分解。[实际分叉与计费](readout-cause-independent-uniform.json)","",
           "## 能预测自己的已作动作，是否就能更好地选复习","",
           "第二项试验只加五个系数。两次摆向后身体对该对象作出一次真正的普通方向选择，并接收自己的动作反馈。环境在生命结束时为其中一个已作动作结算；该结果在做复习选择时不可见。三个系数估计这个选侧的成功率，另两个系数把两项的估计差与原生菜单信号变成复习动作。","",
           "| 独立六出生，四对象 | 实际成功率 | 平均自报概率 | Brier（越低越好） | 复习四项全对减随机，百分点 |",
           "|---|---:|---:|---:|---:|"]
    ss=read("self-assess-independent-uniform-4.json")["results"]
    for k in forecaster.ARMS:
        v=ss[k];lines.append("| "+" | ".join(({"forecast":"用本人已作动作校准","uncredited":"只有复习后的真实收益","shuffled":"早期收益跨世界打乱"}[k],
                            pct(v["early_success"]),pct(v["forecast"]),f"{v['brier']['mean']:.3f}",diff(v["all_minus_random"])))+" |")
    lines+=["","把全部预测系数取反会把概率 p 变为 1−p，再把选择对估计差的斜率取反，最后两个动作的相对倾向完全相同。对两批留出逐个验证，选侧无一改变，而预测的误差明显变化。单靠最终复习行为不能确定内部标量是否具有“我有多可能正确”的含义。正确配对的本人行动后果使它获得可验证的局部概率解释；两种训练并未因此拉开四项回访成绩。[等价变换与校准分箱](self-report-symmetry.json) · [逐评估项](self-assess-independent-uniform-4.json)","",
           "![独立身体对自己动作的估计与复习获益](learned-self-readout.png)"]
    (EVIDENCE/"learned-readouts.md").write_text("\n".join(lines)+"\n",encoding="utf8")


if __name__=="__main__":run()
