"""One actual paired life, showing where different learned rules reach different physical targets."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from . import symmetry_audit as protocols,mixture_flash as flash,world, lifelong_agent as portable
from . import symmetry_world as task
from experiments.seed_evolution.large_runner import _write


def main():
    folder=Path(__file__).resolve().parents[2]/"data/pulse-game/morning-20261119"
    roots=np.random.default_rng(18691153).integers(3_000_000,465_000_000,64).tolist()
    p=protocols.joint(roots,pairs=12,probes=1,span=4,noise=.04,jitter=.08,matched=True)
    agent=portable.new_lives(roots+roots,address="local");agent.reset(2*len(roots))
    sim=flash.FlashBody(p.menu,hidden_after=p.exposed+1,record=True)
    obs=sim.observe();gain=[];slot=[];key_pulse=[]
    while obs is not None:
        action=agent.act(obs)
        if sim.tick_index==5:
            row=np.arange(len(action));active=agent.places.current.clip(min=0)
            gain.append(agent.g[row,active].copy());slot.append(active.copy())
        obs=sim.step(action)
    r=sim.result();n=len(roots);j=p.exposed+1
    correct=task.metrics(p,r)["first"].astype(bool)
    same=(r.observed_reward[:n,:p.exposed]==r.observed_reward[n:,:p.exposed]).all((1,2))
    eligible=np.flatnonzero(correct[:n]&correct[n:]&same&(r.contact_side[:n,j]!=r.contact_side[n:,j]))
    if len(eligible)==0:raise ValueError("No qualifying example")
    ix=int(eligible[0]);pick=[ix,ix+n];labels=["同味规律","反味规律"];colors=["#176F70","#AC4A2B"]
    traces=np.stack(gain,1)[pick]
    position=r.obs[pick,j,:,world.POSITION]
    taste=r.observed_reward[pick,:p.exposed].sum(2)
    record=dict(seed=roots[ix],index=ix,criterion="first paired root both physically correct with opposite sides and identical signed timing",
              roots_considered=n,roots_eligible=len(eligible),gain_at_tick5=traces.tolist(),actual_teaching_sign=taste.tolist(),
              time_series_sign_identical=bool(same[ix]),query_position=position.tolist(),query_motor=r.pulses[pick,j].astype(int).tolist(),
              query_touch=r.contact_side[pick,j].tolist(),query_touch_tick=r.contact_tick[pick,j].tolist(),
              B_anchor_actual_reward=r.reward[pick,p.exposed].tolist(),scalar_feedback_at_query=r.observed_reward[pick,j].tolist())
    _write(folder/"single-life-trace.json",record)
    font=Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams["font.family"]=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"]=False
    fig,axes=plt.subplots(2,1,figsize=(9.0,6.9),gridspec_kw={"height_ratios":[1,1.15]})
    fig.subplots_adjust(top=.83,bottom=.14,left=.125,right=.97,hspace=.65)
    fig.patch.set_facecolor("#FBFCFA")
    fig.text(.125,.948,"同一份新食物，两个生命走向了不同目标",fontsize=17.5,color="#133F3C",weight="bold")
    fig.text(.125,.899,"A 阶段逐拍收到相同正负序列，改变的只有哪种实际气味遇到哪次回执。",fontsize=10,color="#526963")
    t=np.arange(1,p.exposed+1);a=axes[0]
    for k in range(2):a.plot(t,traces[k,:p.exposed],marker="o",markersize=3.4,color=colors[k],lw=1.9,label=labels[k])
    a.axhline(0,color="#9FB1AB",lw=.7);a.set_ylabel("当前地址的规则增益")
    a.set_xlabel("A 阶段的实际请求");a.legend(loc="lower left",ncol=2,frameon=False,fontsize=9)
    a.set_title("实际味觉经历改变了规则权重",loc="left",fontsize=11.5)
    b=axes[1];tick=np.arange(position.shape[1])
    for k in range(2):
        b.plot(tick,position[k],marker="o",markersize=3.4,color=colors[k],lw=2,label=labels[k])
        at=int(r.contact_tick[pick[k],j])+1
        if at<len(tick):b.scatter([at],[position[k,at]],facecolors="white",edgecolors=colors[k],linewidths=2,s=48,zorder=4)
    b.set_xlabel("B 首次自由选择中的物理 tick");b.set_ylabel("身体位置（左 / 右）")
    b.set_title("B 的锚味和左右场景完全相同，空心点为实际触端",loc="left",fontsize=11.5)
    for ax in axes:
        ax.set_axisbelow(True);ax.grid(axis="y",color="#DDE6E0",lw=.8)
        for sp in ("top","right"):ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("#A1B3A8");ax.spines["bottom"].set_color("#A1B3A8")
    fig.text(.125,.041,f"固定参考个体。取预先抽出的 {n} 个世界中第一个两臂都成功且方向相反的实例。总体正确率见主结果图。",fontsize=9,color="#526963")
    fig.savefig(folder/"single-life-trace.png",dpi=175,facecolor=fig.get_facecolor());plt.close(fig)


if __name__=="__main__":main()
