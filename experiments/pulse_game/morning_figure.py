"""Generate the main research figure from the locked physical records and state-fork results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager,ticker


ROOT=Path(__file__).resolve().parents[2]/"data/pulse-game/morning-20261119"
TEAL="#086B68";BLUE="#355FA0";RUST="#AF4B2E";GRAY="#5F6E72";SAND="#BD8B3B";LIGHT="#DAE3E0"
def numbers(x):return 100*x["mean"],100*np.asarray(x["ci95"])


def curve(ax,pos,source,key,color,label,marker):
    values=[numbers(source[k]) for k in key]
    y=np.asarray([v[0] for v in values]);ci=np.stack([v[1] for v in values])
    ax.fill_between(pos,ci[:,0],ci[:,1],color=color,alpha=.13,linewidth=0)
    ax.plot(pos,y,color=color,marker=marker,markersize=5,lw=2.35,label=label,clip_on=False)


def setup(ax,title):
    ax.set_title(title,loc="left",fontsize=13,pad=11,fontweight="bold",color="#153B41")
    ax.set_ylim(0,103);ax.set_yticks([0,25,50,75,100]);ax.yaxis.set_major_formatter(ticker.PercentFormatter(100,decimals=0))
    ax.grid(axis="y",color=LIGHT,linewidth=.8);ax.set_axisbelow(True)
    for k in ("top","right"):ax.spines[k].set_visible(False)
    for k in ("bottom","left"):ax.spines[k].set_color("#ADBDBA")
    ax.tick_params(colors="#34474C",length=2)
    ax.margins(x=.045)


def main():
    font=Path("C:/Windows/Fonts/msyh.ttc")
    if font.is_file():font_manager.fontManager.addfont(str(font));plt.rcParams["font.family"]=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"font.size":10,"axes.unicode_minus":False,"pdf.fonttype":42})
    base=json.loads((ROOT/"learning-and-retention.json").read_text())
    v=base["local_addresses"]["results"];fig,axs=plt.subplots(2,2,figsize=(12.25,9.65))
    fig.patch.set_facecolor("#FAFCFA")
    fig.subplots_adjust(top=.835,bottom=.145,left=.075,right=.98,hspace=.54,wspace=.23)
    fig.text(.075,.957,"从一次味觉反馈，到跨情境的长期学习",fontsize=21,fontweight="bold",color="#133C3F")
    fig.text(.075,.910,"同一生命保持感觉活动、运动神经元膜态和可塑连接。前三图使用 12 个重新训练的出生网络。",fontsize=10.5,color=GRAY)
    ax=axs[0,0];setup(ax,"新环境与旧规律相反时，几次经历能改判？")
    ticks=[0,2,4,8,12]
    for group,c,l,m in (("same",BLUE,"两地规律相同","o"),("conflict",TEAL,"两地规律相反","s")):
        curve(ax,ticks,v,[f"standard/local/{group}/B_after_{i}" for i in ticks],c,l,m)
    ax.set_xticks(ticks);ax.set_xlabel("B 已提供的关系样本对数");ax.legend(frameon=False,loc="center right",fontsize=9)
    ax=axs[0,1];setup(ax,"B 学完后，切回 A 的第一次选择")
    cases=["standard","absent_A","near_and_noisy"];labels=["B 学 12 对","B 学 48 对","情境相近且有噪声"]
    x=np.arange(3)
    for j,(arm,col,lab) in enumerate((("distributed",GRAY,"分布式地址"),("local",TEAL,"局部地址"))):
        offset=(j-.5)*.20
        for i,k in enumerate(cases):
            source=v[f"{k}/{arm}/conflict/return_A"];y,ci=numbers(source)
            seeds=100*np.asarray(source["per_birth"]);jitter=np.linspace(-.037,.037,len(seeds))
            ax.scatter(x[i]+offset+jitter,seeds,color=col,s=9,alpha=.31,zorder=2,linewidths=0)
            ax.errorbar(x[i]+offset,y,[[y-ci[0]],[ci[1]-y]],fmt="o",color=col,ms=6,lw=1.6,capsize=3,
                        label=lab if i==0 else None,zorder=3)
    ax.set_xlim(-.4,2.4);ax.set_xticks(x,labels);ax.legend(frameon=False,loc="lower left",fontsize=9)
    ax=axs[1,0];setup(ax,"B 的规律暗中翻转，原地址还能否再学？")
    regs=["change_1_1","change_2_2","change_3_4","change_5_8","change_9_16"]
    for arm,col,lab,mark in (("local",TEAL,"局部地址持续可更新","s"),("local_no_reopening",RUST,"同一地址按常规 RLS 更新","o")):
        curve(ax,np.arange(5),v,[f"stream/{arm}/{reg}/post" for reg in regs],col,lab,mark)
    ax.set_xticks(range(5),["1","2","3–4","5–8","9–16"]);ax.set_xlabel("翻转后第几次访问 B（当次教学后）")
    ax.legend(frameon=False,loc="center right",fontsize=9)
    ax=axs[1,1];setup(ax,"只给新食物锚味时，哪部分旧状态有用？")
    d=base["forks"];arms=["ongoing","only_higher_rule","only_episode_memory","cold"]
    vals=[numbers(d[a]["all"]) for a in arms];ys=[a[0] for a in vals]
    colors=[TEAL,BLUE,SAND,GRAY];xx=np.arange(4)
    ax.bar(xx,ys,color=colors,width=.62,alpha=.9)
    for i,(y,ci) in enumerate(vals):
        ax.errorbar(i,y,[[y-ci[0]],[ci[1]-y]],fmt="none",ecolor="#173941",capsize=3,linewidth=1.25)
        ax.text(i,y-9,f"{y:.1f}%",ha="center",color="white",fontsize=10,fontweight="bold")
    ax.set_xticks(xx,["完整状态","仅规则权重\n与协方差","仅食物事件\n记忆","冷状态"])
    ax.set_xlabel("同一参考个体，256 个新世界，后续没有成对教学")
    fig.text(.075,.035,"线带和误差棒为 95% 自举区间。前两图 256 个世界根，左下 96 个世界根、每生 2,304 次请求。所有选择均按实际接触计分。",color=GRAY,fontsize=9.2)
    fig.savefig(ROOT/"learning-results.png",dpi=185,facecolor=fig.get_facecolor())
    fig.savefig(ROOT/"learning-results.pdf",facecolor=fig.get_facecolor());plt.close(fig)


if __name__=="__main__":main()


