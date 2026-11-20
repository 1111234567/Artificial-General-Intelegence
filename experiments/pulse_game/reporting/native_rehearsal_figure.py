"""Draw measured behavior in the absence of a supplied replay cue."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT=Path(__file__).resolve().parents[3]
EVIDENCE=ROOT/"docs/evidence/native-rehearsal-20261120"


def main():
    wait=json.loads((EVIDENCE/"wait.json").read_text(encoding="utf8"))["scores"]
    behavior=json.loads((EVIDENCE/"behavior.json").read_text(encoding="utf8"))
    plt.rcParams.update({"font.family":["Microsoft YaHei","DejaVu Sans"],"font.size":10,"axes.unicode_minus":False,
                         "axes.spines.top":False,"axes.spines.right":False})
    fig,(time,changes)=plt.subplots(1,2,figsize=(11.5,4.3),constrained_layout=True)
    ticks=[0,3,6,12,24,48]
    for group,label,color in (("single","20 节点","#6b7280"),("copy","40 节点 · 出生复制","#167362"),
                              ("permuted","40 节点 · 副区入边置换","#bc672a")):
        series=[wait[group]["results"][str(n)]["both"]["mean"]*100 for n in ticks]
        time.plot(ticks,series,marker="o",color=color,label=label)
    time.set(xlabel="B 接触结束后的全零输入拍数",ylabel="A、B 实际两题全对率（%）",xticks=ticks,ylim=(32,66))
    time.set_title("额外计算时间的实际收益");time.grid(axis="y",alpha=.15)
    time.legend(frameon=False,fontsize=8,loc="lower right")
    tests=(("held","整个网络暂停"),("no_quiet_write","只暂停快权重更新"),("cut_return","只截断副区回传电流"))
    for row,(reference,label) in enumerate(tests):
        for x,(seed,group,color) in enumerate((("81921671","留出一","#176757"),("81921701","留出二","#536ea0"))):
            item=behavior[seed]["results"]["close"][f"copy/normal minus copy/{reference}"]["conflict/both"]
            value=item["mean"]*100;low,high=100*np.asarray(item["ci95"])
            changes.errorbar(value,2-row+(0.12 if x==0 else -.12),xerr=[[value-low],[high-value]],
                             fmt="o",color=color,capsize=3,label=group if row==0 else None)
    changes.axvline(0,color="#777",linewidth=1);changes.set_yticks([2,1,0],[v[1] for v in tests])
    changes.set(xlabel="完整网络减干预，两题全对率差（百分点）",ylim=(-.54,2.4))
    changes.set_title("40 节点安静期各条路径的作用")
    changes.legend(loc="lower right",frameon=False,fontsize=8);changes.grid(axis="x",alpha=.15)
    fig.savefig(EVIDENCE/"silence-and-behavior.png",dpi=170)
    plt.close(fig)


if __name__=="__main__":
    main()
