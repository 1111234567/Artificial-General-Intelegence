"""Plot per-seed free-policy results from independently reconstructed counts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.wait_content.audit import ARMS, BASE, ROOT, sha


def main():
    path = BASE/"independent-audit.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["audit"] == "passed"
    output = ROOT/"docs/evidence/wait-content-2026-11-03"
    output.mkdir(parents=True,exist_ok=True)
    values = [r for seed in data["seeds"] for r in seed["outcomes"]]
    (output/"free-behavior.json").write_text(json.dumps(dict(
        source_sha256=sha(path), seeds=data["budgets"], outcomes=values), indent=2), encoding="utf-8")
    names = ("原策略","仅结束时","等待时也学")
    colors = ("#146e74","#a34e19","#6f57a2")
    plt.rcParams.update({"font.family":"Microsoft YaHei","font.size":10,"axes.unicode_minus":False})
    fig, axes = plt.subplots(2,3,figsize=(11.3,6.2))
    fig.subplots_adjust(top=.80,bottom=.11,left=.105,right=.975,wspace=.34,hspace=.42)
    fig.text(.045,.94,"移动一半内容监督后，模型实际做成了多少任务？",weight="bold",fontsize=16)
    fig.text(.045,.88,"每条线连接同一种子的三个策略。完成率按成熟请求计算，空闲误动按实际空闲机会计算。",color="#43535b")
    metrics = (("ordinary","普通完成（%）"),("strict","全程无错完成（%）"),("idle","空闲误动（每千次机会）"))
    for i,(condition, label) in enumerate((("short","短任务"),("long_both","长处理\n长空档"))):
        for j,(key,title) in enumerate(metrics):
            ax=axes[i,j]
            curves=[]
            for seed in range(3):
                rates={r["arm"]:r["rates"] for r in values if r["seed"]==seed and r["condition"]==condition}
                curve=[rates[arm][key] for arm in ARMS]
                curves.append(curve)
                ax.plot(range(3),curve,"o-",lw=1.6,color=colors[seed],label=f"种子 {seed}",alpha=.88)
            top = max(1 if key!="idle" else 5, float(np.max(curves))*1.10)
            ax.set(xticks=range(3),xticklabels=names,xlim=(-.14,2.14),ylim=(-.035*top,top))
            if i==0: ax.set_title(title)
            if j==0: ax.set_ylabel(label,rotation=0,labelpad=26,va="center",weight="bold")
            ax.grid(axis="y",alpha=.2)
            ax.spines[["top","right"]].set_visible(False)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=3,frameon=False)
    fig.savefig(output/"free-behavior.png",dpi=155,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    main()
