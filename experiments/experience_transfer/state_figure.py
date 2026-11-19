"""Display each seed instead of averaging away persistent operating states."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.learn_remember.gru_train import sha


def main():
    root=Path(__file__).resolve().parents[2]
    folder=root/"data/experience-transfer/state-v1"
    output=root/"docs/evidence/experience-transfer-2026-11-03"
    output.mkdir(parents=True,exist_ok=True)
    rows=[]; files={}
    for seed in range(3):
        path=folder/f"seed_{seed:03d}.json"
        d=json.loads(path.read_text(encoding="utf-8"));files[str(path.relative_to(root))]=sha(path)
        for arch in ("plain","symmetric"):
            start=next(q for q in d["probes"] if q["arch"]==arch and q["clamp_ticks"]==0)["distance"]["rms_mean"]
            for q in d["probes"]:
                k=q["clamp_ticks"]
                if q["arch"]!=arch or k==0:
                    continue
                pair={r["entry_state"]:r["counts"] for r in d["records"] if
                    (r["arch"],r["driver"],r["clamp_ticks"])==(arch,q["driver"],k)}
                assert pair["standard"]["matured"]==pair["transplant"]["matured"]
                keep=("matured","completed","clean_completed","idle_actions","idle_opportunities")
                rows.append(dict(seed=seed,arch=arch,driver=q["driver"],ticks=k,
                    initial_rms=start,rms=q["distance"]["rms_mean"],ratio=q["distance"]["rms_mean"]/start,
                    next_action=q["next_action"],pair={arm:{k:c[k] for k in keep} for arm,c in pair.items()}))
    (output/"state-release.json").write_text(json.dumps(dict(rows=rows,source_sha256=files),indent=2),encoding="utf-8")
    plt.rcParams.update({"font.family":"Microsoft YaHei","font.size":10,"axes.unicode_minus":False})
    fig,axs=plt.subplots(2,2,figsize=(10.6,6.6),sharex=True)
    fig.subplots_adjust(left=.12,right=.97,wspace=.24,hspace=.23,top=.81,bottom=.17)
    fig.text(.05,.944,"两份状态读相同输入，然后各自行动",fontsize=16,weight="bold")
    fig.text(.05,.877,"实线为双向共享循环权重的三个种子，虚线为普通 GRU 三种子均值。每次放开后测 1,024 tick。",color="#4c5963")
    colors=("#0d777c","#ad591f","#7152a0")
    ticks=(32,128,512,1024)
    for j,(drv,title) in enumerate((("standard","播放标准状态实际走过的历史"),("transplant","播放移入状态实际走过的历史"))):
        axs[0,j].set_title(title,fontsize=11)
        for i in range(2):
            ax=axs[i,j]
            for arch in ("symmetric","plain"):
                ys=[]
                for s in range(3):
                    seq=[next(r for r in rows if (r["arch"],r["seed"],r["driver"],r["ticks"])==(arch,s,drv,k)) for k in ticks]
                    yy=[r["ratio"] for r in seq] if i==0 else [
                        100*(r["pair"]["transplant"]["clean_completed"]-r["pair"]["standard"]["clean_completed"])/r["pair"]["standard"]["matured"] for r in seq]
                    if arch=="symmetric":
                        ax.plot(range(4),yy,"o-",color=colors[s],ms=4,label=f"共享 · 种子 {s}")
                    ys.append(yy)
                if arch=="plain":
                    ax.plot(range(4),np.mean(ys,axis=0),"x--",color="#525b67",label="普通 GRU")
            ax.axhline(0,color="#8d989c",lw=.7);ax.grid(axis="y",alpha=.18);ax.spines[["top","right"]].set_visible(False)
            if i==0:
                ax.set_ylim(-.07,1.08)
            else:
                ax.set_ylim(-7,52);ax.set_xticks(range(4),[str(k) for k in ticks]);ax.set_xlabel("固定输入时长（tick）")
    axs[0,0].set_ylabel("残留状态距离 / 初始距离")
    axs[1,0].set_ylabel("释放后严格完成率差\n移入减标准（百分点）")
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc="lower center",ncol=4,frameon=False)
    fig.savefig(output/"state-release.png",dpi=150,facecolor="white");plt.close(fig)


if __name__=="__main__":
    main()
