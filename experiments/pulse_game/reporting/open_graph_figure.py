"""Show the registered native graph and its silent, matched orthogonal-preference forks."""
import json
from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np

from ..ring_logic_study import REPO
from .. import unified_live_graph


def figure(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import ListedColormap

    orth=folder/"orthogonal"
    data=json.loads((orth/"summary.json").read_bytes())
    with zipfile.ZipFile(orth/"physical.zip") as z:
        after=np.load(BytesIO(z.read("joint/B/end_mask.npy")))[0,0,0]
    before=unified_live_graph.births([data["births"][0]])[0][0]
    assert before.shape==after.shape==(16,16)
    font=next((n for n in ("Microsoft YaHei","Noto Sans CJK SC","SimHei")
               if n in {f.name for f in font_manager.fontManager.ttflist}),None)
    if font:plt.rcParams["font.family"]=font
    fig,(a,b)=plt.subplots(1,2,figsize=(9.9,3.9),layout="constrained",gridspec_kw={"width_ratios":[1,1.45]})
    colors=["#f0f1ef","#345875","#bd634a","#358363"]
    labels=["保留","退出","新增"] if font else ["Retained","Retired","New"]
    a.imshow(before.astype(int)*2+after.astype(int),vmin=0,vmax=3,
             cmap=ListedColormap([colors[0],colors[3],colors[2],colors[1]]),interpolation="nearest")
    limits=[7.5,9.5,11.5,13.5]
    for x in limits:
        a.axvline(x,color="#afb1ad",linewidth=.8);a.axhline(x,color="#afb1ad",linewidth=.8)
    groups=["感觉","动作","味觉","接触","内部"] if font else ["Senses","Motor","Taste","Touch","Other"]
    a.set(xticks=[3.5,8.5,10.5,12.5,14.5],yticks=[3.5,8.5,10.5,12.5,14.5],
          xticklabels=groups,yticklabels=groups,xlabel="发送节点" if font else "Sending unit",
          ylabel="接收节点" if font else "Receiving unit",title="一条生命的接线变化" if font else "Connectivity in one life")
    a.tick_params(axis="x",labelrotation=36)
    from matplotlib.patches import Patch
    a.legend([Patch(color=c) for c in colors[1:]],labels,loc="upper center",bbox_to_anchor=(.5,-.28),
             ncol=3,frameon=False,handlelength=1,columnspacing=.8)
    conditions=["joint/B/silent","joint/B/activity","joint/B/synapses","placebo/B/silent"]
    names=["保留自身图和状态","仅换另一生命的活动","换另一生命的接线和权重","用随机味觉教学"] if font else [
        "Own graph and state","Swap activity only","Swap synaptic graph","Random teaching taste"]
    y=np.arange(len(conditions))[::-1]
    for i,k in enumerate(conditions):
        e=data["rates"][k];m=e["mean"]*100;lo,hi=np.asarray(e["ci95"])*100
        b.plot([lo,hi],[y[i]]*2,color=colors[1],lw=2)
        b.plot(m,y[i],"o",color=colors[2] if i==2 else colors[1])
        b.text(hi+2,y[i],f"{m:.1f}%",va="center",fontsize=9)
    b.axvline(50,color="#888",linewidth=1,linestyle=":")
    b.set(yticks=y,yticklabels=names,xlim=(0,101),ylim=(-.5,3.6),xticks=[0,25,50,75,100],
          xlabel="在新偏好下实际选对（%）" if font else "Correct physical choice (%)",
          title="正交新偏好：同一状态的静默探针" if font else "Orthogonal preference: silent forks")
    b.spines[["top","right","left"]].set_visible(False);b.tick_params(axis="y",length=0)
    fig.savefig(orth/"graph-and-choices.png",dpi=165)
    plt.close(fig)


if __name__=="__main__":
    figure(REPO/"docs/evidence/open-graph-learning-2026-11-19")
