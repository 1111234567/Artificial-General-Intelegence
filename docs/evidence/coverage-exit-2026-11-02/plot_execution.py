"""Draw full-update timings and four-core scheduling from completed measurements."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
PERF = HERE.parents[2] / "data/coverage-exit/perf"


def main():
    inputs = {}
    def read(name):
        path = PERF / (name + ".json")
        data = path.read_bytes()
        inputs[path.name] = hashlib.sha256(data).hexdigest()
        return json.loads(data)
    window = read("recorded-window-cpu-v1")
    rows = []
    for name,label in (("ports","512 tick · 统一端口 · CPU"),("core","512 tick · 可学接口 · CPU")):
        entry = next(c for c in window["cases"] if c["interface"] == name)
        rows.append((label,entry["median_seconds"]["reference"],entry["median_seconds"]["optimized"]))
    for name,label in (("cpu","完整 RTRL · CPU"),("gpu","完整 RTRL · GPU")):
        data = read(f"block-rtrl-{name}-real-cp")["cases"][0]
        timing = data["timings"]["full_native_train_update"]["median_seconds"]
        rows.append((label,timing["native"],timing["analytic"]))
    schedule = read("block-rtrl-four-core-scheduling")
    layouts = defaultdict(dict)
    for sample in schedule["cases"]:
        if sample["total_measured_updates"] != 12:
            raise ValueError("unequal workload in scheduling table")
        layouts[sample["concurrent_jobs"]][sample["round"]] = sample["updates_per_second"]
    if sorted(layouts) != [1,2,4] or any(sorted(pair) != [0,1] for pair in layouts.values()):
        raise ValueError("incomplete paired scheduling rounds")
    plt.rcParams.update({"font.family":["Microsoft YaHei","DejaVu Sans"],"font.size":10,
                         "svg.fonttype":"none","axes.unicode_minus":False})
    fig,(a,b)=plt.subplots(1,2,figsize=(12,4.7),gridspec_kw={"width_ratios":[1.20,1]})
    fig.subplots_adjust(left=.178,right=.973,top=.78,bottom=.23,wspace=.36)
    for i,(label,old,new) in enumerate(rows):
        a.barh(i-.15,old,height=.25,color="#9aadb5",label="原执行" if i==0 else None)
        a.barh(i+.15,new,height=.25,color="#20766d",label="优化后" if i==0 else None)
        a.text(old+.05,i-.15,f"{old:.2f}",va="center",fontsize=9)
        a.text(new+.05,i+.15,f"{new:.2f}",va="center",fontsize=9)
    a.invert_yaxis()
    a.set_yticks(range(4),[r[0] for r in rows])
    a.set_xlim(0,max(r[1] for r in rows)*1.13)
    a.set_xlabel("秒 / 完整训练更新")
    a.legend(frameon=False,loc="upper right")
    a.set_title("单个模型的训练时间",loc="left",weight="bold",pad=12)
    medians = [median(layouts[n].values()) for n in (1,2,4)]
    b.bar(range(3),medians,width=.55,color="#d2e6e1")
    for i,n in enumerate((1,2,4)):
        b.scatter([i-.06,i+.06],[layouts[n][0],layouts[n][1]],s=28,color="#18584f",zorder=3)
    b.set_xticks(range(3),["1 作业 × 4 线程","2 作业 × 2 线程","4 作业 × 1 线程"])
    b.set_ylabel("完成更新数 / 秒")
    b.set_title("固定四个物理核的总吞吐",loc="left",weight="bold",pad=12)
    b.set_ylim(0,max(max(pair.values()) for pair in layouts.values())*1.14)
    for axis in (a,b):
        for side in ("top","right"):
            axis.spines[side].set_visible(False)
    fig.suptitle("真实存档上的 FP64 训练",x=.035,y=.96,ha="left",weight="bold",fontsize=16)
    fig.text(.035,.085,"左：H=80，8 个并行环境，每次 128 tick，含世界推进和 Adam。恢复、外部写盘与独立审计另计。",fontsize=9)
    fig.text(.035,.035,"右：每种布局完成同样的 12 次 RTRL 更新，两个点为正序、逆序两轮。GPU 另有外部负载。",fontsize=9)
    for ext in ("png","svg"):
        fig.savefig(HERE/f"execution.{ext}",dpi=165,facecolor="white")
    plt.close(fig)
    (HERE/"execution-inputs.json").write_text(json.dumps(inputs,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
