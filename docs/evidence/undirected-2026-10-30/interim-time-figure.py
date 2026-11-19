"""Plot only the completed, separately declared seed-0 time-factor supplement."""
from pathlib import Path
import hashlib
import json
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

folder = Path(__file__).resolve().parent
source = folder / "time-factor-interim-2026-11-01.json"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
data = json.loads(source.read_bytes())
assert data["complete"] and data["additional_forks_completed"] == 32
assert data["independent_trained_seeds"] == 1 and len(data["rows"]) == 64
inputs = {r["raw_path"]: r["raw_sha256"] for r in data["rows"]}
assert all(sha(Path(f)) == digest for f, digest in inputs.items())
models = [(i, m) for i in ("core", "ports") for m in ("undirected", "untied")]
labels = ["可学习接口 · 无向", "可学习接口 · 解绑", "统一端口 · 无向", "统一端口 · 解绑"]
conditions = ["short", "long_wait", "long_idle", "long_both"]
table = []
for interface, mode in models:
    cells = []
    for condition in conditions:
        rows = [r for r in data["rows"] if (r["interface"], r["mode"], r["condition"]) == (interface, mode, condition)]
        assert sorted(r["family"] for r in rows) == [0, 1, 2, 3] and all(r["primary"]["matured"] for r in rows)
        cells.append(100*statistics.mean(r["primary"]["clean_completed"]/r["primary"]["matured"] for r in rows))
    table.append(cells)
plt.rcParams.update({"font.family":["Microsoft YaHei", "DejaVu Sans"], "font.size":10, "axes.unicode_minus":False,
                     "svg.fonttype":"none"})
fig, (a, b) = plt.subplots(1, 2, figsize=(11.8, 4.65), gridspec_kw={"width_ratios":[1.18, 1]})
fig.subplots_adjust(left=.128, right=.973, top=.82, bottom=.24, wspace=.36)
pal = LinearSegmentedColormap.from_list("blue", ["#f1f3f1", "#b2d4cd", "#28786f", "#164e48"])
bound = max(max(v) for v in table)
a.imshow(table, cmap=pal, vmin=0, vmax=max(1,bound), aspect="auto")
a.set_xticks(range(4), ["均短", "仅长处理", "仅长空档", "均长"])
a.set_yticks(range(4), labels)
a.tick_params(axis="both", length=0, pad=7)
a.set_title("冻结测试  |  严格完成率", loc="left", pad=14, weight="bold")
for i in range(4):
    for j in range(4):
        val = table[i][j]
        a.text(j, i, f"{val:.2f}%" if val else "0%", ha="center", va="center",
               color="white" if val > .65*bound else "#153d36", weight="bold")
for axis in (a,b):
    for side in ("top","right"):
        axis.spines[side].set_visible(False)
a.spines["left"].set_visible(False); a.spines["bottom"].set_visible(False)
for k, (interface, mode) in enumerate(models):
    ident = f"seed_000__{interface}__{mode}__recorded512__fixed__warmup"
    entry = data["live_training_coverage"][ident]
    assert sha(Path(entry["path"])) == entry["sha256"]
    rates = [1000*r["target_action_counts"]["FINISH_total"]/r["positions"] for r in entry["ranges"]]
    b.plot(range(3), rates, linestyle="-" if mode=="undirected" else "--", marker="o" if mode=="undirected" else "x",
           color="#277a70" if interface=="core" else "#ae612f", linewidth=1.8, label=labels[k])
b.set_title("实时训练  |  FINISH 标签的出现频率", loc="left", pad=14, weight="bold")
b.set_ylabel("每 1,000 个动作位置")
b.set_xticks(range(3), ["短课程", "宽范围前半", "宽范围后半"])
b.set_ylim(-6, max(150, b.get_ylim()[1]))
b.grid(axis="y", color="#d9deda", linewidth=.7)
b.legend(frameon=False, fontsize=8, loc="upper right")
fig.suptitle("接口改变了行为，也改变了模型随后能学到的经历", x=.045, y=.96, ha="left", fontsize=16, weight="bold")
fig.text(.045,.09,"种子 0，固定图，512 tick 信用。左：主训练末 196,608 tick，各单元等权平均四种任务规则。单因素条件为事后补测。",fontsize=9)
fig.text(.045,.042,"右：三个连续阶段各 65,536 tick，仅计实时监督，不含历史复习。曲线描述各模型实际经历，不单独证明因果。",fontsize=9)
for ext in ("png","svg"):
    fig.savefig(folder / f"interim-time-and-coverage.{ext}", dpi=165, facecolor="white")
plt.close(fig)
print(json.dumps({"source_sha256":sha(source),"raw_inputs":len(inputs),"rows":len(data["rows"]),"training_seeds":1}))
