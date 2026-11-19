"""Separate the original active gate from the paired-price revision."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .active_symmetry_report import LABELS, load, metric, pct


def extract(root):
    original = load(root, "summary")
    updated = load(root, "v2-confirm")
    native = load(root, "v2-offers-confirm")
    prices = load(root, "v2-prices")
    frozen = load(root, "v2-frozen")
    base = {"original": set(original["outputs"]["held_worlds"]), "paired": set(updated["held_worlds"]),
            "matched_offer_frames": set(prices["held_worlds"]), "native_controls": set(native["world_seeds"])}
    values = list(base.values())
    if any(x & y for i,x in enumerate(values) for y in values[:i]):
        raise ValueError("Held-world panels are not disjoint")
    src = Path(__file__).resolve().parent
    for name, sha in frozen["source"].items():
        if hashlib.sha256((src/name).read_bytes()).hexdigest() != sha:
            raise ValueError(f"Changed frozen source {name}")
    for file, sha in frozen["checkpoints"].items():
        if hashlib.sha256((root/file).read_bytes()).hexdigest() != sha:
            raise ValueError(f"Changed frozen checkpoint {file}")
    score = updated["results"]
    gates = {"same_budget_advantage_is_heldout_positive": score["count_matched_random"]["paired_learned_minus"]["net"]["ci95"][0] > 0.,
             "net_more_than_none": score["none"]["paired_learned_minus"]["net"]["ci95"][0] > 0.,
             "physically_positive_fee_response": prices["results"]["paired"]["average_physical_effect"]["ci95"][0] > 0.,
             "exact_hash_manifest": True}
    bound = load(root, "public-bound")["records"]["active-symmetry-v2-confirm.json"]
    return {"v1":original, "v2":updated, "v2_native":native, "price_grid":prices, "frozen":frozen, "checks":gates,
            "public_bound": bound}


def report(data):
    old = data["v1"]["outputs"]
    now = data["v2"]
    cut = data["price_grid"]["results"]
    ordinary = data["v2_native"]["whole_life"]
    table = ["# 探查策略复验 · 同场景价格训练", "",
             "每区有独立隐藏味道规则。A 探查花一张票和实际费用，身体随后经历四对气味和真回执。直接收获也占一张票，"
             "尝到 B 锚以后物理选择 B 目标。实际收获记 +1 或 −1，收获选择的回执不在本生显示。目标位置和探查牌位置随机。"
             "固定左右脉冲路由、公开区域地址、剩余票及费用是工程接口。气味储层的状态在尝味段之间保留，牌路由阶段由独立的固定马达执行。", "",
             "## 保留初版反例", "",
             f"初版回报训练 gate 的实际净收益为 {metric(old['results']['learned']['net'])}，比同生同区同次数同扣费的随机时机高 "
             f"{metric(old['results']['count_matched_random']['paired_learned_minus']['net'])}。不过短寿命的同状态实测中，高费反而更常触发探查。"
             "初版的训练与回报诊断仍保留在[原结果](active-symmetry-summary.md)。", "",
             "修订版只保留六个未经符号约束的线性输入权重，训练时在相同身体、私有规则、气味和采样随机数上实际运行低费／高费、短寿命／长寿命。"
             "外层更新只使用每条身体轨迹真实发生的净回报。两版的训练配对方式、gate 模型大小与轨迹预算都不同（27,648 与 92,160 个训练生命）。", "",
             "## 新世界的身体回报", "",
             "384 个新世界，两个优化种子、两个新出生图。区间在配对世界上计算，优化种子和图先平均。冷启动后第一次实际收获选择以全部生命为分母。", "",
             "| 策略 | 实际净收益/生 | 探查次数/生 | 实付费用 | 首次无回执 B 正确 |", "|---|---:|---:|---:|---:|"]
    for name, label in LABELS.items():
        r = now["results"][name]
        table.append(f"| {label} | {metric(r['net'])} | {r['probes']['mean']:.2f} | {r['paid']['mean']:.2f} | {pct(r['first_correct'])} |")
    a = now["actual_decisions"]["learned"]
    table.extend(["", f"与同世界、同区、同次数和同费用但随机时机比较，修订 gate 的配对净收益差为 {metric(now['results']['count_matched_random']['paired_learned_minus']['net'])}。"
                  f"至少还剩四票时，低置信的实际探查率为 {pct(a['uncertain_and_time'])}，高置信为 {pct(a['confident_and_time'])}。"
                  f"最后一票实际探查率为 {pct(a['last_ticket'])}。随机对照由评估器事后匹配每区探查数。", "",
                  f"理想公开规划获得完美配对和全签名向量时的上界为每生命 {data['public_bound']['perfect_matching_grid_bound']['80']['mean']:.2f}。"
                  "它依未来区域分布计算期望，仍需推断本生私有规则。", "",
                  "## 真正改变同场景的出价", "",
                  "另在 192 个新身体上，每格固定内部置信、剩余票、当前区域、牌的位置、他区费用以及随机性，只改当区费用。"
                  "实际碰牌才计动作。此项只验证 gate 的动作响应。", ""])
    for name, setting in (("原版", cut["original"]), ("修订版", cut["paired"])):
        table.append(f"- {name}的平均低费减高费实探率为 {metric(setting['average_physical_effect'],100)} 个百分点。"
                     f"高费反而触发探索的配对占 {pct(setting['wrong_direction_fraction'])}。")
    table.extend(["", "相同输入下确定 gate 的动作可以对所有身体都相同，因此格点自举区间可以退化。该区间只刻画已冻住的两次训练。", "",
                  "再用不同的 192 个相同身体场景完整运行实际预算反事实。", "",
                  "| 固定票数 | 两区费用 | 探查次数 | 实际净收益 |", "|---:|---:|---:|---:|"])
    for days in (4, 8, 14):
        for c in (.10, .72):
            r = ordinary[str((days,c))]
            table.append(f"| {days} | {c:.2f} | {r['probes']['mean']:.2f} | {metric(r['net'])} |")
    table.extend(["", "[修订冻结协议](active-symmetry-v2-frozen.json) · [修订闭环](active-symmetry-v2-confirm.json) · "
                  "[物理价格格点](active-symmetry-v2-prices.json) · [实际预算反事实](active-symmetry-v2-offers-confirm.json)"])
    return "\n".join(table) + "\n"


def chart(data, save):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14.3, 4.5), layout="constrained")
    ax = axes[0]
    r = data["v2"]["results"]
    stages = ("none", "once", "count_matched_random", "heuristic", "learned_no_confidence", "learned", "public")
    labels = ("No probes", "One per arena", "Random, same fees", "Hand-set gate", "Learned, confidence hidden", "Return-trained gate", "Public Bayesian")
    for j, key in enumerate(stages):
        q = r[key]["net"]
        mid, lo, hi = q["mean"], *q["ci95"]
        ax.errorbar(mid, j, xerr=[[mid-lo], [hi-mid]], color="#207061" if key == "learned" else "#91674b", marker="o", capsize=2)
    ax.set_yticks(range(len(stages)), labels, fontsize=8)
    ax.set_ylim(len(stages)-.5,-.5)
    ax.axvline(0, color="#777", ls=":", lw=.7)
    ax.grid(axis="x", color="#dddddd", lw=.6)
    ax.set_xlabel("Net reward per life")
    ax.set_title("A   Paired-training gate, held worlds", loc="left", fontweight="bold", fontsize=11)
    ax.spines[["top","right"]].set_visible(False)
    scaled = None
    for i, (key, label) in enumerate((("original", "Original gate"), ("paired", "Linear paired-training gate")), start=1):
        aa = axes[i]
        samples = data["price_grid"]["results"][key]["cells"]
        h = sorted({k["remaining"] for k in samples})
        g = sorted({k["confidence"] for k in samples})
        matrix = np.array([[next(q["low_minus_high_probe"]["mean"] for q in samples if q["remaining"] == n and q["confidence"] == x)
                           for x in g] for n in h])
        scaled = aa.imshow(matrix, aspect="auto", cmap="PiYG", vmin=-1,vmax=1, interpolation="nearest")
        aa.set_xticks(range(len(g)), [f"{x:g}" for x in g])
        aa.set_yticks(range(len(h)), h)
        aa.set_xlabel("Stored confidence signal")
        if i == 1:
            aa.set_ylabel("Public tickets remaining")
        aa.set_title(("B   " if i == 1 else "C   ") + label, loc="left", fontweight="bold", fontsize=11)
    bar = fig.colorbar(scaled, ax=axes[1:], fraction=.032, pad=.02)
    bar.set_label("Physical probe rate at low fee minus high fee")
    fig.savefig(save, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    output = extract(args.directory)
    root = args.directory
    (root / "active-symmetry-final.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    (root / "active-symmetry-final.md").write_text(report(output), encoding="utf-8")
    chart(output, root / "active-symmetry-v2-results.png")


if __name__ == "__main__":
    main()
