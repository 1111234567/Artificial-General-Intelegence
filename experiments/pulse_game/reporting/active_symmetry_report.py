"""Write the compact held-out strategic exploration report."""

import argparse
import hashlib
import json
from pathlib import Path

LABELS = {
    "learned": "回报训练 gate", "learned_no_confidence": "训练 gate 隐藏置信", "count_matched_random": "同次数同费用随机时机",
    "none": "不探查", "once": "每区域固定首探一次", "twice": "每区域固定首探两次", "heuristic": "手写信息价值 gate",
    "public": "公开配对与后验规划",
}


def load(root, suffix):
    return json.loads((root / f"active-symmetry-{suffix}.json").read_text(encoding="utf-8"))


def pct(m):
    return f"{100*m['mean']:.1f}%"


def metric(m, scale=1.):
    return f"{scale*m['mean']:+.2f} [{scale*m['ci95'][0]:+.2f}, {scale*m['ci95'][1]:+.2f}]"


def extract(root):
    held, corrected, frozen = (load(root, x) for x in ("confirm", "offers-confirm", "evaluation-frozen"))
    if set(held["held_worlds"]) & set(corrected["world_seeds"]):
        raise ValueError("The budget effect panel was not independently sampled")
    source = Path(__file__).resolve().parents[1]
    okay = {name: hashlib.sha256((source/name).read_bytes()).hexdigest()==sha for name, sha in frozen["source"].items()}
    trained = [load(root, f"trained-{seed}") for seed in held["optimizer_seeds"]]
    checks = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
              for name, digest in frozen["checkpoints"].items()}
    if not (all(okay.values()) and all(checks.values())):
        raise ValueError("A model/code fingerprint no longer agrees with the frozen test")
    for x in trained:
        if (set(x["training_worlds"]) | set(x["development_worlds"])) & (set(held["held_worlds"]) | set(corrected["world_seeds"])):
            raise ValueError("A held scene leaked to training or development")
    findings = held["results"]
    causal = corrected["actual_gate_effects_given_same_body"]
    strategic = {
        "beats_no_probe_on_real_net": findings["none"]["paired_learned_minus"]["net"]["ci95"][0] > 0,
        "beats_same_probe_budget_random_at_real_net": findings["count_matched_random"]["paired_learned_minus"]["net"]["ci95"][0] > 0,
        "more_often_probes_under_more_favorable_display": any(x["physical_probe_difference"]["ci95"][0] > 0 for k, x in causal.items()
                                                               if not k.startswith("unknown")),
        "all_frozen_fingerprints_verified": True,
    }
    bound = load(root, "public-bound")["records"]["active-symmetry-confirm.json"]
    return {"outputs": held, "same_scene_offer_controls": corrected, "frozen_protocol": frozen, "checks": strategic,
            "public_bound": bound,
            "gate_training": [{"optimizer_seed": r["optimizer_seed"], "development_best": r["development_best"],
                               "history": r["history"]} for r in trained]}


def text(data):
    p, tests = data["outputs"], data["same_scene_offer_controls"]
    r = p["results"]
    actual = p["actual_decisions"]["learned"]
    result = ["# 用真实回报学习何时探查", "",
              "每个生命有 4、8 或 14 张票。每张票来到两个区域之一，区域规则独立抽取。公开显示剩余票和两区探查费用。"
              "探查牌与收获牌随机摆在两侧。碰到探查牌扣当区费用、用掉一张票，再亲身尝一组四对气味，收到签名回执。"
              "碰收获牌也用一张票，先实际尝一个 B 锚，再挑一个 B 目标，正确或错误分别记账 +1、−1。第二口的回执延迟到生命以后统计，"
              "避免收获本身变成免费的探查通道。错过物理接触时不给未完成动作计收益。", "",
              "牌到左右脉冲的路由是工程接口。各区域用同一规则的本地 LIF 气味控制器，但各自有固定容量关联权和方向增益。"
              "每次收获的锚权组由当前公开背景触发清空。训练的是 19 参数的探查 gate。它读取当前区域、公开开销、剩余票和当前神经置信，"
              "使用实际对应的身体净收益与 actor–critic 策略梯度，评价器的未选回报与私有规则不进入 gate。", "",
              "## 独立世界的实际净收益", "",
              f"{len(p['held_worlds'])} 个新世界，两次独立训练与两个新出生图。区间先对同世界模型重复取均值，再按世界配对重采样。"
              f"首次盲选是本生第一个收获的 B 选择，没收获也占分母。", "",
              "| 策略 | 每生命净收益 | 探查次数 | 扣费 | 首次盲选正确 |", "|---|---:|---:|---:|---:|"]
    for key, name in LABELS.items():
        row = r[key]
        result.append(f"| {name} | {metric(row['net'])} | {row['probes']['mean']:.2f} | {row['paid']['mean']:.2f} | {pct(row['first_correct'])} |")
    shuffled = r["count_matched_random"]["paired_learned_minus"]
    manual = r["heuristic"]["paired_learned_minus"]
    result.extend(["", f"随机时机对照在每世界、每区精确复用训练 gate 的探查次数和总扣费。训练 gate 对此净收益差为 **{metric(shuffled['net'])}**，"
                   f"对手写信息门为 {metric(manual['net'])}。公开后验规划在观察到的气味上先做配对、再按未来区域分布反推。"
                   f"赋予完美配对并使用全签名向量的理想上界为每生命 {data['public_bound']['perfect_matching_grid_bound']['80']['mean']:.2f}。"
                   "这项规划不参加 gate 训练。", "",
                   "## 改变开销、剩余票和置信", "",
                   f"真实轨迹中，至少剩 4 票且置信低于 .15 时探查率 {pct(actual['uncertain_and_time'])}，置信至少 .60 时为 {pct(actual['confident_and_time'])}。"
                   f"最后一张票实际探查率 {pct(actual['last_ticket'])}。这些比率来自同一决策过程的观察。", "",
                   f"另用 {len(tests['world_seeds'])} 个新世界，在同一身体和零知识状态独立改显示票数或费用，只记录实际碰哪一牌。", "",
                   "| 只改变眼前条件 | 实际探查概率差（百分点） |", "|---|---:|"])
    alternates = {"low_minus_high_when_4_unknown": "4 票时低费减高费",
                  "low_minus_high_when_14_unknown": "14 票时低费减高费",
                  "many_minus_few_when_low_unknown": "低费时 14 票减 4 票",
                  "many_minus_few_when_high_unknown": "高费时 14 票减 4 票",
                  "unknown_minus_known_same_low_long_offer": "同费同票时低置信减高置信"}
    for name, label in alternates.items():
        result.append(f"| {label} | {metric(tests['actual_gate_effects_given_same_body'][name]['physical_probe_difference'],100)} |")
    result.extend(["", "最后一行显式给 gate 一个较高的内部置信值，检验同价格、同寿命的实际动作，和上面的自然轨迹分开解释。"
                   "这项收益和信息价值只针对当前关闭收获反馈的条件。", "",
                   "[冻结协议](active-symmetry-evaluation-frozen.json) · [逐世界数据](active-symmetry-confirm.json) · [成本和寿命反事实](active-symmetry-offers-confirm.json)"])
    return "\n".join(result) + "\n"


def chart(data, target):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figs, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    t, x = data["outputs"]["results"], data["same_scene_offer_controls"]
    labels = ("none", "once", "twice", "count_matched_random", "learned_no_confidence", "heuristic", "learned", "public")
    en = ("No probes", "One per arena", "Two per arena", "Random, same fees", "Learned, confidence hidden",
          "Hand-set gate", "Return-trained gate", "Public Bayes reference")
    ax = axes[0]
    for i, stage in enumerate(labels):
        e = t[stage]["net"]
        point, lo, hi = e["mean"], *e["ci95"]
        ax.errorbar(point, i, xerr=[[point-lo], [hi-point]], color="#206e65" if stage == "learned" else "#916349",
                    marker="o", capsize=2)
    ax.set_yticks(range(len(labels)), en, fontsize=8)
    ax.set_ylim(len(labels)-.5,-.5)
    ax.axvline(0, color="#777", ls=":", lw=.7)
    ax.set_xlabel("Actual net reward per life")
    ax.set_title("A   Held-out net returns", loc="left", fontweight="bold", fontsize=11)

    ax = axes[1]
    for i, (code, label) in enumerate(zip(labels, en, strict=True)):
        e = t[code]["first_correct"]
        point, lo, hi = e["mean"], *e["ci95"]
        ax.errorbar(point, i, xerr=[[point-lo], [hi-point]], color="#206e65" if code == "learned" else "#916349",
                    marker="o", capsize=2)
    ax.set_yticks(range(len(labels)), en, fontsize=8)
    ax.set_ylim(len(labels)-.5,-.5)
    ax.xaxis.set_major_formatter(PercentFormatter(1., decimals=0))
    ax.axvline(.5, color="#777", ls=":", lw=.7)
    ax.set_xlim(.31,1.)
    ax.set_xlabel("Correct no-receipt first harvest")
    ax.set_title("B   First B decision", loc="left", fontweight="bold", fontsize=11)

    ax = axes[2]
    displays = (("low_minus_high_when_4_unknown", "Cheaper, 4 tickets"), ("low_minus_high_when_14_unknown", "Cheaper, 14 tickets"),
                ("many_minus_few_when_low_unknown", "More tickets, cheap"), ("many_minus_few_when_high_unknown", "More tickets, costly"),
                ("unknown_minus_known_same_low_long_offer", "Uncertain vs confident"))
    for i, (key, _) in enumerate(displays):
        s = x["actual_gate_effects_given_same_body"][key]["physical_probe_difference"]
        point, lo, hi = s["mean"], *s["ci95"]
        ax.errorbar(point, i, xerr=[[point-lo], [hi-point]], color="#206e65", marker="o", capsize=2)
    ax.set_yticks(range(len(displays)), [t[1] for t in displays], fontsize=8)
    ax.set_ylim(len(displays)-.5,-.5)
    ax.axvline(0, color="#777", ls=":", lw=.7)
    ax.set_xlabel("Change in physical probe rate")
    ax.xaxis.set_major_formatter(PercentFormatter(1., decimals=0))
    ax.set_title("C   Same body, different offer", loc="left", fontweight="bold", fontsize=11)
    for panel in axes:
        panel.spines[["top", "right"]].set_visible(False)
        panel.grid(axis="x", lw=.55, color="#dddddd")
    figs.savefig(target, dpi=150)
    plt.close(figs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    data = extract(args.directory)
    root = args.directory
    (root / "active-symmetry-summary.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    (root / "active-symmetry-summary.md").write_text(text(data), encoding="utf-8")
    chart(data, root / "active-symmetry-results.png")


if __name__ == "__main__":
    main()
