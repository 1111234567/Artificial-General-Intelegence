"""Summarize fixed-endpoint noisy receipt and post-taste commit comparisons."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "docs/evidence/active-review-20261120"
COHORTS = ("original", "independent")


def read_pair(pattern, field):
    return [json.loads((EVIDENCE / pattern.format(c)).read_text(encoding="utf-8"))[field] for c in COHORTS]


def metric(pair, arm, key, *, percent=False):
    scale, digits = (100, 2) if percent else (1, 3)
    def fmt(x):
        return f"{x * scale:.{digits}f}"
    cells = []
    for group in pair:
        entry = group[arm][key]
        low, high = entry["ci95"]
        cells.append(f'{fmt(entry["mean"])} [{fmt(low)}, {fmt(high)}]')
    return " | ".join(cells)


def table(rows):
    return ["| 实际行为或同身体差值 | 原六个体 | 独立六个体 |", "|---|---:|---:|",
            *[f"| {name} | {value} |" for name, value in rows]]


def agreement_report():
    text = [
        "## 两份回执冲突后会不会再买可靠信息", "",
        "同一对象先出现两次，每次味觉各有 20% 概率与其真实价值相反，中途还遇到另一个可靠对象。随后身体选择是否付费再尝一次，再完成一次实际收获。暴露次数和买价在同一对照中相等。每个身体的所有私有赋值与滑误组合按实际概率加权。有些配对生命的真实价值相反，但到付费门前的感觉、行动和味觉完全相同，评估逐条要求这对生命也作出相同的购买决定。", "",
        "三组均能用原来的局部规则改变图。其中一组另有六个感觉共激活量，在每次实际接触时累积对象方向。开放的一个系数可根据上一刻的运动倾向、刚选的方向和实际味觉，调节这一笔的写入幅度。另两组只计实际接触次数或随机打乱这一修正的符号。累计量通过感觉端进入原图，最终仍用原身体买或跳过。两种接线布局从各自的亲本出发，所有正式组各训练 12,000 步。每个区间分别按出生与新世界重采样。", "",
    ]
    for title, pattern in (
        ("双侧输入，额外独立世界复测", "agreement-bilateral-confirmed-{}-12000.json"),
        ("输入只接到可购买的一侧，主留出", "agreement-local-{}-12000.json"),
    ):
        data = read_pair(pattern, "scores")
        def get(arm, key, pct=False):
            return metric(data, arm, "2/pooled/" + key, percent=pct)
        text.extend([f"### {title}", "", *table([
            ("回执冲突减一致的购买概率，百分点", get("agreement","ambiguity_choice_gap",True)),
            ("实际收获减费用，每生", get("agreement","all/net")),
            ("实际时机减同价同购买率重排，每生", get("agreement","all/same_cost_gain")),
            ("同预算训练，相对只计次数的净资源", get("agreement minus count","all/net")),
            ("同预算训练，相对打乱修正符号的净资源", get("agreement minus scrambled","all/net")),
            ("同一身体保留修正系数减当场移除，净资源", get("agreement","all/remove_confirmation_choice_loss")),
        ]), ""])
    text.extend(["### 增加既往接触次数", "",
                 "每次可靠信息定价 0.25。下表是在所有接触数固定的世界内，回执分歧最大减全部一致的实际购买概率差，单位为百分点。3 次时为 2 比 1 对 3 比 0，4 次时为 2 比 2 对 4 比 0。", ""])
    rows = []
    for name, key in (("双侧", "bilateral"), ("购买侧", "local")):
        data = read_pair(f"agreement-{key}-transfer-{{}}-12000.json", "scores")
        for visits in (3, 4):
            rows.append((f"{name}，{visits} 次", metric(data,"agreement",f"{visits}/pooled/ambiguity_choice_gap",percent=True)))
    text.extend([*table(rows),"",
                 "[双侧复测原组](agreement-bilateral-confirmed-original-12000.json) · [独立组](agreement-bilateral-confirmed-independent-12000.json) · [购买侧原组](agreement-local-original-12000.json) · [独立组](agreement-local-independent-12000.json) · [协议](receipt-agreement-plan.json)", ""])
    return text


def commit_report():
    data = read_pair("commit-confirmed-{}-10000.json", "results")
    rows = []
    for arm, name in (("always","总是写"),("constant","固定概率"),("state","读味觉后的 20 个节点"),
                      ("confirmation","读自身运动与真实味觉是否一致")):
        rows.append((f"{name}，选择写入的比例（%）",metric(data,arm,"pooled/all/commit",percent=True)))
        rows.append((f"{name}，随后两题都答对（%）",metric(data,arm,"pooled/all/native/serial",percent=True)))
    for arm,label in (("state","读节点"),("confirmation","读运动与味觉")):
        rows.append((f"{label}，相对本身体同写入率重排（百分点）",
                     metric(data,arm,"pooled/all/native_minus_same_count/serial",percent=True)))
        rows.append((f"{label}，相对本身体强制每次写（百分点）",
                     metric(data,arm,"pooled/all/native_minus_always/serial",percent=True)))
        rows.append((f"{label}，相对同预算另训的总是写组（百分点）",
                     metric(data,arm+" minus always","pooled/all/native/serial",percent=True)))
    for condition,label in (("reversal","对象刚改变奖惩"),("stable","奖惩保持")):
        rows.append((f"读运动与味觉，{label}时写入（%）",
                     metric(data,"confirmation",f"pooled/{condition}/commit",percent=True)))
    text = ["## 已经尝到反馈后，要不要把这次复习写进图", "",
            "先实际学习 A 和 B，再随机复习一个。两者相隔 15°、45° 或 75°，复习对象的奖惩有四分之一概率已经反转。模型先出手，再尝到实际味觉，才决定是否提交本次对权重、偏置及接线的原有更新。跳过时也经过相同拍数。先前的两次学习照常写入。终点是同一条生命先后答两题且都答对，两题之间没有新味觉。", "",
            "每种方案分别做 10,000 次外层更新，训练时每次把同一份真实回执的写与不写两条延续都实际走完。部署时只抽取一次提交动作。对照有两个含义：同一身体随机重排写入检验它有没有挑对时机；独立训练一个每次都写的身体，检验新增的选择能力有没有提高整个系统。下表是共同终点之外另一套留出世界的结果。", "",
            *table(rows), "",
            "### 写入造成的旧记忆与新记忆差别", "",
            "以下只用开放 20 节点写入门的同一身体，在同一份回执后强制写或跳过，然后从同一结果状态分别问复习对象和另一个对象。报写减跳过的正确率差，单位为百分点。", ""]
    split = []
    for scene,title in (("stable_opposed","起初相反且保持"),("stable_agree","起初一致且保持"),
                        ("new_opposed","起初一致，刚变相反"),("resolved_conflict","起初相反，刚变一致")):
        for target,tag in (("reviewed","复习对象"),("unreviewed","另一个对象")):
            split.append((f"{title}，{tag}",metric(data,"state",f"pooled/{scene}/write_minus_skip/{target}",percent=True)))
    text.extend([*table(split), "", "### 长等待", "",
                 "用训练角度以外的一套世界，在 30° 下分别等 4 拍或 64 拍再答题。下表是 20 节点门的两题连续全对率相对同一身体按相同概率随机分配写入的差，单位为百分点。", ""])
    stress=read_pair("commit-stress-{}-10000.json","results")
    text.extend([*table([(f"等 {delay} 拍",metric(stress,"state",f"30/{delay}/all/native_minus_same_count/serial",percent=True)) for delay in (4,64)]), "",
                 "实际落盘的等待评估覆盖 4 与 64 拍；原方案另提及的 16 拍没有读数。此处的旧内容保持来自同一生命中另一个对象的直接测量。单独的普通 A/B、复合任务迁移表覆盖其它训练组，本次写入门尚缺这套跨任务复测。", "",
                 "[独立世界复测原组](commit-confirmed-original-10000.json) · [独立组](commit-confirmed-independent-10000.json) · [等待与角度原组](commit-stress-original-10000.json) · [独立组](commit-stress-independent-10000.json) · [协议](commit-plan.json)", ""])
    return text


def main():
    lines = ["# 读到冲突反馈以后，是否补查、是否写入", "", *agreement_report(), *commit_report()]
    (EVIDENCE/"receipt-and-writing.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
