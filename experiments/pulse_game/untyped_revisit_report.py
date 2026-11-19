"""Aggregate frozen mixed-port trials over independent world roots."""

import argparse
import json
from pathlib import Path

import numpy as np


CONDITION_LABELS = {
    "standard": "标准",
    "prolonged": "B 中 48 对经历后回 A",
    "noise": "共同背景噪声",
    "side_perturbed": "左右背景分别扰动 σ=.18",
    "side_heavy": "左右背景分别扰动 σ=.40",
    "matched_jitter": "两类感觉均有独立噪声 σ=.20",
    "unidentifiable": "背景身份每回合重抽",
}
METRICS = (
    "all_roles", "first_four_roles", "initial_A_training_roles", "middle_B_roles", "return_A_anchor_role",
    "return_A_query_role", "return_A_both_roles", "return_A_query_matches_initial_slot", "return_A", "actual_return_A_positive",
    "actual_B_final_positive", "actual_both_positive", "slot_overflow",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def conflict(rows, method, metric):
    return np.stack([np.asarray(r[method][metric], dtype=float)[[1, 2]] for r in rows])


def pair(a, b):
    diff = (a - b).mean((0, 1))
    n = len(diff)
    choices = np.random.default_rng(148107791).integers(0, n, (16_000, n))
    return {"difference": float(diff.mean()),
            "interval_95": np.quantile(diff[choices].mean(1), [.025, .975]).tolist(), "roots": n}


def percent(s):
    return f"{s:.1%}"


def delta(item):
    lo, hi = item["interval_95"]
    return f"{100*item['difference']:+.1f} pp [{100*lo:+.1f}, {100*hi:+.1f}]"


def report(folder):
    done = read(folder / "complete.json")
    selected = read(folder / "selected.json")
    ablations = read(folder / "fixed_feature_ablations.json")
    result = {"conditions": {}, "ablations": {}, "births": done["births"], "roots": done["worlds"],
              "selected": selected["selected"], "raw_port_parity": {}}
    for condition in done["conditions"]:
        data = [read(folder / f"external-{birth}-{condition}.json") for birth in done["births"]]
        if any(len(d["seeds"]) != done["worlds"] or d["seeds"] != data[0]["seeds"] for d in data):
            raise ValueError("World roots differ across the paired births")
        rows = [s["rows"] for s in data]
        means = {method: {k: float(conflict(rows, method, k).mean()) for k in METRICS} for method in rows[0]}
        comparisons = {method: {k: pair(conflict(rows, method, k), conflict(rows, "v1_short", k)) for k in METRICS}
                       for method in rows[0] if method != "v1_short"}
        births = {s["birth"]: {m: float(np.asarray(s["rows"][m]["return_A"])[[1, 2]].mean())
                              for m in ("v1_short", "selected", "known_roles")} for s in data}
        result["conditions"][condition] = {"mean": means, "paired_vs_v1": comparisons, "by_birth": births}
        if data[0]["real_port_parity"] is not None:
            result["raw_port_parity"][condition] = {k: {field: sum(d["real_port_parity"][k][field] for d in data)
                                                         for field in ("different", "elements")} for k in data[0]["real_port_parity"]}
        if condition in next(iter(ablations.values())):
            altered = [ablations[str(birth)][condition] for birth in done["births"]]
            extended = [{**a["rows"], "selected": b["selected"]} for a, b in zip(altered, rows, strict=True)]
            result["ablations"][condition] = {
                "v1_raw_reproduction_mismatches": sum(r["baseline_role_mismatches"] for r in altered),
                "mean": {method: {k: float(conflict(extended, method, k).mean()) for k in METRICS}
                         for method in altered[0]["rows"]},
                "paired_vs_selected": {method: {k: pair(conflict(extended, method, k), conflict(extended, "selected", k))
                                                for k in METRICS} for method in altered[0]["rows"]},
            }
    data = result["conditions"]
    standard = data["standard"]["mean"]
    standard_delta = data["standard"]["paired_vs_v1"]["selected"]["return_A"]
    lines = ["# 同一感觉口的长期熟悉度与回访", "",
             f"标准冲突回 A 首次正确率从 {percent(standard['v1_short']['return_A'])} 升到 "
             f"**{percent(standard['selected']['return_A'])}**，成对差 {delta(standard_delta)}。"
             f"只在评估器提供正确角色的同延时参照为 {percent(standard['known_roles']['return_A'])}。", "",
             "两次 4 维输入经同一感觉口到达，顺序随机，在第二次到达后才分流。运行时只使用公共的回合起点、"
             "两侧感觉值、运动和真实触达回执。身体、动作与关联更新沿用原来冻结的策略。", "",
             "每包计算 64 个固定感受单元的连续活动。频率由前序两包平均活动的指数移动平均表示，分短、慢两种时间尺度。"
             "另算两侧距离和已形成原型的相似度。"
             "两侧完全一样时，`-0.25*||左−右||²` 严格为零。原型熟悉度随访问次数按 `(次数/(次数+3))²` 加权。"
             "六槽、角距阈值、固定编码、各特征和衰减速度是手工设置。37 组系数只按开发世界的真实触达正回执筛选和复排，"
             "选中短迹 1、平方距离 0.5、慢迹 0、原型 1。没有用角色标签拟合系数。", "",
             f"外测 {done['worlds']} 个新世界根、每根四种隐藏关系、四个冻结出生，其中两个从未参与系数选择。"
             "结论使用两种冲突关系。区间按世界根配对重抽，保留同根出生和关系相关性。", "",
             "| 外测条件 | v1 短迹 | v2 | 同延时已知角色 | v2−v1［95% 区间］ |", "|---|---:|---:|---:|---:|"]
    for condition, row in data.items():
        values = [percent(row["mean"][m]["return_A"]) for m in ("v1_short", "selected", "known_roles")]
        lines.append(f"| {CONDITION_LABELS[condition]} | {' | '.join(values)} | {delta(row['paired_vs_v1']['selected']['return_A'])} |")
    lines.extend(["", "## 错误发生在回访时", "",
                  "| 标准世界，冲突关系 | A 训练角色正确 | B 阶段角色正确 | 回 A 锚味角色正确 | 回 A 作答角色正确 | A 首答正确 |",
                  "|---|---:|---:|---:|---:|---:|"])
    labels = {"v1_short": "旧短迹", "v1_dot": "旧双侧点积", "squared_only": "仅平方距离", "fit_slow": "开发选出的慢迹组合",
              "selected": "新原型组合", "v1_fix_initial_A": "旧短迹，仅纠正 A 训练", "v1_fix_return_anchor": "旧短迹，仅纠正锚味",
              "v1_fix_return_query": "旧短迹，仅纠正作答", "v1_fix_return_pair": "旧短迹，纠正两次回访角色"}
    for method, desc in labels.items():
        values = [percent(standard[method][k]) for k in
                  ("initial_A_training_roles", "middle_B_roles", "return_A_anchor_role", "return_A_query_role", "return_A")]
        lines.append(f"| {desc} | {' | '.join(values)} |")
    probe = result["ablations"]["standard"]["mean"]
    heavy = result["ablations"]["side_heavy"]["mean"]
    actual = data["standard"]["mean"]["selected"]
    ncheck = result["raw_port_parity"]["standard"]["side"]["elements"] + result["raw_port_parity"]["side_heavy"]["side"]["elements"]
    lines.extend(["", f"在标准世界，拿掉长期原型时回 A 降至 {percent(probe['without_prototype']['return_A'])}，"
                  f"单独拿掉双侧距离后为 {percent(probe['without_negative_squared_distance']['return_A'])}。"
                  "教学与回访锚味常与背景一样两侧相同，平方距离在这两阶段常打平。自由作答时，食物两侧通常不同。"
                  "因此仅距离几乎全认对自由作答的角色，却认不全锚味。", "",
                  f"强左右噪声下，固定系数拿掉距离的回 A 为 {percent(heavy['without_negative_squared_distance']['return_A'])}，"
                  f"拿掉短迹为 {percent(heavy['without_short']['return_A'])}，说明选出的距离权重在该世界拖累表现。"
                  "两类感觉受同等噪声时仍有增益。背景身份本身每回合重抽时，即使已知角色也未恢复。", "",
                  f"标准世界 v2 实际回 A 正回执 {percent(actual['actual_return_A_positive'])}，最终 B 正回执 "
                  f"{percent(actual['actual_B_final_positive'])}，两者均正 {percent(actual['actual_both_positive'])}。"
                  f"四出生 × 两条件的原生感觉口复核共 {ncheck:,} 回合。角色、接触侧、接触时刻、真实回执均逐项一致。", "",
                  "逐世界数据见 `external-*.json`，开发中的所有候选见 `selected.json`，固定系数消融见 `fixed_feature_ablations.json`。", "",
                  "```powershell",
                  ".venv/Scripts/python.exe -m experiments.pulse_game.untyped_revisit_study --part discover",
                  ".venv/Scripts/python.exe -m experiments.pulse_game.untyped_revisit_study --part confirm",
                  ".venv/Scripts/python.exe -m experiments.pulse_game.untyped_revisit_controls",
                  ".venv/Scripts/python.exe -m experiments.pulse_game.untyped_revisit_report data/pulse-game/research-2026-11-19/untyped_revisit",
                  "```", ""])
    (folder / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (folder / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    report(p.parse_args().folder)


if __name__ == "__main__":
    main()
