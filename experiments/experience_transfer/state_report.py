"""Aggregate the fixed same-input experiment without selecting seeds or cases."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.experience_transfer import state_clamp as s


def mean(values):
    return statistics.mean(values)


def pct(x):
    return f"{100*x:.2f}%"


def number(x):
    return f"{x:.3g}"


def aggregate():
    plan = s.checked()
    bundles = [s.e.read(s.OUT / f"seed_{seed:03d}.json") for seed in range(3)]
    records = [r for b in bundles for r in b["records"]]
    probes = [p for b in bundles for p in b["probes"]]
    audits = [s.e.read(s.OUT / f"audit_seed_{seed:03d}.json") for seed in range(3)]
    pairs, groups = [], []
    for arch in s.ARCHES:
        for driver, point in [("shared", 0)] + [(d, k) for d in s.STATES for k in s.POINTS[1:]]:
            selected = []
            for seed in range(3):
                pair = [next(r for r in records if (r["seed"], r["arch"], r["driver"], r["clamp_ticks"], r["entry_state"]) == (seed, arch, driver, point, name)) for name in s.STATES]
                s.assert_matched(*pair)
                probe = next(p for p in probes if (p["seed"], p["arch"], p["driver"], p["clamp_ticks"]) == (seed, arch, driver, point))
                row = dict(arch=arch, seed=seed, driver=driver, clamp_ticks=point,
                    rms=probe["distance"]["rms_mean"], relative_initial_rms=probe["distance"]["relative_initial_rms"],
                    max_lane_rms=probe["distance"]["rms_max"], next_disagree=probe["next_action"]["disagree"],
                    next_slots=probe["next_action"]["slots"], next_active_disagree=probe["next_action"]["active_disagree"],
                    next_active_slots=probe["next_action"]["active_slots"], next_idle_disagree=probe["next_action"]["idle_disagree"],
                    next_idle_slots=probe["next_action"]["idle_slots"], matured=pair[0]["counts"]["matured"],
                    primary_identity_hash=pair[0]["primary_identity_hash"], start_world_hash=pair[0]["start_world_hash"])
                for name, record in zip(s.STATES, pair):
                    for key, value in record["counts"].items():
                        row[name + "_" + key] = value
                    for key, value in record["metrics"].items():
                        row[name + "_" + key] = value
                row["strict_difference"] = row["transplant_strict_completion_rate"] - row["standard_strict_completion_rate"]
                row["idle_difference_per1000"] = row["transplant_idle_actions_per1000"] - row["standard_idle_actions_per1000"]
                selected.append(row)
                pairs.append(row)
            group = dict(arch=arch, driver=driver, clamp_ticks=point, per_seed=selected,
                mean_rms=mean(r["rms"] for r in selected), mean_relative_initial_rms=mean(r["relative_initial_rms"] for r in selected),
                max_lane_rms=max(r["max_lane_rms"] for r in selected),
                next_disagree=sum(r["next_disagree"] for r in selected), next_slots=sum(r["next_slots"] for r in selected),
                matured_by_seed=[r["matured"] for r in selected], matured_total=sum(r["matured"] for r in selected),
                mean_strict_difference=mean(r["strict_difference"] for r in selected),
                strict_difference_sample_sd=statistics.stdev(r["strict_difference"] for r in selected))
            for name in s.STATES:
                group[name + "_strict_mean"] = mean(r[name + "_strict_completion_rate"] for r in selected)
                group[name + "_idle_per1000_mean"] = mean(r[name + "_idle_actions_per1000"] for r in selected)
                group[name + "_clean_total"] = sum(r[name + "_clean_completed"] for r in selected)
            groups.append(group)
    coverage = {key: sum(a["coverage"][key] for a in audits) for key in audits[0]["coverage"]}
    output = dict(protocol_hash=plan["protocol_hash"], groups=groups, paired_rows=pairs,
        audit_coverage=coverage, all_zero_reproduced=all(r["old_zero_reproduced"] for r in records if r["clamp_ticks"] == 0),
        total_release_branches=len(records), total_history_probes=len(probes),
        mature_nonzero_range=[min(r["matured"] for r in pairs if r["clamp_ticks"]), max(r["matured"] for r in pairs if r["clamp_ticks"])],
        report_source_sha256=s.e.sha(__file__))
    s.e.write(s.OUT / "summary.json", output)
    with (s.OUT / "paired.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    render(output)
    print(json.dumps(dict(branches=len(records), audit_coverage=coverage, report=str(s.OUT / "report.md"))), flush=True)


def render(result):
    groups = result["groups"]
    lines = ["# 相同真实输入钳制，再放开行动", "", "2026-11-03。冻结既有 wide 权重，在相同陌生世界比较标准测试 h 与原训练端点前八个 lane 的 h。三种子、普通 GRU 与循环权重双向共享 GRU。", "",
        "**共享模型的起始状态差异在部分种子上能经受长期相同输入。** seed1 在两种真实播放下读完1024 tick后，h RMS仍约为初始距离的98%–99%，下一次同输入的十一动作选择仍有15–16/16处分歧；各自在相同释放世界自由行动1024 tick时，严格完成为0对36–37。这个种子的持续分离不能只归于两种控制器各自制造了不同的输入回执。其他种子对播放历史的反应不同；本次不据此证明吸引子或全局恢复。", "",
        "## 方案与配对", "",
        "0点复用原 evaluation-v1 的1024 tick WAIT背景与完整世界。标准h由该公开背景重算，移入h沿用 state-transfer-v1 训练端点前八lane，未按成败筛选。各模型权重始终冻结。两条播放分别由标准h或移入h的真实自由argmax驱动，另一个h只读取同一个56维公开观察，包括驱动策略产生的真实动作与回执。每种播放内两h收到逐位相同的输入。", "",
        "在0、32、128、512、1024 tick后保存共同完整世界与两份演化h。状态距离测在下一观察前；动作分歧测让两h读同一个下一观察并各自取十一动作argmax，预览不会修改保存状态。随后各h从该播放端点的相同世界自由行动，按未来新发起且成熟的家族3长请求计分；继承请求不入完成分母。idle按每一分支实际空闲机会计分。0点释放2048 tick以复现旧结果，其余释放1024 tick。两种播放之间的世界与回执可以不同，跨播放比较是历史总效应，不是单独换h。", "",
        f"预冻结 [protocol.json](protocol.json)，协议哈希 `{result['protocol_hash']}`。共{result['total_release_branches']}个自由释放分支，每个非0分支有{result['mature_nonzero_range'][0]}–{result['mature_nonzero_range'][1]}个成熟请求；同一比较的三种子合计约240个。下列百分数与idle率先在种子内计算，再等权平均三个种子。RMS按每lane的192维差值计算，再平均；括号为相对各自0点的距离比例。下一动作分歧是三种子的总数，分母48。", "",
        "## 双向共享循环权重", "",
        "| 播放策略 | tick | h RMS（初始比例） | 同输入下一动作分歧 | 释放严格完成 标准→移入 | 差值/百分点 | idle误动/千机会 标准→移入 | 成熟数/种子 |",
        "|---|---:|---:|---:|---:|---:|---:|---|"]
    for arch in ("symmetric", "plain"):
        if arch == "plain":
            lines.extend(["", "## 普通 GRU 对照", "", "| 播放策略 | tick | h RMS（初始比例） | 同输入下一动作分歧 | 释放严格完成 标准→移入 | 差值/百分点 | idle误动/千机会 标准→移入 | 成熟数/种子 |", "|---|---:|---:|---:|---:|---:|---:|---|"])
        for g in (r for r in groups if r["arch"] == arch):
            label = {"shared":"共同起点", "standard":"标准h驱动", "transplant":"移入h驱动"}[g["driver"]]
            lines.append(f"| {label} | {g['clamp_ticks']} | {number(g['mean_rms'])}（{g['mean_relative_initial_rms']:.3f}） | {g['next_disagree']}/{g['next_slots']} | {pct(g['standard_strict_mean'])}→{pct(g['transplant_strict_mean'])} | {100*g['mean_strict_difference']:+.2f} | {g['standard_idle_per1000_mean']:.2f}→{g['transplant_idle_per1000_mean']:.2f} | {'/'.join(map(str,g['matured_by_seed']))} |")
    lines.extend(["", "## 1024 tick后的逐种子残留", "", "均值不能代替每个种子。以下列出全部共享模型的最长播放点，未筛选分歧大的案例。相同播放内两h释放时面对相同完整世界和未来请求。", "",
        "| seed | 播放 | h RMS / 初始比例 | 最大lane RMS | 下一动作分歧 | 严格完成 标准→移入/成熟 | idle误动 标准→移入（各自分母） |", "|---:|---|---:|---:|---:|---:|---|"])
    for r in (r for r in result["paired_rows"] if r["arch"] == "symmetric" and r["clamp_ticks"] == 1024):
        lines.append(f"| {r['seed']} | {r['driver']} | {r['rms']:.6f} / {r['relative_initial_rms']:.4f} | {r['max_lane_rms']:.6f} | {r['next_disagree']}/16 | {r['standard_clean_completed']}→{r['transplant_clean_completed']}/{r['matured']} | {r['standard_idle_actions']}/{r['standard_idle_opportunities']}→{r['transplant_idle_actions']}/{r['transplant_idle_opportunities']} |")
    lines.extend(["", "seed0读标准h生成的历史后，两个h的释放严格完成计数从32 tick起已相同，但h、动作和idle仍有残留差异；读移入h生成的历史，直到1024 tick后仍有严格完成差异。seed1在两种历史中都保留大距离和动作分歧。seed2的释放差异较小。普通GRU在128及更长播放后的配对释放中，严格完成与idle误动分子均相同；128 tick的idle机会分母仍有1–2个位置之差，512及1024 tick时完成与idle分子分母才全部相同。不能据有限计数相同断言状态逐位相同。", "",
        "这些结果说明：部分差异确实依赖闭环产生了哪种公开历史；同时seed1保留了对共同输入也不消失的隐藏状态差异，并且该差异在释放后继续影响真实完成与空闲控制。两者均存在，不应将所有持续行为差异合并归因于外部反馈，也不应把有限长残留提升为动力系统吸引子的证明。", "",
        "## 独立核验与边界", "",
        f"全部{result['total_release_branches']}个释放分支均独立重算逐请求成熟/普通/严格完成、原始请求身份、idle分子分母，并对全部保存的公开输入作GRU逐tick forward重算，逐位验证logits、真argmax与末h。另重放12条真实播放历史的世界、共同输入、两h、动作和回执；使用原 lifelong.evaluate 完整重放36个释放分支，覆盖全部0点和全部1024点。审计将oracle替换为一旦调用即报错的函数，所有核验通过；0点12个分支逐计数、h/world哈希及请求身份精确复现原 state-transfer-v1。审计覆盖：`{json.dumps(result['audit_coverage'],ensure_ascii=False)}`。首次审计用整序列GRU内核重算时，少量logit出现约4.5e-5的CPU浮点差，超过事先3e-5容差；改为原逐tick边界的forward后要求完全相等，未调整结果或放宽容差。", "",
        "三种子仍只有三个独立训练重复。长任务时长固定8–16，间隔16–32，dt0.25；每条陌生公开流仅一个world seed。非0完成分母只包含释放后新请求，避免把钳制期间控制者已造成的请求错误计作被测h的初始能力。idle机会依赖释放后的行动，需同时读分子分母。有限长、特定播放、特定权重的结果无法保证其他起点、其他任务或无限时间的行为。", "",
        "[完整聚合与三种子值](summary.json) · [全部配对CSV](paired.csv) · [seed0](seed_000.json) · [seed1](seed_001.json) · [seed2](seed_002.json) · [审计0](audit_seed_000.json) · [审计1](audit_seed_001.json) · [审计2](audit_seed_002.json)", ""])
    (s.OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    aggregate()
