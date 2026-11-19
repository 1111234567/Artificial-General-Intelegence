"""Render the declared comparisons as readable Chinese tables from the audited bundle."""
from __future__ import annotations

import argparse
import os
import statistics
from pathlib import Path

from experiments.undirected import plot


def three(values, scale=100., signed=False):
    """A missing seed or denominator is never zero and never a smaller mean."""
    if len(values) != 3:
        raise ValueError("expected exactly three declared training seeds")
    defined = [value for value in values if value is not None]
    if len(defined) != 3:
        return f"不可汇总（{len(defined)}/3 有值）"
    mean, sd = statistics.mean(defined) * scale, statistics.stdev(defined) * scale
    return (f"{mean:+.2f}" if signed else f"{mean:.2f}") + f" ± {sd:.2f}"


def paired(*vectors, weights):
    if len(vectors) != len(weights) or any(len(v) != 3 for v in vectors):
        raise ValueError("paired contrast needs matched seeds 0,1,2 for every condition")
    return [None if any(x is None for x in row) else sum(w*x for w, x in zip(weights, row))
            for row in zip(*vectors)]


def audit_disclosure(header):
    comparison = plot._comparison_contract(header)
    rows = comparison["per_seed"]
    failed = [row for row in rows if row["training_trace"]["status"] == "failed"]
    out = ["## 审计范围与对照状态", "",
        "审计通过的范围是来源、各分支内部一致性和计数/比率，状态为 **passed**。"
        "原对照预期是否通过，单独列在下面。", "",
        "首次正式报告的严格相等检查失败后，本报告层作了事后修订。原失败和实际差异均保留。"
        "冻结运行未修改，也未增加数值容差。", "",
        f"原全局检查 `{comparison['check_code']}`：**{comparison['status']}**。"]
    if failed:
        affected = "；".join(f"seed {row['seed']} 首次差异 {row['training_trace']['first_mismatch_tick']:,} tick" for row in failed)
        out += ["固定图与同探测固定图“全部种子逐位相同”的原预期未通过：" + affected + "。"
            "检验完成不表示原预期通过；差异原因尚未确定，不能直接归因为浮点尾数。"]
    else:
        out += ["当前实际记录未发现逐位差异；空历史不构成完整终点等价的证据。"]
    prior = header.get("reporting_revision", {}).get("prior_failed_report")
    if prior is not None:
        out += [f"此前首次正式报告仍保留 **{prior['status']}**：seed {prior['seed']}，"
            f"{prior['first_mismatch_tick']:,} tick；原失败日志 SHA-256：`{prior['sha256']}`。"]
    out += ["", "训练动作与冻结评估的动作、世界及计数分别比较。训练动作相同不推出冻结分叉动作或计数相同。", "",
        "| 种子 | 训练学习态不同块 | 训练动作/世界/数据不同块 | 外界命令不同块 | 冻结评估动作不同分叉 | 冻结评估世界/行为计数不同分叉 | 冻结评估请求/内容诊断不同分叉 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        train, evaluation = row["training_trace"]["groups"], row["frozen_evaluation"]["groups"]
        out.append(f"| {row['seed']} | {train['learning_state']['mismatch_count']} | {train['behavior']['mismatch_count']} | "
                   f"{train['external_stream']['mismatch_count']} | {evaluation['actions']['mismatch_count']} | "
                   f"{evaluation['world_and_behavior_counts']['mismatch_count']} | "
                   f"{evaluation['request_and_content_diagnostics']['mismatch_count']} |")
    out += ["", "“世界/行为计数”仅比较保存的 end_world_state_hash、post_fork_counters、opportunities 和 primary。"
        "“请求/内容诊断”另比较 request_diagnosis、content_by_cohort_phase_age 和 inactive_history。"
        "这两栏不能互相替代。", "",
        "| 种子 | 已列评估字段中整数叶不同的分叉 | 浮点叶不同的分叉 |",
        "|---|---:|---:|"]
    for row in rows:
        numeric = row["frozen_evaluation"]["numeric_leaf_differences"]
        out.append(f"| {row['seed']} | {numeric['integer']['record_count']} | {numeric['float']['record_count']} |")
    differences = [(row["seed"], label, field, value) for row in rows
        for key, label in (("training_trace", "训练"), ("frozen_evaluation", "冻结评估"))
        for field, value in row[key]["fields"].items() if value["mismatch_count"]]
    if differences:
        out += ["", "下表按字段记录精确差异次数和首次 tick；冻结评估在一个 tick 有四家族、两条件的多个分叉。", "",
            "| 种子 | 范围 | 字段 | 不同记录数 | 首次 tick |", "|---|---|---|---:|---:|"]
        for seed, label, field, value in differences:
            out.append(f"| {seed} | {label} | `{field}` | {value['mismatch_count']} | {value['first_mismatch_tick']} |")
    return out + [""]


def render(bundle):
    h, tables = bundle["header"], bundle["tables"]
    m, w, end = h["primary_tick"], h["checkpoint_ticks"][-2], h["checkpoint_ticks"][-1]
    counts = h["counts"]
    branches = {(plot._key(b), b["seed"]): b for b in tables["branches"]}
    variants = [plot._key(v) for v in h["variants"]]
    avg = {(plot._key(r), r["seed"], plot._tick(r), r["condition"]): r for r in tables["family_average_per_seed"]}
    family = {(plot._key(r), r["seed"], plot._tick(r), r["condition"], r["family"]): r for r in tables["per_seed"]}
    def values(v, tick=m, metric="strict_completion", condition="long_both", f=None):
        result = []
        for seed in plot.SEEDS:
            key = (v, seed, tick, condition)
            row = avg.get(key) if f is None else family.get((*key, f))
            result.append(None if branches[v, seed]["status"] != "success" or row is None else row[metric])
        return result
    p = plot._variant
    out = ["# 无向神经图：正式对照结果", "", *audit_disclosure(h),
        f"共 {counts['accounted_branches']} 个分支：同预算训练成功 {counts['successful_branches']} 个，"
        f"数值失败 {counts['numerical_failure_branches']} 个；实际冻结评估 {counts['actual_forks']}/{counts['declared_forks']} 个。",
        "", "表内 ± 为三个独立训练种子的样本标准差。除单列的 family 0 外，先在每个种子内等权平均四种规则的比率。"
        "严格完成要求请求成功且过程中没有错误；普通完成允许中途出错。二者的分母仅含已成熟、符合条件的请求。"
        "空闲误动作以实际空闲机会为分母。", "", "## 14 个条件的行为", "",
        f"主训练末为 {m:,} tick，撤出结束为 {w:,} tick，返回结束为 {end:,} tick。"
        "下表的“长”同时延长处理时间和空档。除明确标注返回末，均取主训练末。", "",
        "| 接口 / 方向 / 信用 / 结构 / 课程 | 短：严格 % | 长：严格 % | 长：普通 % | 长：误动作/千空闲 | 返回末长：严格 % |",
        "|---|---:|---:|---:|---:|---:|"]
    for v in variants:
        out.append("| " + " / ".join(v) + " | " + " | ".join([
            three(values(v, condition="short")), three(values(v)), three(values(v, metric="completion")),
            three(values(v, metric="inactive_per_1000_idle"), 1), three(values(v, end))]) + " |")
    out += ["", "## 同种子配对差", "", "先对同一训练种子做差，再汇总三个差值。完成率的差用百分点，误动作的差用每千次空闲机会；后者正值表示更多误动作。均取主训练末长条件。",
            "", "| 左减右 | 严格完成差 | 普通完成差 | 空闲误动作差 |", "|---|---:|---:|---:|"]
    comparisons = [
        ("core 内：无向 − 解绑（512）", p(interface="core", credit="recorded512"), p(interface="core", credit="recorded512", mode="untied")),
        ("ports 内：无向 − 解绑（512）", p(credit="recorded512"), p(credit="recorded512", mode="untied")),
        ("无向：ports − core（512）", p(credit="recorded512"), p(credit="recorded512", interface="core")),
        ("解绑：ports − core（512）", p(credit="recorded512", mode="untied"), p(credit="recorded512", mode="untied", interface="core")),
        ("ports 内：无向 − 解绑（RTRL）", p(), p(mode="untied")),
        ("512 − 128", p(credit="recorded512"), p(credit="tbptt128")),
        ("RTRL − 512", p(), p(credit="recorded512")),
        ("RTRL − 每块清空敏感度", p(), p(credit="rtrl_reset")),
        ("联合 − 只允许渐退", p(structure="both"), p(structure="retire")),
        ("联合 − 只允许生长", p(structure="both"), p(structure="grow")),
        ("只允许生长 − 同探测固定图", p(structure="grow"), p(structure="fixed_probe")),
        ("只允许渐退 − 同探测固定图", p(structure="retire"), p(structure="fixed_probe")),
        ("从出生起宽范围 − 短到长（128）", p(credit="tbptt128", course="wide"), p(credit="tbptt128")),
        ("从出生起宽范围 − 短到长（RTRL）", p(course="wide"), p())]
    metrics = (("strict_completion", 100), ("completion", 100), ("inactive_per_1000_idle", 1))
    for label, left, right in comparisons:
        cells = [three(paired(values(left, metric=metric), values(right, metric=metric), weights=(1,-1)), scale, True)
                 for metric, scale in metrics]
        out.append("| " + " | ".join([label, *cells]) + " |")
    policies = tuple(p(structure=s) for s in ("both", "retire", "grow", "fixed_probe"))
    interaction = [three(paired(*(values(v, metric=metric) for v in policies), weights=(1,-1,-1,1)), scale, True)
                   for metric, scale in metrics]
    out.append("| 生长效果在允许渐退时的变化¹ | " + " | ".join(interaction) + " |")
    out += ["", "¹ 描述性四组差值：(联合−只退)−(只生−同探测固定)。它量化两种策略合用时的行为差异，不定位到单条边。", "",
            "## 连接实际上发生了什么", "",
            "按训练分支列至最后合法状态的累计数。资格退出是实际退到零并收回资格；探测了减弱不等于退出。历史损伤按整批已接受生长计算。分母为零写作 0/0，表示没有这类提案可评。", "",
            "| 策略 | 种子 | 末次 tick / 状态 | 最后资格边 | 累计出生 | 实际退出 | 接受生长批中历史损失增加/接受批 |",
            "|---|---:|---|---:|---:|---:|---:|"]
    for v in reversed(policies):
        for seed in plot.SEEDS:
            b = branches[v, seed]
            s, d = b["structure_summary"], b["functional_damage_summary"]
            out.append(f"| {v[3]} | {seed} | {b['last_tick']} / {b['status']} | {s['eligible_pairs']} | "
                       f"{s['cumulative_births']} | {s['cumulative_deaths']} | {d['accepted_birth_history_worse']}/{d['accepted_birth_batches']} |")
    out += ["", "## 撤出前学会了多少，撤出后还剩多少", "",
            "只看被撤出的 family 0、长条件、严格完成。旧经历仍可被历史库抽中，其他家族的部分映射也重合。撤出前的地板分不足以证明遗忘。", "",
            "| ports / 无向 / 固定图 / 短到长 | 主训练末 % | 撤出末 % | 撤出末−主末（百分点） | 返回末 % |",
            "|---|---:|---:|---:|---:|"]
    for method in ("tbptt128","recorded512","rtrl","rtrl_reset"):
        v = p(credit=method)
        a, b, c = (values(v, t, f=0) for t in (m, w, end))
        out.append(f"| {method} | {three(a)} | {three(b)} | {three(paired(b,a,weights=(1,-1)), signed=True)} | {three(c)} |")
    out += ["", "## 解释边界", "",
            "- 统一端口与可学习接口从相同函数出生，但参数量及即时反应能力不同。ports 的互不重叠端口至少需要下一 tick 才能让新脉冲影响输出。",
            "- 当前有效线索到合同截止最多 74 tick；128 分段仍能切断更短的因果链。机械校验表明信用可超过 1,024 tick，行为数据不能单独证明模型学会了超长线索保持。",
            "- RTRL 的完整敏感度沿真实记录的参数轨迹传递；不对优化器、结构决策、离散动作或世界求导。两条独立方向的数值校验不证明所有敏感度元素。",
            "- 生长与退出允许独立接受。边数、一次候选损失或图的形状都不能证明识别了可丢弃的记忆，也不能证明出现了功能分工。",
            "", f"协议 SHA-256：`{h['protocol_sha256']}`。原始分母、每个种子的记录、全部家族和五个时间点保存在同套审计表中。"]
    return "\n".join(out) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    content = render(plot.read_audited_bundle(args.report))
    relative = Path(os.path.relpath(args.report.resolve(), args.output.parent.resolve())).as_posix()
    content += f"\n[审计总表](<{relative}>)\n"
    target = args.output.resolve()
    if target.exists():
        if target.read_text(encoding="utf-8") != content:
            raise FileExistsError("refusing to overwrite different result tables")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(target)


if __name__ == "__main__":
    main()
