"""Paired, seed-aware aggregation of the complete single- and two-compartment audits."""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from ..spike_analysis import estimates


AT_TIMES = ("short", "interfere", "long", "serial", "serial_noise", "extrapolate")
DN_TIMES = ("short", "long", "serial", "serial_noise", "extrapolate")
SEARCH_TASKS = ("mixed", "serial_noise")
DN_MODES = ("basal", "add", "gain", "both")
WAIT = {"short": "同步 8–12 拍", "interfere": "同步干扰 12–20 拍", "long": "同步干扰 26–42 拍",
        "serial": "串行 8–15 拍", "serial_noise": "串行干扰 12–20 拍", "extrapolate": "同步干扰 48–68 拍"}


def json_file(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_panels(root):
    attractor, dendrite = {}, {}
    for f in sorted((root / "attractor-overnight-external").glob("*-full.json")):
        x = json_file(f)
        m = re.fullmatch(r"attractor-overnight-(mixed|serial_noise)-(cut|full)-(\d+)", x["training"])
        if not m:
            raise ValueError(f"Unexpected training key in {f}")
        task, design, seed = m.groups()
        index = task, design, int(seed), x["birth"], x["condition"]
        if index in attractor:
            raise ValueError(f"Repeated attractor trial {index}")
        attractor[index] = x
    for f in sorted((root / "dendrite-v1-external").glob("dendrite-v1-*.json")):
        x = json_file(f)
        index = x["mode"], x["search_seed"], x["birth"], x["condition"]
        if index in dendrite:
            raise ValueError(f"Repeated dendrite trial {index}")
        dendrite[index] = x

    def seed_coverage(records, size):
        first = list(records.values())[0]["seeds"]
        if len(records) != size or len(first) != 128 or len(set(first)) != 128:
            raise ValueError(f"Expected {size} complete sets of 128 held worlds")
        if not all(r["seeds"] == first for r in records.values()):
            raise ValueError("The held-world order drifted between cells")
        return first

    a_worlds = seed_coverage(attractor, 288)
    d_worlds = seed_coverage(dendrite, 240)
    at_seeds = tuple(sorted({k[2] for k in attractor}))
    at_graphs = tuple(sorted({k[3] for k in attractor}))
    dn_seeds = tuple(sorted({k[1] for k in dendrite}))
    dn_graphs = tuple(sorted({k[2] for k in dendrite}))
    expected_at = {(t, g, s, b, d) for t in SEARCH_TASKS for g in ("cut", "full") for s in at_seeds
                   for b in at_graphs for d in AT_TIMES}
    expected_dn = {(m, s, b, d) for m in DN_MODES for s in dn_seeds for b in dn_graphs for d in DN_TIMES}
    if set(attractor) != expected_at or set(dendrite) != expected_dn:
        raise ValueError("Incomplete factorial panel")
    return attractor, dendrite, (at_seeds, at_graphs, a_worlds), (dn_seeds, dn_graphs, d_worlds)


def tensor(trials, cells, path):
    block = []
    for row in cells:
        values = []
        for key in row:
            selection = trials[key]
            for i in path:
                selection = selection[i]
            values.append(selection)
        block.append(values)
    out = np.asarray(block, dtype=np.float64)
    if not np.isfinite(out).all() or out.ndim not in (2, 3):
        raise ValueError(f"Invalid array at {path}")
    return out


def matched_stat(data, seeds, graphs):
    if data.shape[:2] != (len(seeds), len(graphs)):
        raise ValueError("Crossed model/graph axes differ")
    integrated = data.mean(axis=-1) if data.ndim == 3 else data
    report = estimates(data.mean(axis=(0, 1))) if data.ndim == 3 else {"mean": float(data.mean())}
    return {**report,
            "by_search_seed": {str(s): float(value) for s, value in zip(seeds, integrated.mean(1), strict=True)},
            "by_birth": {str(b): float(value) for b, value in zip(graphs, integrated.mean(0), strict=True)},
            "per_seed_and_birth": integrated.tolist()}


def cells_at(task, design, condition, seeds, graphs):
    return [[(task, design, seed, birth, condition) for birth in graphs] for seed in seeds]


def cells_dn(mode, condition, seeds, graphs):
    return [[(mode, seed, birth, condition) for birth in graphs] for seed in seeds]


def summarize_at(records, seeds, graphs):
    report = {}
    for task in SEARCH_TASKS:
        waits = {}
        for condition in AT_TIMES:
            full = cells_at(task, "full", condition, seeds, graphs)
            cut = cells_at(task, "cut", condition, seeds, graphs)

            def get(group, *path):
                return tensor(records, group, ("details", *path))

            def stats(a):
                return matched_stat(a, seeds, graphs)

            intact = get(full, "rows", "native", "late")
            after_cut = get(full, "rows", "cut", "late")
            independent = get(cut, "rows", "native", "late")
            rng_control = get(full, "rows", "no_outcome", "late")
            corrected = {
                "native_intact": stats(intact), "trained_without_recurrence": stats(independent),
                "original_genome_with_test_recurrence_cut": stats(after_cut),
                "native_minus_trained_without": stats(intact - independent),
                "native_minus_test_cut": stats(intact - after_cut),
                "recovery_training_without_vs_test_cut": stats(independent - after_cut),
                "native_minus_no_outcome": stats(intact - rng_control),
                "physical_spike_fraction_intact": stats(get(full, "activity")),
                "physical_spike_fraction_cut_trained": stats(get(cut, "activity")),
            }
            for label, group in (("intact", full), ("trained_without", cut)):
                corrected[label + "_linear_public_probe"] = {key: stats(get(group, "linear_probe", key))
                                                             for key in ("context", "cue", "joint")}
                corrected[label + "_final_context_fork"] = {k: stats(get(group, "last_context_fork", k))
                                                            for k in ("switch", "both", "alignment")}
                corrected[label + "_responses"] = {agg: {key: stats(get(group, "rows", agg, key)) for key in ("early", "late", "all", "both")}
                                                     for agg in records[group[0][0]]["details"]["rows"]}
            waits[condition] = corrected
        report[task] = waits
    return report


def summarize_dn(records, seeds, graphs):
    report = {}
    for condition in DN_TIMES:
        controls = cells_dn("basal", condition, seeds, graphs)
        baseline = tensor(records, controls, ("rows", "native", "late"))
        modes = {}

        def stat(data):
            return matched_stat(data, seeds, graphs)

        for mode in DN_MODES:
            keys = cells_dn(mode, condition, seeds, graphs)

            def x(*path):
                return tensor(records, keys, ("rows", *path))

            result = x("native", "late")
            bits = {"native": stat(result), "native_minus_separately_trained_basal": stat(result - baseline),
                    "native_minus_test_cut": stat(result - x("cut", "late")),
                    "native_minus_no_outcome": stat(result - x("no_outcome", "late")),
                    "responses": {label: {key: stat(x(label, key)) for key in ("early", "late", "all", "both", "firing")}
                                  for label in records[keys[0][0]]["rows"]}}
            for name in ("remove_dendrite", "add_only", "gain_only", "rewired"):
                if name in bits["responses"]:
                    bits[f"native_minus_{name}"] = stat(result - x(name, "late"))
            modes[mode] = bits
        candidates = {m: tensor(records, cells_dn(m, condition, seeds, graphs), ("rows", "native", "late"))
                      for m in DN_MODES}
        modes["independent_mode_contrasts"] = {f"{a}_minus_{b}": stat(candidates[a] - candidates[b])
                                              for a, b in (("gain", "add"), ("both", "gain"), ("both", "add"))}
        report[condition] = modes
    return report


def train_trace(records, datasets):
    """Check the training count, pairing and exact holdout separation of saved runs."""
    out = {}
    for family, keyset, worlds, checkpoint in datasets:
        lines = {}
        for name in keyset:
            base = records / name / checkpoint
            item = json_file(base)
            history = item["history"]
            schedule = [(row.get("net", row.get("graph")), tuple(row["worlds"])) for row in history]
            items = [seed for _, line in schedule for seed in line]
            lines[name] = {"population": item["protocol"]["population"], "generations": len(history),
                           "evaluated_candidates": item["protocol"]["population"] * len(history),
                           "unique_training_worlds": len(set(items)), "repeated_training_world_draws": len(items)-len(set(items)),
                           "intersection_held_worlds": len(set(items).intersection(worlds)),
                           "schedule": schedule}
        if family == "attractor":
            matched_groups = [sorted(name for name in lines if f"-{t}-" in name and name.endswith(f"-{s}"))
                              for t in SEARCH_TASKS for s in sorted({int(n.rsplit("-", 1)[-1]) for n in lines})]
        else:
            matched_groups = [sorted(name for name in lines if name.endswith(f"-{s}"))
                              for s in sorted({int(n.rsplit("-", 1)[-1]) for n in lines})]
        aligned = [{"reference": names[0], "compared": names[1:], "matches": {
            name: sum(a == b for a, b in zip(lines[names[0]]["schedule"], lines[name]["schedule"], strict=True))
            for name in names[1:]}} for names in matched_groups]
        for v in lines.values():
            del v["schedule"]
        if family == "attractor":
            validation = np.random.default_rng(982376).integers(100_000, 8_000_000, 96)
        else:
            validation = np.random.default_rng(437091).integers(10_000_000, 470_000_000, 96)
        out[family] = {"runs": lines, "compared_training_schedules": aligned,
                       "development_validation_intersection_held_worlds": len(set(validation).intersection(worlds)),
                       "total_evaluated_candidates": sum(v["evaluated_candidates"] for v in lines.values())}
    return out


def percent(x):
    return f"{100*x['mean']:.1f}"


def change(x):
    lo, hi = x["ci95"]
    return f"{100*x['mean']:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]"


def in_order(values):
    return " / ".join(f"{100*x:+.1f}" for x in values.values())


def one_page(data):
    a, d, provenance = data["attractor"], data["dendrite"], data["scope"]
    rows = ["# 时延发育与第二隔室的独立外测", "",
            f"时延搜索共 {provenance['training']['attractor']['total_evaluated_candidates']:,} 个候选，第二隔室共 "
            f"{provenance['training']['dendrite']['total_evaluated_candidates']:,} 个候选。两套外测各 128 世界，各条件共用对应的三次搜索种子、四个出生图。"
            "下文 n=128，区间先在世界内平均搜索和出生重复，再按世界配对自举。区间描述这些固定策略的世界变异，跨搜索与出生的方向另列。两套外测世界种子互斥。", "",
            f"时延搜索种子 {', '.join(map(str, provenance['attractor']['evolution_seeds']))}，出生 {', '.join(map(str, provenance['attractor']['births']))}。"
            f"第二隔室搜索种子 {', '.join(map(str, provenance['dendrite']['evolution_seeds']))}，出生 {', '.join(map(str, provenance['dendrite']['births']))}。", "",
            "## 递归的训练对照与训后干预", "",
            "数值是后 32 题正确率（%）。每题作一次符号选择，机会水平 25%。每格为正常训练和执行／独立去环训练和执行／正常训练、外测时去环。串行条件标最后一拍线索后的等待，世界先公示 A/B，过 4–12 拍才公示符号。", "",
            "| 等待条件 | 混合训练 | 串行带干扰训练 |", "|---|---:|---:|"]
    for name in AT_TIMES:
        items = [" / ".join(percent(a[task][name][k]) for k in ("native_intact", "trained_without_recurrence",
                                                                  "original_genome_with_test_recurrence_cut")) for task in SEARCH_TASKS]
        rows.append(f"| {WAIT[name]} | {items[0]} | {items[1]} |")
    mix, serial = a["mixed"]["long"], a["serial_noise"]["serial_noise"]
    raw = mix["intact_linear_public_probe"]
    zero = mix["intact_responses"]["no_outcome"]["late"]
    serial_dec = serial["intact_linear_public_probe"]["joint"]
    seq = serial["intact_final_context_fork"]["alignment"]
    rows.extend(["", f"混合训练的同步长等待，正常比独立去环训高 **{change(mix['native_minus_trained_without'])} pp**，比同模型训后去环高 **{change(mix['native_minus_test_cut'])} pp**。"
                 f"后一差值在三次搜索分别为 {in_order(mix['native_minus_test_cut']['by_search_seed'])} pp，四图为 {in_order(mix['native_minus_test_cut']['by_birth'])} pp。"
                 f"串行带干扰训练回到串行带干扰任务时，两项差值为 {change(serial['native_minus_trained_without'])} 和 {change(serial['native_minus_test_cut'])} pp。", "",
                 f"同步长等待的冻结编码线性探针读出背景、符号和联合键分别为 {percent(raw['context'])}%、{percent(raw['cue'])}% 和 {percent(raw['joint'])}%，"
                 f"在线关联正常为 {percent(mix['native_intact'])}%，关闭一生的回执写入为 {percent(zero)}%。串行干扰的联合键线性读出为 {percent(serial_dec)}%，"
                 f"最终同前缀换背景的对齐优势为 {percent(seq)} 个百分点。线性探针只作诊断。", "",
                 "## 第二隔室", "",
                 "表中各模式独立搜索，附加电位可加到膜电位、乘上当下感觉输入，或同时生效。下文训后消融固定入世参数，各臂仍从出生重新在线学习自己的关联权重。", "",
                 "| 等待条件 | 基础 | 加性 | 增益 | 两者 |", "|---|---:|---:|---:|---:|"])
    for name in DN_TIMES:
        row = " | ".join(percent(d[name][m]["native"]) for m in DN_MODES)
        rows.append(f"| {WAIT[name]} | {row} |")
    best = d["serial_noise"]["both"]
    only_gain = d["serial_noise"]["independent_mode_contrasts"]["both_minus_gain"]
    rows.extend(["", f"串行干扰上，两者相对独立基础为 **{change(best['native_minus_separately_trained_basal'])} pp**，三次搜索 {in_order(best['native_minus_separately_trained_basal']['by_search_seed'])} pp，"
                 f"四图 {in_order(best['native_minus_separately_trained_basal']['by_birth'])} pp。同“两者”模型关第二隔室、切递归、关回执后，与原模型差分别为 {change(best['native_minus_remove_dendrite'])}、"
                 f"{change(best['native_minus_test_cut'])}、{change(best['native_minus_no_outcome'])} pp。只留加性和只留增益的训后结果各为 {percent(best['responses']['add_only']['late'])}% 和 {percent(best['responses']['gain_only']['late'])}%。"
                 f"独立训练中的“两者减增益”为 {change(only_gain)} pp。", "",
                 "同一搜索种子下，训练阶段各模式的逐代世界题本和出生图逐项相同。两套外测种子分别与本套训练、开发验证互斥。", "",
                 "[逐条件配对数据与全部种子/四图](attractor-dendrite-summary.json) · [短暂递归路径与慢均值的定位](attractor-motif-mechanism.md)"])
    return "\n".join(rows) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    root = args.directory
    a, d, acr, dcr = read_panels(root)
    training = train_trace(root.parent, [("attractor", sorted({x["training"] for x in a.values()}), acr[2], "search.json"),
                                        ("dendrite", sorted({x["study"] for x in d.values()}), dcr[2], "checkpoint.json")])
    summary = {"scope": {"attractor": {"evolution_seeds": acr[0], "births": acr[1], "world_seeds": acr[2]},
                         "dendrite": {"evolution_seeds": dcr[0], "births": dcr[1], "world_seeds": dcr[2]},
                         "held_world_intersection": len(set(acr[2]).intersection(dcr[2])), "training": training,
                         "attractor_audit_sha256": sorted({row["audit_sha256"] for row in a.values()}),
                         "dendrite_rule_source_sha256": sorted({row["rule_source"] for row in d.values()}),
                         "dendrite_cpu_parity": json_file(root / "dendrite-v1-external" / "cpu-parity.json")},
               "attractor": summarize_at(a, *acr[:2]), "dendrite": summarize_dn(d, *dcr[:2])}
    (root / "attractor-dendrite-summary.json").write_text(json.dumps(summary, separators=(",", ":")) + "\n", encoding="utf-8")
    (root / "attractor-dendrite-summary.md").write_text(one_page(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
