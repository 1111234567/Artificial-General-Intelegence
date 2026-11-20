"""Render the compact Chinese transfer table from paired JSON score cards."""

from argparse import ArgumentParser

import numpy as np

from . import noon_transfer as card
from . import noon_transfer_archive as archive


LABELS = {
    "uniform": "普通对象专家与复习训练起点",
    "parent": "七接触交错亲本",
    "long_only": "修复后仅长题信用",
    "mixed": "修复后长短各半信用",
    "triangle_expert": "难三角专家",
    "native": "图内选择复习训练",
    "random": "外部随机选择训练",
    "uncertain": "外部亲本不确定性选择训练",
    "native_body_uniform_rule": "训练后身体＋起点写法",
    "uniform_body_native_rule": "起点身体＋训练后写法",
}
COHORT_NAMES = {"original": "原出生 731–736", "independent": "独立出生 811–816"}
ORDINARY = (
    ("A 单独", "objects/1/no_review/A"), ("B 单独", "objects/B_alone/B"),
    ("A 学过 B 后", "objects/2/no_review/A"), ("B 学过 A 后", "objects/2/no_review/B"),
    ("A、B 全对", "objects/2/no_review/all"), ("四项全对", "objects/4/no_review/all"),
    ("八项全对", "objects/8/no_review/all"),
)
JOINT = (
    ("冲突 A", "snapshot/context_conflict/context_A"), ("冲突 B", "snapshot/context_conflict/context_B"),
    ("冲突标记", "snapshot/context_conflict/mark"), ("冲突 A、B、标记全对", "snapshot/context_conflict/context_both_mark"),
    ("冲突六项全对", "snapshot/context_conflict/all_six"), ("各映射六项全对", "snapshot/all_maps/all_six"),
    ("冲突串答双收益", "serial/context_conflict/two_paid"), ("旧难三角三答全对", "triangle/hard_three_actual"),
)
CONTRAST = (ORDINARY[4], ORDINARY[5], ORDINARY[6], JOINT[3], JOINT[4], JOINT[6], JOINT[7])


def name(key):
    return LABELS.get(key, LABELS.get(key.rsplit("_", 1)[0], key))


def score(cell):
    return f"{100 * cell['mean']:.2f}%"


def difference(cell):
    mean = 100 * cell["mean"]
    lo, hi = [100 * v for v in cell["ci95"]]
    return f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def table(labels, rows):
    return ["| " + " | ".join(labels) + " |",
            "|" + "|".join("---" for _ in labels) + "|",
            *("| " + " | ".join(row) + " |" for row in rows), ""]


def pooled_contrasts(plans, stage):
    comparisons = [("mixed", "long_only")]
    if stage:
        comparisons += [(f"native_{stage}", model) for model in ("uniform", f"random_{stage}", f"uncertain_{stage}")]
    metrics = [metric for _, metric in CONTRAST]
    models = dict.fromkeys(model for pair in comparisons for model in pair)
    groups = {}
    for cohort, report in plans.items():
        n = len(report["births"])
        intervals = {assay: card.stats.Crossed(n, count, report["seeds"][assay] + 367, 4000)
                     for assay, count in report["worlds"].items()}
        arrays = {}
        for model in models:
            rows = []
            for birth in report["births"]:
                entry = report["hashes"][model][str(birth)]
                file = card.ROOT / entry["trace"]
                archive.verify(file, entry["sha256"])
                with np.load(file) as raw:
                    rows.append(card.score(raw))
            arrays[model] = {metric: np.stack([row[metric] for row in rows]) for metric in metrics}
        cells = {}
        for a, b in comparisons:
            pair = cells[a, b] = {}
            for metric in metrics:
                interval = intervals[metric.split("/", 1)[0]]
                v = arrays[a][metric] - arrays[b][metric]
                pair[metric] = (v.mean(), ((interval.br @ v) * interval.bw).sum(1))
        groups[cohort] = cells
    combined = {}
    for pair in comparisons:
        combined[pair] = {}
        for metric in metrics:
            points, draws = zip(*(group[pair][metric] for group in groups.values()), strict=True)
            combined[pair][metric] = dict(mean=np.mean(points), ci95=np.quantile(np.mean(draws, axis=0), [.025, .975]))
    return combined


def retention_table(data):
    result = ["## 相同接触与味觉下的 A 保留", "",
              "每个出生重放起点图实际做出的收集选侧。各模型得到相同物理菜单、选侧、味觉和随机数。首先学 A，"
              "随后从这次写完的状态分叉，允许后续写入、同接触禁写或等时空等候。每个测点从当时的状态独立读取 A。", ""]
    labels = ["出生", "冻结模型", "学 A 后", "两接触后", "四接触后", "八接触后", "八接触后禁写", "八接触后空等候"]
    rows = []
    for cohort, group in data["cohorts"].items():
        for model, values in group["models"].items():
            cols = ("write/1", "write/2", "write/4", "write/8", "sham/8", "idle/8")
            rows.append([COHORT_NAMES[cohort], name(model), *(score(values[k]) for k in cols)])
    result += table(labels, rows)
    rows = []
    for cohort, group in data["cohorts"].items():
        for comparison in ("native minus uniform", "native minus random", "native minus uncertain"):
            a, b = comparison.split(" minus ")
            values = group["contrasts"][comparison]
            cols = ("write/1", "write/8", "decrement/write/8", "write_minus_sham/8")
            rows.append([COHORT_NAMES[cohort], f"{name(a)}减{name(b)}", *(difference(values[k]) for k in cols)])
    result += table(["出生", "冻结模型差", "首测差", "八接触末测差", "相对各自首测的变化差",
                     "两模型各自写入减禁写的差"], rows)
    return result


def render(stage, swaps):
    plans = {}
    for cohort in COHORT_NAMES:
        suffix = "-swaps" if swaps else ""
        plans[cohort] = archive.read(card.OUT / f"transfer-{cohort}-{stage}{suffix}.json")
        if plans[cohort]["plan_sha256"] != archive.digest(card.PLAN):
            raise ValueError(f"Unmatched study plan: {cohort}")
    result = [f"# 旧任务冻结交叉评估 · {'第 ' + str(stage) + ' 步' if stage else '起点'}", "",
              "两批各六出生，在各自相同的未见世界比较。每个模型以同一具封存身体和局部写法进入所有任务。普通题、七接触快照、真实串答、难三角均允许原有的生内写入，成绩表主列无额外复习。四项及八项另列统一外置复习的因果分叉。",
              "", "普通题各 128 个共享气味世界，七接触和串答分别使用各自 96 个世界，难三角用 128 个旋转世界。"
              "普通题的联合选择来自同一末状态。冲突 A、B、标记和六项同样取同一末状态，表中三种支持顺序等权。"
              "两次串答确实依次行动，首题味觉输入为零，同时保留这次动作后的状态。旧难三角成绩取两个等值映射，"
              "六个支持顺序及两种运动摆向等权。百分点差的括号为出生与世界配对交叉重采样的 95% 区间，共 4,000 次。", ""]
    if stage:
        result.extend(("两项和四项在复习训练的辅助更新中出现，表中使用全新的普通世界。八项、七接触、真实串答和难三角另有各自的负荷或规则。"
                       "复习菜单下的图内选择另用相同模型进行对照。", ""))
    for cohort, data in plans.items():
        result += [f"## {COHORT_NAMES[cohort]}", ""]
        for name_of_test, columns in (("普通气味", ORDINARY), ("七接触、串答与旧难三角", JOINT)):
            result += [f"### {name_of_test}", ""]
            rows = [[name(model), *(score(values[k]) for _, k in columns)] for model, values in data["models"].items()]
            result += table(["冻结模型", *(label for label, _ in columns)], rows)
        result += ["### 配对差，百分点", ""]
        rows = [[f"{name(left)}减{name(right)}", *(difference(values[k]) for _, k in CONTRAST)]
                for pair, values in data["paired"].items() for left, right in (pair.split(" minus "),)]
        result += table(["冻结模型差", *(label for label, _ in CONTRAST)], rows)
        result += ["### 同模型增加一次外置均匀复习，百分点", ""]
        result += ["从相同普通历史和同一待答快照分叉。空等候与实际接触经历相同时钟，禁写得到与实际复习一致的选择和味觉。各行对所有世界计分。", ""]
        rows = []
        for model, values in data["external_opportunity"].items():
            cell = [name(model)]
            for count in (4, 8):
                for control in ("idle", "sham"):
                    cell.append(difference(values[f"{count}/external_uniform minus {control}"]["all"]))
            rows.append(cell)
        result += table(["冻结模型", "四项全对减空等候", "四项全对减禁写", "八项全对减空等候", "八项全对减禁写"], rows)
    result += ["## 两批合并的预定配对差，百分点", "",
               "分别在两批内重采样各自的出生和世界，再对两批等权取平均。", ""]
    pooled = pooled_contrasts(plans, stage)
    rows = [[f"{name(a)}减{name(b)}", *(difference(cells[k]) for _, k in CONTRAST)] for (a, b), cells in pooled.items()]
    result += table(["冻结模型差", *(label for label, _ in CONTRAST)], rows)
    diagnostic = f"transfer-retention-{stage}.json"
    diagnostic_available = (card.OUT / diagnostic).is_file()
    if diagnostic_available:
        result += retention_table(archive.read(card.OUT / diagnostic))
    result += ["## 证据", "", "[冻结方案](transfer-plan.json) · [前一轮检查点和行为复核](transfer-archived-verification.json) · "
              "[本轮起点逐张量验档](transfer-stage-0-custody.json)", ""]
    suffix = "-swaps" if swaps else ""
    files = [f"[{COHORT_NAMES[c]}完整成绩与逐出生原始轨迹索引](transfer-{c}-{stage}{suffix}.json)" for c in COHORT_NAMES]
    verify = f"transfer-verification-{stage}{suffix}.json"
    if (card.OUT / verify).is_file():
        files.append(f"[逐轨迹重算与 SHA 验证]({verify})")
    if stage:
        files.append(f"[第 {stage} 步优化器与模型验档](transfer-stage-{stage}-custody.json)")
        if diagnostic_available:
            files.append(f"[同选侧同味觉的 A 保留分叉]({diagnostic})")
            diagnostic_verification = f"transfer-retention-verification-{stage}.json"
            if (card.OUT / diagnostic_verification).is_file():
                files.append(f"[保留分叉复算]({diagnostic_verification})")
    result.extend((" · ".join(files), ""))
    result.extend(("[交叉跑分](../../../experiments/pulse_game/noon_transfer.py) · "
                   "[成绩表生成](../../../experiments/pulse_game/noon_transfer_report.py) · "
                   "[原始轨迹验算](../../../experiments/pulse_game/noon_transfer_verify.py)", ""))
    path = card.OUT / f"transfer-scorecard-{stage}{suffix}.md"
    path.write_text("\n".join(result), encoding="utf-8")
    return path


def render_tickets():
    from . import noon_transfer_ticket as ticket
    from . import noon_transfer_verify as verify

    labels = {"uniform": "普通对象起点", "ticket_native": "票据图内购买", "ticket_coin": "票据外部分配购买",
              "coverage_evidence": "同票据图继续训练共激活证据",
              "coverage_dose": "同票据图继续训练总剂量对照"}
    headline = (*ORDINARY[4:], JOINT[3], JOINT[4], JOINT[5])
    result = ["# 票据模型与付费菜单共激活模型的旧任务成绩", "",
              "两种票据模型从 uniform 起点各训练 8,000 步。共激活证据和总剂量对照都继承同一票据图内购买模型，"
              "各另训练 8,000 步。旧任务沿用标准原生身体、感觉运动接口和局部写入。付费菜单的共激活读数和增益只在那个菜单使用。"
              "本页沿用主卡相同出生、普通气味几何与七接触世界。", ""]
    bank = {}
    for cohort in COHORT_NAMES:
        path = card.OUT / f"transfer-ticket-coverage-{cohort}-8000.json"
        report = archive.read(path)
        n, seeds, counts = len(report["births"]), report["seeds"], report["worlds"]
        ordinary = card.capacity.make_tape("odor", seeds["objects"], counts["objects"], 7)
        scene = card.six.make(seeds["snapshot"], counts["snapshot"])
        menu_sources = {model: archive.read(card.OUT / f"ticket-{cohort}-8000.json")["hashes"]
                        for model in ("ticket_native", "ticket_coin")}
        optional = card.OUT / f"coverage-{cohort}-first-8000.json"
        if optional.is_file():
            hashes = archive.read(optional)["hashes"]
            menu_sources.update({"coverage_evidence": hashes, "coverage_dose": hashes})
        draws = {key: card.stats.Crossed(n, counts[key], seeds[key] + 367, 4000) for key in ("objects", "snapshot")}
        arrays = {}
        for model, traced in report["hashes"].items():
            rows = []
            for b in report["births"]:
                entry = traced[str(b)]
                raw = card.ROOT / entry["trace"]
                archive.verify(raw, entry["sha256"])
                for source in entry["sources"]:
                    checkpoint = card.ROOT / source["path"]
                    archive.verify(checkpoint, source["sha256"])
                    if model in menu_sources and menu_sources[model][checkpoint.relative_to(card.DATA).as_posix()] != source["sha256"]:
                        raise ValueError(f"Old-task and ticket-menu assays used different models: {checkpoint}")
                with np.load(raw) as saved:
                    values = {**verify.check_objects(saved, ordinary, seeds["objects"]), **verify.check_snapshot(saved, scene)}
                rows.append(values)
            arrays[model] = {k: np.stack([row[k] for row in rows]) for k in rows[0]}
        for model, matrices in arrays.items():
            for key, v in matrices.items():
                rescored = draws[key.split("/")[0]](v)
                for stat, cell in rescored.items():
                    np.testing.assert_allclose(cell, report["models"][model][key][stat], rtol=0, atol=1e-13)
        intervals = {}
        for a, b in ticket.PAIRS:
            intervals[a, b] = {}
            for key, v in arrays[a].items():
                v = v - arrays[b][key]
                interval = draws[key.split("/")[0]]
                stored = report["paired"][f"{a} minus {b}"][key]
                for stat, cell in interval(v).items():
                    np.testing.assert_allclose(cell, stored[stat], rtol=0, atol=1e-13)
                intervals[a, b][key] = (v.mean(), ((interval.br @ v) * interval.bw).sum(1))
        bank[cohort] = intervals
        result += [f"## {COHORT_NAMES[cohort]}", ""]
        for title, chosen in (("普通 A/B 与容量", ORDINARY), ("七接触的同状态终局选择", JOINT[:6])):
            result += [f"### {title}", ""]
            rows = [[labels[m], str(report["updates_after_uniform"][m]), *(score(v[k]) for _, k in chosen)]
                    for m, v in report["models"].items()]
            result += table(["冻结模型", "起点后的累计训练更新", *(label for label, _ in chosen)], rows)
        result += ["### 组内配对差，百分点", ""]
        rows = [[f"{labels[a]}减{labels[b]}", *(difference(report["paired"][f"{a} minus {b}"][k])
                                               for _, k in headline)] for a, b in ticket.PAIRS]
        result += table(["冻结模型差", *(label for label, _ in headline)], rows)
    result += ["## 两批合并配对差，百分点", ""]
    rows = []
    for a, b in ticket.PAIRS:
        row = [f"{labels[a]}减{labels[b]}"]
        for _, key in headline:
            points, draws = zip(*(cells[a, b][key] for cells in bank.values()), strict=True)
            values = {"mean": np.mean(points), "ci95": np.quantile(np.mean(draws, axis=0), [.025, .975])}
            row.append(difference(values))
        rows.append(row)
    result += table(["冻结模型差", *(label for label, _ in headline)], rows)
    retention = card.OUT / "transfer-ticket-retention-8000.json"
    if retention.is_file():
        evidence = archive.read(retention)
        result += ["## 票据训练后的固定选侧和味觉分叉", "",
                   "自然列保留普通部署中的实际收集动作，使用主卡的世界。回放列使用另一批固定几何，"
                   "每个出生先用训练前 uniform 图自然收集，再对三个冻结模型重放同样菜单、物理选侧、实际味觉和写入随机数。"
                   "八接触禁写列保留另外七次接触但禁止它们改图。", ""]
        rows = []
        for cohort, group in evidence["cohorts"].items():
            for model, fixed in group["fixed_receipts"].items():
                natural = group["natural_reference"][model]["objects/2/no_review/all"]
                keys = ("write/1", "pair/B_alone", "pair/A", "sham/2", "pair/B", "pair/both", "write/8", "sham/8")
                rows.append([COHORT_NAMES[cohort], labels[model], score(natural), *(score(fixed[k]) for k in keys)])
        result += table(["出生", "冻结模型", "自然 A/B 全对", "回放 A 单独", "回放 B 单独", "回放 A 学过 B 后",
                         "同 B 接触禁写后 A", "回放 B 学过 A 后", "回放 A/B 全对", "回放八接触后 A",
                         "八接触禁写后 A"], rows)
        rows = []
        for cohort, group in evidence["cohorts"].items():
            for comparison, values in group["paired_fixed_receipts"].items():
                a, b = comparison.split(" minus ")
                keys = ("write/1", "write/2", "decrement/write/2", "write_minus_sham/2", "sham_minus_idle/2")
                rows.append([COHORT_NAMES[cohort], f"{labels[a]}减{labels[b]}", *(difference(values[k]) for k in keys)])
        result += table(["出生", "冻结模型差，百分点", "学 A 后", "学 B 后的旧 A", "各自首测到学 B 后的变化差",
                         "两模型各自写入减禁写的差", "两模型各自禁写减空等候的差"], rows)
        result.extend(("[固定选侧与味觉分叉及逐出生原始轨迹](transfer-ticket-retention-8000.json)", ""))
    result += ["两批各六出生，配对区间从出生与共享几何交叉重采样 4,000 次。每份原始选侧、世界、模型和训练清单均已验档，成绩由行为轨迹重新计分。", "",
               "[原组完整成绩与指纹](transfer-ticket-coverage-original-8000.json) · "
               "[独立组完整成绩与指纹](transfer-ticket-coverage-independent-8000.json) · "
               "[主卡冻结方案](transfer-plan.json) · [复算入口](../../../experiments/pulse_game/noon_transfer_ticket.py)", ""]
    paid = []
    for cohort, title in COHORT_NAMES.items():
        paid.append(f"[{title}基础票据部署](ticket-{cohort}-8000.json)")
        file = f"coverage-{cohort}-first-8000.json"
        if (card.OUT / file).is_file():
            paid.append(f"[{title}追加训练付费菜单部署]({file})")
    result.extend(("付费菜单评估已核对同一批导出，组间沿用各自世界。", "", " · ".join(paid), ""))
    output = card.OUT / "transfer-scorecard-ticket-coverage.md"
    output.write_text("\n".join(result), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--swaps", action="store_true")
    parser.add_argument("--tickets", action="store_true")
    options = parser.parse_args()
    print(render_tickets() if options.tickets else render(options.stage, options.swaps))
