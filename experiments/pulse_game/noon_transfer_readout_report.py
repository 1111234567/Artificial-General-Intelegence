"""Verify and summarize the resource-readout lifetime deployments."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from . import interleave_revisit_train as io
from . import noon_transfer as card, noon_transfer_archive as archive, noon_transfer_custody as custody
from . import noon_transfer_readout as deployment
from .noon_transfer_lifetime_report import delta, value
from .noon_transfer_report import COHORT_NAMES, table


LABELS = {"bare_native": "原图购票", "price": "价格读出", "dose": "价格和总量读出",
          "directional": "价格、总量和方向读出"}
ENVIRONMENTS = {"delayed": "收获味觉整局延迟", "immediate": "收获后立即送味觉"}
PAIRS = (("directional", "price"), ("directional", "dose"), ("dose", "price"),
         ("directional", "bare_native"), ("price", "bare_native"))
MODES = ("natural", "matched_shuffle", "always_pass", "always_buy")


def same_body(raw_index):
    events = ("buy", "fee", "ticket_side", "review_side", "review_taste", "harvest_side", "signed_harvest",
              "delivered_harvest", "distractor_side", "distractor_taste", "food_count_before")
    for environment in raw_index.values():
        for birth, reference in environment["bare_native"].items():
            baseline = card.ROOT / reference["trace"]["path"]
            with np.load(baseline) as prior:
                neutral = {f"{mode}/{key}": prior[f"{mode}/{key}"] for mode in ("always_buy", "always_pass")
                           for key in (*events, "initial", "final_coverage_trace")}
                natural = {k: prior[f"natural/{k}"] for k in events}
            for arm in ("price", "dose", "directional"):
                target = environment[arm][birth]
                if (target["checkpoint"]["sha256"] != reference["checkpoint"]["sha256"]):
                    raise ValueError(f"Readout {arm} used a different physical body at birth {birth}")
                with np.load(card.ROOT / target["trace"]["path"]) as raw:
                    for key, value in neutral.items():
                        np.testing.assert_array_equal(value, raw[key])
                    prefix = np.logical_and.accumulate(natural["buy"] == raw["natural/buy"], axis=1)
                    for key, value in natural.items():
                        np.testing.assert_array_equal(value[prefix], raw[f"natural/{key}"][prefix])


def restore(cohort, seed):
    name = card.OUT / f"transfer-resource-readout-{seed}-{cohort}.json"
    card_report = archive.read(name)
    archive.verify(deployment.PLAN, card_report["protocol"]["sha256"])
    for model in card_report["coefficients"].values():
        for key in ("params", "manifest"):
            entry = model[key]
            archive.verify(card.ROOT / entry["path"], entry["sha256"])
    for environment in card_report["raw"].values():
        for model in environment.values():
            for birth in model.values():
                for file in birth.values():
                    archive.verify(card.ROOT / file["path"], file["sha256"])
    same_body(card_report["raw"])
    groups = {}
    path = card.ROOT / card_report["per_world"]["path"]
    archive.verify(path, card_report["per_world"]["sha256"])
    with np.load(path) as z:
        for key in z.files:
            e, a, m, metric = key.split("/", 3)
            groups.setdefault(e, {}).setdefault(a, {}).setdefault(m, {})[metric] = z[key]
    draws = card.stats.Crossed(len(card_report["births"]), card_report["worlds"], card_report["root_seed"]+992, 4000)
    for env, values in groups.items():
        if deployment.summarize(values, draws) != card_report["results"][env]:
            raise ValueError(f"Results did not reproduce for {seed} {cohort} {env}")
    archive.verify(card.ROOT / card_report["previous_baseline"]["path"], card_report["previous_baseline"]["sha256"])
    return card_report, groups, draws, custody.record(name)


def pooled(groups, draws):
    def estimate(function):
        original, replica = [], []
        for cohort, bank in groups.items():
            v, boot = function(bank), draws[cohort]
            original.append(v.mean())
            replica.append(((boot.br @ v) * boot.bw).sum(1))
        interval = np.quantile(np.mean(replica, axis=0), [.025, .975])
        return {"mean": float(np.mean(original)), "ci95": interval.tolist()}

    results = {}
    for env in groups["original"]:
        models, contrasts = {}, {}
        for arm in LABELS:
            cells = {}
            for k in ("2", "8", "32"):
                key = f"{k}/net"
                rates = {mode: estimate(lambda bank: bank[env][arm][mode][key]) for mode in MODES}
                differences = {mode: estimate(lambda bank: bank[env][arm]["natural"][key] - bank[env][arm][mode][key])
                               for mode in MODES[1:]}
                ledger = {label: estimate(lambda bank: bank[env][arm]["natural"][f"{k}/{label}"])
                          for label in ("gross", "fees", "purchases")}
                ledger["moved_by_shuffle"] = estimate(lambda bank: bank[env][arm]["matched_shuffle"][f"{k}/changed"])
                cells[k] = {"rates": rates, "differences": differences, "ledger": ledger}
            models[arm] = cells
        for a, b in PAIRS:
            phases = {}
            for k in ("2", "8", "32"):
                measure = f"{k}/net"
                phases[k] = {
                    "natural": estimate(lambda data: data[env][a]["natural"][measure] - data[env][b]["natural"][measure]),
                    "allocation": estimate(lambda data: (data[env][a]["natural"][measure] -
                                                         data[env][a]["matched_shuffle"][measure]) -
                                                        (data[env][b]["natural"][measure] -
                                                         data[env][b]["matched_shuffle"][measure])),
                }
            contrasts[f"{a} minus {b}"] = phases
        results[env] = {"models": models, "between": contrasts}
    return results


def run(seeds, closed_after_primary=False):
    torch.set_num_threads(1)
    plan = archive.read(deployment.PLAN)
    primary = plan["primary_training_seed"]
    results = {}
    versions = {}
    for seed in seeds:
        reports, groups, draws, paths = {}, {}, {}, {}
        for cohort in COHORT_NAMES:
            reports[cohort], groups[cohort], draws[cohort], paths[cohort] = restore(cohort, seed)
        results[str(seed)] = {"results": pooled(groups, draws), "reports": paths,
                              "signals": {c: y["signals"] for c, y in reports.items()}}
        versions[str(seed)] = {c: y["coefficients"] for c, y in reports.items()}
    filename = card.OUT / "transfer-resource-readout-pooled.json"
    io._write_json(filename, {"plan": custody.record(deployment.PLAN), "coefficients": versions,
                              "closed_after_primary": closed_after_primary, "seeds": results})

    rows = ["# 同一票据身体上的共享资源读出", "",
            "两批各六出生，沿用 64 个新几何与四种私有历史。所有读出用同一批封存的原票据身体和局部写入。"
            "训练只调整四个共享读出系数，所有系数部署到两批时保持封存。程序在付费门从已发布价格、预览气味和已实际收到的食物味觉痕迹采样一次。"
            "实际扣费、正常收集、收获与局部写入继续相同的物理规则。", "",
            "每种读出各自按同价、同阶段及同一条生命的购买数量重排四次，随后完整回放身体。"
            "表中取每生实际收获减实际费用，阶段通过第 2、8、32 次。两批先独立重采样各自出生和几何，再等权合并。", ""]
    if str(primary) in results:
        rows += ["## 首种子两批合并结论", ""]
        summary = []
        first = results[str(primary)]["results"]
        for env, output in first.items():
            selected = output["models"]["directional"]["32"]
            cost = output["models"]["price"]["32"]["ledger"]
            trace = output["models"]["dose"]["32"]["ledger"]
            summary.append([
                ENVIRONMENTS[env], value(selected["rates"]["natural"]), value(selected["ledger"]["purchases"]),
                f"{value(cost['purchases'])} / {value(trace['purchases'])}",
                delta(output["between"]["directional minus price"]["32"]["natural"]),
                delta(output["between"]["directional minus dose"]["32"]["natural"]),
                delta(selected["differences"]["matched_shuffle"]), delta(selected["differences"]["always_pass"]),
            ])
        rows += table(["反馈", "方向自然净资源", "方向购票数", "价格 / 总量购票数", "方向减价格", "方向减总量",
                       "方向减同额重排", "方向减始终跳过"], summary)
        if closed_after_primary:
            rows.extend(("按结束要求，追加训练种子的部署队列已取消。首种子的两个反馈环境与两批固定出生均已完成。", ""))
    headings = ["模型", "前几次", "自然净资源", "匹配随机净资源", "自然减匹配随机", "始终跳过净资源",
                "始终买净资源", "自然收获", "自然费用", "自然购买数", "重排改动"]
    for seed, contents in results.items():
        rows += [f"## 训练种子 {seed}", ""]
        for env, env_result in contents["results"].items():
            rows += [f"### {ENVIRONMENTS[env]}", ""]
            entries = []
            for model, group in env_result["models"].items():
                for end, y in group.items():
                    q, d, g = y["rates"], y["differences"], y["ledger"]
                    entries.append([LABELS[model], end, value(q["natural"]), value(q["matched_shuffle"]),
                                    delta(d["matched_shuffle"]), value(q["always_pass"]), value(q["always_buy"]),
                                    *(value(g[k]) for k in ("gross", "fees", "purchases")),
                                    f"{100*g['moved_by_shuffle']['mean']:.1f}%"])
            rows += table(headings, entries)
            comparisons = []
            for pair, group in env_result["between"].items():
                a, b = pair.split(" minus ")
                comparisons.append([f"{LABELS[a]}减{LABELS[b]}", *(delta(v["natural"]) for v in group.values()),
                                    *(delta(v["allocation"]) for v in group.values())])
            rows += table(["模型差", "前2次自然资源差", "前8次自然资源差", "前32次自然资源差", "前2次择时差",
                           "前8次择时差", "前32次择时差"], comparisons)
        if int(seed) == primary:
            rows += ["### 付费门上的痕迹分布", "",
                     "方向分量 s 为当前预览与累计矩阵的二次投影。总量为三条对角线的平均。"
                     "单票训练中，预览前最多有8次食物接触，可产生不超过 8/3 的总量。", ""]
            entries = []
            for cohort, conditions in contents["signals"].items():
                for env, models in conditions.items():
                    for arm, group in models.items():
                        for end in ("2", "8", "32"):
                            f = group[end]
                            s, d = f["signal"], f["dose"]
                            entries.append([COHORT_NAMES[cohort], ENVIRONMENTS[env], LABELS[arm], end,
                                            f"{s['median']:.2f} / {s['p95']:.2f}",
                                            f"{d['median']:.2f} / {d['p95']:.2f}",
                                            f"{100*f['fraction_above_8_food_tastes']:.1f}%",
                                            f"{100*f['prob_below_0_01_or_above_0_99']:.1f}%"])
            rows += table(["出生组", "反馈", "模型", "第几次付费门", "s 中位 / P95", "总量中位 / P95",
                           "总量超过 8/3", "概率低于 0.01 或高于 0.99"], entries)
    evidence = ["[部署方案](transfer-resource-readout-plan.json)", "[两批合并与原始证据索引](transfer-resource-readout-pooled.json)",
                "[同身体原票据长生命](transfer-lifetime.md)"]
    for seed in seeds:
        evidence.extend(f"[{seed} {COHORT_NAMES[c]}](transfer-resource-readout-{seed}-{c}.json)" for c in COHORT_NAMES)
    rows.extend((" · ".join(evidence), ""))
    page = card.OUT / "transfer-resource-readout.md"
    page.write_text("\n".join(rows), encoding="utf-8")
    print(page)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1177001])
    parser.add_argument("--closed-after-primary", action="store_true")
    args = parser.parse_args()
    run(args.seeds, args.closed_after_primary)
