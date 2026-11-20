"""Verify and pool the continuous-life ticket evidence across paired births."""

from pathlib import Path

import numpy as np
import torch

from . import interleave_revisit_train as io
from . import noon_transfer as card, noon_transfer_archive as archive
from . import noon_transfer_custody as custody, noon_transfer_lifetime as lifetime
from .noon_transfer_report import COHORT_NAMES, table


MODELS = {"ticket_native": "票据图内购买", "ticket_coin": "票据外部分配训练",
          "coverage_evidence": "追加共激活证据", "coverage_dose": "追加总剂量对照"}
ENVIRONMENTS = {"immediate": "收获后立即送味觉", "delayed": "收获味觉整局延迟"}


def verify_replays(raw, tape, plan):
    native = raw["natural/buy"]
    schedules = lifetime.shuffled_schedules(native, tape, plan)
    alternatives = {"always_buy": np.ones_like(native), "always_pass": np.zeros_like(native),
                    **{f"shuffle_{i+1}": v for i, v in enumerate(schedules)}}
    events = ("review_side", "review_taste", "harvest_side", "signed_harvest", "distractor_side", "distractor_taste")
    for mode, buy in alternatives.items():
        np.testing.assert_array_equal(raw[f"{mode}/buy"], buy)
        np.testing.assert_array_equal(raw[f"{mode}/initial"], raw["natural/initial"])
        same_prefix = np.logical_and.accumulate(buy == native, axis=1)
        for name in events:
            np.testing.assert_array_equal(raw[f"{mode}/{name}"][same_prefix], raw[f"natural/{name}"][same_prefix])


def rebuild(plan, reports):
    cohorts = {}
    checks = {}
    for cohort, record in reports.items():
        path = card.ROOT / record["worlds"]["path"]
        archive.verify(path, record["worlds"]["sha256"])
        archive.verify(card.ROOT / record["models_index"]["path"], record["models_index"]["sha256"])
        if (record["protocol"]["plan"], record["seed"], record["geometry_count"]) != (
                archive.digest(lifetime.PLAN), plan["cohorts"][cohort], plan["geometries_per_birth"]):
            raise ValueError(f"Unexpected long-life result contract: {path}")
        tape = lifetime.make_tape(plan, record["seed"], record["geometry_count"])
        with np.load(path) as world:
            for name, data in tape.items():
                np.testing.assert_array_equal(data.numpy(), world[name])
        bank = {}
        verified_traces = 0
        for environment, models in record["hashes"].items():
            bank[environment] = {}
            for model, rows in models.items():
                instrument = model.removeprefix("coverage_") if model.startswith("coverage_") else None
                worlds = []
                for birth in record["births"]:
                    identity = rows[str(birth)]
                    for entry in identity.values():
                        archive.verify(card.ROOT / entry["path"], entry["sha256"])
                    raw_path = card.ROOT / identity["trace"]["path"]
                    with np.load(raw_path) as raw:
                        verify_replays(raw, tape, plan)
                        worlds.append(lifetime.verify_and_score(raw, tape, plan, environment, instrument))
                    verified_traces += 1
                bank[environment][model] = {mode: {key: np.stack([world[mode][key] for world in worlds])
                                                    for key in worlds[0][mode]} for mode in lifetime.MODES}
        expected = lifetime.intervals(bank, plan, len(record["births"]), record["seed"])
        for key, verified in zip(("rates", "paired", "cross_model"), expected, strict=True):
            if verified != record[key]:
                raise ValueError(f"Long life {key} did not rescore in {cohort}")
        cohorts[cohort] = bank
        checks[cohort] = verified_traces
    return cohorts, checks


def pool(plan, reports, cohorts):
    intervals = {cohort: card.stats.Crossed(len(value["births"]), value["geometry_count"], value["seed"] + 992, 4000)
                 for cohort, value in reports.items()}

    def interval(matrices):
        draws, point = [], []
        for cohort, v in matrices.items():
            sample = intervals[cohort]
            draws.append(((sample.br @ v) * sample.bw).sum(1))
            point.append(v.mean())
        return {"mean": float(np.mean(point)), "ci95": np.quantile(np.mean(draws, axis=0), [.025, .975]).tolist()}

    pooled = {}
    for env in plan["environments"]:
        summaries = {}
        for model in plan["models_in_order"]:
            rates = {}
            differences = {}
            for mode in lifetime.MODES:
                cells = {cohort: bank[env][model][mode] for cohort, bank in cohorts.items()}
                rates[mode] = {k: interval({c: entry[k] for c, entry in cells.items()}) for k in next(iter(cells.values()))}
                if mode != "natural":
                    differences[f"natural minus {mode}"] = {
                        k: interval({c: bank[env][model]["natural"][k] - bank[env][model][mode][k]
                                     for c, bank in cohorts.items()}) for k in next(iter(cells.values()))}
            summaries[model] = {"rates": rates, "paired": differences}
        cross = {}
        for a, b in (("ticket_native", "ticket_coin"), ("coverage_evidence", "coverage_dose"),
                     ("coverage_evidence", "ticket_native"), ("coverage_dose", "ticket_native")):
            key = f"{a} minus {b}"
            cross[key] = {}
            for end in plan["report_through"]:
                metric = f"{end}/net"
                cross[key][str(end)] = {
                    "natural": interval({c: bank[env][a]["natural"][metric] - bank[env][b]["natural"][metric]
                                         for c, bank in cohorts.items()}),
                    "allocation_advantage": interval({
                        c: (bank[env][a]["natural"][metric] - bank[env][a]["matched_shuffle"][metric]) -
                           (bank[env][b]["natural"][metric] - bank[env][b]["matched_shuffle"][metric])
                        for c, bank in cohorts.items()}),
                }
        pooled[env] = {"models": summaries, "cross_model": cross}
    return pooled


def value(x):
    return f"{x['mean']:.3f}"


def delta(x):
    lo, hi = x["ci95"]
    return f"{x['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def rows_for(end, model, rates, comparison):
    def quantity(mode, key):
        return rates[mode][f"{end}/{key}"]

    return [MODELS[model], str(end), value(quantity("natural", "net")), value(quantity("matched_shuffle", "net")),
            delta(comparison["natural minus matched_shuffle"][f"{end}/net"]), value(quantity("always_buy", "net")),
            value(quantity("always_pass", "net")), value(quantity("natural", "gross")),
            value(quantity("natural", "fees")), value(quantity("natural", "purchases")),
            f"{100 * quantity('matched_shuffle', 'changed')['mean']:.1f}%"]


def render(plan, reports, pooled):
    page = ["# 同一身体连续 32 次购票和收获", "",
            "每条生命先遇到两个有味觉的普通对象。随后四个八次循环各呈现六个反复对象和两个新对象，期间另有八次有味觉的普通干扰接触。"
            "每个收获动作由当时图发出，有利侧记 1，其余记 0，减去实际付出的票价。起始和干扰接触供给经验。"
            "每次挑票、接触及收获后都继续使用活着的图和局部写入。", "",
            "每条自然生命的购买量按相同价格和相同阶段固定，四次随机重排后都完整重放身体、味觉、后续选择和写入。"
            "因此在前 2、8、32 次，匹配随机的实际购买数与总费用逐条相同。两批分别各六出生和 64 个新几何，每个几何四种配对私有赋值。"
            "配对区间来自 4,000 次出生和几何的交叉重采样，两批分别抽样后等权合并。表中的资源与费用为每条生命的累计单位。", "",
            "票据两种训练各从同一普通起点走 8,000 步。共激活证据和总剂量组从票据图内购买模型再走各 8,000 步，并共享付费菜单外的原生身体。"
            "它们的六数在整个生命仅随真实非零食物味觉累积。延迟环境在第 32 次之后依序送达实际收获的味觉。", ""]
    headings = ["冻结模型", "前几次", "自然净资源", "匹配随机净资源", "自然减匹配随机", "始终买净资源", "始终跳过净资源",
                "自然获益数", "自然费用", "自然购买数", "随机实际改动比例"]
    for env, title in ENVIRONMENTS.items():
        page += [f"## {title}", ""]
        rows = [rows_for(end, name, item["rates"], item["paired"]) for name, item in pooled[env]["models"].items()
                for end in plan["report_through"]]
        page += table(headings, rows)
        comparison_rows = []
        for comparison, values in pooled[env]["cross_model"].items():
            a, b = comparison.split(" minus ")
            comparison_rows.append([f"{MODELS[a]}减{MODELS[b]}",
                                    *(delta(values[str(end)]["natural"]) for end in plan["report_through"]),
                                    *(delta(values[str(end)]["allocation_advantage"]) for end in plan["report_through"])])
        page += table(["冻结模型差", "前2次自然资源差", "前8次自然资源差", "前32次自然资源差", "前2次采购安排收益差",
                       "前8次采购安排收益差", "前32次采购安排收益差"], comparison_rows)
        for cohort, report in reports.items():
            page += [f"### {COHORT_NAMES[cohort]}", ""]
            rows = [rows_for(end, name, rates, report["paired"][env][name]) for name, rates in report["rates"][env].items()
                    for end in plan["report_through"]]
            page += table(headings, rows)
    page += ["[冻结方案](transfer-lifetime-plan.json) · [原出生完整结果与轨迹索引](transfer-lifetime-original.json) · "
             "[独立出生完整结果与轨迹索引](transfer-lifetime-independent.json) · [两批合并与验档](transfer-lifetime-pooled.json) · "
             "[具身回放实现](../../../experiments/pulse_game/noon_transfer_lifetime.py)", ""]
    result = card.OUT / "transfer-lifetime.md"
    result.write_text("\n".join(page), encoding="utf-8")
    return result


def main():
    torch.set_num_threads(1)
    plan = archive.read(lifetime.PLAN)
    files = {cohort: card.OUT / f"transfer-lifetime-{cohort}.json" for cohort in plan["cohorts"]}
    reports = {cohort: archive.read(path) for cohort, path in files.items()}
    bank, traces = rebuild(plan, reports)
    results = pool(plan, reports, bank)
    artifact = {"cohort_reports": {c: custody.record(p) for c, p in files.items()},
                "source": custody.record(Path(__file__)), "verified_full_life_traces": traces,
                "results": results}
    io._write_json(card.OUT / "transfer-lifetime-pooled.json", artifact)
    print(render(plan, reports, results))


if __name__ == "__main__":
    main()
