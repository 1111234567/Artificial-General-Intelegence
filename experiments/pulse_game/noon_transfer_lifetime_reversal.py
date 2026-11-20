"""Replay the immediate-feedback lives after an unannounced food-law reversal."""

from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import torch

from . import interleave_revisit_train as io
from . import noon_transfer as card, noon_transfer_archive as archive, noon_transfer_custody as custody
from . import noon_transfer_lifetime as life, noon_transfer_lifetime_report as report
from .noon_transfer_report import COHORT_NAMES, table


PLAN = card.OUT / "transfer-lifetime-reversal-plan.json"
OUT = life.OUTPUT / "reversal"


def reversal(tape, plan, switch_after):
    later = torch.arange(plan["opportunities"]) >= switch_after
    earlier_objects = tape["target_ids"] < plan["recurring_objects"]
    changed = later & earlier_objects
    tape = dict(tape)
    for key in ("review_value", "final_value"):
        tape[key] = torch.where(changed, -tape[key], tape[key])
    return tape


def late(scores, pre, end):
    return {mode: {k: values[f"{end}/{k}"] - values[f"{pre}/{k}"]
                   for k in ("net", "fees", "gross", "purchases")}
            for mode, values in scores.items()}


def prefix_equal(base, treatment, end):
    steps = base["natural/buy"].shape[1]
    for key in base.files:
        if key.endswith("/initial"):
            np.testing.assert_array_equal(base[key], treatment[key])
        if "/" in key and base[key].ndim == 2 and base[key].shape[1] == steps:
            value, compare = base[key][:, :end], treatment[key][:, :end]
            if value.dtype.kind == "f":
                np.testing.assert_allclose(value, compare, rtol=0, atol=2e-6)
            else:
                np.testing.assert_array_equal(value, compare)


@torch.no_grad()
def run(cohort):
    torch.set_num_threads(1)
    protocol = archive.read(PLAN)
    plan = archive.read(life.PLAN)
    original_report = card.OUT / f"transfer-lifetime-{cohort}.json"
    frozen = archive.read(original_report)
    supplement = archive.read(card.ROOT / frozen["models_index"]["path"])
    if (frozen["protocol"]["plan"], frozen["seed"]) != (archive.digest(life.PLAN), plan["cohorts"][cohort]):
        raise ValueError("The comparison life archive came from a different plan")
    plan = deepcopy(plan)
    plan["report_through"] = protocol["measure_through"]
    pre, end = plan["report_through"]
    if pre != protocol["switch_after"]:
        raise ValueError("The switch and the end of the first measurement window must coincide")
    env = protocol["environment"]
    original = life.make_tape(plan, frozen["seed"], frozen["geometry_count"])
    flipped = reversal(original, plan, pre)
    directory = OUT / cohort
    directory.mkdir(parents=True, exist_ok=True)
    signature = json.dumps({"flip_plan": archive.digest(PLAN), "source_sha256": archive.digest(Path(__file__)),
                            "parent_life": frozen["protocol"]}, sort_keys=True)
    cells, paths = {}, {}
    for model in plan["models_in_order"]:
        cells[model] = {"reference": [], "reversal": []}
        paths[model] = {}
        for b in frozen["births"]:
            started = perf_counter()
            baseline = frozen["hashes"][env][model][str(b)]["trace"]
            path = card.ROOT / baseline["path"]
            archive.verify(path, baseline["sha256"])
            flipped_score, trace = life.assess_birth(cohort, model, b, flipped, plan, env,
                                                      supplement, signature, directory)
            with np.load(path) as reference, np.load(card.ROOT / trace["trace"]["path"]) as changed:
                prefix_equal(reference, changed, pre)
                baseline_score = life.verify_and_score(
                    reference, original, plan, env, model.removeprefix("coverage_") if model.startswith("coverage_") else None)
            cells[model]["reference"].append(late(baseline_score, pre, end))
            cells[model]["reversal"].append(late(flipped_score, pre, end))
            paths[model][str(b)] = dict(reference=baseline, reversal=trace)
            print(cohort, model, b, f"{perf_counter()-started:.2f}s", flush=True)
    bootstrap = card.stats.Crossed(len(frozen["births"]), frozen["geometry_count"], frozen["seed"] + 1559, 4000)
    summaries = {}
    bank = {}
    for model, conditions in cells.items():
        entries = {event: {mode: {k: np.stack([row[mode][k] for row in rows]) for k in rows[0][mode]}
                           for mode in life.MODES} for event, rows in conditions.items()}
        bank[model] = {}
        summaries[model] = {}
        for event, arms in entries.items():
            summaries[model][event] = {mode: {k: bootstrap(v) for k, v in measurements.items()}
                                       for mode, measurements in arms.items()}
            bank[model][f"{event}/selection"] = arms["natural"]["net"] - arms["matched_shuffle"]["net"]
            for mode, values in arms.items():
                bank[model][f"{event}/{mode}"] = values["net"]
        bank[model]["reversal/natural minus reference"] = bank[model]["reversal/natural"] - bank[model]["reference/natural"]
        bank[model]["reversal/selection minus reference"] = bank[model]["reversal/selection"] - bank[model]["reference/selection"]
        summaries[model]["comparisons"] = {k: bootstrap(v) for k, v in bank[model].items()
                                           if k.endswith("selection") or k.endswith("reference")}
    raw = directory / "paired-late-resource.npz"
    io._write_raw(raw, **{f"{model}/{k}": v for model, items in bank.items() for k, v in items.items()})
    artifact = {"cohort": cohort, "births": frozen["births"], "geometry_count": frozen["geometry_count"],
                "seed": frozen["seed"], "tracked_window": [pre + 1, end], "plan": custody.record(PLAN),
                "first_half": "Verified the first 16 opportunities against the original immediate-feedback traces.",
                "reference": custody.record(original_report), "paired_resource": custody.record(raw),
                "results": summaries, "traces": paths}
    io._write_json(card.OUT / f"transfer-lifetime-reversal-{cohort}.json", artifact)


def combine():
    files = {c: card.OUT / f"transfer-lifetime-reversal-{c}.json" for c in COHORT_NAMES}
    records = {c: archive.read(p) for c, p in files.items()}
    results = {}
    draws, arrays = {}, {}
    for c, y in records.items():
        draws[c] = card.stats.Crossed(len(y["births"]), y["geometry_count"], y["seed"] + 1559, 4000)
        raw = card.ROOT / y["paired_resource"]["path"]
        archive.verify(raw, y["paired_resource"]["sha256"])
        with np.load(raw) as z:
            arrays[c] = dict(z)
    for model in report.MODELS:
        results[model] = {}
        metrics = [k.removeprefix(model + "/") for k in arrays["original"] if k.startswith(model + "/")]
        for metric in metrics:
            values, samples = [], []
            for c, draw in draws.items():
                x = arrays[c][f"{model}/{metric}"]
                values.append(x.mean())
                samples.append(((draw.br @ x) * draw.bw).sum(1))
            results[model][metric] = {"mean": float(np.mean(values)),
                                      "ci95": np.quantile(np.mean(samples, axis=0), [.025, .975]).tolist()}
    output = card.OUT / "transfer-lifetime-reversal-pooled.json"
    io._write_json(output, {"inputs": {c: custody.record(p) for c, p in files.items()}, "results": results})
    page = ["# 六个反复对象在生命中段无提示翻转", "",
            "沿用两批新世界的相同几何、价格和身体。第16次以后，只将六个反复对象的真实好坏侧反过来。"
            "每次收获后即送实际味觉，后半净资源以第17至32次的实际收益减购票费。"
            "翻转和未翻转在前16次的选择和回执逐轨迹核对相同，包括所有严格配额的重放。", ""]
    measures = ("reference/natural", "reversal/natural", "reversal/matched_shuffle", "reversal/always_buy",
                "reversal/always_pass", "reversal/natural minus reference", "reference/selection",
                "reversal/selection", "reversal/selection minus reference")
    for name, data in (("两批合并", results), *((COHORT_NAMES[c], y["results"]) for c, y in records.items())):
        page += [f"## {name}", ""]
        rows = []
        for model, cells in data.items():
            if name != "两批合并":
                cells = {**{f"{event}/{mode}": values["net"] for event in ("reference", "reversal")
                             for mode, values in cells[event].items()}, **cells["comparisons"]}
            rows.append([report.MODELS[model], *(report.value(cells[key]) for key in measures[:5]),
                         *(report.delta(cells[key]) for key in measures[5:])])
        page += table(["冻结模型", "未翻转自然净资源", "翻转自然净资源", "翻转匹配随机", "翻转始终买",
                       "翻转始终跳过", "自然资源翻转差", "未翻转采购时机优势",
                       "翻转采购时机优势", "采购时机优势前后改变"], rows)
    page += ["[翻转方案](transfer-lifetime-reversal-plan.json) · [原出生证据](transfer-lifetime-reversal-original.json) · "
             "[独立出生证据](transfer-lifetime-reversal-independent.json) · [合并区间](transfer-lifetime-reversal-pooled.json)", ""]
    output = card.OUT / "transfer-lifetime-reversal.md"
    output.write_text("\n".join(page), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--cohort", choices=tuple(COHORT_NAMES))
    group.add_argument("--combine", action="store_true")
    args = p.parse_args()
    if args.combine:
        combine()
    else:
        run(args.cohort)
