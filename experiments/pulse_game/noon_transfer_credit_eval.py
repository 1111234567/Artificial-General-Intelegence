"""Evaluate persistent ticket credit on short and extended fresh lives."""

from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import torch

from . import interleave_revisit_train as io
from . import noon_ticket_eval as single
from . import noon_transfer as card, noon_transfer_archive as archive, noon_transfer_custody as custody
from . import noon_transfer_credit_train as train, noon_transfer_lifetime as long
from . import noon_transfer_lifetime_report as reporter
from .noon_transfer_report import COHORT_NAMES, table


MODELS = ("ticket_native", *train.ARMS)
NAMES = {"ticket_native": "原单次票据图", "current": "连续身体、当次信用", "future": "连续身体、累计信用"}
PAIRS = (("current", "ticket_native"), ("future", "ticket_native"), ("future", "current"))
BEHAVIOR = ("gross", "fees", "purchases", "price_5_purchases", "price_25_purchases", "price_55_purchases")


def train_index(cohort, arm, step, births, parent):
    directory = train.ROOT / cohort / "training" / arm
    manifest = directory / "run.json"
    optimizer = directory / f"optimizer-{step:06d}.pt"
    trace = directory / f"trace-{step:06d}.npz"
    spec = archive.read(manifest)
    archive.verify(train.PLAN, spec["plan"]["sha256"])
    if spec["births"] != births or spec["parents"] != parent:
        raise ValueError(f"Training lineage mismatch at {manifest}")
    payload = torch.load(optimizer, map_location="cpu", weights_only=True)
    if payload["manifest_sha256"] != archive.digest(manifest) or payload["step"] != step:
        raise ValueError(f"Unexpected frozen optimizer at {optimizer}")
    if {int(s["step"]) for s in payload["optimizer"]["state"].values()} != {step}:
        raise ValueError(f"Unequal optimizer step counters at {optimizer}")
    with np.load(trace) as z:
        np.testing.assert_array_equal(z["births"], births)
        np.testing.assert_array_equal(z["labels"], train.TRACE)
        np.testing.assert_array_equal(z["values"], payload["trace"].numpy())
    body, initial = {}, {}
    for i, b in enumerate(births):
        folder = train.ROOT / cohort
        baseline = card.checkpoint.load_checkpoint(card.DATA / "tickets" / cohort, "odor", "native", b, 8000)
        first = card.checkpoint.load_checkpoint(folder, "odor", arm, b, 0)
        archive.assert_same_parameters(baseline, first)
        initial[str(b)] = custody.record(card.checkpoint.checkpoint_path(folder, "odor", arm, b, 0))
        f = card.checkpoint.checkpoint_path(folder, "odor", arm, b, step)
        g, r = card.checkpoint.load_checkpoint(folder, "odor", arm, b, step)
        for key in ("weights", "bias", "present"):
            torch.testing.assert_close(getattr(g, key), payload["graph"][key][i:i+1], rtol=0, atol=0)
        for k, coeff in r.state_dict().items():
            torch.testing.assert_close(coeff, payload["rule"][k][i:i+1], rtol=0, atol=0)
        body[str(b)] = custody.record(f)
    values = payload["trace"].numpy()
    return {"manifest": custody.record(manifest), "optimizer": custody.record(optimizer), "trace": custody.record(trace),
            "updates": step, "models": body, "initial_matches_parent": initial,
            "f64_retries": int(values[:, :, -1].max(1).sum()),
            "median_unclipped_gradient_norm": float(np.median(values[:, :, 4])),
            "clipped_birth_update_fraction": float((values[:, :, 4] > train.CLIP).mean()),
            "last_128_average_net_by_birth": values[-128:, :, 1].mean(0).tolist()}


def single_score(raw):
    fields = {"natural": "utility", "matched_shuffle": "matched_random", "always_buy": "forced_buy",
              "always_pass": "forced_skip"}
    rows = {}
    for mode, key in fields.items():
        y = single.average_worlds(raw[key])[:, list(single.TRAIN_PRICES)].mean(1)
        rows[mode] = {"1/net": y}
    for key in ("gross", "buy"):
        y = single.average_worlds(raw[key])[:, list(single.TRAIN_PRICES)].mean(1)
        rows["natural"][f"1/{key}"] = y
    rows["natural"]["1/fees"] = rows["natural"]["1/gross"] - rows["natural"]["1/net"]
    rows["natural"]["1/purchases"] = rows["natural"]["1/buy"]
    for p in single.TRAIN_PRICES:
        amount = round(100 * single.PRICES[p])
        rows["natural"][f"1/price_{amount}_purchases"] = single.average_worlds(raw["buy"])[:, p]
    return rows


def long_score(raw, tape, plan):
    cells = long.verify_and_score(raw, tape, plan, "delayed", None)
    price, paid = tape["price"].numpy(), raw["natural/buy"]
    for k in plan["report_through"]:
        for fee in plan["prices_cents"]:
            assigned = (paid[:, :k] & (price[:, :k] == fee)).sum(1)
            cells["natural"][f"{k}/price_{fee}_purchases"] = assigned.reshape(-1, plan["private_histories_per_geometry"]).mean(1)
    return cells


def summarize(group, interval):
    def compare(a, b):
        return {k: interval(v - b[k]) for k, v in a.items() if k in b}

    rates = {name: {mode: {k: interval(v) for k, v in values.items()} for mode, values in arms.items()}
             for name, arms in group.items()}
    within = {name: {f"natural minus {mode}": compare(choices["natural"], choices[mode])
                     for mode in long.MODES[1:]} for name, choices in group.items()}
    between = {}
    for a, b in PAIRS:
        aa, bb = group[a], group[b]
        a_lift = {k: v - aa["matched_shuffle"][k] for k, v in aa["natural"].items() if k in aa["matched_shuffle"]}
        b_lift = {k: v - bb["matched_shuffle"][k] for k, v in bb["natural"].items() if k in bb["matched_shuffle"]}
        between[f"{a} minus {b}"] = {"natural": compare(aa["natural"], bb["natural"]), "allocation": compare(a_lift, b_lift)}
    return {"rates": rates, "within": within, "between": between}


def record_root(record, directory, name, signature, model_ref, generate):
    file = directory / name
    if not file.is_file():
        io._write_raw(file, signature=signature, checkpoint_sha256=model_ref["sha256"], **generate())
    record[name] = custody.record(file)
    return file


@torch.no_grad()
def run(cohort):
    torch.set_num_threads(1)
    plan = archive.read(train.PLAN)
    step = plan["updates"]
    base_plan = archive.read(long.PLAN)
    first = card.OUT / f"transfer-lifetime-{cohort}.json"
    prior = archive.read(first)
    births = prior["births"]
    supplement = archive.read(card.ROOT / prior["models_index"]["path"])
    parent = supplement["model_custody"]["ticket_native"]["models"]
    archive.verify(long.PLAN, prior["protocol"]["plan"])
    env = "delayed"
    n, seed = prior["geometry_count"], prior["seed"]
    tape = long.make_tape(base_plan, seed, n)
    single_seed = plan["single_eval"]["seeds"][cohort]
    n_short = plan["single_eval"]["worlds"]
    hashes = {arm: train_index(cohort, arm, step, births, parent) for arm in train.ARMS}
    inherited_specs = []
    for source in hashes.values():
        common = archive.read(card.ROOT / source["manifest"]["path"])
        common.pop("arm")
        inherited_specs.append(common)
    if inherited_specs[0] != inherited_specs[1]:
        raise ValueError(f"Training arms used different inherited protocols in {cohort}")
    directory = train.ROOT / cohort / "evaluation"
    directory.mkdir(parents=True, exist_ok=True)
    signature = json.dumps({"plan": archive.digest(train.PLAN), "evaluator": archive.digest(Path(__file__)),
                            "previous_worlds": prior["protocol"], "one_offer_evaluator": archive.digest(Path(single.__file__))},
                           sort_keys=True)
    arrays = {"long": {}, "single": {}}
    traces = {}
    for model in MODELS:
        rows = {k: [] for k in arrays}
        trace = traces[model] = {}
        for b in births:
            start = perf_counter()
            if model == "ticket_native":
                folder = card.DATA / "tickets" / cohort
                original = prior["hashes"][env][model][str(b)]["trace"]
                archive.verify(card.ROOT / original["path"], original["sha256"])
                trace[f"base-long-{b}"] = original
                with np.load(card.ROOT / original["path"]) as raw:
                    rows["long"].append(long_score(raw, tape, base_plan))
                ref = parent[str(b)]
                graph, rule = card.checkpoint.load_checkpoint(folder, "odor", "native", b, 8000)
            else:
                ref = hashes[model]["models"][str(b)]
                folder = train.ROOT / cohort
                graph, rule = card.checkpoint.load_checkpoint(folder, "odor", model, b, step)
                f = record_root(trace, directory, f"delayed-{model}-{b}.npz", signature, ref,
                                lambda: long.routes(graph, rule, tape, base_plan, env, None, None))
                with np.load(f) as raw:
                    if (str(raw["signature"]), str(raw["checkpoint_sha256"])) != (signature, ref["sha256"]):
                        raise ValueError(f"Stale long evaluation trace: {f}")
                    reporter.verify_replays(raw, tape, base_plan)
                    rows["long"].append(long_score(raw, tape, base_plan))
            archive.verify(card.ROOT / ref["path"], ref["sha256"])
            f = record_root(trace, directory, f"one-{model}-{b}.npz", signature, ref,
                            lambda: single.batch_eval(graph, rule, single_seed, n_short, False))
            with np.load(f) as raw:
                if (str(raw["signature"]), str(raw["checkpoint_sha256"])) != (signature, ref["sha256"]):
                    raise ValueError(f"Stale one-offer evaluation trace: {f}")
                rows["single"].append(single_score(raw))
            print(cohort, model, b, f"{perf_counter()-start:.2f}s", flush=True)
        for assay, seen in rows.items():
            arrays[assay][model] = {mode: {k: np.stack([row[mode][k] for row in seen]) for k in seen[0][mode]}
                                    for mode in long.MODES}
    draws = {"long": card.stats.Crossed(len(births), n, seed+992, 4000),
             "single": card.stats.Crossed(len(births), n_short, single_seed+992, 4000)}
    raw_bank = directory / "per-world.npz"
    io._write_raw(raw_bank, **{f"{assay}/{model}/{mode}/{k}": v for assay, bodies in arrays.items()
                             for model, routes in bodies.items() for mode, values in routes.items() for k, v in values.items()})
    evidence = {"cohort": cohort, "births": births, "updates": step, "plan": custody.record(train.PLAN),
                "worlds": {"single": {"seed": single_seed, "worlds": n_short}, "long": {"seed": seed, "worlds": n}},
                "original_long": custody.record(first), "source": custody.record(Path(__file__)),
                "training": hashes, "paths": traces, "arrays": custody.record(raw_bank),
                "results": {k: summarize(v, draws[k]) for k, v in arrays.items()}}
    io._write_json(card.OUT / f"transfer-continuous-credit-{cohort}.json", evidence)


def combine():
    files = {c: card.OUT / f"transfer-continuous-credit-{c}.json" for c in COHORT_NAMES}
    reports = {c: archive.read(f) for c, f in files.items()}
    steps = {y["updates"] for y in reports.values()}
    if len(steps) != 1:
        raise ValueError("Branches do not share a training budget")
    step, = steps
    assays = ("single", "long")
    bank, draws = {}, {}
    for cohort, y in reports.items():
        archive.verify(train.PLAN, y["plan"]["sha256"])
        for items in y["paths"].values():
            for record in items.values():
                archive.verify(card.ROOT / record["path"], record["sha256"])
        raw = card.ROOT / y["arrays"]["path"]
        archive.verify(raw, y["arrays"]["sha256"])
        bank[cohort] = {}
        with np.load(raw) as z:
            for key in z.files:
                assay, model, mode, metric = key.split("/", 3)
                bank[cohort].setdefault(assay, {}).setdefault(model, {}).setdefault(mode, {})[metric] = z[key]
        draws[cohort] = {}
        for assay in assays:
            world = y["worlds"][assay]
            boot = card.stats.Crossed(len(y["births"]), world["worlds"], world["seed"]+992, 4000)
            draws[cohort][assay] = boot
            if summarize(bank[cohort][assay], boot) != y["results"][assay]:
                raise ValueError(f"Continuous credit {cohort} {assay} rates did not rescore")

    def pooled(assay, sample):
        point, draws_by_cohort = [], []
        for c, value in bank.items():
            x = sample(value[assay])
            trial = draws[c][assay]
            point.append(x.mean())
            draws_by_cohort.append(((trial.br @ x) * trial.bw).sum(1))
        return {"mean": float(np.mean(point)), "ci95": np.quantile(np.mean(draws_by_cohort, axis=0), [.025, .975]).tolist()}

    by_assay = {}
    for assay in assays:
        ends = (1,) if assay == "single" else (2, 8, 32)
        models, between = {}, {}
        for model in MODELS:
            models[model] = {}
            for end in ends:
                k = f"{end}/net"
                rates = {mode: pooled(assay, lambda group: group[model][mode][k]) for mode in long.MODES}
                diffs = {mode: pooled(assay, lambda group: group[model]["natural"][k] - group[model][mode][k])
                         for mode in long.MODES[1:]}
                behavior = {v: pooled(assay, lambda g: g[model]["natural"][f"{end}/{v}"]) for v in BEHAVIOR}
                if assay == "long":
                    behavior["rescheduled_share"] = pooled(assay, lambda g: g[model]["matched_shuffle"][f"{end}/changed"])
                models[model][str(end)] = {"rates": rates, "differences": diffs, "behavior": behavior}
        for a, b in PAIRS:
            between[f"{a} minus {b}"] = {}
            for end in ends:
                k = f"{end}/net"
                between[f"{a} minus {b}"][str(end)] = {
                    "natural": pooled(assay, lambda g: g[a]["natural"][k] - g[b]["natural"][k]),
                    "allocation": pooled(assay, lambda g: (g[a]["natural"][k] - g[a]["matched_shuffle"][k]) -
                                                        (g[b]["natural"][k] - g[b]["matched_shuffle"][k])),
                    "behavior": {v: pooled(assay, lambda g: g[a]["natural"][f"{end}/{v}"] -
                                                            g[b]["natural"][f"{end}/{v}"])
                                 for v in BEHAVIOR},
                }
        by_assay[assay] = {"models": models, "between": between}
    output = card.OUT / "transfer-continuous-credit-pooled.json"
    io._write_json(output, dict(checkpoints=step, evidence={c: custody.record(f) for c, f in files.items()}, results=by_assay))

    lines = ["# 连续身体的当次信用与以后实际净资源信用", "",
             f"两批各六出生，同一起点分到两种训练，均做 {step:,} 次更新。每步真的完成八次连续购票、复习和收获。"
             "当次信用给本轮随机动作，累计信用还包括此后实际收获与费用。收获后写入按发生时点归于以后回报。"
             "两臂共享物理规则、世界、随机数、感知和局部写入权限。身体改变后，其自行选侧和真实经历可以分化。", "",
             "单次留出用三种价格与配对的已知、未见和仅见未得味觉的历史。32 次留出沿用延迟味觉长生命，"
             "按同价格及同阶段重排每条自然轨迹的购买并完整重放。区间以出生和世界分别重采样 4,000 次，两批等权。", ""]
    for assay, data in by_assay.items():
        label = "单次留出" if assay == "single" else "收获味觉整局延迟的长生命"
        lines += [f"## {label}", ""]
        rows = []
        for model, points in data["models"].items():
            for end, row in points.items():
                rates, contrasts = row["rates"], row["differences"]
                rows.append([NAMES[model], end, *(reporter.value(rates[mode]) for mode in long.MODES),
                             reporter.delta(contrasts["matched_shuffle"]), reporter.delta(contrasts["always_pass"])])
        lines += table(["冻结模型", "前几次", "自然净资源", "始终买", "始终跳过", "配额随机",
                        "自然减配额随机", "自然减始终跳过"], rows)
        rows = []
        for pair, points in data["between"].items():
            a, b = pair.split(" minus ")
            for end, row in points.items():
                rows.append([f"{NAMES[a]}减{NAMES[b]}", end, reporter.delta(row["natural"]),
                             reporter.delta(row["allocation"])])
        lines += table(["冻结模型差", "前几次", "自然净资源差", "自然减配额随机的组间差"], rows)
        rows = []
        for model, points in data["models"].items():
            for end, row in points.items():
                actual = row["behavior"]
                counts = [reporter.value(actual[k]) for k in BEHAVIOR]
                moved = actual.get("rescheduled_share")
                if moved:
                    counts.append(f"{100*moved['mean']:.1f}%")
                rows.append([NAMES[model], end, *counts])
        labels = ["冻结模型", "前几次", "自然收获", "费用", "购买量", "价格0.05购买", "价格0.25购买", "价格0.55购买"]
        if assay == "long":
            labels.append("重排实际改动比例")
            lines.extend(("各价格栏为到该时点每条生命的累计实际购买数。", ""))
        else:
            lines.extend(("各价格栏为面对该价格时的实际购买率。", ""))
        lines += table(labels, rows)
        rows = [[end, *(reporter.delta(row["behavior"][k]) for k in BEHAVIOR)]
                for end, row in data["between"]["future minus current"].items()]
        lines += table(["累计减当次信用的差", *labels[2:8]], rows)
    lines += ["[冻结方案与吞吐](transfer-continuous-credit-plan.json) · "
              "[原出生行为及训练验档](transfer-continuous-credit-original.json) · "
              "[独立出生行为及训练验档](transfer-continuous-credit-independent.json) · "
              "[两批合并区间](transfer-continuous-credit-pooled.json)", ""]
    page = card.OUT / "transfer-continuous-credit.md"
    page.write_text("\n".join(lines), encoding="utf-8")
    print(page)


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    commands = p.add_mutually_exclusive_group(required=True)
    commands.add_argument("--cohort", choices=tuple(COHORT_NAMES))
    commands.add_argument("--combine", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.combine:
        combine()
    else:
        run(args.cohort)
