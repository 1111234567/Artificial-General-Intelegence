"""Verify and summarize complete, paired three-seed consolidation runs.

Usage (each input is a run.py output root, not one method's subdirectory)::

    python experiments/consolidation/summarize.py --inputs RUN0 RUN1 RUN2 --output REPORT

No training or checkpoint selection occurs here.  summary.json embeds every
result and its manifests/verification records, so the figures can be redrawn
without the original run directories or model checkpoints.  Error bars and
bands are sample standard deviations across seeds, not confidence intervals.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


METHODS = ("online", "repeat", "replay")
ORDERS = ("blocked", "recurrent")
CONDITIONS = ("A0", "A1", "B0", "C0")
METRICS = (*CONDITIONS, "current_mean", "all4_mean")
COLORS = {"online": "#4c78a8", "repeat": "#f28e2b", "replay": "#59a14f"}
LABELS = {"online": "Online (1x updates)", "repeat": "Repeat (2x updates)", "replay": "Replay (2x updates)"}
_RUNTIME_CONFIG = {"device", "threads"}


def _fail(message: str) -> None:
    raise ValueError(message)


def _read_json(path: Path):
    def reject_constant(value):
        _fail(f"{path}: non-finite JSON value {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash(value, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        _fail(f"{where}: expected a SHA256 hexadecimal digest")
    return value


def _seeds(value) -> list[int]:
    raw = value.split(",") if isinstance(value, str) else value
    if not isinstance(raw, list) or not raw:
        _fail("manifest seeds must be a nonempty list or comma-separated string")
    seeds = []
    for item in raw:
        if isinstance(item, bool) or (not isinstance(item, (int, str))):
            _fail(f"invalid seed: {item!r}")
        seed = int(item)
        if seed < 0:
            _fail(f"invalid seed: {item!r}")
        seeds.append(seed)
    if len(set(seeds)) != len(seeds):
        _fail("manifest contains duplicate seeds")
    return seeds


def _schedule(order: str, config: dict) -> list[str]:
    if order == "blocked":
        return [c for c in ("A0", "B0", "C0") for _ in range(config["rounds"])] + [
            c for c in ("A1", "B0", "C0") for _ in range(config["revision_rounds"])
        ]
    return ["A0", "B0", "C0"] * config["rounds"] + ["A1", "B0", "C0"] * config["revision_rounds"]


def _accuracy(value, where: str, allow_none: bool = False) -> None:
    if allow_none and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        _fail(f"{where}: expected finite accuracy in [0, 1], got {value!r}")


def _evaluation(value: dict, config: dict, where: str) -> None:
    split = config["eval_split"]
    n = config["n_val"] if split == "val" else config["n_test"]
    if set(value) != {split, "transfer_" + split}:
        _fail(f"{where}: missing or unexpected evaluation domains")
    for domain, conditions in value.items():
        if set(conditions) != set(CONDITIONS):
            _fail(f"{where}/{domain}: expected all four conditions")
        for condition, metric in conditions.items():
            prefix = f"{where}/{domain}/{condition}"
            if metric["n"] != n:
                _fail(f"{prefix}: evaluation sample count differs from config")
            _accuracy(metric["accuracy"], prefix)
            counts = 0
            for group in ("ab_agree", "ab_disagree"):
                subset = metric[group]
                if not isinstance(subset["n"], int) or not 0 <= subset["n"] <= n:
                    _fail(f"{prefix}/{group}: invalid subset count")
                counts += subset["n"]
                _accuracy(subset["accuracy"], f"{prefix}/{group}", allow_none=subset["n"] == 0)
                if (subset["accuracy"] is None) != (subset["n"] == 0):
                    _fail(f"{prefix}/{group}: an empty subset must have null accuracy")
            if counts != n:
                _fail(f"{prefix}: agreement subsets do not partition evaluation inputs")


def _validate_result(result: dict, manifest: dict, path: Path) -> None:
    config = manifest["config"]
    if result["config"] != config:
        _fail(f"{path}: result config differs from its manifest")
    order, method = result["order"], result["method"]
    expected_schedule = _schedule(order, config)
    trajectory = result["trajectory"]
    if len(trajectory) != len(expected_schedule):
        _fail(f"{path}: incomplete block trajectory")
    _hash(result["initial_hash"], f"{path}/initial_hash")
    _hash(result["buffer_hash"], f"{path}/buffer_hash")
    _hash(result["checkpoint_sha256"], f"{path}/checkpoint_sha256")
    if set(result["external_stream_hashes"]) != set(CONDITIONS):
        _fail(f"{path}: incomplete external stream hashes")
    for condition, digest in result["external_stream_hashes"].items():
        _hash(digest, f"{path}/external_stream_hashes/{condition}")
    _evaluation(result["initial"], config, f"{path}/initial")
    _evaluation(result["final"], config, f"{path}/final")
    previous = result["initial"]
    visits = Counter()
    last_seen = {c: None for c in CONDITIONS}
    block_examples = config["steps_per_block"] * config["batch"]
    factor = 1 if method == "online" else 2
    for block, (item, condition) in enumerate(zip(trajectory, expected_schedule)):
        visits[condition] += 1
        expected_updates = (block + 1) * config["steps_per_block"] * factor
        if (item["block"], item["condition"], item["visit"]) != (block, condition, visits[condition]):
            _fail(f"{path}: incorrect condition, visit, or block number at block {block}")
        if item["new_examples"] != (block + 1) * block_examples or item["optimizer_steps"] != expected_updates:
            _fail(f"{path}: incorrect external-example or optimizer budget at block {block}")
        if item["before"] != previous:
            _fail(f"{path}: before/after evaluation chain is broken at block {block}")
        before_age = {c: None if last is None else block * config["steps_per_block"] - last for c, last in last_seen.items()}
        last_seen[condition] = (block + 1) * config["steps_per_block"]
        after_age = {c: None if last is None else (block + 1) * config["steps_per_block"] - last for c, last in last_seen.items()}
        if item["before_age_external_steps"] != before_age or item["after_age_external_steps"] != after_age:
            _fail(f"{path}: return-age records disagree with the schedule at block {block}")
        _accuracy(item["prequential_accuracy"], f"{path}/block{block}/prequential")
        _accuracy(item["first_batch_before_feedback"], f"{path}/block{block}/first_batch_before_feedback")
        _evaluation(item["after"], config, f"{path}/block{block}/after")
        previous = item["after"]
    total_examples = len(expected_schedule) * block_examples
    total_updates = len(expected_schedule) * config["steps_per_block"] * factor
    if result["final"] != previous:
        _fail(f"{path}: final evaluation differs from the final block")
    expected_counts = {
        "new_examples": total_examples,
        "optimizer_steps": total_updates,
        "training_examples": total_updates * config["batch"],
        "buffer_seen": total_examples,
        "buffer_size": min(config["capacity"], total_examples),
    }
    for field, expected in expected_counts.items():
        if result[field] != expected:
            _fail(f"{path}: {field} is {result[field]!r}, expected {expected}")
    replayed = result["replayed_examples"]
    if not isinstance(replayed, int) or not 0 <= replayed < total_examples or (method != "replay" and replayed != 0):
        _fail(f"{path}: invalid replayed-example count")
    if _read_json(path.with_name("trajectory.json")) != trajectory:
        _fail(f"{path}: standalone trajectory differs from results.json")
    if _digest(path.with_name("final.pt")) != result["checkpoint_sha256"]:
        _fail(f"{path}: final checkpoint hash mismatch")


def collect_inputs(paths: list[Path]) -> tuple[list[dict], list[dict], dict, dict]:
    """Read completed roots and reject partial, duplicate, or unpaired evidence."""
    roots = [p.resolve() for p in paths]
    if len(set(roots)) != len(roots):
        _fail("duplicate input run directory")
    inputs, records = [], []
    canonical_config = None
    canonical_sources = None
    by_key = {}
    world_hashes = {}
    for input_id, root in enumerate(roots):
        manifest = _read_json(root / "manifest.json")
        index = _read_json(root / "index.json")
        verification = _read_json(root / "verification.json")
        config = manifest["config"]
        required = {"input_dim", "classes", "width", "n_train", "n_val", "n_test", "batch", "capacity", "steps_per_block", "rounds", "revision_rounds", "lr", "eval_split"}
        if not required <= set(config) or config["eval_split"] not in ("val", "test"):
            _fail(f"{root}: missing core config fields or invalid evaluation split")
        main_config = {k: v for k, v in config.items() if k not in _RUNTIME_CONFIG}
        if canonical_config is None:
            canonical_config = main_config
        elif main_config != canonical_config:
            _fail(f"{root}: core configuration differs between input roots")
        seeds = _seeds(manifest["seeds"])
        methods, orders = manifest["methods"], manifest["orders"]
        if not methods or len(set(methods)) != len(methods) or not set(methods) <= set(METHODS):
            _fail(f"{root}: invalid or duplicate methods")
        if not orders or len(set(orders)) != len(orders) or not set(orders) <= set(ORDERS):
            _fail(f"{root}: invalid or duplicate orders")
        if verification.get("source_unchanged") is not True:
            _fail(f"{root}: source-unchanged verification is missing or false")
        if set(verification["world_hashes"]) != {str(seed) for seed in seeds}:
            _fail(f"{root}: incomplete world hashes")
        for seed in seeds:
            value = _hash(verification["world_hashes"][str(seed)], f"{root}/world_hashes/{seed}")
            if seed in world_hashes and world_hashes[seed] != value:
                _fail(f"seed {seed}: teacher/world hash differs across inputs")
            world_hashes[seed] = value
        sources = manifest["source_sha256"]
        if canonical_sources is None:
            canonical_sources = sources
        elif sources != canonical_sources:
            _fail(f"{root}: source hashes differ between inputs")
        source_names = [Path(name).name for name in sources]
        if len(set(source_names)) != len(source_names) or not {"run.py", "world.py", "memory.py", "model.py"} <= set(source_names):
            _fail(f"{root}: core source snapshots are missing or ambiguous")
        for name, expected in sources.items():
            _hash(expected, f"{root}/source_sha256/{name}")
            if _digest(root / "source" / Path(name).name) != expected:
                _fail(f"{root}: source snapshot hash mismatch for {name}")
        completed = index["completed"]
        if not isinstance(completed, list) or len(set(completed)) != len(completed):
            _fail(f"{root}: invalid or duplicate index entries")
        expected_keys = {(seed, order, method) for seed in seeds for order in orders for method in methods}
        found_keys = set()
        indexed_paths = set()
        for relative in completed:
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or path.name != "results.json":
                _fail(f"{root}: invalid indexed result path {relative!r}")
            result = _read_json(path)
            seed, order, method = result["seed"], result["order"], result["method"]
            key = (seed, order, method)
            if key not in expected_keys or key in found_keys or key in by_key:
                _fail(f"{path}: unexpected or duplicate seed/order/method {key}")
            _validate_result(result, manifest, path)
            found_keys.add(key)
            indexed_paths.add(path)
            record = {"input_id": input_id, "relative_path": path.relative_to(root).as_posix(), "result": result}
            records.append(record)
            by_key[key] = result
        if found_keys != expected_keys:
            _fail(f"{root}: incomplete index; missing {sorted(expected_keys - found_keys)}")
        disk_paths = {p.resolve() for p in root.rglob("results.json")}
        if indexed_paths != disk_paths:
            _fail(f"{root}: indexed results differ from result files on disk")
        inputs.append({"input_id": input_id, "path": str(root), "manifest": manifest, "index": index, "verification": verification})
    combination_seeds = {}
    for order in ORDERS:
        for method in METHODS:
            seeds = sorted(seed for seed, o, m in by_key if o == order and m == method)
            if len(seeds) != 3:
                _fail(f"{order}/{method}: expected exactly 3 independent seeds, found {seeds}")
            combination_seeds[(order, method)] = seeds
    seed_ids = combination_seeds[(ORDERS[0], METHODS[0])]
    if any(seeds != seed_ids for seeds in combination_seeds.values()):
        _fail("methods/orders do not use the same three seed IDs")
    initial_hashes, external_hashes, buffer_hashes = {}, {}, {}
    parameter_counts = {record["result"]["parameters"] for record in records}
    if len(parameter_counts) != 1:
        _fail("parameter counts differ between runs")
    for seed in seed_ids:
        paired = [by_key[(seed, order, method)] for order in ORDERS for method in METHODS]
        if len({r["initial_hash"] for r in paired}) != 1:
            _fail(f"seed {seed}: model initialization differs across methods/orders")
        initial_hashes[str(seed)] = paired[0]["initial_hash"]
        if any(r["external_stream_hashes"] != paired[0]["external_stream_hashes"] for r in paired):
            _fail(f"seed {seed}: per-context external streams differ across methods/orders")
        external_hashes[str(seed)] = paired[0]["external_stream_hashes"]
        buffer_hashes[str(seed)] = {}
        for order in ORDERS:
            hashes = {by_key[(seed, order, method)]["buffer_hash"] for method in METHODS}
            if len(hashes) != 1:
                _fail(f"seed {seed}/{order}: final buffer differs across methods")
            buffer_hashes[str(seed)][order] = next(iter(hashes))
    verification = {
        "passed": True,
        "checks": [
            "Each input manifest's full Cartesian product is complete, with no unindexed or duplicate results.",
            "All six order/method combinations contain the same exactly three independent seed IDs.",
            "Core configurations, parameter counts, and verified source snapshots match.",
            "Per-seed teacher/world hashes, initial parameters, and per-condition external streams match.",
            "Per-seed/order final buffer hashes match across methods.",
            "Schedules, evaluation chains, checkpoints, return ages, and external/update budgets are complete.",
        ],
        "seed_ids": seed_ids,
        "runs": len(records),
        "parameters": next(iter(parameter_counts)),
        "source_sha256": canonical_sources,
        "paired_world_hashes": {str(seed): world_hashes[seed] for seed in seed_ids},
        "paired_initial_hashes": initial_hashes,
        "paired_external_stream_hashes": external_hashes,
        "paired_buffer_hashes": buffer_hashes,
    }
    return inputs, records, canonical_config, verification


def _stats(values: dict[int, float | None]) -> dict:
    valid = [float(v) for v in values.values() if v is not None]
    complete = len(valid) == len(values)
    return {
        "mean": statistics.mean(valid) if complete and valid else None,
        "sample_sd": statistics.stdev(valid) if complete and len(valid) > 1 else None,
        "n_seeds": len(values),
        "n_valid": len(valid),
        "values_by_seed": {str(seed): value for seed, value in sorted(values.items())},
    }


def _metrics(evaluation: dict, domain: str) -> dict[str, float]:
    values = {condition: evaluation[domain][condition]["accuracy"] for condition in CONDITIONS}
    values["current_mean"] = statistics.mean(values[c] for c in ("A1", "B0", "C0"))
    values["all4_mean"] = statistics.mean(values[c] for c in CONDITIONS)
    return values


def _snapshot(runs: list[dict], selector, domains: tuple[str, str]) -> dict:
    return {
        domain: {
            metric: _stats({r["seed"]: _metrics(selector(r), domain)[metric] for r in runs})
            for metric in METRICS
        }
        for domain in domains
    }


def _first(run: dict, condition: str) -> dict:
    return next(item for item in run["trajectory"] if item["condition"] == condition)


def _last_a1(run: dict) -> dict:
    return next(item for item in reversed(run["trajectory"]) if item["condition"] == "A1")


def _curve(run: dict, domain: str) -> dict:
    config = run["config"]
    revision = _first(run, "A1")
    start = revision["new_examples"] - config["steps_per_block"] * config["batch"]
    xs, historical, current = [], [], []

    def append(examples: int, evaluation: dict, revised: bool):
        xs.append(examples)
        historical.append(evaluation[domain]["A0"]["accuracy"])
        active_a = "A1" if revised else "A0"
        current.append(statistics.mean(evaluation[domain][c]["accuracy"] for c in (active_a, "B0", "C0")))

    append(0, run["initial"], False)
    for item in run["trajectory"]:
        if item["block"] == revision["block"]:
            # Duplicate x shows the supplied rule change at a fixed parameter
            # state, rather than smearing it across an earlier training block.
            append(start, item["before"], True)
        append(item["new_examples"], item["after"], item["block"] >= revision["block"])
    return {"external_examples": xs, "revision_start": start, "historical_A0": historical, "current_mean": current}


def aggregate_runs(records: list[dict], config: dict) -> dict:
    split = config["eval_split"]
    domains = (split, "transfer_" + split)
    result = {}
    for order in ORDERS:
        result[order] = {}
        for method in METHODS:
            runs = sorted(
                [r["result"] for r in records if r["result"]["order"] == order and r["result"]["method"] == method],
                key=lambda r: r["seed"],
            )
            summary = {
                "seeds": [r["seed"] for r in runs],
                "final": _snapshot(runs, lambda r: r["final"], domains),
                "revision_end": _snapshot(runs, lambda r: _last_a1(r)["after"], domains),
                "first_A1_before": _snapshot(runs, lambda r: _first(r, "A1")["before"], domains),
                "first_A1_after": _snapshot(runs, lambda r: _first(r, "A1")["after"], domains),
                "first_B_before": {},
                "return_before": {},
                "resources": {
                    field: _stats({r["seed"]: r[field] for r in runs})
                    for field in ("new_examples", "optimizer_steps", "training_examples", "replayed_examples", "buffer_size", "buffer_bytes", "elapsed_seconds", "peak_cuda_bytes")
                },
                "curves": {},
            }
            for domain in domains:
                groups = {}
                for group in ("ab_agree", "ab_disagree"):
                    before = {r["seed"]: _first(r, "B0")["before"][domain]["B0"][group]["accuracy"] for r in runs}
                    initial = {r["seed"]: r["initial"][domain]["B0"][group]["accuracy"] for r in runs}
                    gain = {seed: None if before[seed] is None or initial[seed] is None else before[seed] - initial[seed] for seed in before}
                    groups[group] = {
                        "accuracy": _stats(before), "initial_accuracy": _stats(initial), "gain_from_initial": _stats(gain),
                        "n_examples": _stats({r["seed"]: _first(r, "B0")["before"][domain]["B0"][group]["n"] for r in runs}),
                    }
                summary["first_B_before"][domain] = {
                    "B0_accuracy": _stats({r["seed"]: _first(r, "B0")["before"][domain]["B0"]["accuracy"] for r in runs}),
                    **groups,
                }
                curves = {r["seed"]: _curve(r, domain) for r in runs}
                reference = curves[runs[0]["seed"]]
                if any(c["external_examples"] != reference["external_examples"] or c["revision_start"] != reference["revision_start"] for c in curves.values()):
                    _fail(f"{order}/{method}: curve coordinates differ across seeds")
                summary["curves"][domain] = {
                    "external_examples": reference["external_examples"], "revision_start": reference["revision_start"],
                    **{
                        metric: [_stats({seed: curve[metric][i] for seed, curve in curves.items()}) for i in range(len(reference["external_examples"]))]
                        for metric in ("historical_A0", "current_mean")
                    },
                }
            for condition in CONDITIONS:
                by_seed = {r["seed"]: [item for item in r["trajectory"] if item["condition"] == condition and item["visit"] > 1] for r in runs}
                count = len(next(iter(by_seed.values())))
                visits = []
                for index in range(count):
                    items = {seed: items[index] for seed, items in by_seed.items()}
                    reference = next(iter(items.values()))
                    visits.append({
                        "visit": reference["visit"], "block": reference["block"],
                        "external_examples": reference["new_examples"] - config["steps_per_block"] * config["batch"],
                        "age_external_steps": _stats({seed: item["before_age_external_steps"][condition] for seed, item in items.items()}),
                        "accuracy": {domain: _stats({seed: item["before"][domain][condition]["accuracy"] for seed, item in items.items()}) for domain in domains},
                    })
                summary["return_before"][condition] = {
                    "n_returns_per_seed": count,
                    "mean_across_returns": {
                        domain: _stats({seed: statistics.mean(item["before"][domain][condition]["accuracy"] for item in items) if items else None for seed, items in by_seed.items()})
                        for domain in domains
                    },
                    "by_visit": visits,
                }
            result[order][method] = summary
    return result


def _cell(stats: dict, percent: bool = True) -> str:
    if stats["mean"] is None or stats["sample_sd"] is None:
        return "NA"
    scale = 100 if percent else 1
    return f"{stats['mean'] * scale:.2f} ± {stats['sample_sd'] * scale:.2f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def write_tables(path: Path, report: dict) -> None:
    config, aggregate = report["config"], report["aggregates"]
    split = config["eval_split"]
    domains = (split, "transfer_" + split)
    lines = [
        "# Consolidation experiment tables", "",
        "All accuracies are percentages, reported as **mean ± sample SD across three independent seeds**. "
        "Every seed is weighted equally; evaluation examples are not pooled across seeds. "
        "Gain values are percentage-point changes, paired within each seed. NA means the requested metric is undefined for at least one seed.", "",
        "**Online uses half the optimizer updates and training-example presentations. Only repeat versus replay is the compute-matched comparison.** "
        "External observations, initialization, teachers, and per-context streams are paired. All methods retain the same buffer; only replay uses its historical records for learning.", "",
        f"Evaluation split: `{split}`; seeds: `{report['verification']['seed_ids']}`. "
        "The version cue is an observed context-local rule change, not autonomous change detection. "
        "The held-out positive-positive conjunction is a narrow transfer probe.", "",
        "`current_mean` = mean(A1, B0, C0); `all4_mean` = mean(A0, A1, B0, C0). "
        "A0 measures retention of the superseded historical rule. A1 is evaluated separately; historical-rule retention is not required for present-rule correctness.", "",
    ]
    for title, field in (("Final checkpoint", "final"), ("Revision end: immediately after the last A1 block", "revision_end"), ("First A1 block: before feedback", "first_A1_before"), ("First A1 block: after training", "first_A1_after")):
        lines += [f"## {title}", ""]
        for domain in domains:
            rows = [[order, method, *[_cell(aggregate[order][method][field][domain][metric]) for metric in METRICS]] for order in ORDERS for method in METHODS]
            lines += [f"### {domain}", "", _table(["Order", "Method", *METRICS], rows), ""]
    lines += ["## B0 transfer before its first feedback", "", "B0 is evaluated just before its first B0 block. The agreement subsets use the environment's actual A0/B0 agreement on each evaluation input. Gain subtracts the same seed's untrained initial B0 accuracy; it is not a separate training stage.", ""]
    for domain in domains:
        rows = []
        for order in ORDERS:
            for method in METHODS:
                item = aggregate[order][method]["first_B_before"][domain]
                rows.append([order, method, _cell(item["B0_accuracy"]), _cell(item["ab_agree"]["accuracy"]), _cell(item["ab_agree"]["gain_from_initial"]), _cell(item["ab_disagree"]["accuracy"]), _cell(item["ab_disagree"]["gain_from_initial"])])
        lines += [f"### {domain}", "", _table(["Order", "Method", "B0 overall", "A/B agree", "Agree gain (pp)", "A/B disagree", "Disagree gain (pp)"], rows), ""]
    lines += ["## Accuracy before returning to each condition", "", "A return has visit >= 2. First average return accuracies within each seed, then compute the three-seed mean and sample SD. Consecutive blocked visits are included; their gap is zero. Detailed per-return measurements follow.", ""]
    for domain in domains:
        rows = [[order, method, *[_cell(aggregate[order][method]["return_before"][c]["mean_across_returns"][domain]) for c in CONDITIONS]] for order in ORDERS for method in METHODS]
        lines += [f"### {domain}", "", _table(["Order", "Method", *CONDITIONS], rows), ""]
    lines += ["### Every return", "", "Gap is the number of intervening external minibatch steps, not optimizer updates. The example count is measured before the returning block.", ""]
    rows = []
    for order in ORDERS:
        for method in METHODS:
            for condition in CONDITIONS:
                for item in aggregate[order][method]["return_before"][condition]["by_visit"]:
                    rows.append([order, method, condition, str(item["visit"]), str(item["external_examples"]), _cell(item["age_external_steps"], percent=False), *[_cell(item["accuracy"][domain]) for domain in domains]])
    lines += [_table(["Order", "Method", "Condition", "Visit", "External examples", "Gap steps", *domains], rows), "", "## Measured budgets and resources", "", "Wall time and peak allocation are descriptive measurements, not matched-compute evidence. Model/buffer sizes and all paired hashes are embedded in summary.json.", ""]
    fields = ("new_examples", "optimizer_steps", "training_examples", "replayed_examples", "buffer_bytes", "elapsed_seconds", "peak_cuda_bytes")
    rows = [[order, method, *[_cell(aggregate[order][method]["resources"][field], percent=False) for field in fields]] for order in ORDERS for method in METHODS]
    lines += [_table(["Order", "Method", *fields], rows), "", "## Verification", ""]
    lines += ["- " + check for check in report["verification"]["checks"]]
    lines += ["", "Figures: trajectories.png / trajectories.svg and final_accuracy.png / final_accuracy.svg. "
              "The trajectory bands and final error bars show sample SD. The current-rule curve uses A0/B0/C0 before the revision cue, then A1/B0/C0. "
              "Its vertical jump at the cue evaluates the new rule with the same pre-update parameters.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figures(output: Path, report: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    aggregate, split = report["aggregates"], report["config"]["eval_split"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), sharex=True, sharey=True)
    for row, order in enumerate(ORDERS):
        for column, (metric, title) in enumerate((("historical_A0", "Historical A0"), ("current_mean", "Current valid rules"))):
            ax = axes[row, column]
            for method in METHODS:
                curve = aggregate[order][method]["curves"][split]
                xs = curve["external_examples"]
                means = [s["mean"] for s in curve[metric]]
                sds = [s["sample_sd"] for s in curve[metric]]
                ax.plot(xs, means, color=COLORS[method], lw=1.8, label=LABELS[method])
                ax.fill_between(xs, [max(0, m - s) for m, s in zip(means, sds)], [min(1, m + s) for m, s in zip(means, sds)], color=COLORS[method], alpha=.14, linewidth=0)
            revision = aggregate[order][METHODS[0]]["curves"][split]["revision_start"]
            ax.axvline(revision, color="#555555", ls=":", lw=1.3)
            ax.set_title(f"{order.capitalize()}: {title}", loc="left", fontsize=11)
            ax.set_ylim(0, 1)
            ax.set_xlim(0, xs[-1])
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x / 1000:g}k" if x else "0"))
            ax.grid(axis="y", alpha=.18)
            if column == 0:
                ax.set_ylabel("Accuracy")
            if row == 1:
                ax.set_xlabel("External examples")
    handles = [Line2D([0], [0], color=COLORS[m], lw=2, label=LABELS[m]) for m in METHODS]
    handles.append(Line2D([0], [0], color="#555555", lw=1.3, ls=":", label="Observed revision cue"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.5, .975))
    fig.suptitle(f"Consolidation trajectories ({split}; mean ± sample SD, 3 seeds)", y=1.015, fontsize=14)
    fig.text(.5, .005, "Current rules: A0/B0/C0 before the cue; A1/B0/C0 after. Only repeat and replay have matched update budgets.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .925))
    for suffix in ("png", "svg"):
        fig.savefig(output / f"trajectories.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    categories = ((split, "current_mean", "Current\nA1/B0/C0"), (split, "A0", "Historical\nA0"), ("transfer_" + split, "current_mean", "Transfer current\nA1/B0/C0"))
    for ax, order in zip(axes, ORDERS):
        for index, method in enumerate(METHODS):
            points = [aggregate[order][method]["final"][domain][metric] for domain, metric, _ in categories]
            positions = [i + (index - 1) * .24 for i in range(len(categories))]
            ax.bar(positions, [s["mean"] for s in points], width=.22, yerr=[s["sample_sd"] for s in points], capsize=3, color=COLORS[method], label=LABELS[method], error_kw={"elinewidth": 1})
        ax.set_xticks(range(len(categories)), [label for _, _, label in categories])
        ax.set_title(order.capitalize(), loc="left")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Final accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, 1.025))
    fig.suptitle("Final checkpoint: mean ± sample SD across 3 seeds", y=1.095, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    for suffix in ("png", "svg"):
        fig.savefig(output / f"final_accuracy.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True, help="Complete run.py roots, each with manifest/index/verification")
    parser.add_argument("--output", type=Path, required=True, help="Empty or new report directory")
    args = parser.parse_args()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("refusing to overwrite a nonempty output directory")
    try:
        inputs, records, config, verification = collect_inputs(args.inputs)
        aggregates = aggregate_runs(records, config)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    report = {
        "schema": "consolidation-summary-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "definitions": {
            "uncertainty": "Sample SD across the same three independent seeds; no pooling across evaluation examples.",
            "current_mean": "Mean of A1/B0/C0 accuracy at reported final/revision snapshots.",
            "all4_mean": "Mean of A0/A1/B0/C0 accuracy, including the superseded A0 rule.",
            "current_curve": "Mean of A0/B0/C0 before the observed revision cue, then A1/B0/C0; duplicate cue x shows the instantaneous rule switch.",
            "revision_end": "Evaluation immediately after the last A1 training block, which may precede later B0/C0 blocks.",
            "first_B_before": "B0 before its first feedback; actual A0/B0 agreement subsets and paired gains over initial B0 accuracy.",
            "return_before": "Before each condition's visits >= 2; per-condition averages are taken within seed before computing cross-seed SD.",
            "compute": "Online uses half the optimizer updates/training presentations. Only repeat and replay are compute matched.",
        },
        "verification": verification,
        "inputs": inputs,
        "full_run_results": records,
        "aggregates": aggregates,
        "artifacts": ["summary.json", "tables.md", "trajectories.png", "trajectories.svg", "final_accuracy.png", "final_accuracy.svg"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_figures(args.output, report)
    write_tables(args.output / "tables.md", report)
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "runs": verification["runs"], "seeds": verification["seed_ids"], "verification": "passed", "artifacts": report["artifacts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
