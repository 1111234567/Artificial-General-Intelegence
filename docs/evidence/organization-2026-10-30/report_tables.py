"""Recreate the report's selected tables from passed, SHA-bound CSVs."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import statistics as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BINDINGS = {}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows(folder, name):
    folder = HERE / folder
    manifest = folder / "artifacts.json"
    bindings = json.loads(manifest.read_text(encoding="utf-8"))
    path = folder / f"{name}.csv"
    assert sha(path) == bindings[path.name]
    BINDINGS[str(manifest)] = sha(manifest)
    BINDINGS[str(path)] = sha(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def fmt(values, scale=1., decimals=2):
    assert len(values) == 3
    values = [value * scale for value in values]
    return f"{st.mean(values):.{decimals}f} ± {st.stdev(values):.{decimals}f}"


def table(columns, values):
    return "\n".join(["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns),
        *("| " + " | ".join(map(str, row)) + " |" for row in values)])


def main(fill=False):
    acq = rows("acquisition-aggregate-v1", "per_seed")
    old = rows("acquisition-aggregate-v1", "baseline_per_seed")
    main_rows = rows("aggregate-v1", "family_average_summary")
    training = rows("aggregate-v1", "training_per_seed")
    probes = rows("aggregate-v1", "functional_probes")

    def acq_values(data, architecture, coverage, condition, metric):
        result = []
        for seed in (0, 1, 2):
            selected = [r for r in data if (r["architecture"], r["coverage"], r["condition"], r["tick"], r["seed"])
                        == (architecture, coverage, condition, "131072", str(seed))]
            assert {int(r["family"]) for r in selected} == {0, 1, 2, 3} and len(selected) == 4
            result.append(st.mean(float(r[metric]) for r in selected))
        return result

    def summary(field, rule, condition, tick, metric):
        selected = [r for r in main_rows if (r["field_mode"], r["structural_rule"], r["condition"], r["tick"], r["metric"])
                    == (field, rule, condition, str(tick), metric)]
        assert len(selected) == 1 and int(selected[0]["n"]) == 3
        return [r["value"] for r in sorted(json.loads(selected[0]["per_seed"]), key=lambda x: x["seed"])]

    def trait(field, rule, key):
        selected = sorted((r for r in training if (r["field_mode"], r["structural_rule"]) == (field, rule)),
                          key=lambda r: int(r["seed"]))
        assert [int(r["seed"]) for r in selected] == [0, 1, 2]
        return [json.loads(r["structure"])[key] for r in selected]

    selections = {}
    data = []
    for architecture, coverage in (("fixed_shrink", "wide"), ("open_shrink", "wide"), ("fixed_shrink", "narrow")):
        for condition in ("short", "long_both"):
            before = acq_values(old, architecture, coverage, condition, "strict_completion")
            after = acq_values(acq, architecture, coverage, condition, "strict_completion")
            ordinary = [acq_values(source, architecture, coverage, condition, "completion") for source in (old, acq)]
            idle = [acq_values(source, architecture, coverage, condition, "inactive_per_1000_idle") for source in (old, acq)]
            data.append([f"{architecture.split('_')[0]}/{coverage}，{condition}",
                f"{100 * st.mean(before):.2f} → {100 * st.mean(after):.2f}",
                fmt([b - a for a, b in zip(before, after)], 100),
                f"{100 * st.mean(ordinary[0]):.2f} → {100 * st.mean(ordinary[1]):.2f}",
                f"{st.mean(idle[0]):.2f} → {st.mean(idle[1]):.2f}"])
    selections["acquisition_table"] = table(["条件", "严格：旧→零", "严格配对差", "普通：旧→零", "idle：旧→零"], data)

    data = []
    for field in ("plain", "parallel", "coupled"):
        data.append([field, fmt(summary(field, "passive", "long_both", 131072, "strict_completion"), 100),
            fmt(summary(field, "passive", "long_both", 172032, "strict_completion"), 100),
            fmt(summary(field, "passive", "long_both", 172032, "completion"), 100),
            fmt(summary(field, "passive", "long_both", 172032, "inactive_per_1000_idle"))])
    selections["topology_table"] = table(["场", "主训末严格", "最终严格", "最终普通", "最终idle/千机会"], data)

    variants = [("plain", f"functional_{rule}") for rule in ("fixed", "task", "magnitude", "random")]
    variants += [("coupled", f"functional_{rule}") for rule in ("fixed", "task")]
    data = []
    for field, rule in variants:
        data.append([f"{field}/{rule.removeprefix('functional_')}",
            fmt(summary(field, rule, "long_both", 172032, "strict_completion"), 100),
            fmt(summary(field, rule, "long_both", 172032, "completion"), 100),
            fmt(summary(field, rule, "long_both", 172032, "inactive_per_1000_idle"))])
    selections["functional_table"] = table(["策略", "严格完成（%）", "普通完成（%）", "idle/千机会"], data)
    data = []
    for field, rule in variants:
        data.append([f"{field}/{rule.removeprefix('functional_')}", *[
            fmt(trait(field, rule, key), decimals=1) for key in
            ("active_edges", "active_cross_bank_bridges", "cumulative_births", "cumulative_deaths")]])
    selections["capacity_table"] = table(["策略", "非零边", "非零跨部分边", "累计出生", "累计退出"], data)

    data, history = [], []
    for field, rule in variants:
        if rule == "functional_fixed":
            continue
        selected = [r for r in probes if (r["field_mode"], r["structural_rule"]) == (field, rule)]
        assert len(selected) == 501
        births = [r for r in selected if r["birth_accepted"] == "True"]
        exits = [r for r in selected if r["exit_accepted"] == "True"]
        harmful = lambda r: json.loads(r["full_ablation_damage"])["mixed"] > 0
        label = f"{field}/{rule.removeprefix('functional_')}"
        data.append([label, f"{len(births)}/501", f"{len(exits)}/501",
                     f"{sum(map(harmful, selected))}/501", f"{sum(map(harmful, exits))}/{len(exits)}"])
        rates = []
        for accepted, key in ((births, "birth_damage"), (exits, "half_exit_damage")):
            values = []
            for seed in (0, 1, 2):
                subset = [r for r in accepted if int(r["seed"]) == seed]
                assert subset
                values.append(sum(json.loads(r[key])["history"] > 0 for r in subset) / len(subset))
            rates.append(fmt(values, 100))
        history.append([label, *rates])
    selections["probe_table"] = table(["策略", "接受出生", "接受减弱", "全删会增加mixed loss", "减弱可接受但全删有害"], data)
    selections["history_table"] = table(["策略", "已接受出生中历史loss恶化（%）", "已接受减弱中历史loss恶化（%）"], history)
    BINDINGS[str(Path(__file__).resolve())] = sha(__file__)
    (HERE / "selected-tables.json").write_text(json.dumps(dict(format="organization-selected-tables-v1",
        inputs_sha256=BINDINGS, independent_training_seeds=3,
        note="Four-family rates averaged within seed before mean/sampleSD. Trial batches are descriptive repeated measurements.",
        tables=selections), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if fill:
        path = ROOT / "docs/organization-experiment-2026-10-30.md"
        text = path.read_text(encoding="utf-8")
        for key, value in selections.items():
            marker = f"@@{key}@@"
            assert text.count(marker) == 1, marker
            text = text.replace(marker, value)
        path.write_text(text, encoding="utf-8")
    print(json.dumps(dict(tables=list(selections), bound_inputs=len(BINDINGS))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fill-report", action="store_true")
    main(parser.parse_args().fill_report)
