"""Summarize raw context-local tests by independent experimental roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def difference(new, old):
    delta = np.asarray(new) - np.asarray(old)
    indices = np.random.default_rng(204916).integers(len(delta), size=(10_000, len(delta)))
    return {"mean": float(delta.mean()), "ci95": np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist()}


def recall(record):
    return np.asarray(record["rows"]["return_A"])[1:3].mean(axis=0)


def one_model(report):
    groups = report["reports"]
    scenes = {}
    for context in ("separated", "nearby", "indistinguishable"):
        if context not in groups:
            continue
        cases = groups[context]
        fields = {}
        for mode, record in cases.items():
            scores = record["summary"]
            fields[mode] = {
                "conflict_A": scores["return_A"]["conflict"], "conflict_B48": scores["B_after_48"]["conflict"],
                "conflict_both": scores["both"]["conflict"], "same_B0": scores["B_after_0"]["same"],
                "conflict_B0": scores["B_after_0"]["conflict"],
                "addresses": record.get("addresses"), "default_rule": record.get("default_rule"),
            }
        scenes[context] = fields
    novel = {name: record["summary"]["return_A"] for name, record in groups.get("novel_cue", {}).get("arms", {}).items()}
    same = groups.get("same_body_native")
    same_decay = groups.get("same_body_global_decay")
    return {
        "retention": report["frozen_rule_retention"], "cases": scenes, "novel_cue": novel,
        "separated_local_minus_native_A": difference(recall(groups["separated"]["local"]), recall(groups["separated"]["native"])),
        "same_body_local_minus_native_A": difference(recall(same["local"]), recall(same["native"])) if same else None,
        "same_body_local_minus_global_decay_A": difference(recall(same_decay["local"]), recall(same_decay["global_decay"])) if same_decay else None,
        "same_body_conflict_A": {mode: float(recall(record).mean()) for mode, record in same.items() if mode in ("local", "native")} if same else None,
    }


def multi(report):
    result = {"model": report["model"], "roots": len(report["roots"])}
    for count in (3, 4):
        arms = report["arms"][f"{count}_visible_habitats"]
        arm_values = {mode: record["summary"] for mode, record in arms.items()}
        returns = lambda mode: np.asarray(arms[mode]["rows"]["returned_correct"]).mean(axis=1)
        result[str(count)] = {"arms": arm_values, "6_minus_2": difference(returns("6_slots"), returns("2_slots")),
                              "6_minus_3": difference(returns("6_slots"), returns("3_slots"))}
    result["cue_collision"] = report["arms"]["4_habitats_2_cues_identical"]["summary"]
    result["new_rule_twins"] = report["arms"]["new_rule_twins"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.reports.glob("rls-learned-*-full-*.json"))
    reports = {p.stem: read(p) for p in paths}
    roots = [record["root_seeds"] for record in reports.values()]
    if roots and any(row != roots[0] for row in roots):
        raise ValueError("cannot pool differing roots across fixed models")
    results = {"binary_models": {name: one_model(record) for name, record in reports.items()}}
    paired_records = [r["reports"]["same_body_native"] for r in reports.values() if "same_body_native" in r["reports"]]
    if paired_records:
        diffs = [recall(r["local"]) - recall(r["native"]) for r in paired_records]
        effect = np.stack(diffs).mean(axis=0)
        results["fixed_models_same_body_averaged_by_root"] = {"roots": len(effect), "fixed_models": len(diffs), **difference(effect, np.zeros_like(effect))}
    multi_files = sorted(args.reports.glob("multihabitat-*.json"))
    results["multi"] = {p.stem: multi(read(p)) for p in multi_files}
    bank_file = args.reports / "banked-native.json"
    if bank_file.exists():
        banked = read(bank_file)
        results["native_code_banked"] = {
            name: {scene: entry["banked_minus_native_return_A"] for scene, entry in r["conditions"].items()}
            for name, r in banked["models"].items()
        }
    args.out.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
