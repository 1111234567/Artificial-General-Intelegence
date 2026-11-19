"""Separate local storage from changing the original learned route code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import habitat_world
from .context_local_run import agent, make, phenotypes, same_body, score


def contrast(banked, native):
    rows = lambda r: np.asarray(r["rows"]["return_A"])[1:3].mean(axis=0)
    delta = rows(banked) - rows(native)
    sampled = np.random.default_rng(434832).integers(len(delta), size=(5000, len(delta)))
    return {"mean": float(delta.mean()), "ci95": np.quantile(delta[sampled].mean(axis=1), [.025, .975]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=128)
    parser.add_argument("--root-seed", type=int, default=19854495)
    parser.add_argument("--model-filter")
    args = parser.parse_args()
    torch.set_num_threads(2)
    roots = np.random.default_rng(args.root_seed).choice(470_000_000, size=args.worlds, replace=False).astype(np.int64).tolist()
    all_cases = {}
    for name, config, identity in phenotypes():
        if args.model_filter and args.model_filter not in name:
            continue
        conditions = {}
        for scene in ("separated", "nearby"):
            protocol = make(roots, scene)
            records = {}
            for arm in ("native", "local", "banked_native"):
                brain = agent(config, roots * 4, arm)
                body = habitat_world.rollout(brain, protocol.arena, silent_probes=True)
                records[arm] = score(protocol, body, brain, len(roots))
            benefit = contrast(records["banked_native"], records["native"])
            conditions[scene] = {"arms": records, "banked_minus_native_return_A": benefit}
        body_match = same_body(config, make(roots, "separated"), roots * 4, len(roots), "native", reference="banked_native")
        body_match["banked_minus_native_return_A"] = contrast(body_match["banked_native"], body_match["native"])
        conditions["same_body_native_vs_banked_native"] = body_match
        all_cases[name] = {"genotype": identity, "conditions": conditions}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {"root_seed": args.root_seed, "roots": roots, "models": all_cases, "uses_original_learned_code_in_all_banked_arms": True}
    args.out.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
