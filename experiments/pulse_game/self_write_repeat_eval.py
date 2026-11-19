"""Evaluate repeated-contact bootstrap checkpoints with NumPy deployment only."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from . import self_write_core as core
from .self_write_transfer import writer_delta


EVIDENCE = Path("docs/evidence/self-writing-2026-11-19/repeat")


def world(seed, lives, contacts, pairs, noise, switch):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(lives, pairs, 3))
    vectors = (1.35 * vectors / np.linalg.norm(vectors, axis=2, keepdims=True)).astype(np.float32)
    good = rng.choice(np.array([-1, 1], dtype=np.int8), (lives, pairs))
    order = rng.integers(pairs, size=(lives, contacts))
    orientation = rng.choice(np.array([-1, 1], dtype=np.int8), size=(lives, contacts))
    delivery_sign = np.where(rng.random((lives, contacts)) < noise, -1, 1)
    rows = np.arange(lives)[:, None]
    left = vectors[rows, order] * orientation[:, :, None]
    target = good[rows, order] * orientation
    if switch is not None:
        target[:, switch:] *= -1
    rewards = target * delivery_sign
    return left, target, rewards


def deploy(birth, tape, key, arm):
    left, target, rewards = tape
    lives, contacts = target.shape
    graph = core.repeated_graph(birth, lives)
    initial = graph
    h = core.rest(graph)
    action_correct = np.empty((lives, contacts), dtype=bool)
    probe_correct = np.empty((lives, contacts), dtype=np.float32)
    tastes = np.empty_like(target)
    edits = np.empty_like(target, dtype=np.int16)
    writer_distance = np.empty_like(target, dtype=np.float32)
    for t in range(contacts):
        cue = left[:, t]
        expected = target[:, t]
        taste_by_side = np.stack((-rewards[:, t], rewards[:, t]), axis=1).astype(np.float32)
        graph, h, side, taste, _, receipt = core.step(
            graph, h, cue, -cue, taste_by_side, key + t, arm=arm, view="own",
        )
        action_correct[:, t] = side == expected
        tastes[:, t] = taste
        edits[:, t] = receipt.edge_count
        writer_distance[:, t] = writer_delta(initial, graph)[0]
        probe_correct[:, t] = 0
        for sign in (-1, 1):
            probe = core.advance(graph, core.rest(graph), core.sense(sign * cue, -sign * cue), core.MOTOR_TICKS)
            probe_correct[:, t] += 0.5 * (core.motor(probe) * sign * expected > 0)
    return {
        "actual_choice": action_correct,
        "postwrite_probe": probe_correct,
        "tastes": tastes,
        "edge_count": edits,
        "writer_delta_l2": writer_distance,
    }


def summarize(records, plan):
    roots = plan["roots"]
    n_birth = len(roots)
    n_lives = plan["evaluation_lives"]
    rng = np.random.default_rng(plan["uncertainty_seed"])
    birth_draws = rng.integers(n_birth, size=(plan["bootstrap_draws"], n_birth, 1))
    life_draws = rng.integers(n_lives, size=(plan["bootstrap_draws"], n_birth, n_lives))
    results = {}
    scenarios = ("stable_simple", "novel_noise_switch")
    contrast_groups = {
        "repeated active minus one-contact active": ("repeat_act", "one_act"),
        "repeated active minus repeated primitive": ("repeat_act", "repeat_prim"),
    }
    conditions = {
        "repeat_act": "repeat|active",
        "repeat_prim": "repeat|primitive",
        "repeat_off": "repeat|none",
        "one_act": "one_contact|active",
        "one_prim": "one_contact|primitive",
        "one_off": "one_contact|none",
    }
    for scenario in scenarios:
        results[scenario] = {}
        for metric in ("actual_choice", "postwrite_probe", "writer_delta_l2", "edge_count"):
            affected = {
                name: np.stack([records[f"{root}|{scenario}|{arm}|{metric}"] for root in roots])
                for name, arm in conditions.items()
            }
            curves = {name: data.mean(axis=(0, 1)).tolist() for name, data in affected.items()}
            effects = {}
            if metric in ("actual_choice", "postwrite_probe"):
                for name in plan["paired_comparisons"]:
                    main, control = contrast_groups[name]
                    delta = affected[main].astype(float) - affected[control]
                    windows = {}
                    for first, last in plan["primary_windows"]:
                        by_life = delta[:, :, first - 1:last].mean(axis=2)
                        simulated = by_life[birth_draws, life_draws].mean(axis=(1, 2))
                        windows[f"{first}-{last}"] = {
                            "mean": float(by_life.mean()),
                            "ci95": np.quantile(simulated, [0.025, 0.975]).tolist(),
                            "by_root": by_life.mean(axis=1).tolist(),
                        }
                    effects[name] = windows
            results[scenario][metric] = {"conditions": curves, "paired_effects": effects}
    return results


def run(evidence, previous):
    plan_bytes = (evidence / "protocol.json").read_bytes()
    plan = json.loads(plan_bytes)
    lives = plan["evaluation_lives"]
    contacts = plan["evaluation_contacts"]
    output = {}
    provenance = {}
    for root in plan["roots"]:
        seed = plan["evaluation_root"] + root * 1000
        tapes = {
            "stable_simple": world(seed, lives, contacts, 1, 0.0, None),
            "novel_noise_switch": world(
                seed + 1, lives, contacts, plan["novel_pairs"], plan["novel_noise"], plan["novel_switch_contact"],
            ),
        }
        for scenario, tape in tapes.items():
            for field, values in zip(("left", "target", "signed_delivery"), tape, strict=True):
                output[f"{root}|{scenario}|world|{field}"] = values
        paths = {"repeat": evidence / f"bootstrap-repeat-{root}.npz",
                 "one_contact": previous / f"bootstrap-own-{root}.npz"}
        provenance[str(root)] = {}
        for bootstrap, path in paths.items():
            provenance[str(root)][bootstrap] = sha256(path.read_bytes()).hexdigest()
            with np.load(path, allow_pickle=False) as model:
                birth = core.Graph(model["w"], model["b"], model["mask"])
            for scenario, tape in tapes.items():
                for arm in plan["arms"]:
                    results = deploy(birth, tape, seed + 100, arm)
                    for metric, values in results.items():
                        output[f"{root}|{scenario}|{bootstrap}|{arm}|{metric}"] = values
        print(f"Completed held-out lives for root {root}", flush=True)
    raw = evidence / "repeat_traces.npz"
    np.savez_compressed(raw, **output)
    report = {
        "protocol_sha256": sha256(plan_bytes).hexdigest(),
        "core_sha256": sha256(Path(core.__file__).read_bytes()).hexdigest(),
        "traces_sha256": sha256(raw.read_bytes()).hexdigest(),
        "checkpoints": provenance,
        "results": summarize(output, plan),
    }
    (evidence / "repeat_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--previous", type=Path, default=EVIDENCE.parent / "core")
    args = parser.parse_args()
    run(args.evidence, args.previous)


if __name__ == "__main__":
    main()
