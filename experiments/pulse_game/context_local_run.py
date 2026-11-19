"""CPU tests of locally addressed rules in the frozen physical game."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import habitat_deploy, habitat_world, unified_audit, unified_confirm, unified_physical, unified_world
from .attractor_search import OUT
from .context_local import BoundNativeForager, LocalForager


ROOT_SEED = 84590166
ASSIGNMENT_SEED = 338803
LONG = "long_competing_run"
SOURCES = ("native", "shared", "local", "global_decay", "no_default")


def digest(x) -> str:
    return hashlib.sha256(Path(x).read_bytes()).hexdigest()


def phenotypes():
    for file in sorted((OUT / "unified-physical-evolution-v1").glob("*--full-*/frozen.json")):
        frozen = json.loads(file.read_text())
        settings = frozen["settings"]
        route_source = OUT / "unified-transfer-v1" / (settings["model"] + ".pt")
        if digest(route_source) != settings["model_sha"]:
            raise ValueError(f"frozen route parent changed for {file.parent.name}")
        front, route, movement = unified_audit.resources(OUT / "unified-transfer-v1", settings["model"])
        config = unified_physical.materialize(front, route, movement, np.asarray(frozen["best"]["genome"]), settings["mode"])
        yield file.parent.name, config, {"genotype": digest(file), "route": digest(route_source)}


def make(roots, contexts):
    protocol = unified_confirm.make(roots, LONG)
    if contexts == "separated":
        return protocol
    if contexts == "indistinguishable":
        protocol.arena.appearance[:] = np.sqrt(0.5)
        protocol.arena.tags[:] = np.sqrt(0.5)
        return protocol
    if contexts != "nearby":
        raise ValueError("unknown sensory scenario")
    settings = {**unified_audit.SETTINGS[LONG], "appearance": "nearby"}
    family = protocol.arena.relation
    altered = unified_world.make(roots * 4, force=family, nonce=19176531, **settings)
    for key in ("chemicals", "probability", "realized"):
        np.testing.assert_array_equal(getattr(altered.arena.menu, key), getattr(protocol.arena.menu, key))
    for key in ("wind", "distances", "ctx"):
        np.testing.assert_array_equal(getattr(altered.arena.menu.body, key), getattr(protocol.arena.menu.body, key))
    return altered


def agent(config, seeds, arm, *, record=False):
    front, rule, movement = config
    if arm == "native":
        return habitat_deploy.Forager(front, rule, seeds, start=-1, motor=movement)
    if arm == "banked_native":
        return BoundNativeForager(front, rule, seeds, motor=movement)
    return LocalForager(
        front, rule, seeds, motor=movement, slots=1 if arm == "shared" else 3,
        radius=0.35 if arm == "merge_nearby" else 0.09, inherit=arm != "no_default",
        local_retention=arm != "global_decay", record_rules=record,
    )


def score(protocol, result, solver, n):
    observations = unified_world.measures(protocol, result)
    first_side = observations["B_initial_side"].reshape(4, n)
    np.testing.assert_array_equal(first_side[0], first_side[1])
    np.testing.assert_array_equal(first_side[2], first_side[3])
    raw = {name: value.reshape(4, n).tolist() for name, value in observations.items()}
    summary = {name: {"same": float(np.mean(value[::3])), "conflict": float(np.mean(value[1:3]))} for name, value in raw.items() if not name.endswith("_side")}
    occurrence = int(protocol.arena.probe.sum())
    np.testing.assert_array_equal(result.observed_reward[protocol.arena.probe], 0)
    output = {"summary": summary, "rows": raw, "silent_probe_opportunities": occurrence}
    if isinstance(solver, LocalForager):
        slots = (solver.addresses.counts > 0).sum(axis=1)
        output["addresses"] = {"allocated_histogram": np.bincount(slots, minlength=4).tolist(), "overflow_pulses": int(solver.addresses.overflows.sum())}
        if solver.default_records:
            before_B, back_A = (solver.default_records[i] for i in (protocol.arena.start_test, protocol.checks["return_A"]))
            expected = protocol.arena.relation[:, 0]
            output["default_rule"] = {
                "old_correct_sign": float((before_B * expected > 0).mean()),
                "return_correct_sign": float((back_A * expected > 0).mean()),
                "absolute_old_mean": float(np.abs(before_B).mean()), "absolute_return_mean": float(np.abs(back_A).mean()),
            }
    return output


def fresh_context(protocol, *, opposing):
    """Retain the last actual anchor and query physics, with an unseen cue."""
    t = protocol.checks["return_A"]
    world = protocol.arena
    angles = np.linspace(0., np.pi / 2, 101)
    basis = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    similarity = (world.tags @ basis.T).max(1)
    novel = basis[similarity.argmin(1)].astype(np.float32)
    changed = world.appearance.copy()
    changed[:, t - 1:t + 1] = novel[:, None]
    menu = world.menu
    probabilities, signs = menu.probability.copy(), menu.realized.copy()
    if opposing:
        probabilities[:, t] = probabilities[:, t, ::-1]
        signs[:, t] = signs[:, t, ::-1]
    menu = replace(menu, probability=probabilities, realized=signs)
    altered = replace(world, menu=menu, appearance=changed, truths=probabilities.argmax(2))
    return replace(protocol, arena=altered)


def new_cue(config, protocol, seeds, n):
    reports, outputs = {}, []
    for label, contrary in (("novel_rule_matches_first", False), ("novel_rule_opposes_first", True)):
        changed = fresh_context(protocol, opposing=contrary)
        solver = agent(config, seeds, "local")
        result = habitat_world.rollout(solver, changed.arena, silent_probes=True)
        reports[label] = score(changed, result, solver, n)
        outputs.append(result)
    for field in ("contact_id", "contact_side", "contact_tick", "observed_reward"):
        np.testing.assert_array_equal(getattr(outputs[0], field), getattr(outputs[1], field))
    return {"actual_contacts_and_shown_payoff_equal_between_hidden_novel_rules": True, "arms": reports}


def same_body(config, protocol, seeds, n, challenger, *, reference="local"):
    a, b = (agent(config, seeds, arm) for arm in (reference, challenger))
    simulators = [habitat_world.Body(protocol.arena, silent_probes=True) for _ in range(2)]
    for brain in (a, b):
        brain.reset(len(seeds))
    observations = [body.observe() for body in simulators]
    # A root's matched family variants must have the same actuator assignment.
    # Otherwise differing drivers could alter an untrained B choice before a
    # single piece of B-specific evidence has been seen.
    drive_first = np.tile(np.random.default_rng(ASSIGNMENT_SEED).integers(2, size=n).astype(bool), 4)
    cutoff = protocol.checks["return_A"]
    while observations[0] is not None:
        trial, tick = simulators[0].trial_index, simulators[0].tick_index
        if trial < cutoff or (trial == cutoff and tick == 0):
            np.testing.assert_array_equal(*observations)
        commands = [brain.act(obs) for brain, obs in zip((a, b), observations, strict=True)]
        if trial < cutoff:
            pulse = np.where(drive_first[:, None], commands[0], commands[1])
            commands = [pulse, pulse]
        observations = [body.step(pulse) for body, pulse in zip(simulators, commands, strict=True)]
    results = [body.result() for body in simulators]
    for f in ("contact_id", "contact_side", "contact_tick", "observed_reward"):
        np.testing.assert_array_equal(*(getattr(r, f)[:, :cutoff] for r in results))
    return {
        "same_motion_and_earned_observations_before_A_query": True,
        reference: score(protocol, results[0], a, n), challenger: score(protocol, results[1], b, n),
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--worlds", type=int, default=128)
    cli.add_argument("--root-seed", type=int, default=ROOT_SEED)
    cli.add_argument("--model-filter")
    cli.add_argument("--controls", action="store_true")
    args = cli.parse_args()
    torch.set_num_threads(2)
    roots = np.random.default_rng(args.root_seed).choice(470_000_000, size=args.worlds, replace=False).astype(np.int64).tolist()
    root_heldout = OUT / "research-2026-11-19" / "unified-physical-sealed" / "complete.json"
    if root_heldout.exists():
        np.testing.assert_equal(len(set(roots).intersection(json.loads(root_heldout.read_text())["roots"])), 0)
    seeds = roots * 4
    args.out.mkdir(parents=True, exist_ok=True)
    for name, config, origin in phenotypes():
        if args.model_filter and args.model_filter not in name:
            continue
        opened = perf_counter()
        reports = {}
        scenes = ("separated", "nearby", "indistinguishable") if args.controls else ("separated",)
        for scene in scenes:
            protocol = make(roots, scene)
            variants = SOURCES if scene == "separated" else ("native", "local", "merge_nearby")
            reports[scene] = {}
            for variant in variants:
                solver = agent(config, seeds, variant, record=True)
                trials = habitat_world.rollout(solver, protocol.arena, silent_probes=True)
                reports[scene][variant] = score(protocol, trials, solver, len(roots))
        if args.controls:
            protocol = make(roots, "separated")
            reports["novel_cue"] = new_cue(config, protocol, seeds, len(roots))
            reports["same_body_native"] = same_body(config, protocol, seeds, len(roots), "native")
            reports["same_body_global_decay"] = same_body(config, protocol, seeds, len(roots), "global_decay")
        out = {"name": name, "root_seeds": roots, "root_seed": args.root_seed, "genotype": origin, "family_order": ["++", "+-", "-+", "--"],
               "only_public_receptor_and_taste_used": True, "default_angular_radius": 0.09, "shared_body_random_seed": ASSIGNMENT_SEED,
               "frozen_rule_retention": config[1]["retention"], "frozen_rule_updater": config[1]["updater"],
               "protocol": unified_audit.SETTINGS[LONG], "reports": reports, "seconds": perf_counter() - opened}
        result_path = args.out / f"{name}.json"
        provisional = result_path.with_suffix(".pending")
        provisional.write_text(json.dumps(out, allow_nan=False), encoding="utf-8")
        provisional.replace(result_path)
        digestible = {arm: round(r["summary"]["return_A"]["conflict"], 3) for arm, r in reports["separated"].items()}
        print(name, json.dumps(digestible), round(out["seconds"], 2), flush=True)


if __name__ == "__main__":
    main()
