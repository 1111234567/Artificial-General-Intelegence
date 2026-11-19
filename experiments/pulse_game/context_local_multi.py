"""Out-of-count tests of small public context-address banks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np
import torch

from . import habitat_world as habitat, mixture_world as food, symmetry_world, world
from .context_local import LocalForager
from .context_local_run import phenotypes


MULTI_SEED = 5665029
WORLD_NONCE = 84214583
PAIRS = 12


@dataclass
class Multi:
    arena: habitat.MultiWorld
    novel: np.ndarray
    learned: np.ndarray
    returned: np.ndarray
    taught: np.ndarray


def make_multi(roots, count: int, *, collision: bool = False) -> Multi:
    if count not in (3, 4):
        raise ValueError("this extrapolation uses three or four habitats")
    n = len(roots)
    rules = np.empty((n, count), dtype=np.int8)
    tags = np.empty((n, count, 2), dtype=np.float32)
    orders = np.empty((n, count), dtype=np.int8)
    for i, seed in enumerate(roots):
        rule, tag, order = np.random.SeedSequence([seed, WORLD_NONCE]).spawn(3)
        rules[i] = np.random.default_rng(rule).choice([-1, 1], size=count)
        rng = np.random.default_rng(tag)
        left = rng.uniform(0.02, 0.36)
        span = rng.uniform(0.82, 1.15)
        theta = left + span * np.linspace(0, 1, count)
        tags[i] = np.stack((np.cos(theta), np.sin(theta)), axis=1)[rng.permutation(count)]
        orders[i] = np.random.default_rng(order).permutation(count)
    if collision:
        tags[:, -1] = tags[:, 0]
    teacher = []
    for c in range(count):
        source = [int(np.random.SeedSequence([seed, WORLD_NONCE, c, 819]).generate_state(1)[0]) for seed in roots]
        teacher.append(symmetry_world.make(source, pairs=PAIRS, probes=3, interleave=4, noise=.09, jitter=.12, family=rules[:, c], balanced_stream=True).menu)
    steps = 2 * PAIRS + 6
    length = count * steps
    body = world.make_tape(roots, trials=length, distance_jitter=.13, wind_scale=.07)
    # Base geometry has two plumbing context IDs. Body.observe replaces every
    # actual context flash with the physical vector from appearance below.
    body = replace(body, ctx=np.zeros_like(body.ctx))
    odors = np.empty((n, length, 2, food.DIM), dtype=np.float32)
    pay = np.empty((n, length, 2), dtype=np.float32)
    signed = np.empty((n, length, 2), dtype=np.int8)
    cues = np.empty((n, length, 2), dtype=np.float32)
    probe = np.zeros((n, length), dtype=bool)
    taught = np.zeros_like(probe)
    novel = np.empty((n, count), dtype=np.int64)
    learned = np.empty_like(novel)
    returned = np.empty_like(novel)
    for row in range(n):
        index = 0

        def copy(c, left, right, *, is_probe=False, is_teaching=False):
            nonlocal index
            source = teacher[c]
            destination = slice(index, index + right - left)
            odors[row, destination] = source.chemicals[row, left:right]
            pay[row, destination] = source.probability[row, left:right]
            signed[row, destination] = source.realized[row, left:right]
            cues[row, destination] = tags[row, c]
            if is_probe:
                probe[row, index + 1] = True
            if is_teaching:
                taught[row, destination] = True
            index += right - left
            return index - 1

        for c in range(count):
            novel[row, c] = copy(c, 2*PAIRS, 2*PAIRS + 2, is_probe=True)
            copy(c, 0, 2*PAIRS, is_teaching=True)
            learned[row, c] = copy(c, 2*PAIRS + 2, 2*PAIRS + 4, is_probe=True)
        for c in orders[row]:
            returned[row, c] = copy(c, 2*PAIRS + 4, 2*PAIRS + 6, is_probe=True)
        if index != length:
            raise AssertionError("nonmatching body and exposure lengths")
    menu = food.Foods(body, odors, pay, signed, np.empty(0), "local_address_count", 0)
    arena = habitat.MultiWorld(menu, tags, cues, rules, returned, probe, 0, pay.argmax(axis=2))
    return Multi(arena, novel, learned, returned, taught)


def test_model(config, episode: Multi, slots: int, *, inherit=True):
    front, rules, movement = config
    solver = LocalForager(front, rules, episode.arena.menu.body.seeds, motor=movement, slots=slots, inherit=inherit, record_rules=True)
    result = habitat.rollout(solver, episode.arena, silent_probes=True)
    np.testing.assert_array_equal(result.observed_reward[episode.arena.probe], 0)
    return result, solver


def grade(episode, result, solver):
    n, count = episode.returned.shape
    row = np.arange(n)[:, None]
    truth = episode.arena.truths
    choices = result.contact_side
    right = (choices == truth) & (choices >= 0)
    new, immediate, old = (right[row, stage] for stage in (episode.novel, episode.learned, episode.returned))
    actual = episode.arena.relation
    different_from_first = actual[:, 1:] != actual[:, :1]
    assigned = {index: state for index, state, _ in solver.history}
    sequence = np.stack([assigned[i] for i in range(right.shape[1])], axis=1)
    first_address, return_address = (sequence[row, stage] for stage in (episode.novel, episode.returned))
    collisions = np.stack([(return_address[:, a] == return_address[:, b]) for a in range(count) for b in range(a + 1, count)], axis=1).sum(axis=1)
    prior_correct = new[:, 1:]
    mean = lambda values: float(values.mean()) if values.size else None
    summary = {
        "novel_default_when_matches_first": mean(prior_correct[~different_from_first]),
        "novel_default_when_opposes_first": mean(prior_correct[different_from_first]),
        "initial_novel_opportunities": int(prior_correct.size),
        "initial_novel_matches_first_count": int((~different_from_first).sum()),
        "trained_query_choices": float(immediate.mean()),
        "return_correct": float(old.mean()),
        "return_correct_all_habitats": float(old.all(axis=1).mean()),
        "return_correct_first_habitat": float(old[:, 0].mean()),
        "return_correct_first_if_last_opposes": mean(old[:, 0][different_from_first[:, -1]]),
        "first_last_opposes_opportunities": int(different_from_first[:, -1].sum()),
        "return_opportunities": int(old.size),
        "return_contact_rate": float((choices[row, episode.returned] >= 0).mean()),
        "teaching_contact_rate": float((choices[episode.taught] >= 0).mean()),
        "allocated_histogram": np.bincount((solver.addresses.counts > 0).sum(axis=1), minlength=count + 1).tolist(),
        "overflow_pulses": int(solver.addresses.overflows.sum()),
        "return_address_matches_first_assignment": float((return_address == first_address).mean()),
        "lives_with_colliding_return_addresses": int((collisions > 0).sum()),
        "mean_habitat_pairs_sharing_address_on_return": float(collisions.mean()),
    }
    return {"summary": summary, "rows": {
        "returned_correct": old.tolist(), "new_correct": new.tolist(), "return_address": return_address.tolist(),
        "initial_address": first_address.tolist(), "private_families_scoring_only": actual.tolist(),
    }}


def check_private_novel_rule(config, episode):
    row = np.arange(episode.novel.shape[0])[:, None]
    indices = episode.novel[:, 1:]
    w = episode.arena
    prob, realized = w.menu.probability.copy(), w.menu.realized.copy()
    prob[row, indices] = prob[row, indices][:, :, ::-1]
    realized[row, indices] = realized[row, indices][:, :, ::-1]
    altered = replace(episode, arena=replace(w, menu=replace(w.menu, probability=prob, realized=realized), truths=prob.argmax(2)))
    a, _ = test_model(config, episode, 6)
    b, _ = test_model(config, altered, 6)
    for field in ("contact_id", "contact_side", "contact_tick", "observed_reward"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    score = lambda e, r: float((r.contact_side[row, indices] == e.arena.truths[row, indices]).mean())
    return {"private_unobserved_flip_keeps_entire_action_and_shown_payoff_history": True,
            "original_truth_accuracy": score(episode, a), "flipped_silent_truth_accuracy": score(altered, b),
            "visible_cue_does_not_report_the_hidden_novel_rule": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=192)
    parser.add_argument("--root-seed", type=int, default=MULTI_SEED)
    parser.add_argument("--model-filter", default="1373")
    args = parser.parse_args()
    torch.set_num_threads(2)
    name, config, origin = next(item for item in phenotypes() if args.model_filter in item[0])
    roots = np.random.default_rng(args.root_seed).choice(470_000_000, size=args.worlds, replace=False).astype(np.int64).tolist()
    result = {}
    for count in (3, 4):
        scene = make_multi(roots, count)
        result[f"{count}_visible_habitats"] = {}
        for slots in (2, 3, 4, 6):
            record, solver = test_model(config, scene, slots)
            result[f"{count}_visible_habitats"][f"{slots}_slots"] = grade(scene, record, solver)
        record, solver = test_model(config, scene, 6, inherit=False)
        result[f"{count}_visible_habitats"]["6_slots_no_default"] = grade(scene, record, solver)
        if count == 4:
            aliased = make_multi(roots, count, collision=True)
            np.testing.assert_array_equal(scene.arena.menu.chemicals, aliased.arena.menu.chemicals)
            np.testing.assert_array_equal(scene.arena.menu.realized, aliased.arena.menu.realized)
            record, solver = test_model(config, aliased, 6)
            result["4_habitats_2_cues_identical"] = grade(aliased, record, solver)
            result["new_rule_twins"] = check_private_novel_rule(config, scene)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {"model": name, "genotype": origin, "root_seed": args.root_seed, "roots": roots, "pairs_per_habitat": PAIRS, "arms": result}
    args.out.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
