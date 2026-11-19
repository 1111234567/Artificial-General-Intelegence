"""Match the survey and visit bodies across all four transition/value cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from experiments.pulse_game.twostep_latent import LatentPolicy
from experiments.pulse_game.twostep_rewire import CASES, CASE_NAMES, RewireTape, RewireWorld, condition, make_rewire_tape, profile
from experiments.pulse_game.twostep_rewire_train import load as load_typed
from experiments.pulse_game.twostep_untyped import UntypedPolicy, UntypedWorld, load as load_untyped
from experiments.pulse_game.twostep_world import OBS_SIZE, ROOM_ID
from experiments.pulse_game.world import Policy, REWARD_NEG, REWARD_POS


def matched_four(tape: RewireTape, factory: Callable[[], Policy], build: Callable[[RewireTape], RewireWorld], *, seed: int) -> dict:
    selected = np.random.default_rng(seed).integers(4, size=tape.n)
    cases = [condition(tape, route=route, value=value) for route, value in CASES]
    worlds = [build(scene) for scene in cases]
    agents = [factory() for _ in cases]
    for policy in agents:
        policy.reset(tape.n)
    obs = [world.observe() for world in worlds]
    outcomes = [*range(ROOM_ID.start, ROOM_ID.stop), REWARD_POS, REWARD_NEG]
    fixed = [i for i in range(OBS_SIZE) if i not in outcomes]
    row = np.arange(tape.n)
    while obs[0] is not None:
        t = worlds[0].episode_index
        k = worlds[0].tick_index
        if t < tape.probe:
            for counterfactual in obs[1:]:
                np.testing.assert_array_equal(obs[0][:, fixed], counterfactual[:, fixed])
        if t == tape.probe and k == 0:
            for counterfactual in obs[1:]:
                np.testing.assert_array_equal(obs[0], counterfactual)
        command = [policy.act(frame) for policy, frame in zip(agents, obs)]
        if t < tape.probe:
            shared = np.stack(command)[selected, row]
            command = [shared] * 4
        obs = [world.step(move) for world, move in zip(worlds, command)]

    result = [world.result() for world in worlds]
    for alternative in result[1:]:
        for field in ("entry_id", "entry_side", "entry_tick", "fruit_id", "fruit_side", "fruit_tick"):
            np.testing.assert_array_equal(getattr(result[0], field)[:, :tape.probe], getattr(alternative, field)[:, :tape.probe])
    correct = np.stack([real.effective_entry[:, tape.probe] for real in result])
    return {
        "same_sampled_motion_before_probe": True,
        "same_first_probe_raw_observation": True,
        "n_pairs": tape.n,
        "sharing_init_counts": np.bincount(selected, minlength=4).tolist(),
        "conditions": {name: profile(scenario, data) for name, scenario, data in zip(CASE_NAMES, cases, result)},
        "correct_opportunities": int(correct.sum()),
        "opportunities": correct.size,
        "correct_all_four": int(correct.all(axis=0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--untyped", action="store_true")
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--lives", type=int, default=256)
    parser.add_argument("--policy-seed", type=int, default=917615)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, metadata = (load_untyped if args.untyped else load_typed)(args.model)
    seen = set(range(*metadata["development"]))
    for lo, hi in metadata["world_ranges"]:
        seen.update(range(lo, hi))
    seeded = range(args.seed_start, args.seed_start + args.lives)
    if seen.intersection(seeded):
        parser.error("provide independent worlds")
    actor = UntypedPolicy if args.untyped else LatentPolicy
    settings = {} if args.untyped else {"erase_stage_tags": metadata["erase_stage_tags"]}
    world_type = UntypedWorld if args.untyped else RewireWorld
    results = {}
    for rare in (0.0, 0.2):
        tape = make_rewire_tape(seeded, rare_probability=rare, distance_jitter=0.22, wind_scale=0.12)
        results[f"rare_{rare}"] = matched_four(tape, lambda: actor(model, seed=args.policy_seed), lambda s: world_type(s, **settings), seed=args.policy_seed + 13)
    print(json.dumps({"world_seeds": [seeded.start, seeded.stop - 1], "policy_seed": args.policy_seed, "conditions": results}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
