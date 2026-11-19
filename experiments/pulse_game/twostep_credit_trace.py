"""Measure movement on held-out probes with no new training signal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pulse_game.twostep_latent import LatentPolicy
from experiments.pulse_game.twostep_latent_train import load as load_core
from experiments.pulse_game.twostep_rewire import make_rewire_tape, run
from experiments.pulse_game.twostep_untyped import UntypedPolicy, UntypedWorld, load as load_untyped
from experiments.pulse_game.twostep_world import make_tape, rollout
from experiments.pulse_game.world import ACTIVE, POSITION, TICKS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--lives", type=int, default=64)
    parser.add_argument("--policy-seed", type=int, default=801227)
    parser.add_argument("--untyped", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    seeds = range(args.seed_start, args.seed_start + args.lives)
    model, info = (load_untyped if args.untyped else load_core)(args.model)
    seen = set(range(*info["development"]))
    for start, stop in info["world_ranges"]:
        seen.update(range(start, stop))
    if seen.intersection(seeds):
        parser.error("probe worlds must be independent")
    build = make_rewire_tape if args.untyped else make_tape
    tape = build(seeds, rare_probability=0.2, distance_jitter=0.22, wind_scale=0.12)
    if args.untyped:
        result = run(lambda: UntypedPolicy(model, seed=args.policy_seed), tape, world_type=UntypedWorld, record=True)
        old = tape.ordinary
    else:
        result = rollout(LatentPolicy(model, seed=args.policy_seed, erase_stage_tags=info.get("erase_stage_tags", False)), tape, record=True)
        old = tape.acquisition
    live = result.obs[:, tape.probe, :TICKS, ACTIVE] > 0
    pulses = result.pulses[:, tape.probe, :TICKS].any(axis=-1) & live
    position = np.abs(result.obs[:, tape.probe, :TICKS, POSITION])
    print(json.dumps({
        "world_seeds": [seeds.start, seeds.stop - 1], "policy_seed": args.policy_seed,
        "active_probe_ticks": int(live.sum()), "probe_ticks_with_a_motor_pulse": int(pulses.sum()),
        "probe_lives_with_a_motor_pulse": int(pulses.any(axis=1).sum()), "actual_probe_entrance_contacts": int((result.probe_entry >= 0).sum()),
        "max_probe_abs_position": float(position.max()), "nearest_possible_probe_endpoint": float(tape.distances[:, tape.probe, 0].min()),
        "actual_old_entry_contacts": int((result.entry_id[:, :old] >= 0).sum()), "actual_old_fruit_contacts": int((result.fruit_id[:, :old] >= 0).sum()),
        "old_opportunities": tape.n * old,
    }, indent=2))


if __name__ == "__main__":
    main()
