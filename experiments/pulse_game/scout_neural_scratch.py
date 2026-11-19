"""Start a sparse scout motor controller with random weights and reward alone."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.pulse_game.scout_neural import NeuralScoutConfig, ScoutNetPolicy, ScoutPulseNet
from experiments.pulse_game.scout_neural_partition import table_filtered
from experiments.pulse_game.scout_neural_reward import sample, update
from experiments.pulse_game.scout_neural_train import DEVELOPMENT, POLICY_SEED, evaluate, load, measure, save, tape_for, used_world_seeds
from experiments.pulse_game.scout_world import ScoutTape, ScoutWorld, rollout_scout
from experiments.pulse_game.world import _deranged_rows


def _all_sensitive(tape: ScoutTape) -> ScoutTape:
    """Training curriculum with the same inputs, physics and reward channels."""
    sensitive = np.ones_like(tape.sensitive)
    new_map = tape.good_map.copy()
    new_map[:, 1] = 1 - new_map[:, 0]
    correct = new_map[np.arange(tape.n)[:, None], tape.ctx, tape.cue]
    return replace(tape, sensitive=sensitive, good_map=new_map, good_id=correct)


class ProbeApproachWorld(ScoutWorld):
    """Train from positions along a potential approach while keeping pulse control."""

    def __init__(self, tape: ScoutTape, strength: float) -> None:
        self.strength = strength
        super().__init__(tape)

    def _load_trial(self) -> None:
        super()._load_trial()
        side = 2 * self.tape.scout_side[:, self.trial_index] - 1
        self._position[:] = 0.30 * self.strength * side
        self._velocity[:] = 0.32 * self.strength * side


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--train-start", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lives", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=48)
    parser.add_argument("--nominal-rounds", type=int, default=8)
    parser.add_argument("--all-sensitive-rounds", type=int, default=0)
    parser.add_argument("--pose-hold-rounds", type=int, default=0)
    parser.add_argument("--pose-fade-rounds", type=int, default=0)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--symmetry", choices=("none", "physical", "physical_context"))
    parser.add_argument("--map-parity", type=int, choices=(0, 1))
    parser.add_argument("--outer-sham", action="store_true", help="shuffle PPO reward targets across independent lives")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.lives, args.rounds, args.epochs, args.batch, args.threads) < 1:
        parser.error("training sizes must be positive")
    if args.outer_sham and args.lives < 2:
        parser.error("an outer sham requires at least two independent lives")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.model_seed)
    if args.warm_start is None:
        model = ScoutPulseNet(NeuralScoutConfig(symmetry=args.symmetry or "none"))
        metadata = {"model_seed": args.model_seed, "teacher_lives": 0, "teacher_start": 0, "agent_dagger_starts": [], "reward_ranges": []}
        if args.map_parity is not None:
            metadata["map_parity"] = args.map_parity
            metadata["map_codes"] = []
            metadata["filtered_reward_seed_rosters"] = []
    else:
        model, metadata = load(args.warm_start)
        if metadata["teacher_lives"] or metadata["agent_dagger_starts"]:
            parser.error("the warm start must have no demonstration training")
        if args.map_parity is not None and args.map_parity != metadata.get("map_parity"):
            parser.error("a warm start must preserve the checkpoint's table split")
        if args.symmetry is not None and args.symmetry != model.config.symmetry:
            reparameterized = ScoutPulseNet(replace(model.config, symmetry=args.symmetry))
            reparameterized.load_state_dict(model.state_dict())
            model = reparameterized
    map_parity = metadata.get("map_parity")
    if map_parity is None:
        development = tape_for(DEVELOPMENT)
    else:
        development = table_filtered(1587000, 128, map_parity)
        metadata["world_seed_rosters"] = {"development": development.seeds}
    stages = metadata.setdefault("reward_stages", [])
    stages.append({
        "train_start": args.train_start, "lives": args.lives, "rounds": args.rounds, "nominal_rounds": args.nominal_rounds,
        "all_sensitive_rounds": args.all_sensitive_rounds, "temperature": args.temperature, "lr": args.lr,
        "epochs": args.epochs, "batch": args.batch,
        "pose_hold_rounds": args.pose_hold_rounds, "pose_fade_rounds": args.pose_fade_rounds,
        "symmetry": model.config.symmetry,
        "outer_sham": args.outer_sham,
    })
    used = used_world_seeds(metadata)
    stride = args.lives if map_parity is None else 4 * args.lives
    if used.intersection(range(args.train_start, args.train_start + args.rounds * stride)):
        parser.error("training lives overlap development or earlier training")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    start = perf_counter()
    for step in range(args.rounds):
        first = args.train_start + step * stride
        seeds = range(first, first + args.lives)
        nominal = step < args.nominal_rounds
        if map_parity is None:
            tape = tape_for(seeds, nominal=nominal)
        else:
            tape = table_filtered(first, args.lives, map_parity, nominal=nominal)
        if step < args.all_sensitive_rounds:
            tape = _all_sensitive(tape)
        strength = 0.0
        if step < args.pose_hold_rounds:
            strength = 1.0
        elif step < args.pose_hold_rounds + args.pose_fade_rounds:
            strength = 1.0 - (step - args.pose_hold_rounds + 1) / args.pose_fade_rounds
        world = ProbeApproachWorld(tape, strength) if strength else None
        experience = sample(model, tape, seed=POLICY_SEED + step, temperature=args.temperature, world=world)
        if args.outer_sham:
            peers = torch.from_numpy(_deranged_rows(tape.seeds, 157360 + step))
            peer_targets = experience.returns.reshape(tape.n, -1)[peers].reshape(experience.returns.shape)
            experience = replace(experience, returns=peer_targets)
        stats = update(model, experience, optimizer=optimizer, epochs=args.epochs, batch=args.batch, seed=310681 + step)
        if map_parity is None:
            metadata["reward_ranges"].append([seeds.start, seeds.stop])
        else:
            metadata["filtered_reward_seed_rosters"].append(tape.seeds)
            metadata["map_codes"] = sorted(set(metadata["map_codes"]).union(int(code) for code in tape.map_codes))
        record = {"round": step, "sampled_success": experience.success, "ppo": stats, "seconds": perf_counter() - start}
        if step % 4 == 0 or step == args.rounds - 1:
            result = rollout_scout(ScoutNetPolicy(model, seed=POLICY_SEED), development)
            record["development"] = measure(result, development)
            save(args.output, model, metadata)
        print(json.dumps(record), flush=True)
    print(json.dumps({"stage": "eval", "results": evaluate(model, development)}), flush=True)


if __name__ == "__main__":
    main()
