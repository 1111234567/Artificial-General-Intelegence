"""Learn to retain or replace latent route associations from terminal reward."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import Tensor, nn

from experiments.pulse_game.scout_neural import NeuralScoutConfig
from experiments.pulse_game.twostep_latent import Associations, LatentPolicy, LatentTwoStep, LearnedMemory, MemoryConfig, MemoryState
from experiments.pulse_game.twostep_latent_reward import sample, update
from experiments.pulse_game.twostep_latent_train import load as load_core, save
from experiments.pulse_game.twostep_neural_train import POLICY_SEED
from experiments.pulse_game.twostep_rewire import RewireTape, RewireWorld, evaluate, make_rewire_tape


DEVELOPMENT = range(3180000, 3180128)


class RewireMemory(LearnedMemory):
    """Learn a lower bound on the actual event-conditioned route write step."""

    def __init__(self, config: MemoryConfig, *, adaptive: bool) -> None:
        super().__init__(config)
        self.adaptive = adaptive
        self.route_floor_logit = nn.Parameter(torch.tensor(-2.0))

    def update(
        self, state: MemoryState, assoc: Associations, entry: Tensor, room: Tensor, fruit: Tensor, sign: Tensor,
    ) -> MemoryState:
        default = super().update(state, assoc, entry, room, fruit, sign)
        if not self.adaptive:
            return default
        live_entry = entry * room.sum(dim=1, keepdim=True)
        count = (default.entrance_counts * live_entry).sum(dim=1).clamp_min(1)
        gain = 1 + self.route_floor_logit.sigmoid() * (count - 1)
        routes = state.routes + (default.routes - state.routes) * gain[:, None, None]
        return replace(default, routes=routes)


class RewireLatent(LatentTwoStep):
    def __init__(
        self, *, adaptive: bool, motor: NeuralScoutConfig = NeuralScoutConfig(symmetry="physical"), memory: MemoryConfig = MemoryConfig(),
    ) -> None:
        super().__init__(motor, memory)
        initial_bank = self.memory
        self.memory = RewireMemory(memory, adaptive=adaptive)
        result = self.memory.load_state_dict(initial_bank.state_dict(), strict=False)
        if result.unexpected_keys or result.missing_keys != ["route_floor_logit"]:
            raise RuntimeError("new route bank does not preserve the core encoder")


class RewardingRewireWorld(RewireWorld):
    def _feedback(self, receipt: np.ndarray) -> np.ndarray:
        if self.stage == 1 and self.episode_index == self.tape.probe:
            return self._reward[:, self.episode_index] * receipt
        return super()._feedback(receipt)


def training_tape(seeds: range, *, seed: int, rare_probability: float = 0.2) -> RewireTape:
    base = make_rewire_tape(seeds, rare_probability=rare_probability, distance_jitter=0.22, wind_scale=0.12)
    rng = np.random.default_rng(seed)
    route = rng.integers(2, size=base.n).astype(bool)
    volume = rng.integers(2, size=base.n).astype(bool)
    return replace(base, reverse_route=route, final_good_room=base.initial_good_room ^ volume)


def load(path: Path) -> tuple[RewireLatent, dict]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    meta = data["metadata"]
    model = RewireLatent(adaptive=meta["adaptive"], motor=NeuralScoutConfig(**data["motor"]), memory=MemoryConfig(**data["memory"]))
    model.load_state_dict(data["model"])
    return model.eval(), meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    fit = actions.add_parser("train")
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--train-start", type=int, required=True)
    fit.add_argument("--model-seed", type=int, default=477312)
    fit.add_argument("--warm-start", type=Path)
    fit.add_argument("--cumulative", action="store_true")
    fit.add_argument("--show-stage-tags", action="store_true")
    fit.add_argument("--lives", type=int, default=96)
    fit.add_argument("--rounds", type=int, default=112)
    fit.add_argument("--batch", type=int, default=8)
    fit.add_argument("--epochs", type=int, default=2)
    fit.add_argument("--lr", type=float, default=0.001)
    fit.add_argument("--probe-weight", type=float, default=6)
    fit.add_argument("--rare", type=float, default=0.2)
    fit.add_argument("--stable-rounds", type=int, default=0)
    fit.add_argument("--deterministic-rounds", type=int, default=0)
    fit.add_argument("--short-rounds", type=int, default=0)
    test = actions.add_parser("confirm")
    test.add_argument("--model", type=Path, required=True)
    test.add_argument("--seed-start", type=int, required=True)
    test.add_argument("--lives", type=int, default=256)
    test.add_argument("--policy-seed", type=int, default=248416)
    test.add_argument("--post-switch-attempts", type=int)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.action == "train":
        torch.manual_seed(args.model_seed)
        model = RewireLatent(adaptive=not args.cumulative)
        parent_ranges = []
        if args.warm_start:
            parent, meta = load_core(args.warm_start)
            result = model.load_state_dict(parent.state_dict(), strict=False)
            if result.unexpected_keys or result.missing_keys != ["memory.route_floor_logit"]:
                parser.error("warm checkpoint must match the learned-address actor format")
            parent_ranges = [*meta["world_ranges"], meta["development"]]
        forbidden = set(DEVELOPMENT)
        for lo, hi in parent_ranges:
            forbidden.update(range(lo, hi))
        if forbidden.intersection(range(args.train_start, args.train_start + args.lives * args.rounds)):
            parser.error("training overlaps development or earlier experience")
        dev = make_rewire_tape(DEVELOPMENT, rare_probability=args.rare, distance_jitter=0.22, wind_scale=0.12)
        pair = lambda: LatentPolicy(model, seed=POLICY_SEED)
        metadata = {
            "world_ranges": parent_ranges.copy(), "development": [DEVELOPMENT.start, DEVELOPMENT.stop],
            "model_seed": args.model_seed, "erase_stage_tags": not args.show_stage_tags, "adaptive": not args.cumulative,
            "warm_start_file": args.warm_start.name if args.warm_start else None,
            "reward_fit": {key: getattr(args, key) for key in ("train_start", "lives", "rounds", "batch", "epochs", "lr", "probe_weight", "rare", "stable_rounds", "deterministic_rounds", "short_rounds")},
        }
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
        begin = perf_counter()
        for iteration in range(args.rounds):
            start = args.train_start + iteration * args.lives
            rare = 0.0 if iteration < args.deterministic_rounds else args.rare
            tape = training_tape(range(start, start + args.lives), seed=args.model_seed + iteration, rare_probability=rare)
            if iteration < args.stable_rounds:
                tape = replace(tape, reverse_route=np.zeros(tape.n, dtype=bool), final_good_room=tape.initial_good_room)
            if iteration < args.short_rounds:
                tape = replace(tape, distances=tape.distances * 0.72)
            world = RewardingRewireWorld(tape, erase_stage_tags=not args.show_stage_tags)
            experience = sample(model, tape, seed=POLICY_SEED + iteration, temperature=1.0, world=world)
            stats = update(
                model, experience, optimizer, epochs=args.epochs, batch=args.batch, seed=659906 + iteration,
                probe_weight=args.probe_weight, omit_episodes=slice(tape.ordinary, tape.acquisition),
            )
            metadata["world_ranges"].append([start, start + args.lives])
            report = {"iteration": iteration, "sampled_success": experience.success, "sampled_probe_success": experience.probe_success,
                      "optimizer": stats, "seconds": perf_counter() - begin, "route_update_floor": float(model.memory.route_floor_logit.sigmoid().detach())}
            if iteration % 8 == 0 or iteration == args.rounds - 1:
                report["development"] = evaluate(pair, dev, erase_stage_tags=not args.show_stage_tags)
                save(args.output, model, metadata)
            print(json.dumps(report, allow_nan=False), flush=True)
        return

    model, metadata = load(args.model)
    used = set(range(*metadata["development"]))
    for a, b in metadata["world_ranges"]:
        used.update(range(a, b))
    seeds = range(args.seed_start, args.seed_start + args.lives)
    if used.intersection(seeds):
        parser.error("confirmed lives must be held out")
    pair = lambda: LatentPolicy(model, seed=args.policy_seed)
    result = {}
    for rare in (0.0, 0.2):
        tape = make_rewire_tape(seeds, rare_probability=rare, distance_jitter=0.22, wind_scale=0.12)
        if args.post_switch_attempts is not None:
            if not 1 <= args.post_switch_attempts <= tape.survey:
                parser.error("post-switch attempts must fit the route survey")
            tape = replace(tape, switch_at=np.full(tape.n, tape.acquisition - args.post_switch_attempts))
        erase = metadata["erase_stage_tags"]
        result[f"rare_{rare}"] = {
            "intact": evaluate(pair, tape, erase_stage_tags=erase),
            "hide_survey_rooms": evaluate(pair, tape, erase_stage_tags=erase, hide_survey_room=True),
            "hide_value_visit_rooms": evaluate(pair, tape, erase_stage_tags=erase, hide_visit_room=True),
        }
    print(json.dumps({"world_seeds": [seeds.start, seeds.stop - 1], "policy_seed": args.policy_seed, "stage_tags_erased": metadata["erase_stage_tags"], "post_switch_attempts": args.post_switch_attempts, "conditions": result}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
