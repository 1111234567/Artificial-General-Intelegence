"""Train learned fast banks and sparse motor dynamics from terminal reward alone."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from experiments.pulse_game.twostep_distinct import DistinctFruitTape, fruit_score, make_distinct_tape
from experiments.pulse_game.twostep_eval import paired
from experiments.pulse_game.twostep_latent import LatentPolicy, LatentTwoStep, RAW_SIZE
from experiments.pulse_game.twostep_latent_train import save, sequence
from experiments.pulse_game.twostep_neural import PULSES
from experiments.pulse_game.twostep_neural_train import DEVELOPMENT, POLICY_SEED, _train_tape
from experiments.pulse_game.twostep_reward import RewardingProbeWorld, discounted_life_returns, terminal_returns
from experiments.pulse_game.twostep_world import ENTRANCE_RECEIPT, EPISODE_TICKS, ROOM_ID, TwoStepTape, TwoStepWorld, make_tape
from experiments.pulse_game.world import ACTIVE, CONTACT_ID, REWARD_NEG, REWARD_POS


@dataclass
class LifeTrajectories:
    observations: Tensor
    episodes: Tensor
    motor_initial: Tensor
    actions: Tensor
    logp: Tensor
    terminal_return: Tensor
    temperature: float
    success: float
    probe_success: float


@torch.no_grad()
def sample(
    model: LatentTwoStep, tape: TwoStepTape, *, seed: int, temperature: float, world: TwoStepWorld | None = None,
    erase_stage_tags: bool = False, terminal_only: bool = False, positive_actual: bool = False, future_discount: float = 0.0,
) -> LifeTrajectories:
    policy = LatentPolicy(model, seed=seed, erase_stage_tags=erase_stage_tags)
    policy.reset(tape.n)
    if world is None:
        world = RewardingProbeWorld(tape)
    rng = torch.Generator().manual_seed(seed + 14032)
    shape = (tape.n, tape.episodes, EPISODE_TICKS)
    frames = np.empty((*shape, RAW_SIZE), dtype=np.float32)
    indices = np.empty(shape, dtype=np.int64)
    logp = np.empty(shape, dtype=np.float32)
    starts = np.empty((*shape[:2], *policy.activity.shape[1:]), dtype=np.float32)
    obs = world.observe()
    while obs is not None:
        episode, tick = world.episode_index, world.tick_index
        if tick == 0:
            starts[:, episode] = policy.activity.numpy()
        distribution = (policy.logits(obs) / temperature).log_softmax(dim=1)
        actions = torch.multinomial(distribution.exp(), 1, generator=rng)
        move = actions[:, 0].numpy()
        frames[:, episode, tick] = policy.public_frame(obs)
        indices[:, episode, tick] = move
        logp[:, episode, tick] = distribution.gather(1, actions)[:, 0].numpy()
        obs = world.step(PULSES[move])
    public = torch.from_numpy(frames)
    fields = (ENTRANCE_RECEIPT, ROOM_ID, CONTACT_ID, slice(REWARD_POS, REWARD_NEG + 1))
    tokens = torch.stack([public[..., field].amax(dim=2) for field in fields], dim=2)
    shown = world.result().shown
    positive = (shown > 0).any(axis=-1)
    payoff = discounted_life_returns(terminal_returns(shown, only_contact=terminal_only, positive_only=positive_actual), future_discount)
    return LifeTrajectories(
        public, tokens, torch.from_numpy(starts), torch.from_numpy(indices), torch.from_numpy(logp),
        torch.from_numpy(payoff), temperature, float(positive.mean()), float(positive[:, tape.probe].mean()),
    )


def update(
    model: LatentTwoStep, data: LifeTrajectories, optimizer: torch.optim.Optimizer, *, epochs: int, batch: int, seed: int, probe_weight: float,
    omit_episodes: slice | None = None, event_projection: Callable[[Tensor], Tensor] | None = None,
) -> dict:
    rng = torch.Generator().manual_seed(seed)
    scores = []
    for _ in range(epochs):
        order = torch.randperm(len(data.observations), generator=rng)
        for rows in order.split(batch):
            public = data.observations[rows]
            events = data.episodes[rows]
            if event_projection is not None:
                events = event_projection(events)
            logits, values, _ = sequence(model, public, events, initial_activity=data.motor_initial[rows], auxiliary=False)
            live = public[..., ACTIVE].bool()
            if omit_episodes is not None:
                live[:, omit_episodes] = False
            weight = torch.ones_like(public[..., ACTIVE])
            weight[:, -1] += probe_weight
            weight = weight[live]
            lp = (logits / data.temperature).log_softmax(dim=-1)
            taken = lp.gather(-1, data.actions[rows].unsqueeze(-1)).squeeze(-1)
            log_ratio = (taken - data.logp[rows])[live]
            ratio = log_ratio.exp()
            actual = data.terminal_return[rows].unsqueeze(-1).expand_as(live)[live]
            values = values[live]
            advantage = actual - values.detach()
            advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(0.2)
            surrogate = torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)
            critic_error = (values - actual).square()
            entropy = -(lp[live].exp() * lp[live]).sum(dim=-1)
            loss = ((-surrogate + 0.25 * critic_error - 0.004 * entropy) * weight).sum() / weight.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scores.append((float(loss.detach()), float((ratio - 1 - log_ratio).mean().detach())))
    return {"loss": float(np.mean([v[0] for v in scores])), "kl": float(np.mean([v[1] for v in scores]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-start", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, default=345774)
    parser.add_argument("--distinct-fruit", action="store_true")
    parser.add_argument("--lives", type=int, default=96)
    parser.add_argument("--rounds", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--short-rounds", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--probe-weight", type=float, default=3.0)
    parser.add_argument("--erase-stage-tags", action="store_true")
    credit = parser.add_mutually_exclusive_group()
    credit.add_argument("--terminal-only", action="store_true", help="use zero return without an actual terminal receipt")
    credit.add_argument("--positive-actual", action="store_true", help="optimize observed positive endpoint receipts")
    parser.add_argument("--positive-warm-rounds", type=int, default=0)
    parser.add_argument("--future-discount", type=float, default=0.0)
    args = parser.parse_args()
    if args.positive_warm_rounds and (not args.terminal_only or not 0 < args.positive_warm_rounds <= args.rounds):
        parser.error("positive warm rounds precede the signed-receipt objective")
    torch.set_num_threads(2)
    torch.manual_seed(args.model_seed)
    model = LatentTwoStep()
    make = make_distinct_tape if args.distinct_fruit else make_tape
    development = make(DEVELOPMENT, rare_probability=0.2, distance_jitter=0.22, wind_scale=0.12)
    if set(DEVELOPMENT).intersection(range(args.train_start, args.train_start + args.lives * args.rounds)):
        parser.error("reinforcement lives overlap development")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    metadata = {
        "world_ranges": [], "development": [DEVELOPMENT.start, DEVELOPMENT.stop], "model_seed": args.model_seed,
        "distinct_fruit": args.distinct_fruit, "erase_stage_tags": args.erase_stage_tags,
        "reward_fit": {key: getattr(args, key) for key in ("lives", "rounds", "batch", "epochs", "lr", "temperature", "short_rounds", "probe_weight", "terminal_only", "positive_actual", "positive_warm_rounds", "future_discount")},
    }
    started = perf_counter()
    for step in range(args.rounds):
        lo = args.train_start + step * args.lives
        seeds = range(lo, lo + args.lives)
        tape = _train_tape(seeds, seed=args.model_seed + step, distinct=args.distinct_fruit)
        if step < args.short_rounds:
            tape = replace(tape, distances=tape.distances * 0.72)
        early_positive = step < args.positive_warm_rounds
        gamma = 0.0 if early_positive else args.future_discount
        trajectories = sample(
            model, tape, seed=POLICY_SEED + step, temperature=args.temperature, erase_stage_tags=args.erase_stage_tags,
            terminal_only=args.terminal_only, positive_actual=args.positive_actual or early_positive, future_discount=gamma,
        )
        stats = update(model, trajectories, optimizer, epochs=args.epochs, batch=args.batch, seed=691949 + step, probe_weight=args.probe_weight)
        metadata["world_ranges"].append([seeds.start, seeds.stop])
        output = {"round": step, "sampled_success": trajectories.success, "sampled_probe_success": trajectories.probe_success, "optimizer": stats, "seconds": perf_counter() - started}
        if step % 4 == 0 or step == args.rounds - 1:
            result, a, b = paired(lambda: LatentPolicy(model, seed=POLICY_SEED, erase_stage_tags=args.erase_stage_tags), development)
            output["development"] = result
            if isinstance(development, DistinctFruitTape):
                output["conditional_fruit"] = {"stay": fruit_score(development, a, reversed=False), "reversed": fruit_score(development, b, reversed=True)}
            save(args.output, model, metadata)
        print(json.dumps(output, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
