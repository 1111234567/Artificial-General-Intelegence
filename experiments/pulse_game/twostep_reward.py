"""Fit two-step motor logits with actual terminal outcomes and no action labels."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.pulse_game.scout_neural import NeuralScoutConfig
from experiments.pulse_game.scout_neural_reward import Sampled, update
from experiments.pulse_game.twostep_distinct import DistinctFruitTape, make_distinct_tape
from experiments.pulse_game.twostep_eval import paired
from experiments.pulse_game.twostep_neural import INPUT_SIZE, PULSES, RouteFeatures, TwoStepNetPolicy, TwoStepPulseNet
from experiments.pulse_game.twostep_neural_train import DEVELOPMENT, POLICY_SEED, _train_tape, load, save
from experiments.pulse_game.twostep_world import EPISODE_TICKS, TwoStepTape, TwoStepWorld, make_tape
from experiments.pulse_game.world import _deranged_rows


class RewardingProbeWorld(TwoStepWorld):
    """A training world which also pays after the final episode's two actions."""

    def _feedback(self, receipt: np.ndarray) -> np.ndarray:
        if self.stage == 1 and self.episode_index == self.tape.probe:
            return self._reward[:, self.episode_index] * receipt
        return super()._feedback(receipt)


def terminal_returns(shown: np.ndarray, *, only_contact: bool = False, positive_only: bool = False) -> np.ndarray:
    """Score actual receipt signs, optionally leaving uncompleted episodes at zero."""
    signed = shown.sum(axis=-1, dtype=np.float32)
    if positive_only:
        return (signed > 0).astype(np.float32)
    return signed if only_contact else np.where(signed > 0, 1.0, -1.0).astype(np.float32)


def discounted_life_returns(terminal: np.ndarray, gamma: float) -> np.ndarray:
    """Credit future actual receipts to earlier decisions without artificial outcomes."""
    if not 0 <= gamma < 1:
        raise ValueError("episode discount must be in [0, 1)")
    if gamma == 0:
        return terminal
    discounted = terminal.copy()
    for episode in range(discounted.shape[1] - 2, -1, -1):
        discounted[:, episode] += gamma * discounted[:, episode + 1]
    return discounted * (1 - gamma)


@torch.no_grad()
def sample(
    model: TwoStepPulseNet, tape: TwoStepTape, *, seed: int, temperature: float, extra_probe: int = 0, outer_sham: bool = False,
) -> tuple[Sampled, float]:
    world = RewardingProbeWorld(tape)
    features = RouteFeatures(tape.n, seed=seed, mean_visits=isinstance(tape, DistinctFruitTape))
    state = model.initial(tape.n)
    matrix = model.prepare()
    random = torch.Generator().manual_seed(seed + 306177)
    shape = tape.n, tape.episodes, EPISODE_TICKS
    x = np.empty((*shape, INPUT_SIZE), dtype=np.float32)
    choices = np.empty(shape, dtype=np.int64)
    logp = np.empty(shape, dtype=np.float32)
    state_shape = state.shape[1:]
    initial = np.empty((tape.n, tape.episodes, *state_shape), dtype=np.float32)
    obs = world.observe()
    while obs is not None:
        episode, tick = world.episode_index, world.tick_index
        if tick == 0:
            initial[:, episode] = state.numpy()
        raw = features.observe(obs)
        logits, state = model(torch.from_numpy(raw), state, matrix)
        lp = (logits / temperature).log_softmax(dim=-1)
        sampled = torch.multinomial(lp.exp(), 1, generator=random)
        chosen = sampled.squeeze(1).numpy()
        x[:, episode, tick] = raw
        choices[:, episode, tick] = chosen
        logp[:, episode, tick] = lp.gather(1, sampled).squeeze(1).numpy()
        obs = world.step(PULSES[chosen])

    result = world.result()
    actual = (result.shown > 0).any(axis=-1)
    terminal = terminal_returns(result.shown)
    if outer_sham:
        terminal = terminal[_deranged_rows(tape.seeds, seed + 42105)]
    returns = np.broadcast_to(terminal[..., None], shape).copy()
    cases = np.arange(tape.n * tape.episodes)
    if extra_probe:
        cases = np.concatenate((cases, np.tile(cases.reshape(tape.n, tape.episodes)[:, tape.probe], extra_probe)))
    return Sampled(
        torch.from_numpy(x.reshape(-1, EPISODE_TICKS, INPUT_SIZE)[cases]),
        torch.from_numpy(choices.reshape(-1, EPISODE_TICKS)[cases]),
        torch.from_numpy(initial.reshape(-1, *state_shape)[cases]),
        torch.from_numpy(logp.reshape(-1, EPISODE_TICKS)[cases]),
        torch.from_numpy(returns.reshape(-1, EPISODE_TICKS)[cases]),
        temperature,
        float(actual.mean()),
    ), float(actual[:, tape.probe].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-start", required=True, type=int)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--model-seed", type=int, default=503573)
    parser.add_argument("--distinct-fruit", action="store_true")
    parser.add_argument("--symmetry", choices=("none", "physical", "physical_context", "full"), default="physical_context")
    parser.add_argument("--lives", type=int, default=96)
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--short-rounds", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--extra-probe", type=int, default=3)
    parser.add_argument("--outer-sham", action="store_true")
    parser.add_argument("--all-rare", action="store_true", help="apply the rare transition probability in every life")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.lives, args.rounds, args.epochs, args.batch, args.threads) < 1:
        parser.error("training sizes must be positive")
    if args.outer_sham and args.lives < 2:
        parser.error("outer sham requires multiple independent scores")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.model_seed)
    if args.warm_start is None:
        model = TwoStepPulseNet(NeuralScoutConfig(symmetry=args.symmetry))
        metadata = {"world_ranges": [], "development": [DEVELOPMENT.start, DEVELOPMENT.stop], "model_seed": args.model_seed, "distinct_fruit": args.distinct_fruit}
    else:
        model, metadata = load(args.warm_start)
        if args.distinct_fruit and not metadata.get("distinct_fruit", False):
            parser.error("warm-start environment must use the same fruit protocol")
    distinct = metadata.get("distinct_fruit", False)
    used = set(range(*metadata["development"]))
    for start, stop in metadata["world_ranges"]:
        used.update(range(start, stop))
    if used.intersection(range(args.train_start, args.train_start + args.rounds * args.lives)):
        parser.error("new reinforcement lives overlap prior training or development")
    configs = metadata.setdefault("reward_fit", [])
    configs.append({
        "train_start": args.train_start, "rounds": args.rounds, "short_rounds": args.short_rounds, "lives": args.lives,
        "temperature": args.temperature, "extra_probe": args.extra_probe, "lr": args.lr, "epochs": args.epochs, "batch": args.batch,
        "outer_sham": args.outer_sham,
        "all_rare": args.all_rare,
    })
    make = make_distinct_tape if distinct else make_tape
    dev = make(DEVELOPMENT, rare_probability=0.2, distance_jitter=0.22, wind_scale=0.12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    begin = perf_counter()
    for step in range(args.rounds):
        lo = args.train_start + step * args.lives
        seeds = range(lo, lo + args.lives)
        training = _train_tape(seeds, seed=args.model_seed + step, distinct=distinct, mix_rare=not args.all_rare)
        if step < args.short_rounds:
            training = replace(training, distances=training.distances * 0.72)
        experience, probe = sample(
            model, training, seed=POLICY_SEED + step, temperature=args.temperature, extra_probe=args.extra_probe, outer_sham=args.outer_sham,
        )
        stats = update(model, experience, optimizer=optimizer, epochs=args.epochs, batch=args.batch, seed=531045 + step)
        metadata["world_ranges"].append([seeds.start, seeds.stop])
        summary = {"step": step, "sampled_positive_rate": experience.success, "sampled_probe_positive_rate": probe, "optimizer": stats, "seconds": perf_counter() - begin}
        if step % 4 == 0 or step == args.rounds - 1:
            check, _, _ = paired(lambda: TwoStepNetPolicy(model, seed=POLICY_SEED, mean_visits=distinct), dev)
            summary["development"] = check
            save(args.output, model, metadata)
        print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
