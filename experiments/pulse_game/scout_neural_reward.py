"""Fine-tune neural scout motor weights using sampled actions and true rewards."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from experiments.pulse_game.scout_neural import INPUT_SIZE, ReceiptFeatures, ScoutNetPolicy, ScoutPulseNet, _PULSES
from experiments.pulse_game.scout_neural_train import DEVELOPMENT, POLICY_SEED, TRAIN_REWARD, evaluate, load, measure, save, tape_for
from experiments.pulse_game.scout_world import ScoutTape, ScoutWorld, rollout_scout
from experiments.pulse_game.world import ACTIVE, TICKS


CHUNK_TRIALS = 2
CHUNK = CHUNK_TRIALS * TICKS


@dataclass
class Sampled:
    inputs: Tensor
    actions: Tensor
    initial: Tensor
    log_probability: Tensor
    returns: Tensor
    temperature: float
    success: float


@torch.no_grad()
def sample(
    model: ScoutPulseNet,
    tape: ScoutTape,
    *,
    seed: int,
    temperature: float,
    world: ScoutWorld | None = None,
) -> Sampled:
    if world is None:
        world = ScoutWorld(tape)
    features = ReceiptFeatures(tape.n, seed=seed)
    state = model.initial(tape.n)
    weights = model.prepare()
    rng = torch.Generator().manual_seed(seed + 83147)
    shape = (tape.n, tape.trials, TICKS)
    x = np.empty((*shape, INPUT_SIZE), dtype=np.float32)
    actual = np.empty(shape, dtype=np.int64)
    old = np.empty(shape, dtype=np.float32)
    state_shape = state.shape[1:]
    initial = np.empty((tape.n, tape.trials // CHUNK_TRIALS, *state_shape), dtype=np.float32)
    obs = world.observe()
    while obs is not None:
        trial, tick = world.trial_index, world.tick_index
        if tick == 0 and trial % CHUNK_TRIALS == 0:
            initial[:, trial // CHUNK_TRIALS] = state.numpy()
        input_values = features.observe(obs)
        logits, state = model(torch.from_numpy(input_values), state, weights)
        log_probability = (logits / temperature).log_softmax(dim=-1)
        action = torch.multinomial(log_probability.exp(), 1, generator=rng).squeeze(1)
        a = action.numpy()
        x[:, trial, tick] = input_values
        actual[:, trial, tick] = a
        old[:, trial, tick] = log_probability.gather(1, action[:, None]).squeeze(1).numpy()
        obs = world.step(_PULSES[a])
    # Successful contact emits +1. Incorrect contact and silence both fail the task.
    shown = world.result().observed_reward
    success = (shown > 0).any(axis=-1)
    reward = np.broadcast_to(np.where(success, 1.0, -1.0).astype(np.float32)[..., None], shape).copy()
    return Sampled(
        torch.from_numpy(x.reshape(-1, CHUNK, INPUT_SIZE)),
        torch.from_numpy(actual.reshape(-1, CHUNK)),
        torch.from_numpy(initial.reshape(-1, *state_shape)),
        torch.from_numpy(old.reshape(-1, CHUNK)),
        torch.from_numpy(reward.reshape(-1, CHUNK)),
        temperature,
        float(success.mean()),
    )


def update(
    model: ScoutPulseNet,
    samples: Sampled,
    *,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    batch: int,
    seed: int,
) -> dict[str, float]:
    rng = torch.Generator().manual_seed(seed)
    loss_history = []
    kl_history = []
    for _ in range(epochs):
        batches = torch.randperm(len(samples.inputs), generator=rng).split(batch)
        for selected in batches:
            x, actual = samples.inputs[selected], samples.actions[selected]
            live = x[..., ACTIVE].bool()
            logits, values = model.sequence(x, samples.initial[selected])
            lp = F.log_softmax(logits / samples.temperature, dim=-1)
            lp_actual = lp.gather(-1, actual.unsqueeze(-1)).squeeze(-1)[live]
            log_ratio = lp_actual - samples.log_probability[selected][live]
            ratio = log_ratio.exp()
            returns = samples.returns[selected][live]
            values = values[live]
            advantage = returns - values.detach()
            advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(0.2)
            actor = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage).mean()
            critic = F.mse_loss(values, returns)
            entropy = -(lp[live].exp() * lp[live]).sum(dim=-1).mean()
            loss = actor + 0.25 * critic - 0.004 * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            loss_history.append(float(loss.detach()))
            kl_history.append(float((ratio - 1 - log_ratio).mean().detach()))
    return {"loss": float(np.mean(loss_history)), "kl": float(np.mean(kl_history))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--lives", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=192)
    parser.add_argument("--lr", type=float, default=0.00015)
    parser.add_argument("--temperature", type=float, default=1.3)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model, metadata = load(args.input)
    if metadata.get("reward_ranges"):
        parser.error("start reward fine-tuning from the BC/DAgger checkpoint")
    metadata["reward_ranges"] = []
    development = tape_for(DEVELOPMENT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    start = perf_counter()
    for step in range(args.rounds):
        seed_start = TRAIN_REWARD + step * 512
        seeds = range(seed_start, seed_start + args.lives)
        sampled = sample(model, tape_for(seeds), seed=POLICY_SEED + step, temperature=args.temperature)
        stats = update(model, sampled, optimizer=optimizer, epochs=args.epochs, batch=args.batch, seed=310681 + step)
        outcome = measure(rollout_scout(ScoutNetPolicy(model, seed=POLICY_SEED), development), development)
        metadata["reward_ranges"].append([seeds.start, seeds.stop])
        print(json.dumps({"round": step, "sampled_success": sampled.success, "ppo": stats, "development": outcome, "seconds": perf_counter() - start}), flush=True)
    save(args.output, model, metadata)
    print(json.dumps({"stage": "eval", "results": evaluate(model, development), "elapsed": perf_counter() - start}), flush=True)


if __name__ == "__main__":
    main()
