"""Train one-frame recall and actual movement solely with terminal rewards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.pulse_game.scout_neural import NeuralScoutConfig
from experiments.pulse_game.scout_neural_reward import Sampled, update
from experiments.pulse_game.twostep_eval import paired
from experiments.pulse_game.twostep_flash import FlashPolicy
from experiments.pulse_game.twostep_flash_control import DELAYS, TOTAL_QUIET
from experiments.pulse_game.twostep_neural import INPUT_SIZE, PULSES, RouteFeatures, TwoStepPulseNet
from experiments.pulse_game.twostep_neural_train import DEVELOPMENT, POLICY_SEED, _train_tape, load, save
from experiments.pulse_game.twostep_reward import RewardingProbeWorld
from experiments.pulse_game.twostep_world import OBS_SIZE, STAGE_START, TwoStepTape, make_tape
from experiments.pulse_game.world import ACTIVE, TICKS


PREAMBLE = TOTAL_QUIET + 1
PHASE = TICKS + PREAMBLE
FRAMES = 2 * PHASE


@torch.no_grad()
def sample(
    model: TwoStepPulseNet, tape: TwoStepTape, *, seed: int, temperature: float,
    delays: tuple[int, ...], extra_probe: int, hold_fraction: float = 0.0,
) -> tuple[Sampled, float]:
    observer = RouteFeatures(tape.n, seed=seed)
    activity = model.initial(tape.n)
    recurrent = model.prepare()
    rng = torch.Generator().manual_seed(seed + 720008)
    schedule = np.random.default_rng(seed + 42727)
    world = RewardingProbeWorld(tape)
    rows = np.arange(tape.n)
    shape = (tape.n, tape.episodes, FRAMES)
    inputs = np.zeros((*shape, INPUT_SIZE), dtype=np.float32)
    moves = np.zeros(shape, dtype=np.int64)
    logp = np.zeros(shape, dtype=np.float32)
    hidden_shape = activity.shape[1:]
    start = np.empty((*shape[:2], *hidden_shape), dtype=np.float32)
    held = np.zeros(tape.n, dtype=bool)
    obs = world.observe()
    while obs is not None:
        episode, phase, local = world.episode_index, world.stage, world.local_tick
        base = phase * PHASE
        raw = observer.observe(obs)
        if world.tick_index == 0:
            start[:, episode] = activity.numpy()
        if local == 0:
            cue = raw.copy()
            cue[:, ACTIVE] = 0
            lag = schedule.permutation(np.resize(delays, tape.n))
            offset = TOTAL_QUIET - lag
            if hold_fraction:
                held = schedule.random(tape.n) < hold_fraction
                active_read = (np.arange(PREAMBLE)[None, :] > offset[:, None]) & held[:, None]
                inputs[:, episode, base:base + PREAMBLE, OBS_SIZE:] = cue[:, None, OBS_SIZE:] * active_read[..., None]
            inputs[rows, episode, base + offset] = cue
            for step in range(PREAMBLE):
                frame = torch.from_numpy(inputs[:, episode, base + step])
                _, activity = model(frame, activity, recurrent)
        real = base + PREAMBLE + local
        raw[~held, OBS_SIZE:] = 0
        inputs[:, episode, real] = raw
        logits, activity = model(torch.from_numpy(raw), activity, recurrent)
        lp = (logits / temperature).log_softmax(dim=-1)
        actions = torch.multinomial(lp.exp(), 1, generator=rng)
        selected = actions[:, 0].numpy()
        moves[:, episode, real] = selected
        logp[:, episode, real] = lp.gather(1, actions)[:, 0].numpy()
        obs = world.step(PULSES[selected])
    result = world.result()
    positive = (result.shown > 0).any(axis=-1)
    terminal = np.where(positive, 1.0, -1.0).astype(np.float32)
    targets = np.broadcast_to(terminal[..., None], shape).copy()
    selection = np.arange(tape.n * tape.episodes)
    if extra_probe:
        probe = selection.reshape(tape.n, tape.episodes)[:, tape.probe]
        selection = np.concatenate((selection, np.tile(probe, extra_probe)))
    return Sampled(
        torch.from_numpy(inputs.reshape(-1, FRAMES, INPUT_SIZE)[selection]),
        torch.from_numpy(moves.reshape(-1, FRAMES)[selection]),
        torch.from_numpy(start.reshape(-1, *hidden_shape)[selection]),
        torch.from_numpy(logp.reshape(-1, FRAMES)[selection]),
        torch.from_numpy(targets.reshape(-1, FRAMES)[selection]),
        temperature,
        float(positive.mean()),
    ), float(positive[:, tape.probe].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-start", type=int, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--lives", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=128)
    parser.add_argument("--curriculum-rounds", type=int, default=20)
    parser.add_argument("--quiet-leak-init", type=float, default=-3.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--extra-probe", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=192)
    parser.add_argument("--hold-fraction", type=float, default=0.0)
    parser.add_argument("--warm-start", type=Path)
    args = parser.parse_args()
    if not 0 <= args.hold_fraction <= 1:
        parser.error("hold fraction must be in [0, 1]")
    torch.set_num_threads(2)
    torch.manual_seed(args.model_seed)
    if args.warm_start is None:
        model = TwoStepPulseNet(NeuralScoutConfig(symmetry="physical_context"))
        metadata = {"world_ranges": [], "development": [DEVELOPMENT.start, DEVELOPMENT.stop], "model_seed": args.model_seed}
    else:
        model, metadata = load(args.warm_start)
        if metadata.get("distinct_fruit", False):
            parser.error("use a core two-step checkpoint for this curriculum")
    with torch.no_grad():
        model.gate.bias.fill_(args.quiet_leak_init)
        model.gate.weight[:, STAGE_START] += 3
        model.gate.weight[:, ACTIVE] -= 1
    metadata.update({
        "protocol": "mixed" if args.hold_fraction else "flash", "delays": DELAYS, "reward_only": "fit_protocol" not in metadata,
        "flash_fit": {k: getattr(args, k) for k in ("lives", "rounds", "curriculum_rounds", "quiet_leak_init", "lr", "temperature", "extra_probe", "epochs", "batch", "hold_fraction")},
    })
    used = set(range(*metadata["development"]))
    for left, right in metadata["world_ranges"]:
        used.update(range(left, right))
    if used.intersection(range(args.train_start, args.train_start + args.lives * args.rounds)):
        parser.error("training overlaps development or earlier training")
    test_tape = make_tape(DEVELOPMENT, rare_probability=0.2, distance_jitter=0.22, wind_scale=0.12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    clock = perf_counter()
    for step in range(args.rounds):
        lo = args.train_start + step * args.lives
        seeds = range(lo, lo + args.lives)
        tape = _train_tape(seeds, seed=args.model_seed + step)
        if step < args.curriculum_rounds:
            delays = (0, 2, 4)
        elif step < 2 * args.curriculum_rounds:
            delays = (4, 8, 16)
        else:
            delays = DELAYS
        corpus, probes = sample(
            model, tape, seed=POLICY_SEED + step, temperature=args.temperature, delays=delays,
            extra_probe=args.extra_probe, hold_fraction=args.hold_fraction,
        )
        stats = update(model, corpus, optimizer=optimizer, epochs=args.epochs, batch=args.batch, seed=521022 + step)
        metadata["world_ranges"].append([seeds.start, seeds.stop])
        summary = {"step": step, "sampled_success": corpus.success, "sampled_probe_success": probes, "optimizer": stats, "seconds": perf_counter() - clock}
        if step % 8 == 0 or step == args.rounds - 1:
            development = {}
            for delay in DELAYS:
                factory = lambda: FlashPolicy(model, seed=POLICY_SEED, silence=delay, pre_silence=TOTAL_QUIET - delay)
                development[str(delay)], _, _ = paired(factory, test_tape)
            summary["development"] = development
            save(args.output, model, metadata)
        print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
