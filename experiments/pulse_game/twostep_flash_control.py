"""Matched-duration recall, state swap, and equal-dose state controls."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import Tensor

from experiments.pulse_game.scout_neural import NeuralScoutConfig
from experiments.pulse_game.twostep_eval import _episode_stats, _paired_diff, paired
from experiments.pulse_game.twostep_flash import FlashPolicy
from experiments.pulse_game.twostep_neural import INPUT_SIZE, TwoStepPulseNet
from experiments.pulse_game.twostep_neural_train import DEVELOPMENT, POLICY_SEED, Corpus, _train_tape, clone, collect, load, save
from experiments.pulse_game.twostep_world import OBS_SIZE, STAGE_START, TwoStepTape, TwoStepWorld, make_tape, set_reversal
from experiments.pulse_game.world import ACTIVE, REWARD_NEG, REWARD_POS, TICKS


DELAYS = (8, 16, 32)
TOTAL_QUIET = max(DELAYS)


def scheduled_corpus(
    base: Corpus, *, seed: int, protocol: str, delays: tuple[int, ...] = DELAYS, balanced: bool = True,
) -> Corpus:
    """Move a one-frame cue within the same waiting window in every episode."""
    rng = np.random.default_rng(seed)
    n = len(base.sensory)
    times = rng.permutation(np.resize(delays, n)) if balanced else rng.choice(delays, size=n)
    offsets = torch.from_numpy(TOTAL_QUIET - times)
    held = np.full(n, protocol == "hold", dtype=bool)
    if protocol == "mixed":
        held = rng.random(n) < 0.5
    rows = torch.arange(n)
    x_parts, y_parts, weights = [], [], []
    idle_shape = (n, TOTAL_QUIET + 1)
    idle_y = base.labels.new_zeros(idle_shape)
    idle_weight = base.weight.new_full(idle_shape, 0.06)
    for start in (0, TICKS):
        stage = slice(start, start + TICKS)
        idle = base.sensory.new_zeros((*idle_shape, INPUT_SIZE))
        cue = base.sensory[:, start].clone()
        cue[:, ACTIVE] = 0
        memory = cue[:, OBS_SIZE:]
        visible = (torch.arange(TOTAL_QUIET + 1)[None, :] > offsets[:, None]) & torch.from_numpy(held)[:, None]
        idle[:, :, OBS_SIZE:] = memory[:, None, :] * visible[:, :, None]
        idle[rows, offsets] = cue
        movement = base.sensory[:, stage].clone()
        movement[~held, :, OBS_SIZE:] = 0
        x_parts.extend((idle, movement))
        y_parts.extend((idle_y, base.labels[:, stage]))
        weights.extend((idle_weight, base.weight[:, stage]))
    return Corpus(torch.cat(x_parts, dim=1), torch.cat(y_parts, dim=1), torch.cat(weights, dim=1))


def equal_dose_state(state: Tensor, target: Tensor, rng: torch.Generator) -> Tensor:
    """Preserve target norm and displacement, with a different direction."""
    shape = state.shape
    x, y = state.flatten(1), target.flatten(1)
    axis = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
    along = (y * axis).sum(dim=1, keepdim=True) * axis
    transverse = y - along
    magnitude = transverse.norm(dim=1, keepdim=True)
    oriented = transverse / magnitude.clamp_min(1e-8)
    noise = torch.randn(x.shape, dtype=x.dtype, generator=rng)
    noise -= (noise * axis).sum(dim=1, keepdim=True) * axis
    noise -= (noise * oriented).sum(dim=1, keepdim=True) * oriented
    alternative = along + magnitude * noise / noise.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return alternative.reshape(shape)


def equal_cut_state(state: Tensor, rng: torch.Generator) -> Tensor:
    """Match the displacement of zeroing the state while retaining its norm."""
    x = state.flatten(1)
    magnitude = x.norm(dim=1, keepdim=True)
    axis = x / magnitude.clamp_min(1e-8)
    noise = torch.randn(x.shape, dtype=x.dtype, generator=rng)
    orthogonal = noise - (noise * axis).sum(dim=1, keepdim=True) * axis
    unit = orthogonal / orthogonal.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return (0.5 * x + (3**0.5 / 2) * magnitude * unit).reshape_as(state)


@dataclass
class ProbePair:
    worlds: tuple[TwoStepWorld, TwoStepWorld]
    policies: tuple[FlashPolicy, FlashPolicy]
    observations: tuple[np.ndarray, np.ndarray]


def match_acquisition_and_visits(model: TwoStepPulseNet, tape: TwoStepTape, *, delay: int, hold: bool, seed: int) -> ProbePair:
    stay_tape, reverse_tape = set_reversal(tape, False), set_reversal(tape, True)
    world_a, world_b = TwoStepWorld(stay_tape), TwoStepWorld(reverse_tape)
    solver = lambda: FlashPolicy(
        model, seed=seed, silence=delay, pre_silence=TOTAL_QUIET - delay, hold=hold,
    )
    a, b = solver(), solver()
    a.reset(tape.n)
    b.reset(tape.n)
    obs_a, obs_b = world_a.observe(), world_b.observe()
    use_a = np.random.default_rng(seed + 1911).integers(2, size=tape.n).astype(bool)
    nonfeedback = [key for key in range(OBS_SIZE) if key not in (REWARD_POS, REWARD_NEG)]
    while world_a.episode_index < tape.probe:
        command_a, command_b = a.act(obs_a), b.act(obs_b)
        if world_a.is_visit:
            np.testing.assert_array_equal(obs_a[:, nonfeedback], obs_b[:, nonfeedback])
            shared = np.where(use_a[:, None], command_a, command_b)
            command_a = command_b = shared
        obs_a, obs_b = world_a.step(command_a), world_b.step(command_b)
    np.testing.assert_array_equal(obs_a, obs_b)
    return ProbePair((world_a, world_b), (a, b), (obs_a, obs_b))


def compare_states(prefix: ProbePair, *, seed: int, mode: str) -> tuple[dict, np.ndarray]:
    original = prefix.policies[0].model
    worlds = prefix.worlds
    fork = deepcopy(prefix, memo={id(original): original, id(worlds[0].tape): worlds[0].tape, id(worlds[1].tape): worlds[1].tape})
    world_a, world_b = fork.worlds
    stay_tape, reverse_tape = world_a.tape, world_b.tape
    tape = reverse_tape
    a, b = fork.policies
    obs_a, obs_b = fork.observations
    a.lesion = b.lesion = mode if mode in ("cut_edges", "cut_silence_edges") else "intact"
    rng = torch.Generator().manual_seed(seed + 11210)
    state_dose = None
    while obs_a is not None:
        prepared_a, prepared_b = a.prepare(obs_a), b.prepare(obs_b)
        if world_a.episode_index == tape.probe and world_a.tick_index == 0:
            np.testing.assert_array_equal(obs_a, obs_b)
            initial_a, initial_b = a.state, b.state
            dose = (initial_a - initial_b).flatten(1).norm(dim=1)
            state_dose = {"mean_pair_l2_before_external_swap": float(dose.mean())}
            if mode == "swap_states":
                a.state, b.state = initial_b, initial_a
            elif mode == "equal_dose":
                a.state = equal_dose_state(initial_a, initial_b, rng)
                b.state = equal_dose_state(initial_b, initial_a, rng)
                state_dose["max_norm_error"] = float(torch.maximum(
                    (a.state.flatten(1).norm(dim=1) - initial_b.flatten(1).norm(dim=1)).abs(),
                    (b.state.flatten(1).norm(dim=1) - initial_a.flatten(1).norm(dim=1)).abs(),
                ).max())
                state_dose["max_dose_error"] = float(torch.maximum(
                    ((a.state - initial_a).flatten(1).norm(dim=1) - dose).abs(),
                    ((b.state - initial_b).flatten(1).norm(dim=1) - dose).abs(),
                ).max())
            elif mode == "reset_after_read":
                a.state = torch.zeros_like(initial_a)
                b.state = torch.zeros_like(initial_b)
            elif mode == "equal_cut_dose":
                a.state = equal_cut_state(initial_a, rng)
                b.state = equal_cut_state(initial_b, rng)
                state_dose["max_cut_dose_error"] = float(torch.maximum(
                    ((a.state - initial_a).flatten(1).norm(dim=1) - initial_a.flatten(1).norm(dim=1)).abs(),
                    ((b.state - initial_b).flatten(1).norm(dim=1) - initial_b.flatten(1).norm(dim=1)).abs(),
                ).max())
        obs_a, obs_b = world_a.step(a.drive(prepared_a)), world_b.step(b.drive(prepared_b))
    ra, rb = world_a.result(), world_b.result()
    success_a, success_b = ra.effective_entry[:, tape.probe], rb.effective_entry[:, tape.probe]
    result = {
        "stay": _episode_stats(stay_tape, ra), "reversed": _episode_stats(reverse_tape, rb),
        "balanced_optimal_entrance": float((success_a.mean() + success_b.mean()) / 2),
        "both_first_probes_optimal": float((success_a & success_b).mean()),
        "both_first_probes_opposite_target": float(((ra.probe_entry == tape.final_entrance) & (rb.probe_entry == tape.initial_entrance)).mean()),
        "state_dose": state_dose,
        "visit_motion_and_nonfeedback_observations_matched": True,
    }
    return result, 0.5 * (success_a.astype(np.float32) + success_b)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    tasks = parser.add_subparsers(dest="command", required=True)
    fit = tasks.add_parser("train")
    fit.add_argument("--world-start", type=int, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--seed", type=int, required=True)
    fit.add_argument("--lives", type=int, default=384)
    fit.add_argument("--epochs", type=int, default=24)
    fit.add_argument("--protocol", choices=("flash", "hold", "mixed"), default="flash")
    fit.add_argument("--warm-epochs", type=int, default=0)
    fit.add_argument("--quiet-leak-init", type=float)
    fit.add_argument("--unbalanced-delay", action="store_true")
    test = tasks.add_parser("confirm")
    test.add_argument("--model", type=Path, required=True)
    test.add_argument("--seed-start", type=int, required=True)
    test.add_argument("--lives", type=int, default=256)
    test.add_argument("--policy-seed", type=int, default=967284)
    test.add_argument("--flash-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "train":
        seeds = range(args.world_start, args.world_start + args.lives)
        if set(DEVELOPMENT).intersection(seeds):
            parser.error("training overlaps development")
        started = perf_counter()
        torch.manual_seed(args.seed)
        model = TwoStepPulseNet(NeuralScoutConfig(hidden=88, degree=14, symmetry="physical_context"))
        if args.quiet_leak_init is not None:
            with torch.no_grad():
                model.gate.bias.fill_(args.quiet_leak_init)
                model.gate.weight[:, STAGE_START] += 3
                model.gate.weight[:, ACTIVE] -= 1
        source = collect(_train_tape(seeds, seed=args.seed), seed=POLICY_SEED, probe_weight=12)
        warm_loss = []
        if args.warm_epochs:
            warm = scheduled_corpus(source, seed=args.seed + 2404, protocol=args.protocol, delays=(0, 2, 4), balanced=not args.unbalanced_delay)
            warm_loss = clone(model, warm, epochs=args.warm_epochs, batch=192, rate=0.002, seed=args.seed - 1)
        data = scheduled_corpus(source, seed=args.seed + 2405, protocol=args.protocol, balanced=not args.unbalanced_delay)
        losses = clone(model, data, epochs=args.epochs, batch=192, rate=0.002, seed=args.seed)
        dev = make_tape(DEVELOPMENT, rare_probability=0.2, distance_jitter=0.22, wind_scale=0.12)
        development = {}
        for delay in DELAYS:
            policy = lambda: FlashPolicy(model, seed=POLICY_SEED, silence=delay, pre_silence=TOTAL_QUIET - delay, hold=args.protocol == "hold")
            report, _, _ = paired(policy, dev)
            development[str(delay)] = report
        save(args.output, model, {
            "world_ranges": [[seeds.start, seeds.stop]], "development": [DEVELOPMENT.start, DEVELOPMENT.stop],
            "model_seed": args.seed, "protocol": args.protocol, "epochs": args.epochs, "delays": DELAYS,
            "warm_epochs": args.warm_epochs, "quiet_leak_init": args.quiet_leak_init,
            "balanced_delay": not args.unbalanced_delay,
        })
        print(json.dumps({"warm_losses": warm_loss, "losses": losses, "development": development, "seconds": perf_counter() - started}, allow_nan=False))
        return

    model, metadata = load(args.model)
    used = set(range(*metadata["development"]))
    for start, stop in metadata["world_ranges"]:
        used.update(range(start, stop))
    seeds = range(args.seed_start, args.seed_start + args.lives)
    if used.intersection(seeds):
        parser.error("probe worlds must be held out")
    tape = make_tape(seeds, rare_probability=0.2, distance_jitter=0.22, wind_scale=0.12)
    reports = {}
    input_modes = (False,) if args.flash_only else (False, True)
    for hold in input_modes:
        by_delay = {}
        for delay in DELAYS:
            by_mode = {}
            intact = None
            initial_pair = match_acquisition_and_visits(model, tape, delay=delay, hold=hold, seed=args.policy_seed)
            for mode in ("intact", "reset_after_read", "equal_cut_dose", "cut_edges", "cut_silence_edges", "swap_states", "equal_dose"):
                report, scores = compare_states(initial_pair, seed=args.policy_seed, mode=mode)
                if intact is None:
                    intact = scores
                report["intact_minus_case"] = _paired_diff(intact - scores)
                by_mode[mode] = report
            by_delay[str(delay)] = by_mode
        reports["held" if hold else "flash"] = by_delay
    print(json.dumps({"world_seeds": [seeds.start, seeds.stop - 1], "policy_seed": args.policy_seed, "conditions": reports}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
