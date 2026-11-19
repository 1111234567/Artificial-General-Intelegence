"""Pulse the associative readout before a sensor-silent interval and a real choice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.pulse_game.scout_neural import NeuralScoutConfig
from experiments.pulse_game.twostep_eval import _accuracy, _paired_diff, paired
from experiments.pulse_game.twostep_neural import INPUT_SIZE, PULSES, RouteFeatures, TwoStepPulseNet
from experiments.pulse_game.twostep_neural_train import DEVELOPMENT, POLICY_SEED, Corpus, _train_tape, clone, collect, load, save
from experiments.pulse_game.twostep_world import ENTRANCE_STAGE, OBS_SIZE, VISIT, make_tape
from experiments.pulse_game.world import ACTIVE, TICKS


class FlashPolicy:
    """Run extra neural ticks at each phase boundary while the body waits."""

    def __init__(
        self, model: TwoStepPulseNet, *, seed: int, silence: int, lesion: str = "intact",
        hold: bool = False, pre_silence: int = 0,
    ) -> None:
        self.model = model
        self.seed = seed
        self.silence = silence
        self.lesion = lesion
        self.hold = hold
        self.pre_silence = pre_silence

    @torch.no_grad()
    def reset(self, n: int) -> None:
        self.features = RouteFeatures(n, seed=self.seed)
        self.state = self.model.initial(n)
        self.weights = self.model.prepare()
        self.cut = torch.zeros_like(self.weights)
        self.blank = torch.zeros(n, INPUT_SIZE)
        self.tick = 0
        self.visited = False

    @torch.no_grad()
    def prepare(self, obs: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        self.visited |= bool(obs[0, VISIT])
        probe_entrance = self.visited and bool(obs[0, ENTRANCE_STAGE])
        weights = self.cut if probe_entrance and self.lesion == "cut_edges" else self.weights
        delay_weights = self.cut if probe_entrance and self.lesion == "cut_silence_edges" else weights
        frame = torch.from_numpy(self.features.observe(obs))
        if self.tick % TICKS == 0:
            if self.pre_silence:
                for _ in range(self.pre_silence):
                    _, self.state = self.model(self.blank, self.state, weights)
            read = frame.clone()
            read[:, ACTIVE] = 0
            if probe_entrance and self.lesion == "erase_recall":
                read[:, OBS_SIZE:] = 0
            _, self.state = self.model(read, self.state, weights)
            silent = self.blank
            if self.hold:
                silent = self.blank.clone()
                silent[:, OBS_SIZE:] = read[:, OBS_SIZE:]
            for _ in range(self.silence):
                _, self.state = self.model(silent, self.state, delay_weights)
            if probe_entrance and self.lesion == "reset_after_read":
                self.state.zero_()
        if not self.hold:
            frame[:, OBS_SIZE:] = 0
        return frame, weights

    @torch.no_grad()
    def drive(self, prepared: tuple[torch.Tensor, torch.Tensor]) -> np.ndarray:
        frame, weights = prepared
        logits, self.state = self.model(frame, self.state, weights)
        self.tick += 1
        return PULSES[logits.argmax(dim=-1).numpy()]

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.drive(self.prepare(obs))


def pulse_corpus(base: Corpus, silence: int) -> Corpus:
    """Align the extra neural ticks with the factual physical action targets."""
    x_parts, y_parts, weights = [], [], []
    count = len(base.sensory)
    quiet_x = base.sensory.new_zeros(count, silence, INPUT_SIZE)
    quiet_y = base.labels.new_zeros(count, 1 + silence)
    quiet_weight = base.weight.new_full((count, 1 + silence), 0.06)
    for start in (0, TICKS):
        stage = slice(start, start + TICKS)
        cue = base.sensory[:, start:start + 1].clone()
        cue[..., ACTIVE] = 0
        motor = base.sensory[:, stage].clone()
        motor[..., OBS_SIZE:] = 0
        x_parts.extend((cue, quiet_x, motor))
        y_parts.extend((quiet_y, base.labels[:, stage]))
        weights.extend((quiet_weight, base.weight[:, stage]))
    return Corpus(torch.cat(x_parts, dim=1), torch.cat(y_parts, dim=1), torch.cat(weights, dim=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--world-start", type=int, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--lives", type=int, default=384)
    fit.add_argument("--epochs", type=int, default=20)
    fit.add_argument("--silence", type=int, default=12)
    fit.add_argument("--seed", type=int, default=938132)
    confirm = commands.add_parser("confirm")
    confirm.add_argument("--model", type=Path, required=True)
    confirm.add_argument("--seed-start", type=int, required=True)
    confirm.add_argument("--lives", type=int, default=256)
    confirm.add_argument("--policy-seed", type=int, default=764191)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "train":
        seeds = range(args.world_start, args.world_start + args.lives)
        if set(DEVELOPMENT).intersection(seeds):
            parser.error("training overlaps development")
        started = perf_counter()
        torch.manual_seed(args.seed)
        model = TwoStepPulseNet(NeuralScoutConfig(hidden=88, degree=14, symmetry="physical_context"))
        data = _train_tape(seeds, seed=args.seed)
        augmented = pulse_corpus(collect(data, seed=POLICY_SEED, probe_weight=12), args.silence)
        losses = clone(model, augmented, epochs=args.epochs, batch=192, rate=0.002, seed=args.seed)
        tape = make_tape(DEVELOPMENT, rare_probability=0.2, distance_jitter=0.22, wind_scale=0.12)
        report, _, _ = paired(lambda: FlashPolicy(model, seed=POLICY_SEED, silence=args.silence), tape)
        metadata = {
            "world_ranges": [[seeds.start, seeds.stop]], "development": [DEVELOPMENT.start, DEVELOPMENT.stop],
            "model_seed": args.seed, "flash_silence": args.silence, "epochs": args.epochs,
        }
        save(args.output, model, metadata)
        print(json.dumps({"losses": losses, "seconds": perf_counter() - started, "development": report}, allow_nan=False))
        return

    model, meta = load(args.model)
    used = set(range(*meta["development"]))
    for start, stop in meta["world_ranges"]:
        used.update(range(start, stop))
    seeds = range(args.seed_start, args.seed_start + args.lives)
    if used.intersection(seeds):
        parser.error("probe worlds must be held out")
    cases = {}
    for rare in (0.0, 0.2):
        tape = make_tape(seeds, rare_probability=rare, distance_jitter=0.22, wind_scale=0.12)
        results = {}
        true = None
        for lesion in ("intact", "erase_recall", "reset_after_read", "cut_edges"):
            report, a, b = paired(lambda: FlashPolicy(model, seed=args.policy_seed, silence=meta["flash_silence"], lesion=lesion), tape)
            actual = _accuracy(a, b)
            if true is None:
                true = actual
            report["paired_intact_minus_case"] = _paired_diff(true - actual)
            results[lesion] = report
        cases[f"rare_{rare}"] = results
    print(json.dumps({"world_seeds": [seeds.start, seeds.stop - 1], "policy_seed": args.policy_seed, "silence": meta["flash_silence"], "conditions": cases}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
