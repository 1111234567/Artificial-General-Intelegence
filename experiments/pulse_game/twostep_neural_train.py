"""Fit and confirm a two-step motor network with public-history demonstrations."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from experiments.pulse_game.scout_neural import NeuralScoutConfig
from experiments.pulse_game.twostep_agent import DirectedRoutePolicy
from experiments.pulse_game.twostep_distinct import DistinctFruitTape, fruit_score, make_distinct_tape
from experiments.pulse_game.twostep_eval import evaluate_revaluation, paired, _accuracy, _paired_diff
from experiments.pulse_game.twostep_neural import INPUT_SIZE, PULSES, RouteFeatures, TwoStepNetPolicy, TwoStepPulseNet
from experiments.pulse_game.twostep_world import ENTRANCE_STAGE, EPISODE_TICKS, STAGE_START, TwoStepTape, TwoStepWorld, make_tape
from experiments.pulse_game.world import ACTIVE


MODEL_SEED = 70235
POLICY_SEED = 662831
TEACHER_START = 2220000
DAGGER_START = 2250000
DEVELOPMENT = range(2180000, 2180128)


def _train_tape(seeds: range, *, seed: int, distinct: bool = False, mix_rare: bool = True) -> TwoStepTape:
    make = make_distinct_tape if distinct else make_tape
    tape = make(seeds, rare_probability=0.2, distance_jitter=0.22, wind_scale=0.12)
    rng = np.random.default_rng(seed)
    reversal = rng.integers(2, size=tape.n).astype(bool)
    rare_allowed = rng.integers(2, size=tape.n).astype(bool)
    final = tape.initial_good_room ^ reversal
    rare = (tape.rare & rare_allowed[:, None]) if mix_rare else tape.rare
    return replace(tape, final_good_room=final, rare=rare)


@dataclass
class Corpus:
    sensory: Tensor
    labels: Tensor
    weight: Tensor


@torch.no_grad()
def collect(
    tape: TwoStepTape, *, seed: int, model: TwoStepPulseNet | None = None, teacher_mix: float = 0.0, probe_weight: float = 5.0,
) -> Corpus:
    shape = (tape.n, tape.episodes, EPISODE_TICKS)
    x = np.empty((*shape, INPUT_SIZE), dtype=np.float32)
    y = np.empty(shape, dtype=np.int64)
    weight = np.empty(shape, dtype=np.float32)
    rng = np.random.default_rng(seed + 1)
    mean_visits = isinstance(tape, DistinctFruitTape)
    source = DirectedRoutePolicy(seed, mean_visits=mean_visits)
    source.reset(tape.n)
    features = RouteFeatures(tape.n, seed=seed, mean_visits=mean_visits)
    state = model.initial(tape.n) if model is not None else None
    matrix = model.prepare() if model is not None else None
    world = TwoStepWorld(tape)
    obs = world.observe()
    while obs is not None:
        episode, tick = world.episode_index, world.tick_index
        inputs = features.observe(obs)
        desired = source.act(obs)
        x[:, episode, tick] = inputs
        y[:, episode, tick] = desired[:, 0] + 2 * desired[:, 1]
        importance = 1 + 0.7 * obs[:, STAGE_START]
        if episode == tape.probe:
            importance += probe_weight * obs[:, ENTRANCE_STAGE]
        weight[:, episode, tick] = 0.04 + obs[:, ACTIVE] * importance
        actions = desired
        if model is not None:
            logits, state = model(torch.from_numpy(inputs), state, matrix)
            proposed = PULSES[logits.argmax(dim=-1).numpy()]
            actions = np.where(rng.random(tape.n)[:, None] < teacher_mix, desired, proposed)
        obs = world.step(actions)
    return Corpus(
        torch.from_numpy(x.reshape(-1, EPISODE_TICKS, INPUT_SIZE)),
        torch.from_numpy(y.reshape(-1, EPISODE_TICKS)),
        torch.from_numpy(weight.reshape(-1, EPISODE_TICKS)),
    )


def clone(model: TwoStepPulseNet, corpus: Corpus, *, epochs: int, batch: int, rate: float, seed: int) -> list[float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=0.0002)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(epochs):
        batches = torch.randperm(len(corpus.sensory), generator=generator).split(batch)
        running = 0.0
        for indices in batches:
            x, y, weight = corpus.sensory[indices], corpus.labels[indices], corpus.weight[indices]
            logits, _ = model.sequence(x, model.initial(len(x)))
            loss = (F.cross_entropy(logits.transpose(1, 2), y, reduction="none") * weight).sum() / weight.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running += float(loss.detach())
        losses.append(running / len(batches))
    return losses


def save(path: Path, model: TwoStepPulseNet, metadata: dict) -> None:
    torch.save({"model": model.state_dict(), "config": asdict(model.config), "metadata": metadata}, path)


def load(path: Path) -> tuple[TwoStepPulseNet, dict]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    model = TwoStepPulseNet(NeuralScoutConfig(**data["config"]))
    model.load_state_dict(data["model"])
    return model.eval(), data["metadata"]


def confirm(model: TwoStepPulseNet, tape: TwoStepTape, *, seed: int, record: bool = False) -> dict:
    distinct = isinstance(tape, DistinctFruitTape)
    policy = lambda: TwoStepNetPolicy(model, seed=seed, mean_visits=distinct)
    report = evaluate_revaluation(policy, tape, seed=seed, record=record)
    _, stay, reverse = paired(policy, tape)
    actual = _accuracy(stay, reverse)
    interventions = {}
    cases = ["swap_routes", "swap_rooms", "swap_both", "erase_routes", "freeze_visit_writes"]
    if distinct:
        cases.append("swap_arrival_room")
        report["conditional_fruit"] = {"stay": fruit_score(tape, stay, reversed=False), "reversed": fruit_score(tape, reverse, reversed=True)}
    for label in cases:
        freeze = label == "freeze_visit_writes"
        alternative, a, b = paired(
            lambda: TwoStepNetPolicy(
                model, seed=seed, intervention="none" if freeze else label, mean_visits=distinct,
                write_visit_values=not freeze,
            ),
            tape,
        )
        alternative["paired_true_minus_condition"] = _paired_diff(actual - _accuracy(a, b))
        alternative["entrance_flipped_vs_untouched"] = float(((a.probe_entry == 1 - stay.probe_entry) & (b.probe_entry == 1 - reverse.probe_entry)).mean())
        alternative["entrances_same_as_untouched"] = float(((a.probe_entry == stay.probe_entry) & (b.probe_entry == reverse.probe_entry)).mean())
        if distinct and label == "swap_arrival_room":
            alternative["conditional_fruit"] = {"stay": fruit_score(tape, a, reversed=False), "reversed": fruit_score(tape, b, reversed=True)}
            alternative["fruit_choice_flipped_both"] = float(((a.fruit_id[:, -1] == 1 - stay.fruit_id[:, -1]) & (b.fruit_id[:, -1] == 1 - reverse.fruit_id[:, -1])).mean())
        interventions[label] = alternative
    report["probe_memory_interventions"] = interventions
    return report


def train(args) -> None:
    begin = perf_counter()
    torch.manual_seed(MODEL_SEED)
    model = TwoStepPulseNet(NeuralScoutConfig(hidden=args.hidden, degree=args.degree, symmetry=args.symmetry))
    make = make_distinct_tape if args.distinct_fruit else make_tape
    dev = make(DEVELOPMENT, rare_probability=0.2, distance_jitter=0.22, wind_scale=0.12)
    starts = [TEACHER_START, *(DAGGER_START + 2000 * i for i in range(args.dagger))]
    metadata = {
        "world_ranges": [[lo, lo + args.lives] for lo in starts], "development": [DEVELOPMENT.start, DEVELOPMENT.stop],
        "model_seed": MODEL_SEED, "fit_protocol": {"epochs": args.epochs, "dagger_epochs": args.dagger_epochs, "batch": args.batch, "probe_weight": args.probe_weight},
        "distinct_fruit": args.distinct_fruit,
    }
    initial = collect(_train_tape(range(TEACHER_START, TEACHER_START + args.lives), seed=MODEL_SEED, distinct=args.distinct_fruit), seed=POLICY_SEED, probe_weight=args.probe_weight)
    losses = clone(model, initial, epochs=args.epochs, batch=args.batch, rate=0.002, seed=MODEL_SEED)
    probe, _, _ = paired(lambda: TwoStepNetPolicy(model, seed=POLICY_SEED, mean_visits=args.distinct_fruit), dev)
    print(json.dumps({"stage": "clone", "losses": losses, "development": probe, "seconds": perf_counter() - begin}), flush=True)
    for index, start in enumerate(starts[1:]):
        seed = MODEL_SEED + index + 1
        extra = collect(
            _train_tape(range(start, start + args.lives), seed=seed, distinct=args.distinct_fruit), seed=POLICY_SEED, model=model, teacher_mix=0.06,
            probe_weight=args.probe_weight,
        )
        combined = Corpus(
            torch.cat((initial.sensory, extra.sensory)),
            torch.cat((initial.labels, extra.labels)),
            torch.cat((initial.weight, extra.weight)),
        )
        losses = clone(model, combined, epochs=args.dagger_epochs, batch=args.batch, rate=0.0009, seed=seed)
        probe, _, _ = paired(lambda: TwoStepNetPolicy(model, seed=POLICY_SEED, mean_visits=args.distinct_fruit), dev)
        print(json.dumps({"stage": "dagger", "round": index, "losses": losses, "development": probe, "seconds": perf_counter() - begin}), flush=True)
    save(args.output, model, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    fitting = command.add_parser("train")
    fitting.add_argument("--lives", type=int, default=256)
    fitting.add_argument("--epochs", type=int, default=8)
    fitting.add_argument("--dagger", type=int, default=2)
    fitting.add_argument("--dagger-epochs", type=int, default=4)
    fitting.add_argument("--batch", type=int, default=192)
    fitting.add_argument("--hidden", type=int, default=64)
    fitting.add_argument("--degree", type=int, default=12)
    fitting.add_argument("--symmetry", choices=("none", "physical", "physical_context", "full"), default="none")
    fitting.add_argument("--probe-weight", type=float, default=5.0)
    fitting.add_argument("--distinct-fruit", action="store_true")
    fitting.add_argument("--output", type=Path, required=True)
    test = command.add_parser("confirm")
    test.add_argument("--model", type=Path, required=True)
    test.add_argument("--seed-start", type=int, required=True)
    test.add_argument("--lives", type=int, default=256)
    test.add_argument("--policy-seed", type=int, default=888265)
    test.add_argument("--record-proof", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "train":
        train(args)
        return
    model, metadata = load(args.model)
    used = set(range(*metadata["development"]))
    for start, stop in metadata["world_ranges"]:
        used.update(range(start, stop))
    seeds = range(args.seed_start, args.seed_start + args.lives)
    if used.intersection(seeds):
        parser.error("confirmatory lives overlap training or development")
    tests = {}
    make = make_distinct_tape if metadata.get("distinct_fruit", False) else make_tape
    for rare in (0.0, 0.2):
        tape = make(seeds, rare_probability=rare, distance_jitter=0.22, wind_scale=0.12)
        tests[f"rare_{rare}"] = confirm(model, tape, seed=args.policy_seed, record=args.record_proof)
    print(json.dumps({"world_seeds": [seeds.start, seeds.stop - 1], "policy_seed": args.policy_seed, "conditions": tests}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
