"""Train neural scout pulses using public teacher actions and closed-loop data."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from experiments.pulse_game.scout_agent import LocalScoutPolicy, ScoutConfig
from experiments.pulse_game.scout_neural import INPUT_SIZE, NeuralScoutConfig, ReceiptFeatures, ScoutNetPolicy, ScoutPulseNet, _PULSES
from experiments.pulse_game.scout_world import SCOUT_HIT, SCOUT_SIDE, ScoutTape, ScoutWorld, make_context_replay, make_scout_tape, rollout_scout
from experiments.pulse_game.world import ACTIVE, CTX, TICKS, assert_unseen, make_yoke, swap_sides


MODEL_SEED = 67851
POLICY_SEED = 900584
DEVELOPMENT = range(853000, 853128)
TRAIN_TEACHER = 780000
TRAIN_DAGGER = 805000
TRAIN_REWARD = 825000


def tape_for(seeds: Sequence[int], *, nominal: bool = False) -> ScoutTape:
    if nominal:
        return make_scout_tape(seeds)
    return swap_sides(make_scout_tape(seeds, distance_jitter=0.22, wind_scale=0.12), first_trial=32)


@dataclass
class Examples:
    inputs: Tensor
    actions: Tensor

    @classmethod
    def from_lives(cls, x: np.ndarray, a: np.ndarray, *, trials_per_chunk: int = 2) -> Examples:
        steps = trials_per_chunk * TICKS
        return cls(torch.from_numpy(x.reshape(-1, steps, INPUT_SIZE)), torch.from_numpy(a.reshape(-1, steps)))


@torch.no_grad()
def collect(tape: ScoutTape, *, seed: int, model: ScoutPulseNet | None = None, teacher_mix: float = 0.0) -> Examples:
    """Query a public-history teacher at the observations the body really saw."""
    teacher_cfg = replace(ScoutConfig(), actor_weight=0.0, decision_noise=0.0)
    teacher = LocalScoutPolicy(teacher_cfg, seed=seed)
    teacher.reset(tape.n)
    state = model.initial(tape.n) if model is not None else None
    weights = model.prepare() if model is not None else None
    features = ReceiptFeatures(tape.n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    x = np.empty((tape.n, tape.trials, TICKS, INPUT_SIZE), dtype=np.float32)
    actions = np.empty((tape.n, tape.trials, TICKS), dtype=np.int64)
    world = ScoutWorld(tape)
    obs = world.observe()
    while obs is not None:
        trial, tick = world.trial_index, world.tick_index
        input_values = features.observe(obs)
        desired = teacher.act(obs)
        x[:, trial, tick] = input_values
        actions[:, trial, tick] = desired[:, 0] + 2 * desired[:, 1]
        actual = desired
        if model is not None:
            logits, state = model(torch.from_numpy(input_values), state, weights)
            predicted = _PULSES[logits.argmax(dim=-1).numpy()]
            actual = np.where(rng.random(tape.n)[:, None] < teacher_mix, desired, predicted)
        obs = world.step(actual)
    return Examples.from_lives(x, actions)


def behavior_clone(model: ScoutPulseNet, data: Examples, *, epochs: int, batch: int, seed: int, lr: float) -> list[float]:
    rng = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.00025)
    history = []
    for _ in range(epochs):
        total = 0.0
        batches = torch.randperm(len(data.inputs), generator=rng).split(batch)
        for selected in batches:
            x, expected = data.inputs[selected], data.actions[selected]
            logits, _ = model.sequence(x, model.initial(len(x)))
            loss_by_tick = F.cross_entropy(logits.flatten(0, 1), expected.flatten(), reduction="none").unflatten(0, expected.shape)
            weight = 0.06 + x[..., ACTIVE] * (0.94 + 4.0 * x[..., SCOUT_HIT])
            loss = (loss_by_tick * weight).sum() / weight.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss.detach())
        history.append(total / len(batches))
    return history


def measure(result, tape: ScoutTape) -> dict[str, float]:
    correct_side = tape.left_id != tape.good_id
    mandatory = result.sensitive_trial & (tape.scout_side != correct_side)
    late = np.arange(tape.trials) >= tape.trials - 16
    return {
        "accuracy": result.accuracy,
        "late_accuracy": float(result.correct[:, late].mean()),
        "contact": result.contact_rate,
        "probe": result.probe_rate,
        "reverse": result.reverse_rate,
        "must_reverse_correct": float((result.reversed & result.correct)[mandatory].mean()),
        "must_reverse_correct_late": float((result.reversed & result.correct)[mandatory & late].mean()),
        "mean_ticks": float(np.where(result.timed_out, TICKS - 1, result.latency_ticks).mean()),
        "probe_sensitive_late": float(result.probed[result.sensitive_trial & late].mean()),
        "probe_insensitive_late": float(result.probed[~result.sensitive_trial & late].mean()),
    }


@torch.no_grad()
def receipt_audit(model: ScoutPulseNet, tape: ScoutTape, *, seed: int) -> dict[str, int]:
    """Fork the same recurrent state at beacon choice and context reception."""
    world = ScoutWorld(tape)
    policy = ScoutNetPolicy(model, seed=seed)
    policy.reset(tape.n)
    sensitive = tape.sensitive[np.arange(tape.n)[:, None], tape.cue]
    good_side = (tape.left_id != tape.good_id).astype(np.int8)
    mandatory = sensitive & (tape.scout_side != good_side)
    seen = turned = flipped_turn = 0
    choose_scout_sensitive = choose_scout_insensitive = 0
    rows = np.arange(tape.n)
    obs = world.observe()
    while obs is not None:
        trial = world.trial_index
        late_start = world.tick_index == 0 and trial >= tape.trials - 16
        probe = mandatory[:, trial] & (obs[:, SCOUT_HIT] == 1)
        selected = np.flatnonzero(probe) if trial >= tape.trials - 16 else np.empty(0, dtype=np.intp)
        if selected.size or late_start:
            changed = obs.copy()
            if late_start:
                changed[:, SCOUT_SIDE] = obs[:, SCOUT_SIDE][:, ::-1]
            else:
                changed[selected, CTX] = obs[selected, CTX][:, ::-1]
            alternatives = deepcopy(policy.features).observe(changed)
            logits, _ = model(torch.from_numpy(alternatives), policy.activity, policy.weights)
            alternative = _PULSES[logits.argmax(dim=-1).numpy()]
        pulses = policy.act(obs)
        if late_start:
            side = tape.scout_side[:, trial]
            directed = pulses[rows, side] & alternative[rows, 1 - side]
            choose_scout_sensitive += int((directed & sensitive[:, trial]).sum())
            choose_scout_insensitive += int((directed & ~sensitive[:, trial]).sum())
        if selected.size:
            side = good_side[selected, trial]
            real_turn = pulses[selected, side]
            changes_side = alternative[selected, 1 - side]
            seen += selected.size
            turned += int(real_turn.sum())
            flipped_turn += int((real_turn & changes_side).sum())
        obs = world.step(pulses)
    return {
        "eligible": int(mandatory[:, -16:].sum()), "physically_probed": seen,
        "turned_on_real_receipt": turned, "opposite_pulse_when_flipped": flipped_turn,
        "late_sensitive_starts": int(sensitive[:, -16:].sum()),
        "late_sensitive_started_toward_both_beacon_locations": choose_scout_sensitive,
        "late_insensitive_starts": int((~sensitive[:, -16:]).sum()),
        "late_insensitive_started_toward_both_beacon_locations": choose_scout_insensitive,
    }


def evaluate(model: ScoutPulseNet, tape: ScoutTape, *, seed: int = POLICY_SEED) -> dict:
    def run(**condition):
        return rollout_scout(ScoutNetPolicy(model, seed=seed), tape, **condition)

    true = run()
    conditions = {
        "true": true,
        "sensor_off": run(sensor="none"),
        "sensor_replay": run(sensor="replay", replay=make_context_replay(true, seed=633916)),
        "feedback_off": run(feedback="none"),
        "feedback_yoked": run(feedback="yoked", yoke=make_yoke(true, seed=924303)),
        "synapse_off": rollout_scout(ScoutNetPolicy(model, seed=seed, write=False), tape),
    }
    return {condition: measure(result, tape) for condition, result in conditions.items()}


def save(path: Path, model: ScoutPulseNet, metadata: dict) -> None:
    torch.save({"model": model.state_dict(), "config": asdict(model.config), "metadata": metadata}, path)


def load(path: Path) -> tuple[ScoutPulseNet, dict]:
    state = torch.load(path, weights_only=True, map_location="cpu")
    model = ScoutPulseNet(NeuralScoutConfig(**state["config"]))
    model.load_state_dict(state["model"])
    return model.eval(), state["metadata"]


def used_world_seeds(metadata: dict) -> set[int]:
    if "world_seed_rosters" in metadata:
        seen = {seed for roster in metadata["world_seed_rosters"].values() for seed in roster}
    else:
        seen = set(range(*metadata.get("development", (DEVELOPMENT.start, DEVELOPMENT.stop))))
        for start in [metadata["teacher_start"], *metadata["agent_dagger_starts"]]:
            seen.update(range(start, start + metadata["teacher_lives"]))
    for start, stop in metadata.get("reward_ranges", []):
        seen.update(range(start, stop))
    for filtered in metadata.get("filtered_reward_seed_rosters", []):
        seen.update(filtered)
    return seen


def train(args) -> None:
    start = perf_counter()
    torch.manual_seed(MODEL_SEED)
    model = ScoutPulseNet(NeuralScoutConfig(hidden=args.hidden, degree=args.degree))
    development = tape_for(DEVELOPMENT)
    initial_tape = tape_for(range(TRAIN_TEACHER, TRAIN_TEACHER + args.lives))
    assert_unseen(initial_tape, development)
    initial = collect(initial_tape, seed=POLICY_SEED)
    history = behavior_clone(model, initial, epochs=args.epochs, batch=args.batch, seed=MODEL_SEED, lr=0.002)
    print(json.dumps({"stage": "clone", "loss": history, "development": measure(rollout_scout(ScoutNetPolicy(model, seed=POLICY_SEED), development), development), "seconds": perf_counter() - start}), flush=True)
    for round_index in range(args.dagger):
        lo = TRAIN_DAGGER + 2000 * round_index
        tape = tape_for(range(lo, lo + args.lives))
        assert_unseen(tape, development)
        assert_unseen(initial_tape, tape)
        additional = collect(tape, seed=POLICY_SEED, model=model, teacher_mix=max(0, 0.05 - 0.02 * round_index))
        data = Examples(torch.cat((initial.inputs, additional.inputs)), torch.cat((initial.actions, additional.actions)))
        history = behavior_clone(model, data, epochs=args.dagger_epochs, batch=args.batch, seed=MODEL_SEED + round_index + 1, lr=0.0009)
        print(json.dumps({"stage": "dagger", "round": round_index, "loss": history, "development": measure(rollout_scout(ScoutNetPolicy(model, seed=POLICY_SEED), development), development), "seconds": perf_counter() - start}), flush=True)
    metadata = {
        "model_seed": MODEL_SEED, "teacher_lives": args.lives, "teacher_start": TRAIN_TEACHER,
        "agent_dagger_starts": [TRAIN_DAGGER + 2000 * i for i in range(args.dagger)],
        "development": [DEVELOPMENT.start, DEVELOPMENT.stop],
    }
    save(args.output, model, metadata)
    print(json.dumps({"stage": "eval", "results": evaluate(model, development), "elapsed": perf_counter() - start}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2)
    command = parser.add_subparsers(dest="command", required=True)
    take = command.add_parser("train")
    take.add_argument("--lives", type=int, default=256)
    take.add_argument("--epochs", type=int, default=8)
    take.add_argument("--dagger", type=int, default=2)
    take.add_argument("--dagger-epochs", type=int, default=4)
    take.add_argument("--batch", type=int, default=192)
    take.add_argument("--hidden", type=int, default=64)
    take.add_argument("--degree", type=int, default=12)
    take.add_argument("--output", type=Path, required=True)
    check = command.add_parser("eval")
    check.add_argument("--model", type=Path, required=True)
    check.add_argument("--seed-start", type=int, required=True)
    check.add_argument("--lives", type=int, default=128)
    check.add_argument("--nominal", action="store_true")
    check.add_argument("--policy-seed", type=int, default=POLICY_SEED)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.command == "train":
        train(args)
        return
    model, metadata = load(args.model)
    tape = tape_for(range(args.seed_start, args.seed_start + args.lives), nominal=args.nominal)
    overlap = used_world_seeds(metadata).intersection(tape.seeds)
    if overlap:
        parser.error(f"evaluation overlaps training or development with {len(overlap)} seeds")
    print(json.dumps({
        "seeds": [tape.seeds[0], tape.seeds[-1]],
        "results": evaluate(model, tape, seed=args.policy_seed),
        "receipt_audit": receipt_audit(model, tape, seed=args.policy_seed),
    }, indent=2))


if __name__ == "__main__":
    main()
