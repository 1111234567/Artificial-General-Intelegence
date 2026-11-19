"""Train associative encodings through earlier public contacts and later actions."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter

import torch
from torch import Tensor
from torch.nn import functional as F

from experiments.pulse_game.scout_neural import NeuralScoutConfig
from experiments.pulse_game.twostep_distinct import fruit_score, make_distinct_tape
from experiments.pulse_game.twostep_eval import evaluate_revaluation, paired
from experiments.pulse_game.twostep_latent import Associations, LatentPolicy, LatentTwoStep, MemoryConfig, RAW_SIZE
from experiments.pulse_game.twostep_neural_train import DEVELOPMENT, POLICY_SEED, _train_tape, collect
from experiments.pulse_game.twostep_world import ENTRANCE_RECEIPT, EPISODE_TICKS, OBS_SIZE, ROOM_ID, make_tape
from experiments.pulse_game.world import CONTACT_ID, REWARD_NEG, REWARD_POS


@dataclass
class LifeExamples:
    public: Tensor
    events: Tensor
    action: Tensor
    weight: Tensor


def _public_data(seeds: range, *, seed: int, distinct: bool, probe_weight: float) -> LifeExamples:
    tape = _train_tape(seeds, seed=seed, distinct=distinct)
    corpus = collect(tape, seed=POLICY_SEED, probe_weight=probe_weight)
    public = torch.cat((corpus.sensory[..., :OBS_SIZE], corpus.sensory[..., -1:]), dim=-1)
    public = public.reshape(tape.n, tape.episodes, EPISODE_TICKS, RAW_SIZE)
    tokens = (ENTRANCE_RECEIPT, ROOM_ID, CONTACT_ID, slice(REWARD_POS, REWARD_NEG + 1))
    events = torch.stack([public[..., field].amax(dim=2) for field in tokens], dim=2)
    return LifeExamples(public, events, corpus.labels.reshape(tape.n, -1, EPISODE_TICKS), corpus.weight.reshape(tape.n, -1, EPISODE_TICKS))


def _decode_pair(decoder, latent: Tensor, keyed: Tensor, target: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    addresses = keyed.argmax(dim=-1)
    rows = torch.arange(len(latent), device=latent.device)
    predicted = decoder(latent[rows, addresses])
    loss = F.cross_entropy(predicted, target.argmax(dim=-1), reduction="none")
    return (loss * valid).sum(), valid.sum()


def sequence(
    model: LatentTwoStep, public: Tensor, events: Tensor, *, initial_activity: Tensor | None = None, auxiliary: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    bank = model.memory
    state = bank.initial(len(public))
    assoc = bank.prepare()
    history = []
    total_aux = public.new_zeros(())
    total_valid = public.new_zeros(())
    for ep in events.unbind(dim=1):
        entry, room, fruit, sign = ep.unbind(dim=1)
        history.append(bank.view(state, assoc))
        if auxiliary:
            routes, values = bank.recall(state, assoc)
            link = (entry.sum(dim=1) > 0) & (room.sum(dim=1) > 0)
            pair = (room.unsqueeze(2) * fruit.unsqueeze(1)).flatten(1)
            rewarded = (pair.sum(dim=1) > 0) & (sign.sum(dim=1) > 0)
            route_loss, route_valid = _decode_pair(bank.decode_room, routes, entry, room, link)
            sign_loss, sign_valid = _decode_pair(bank.decode_sign, values, pair, sign, rewarded)
            total_aux = total_aux + route_loss + sign_loss
            total_valid = total_valid + route_valid + sign_valid
        state = bank.update(state, assoc, entry, room, fruit, sign)
    memory = torch.stack(history, dim=1).unsqueeze(2).expand(-1, -1, EPISODE_TICKS, -1)
    all_inputs = torch.cat((public, memory), dim=-1).flatten(0, 1)
    activity = model.motor.initial(len(all_inputs)) if initial_activity is None else initial_activity.flatten(0, 1)
    motor_logits, critic = model.motor.sequence(all_inputs, activity)
    return motor_logits.reshape(*public.shape[:3], 3), critic.reshape(*public.shape[:3]), total_aux / total_valid.clamp_min(1)


def _separation(assoc: Associations) -> Tensor:
    def loss(keys: Tensor) -> Tensor:
        target = torch.eye(len(keys), device=keys.device, dtype=keys.dtype)
        return (keys @ keys.T - target).square().mean()

    return loss(assoc.route_keys) + loss(assoc.outcome_keys)


def fit(model: LatentTwoStep, data: LifeExamples, *, epochs: int, batch: int, seed: int, aux: float):
    g = torch.Generator().manual_seed(seed + 9005)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.00025)
    for epoch in range(epochs):
        average = 0.0
        batches = torch.randperm(len(data.public), generator=g).split(batch)
        for rows in batches:
            outputs, _, auxiliary = sequence(model, data.public[rows], data.events[rows], auxiliary=aux > 0)
            expected = data.action[rows]
            weight = data.weight[rows]
            per_tick = F.cross_entropy(outputs.flatten(0, 2), expected.flatten(), reduction="none")
            motor_loss = (per_tick * weight.flatten()).sum() / weight.sum()
            loss = motor_loss + aux * auxiliary + 0.008 * _separation(model.memory.prepare())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            average += float(loss.detach())
        yield average / len(batches)


def save(path: Path, model: LatentTwoStep, metadata: dict) -> None:
    torch.save({"model": model.state_dict(), "motor": asdict(model.motor.config), "memory": asdict(model.memory.config), "metadata": metadata}, path)


def load(path: Path) -> tuple[LatentTwoStep, dict]:
    record = torch.load(path, map_location="cpu", weights_only=True)
    model = LatentTwoStep(NeuralScoutConfig(**record["motor"]), MemoryConfig(**record["memory"]))
    model.load_state_dict(record["model"])
    return model.eval(), record["metadata"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--world-start", type=int, required=True)
    train.add_argument("--seed", type=int, default=79489)
    train.add_argument("--lives", type=int, default=384)
    train.add_argument("--epochs", type=int, default=24)
    train.add_argument("--batch", type=int, default=12)
    train.add_argument("--aux", type=float, default=0.15)
    train.add_argument("--distinct-fruit", action="store_true")
    test = commands.add_parser("confirm")
    test.add_argument("--model", type=Path, required=True)
    test.add_argument("--seed-start", type=int, required=True)
    test.add_argument("--lives", type=int, default=256)
    test.add_argument("--policy-seed", type=int, default=227076)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "train":
        torch.manual_seed(args.seed)
        model = LatentTwoStep()
        start = perf_counter()
        seeds = range(args.world_start, args.world_start + args.lives)
        if set(DEVELOPMENT).intersection(seeds):
            parser.error("training overlaps development")
        make = make_distinct_tape if args.distinct_fruit else make_tape
        dev = make(DEVELOPMENT, rare_probability=0.2, distance_jitter=0.22, wind_scale=0.12)
        data = _public_data(seeds, seed=args.seed, distinct=args.distinct_fruit, probe_weight=12)
        metadata = {"world_ranges": [[seeds.start, seeds.stop]], "development": [DEVELOPMENT.start, DEVELOPMENT.stop], "model_seed": args.seed, "aux": args.aux, "epochs": args.epochs, "batch": args.batch, "distinct_fruit": args.distinct_fruit}
        for epoch, loss in enumerate(fit(model, data, epochs=args.epochs, batch=args.batch, seed=args.seed, aux=args.aux), 1):
            info = {"epoch": epoch, "loss": loss, "seconds": perf_counter() - start}
            if epoch % 4 == 0 or epoch == args.epochs:
                report, stay, flip = paired(lambda: LatentPolicy(model, seed=POLICY_SEED), dev)
                info["development"] = report
                if args.distinct_fruit:
                    info["conditional_fruit"] = {"stay": fruit_score(dev, stay, reversed=False), "reversed": fruit_score(dev, flip, reversed=True)}
                save(args.output, model, metadata)
            print(json.dumps(info, allow_nan=False), flush=True)
        return

    model, metadata = load(args.model)
    used = set(range(*metadata["development"]))
    for start, stop in metadata["world_ranges"]:
        used.update(range(start, stop))
    seeds = range(args.seed_start, args.seed_start + args.lives)
    if used.intersection(seeds):
        parser.error("confirmatory lives overlap training or development")
    make = make_distinct_tape if metadata["distinct_fruit"] else make_tape
    experiments = {}
    erase = metadata.get("erase_stage_tags", False)
    for rare in (0.0, 0.2):
        tape = make(seeds, rare_probability=rare, distance_jitter=0.22, wind_scale=0.12)
        factory = lambda: LatentPolicy(model, seed=args.policy_seed, erase_stage_tags=erase)
        report = evaluate_revaluation(factory, tape, seed=args.policy_seed)
        cases = {}
        for mode in ("none", "swap_routes", "swap_rooms", "swap_both", "erase_routes", "freeze_visit_writes"):
            current, a, b = paired(lambda: LatentPolicy(model, seed=args.policy_seed, intervention=mode, erase_stage_tags=erase), tape)
            if metadata["distinct_fruit"]:
                current["conditional_fruit"] = {"stay": fruit_score(tape, a, reversed=False), "reversed": fruit_score(tape, b, reversed=True)}
            cases[mode] = current
        report["bank_interventions"] = cases
        experiments[f"rare_{rare}"] = report
    print(json.dumps({"world_seeds": [seeds.start, seeds.stop - 1], "policy_seed": args.policy_seed, "conditions": experiments}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
