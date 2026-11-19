"""Learn contact roles with one tactile channel and no stage labels."""

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
from experiments.pulse_game.twostep_latent import RAW_SIZE, MemoryConfig
from experiments.pulse_game.twostep_latent_reward import LifeTrajectories, update
from experiments.pulse_game.twostep_latent_train import load as load_core, save
from experiments.pulse_game.twostep_neural import PULSES
from experiments.pulse_game.twostep_neural_train import POLICY_SEED
from experiments.pulse_game.twostep_rewire import RewireTape, RewireWorld, evaluate, make_rewire_tape
from experiments.pulse_game.twostep_rewire_train import RewireLatent, load as load_rewire, training_tape
from experiments.pulse_game.twostep_reward import discounted_life_returns, terminal_returns
from experiments.pulse_game.twostep_world import ENTRANCE_RECEIPT, EPISODE_TICKS, ROOM_ID
from experiments.pulse_game.world import CONTACT_ID, REWARD_NEG, REWARD_POS, TRIAL_START


DEVELOPMENT = range(3250000, 3250128)


class UntypedWorld(RewireWorld):
    """Observe one common tactile identity pulse at either physical endpoint."""

    def __init__(self, tape: RewireTape, *, training: bool = False, **kwargs) -> None:
        self.training = training
        super().__init__(tape, erase_stage_tags=True, **kwargs)

    def _feedback(self, receipt: np.ndarray) -> np.ndarray:
        if self.training and self.stage == 1 and self.episode_index == self.tape.probe:
            return self._reward[:, self.episode_index] * receipt
        return super()._feedback(receipt)

    def observe(self) -> np.ndarray:
        public = super().observe()
        public[:, CONTACT_ID] += public[:, ENTRANCE_RECEIPT]
        public[:, ENTRANCE_RECEIPT] = 0
        return public


class UntypedLatent(RewireLatent):
    def __init__(
        self, *, adaptive: bool = True, motor: NeuralScoutConfig = NeuralScoutConfig(symmetry="physical"),
        memory: MemoryConfig = MemoryConfig(),
    ) -> None:
        super().__init__(adaptive=adaptive, motor=motor, memory=memory)
        self.contact_role = nn.Sequential(nn.Linear(6, 16), nn.Tanh(), nn.Linear(16, 1))

    def role_attention(self, events: Tensor, *, uniform: bool = False) -> Tensor:
        identity = events[..., :2]
        present = identity.sum(dim=-1) > 0
        score = torch.zeros_like(identity[..., 0]) if uniform else self.contact_role(events).squeeze(-1)
        route_score = score.masked_fill(~present, -torch.inf)
        null_score = torch.zeros_like(route_score[..., :1])
        return torch.cat((route_score, null_score), dim=-1).softmax(dim=-1)

    def classify(self, events: Tensor, room: Tensor, *, uniform: bool = False) -> Tensor:
        identity, reward = events[..., :2], events[..., 4:6]
        route_attention = self.role_attention(events, uniform=uniform)[..., :2]
        route_contact = (route_attention.unsqueeze(-1) * identity).sum(dim=-2)
        signed = reward.sum(dim=-1, keepdim=True)
        valued_contact = (identity * signed).sum(dim=-2)
        return torch.stack((route_contact, room, valued_contact, reward.sum(dim=-2)), dim=-2)

    def decode_episodes(self, packed: Tensor) -> Tensor:
        events = packed[..., :12].reshape(*packed.shape[:-1], 2, 6)
        return self.classify(events, packed[..., 12:14])


class UntypedPolicy:
    def __init__(self, model: UntypedLatent, *, seed: int, uniform_roles: bool = False) -> None:
        self.model = model
        self.seed = seed
        self.uniform_roles = uniform_roles

    @torch.no_grad()
    def reset(self, n: int) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.assoc = self.model.memory.prepare()
        self.memory = self.model.memory.initial(n)
        self.activity = self.model.motor.initial(n)
        self.weights = self.model.motor.prepare()
        self.encounters = np.zeros((n, 2, 6), dtype=np.float32)
        self.room = np.zeros((n, 2), dtype=np.float32)
        self.number = np.zeros(n, dtype=np.int64)
        self.coin = np.zeros((n, 1), dtype=np.float32)

    @torch.no_grad()
    def logits(self, obs: np.ndarray) -> Tensor:
        if obs[0, TRIAL_START]:
            events = self.model.classify(torch.from_numpy(self.encounters), torch.from_numpy(self.room), uniform=self.uniform_roles)
            self.memory = self.model.memory.update(self.memory, self.assoc, *events.unbind(dim=1))
            self.read = self.model.memory.view(self.memory, self.assoc)
            self.encounters.fill(0)
            self.room.fill(0)
            self.number.fill(0)
            self.coin[:, 0] = 2 * self.rng.integers(2, size=len(self.coin)) - 1
        self.room = np.maximum(self.room, obs[:, ROOM_ID])
        contacts = np.flatnonzero(obs[:, CONTACT_ID].sum(axis=1) == 1)
        observable = np.column_stack((obs[contacts, CONTACT_ID], obs[contacts, ROOM_ID], obs[contacts, REWARD_POS:REWARD_NEG + 1]))
        self.encounters[contacts, self.number[contacts]] = observable
        self.number[contacts] += 1
        frame = torch.from_numpy(np.column_stack((obs, self.coin)))
        output, self.activity = self.model.motor(torch.cat((frame, self.read), dim=1), self.activity, self.weights)
        return output

    def act(self, obs: np.ndarray) -> np.ndarray:
        return PULSES[self.logits(obs).argmax(dim=-1).numpy()]


@torch.no_grad()
def sample(
    model: UntypedLatent, tape: RewireTape, *, seed: int, terminal_only: bool = False, positive_actual: bool = False,
    future_discount: float = 0.0,
) -> LifeTrajectories:
    agent = UntypedPolicy(model, seed=seed)
    agent.reset(tape.n)
    world = UntypedWorld(tape, training=True)
    rng = torch.Generator().manual_seed(seed + 97011)
    shape = tape.n, tape.episodes, EPISODE_TICKS
    frames = np.empty((*shape, RAW_SIZE), dtype=np.float32)
    actions = np.empty(shape, dtype=np.int64)
    likelihood = np.empty(shape, dtype=np.float32)
    begins = np.empty((*shape[:2], *agent.activity.shape[1:]), dtype=np.float32)
    episodes = np.empty((*shape[:2], 14), dtype=np.float32)
    obs = world.observe()
    while obs is not None:
        ep, tick = world.episode_index, world.tick_index
        if tick == 0:
            begins[:, ep] = agent.activity.numpy()
            if ep:
                episodes[:, ep - 1, :12] = agent.encounters.reshape(tape.n, 12)
                episodes[:, ep - 1, 12:] = agent.room
        chance = agent.logits(obs).log_softmax(dim=-1)
        chosen = torch.multinomial(chance.exp(), 1, generator=rng)
        signal = chosen[:, 0].numpy()
        frames[:, ep, tick, :-1] = obs
        frames[:, ep, tick, -1] = agent.coin[:, 0]
        actions[:, ep, tick] = signal
        likelihood[:, ep, tick] = chance.gather(1, chosen)[:, 0].numpy()
        obs = world.step(PULSES[signal])
    episodes[:, -1, :12] = agent.encounters.reshape(tape.n, 12)
    episodes[:, -1, 12:] = agent.room
    shown = world.result().shown
    scored = (shown > 0).any(axis=-1)
    credit = discounted_life_returns(terminal_returns(shown, only_contact=terminal_only, positive_only=positive_actual), future_discount)
    return LifeTrajectories(
        torch.from_numpy(frames), torch.from_numpy(episodes), torch.from_numpy(begins), torch.from_numpy(actions),
        torch.from_numpy(likelihood), torch.from_numpy(credit),
        1.0, float(scored.mean()), float(scored[:, tape.probe].mean()),
    )


def load(path: Path) -> tuple[UntypedLatent, dict]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    model = UntypedLatent(motor=NeuralScoutConfig(**data["motor"]), memory=MemoryConfig(**data["memory"]))
    model.load_state_dict(data["model"])
    return model.eval(), data["metadata"]


@torch.no_grad()
def inspect_roles(model: UntypedLatent, tape: RewireTape, *, seed: int) -> dict:
    """Diagnose the treated real encounters after freezing the actor."""
    agent = UntypedPolicy(model, seed=seed)
    agent.reset(tape.n)
    world = UntypedWorld(tape)
    seen: dict[str, list[np.ndarray]] = {"actual_entry": [], "retained_no_entry": [], "actual_rewarded_contact": []}
    obs = world.observe()
    while obs is not None:
        if world.tick_index == 0 and world.episode_index:
            old = world.episode_index - 1
            weights = model.role_attention(torch.from_numpy(agent.encounters)).numpy()
            context = agent.room.sum(axis=1) == 1
            contact = agent.number > 0
            if old < tape.acquisition:
                seen["actual_entry"].append(weights[context & contact, 0])
                earned_terminal = context & (agent.number == 2)
                seen["actual_rewarded_contact"].append(weights[earned_terminal, 1])
            else:
                seen["retained_no_entry"].append(weights[context & contact, 2])
        obs = world.step(agent.act(obs))
    result = {}
    for name, arrays in seen.items():
        values = np.concatenate(arrays) if arrays else np.empty(0)
        result[name] = {"opportunities": len(values), "mean_attention": float(values.mean()) if len(values) else None}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    fit = actions.add_parser("train")
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--train-start", type=int, required=True)
    fit.add_argument("--model-seed", type=int, required=True)
    fit.add_argument("--warm-start", type=Path)
    fit.add_argument("--from-rewire", action="store_true")
    fit.add_argument("--from-untyped", action="store_true")
    fit.add_argument("--lives", type=int, default=96)
    fit.add_argument("--rounds", type=int, default=112)
    fit.add_argument("--batch", type=int, default=8)
    fit.add_argument("--epochs", type=int, default=2)
    fit.add_argument("--lr", type=float, default=0.001)
    fit.add_argument("--probe-weight", type=float, default=6.0)
    credit = fit.add_mutually_exclusive_group()
    credit.add_argument("--terminal-only", action="store_true", help="use zero return for an uncompleted episode")
    credit.add_argument("--positive-actual", action="store_true", help="optimize observed positive endpoint receipts")
    fit.add_argument("--positive-warm-rounds", type=int, default=0)
    fit.add_argument("--future-discount", type=float, default=0.0)
    fit.add_argument("--stable-rounds", type=int, default=0)
    fit.add_argument("--deterministic-rounds", type=int, default=0)
    fit.add_argument("--short-rounds", type=int, default=0)
    test = actions.add_parser("confirm")
    test.add_argument("--model", type=Path, required=True)
    test.add_argument("--seed-start", type=int, required=True)
    test.add_argument("--lives", type=int, default=256)
    test.add_argument("--policy-seed", type=int, default=444816)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.action == "train":
        if args.positive_warm_rounds and (not args.terminal_only or not 0 < args.positive_warm_rounds <= args.rounds):
            parser.error("positive warm rounds precede the signed-receipt objective")
        torch.manual_seed(args.model_seed)
        model = UntypedLatent()
        provenance = []
        if args.warm_start:
            if args.from_rewire and args.from_untyped:
                parser.error("choose one parent model format")
            teacher_free_load = load if args.from_untyped else load_rewire if args.from_rewire else load_core
            parent, old = teacher_free_load(args.warm_start)
            differences = model.load_state_dict(parent.state_dict(), strict=False)
            allowed = ("memory.route_floor_logit", "contact_role.")
            if differences.unexpected_keys or not all(any(key.startswith(prefix) for prefix in allowed) for key in differences.missing_keys):
                parser.error("warm-start input differs from the two-step latent architecture")
            provenance = [*old["world_ranges"], old["development"]]
        seen = set(DEVELOPMENT)
        for lo, hi in provenance:
            seen.update(range(lo, hi))
        if seen.intersection(range(args.train_start, args.train_start + args.lives * args.rounds)):
            parser.error("training must use new worlds")
        metadata = {"world_ranges": provenance, "development": [DEVELOPMENT.start, DEVELOPMENT.stop], "model_seed": args.model_seed,
                    "warm_start_file": args.warm_start.name if args.warm_start else None,
                    "reward_fit": {key: getattr(args, key) for key in ("train_start", "lives", "rounds", "batch", "epochs", "lr", "probe_weight", "stable_rounds", "deterministic_rounds", "short_rounds", "terminal_only", "positive_actual", "positive_warm_rounds", "future_discount")}}
        dev = make_rewire_tape(DEVELOPMENT, rare_probability=0.2, distance_jitter=0.22, wind_scale=0.12)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
        started = perf_counter()
        for step in range(args.rounds):
            lo = args.train_start + step * args.lives
            rare = 0.0 if step < args.deterministic_rounds else 0.2
            tape = training_tape(range(lo, lo + args.lives), seed=args.model_seed + step, rare_probability=rare)
            if step < args.stable_rounds:
                tape = replace(tape, reverse_route=np.zeros(tape.n, dtype=bool), final_good_room=tape.initial_good_room)
            if step < args.short_rounds:
                tape = replace(tape, distances=tape.distances * 0.72)
            early_positive = step < args.positive_warm_rounds
            gamma = 0.0 if early_positive else args.future_discount
            observed = sample(
                model, tape, seed=POLICY_SEED + step, terminal_only=args.terminal_only,
                positive_actual=args.positive_actual or early_positive, future_discount=gamma,
            )
            omit = None if gamma else slice(tape.ordinary, tape.acquisition)
            fit_metrics = update(
                model, observed, optimizer, epochs=args.epochs, batch=args.batch, seed=988301 + step,
                probe_weight=args.probe_weight, omit_episodes=omit, event_projection=model.decode_episodes,
            )
            metadata["world_ranges"].append([lo, lo + args.lives])
            line = {"step": step, "sampled_success": observed.success, "sampled_probe_success": observed.probe_success,
                    "metrics": fit_metrics, "seconds": perf_counter() - started}
            if step % 8 == 0 or step == args.rounds - 1:
                line["development"] = evaluate(lambda: UntypedPolicy(model, seed=POLICY_SEED), dev, world_type=UntypedWorld)
                save(args.output, model, metadata)
            print(json.dumps(line, allow_nan=False), flush=True)
        return

    model, info = load(args.model)
    seen = set(range(*info["development"]))
    for start, stop in info["world_ranges"]:
        seen.update(range(start, stop))
    worlds = range(args.seed_start, args.seed_start + args.lives)
    if seen.intersection(worlds):
        parser.error("held-out seeds must differ from training and development")
    results = {}
    factory = lambda: UntypedPolicy(model, seed=args.policy_seed)
    for rare in (0.0, 0.2):
        tape = make_rewire_tape(worlds, rare_probability=rare, distance_jitter=0.22, wind_scale=0.12)
        results[f"rare_{rare}"] = {
            "intact": evaluate(factory, tape, world_type=UntypedWorld),
            "hide_survey_rooms": evaluate(factory, tape, world_type=UntypedWorld, hide_survey_room=True),
            "hide_visit_rooms": evaluate(factory, tape, world_type=UntypedWorld, hide_visit_room=True),
            "uniform_contact_roles": evaluate(lambda: UntypedPolicy(model, seed=args.policy_seed, uniform_roles=True), tape, world_type=UntypedWorld),
            "role_diagnostics": inspect_roles(model, tape, seed=args.policy_seed),
        }
    print(json.dumps({"world_seeds": [worlds.start, worlds.stop - 1], "policy_seed": args.policy_seed, "conditions": results}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
