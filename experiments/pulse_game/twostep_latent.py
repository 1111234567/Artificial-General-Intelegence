"""Learned associative addresses and payloads for the two-step motor actor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from experiments.pulse_game.scout_neural import NeuralScoutConfig, ScoutPulseNet
from experiments.pulse_game.twostep_neural import PULSES
from experiments.pulse_game.twostep_world import AVAILABLE, ENTRANCE_RECEIPT, ENTRANCE_STAGE, OBS_SIZE, ROOM_ID, ROOM_STAGE, STAGE_START, VISIT
from experiments.pulse_game.world import CONTACT_ID, LEFT_DISTANCE, LEFT_ID, POSITION, REWARD_NEG, REWARD_POS, RIGHT_DISTANCE, RIGHT_ID, TRIAL_START, VELOCITY


RAW_SIZE = OBS_SIZE + 1


@dataclass(frozen=True)
class MemoryConfig:
    route_key: int = 8
    value_key: int = 12
    payload: int = 6


@dataclass
class Associations:
    route_keys: Tensor
    outcome_keys: Tensor
    room_codes: Tensor
    outcome_codes: Tensor
    outcome_rate: Tensor


@dataclass
class MemoryState:
    routes: Tensor
    outcomes: Tensor
    entrance_counts: Tensor
    fruit_counts: Tensor


class LearnedMemory(nn.Module):
    """Separate learned key-value banks with writes at actual pair contacts.

    The experiment defines which public events can bind. The key embeddings,
    signed payload, and motor interpretation are learned parameters. An
    episode's writes enter memory at the next public start pulse.
    """

    def __init__(self, config: MemoryConfig = MemoryConfig()) -> None:
        super().__init__()
        self.config = config
        self.entrance_keys = nn.Linear(2, config.route_key, bias=False)
        self.outcome_keys = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, config.value_key, bias=False))
        self.room_encoder = nn.Linear(2, config.payload, bias=False)
        self.sign_encoder = nn.Linear(2, config.payload, bias=False)
        self.outcome_rate_logit = nn.Parameter(torch.tensor(1.1))
        self.decode_room = nn.Linear(config.payload, 2)
        self.decode_sign = nn.Linear(config.payload, 2)
        eye = torch.eye(2)
        self.register_buffer("identity", eye)
        self.register_buffer("pairs", torch.cat((eye.repeat_interleave(2, dim=0), eye.repeat(2, 1)), dim=-1))

    def prepare(self) -> Associations:
        return Associations(
            F.normalize(self.entrance_keys(self.identity), dim=-1),
            F.normalize(self.outcome_keys(self.pairs), dim=-1),
            self.room_encoder(self.identity).tanh(),
            self.sign_encoder(self.identity).tanh(),
            self.outcome_rate_logit.sigmoid(),
        )

    def initial(self, batch: int) -> MemoryState:
        c = self.config
        zeros = self.identity.new_zeros
        return MemoryState(zeros(batch, c.route_key, c.payload), zeros(batch, c.value_key, c.payload), zeros(batch, 2), zeros(batch, 4))

    def recall(self, state: MemoryState, assoc: Associations) -> tuple[Tensor, Tensor]:
        return assoc.route_keys @ state.routes, assoc.outcome_keys @ state.outcomes

    def update(
        self, state: MemoryState, assoc: Associations, entry: Tensor, room: Tensor, fruit: Tensor, sign: Tensor,
    ) -> MemoryState:
        room_valid = room.sum(dim=-1, keepdim=True)
        e_gate = entry * room_valid
        pair = (room.unsqueeze(2) * fruit.unsqueeze(1)).flatten(1)
        v_gate = pair * sign.sum(dim=-1, keepdim=True)
        e_count = state.entrance_counts + e_gate
        v_count = state.fruit_counts + v_gate
        ek = e_gate @ assoc.route_keys
        vk = v_gate @ assoc.outcome_keys
        route_rate = 1 / (e_count * e_gate).sum(dim=-1, keepdim=True).clamp_min(1)
        route_prediction = (ek.unsqueeze(1) @ state.routes).squeeze(1)
        value_prediction = (vk.unsqueeze(1) @ state.outcomes).squeeze(1)
        route_delta = route_rate * ((room @ assoc.room_codes) - route_prediction)
        value_delta = assoc.outcome_rate * ((sign @ assoc.outcome_codes) - value_prediction)
        return MemoryState(
            state.routes + ek.unsqueeze(-1) * route_delta.unsqueeze(1),
            state.outcomes + vk.unsqueeze(-1) * value_delta.unsqueeze(1),
            e_count,
            v_count,
        )

    def view(self, state: MemoryState, assoc: Associations, *, intervention: str = "none") -> Tensor:
        routes, outcomes = self.recall(state, assoc)
        count = state.fruit_counts
        if intervention in ("swap_routes", "swap_both"):
            routes = routes.flip(1)
        if intervention in ("swap_rooms", "swap_both"):
            outcomes = outcomes[:, [2, 3, 0, 1]]
            count = count[:, [2, 3, 0, 1]]
        if intervention == "erase_routes":
            routes = torch.zeros_like(routes)
        return torch.cat((routes.flatten(1), outcomes.flatten(1), state.entrance_counts.clamp_max(6) / 6, count.clamp_max(3) / 3), dim=1)


class LatentMotor(ScoutPulseNet):
    """Apply the physical mirror to public fields while recall stays attached to identity."""

    def __init__(self, config: NeuralScoutConfig, *, input_size: int) -> None:
        self.input_size = input_size
        super().__init__(config)
        h = config.hidden
        self.sensory = nn.Sequential(nn.Linear(input_size, h), nn.Tanh(), nn.Linear(h, h))
        self.gate = nn.Linear(input_size, h)
        with torch.no_grad():
            self.gate.bias.fill_(0.65)

    def _setup_symmetry(self, symmetry: str) -> None:
        if symmetry not in ("none", "physical"):
            raise ValueError("learned latent features support the physical mirror")
        order = np.arange(self.input_size)
        sign = np.ones(self.input_size, dtype=np.float32)
        indices, signs, motors = [order.copy()], [sign.copy()], [[0, 1, 2]]
        if symmetry == "physical":
            order[LEFT_ID] = np.arange(RIGHT_ID.start, RIGHT_ID.stop)
            order[RIGHT_ID] = np.arange(LEFT_ID.start, LEFT_ID.stop)
            order[[LEFT_DISTANCE, RIGHT_DISTANCE]] = [RIGHT_DISTANCE, LEFT_DISTANCE]
            order[AVAILABLE] = order[AVAILABLE][::-1]
            sign[[POSITION, VELOCITY]] = -1
            indices.append(order)
            signs.append(sign)
            motors.append([0, 2, 1])
        self._orbit = len(motors)
        self.register_buffer("_input_orders", torch.from_numpy(np.stack(indices)), persistent=False)
        self.register_buffer("_input_signs", torch.from_numpy(np.stack(signs)), persistent=False)
        self.register_buffer("_outputs", torch.tensor(motors), persistent=False)


class LatentTwoStep(nn.Module):
    def __init__(self, motor: NeuralScoutConfig = NeuralScoutConfig(symmetry="physical"), memory: MemoryConfig = MemoryConfig()) -> None:
        super().__init__()
        self.memory = LearnedMemory(memory)
        self.motor = LatentMotor(motor, input_size=RAW_SIZE + 6 * memory.payload + 6)


class LatentPolicy:
    """One life state, with online associative writes from the observed pulses."""

    def __init__(self, model: LatentTwoStep, *, seed: int, intervention: str = "none", erase_stage_tags: bool = False) -> None:
        if intervention not in ("none", "swap_routes", "swap_rooms", "swap_both", "erase_routes", "freeze_visit_writes"):
            raise ValueError("unknown learned-memory intervention")
        self.model = model
        self.seed = seed
        self.intervention = intervention
        self.erase_stage_tags = erase_stage_tags

    @torch.no_grad()
    def reset(self, n: int) -> None:
        self._rng = np.random.default_rng(self.seed)
        self.assoc = self.model.memory.prepare()
        self.memory = self.model.memory.initial(n)
        self.activity = self.model.motor.initial(n)
        self.graph = self.model.motor.prepare()
        self.events = np.zeros((n, 4, 2), dtype=np.float32)
        self.coin = np.zeros((n, 1), dtype=np.float32)
        self.visited = False
        self.previous_visit = False

    @torch.no_grad()
    def logits(self, obs: np.ndarray) -> Tensor:
        if obs[0, TRIAL_START]:
            if not (self.previous_visit and self.intervention == "freeze_visit_writes"):
                tokens = torch.from_numpy(self.events).unbind(dim=1)
                self.memory = self.model.memory.update(self.memory, self.assoc, *tokens)
            self.events.fill(0)
            self.coin[:, 0] = 2 * self._rng.integers(2, size=len(self.coin)) - 1
            self.previous_visit = bool(obs[0, VISIT])
            self.visited |= self.previous_visit
            intervene = self.visited and bool(obs[0, ENTRANCE_STAGE])
            if intervene:
                self.recalled = self.model.memory.view(self.memory, self.assoc, intervention=self.intervention)
                self.normal_read = self.model.memory.view(self.memory, self.assoc)
            else:
                self.recalled = self.model.memory.view(self.memory, self.assoc)
                self.normal_read = self.recalled
        fields = (ENTRANCE_RECEIPT, ROOM_ID, CONTACT_ID, slice(REWARD_POS, REWARD_NEG + 1))
        for index, field in enumerate(fields):
            self.events[:, index] = np.maximum(self.events[:, index], obs[:, field])
        if not obs[0, ENTRANCE_STAGE]:
            self.recalled = self.normal_read
        public = torch.from_numpy(self.public_frame(obs))
        logits, self.activity = self.model.motor(torch.cat((public, self.recalled), dim=1), self.activity, self.graph)
        return logits

    def act(self, obs: np.ndarray) -> np.ndarray:
        return PULSES[self.logits(obs).argmax(dim=-1).cpu().numpy()]

    def public_frame(self, obs: np.ndarray) -> np.ndarray:
        frame = np.column_stack((obs, self.coin))
        if self.erase_stage_tags:
            frame[:, [ENTRANCE_STAGE, ROOM_STAGE, STAGE_START, VISIT]] = 0
        return frame
