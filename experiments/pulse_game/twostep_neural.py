"""Sparse motor network receiving public observations and separate fast banks."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import nn

from experiments.pulse_game.scout_neural import NeuralScoutConfig, ScoutPulseNet
from experiments.pulse_game.twostep_agent import RouteMemory
from experiments.pulse_game.twostep_world import AVAILABLE, ENTRANCE_RECEIPT, ENTRANCE_STAGE, OBS_SIZE, ROOM_ID, ROOM_STAGE, VISIT
from experiments.pulse_game.world import CONTACT_ID, LEFT_DISTANCE, LEFT_ID, POSITION, RIGHT_DISTANCE, RIGHT_ID, TRIAL_START, VELOCITY


INPUT_SIZE = OBS_SIZE + 15
ROUTES = slice(OBS_SIZE, OBS_SIZE + 4)
VALUES = slice(OBS_SIZE + 4, OBS_SIZE + 8)
FRUIT_EXPOSURE = slice(OBS_SIZE + 8, OBS_SIZE + 12)
ENTRANCE_EXPOSURE = slice(OBS_SIZE + 12, OBS_SIZE + 14)
PULSES = np.array([[0, 0], [1, 0], [0, 1]], dtype=bool)


class TwoStepPulseNet(ScoutPulseNet):
    """Same directed recurrence and motor classes, fitted to a new input format."""

    def __init__(self, config: NeuralScoutConfig = NeuralScoutConfig()) -> None:
        super().__init__(config)
        size = config.hidden
        self.sensory = nn.Sequential(nn.Linear(INPUT_SIZE, size), nn.Tanh(), nn.Linear(size, size))
        self.gate = nn.Linear(INPUT_SIZE, size)
        with torch.no_grad():
            self.gate.bias.fill_(0.65)

    def _setup_symmetry(self, symmetry: str) -> None:
        if symmetry not in ("none", "physical", "physical_context", "full"):
            raise ValueError("unknown two-step label symmetry")
        mirrors = (False,) if symmetry == "none" else (False, True)
        room_swaps = (False, True) if symmetry in ("physical_context", "full") else (False,)
        object_swaps = (False, True) if symmetry == "full" else (False,)
        input_indices, input_signs, motor_indices = [], [], []
        for mirror in mirrors:
            for room in room_swaps:
                for objects in object_swaps:
                    index = np.arange(INPUT_SIZE)
                    sign = np.ones(INPUT_SIZE, dtype=np.float32)
                    if mirror:
                        index[LEFT_ID] = np.arange(RIGHT_ID.start, RIGHT_ID.stop)
                        index[RIGHT_ID] = np.arange(LEFT_ID.start, LEFT_ID.stop)
                        index[[LEFT_DISTANCE, RIGHT_DISTANCE]] = [RIGHT_DISTANCE, LEFT_DISTANCE]
                        index[AVAILABLE] = index[AVAILABLE][::-1]
                        sign[[POSITION, VELOCITY]] = -1
                    if room:
                        index[ROOM_ID] = index[ROOM_ID][::-1]
                        index[ROUTES] = index[ROUTES][[1, 0, 3, 2]]
                        index[VALUES] = index[VALUES][[2, 3, 0, 1]]
                        index[FRUIT_EXPOSURE] = index[FRUIT_EXPOSURE][[2, 3, 0, 1]]
                    if objects:
                        for field in (LEFT_ID, RIGHT_ID, CONTACT_ID, ENTRANCE_RECEIPT, ENTRANCE_EXPOSURE):
                            index[field] = index[field][::-1]
                        index[ROUTES] = index[ROUTES][[2, 3, 0, 1]]
                        index[VALUES] = index[VALUES][[1, 0, 3, 2]]
                        index[FRUIT_EXPOSURE] = index[FRUIT_EXPOSURE][[1, 0, 3, 2]]
                        sign[-1] = -1
                    input_indices.append(index)
                    input_signs.append(sign)
                    motor_indices.append([0, 2, 1] if mirror else [0, 1, 2])
        self._orbit = len(motor_indices)
        self.register_buffer("_input_orders", torch.from_numpy(np.stack(input_indices)), persistent=False)
        self.register_buffer("_input_signs", torch.from_numpy(np.stack(input_signs)), persistent=False)
        self.register_buffer("_outputs", torch.tensor(motor_indices), persistent=False)


class RouteFeatures:
    """The readout contains separate transition and terminal-value memories."""

    def __init__(
        self,
        n: int,
        *,
        seed: int,
        intervention: Literal["none", "swap_routes", "swap_rooms", "swap_both", "erase_routes", "swap_arrival_room"] = "none",
        write_visit_values: bool = True,
        mean_visits: bool = False,
    ) -> None:
        if intervention not in ("none", "swap_routes", "swap_rooms", "swap_both", "erase_routes", "swap_arrival_room"):
            raise ValueError("invalid two-step memory intervention")
        self.memory = RouteMemory(n, write_visit_values=write_visit_values, mean_visits=mean_visits)
        self.intervention = intervention
        self._rng = np.random.default_rng(seed)
        self._coin = np.zeros((n, 1), dtype=np.float32)
        self._visited = False

    def observe(self, obs: np.ndarray) -> np.ndarray:
        self._visited |= bool(obs[0, VISIT])
        if obs[0, TRIAL_START]:
            self._coin[:, 0] = 2 * self._rng.integers(2, size=len(self._coin)) - 1
        features = np.column_stack((self.memory.neural_features(obs), self._coin))
        if self._visited and obs[0, ENTRANCE_STAGE]:
            if self.intervention in ("swap_routes", "swap_both"):
                features[:, ROUTES] = features[:, ROUTES][:, [1, 0, 3, 2]]
            if self.intervention in ("swap_rooms", "swap_both"):
                features[:, VALUES] = features[:, VALUES][:, [2, 3, 0, 1]]
                features[:, FRUIT_EXPOSURE] = features[:, FRUIT_EXPOSURE][:, [2, 3, 0, 1]]
            if self.intervention == "erase_routes":
                features[:, ROUTES] = 0
        if self._visited and self.intervention == "swap_arrival_room":
            arrived = (obs[:, ROOM_STAGE] > 0) & (obs[:, VISIT] == 0)
            features[arrived, ROOM_ID] = features[arrived, ROOM_ID][:, ::-1]
        return features


class TwoStepNetPolicy:
    """Convert the learned motor logits to pulses for the NumPy simulator."""

    def __init__(
        self, model: TwoStepPulseNet, *, seed: int, intervention: str = "none",
        write_visit_values: bool = True, mean_visits: bool = False,
    ) -> None:
        self.model = model
        self.seed = seed
        self.intervention = intervention
        self.write_visit_values = write_visit_values
        self.mean_visits = mean_visits

    def reset(self, n: int) -> None:
        self.features = RouteFeatures(
            n, seed=self.seed, intervention=self.intervention, write_visit_values=self.write_visit_values, mean_visits=self.mean_visits,
        )
        self.activity = self.model.initial(n)
        self.weights = self.model.prepare().detach()

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        raw = torch.as_tensor(self.features.observe(obs), device=self.activity.device)
        logits, self.activity = self.model(raw, self.activity, self.weights)
        return PULSES[logits.argmax(dim=-1).cpu().numpy()]
