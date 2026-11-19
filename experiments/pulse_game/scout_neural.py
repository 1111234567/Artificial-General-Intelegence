"""Sparse recurrent neural motor control from public scout observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from experiments.pulse_game.scout_world import SCOUT_OBS_SIZE, SCOUT_SIDE
from experiments.pulse_game.world import (
    CONTACT_ID, CTX, CUE, CUES, LEFT_DISTANCE, LEFT_ID, POSITION, REWARD_NEG, REWARD_POS,
    RIGHT_DISTANCE, RIGHT_ID, TRIAL_START, VELOCITY,
)


INPUT_SIZE = SCOUT_OBS_SIZE + 6
_MAPS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
_PULSES = np.array([[0, 0], [1, 0], [0, 1]], dtype=bool)


class ReceiptFeatures:
    """Cue-local fast weights driven by public context and outcome pulses."""

    def __init__(self, n: int, *, seed: int, write: bool = True, read: bool = True) -> None:
        self._rng = np.random.default_rng(seed)
        self._write = write
        self._read = read
        self._rows = np.arange(n)
        self._weights = np.zeros((n, CUES, 4), dtype=np.float32)
        self._ctx = np.zeros((n, 2), dtype=np.float32)
        self._random = np.zeros((n, 2), dtype=np.float32)
        likelihood = np.where(_MAPS.T[:, None] == np.arange(2)[None, :, None], 0.95, 0.05)
        likelihood = np.concatenate([likelihood, likelihood.mean(axis=0, keepdims=True)])
        self._log_likelihood = np.log(likelihood).astype(np.float32)

    def observe(self, obs: np.ndarray) -> np.ndarray:
        cue = obs[:, CUE].argmax(axis=1)
        if obs[0, TRIAL_START]:
            self._ctx.fill(0)
            self._random[:, 0] = 2 * self._rng.integers(2, size=len(self._rows)) - 1
            self._random[:, 1] = 2 * self._rng.random(len(self._rows)) - 1
        self._ctx *= 0.94
        self._ctx += obs[:, CTX]
        if self._write:
            self._update(obs, cue)
        probabilities = np.exp(self._weights[self._rows, cue])
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        if not self._read:
            probabilities.fill(0.25)
        return np.column_stack((obs, probabilities - 0.25, self._random))

    def _update(self, obs: np.ndarray, cue: np.ndarray) -> None:
        receipt = obs[:, CONTACT_ID]
        sign = obs[:, REWARD_POS] - obs[:, REWARD_NEG]
        tagged = np.flatnonzero((receipt.sum(axis=1) == 1) & (sign != 0))
        context = self._ctx[tagged]
        ctx = np.where(context.max(axis=1) > 0.2, context.argmax(axis=1), 2)
        actual = receipt[tagged].argmax(axis=1)
        best = np.where(sign[tagged] > 0, actual, 1 - actual)
        weights = 0.995 * self._weights[tagged, cue[tagged]] + self._log_likelihood[ctx, best]
        weights -= weights.max(axis=1, keepdims=True)
        self._weights[tagged, cue[tagged]] = np.maximum(weights, -32)


@dataclass(frozen=True)
class NeuralScoutConfig:
    hidden: int = 64
    degree: int = 12
    graph_seed: int = 77203
    symmetry: str = "none"


class ScoutPulseNet(nn.Module):
    """Map each tick to silence, a left pulse, or a right pulse."""

    def __init__(self, config: NeuralScoutConfig = NeuralScoutConfig()) -> None:
        super().__init__()
        self.config = config
        size = config.hidden
        if not 1 <= config.degree < size:
            raise ValueError("degree must be smaller than the recurrent population")
        generator = torch.Generator().manual_seed(config.graph_seed)
        scores = torch.rand(size, size, generator=generator)
        scores.fill_diagonal_(-1)
        inputs = scores.topk(config.degree, dim=1).indices
        self.register_buffer("edges", torch.zeros_like(scores).scatter_(1, inputs, 1))
        self.recurrent = nn.Parameter(torch.randn(size, size) * (0.22 / config.degree**0.5))
        self.sensory = nn.Sequential(nn.Linear(INPUT_SIZE, size), nn.Tanh(), nn.Linear(size, size))
        self.gate = nn.Linear(INPUT_SIZE, size)
        self.motor = nn.Linear(size, 3)
        self.critic = nn.Linear(size, 1)
        self._setup_symmetry(config.symmetry)
        with torch.no_grad():
            self.gate.bias.fill_(0.65)

    def _setup_symmetry(self, symmetry: str) -> None:
        if symmetry not in ("none", "physical", "physical_context"):
            raise ValueError("unknown scout symmetry")
        groups = [(False, False)]
        if symmetry != "none":
            groups.append((True, False))
        if symmetry == "physical_context":
            groups.extend([(False, True), (True, True)])
        input_orders, input_signs, outputs = [], [], []
        for mirror, swap_context in groups:
            indices = np.arange(INPUT_SIZE)
            sign = np.ones(INPUT_SIZE, dtype=np.float32)
            if mirror:
                indices[LEFT_ID] = np.arange(RIGHT_ID.start, RIGHT_ID.stop)
                indices[RIGHT_ID] = np.arange(LEFT_ID.start, LEFT_ID.stop)
                indices[[LEFT_DISTANCE, RIGHT_DISTANCE]] = [RIGHT_DISTANCE, LEFT_DISTANCE]
                indices[SCOUT_SIDE] = indices[SCOUT_SIDE][::-1]
                sign[[POSITION, VELOCITY]] = -1
            if swap_context:
                indices[CTX] = indices[CTX][::-1]
                mixed_maps = [SCOUT_OBS_SIZE + 1, SCOUT_OBS_SIZE + 2]
                indices[mixed_maps] = indices[mixed_maps][::-1]
            input_orders.append(indices)
            input_signs.append(sign)
            outputs.append([0, 2, 1] if mirror else [0, 1, 2])
        self._orbit = len(groups)
        self.register_buffer("_input_orders", torch.from_numpy(np.stack(input_orders)), persistent=False)
        self.register_buffer("_input_signs", torch.from_numpy(np.stack(input_signs)), persistent=False)
        self.register_buffer("_outputs", torch.tensor(outputs), persistent=False)

    def initial(self, worlds: int) -> Tensor:
        shape = (worlds, self.config.hidden) if self._orbit == 1 else (worlds, self._orbit, self.config.hidden)
        return self.recurrent.new_zeros(shape)

    def _observation_drives(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if self._orbit > 1:
            x = x[..., self._input_orders] * self._input_signs
        return self.sensory(x), self.gate(x).sigmoid()

    def _motor_logits(self, state: Tensor) -> Tensor:
        logits = self.motor(state)
        if self._orbit > 1:
            logits = logits.gather(-1, self._outputs.expand(logits.shape)).mean(dim=-2)
        return logits

    def prepare(self) -> Tensor:
        return self.recurrent * self.edges

    def _step(self, drive: Tensor, gate: Tensor, state: Tensor, weights: Tensor) -> Tensor:
        return torch.lerp(state, torch.tanh(drive + F.linear(state, weights)), gate)

    def forward(self, x: Tensor, state: Tensor, weights: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if weights is None:
            weights = self.prepare()
        drive, gate = self._observation_drives(x)
        state = self._step(drive, gate, state, weights)
        return self._motor_logits(state), state

    def sequence(self, x: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        drive, gate = self._observation_drives(x)
        weights = self.prepare()
        states = []
        for sensory, update in zip(drive.transpose(0, 1), gate.transpose(0, 1)):
            state = self._step(sensory, update, state, weights)
            states.append(state)
        activity = torch.stack(states, dim=1)
        value = self.critic(activity).squeeze(-1)
        if self._orbit > 1:
            value = value.mean(dim=-1)
        return self._motor_logits(activity), value


class ScoutNetPolicy:
    """Detached inference adapter for the original NumPy physics."""

    def __init__(self, model: ScoutPulseNet, *, seed: int = 0, write: bool = True, read: bool = True) -> None:
        self.model = model
        self.seed = seed
        self.write = write
        self.read = read

    def reset(self, n: int) -> None:
        self.features = ReceiptFeatures(n, seed=self.seed, write=self.write, read=self.read)
        self.activity = self.model.initial(n)
        self.weights = self.model.prepare().detach()

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(self.features.observe(obs), device=self.activity.device)
        logits, self.activity = self.model(x, self.activity, self.weights)
        return _PULSES[logits.argmax(dim=-1).cpu().numpy()]
