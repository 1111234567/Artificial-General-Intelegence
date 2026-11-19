"""Raw-input recurrent scout with trainable associative fast weights."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .meta_model import directed_mask
from .scout_world import SCOUT_OBS_SIZE
from .world import CONTACT_ID, REWARD_NEG, REWARD_POS


PULSES = np.array(((0, 0), (1, 0), (0, 1)), dtype=bool)


@dataclass(frozen=True)
class ScoutModelConfig:
    hidden: int = 56
    degree: int = 10
    graph_seed: int = 136831
    keys: int = 12
    values: int = 16
    projection: int = 48
    fast_enabled: bool = True
    readout_motor: bool = False
    endpoint_value: bool = False
    indexed_outcomes: bool = False
    indexed_min_write: float = 0.0
    body_model: bool = False


@dataclass
class ScoutState:
    activity: Tensor
    fast: Tensor
    trace: Tensor


@dataclass
class ScoutStep:
    logits: Tensor
    state: ScoutState
    read: Tensor
    payload: Tensor
    gate_logits: Tensor
    key: Tensor
    write_gain: Tensor


class ScoutMetaPolicy(nn.Module):
    """Learn all routing from publicly observed raw channels and recurrent activity."""

    def __init__(self, config: ScoutModelConfig = ScoutModelConfig()):
        super().__init__()
        self.config = config
        if not 1 <= config.degree < config.hidden:
            raise ValueError("degree must be between 1 and hidden-1")
        if config.endpoint_value and not config.readout_motor:
            raise ValueError("The endpoint critic requires a readout motor")
        if config.indexed_outcomes and (config.values != 2 or not config.fast_enabled):
            raise ValueError("Indexed outcomes require two fast value columns")
        if config.indexed_min_write and (not config.indexed_outcomes or not 0 <= config.indexed_min_write <= 1):
            raise ValueError("A minimum write gain is available for indexed outcomes and must be in 0..1")
        width = config.hidden
        self.drive = nn.Sequential(nn.Linear(SCOUT_OBS_SIZE, width), nn.Tanh(), nn.Linear(width, width))
        self.update = nn.Linear(SCOUT_OBS_SIZE, width)
        self.recurrent = nn.Parameter(torch.randn(width, width) * 0.18)
        self.register_buffer("edges", directed_mask(config))
        projection_size = config.keys + 2 if config.indexed_outcomes else config.keys + 2 * config.values + 2
        self.associate = nn.Sequential(
            nn.Linear(SCOUT_OBS_SIZE + width, config.projection), nn.Tanh(),
            nn.Linear(config.projection, projection_size),
        )
        self.memory_drive = nn.Linear(config.values, width, bias=False)
        if config.readout_motor:
            pulse_head = nn.Linear(48, 3)
            input_width = width + 7 + 2 * config.endpoint_value
            self.motor = nn.Sequential(nn.Linear(input_width, 48), nn.Tanh(), pulse_head)
        else:
            pulse_head = nn.Linear(width, 3)
            self.motor = pulse_head
        self.context_probe = nn.Linear(width, 3)
        self.preference_probe = nn.Linear(config.values, 2)
        self.seen_probe = nn.Linear(config.values, 2)
        self.critic = nn.Linear(width, 1)
        if config.endpoint_value:
            self.endpoint_probe = nn.Sequential(nn.Linear(width + config.values, 40), nn.Tanh(), nn.Linear(40, 2))
        if config.body_model:
            self.body_predictor = nn.Sequential(nn.Linear(width + 3, 40), nn.Tanh(), nn.Linear(40, 4))
        with torch.no_grad():
            self.update.bias.copy_(torch.linspace(-2.0, 1.0, width))
            self.associate[-1].bias[-2:].copy_(torch.tensor([-0.1, -1.0]))
            pulse_head.bias.copy_(torch.tensor([-0.4, 0.1, 0.0]))

    def initial(self, batch: int) -> ScoutState:
        zero = self.recurrent.new_zeros
        return ScoutState(zero(batch, self.config.hidden), zero(batch, self.config.keys, self.config.values),
                          zero(batch, self.config.keys))

    def prepared(self) -> Tensor:
        return self.recurrent * self.edges

    def forward(self, obs, state, weights=None, encoded=None, *, write_enabled=True, read_enabled=True):
        if weights is None:
            weights = self.prepared()
        drive, update = (self.drive(obs), self.update(obs).sigmoid()) if encoded is None else encoded
        excitation = drive + F.linear(state.activity, weights)
        latent = torch.tanh(excitation)
        config = self.config
        projected = self.associate(torch.cat((obs, latent), dim=1))
        raw_key = projected[:, :config.keys]
        gate_logits = projected[:, -2:]
        candidate = F.normalize(raw_key, dim=1, eps=1e-6)
        capture, write = (0.25 * gate_logits + 0.5).clamp(0, 1).split(1, dim=1)
        trace = torch.lerp(state.trace, candidate, capture)
        key = F.normalize(trace, dim=1, eps=1e-6)
        fast = state.fast
        read = (key.unsqueeze(1) @ fast).squeeze(1)
        if config.indexed_outcomes:
            # Only the actual endpoint column receives the observed signed result.
            sign = obs[:, REWARD_POS:REWARD_POS + 1] - obs[:, REWARD_NEG:REWARD_NEG + 1]
            payload = sign.expand(-1, 2)
            gain = config.indexed_min_write + (1 - config.indexed_min_write) * write
            write_gain = gain * sign.abs() * obs[:, CONTACT_ID]
        else:
            raw_value, channel_gate = projected[:, config.keys:-2].split(config.values, dim=1)
            payload = raw_value.tanh()
            write_gain = write * channel_gate.sigmoid()
        if config.fast_enabled and write_enabled:
            correction = write_gain * (payload - read)
            fast = fast + key.unsqueeze(-1) * correction.unsqueeze(1)
            read = read + key.square().sum(1, keepdim=True) * correction
        if not config.fast_enabled or not read_enabled:
            read = torch.zeros_like(read)
        logits, activity = self.action_from_read(state.activity, excitation, update, read)
        return ScoutStep(logits, ScoutState(activity, fast, trace), read, payload, gate_logits, key, write_gain)

    def action_from_read(self, previous, excitation, update, read):
        config = self.config
        next_activity = torch.tanh(excitation + self.memory_drive(read))
        activity = torch.lerp(previous, next_activity, update)
        motor_input = activity
        if config.readout_motor:
            pieces = [activity, self.preference_probe(read), self.seen_probe(read), self.context_probe(activity)]
            if config.endpoint_value:
                pieces.append(self.endpoint_probe(torch.cat((activity, read), dim=1)))
            motor_input = torch.cat(pieces, dim=1)
        return self.motor(motor_input), activity

    def response_with_read(self, obs, previous, weights, read):
        excitation = self.drive(obs) + F.linear(previous, weights)
        return self.action_from_read(previous, excitation, self.update(obs).sigmoid(), read)


class ScoutRawAgent:
    """Only the 24 public channels and per-life model state enter actual actions."""

    def __init__(self, model: ScoutMetaPolicy, *, write_enabled: bool = True, read_enabled: bool = True,
                 recurrence_enabled: bool = True):
        self.model = model.eval()
        self.write_enabled = write_enabled
        self.read_enabled = read_enabled
        self.recurrence_enabled = recurrence_enabled
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def reset(self, n: int):
        self.state = self.model.initial(n)
        self.weights = self.model.prepared()
        if not self.recurrence_enabled:
            self.weights = torch.zeros_like(self.weights)

    @torch.no_grad()
    def logits(self, obs):
        x = torch.as_tensor(obs, device=self.device, dtype=self.weights.dtype)
        result = self.model(x, self.state, self.weights, write_enabled=self.write_enabled, read_enabled=self.read_enabled)
        self.state = result.state
        self.last_step = result
        return result.logits

    @torch.no_grad()
    def act(self, obs):
        return PULSES[self.logits(obs).argmax(dim=1).cpu().numpy()]


def model_payload(model: ScoutMetaPolicy):
    return {"config": asdict(model.config), "model": model.state_dict()}


def load_scout_model(path: str | Path, device: str = "cpu") -> ScoutMetaPolicy:
    saved = torch.load(path, map_location=device, weights_only=True)
    model = ScoutMetaPolicy(ScoutModelConfig(**saved["config"])).to(device)
    model.load_state_dict(saved["model"])
    return model.eval()
