"""Sparse recurrent motor policy with feedback-modified fast synapses."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    hidden: int = 32
    degree: int = 6
    keys: int = 16
    values: int = 8
    graph_seed: int = 280976
    fast_enabled: bool = True
    aux_probe: bool = False
    memory_kind: str = "explicit"
    association_hidden: int = 0


@dataclass
class State:
    activity: Tensor
    fast: Tensor
    trace: Tensor | None = None
    association: "Association | None" = None

    def detached(self):
        trace = None if self.trace is None else self.trace.detach()
        return State(self.activity.detach(), self.fast.detach(), trace)


@dataclass
class Association:
    gates: Tensor
    value: Tensor


class Policy(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        if not 1 <= config.degree < config.hidden:
            raise ValueError("degree must be between 1 and hidden - 1")
        self.address = nn.Sequential(nn.Linear(6, 24), nn.Tanh(), nn.Linear(24, config.keys))
        self.receipt = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, config.values), nn.Tanh())
        self.write_logit = nn.Parameter(torch.tensor(1.5))
        self.drive = nn.Linear(20 + config.values, config.hidden)
        self.recurrent = nn.Parameter(torch.randn(config.hidden, config.hidden) * 0.2)
        self.register_buffer("edges", directed_mask(config))
        self.leak_logit = nn.Parameter(torch.full((config.hidden,), 0.8))
        self.motor = nn.Linear(config.hidden, 2)
        with torch.no_grad():
            self.motor.bias.copy_(torch.tensor([0.03, -0.03]))
        if config.aux_probe:
            self.preference_probe = nn.Linear(config.values if config.fast_enabled else config.hidden, 1)

    def initial_state(self, batch: int) -> State:
        return State(
            self.recurrent.new_zeros(batch, self.config.hidden),
            self.recurrent.new_zeros(batch, self.config.keys, self.config.values),
        )

    def prepared_recurrence(self) -> tuple[Tensor, Tensor]:
        return self.recurrent * self.edges, self.leak_logit.sigmoid()

    def encode_key(self, cue_context: Tensor) -> Tensor:
        return F.normalize(self.address(cue_context), dim=-1, eps=1e-6)

    def decode_preference(self, state: State, key: Tensor) -> Tensor:
        observed = (key.unsqueeze(1) @ state.fast).squeeze(1) if self.config.fast_enabled else state.activity
        return self.preference_probe(observed).squeeze(1)

    def forward(
        self,
        obs: Tensor,
        state: State,
        *,
        key: Tensor | None = None,
        recurrent: tuple[Tensor, Tensor] | None = None,
        write_enabled: bool = True,
        read_enabled: bool = True,
    ) -> tuple[Tensor, State]:
        if key is None:
            key = self.encode_key(obs[:, :6])
        if recurrent is None:
            recurrent = self.prepared_recurrence()
        weights, leak = recurrent
        fast = state.fast
        memory = (key.unsqueeze(1) @ fast).squeeze(1)
        if self.config.fast_enabled and write_enabled:
            sensed = obs[:, 13:17]
            has_feedback = sensed[:, :2].sum(dim=1, keepdim=True).clamp(0, 1)
            has_feedback = has_feedback * sensed[:, 2:].sum(dim=1, keepdim=True).clamp(0, 1)
            correction = (self.receipt(sensed) - memory) * (self.write_logit.sigmoid() * has_feedback)
            fast = fast + key.unsqueeze(-1) * correction.unsqueeze(1)
            memory = memory + (key.square().sum(dim=-1, keepdim=True) * correction)
        if not self.config.fast_enabled or not read_enabled:
            memory = torch.zeros_like(memory)
        drive = self.drive(torch.cat((obs, memory), dim=-1))
        activity = torch.lerp(state.activity, torch.tanh(drive + F.linear(state.activity, weights)), leak)
        return self.motor(activity), State(activity, fast)


class TemporalPolicy(nn.Module):
    """All address, payload and gate inputs derive from full public observations.

    A latent trace spans the gap between observing a cue and earning a receipt.
    The model receives the same 20 channels during encoding, movement and writing.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        if not config.fast_enabled or not 1 <= config.degree < config.hidden:
            raise ValueError("The temporal policy requires fast weights and a valid directed graph")
        self.config = config
        self.drive = nn.Linear(20, config.hidden)
        self.recurrent = nn.Parameter(torch.randn(config.hidden, config.hidden) * 0.2)
        self.register_buffer("edges", directed_mask(config))
        self.leak_logit = nn.Parameter(torch.full((config.hidden,), 0.8))
        association_input = 20 + config.hidden
        association_output = config.keys + config.values + 2
        if config.association_hidden:
            final = nn.Linear(config.association_hidden, association_output)
            self.associate = nn.Sequential(nn.Linear(association_input, config.association_hidden), nn.Tanh(), final)
        else:
            final = nn.Linear(association_input, association_output)
            self.associate = final
        self.memory_drive = nn.Linear(config.values, config.hidden, bias=False)
        self.motor = nn.Linear(config.hidden, 2)
        with torch.no_grad():
            self.motor.bias.copy_(torch.tensor([0.03, -0.03]))
            final.bias[-2:].fill_(-1.0)
        if config.aux_probe:
            self.preference_probe = nn.Linear(config.values, 1)

    def initial_state(self, batch: int) -> State:
        base = self.recurrent
        return State(base.new_zeros(batch, self.config.hidden),
                     base.new_zeros(batch, self.config.keys, self.config.values), base.new_zeros(batch, self.config.keys))

    def prepared_recurrence(self) -> tuple[Tensor, Tensor]:
        return self.recurrent * self.edges, self.leak_logit.sigmoid()

    def decode_preference(self, state: State, key: Tensor | None = None) -> Tensor:
        key = F.normalize(state.trace, dim=-1, eps=1e-6)
        return self.preference_probe((key.unsqueeze(1) @ state.fast).squeeze(1)).squeeze(1)

    def forward(
        self,
        obs: Tensor,
        state: State,
        *,
        key: Tensor | None = None,
        recurrent: tuple[Tensor, Tensor] | None = None,
        write_enabled: bool = True,
        read_enabled: bool = True,
    ) -> tuple[Tensor, State]:
        if key is not None:
            raise ValueError("Temporal key must be formed from the policy's own observations")
        if recurrent is None:
            recurrent = self.prepared_recurrence()
        weights, leak = recurrent
        excitation = self.drive(obs) + F.linear(state.activity, weights)
        latent = torch.tanh(excitation)
        projected = self.associate(torch.cat((obs, latent), dim=1))
        raw_key, raw_value, gate_logits = projected.split((self.config.keys, self.config.values, 2), dim=1)
        candidate = F.normalize(raw_key, dim=-1, eps=1e-6)
        capture, write = (0.25 * gate_logits + 0.5).clamp(0, 1).split(1, dim=1)
        trace = torch.lerp(state.trace, candidate, capture)
        key = F.normalize(trace, dim=-1, eps=1e-6)
        fast = state.fast
        read = (key.unsqueeze(1) @ fast).squeeze(1)
        value = torch.tanh(raw_value)
        if write_enabled:
            correction = write * (value - read)
            fast = fast + key.unsqueeze(-1) * correction.unsqueeze(1)
            read = read + key.square().sum(dim=-1, keepdim=True) * correction
        if not read_enabled:
            read = torch.zeros_like(read)
        target = torch.tanh(excitation + self.memory_drive(read))
        activity = torch.lerp(state.activity, target, leak)
        return self.motor(activity), State(activity, fast, trace, Association(gate_logits, value))


class GatedPolicy(nn.Module):
    """General gated recurrent graph with no address routing or plastic weights."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.fast_enabled or not 1 <= config.degree < config.hidden:
            raise ValueError("The gated recurrent control requires a valid graph and no fast weights")
        self.config = config
        self.drive = nn.Linear(20, 3 * config.hidden)
        self.recurrent = nn.Parameter(torch.randn(3 * config.hidden, config.hidden) * 0.2)
        self.register_buffer("edges", directed_mask(config).repeat(3, 1))
        self.leak_logit = nn.Parameter(torch.linspace(-6.0, 0.8, config.hidden))
        self.motor = nn.Sequential(nn.Linear(config.hidden, 24), nn.Tanh(), nn.Linear(24, 2))
        with torch.no_grad():
            self.motor[-1].bias.copy_(torch.tensor([0.03, -0.03]))
        if config.aux_probe:
            self.preference_probe = nn.Linear(config.hidden, 1)

    def initial_state(self, batch: int) -> State:
        zero = self.recurrent.new_zeros
        return State(zero(batch, self.config.hidden), zero(batch, 0, 0))

    def prepared_recurrence(self) -> tuple[Tensor, Tensor]:
        return self.recurrent * self.edges, self.leak_logit

    def decode_preference(self, state: State, key: Tensor | None = None) -> Tensor:
        return self.preference_probe(state.activity).squeeze(1)

    def forward(
        self,
        obs: Tensor,
        state: State,
        *,
        key: Tensor | None = None,
        recurrent: tuple[Tensor, Tensor] | None = None,
        write_enabled: bool = True,
        read_enabled: bool = True,
    ) -> tuple[Tensor, State]:
        if recurrent is None:
            recurrent = self.prepared_recurrence()
        weights, update_bias = recurrent
        update, reset, proposal = self.drive(obs).chunk(3, dim=1)
        gate_weights, proposal_weights = weights.split((2 * self.config.hidden, self.config.hidden), dim=0)
        update_r, reset_r = F.linear(state.activity, gate_weights).chunk(2, dim=1)
        update_gain = (update + update_r + update_bias).sigmoid()
        reset_gain = (reset + reset_r).sigmoid()
        activation = (proposal + F.linear(state.activity * reset_gain, proposal_weights)).tanh()
        activity = torch.lerp(state.activity, activation, update_gain)
        return self.motor(activity), State(activity, state.fast)


def directed_mask(config: ModelConfig) -> Tensor:
    generator = torch.Generator().manual_seed(config.graph_seed)
    matrix = torch.zeros(config.hidden, config.hidden)
    for receiver in range(config.hidden):
        ring_parent = (receiver - 1) % config.hidden
        candidates = torch.randperm(config.hidden, generator=generator)
        candidates = candidates[(candidates != receiver) & (candidates != ring_parent)]
        matrix[receiver, ring_parent] = 1
        matrix[receiver, candidates[: config.degree - 1]] = 1
    return matrix


def create_policy(config: ModelConfig) -> Policy | TemporalPolicy | GatedPolicy:
    if config.memory_kind == "explicit":
        return Policy(config)
    if config.memory_kind == "temporal":
        return TemporalPolicy(config)
    if config.memory_kind == "gated":
        return GatedPolicy(config)
    raise ValueError(f"Unknown memory kind: {config.memory_kind}")


def load_policy(path: str | Path | BinaryIO, device: str | torch.device = "cpu") -> Policy | TemporalPolicy | GatedPolicy:
    saved = torch.load(path, map_location=device, weights_only=True)
    policy = create_policy(ModelConfig(**saved["config"])).to(device)
    policy.load_state_dict(saved["model"])
    return policy.eval()


def policy_state_dict(policy) -> dict:
    return {"config": asdict(policy.config), "model": policy.state_dict()}


def recurrent_terms(policy, enabled=True):
    weights, control = policy.prepared_recurrence()
    if enabled:
        return weights, control
    update_now = 40 if policy.config.memory_kind == "gated" else 1
    return torch.zeros_like(weights), torch.full_like(control, update_now)


class RolloutAgent:
    """Deterministic observation-only adapter. Call reset once for each life."""

    def __init__(self, policy: Policy | TemporalPolicy | GatedPolicy, *, write_enabled: bool = True, read_enabled: bool = True,
                 recurrence_enabled: bool = True):
        self.policy = policy.eval()
        self.write_enabled = write_enabled
        self.read_enabled = read_enabled
        self.recurrence_enabled = recurrence_enabled
        self.device = next(policy.parameters()).device
        self.reset()

    @torch.no_grad()
    def reset(self, n: int = 1):
        self.state = self.policy.initial_state(n)
        self.recurrent = recurrent_terms(self.policy, self.recurrence_enabled)

    @torch.no_grad()
    def act(self, observation):
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        logits, self.state = self.policy(
            obs, self.state, recurrent=self.recurrent,
            write_enabled=self.write_enabled, read_enabled=self.read_enabled,
        )
        right = logits[:, 1] > logits[:, 0]
        return torch.stack((~right, right), dim=1).cpu().numpy()
