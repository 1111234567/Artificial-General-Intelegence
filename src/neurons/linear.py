"""Stable linear recurrent cells for dense streams and sparse timed pulses.

The transition parameterization and row-vector convention match the frozen
TasteCircuit sensory graph. New tasks choose their own input size and loss.
"""

import numpy as np
import torch
from torch import Tensor, nn


def _scan(drives: Tensor, transition: Tensor, initial: Tensor) -> Tensor:
    """All states of h[t] = h[t-1] @ transition + drives[t]. Time is axis 0."""
    if not drives.size(0):
        return drives
    states = torch.cat(((initial @ transition + drives[0]).unsqueeze(0), drives[1:]))
    stride = 1
    power = transition
    while stride < len(states):
        states = torch.cat((states[:stride], states[stride:] + states[:-stride] @ power))
        stride *= 2
        if stride < len(states):
            power = power @ power
    return states


def linear_sequence(inputs: Tensor, state: Tensor, transition: Tensor, sensors: Tensor,
                    *, parallel: bool = True) -> tuple[Tensor, Tensor]:
    """Use an existing dense transition and input matrix, including exported weights.

    Inputs are [time, batch, input], states [batch, width]. Matrix columns are
    receivers: one tick is state @ transition + input @ sensors.
    """
    if (sensors.ndim != 2 or inputs.ndim != 3 or inputs.size(2) != sensors.size(0)
            or state.shape != (inputs.size(1), sensors.size(1))
            or transition.shape != (sensors.size(1), sensors.size(1))):
        raise ValueError("Incompatible inputs, state, transition or sensors")
    drives = inputs @ sensors
    if not len(drives):
        return drives, state
    if parallel:
        states = _scan(drives, transition, state)
    else:
        history = []
        for drive in drives.unbind():
            state = state @ transition + drive
            history.append(state)
        states = torch.stack(history)
    return states, states[-1]


class LinearGraph(nn.Module):
    """Stable directed or diagonal dynamics with a trainable input projection."""

    def __init__(self, input_size: int, width: int, *, graph: str = "rotations",
                 graph_seed: int = 19391):
        super().__init__()
        if input_size <= 0 or width <= 0 or graph not in ("diagonal", "rotations"):
            raise ValueError("Expected positive dimensions and graph='diagonal' or 'rotations'")
        if graph == "rotations" and width % 2:
            raise ValueError("Pairwise rotations require an even width")
        self.input_size, self.width, self.graph = input_size, width, graph
        # Keep the names and initial draws used by existing sensory checkpoints.
        rng = np.random.default_rng(graph_seed)
        matchings = np.stack([rng.permutation(width) for _ in range(3)])
        self.register_buffer("matchings", torch.tensor(matchings, dtype=torch.long))
        self.angles = nn.Parameter(.33 * torch.randn(3, width // 2), requires_grad=graph == "rotations")
        self.log_decay = nn.Parameter(torch.linspace(-.8, 1.65, width))
        self.sensors = nn.Parameter(torch.randn(input_size, width) * .33)

    def matrix(self) -> Tensor:
        decay = .28 + .70 * self.log_decay.sigmoid()
        if self.graph == "diagonal":
            return torch.diag(decay)
        basis = torch.eye(self.width, device=decay.device, dtype=decay.dtype)
        for angle, order in zip(self.angles, self.matchings, strict=True):
            pairs = basis.index_select(1, order).reshape(self.width, -1, 2)
            a, b = pairs.unbind(-1)
            c, s = angle.cos(), angle.sin()
            rotated = torch.stack((a * c - b * s, a * s + b * c), dim=-1)
            basis = rotated.reshape_as(basis).index_select(1, order.argsort())
        return basis * decay

    def initial_state(self, batch_size: int) -> Tensor:
        return self.sensors.new_zeros(batch_size, self.width)

    def step(self, inputs: Tensor, state: Tensor) -> Tensor:
        """One physical tick with [batch, input] and [batch, width] tensors."""
        if inputs.ndim != 2 or inputs.size(1) != self.input_size or state.shape != (inputs.size(0), self.width):
            raise ValueError("Expected [batch, input_size] inputs and [batch, width] state")
        return state @ self.matrix() + inputs @ self.sensors

    def forward(self, inputs: Tensor, state: Tensor, *, parallel: bool = True) -> tuple[Tensor, Tensor]:
        """Time-major [time, batch, input]; keep state and gradients across calls."""
        return linear_sequence(inputs, state, self.matrix(), self.sensors, parallel=parallel)

    def pulse_blocks(self, pulses: Tensor, ticks: Tensor, read_at: Tensor,
                     state: Tensor, *, block_ticks: int) -> tuple[Tensor, Tensor]:
        """Sample states without unrolling silent ticks; also return the final block end.

        pulses: [blocks, batch, events, input_size]. ticks: integer offsets of
        those events (-1 = padding). read_at: integer [blocks, batch] query
        offsets. Tick zero includes that tick's input. Events may coincide.
        """
        if (block_ticks <= 0 or pulses.ndim != 4 or pulses.shape[-1] != self.input_size
                or pulses.shape[:-1] != ticks.shape or pulses.shape[:2] != read_at.shape
                or state.shape != (pulses.size(1), self.width)):
            raise ValueError("Invalid pulse block dimensions")
        if ticks.dtype not in (torch.int32, torch.int64) or read_at.dtype not in (torch.int32, torch.int64):
            raise ValueError("Tick indices must be int32 or int64")
        if ticks.device != pulses.device or read_at.device != pulses.device:
            raise ValueError("Pulses and indices must share a device")
        if torch.any((ticks < -1) | (ticks >= block_ticks)) or torch.any((read_at < 0) | (read_at >= block_ticks)):
            raise ValueError("Tick outside its block")
        if not pulses.size(0):
            return state.new_empty(0, *state.shape), state

        transition = self.matrix()
        powers = [torch.eye(self.width, device=transition.device, dtype=transition.dtype), transition]
        for i in range(2, block_ticks + 1):
            powers.append(powers[i // 2] @ powers[i - i // 2])
        powers = torch.stack(powers)
        kernel = self.sensors @ powers

        def pulse_drive(lags: Tensor) -> Tensor:
            valid = (ticks >= 0) & (lags >= 0)
            # Mask the inputs too, so padding is ignored even if it contains NaN.
            observed = torch.where(valid.unsqueeze(-1), pulses, 0)
            matrices = kernel[lags.clamp(0, block_ticks - 1).long()]
            return (observed.unsqueeze(-2) @ matrices).squeeze(-2).sum(2)

        end_drives = pulse_drive(block_ticks - 1 - ticks)
        ends = _scan(end_drives, powers[block_ticks], state)
        before = torch.cat((state.unsqueeze(0), ends[:-1]))
        carried = (before.unsqueeze(-2) @ powers[read_at.long() + 1]).squeeze(-2)
        return carried + pulse_drive(read_at.unsqueeze(-1) - ticks), ends[-1]
