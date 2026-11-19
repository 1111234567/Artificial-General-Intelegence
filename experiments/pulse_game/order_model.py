"""Matched leaky sensory circuits and public action–outcome training."""

import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import order_world, world


WIDTH = 32


@torch.jit.script
def update(projected: torch.Tensor, state: torch.Tensor, gain: torch.Tensor,
           edges: torch.Tensor, reset: torch.Tensor, intervention: int):
    states = torch.jit.annotate(list[torch.Tensor], [])
    for i in range(projected.size(0)):
        if intervention == 2:
            state = torch.where(reset[i].unsqueeze(-1), 0., state)
        candidate, leak = projected[i].chunk(2, dim=-1)
        if intervention == 3:
            state = torch.tanh(candidate)
        else:
            keep = torch.sigmoid(leak)
            if intervention != 1:
                candidate = candidate + state.index_select(-1, edges) * gain
            state = keep * state + (1. - keep) * torch.tanh(candidate)
        states.append(state)
    return torch.stack(states), state


class Circuit(nn.Module):
    def __init__(self, birth, topology, width=WIDTH):
        super().__init__()
        if topology not in ("graph", "self"):
            raise ValueError(f"Unknown topology {topology}")
        rng = torch.Generator().manual_seed(birth)
        self.width = width
        self.drive = nn.Parameter(torch.randn(order_world.SENSES, 2 * width, generator=rng) * .35)
        self.bias = nn.Parameter(torch.cat((torch.zeros(width), torch.full((width,), 1.8))))
        self.gain = nn.Parameter(.82 + .07 * torch.randn(width, generator=rng))
        self.readout = nn.Parameter(torch.randn(width, generator=rng) / math.sqrt(width))
        self.offset = nn.Parameter(torch.zeros(()))
        indices = torch.arange(width)
        self.register_buffer("edges", ((indices - 1) % width if topology == "graph" else indices),
                             persistent=False)

    def start(self, lives):
        return self.drive.new_zeros((lives, self.width))

    def forward(self, observation, state, intervention=0, reset=None):
        if reset is None:
            reset = torch.empty((0, 0), dtype=torch.bool, device=observation.device)
        states, state = update(observation @ self.drive + self.bias, state, self.gain,
                               self.edges, reset, intervention)
        return states @ self.readout + self.offset, state


@torch.no_grad()
def run(circuit, tape, intervention=0):
    """The controller advances every tick, including across contact and request boundaries."""
    state = circuit.start(tape.body.n)
    sensory_ticks = tape.timing.sensory_ticks
    decision = []
    for req, observations in enumerate(tape.observations):
        reset = None
        if intervention == 2:
            reset = torch.zeros(observations.shape[:2], dtype=torch.bool)
            reset[tape.target_ticks[:, req, 1], torch.arange(tape.body.n)] = True
        logits, state = circuit(torch.from_numpy(observations), state, intervention, reset)
        decision.append(logits[sensory_ticks:sensory_ticks + world.TICKS].numpy() >= 0)
    return order_world.physics(tape, np.stack(decision))


def pool(tape):
    """The only training targets come from the exploratory body's observed receipts."""
    physical = order_world.explore(tape)
    side, taste = order_world.contact_receipts(physical)
    return (np.ascontiguousarray(tape.observations.transpose(2, 0, 1, 3)), side.T, taste.T,
            tape.timing.sensory_ticks)


def train(circuit, pools, *, seed, rate, checkpoints, batch_size, steps_per_step_record,
          validate, first=0, optimizer_state=None):
    """Fit chosen-action taste equally in both arms using continuous two-request examples."""
    optimizer = torch.optim.Adam(circuit.parameters(), lr=rate)
    rng = np.random.default_rng(seed)
    history = []
    for iteration in range(1, max(checkpoints) + 1):
        pool_id = (iteration - 1) % len(pools)
        observations, side, taste, sensory_ticks = pools[pool_id]
        lives = rng.choice(observations.shape[0], batch_size, replace=False)
        previous = rng.integers(0, observations.shape[1] - 1, batch_size)
        if iteration <= first:
            continue
        if iteration == first + 1 and optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

        begin = time.perf_counter()
        # The preceding public request provides the initial state. Gradients span
        # the entire scored request; its preceding request still gets fully experienced.
        with torch.no_grad():
            warm = torch.from_numpy(observations[lives, previous].transpose(1, 0, 2))
            _, state = circuit(warm, circuit.start(len(lives)))
        index = previous + 1
        x = torch.from_numpy(observations[lives, index].transpose(1, 0, 2))
        public_side = torch.from_numpy(side[lives, index])
        receipt = torch.from_numpy(taste[lives, index])
        if not torch.any(receipt != 0):
            continue

        prediction, _ = circuit(x, state)
        movement = prediction[sensory_ticks:sensory_ticks + world.MOVEMENT_TICKS, receipt != 0]
        action_sign = (2 * public_side[receipt != 0] - 1).to(movement.dtype)
        targets = (receipt[receipt != 0] > 0).to(movement.dtype).expand_as(movement)
        loss = F.binary_cross_entropy_with_logits(movement * action_sign, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iteration % steps_per_step_record == 0:
            history.append({"step": iteration, "loss": loss.item(),
                            "update_seconds": time.perf_counter() - begin})
        if iteration in checkpoints:
            validate(circuit, iteration, history, optimizer.state_dict())
    return history
