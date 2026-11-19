"""Native GRU policies, shared recurrent weights, and supervised sequence fitting."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

from .training import train_chunk


class SharedPairs(nn.Module):
    """Expand one trainable value per unordered pair and gate, including self-pairs."""

    def __init__(self, width: int):
        super().__init__()
        i, j = torch.triu_indices(width, width)
        index = torch.empty(width, width, dtype=torch.long)
        pair = torch.arange(i.numel())
        index[i, j] = pair
        index[j, i] = pair
        self.register_buffer("index", index)
        self.register_buffer("i", i)
        self.register_buffer("j", j)

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        width = self.index.shape[0]
        return packed[:, self.index].reshape(3 * width, width).contiguous()

    def right_inverse(self, dense: torch.Tensor) -> torch.Tensor:
        width = self.index.shape[0]
        return dense.reshape(3, width, width)[:, self.i, self.j].contiguous()


class GRU(nn.Module):
    """Native one-layer GRU, returning hidden activities for any task readout."""

    def __init__(self, input_size: int, width: int):
        super().__init__()
        self.width, self.input_size = int(width), int(input_size)
        self.core = nn.GRU(self.input_size, self.width, num_layers=1)

    def initial_state(self, batch_size: int, device=None) -> torch.Tensor:
        return self.core.weight_ih_l0.new_zeros(batch_size, self.width, device=device)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor):
        values, state = self.core(obs, hidden[None])
        return values, state[0]

    def step(self, obs: torch.Tensor, state: torch.Tensor, dt=None, max_step=None):
        output, state = self.forward(obs[None], state)
        return output[0], state


class GRUAgent(GRU):
    """Single-layer native GRU with one categorical readout per job.

    Batches use [time, batch, feature] input, [batch, width] state and
    [time, batch, job, action] logits. Both entry points return the next state.
    The constructor and parameter names match the learn_remember checkpoints.
    """

    def __init__(self, width: int = 192, input_size: int = 56,
                 n_jobs: int = 2, n_actions: int = 11):
        super().__init__(input_size, width)
        self.n_jobs, self.n_actions = int(n_jobs), int(n_actions)
        self.head = nn.Linear(self.width, self.n_jobs * self.n_actions)

    def config(self) -> dict[str, int]:
        return dict(width=self.width, input_size=self.input_size,
                    n_jobs=self.n_jobs, n_actions=self.n_actions)

    def initial_state(self, batch_size: int, device=None) -> torch.Tensor:
        return self.head.weight.new_zeros(batch_size, self.width, device=device)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor):
        values, state = super().forward(obs, hidden)
        logits = self.head(values).reshape(*obs.shape[:2], self.n_jobs, self.n_actions)
        return logits, state

    def parameter_report(self):
        recurrent = getattr(self.core, "parametrizations", {})
        shared = "weight_hh_l0" in recurrent and isinstance(recurrent["weight_hh_l0"][0], SharedPairs)
        report = dict(kind="shared_recurrent_gru" if shared else "plain_single_layer_gru",
                      **self.config(), total_parameters=sum(p.numel() for p in self.parameters()),
                      state_scalars=self.width, public_inputs_only=True)
        if shared:
            report["unique_hidden_coordinates"] = 3 * self.width * (self.width + 1) // 2
        return report


def share_recurrent_weights(model: GRU) -> GRU:
    """Tie the recurrent matrices in place before creating the optimizer.

    The upper triangles supply the initial values, matching symmetric_gru.
    The GRU's input matrices, biases and readout retain their ordinary weights.
    """
    if parametrize.is_parametrized(model.core, "weight_hh_l0"):
        raise ValueError("recurrent weights already have a parametrization")
    rule = SharedPairs(model.width).to(model.core.weight_hh_l0.device)
    parametrize.register_parametrization(model.core, "weight_hh_l0", rule, unsafe=True)
    return model


def deploy_dense_gru(model: GRU, *, device="cpu") -> GRU:
    """Export an ordinary GRU with dense weights and the historical key layout.

    The output is a separate model. Keep the packed training state and its
    optimizer state when resuming shared-weight training.
    """
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        if isinstance(model, GRUAgent):
            public = GRUAgent(**model.config()).to(dtype=model.head.weight.dtype)
        else:
            public = GRU(model.input_size, model.width).to(dtype=model.core.weight_ih_l0.dtype)
    weights = {key: value for key, value in model.state_dict().items()
               if not key.startswith("core.parametrizations.weight_hh_l0.")}
    weights["core.weight_hh_l0"] = model.core.weight_hh_l0.detach()
    public.load_state_dict(weights)
    return public.to(device=device)


def fit_gru_sequence(model: GRUAgent, optimizer: torch.optim.Optimizer,
                     observations: torch.Tensor, targets: torch.Tensor, anchor: torch.Tensor,
                     *, block_size: int, epochs: int = 1,
                     class_weights: torch.Tensor | None = None, max_grad_norm: float | None = 1.0):
    """Train on an ordered trajectory with truncation between contiguous blocks.

    Targets contain integer class indices with shape [time, batch, job]. Each
    epoch starts from the same observed anchor. The returned hidden state is
    detached and comes from the last training forward in the last epoch.
    """
    if block_size < 1 or epochs < 1:
        raise ValueError("block_size and epochs must be positive")
    if observations.ndim != 3 or not observations.numel() or observations.shape[2] != model.input_size:
        raise ValueError("observations must have shape [time, batch, input_size] with nonempty axes")
    ticks, batch = observations.shape[:2]
    if targets.shape != (ticks, batch, model.n_jobs) or targets.dtype != torch.long:
        raise ValueError("targets must have shape [time, batch, n_jobs] and dtype torch.long")
    if anchor.shape != (batch, model.width):
        raise ValueError("anchor must have shape [batch, width]")

    device = model.head.weight.device
    observations, targets = observations.to(device), targets.to(device)
    anchor = anchor.detach().to(device)
    weights = None
    if class_weights is not None:
        if class_weights.shape != (model.n_actions,):
            raise ValueError("one class weight per action is required")
        weights = class_weights.to(device=device, dtype=model.head.weight.dtype)

    model.train()
    loss_total = 0.0
    correct = updates = 0
    chunk_logits = None

    def supervised_loss(logits):
        nonlocal chunk_logits
        chunk_logits = logits.detach()
        return F.cross_entropy(logits.flatten(0, 2), labels.flatten(), weight=weights)

    for _ in range(epochs):
        hidden = anchor
        for start in range(0, ticks, block_size):
            stop = start + block_size
            labels = targets[start:stop]
            loss, hidden = train_chunk(model, optimizer, observations[start:stop], hidden,
                                       supervised_loss, max_grad_norm=max_grad_norm)
            loss_total += float(loss)
            correct += int((chunk_logits.argmax(-1) == labels).sum())
            updates += 1

    return dict(mean_loss=loss_total / updates,
                training_all_action_accuracy=correct / (epochs * targets.numel()),
                gradient_updates=updates, gradient_lane_ticks=epochs * ticks * batch,
                hidden=hidden)


def attach_frozen_encoder_cache(model: nn.Module):
    """Attach the audited cache to a frozen native ``model.encoder`` during CPU training.

    Use as a context manager with read-only inputs. The implementation bypasses
    cached results for unsupported dtypes, devices, state and gradient modes.
    """
    from experiments.seed_evolution.encoder_cache import attach_frozen_encoder_cache as attach

    return attach(model)
