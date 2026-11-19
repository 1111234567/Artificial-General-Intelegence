"""Function-preserving native policy with a diagnostic signed-time readout.

The recurrent policy consumes only the original public observation.  Private
times enter target construction and loss computation only; the time readout
never participates in native action decoding or recurrent state updates.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.learn_remember.gru_train import GRUAgent as OriginalGRUAgent


ARMS = ("probe", "time", "action", "joint")
TIME_SCALE = 16.
N_ACTIONS = 11


def choose(logits: torch.Tensor) -> torch.Tensor:
    """Unfiltered original eleven-action argmax, including its tie rule."""
    if logits.ndim < 1 or logits.shape[-1] != N_ACTIONS:
        raise ValueError("native action logits must have final dimension 11")
    return logits.argmax(dim=-1)


class RemainingTimeGRUAgent(OriginalGRUAgent):
    """The unchanged native GRU/core/head plus one linear value per slot."""

    def __init__(self, width=192, input_size=56, n_jobs=2, n_actions=N_ACTIONS):
        if int(n_actions) != N_ACTIONS:
            raise ValueError("remaining-time policy requires 11 native actions")
        super().__init__(width=width, input_size=input_size, n_jobs=n_jobs,
                         n_actions=n_actions)
        self.time_head = nn.Linear(self.width, self.n_jobs)
        nn.init.zeros_(self.time_head.weight)
        nn.init.zeros_(self.time_head.bias)
        self.set_arm("joint")

    @classmethod
    def from_original(cls, original: OriginalGRUAgent, arm: str):
        """Copy both native modules exactly, with independent parameter storage."""
        if original.n_actions != N_ACTIONS:
            raise ValueError("source must be the original eleven-action GRU")
        reference = next(original.parameters())
        model = cls(**original.config()).to(device=reference.device,
                                             dtype=reference.dtype)
        model.core.load_state_dict(original.core.state_dict(), strict=True)
        model.head.load_state_dict(original.head.state_dict(), strict=True)
        model.set_arm(arm)
        model.train(original.training)
        return model

    def set_arm(self, arm: str):
        if arm not in ARMS:
            raise ValueError(f"unknown arm: {arm}")
        self.arm = arm
        self.core.requires_grad_(arm != "probe")
        self.head.requires_grad_(arm in ("action", "joint"))
        self.time_head.requires_grad_(True)
        return self

    def forward_with_time(self, obs, hidden, *, detach_time_features=None):
        """Return native logits, auxiliary values and state from one core pass.

        ``forward`` and ``step`` remain the inherited original implementations
        and do not compute the auxiliary head.  The action control detaches
        only the auxiliary features, retaining the ordinary action gradient.
        """
        if detach_time_features is None:
            detach_time_features = self.arm == "action"
        values, state = self.core(obs, hidden[None])
        logits = self.head(values).reshape(*obs.shape[:2], self.n_jobs,
                                           self.n_actions)
        features = values.detach() if detach_time_features else values
        return logits, self.time_head(features), state[0]

    choose = staticmethod(choose)

    def policy_parameters(self):
        """The clip group is core plus native action head, never the time head."""
        return [p for module in (self.core, self.head) for p in module.parameters()
                if p.requires_grad]

    def time_parameters(self):
        return [p for p in self.time_head.parameters() if p.requires_grad]

    def parameter_report(self):
        return dict(kind="plain_gru_native_action_auxiliary_remaining_time",
            **self.config(), arm=self.arm,
            total_parameters=sum(p.numel() for p in self.parameters()),
            trainable_parameters=sum(p.numel() for p in self.parameters()
                                     if p.requires_grad),
            core_parameters=sum(p.numel() for p in self.core.parameters()),
            head_parameters=sum(p.numel() for p in self.head.parameters()),
            time_head_parameters=sum(p.numel() for p in self.time_head.parameters()),
            state_scalars=self.width, public_inputs_only=True,
            initial_time_head="all weights and biases exactly zero",
            inference="original native 11-way argmax; auxiliary output diagnostic only")


def remaining_time_targets(ready: torch.Tensor, now, active: torch.Tensor,
                           started: torch.Tensor):
    """Signed physical remaining time /16, on all active started finite rows.

    No eligibility, future cancellation/replacement, deadline, or outcome is
    accepted.  A one-dimensional ``now`` is the leading time axis; scalars and
    already broadcastable clock tensors are also supported.  Masked targets
    are finite zero, so unstarted infinity sentinels cannot contaminate MSE.
    """
    if not torch.is_floating_point(ready):
        raise TypeError("ready must be floating point")
    if active.shape != ready.shape or started.shape != ready.shape:
        raise ValueError("ready, active and started shapes must match")
    if active.dtype != torch.bool or started.dtype != torch.bool:
        raise TypeError("active and started must be boolean masks")
    now = torch.as_tensor(now, dtype=ready.dtype, device=ready.device)
    if not bool(torch.isfinite(now).all()):
        raise ValueError("clock must be finite")
    if now.ndim == 1 and ready.ndim > 1 and now.shape[0] == ready.shape[0]:
        now = now.reshape(now.shape[0], *([1] * (ready.ndim - 1)))
    try:
        now = torch.broadcast_to(now, ready.shape)
    except RuntimeError as exc:
        raise ValueError("clock must broadcast to ready shape") from exc
    mask = active & started & torch.isfinite(ready)
    target = torch.zeros_like(ready)
    target[mask] = (ready[mask] - now[mask]) / TIME_SCALE
    return target, mask


def time_loss(prediction: torch.Tensor, target: torch.Tensor,
              mask: torch.Tensor):
    """Mean squared error divided by the actual active-started row count."""
    if prediction.shape != target.shape or mask.shape != prediction.shape:
        raise ValueError("time prediction, target and mask shapes must match")
    if mask.dtype != torch.bool:
        raise TypeError("time mask must be boolean")
    rows = mask.sum()
    if bool(rows):
        if not bool(torch.isfinite(target[mask]).all()):
            raise ValueError("supervised time targets must be finite")
        mse = F.mse_loss(prediction[mask], target[mask], reduction="mean")
    else:
        # An empty sum is a differentiable zero even with masked NaN sentinels.
        mse = prediction.reshape(-1)[:0].sum()
    return dict(total=mse, mse=mse, rows=rows)


def native_action_loss(logits: torch.Tensor, target: torch.Tensor,
                       wait_weight: float = .25):
    """The original CE: every slot, WAIT .25, sum-of-class-weights mean."""
    if logits.ndim < 2 or logits.shape[-1] != N_ACTIONS:
        raise ValueError("native action logits must have final dimension 11")
    if target.shape != logits.shape[:-1]:
        raise ValueError("action targets must match native score rows")
    if not math.isfinite(wait_weight) or wait_weight <= 0:
        raise ValueError("WAIT class weight must be finite and positive")
    weights = logits.new_ones(N_ACTIONS)
    weights[0] = wait_weight
    if target.numel() == 0:
        return logits.reshape(-1)[:0].sum()
    return F.cross_entropy(logits.reshape(-1, N_ACTIONS), target.reshape(-1),
                           weight=weights, reduction="mean")


def experiment_loss(logits: torch.Tensor, time_prediction: torch.Tensor,
                    action_target: torch.Tensor, time_target: torch.Tensor,
                    time_mask: torch.Tensor, arm: str,
                    wait_weight: float = .25):
    """Fixed unit coefficients; routing of auxiliary features is in forward."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    action = native_action_loss(logits, action_target, wait_weight)
    timing = time_loss(time_prediction, time_target, time_mask)
    total = timing["total"]
    if arm in ("action", "joint"):
        total = action + total
    return dict(total=total, action=action, time=timing["mse"],
                action_rows=action_target.new_tensor(action_target.numel()),
                time_rows=timing["rows"])


def clip_gradients(model: RemainingTimeGRUAgent, max_norm: float = 1.):
    """Clip policy and auxiliary-head gradients independently.

    This prevents a large detached-probe gradient from rescaling the action
    control's native gradient, even when both groups use the same Adam object.
    Returned values are the separate norms before clipping.
    """
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("gradient norm limit must be finite and positive")
    return dict(policy=torch.nn.utils.clip_grad_norm_(model.policy_parameters(),
                                                      max_norm),
                time_head=torch.nn.utils.clip_grad_norm_(model.time_parameters(),
                                                         max_norm))


# Agreed short alias for the training entrypoint.
loss = experiment_loss
