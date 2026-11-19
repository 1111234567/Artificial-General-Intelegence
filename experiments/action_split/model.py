"""A plain GRU with separate learned action-mode and conditional-content heads.

The core consumes the same public observations as the original GRU.  Decoding
always produces one of the world's original eleven action codes; no state of
the world, legality test, clock, or oracle is consulted by ``choose``.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from experiments.learn_remember.gru_train import GRUAgent as OriginalGRUAgent


MODE = slice(0, 3)
BEGIN = slice(3, 5)
FINISH = slice(5, 13)
N_SCORES = 13
WAIT_MODE, BEGIN_MODE, FINISH_MODE = 0, 1, 2


def choose(scores: torch.Tensor) -> torch.Tensor:
    """Group-first argmax, followed by native BEGIN/FINISH content decoding.

    Argmax uses PyTorch's first-index tie rule.  Consequently all-zero new
    heads choose WAIT, including when individual conditional scores tie.
    """
    if scores.ndim < 1 or scores.shape[-1] != N_SCORES:
        raise ValueError("split action scores must have final dimension 13")
    mode = scores[..., MODE].argmax(dim=-1)
    begin = 1 + scores[..., BEGIN].argmax(dim=-1)
    finish = 3 + scores[..., FINISH].argmax(dim=-1)
    return torch.where(mode == BEGIN_MODE, begin,
                       torch.where(mode == FINISH_MODE, finish, torch.zeros_like(mode)))


class SplitGRUAgent(OriginalGRUAgent):
    """The original recurrent architecture with 13 scores for each job."""

    def __init__(self, width=192, input_size=56, n_jobs=2, n_actions=N_SCORES):
        if int(n_actions) != N_SCORES:
            raise ValueError("split policy requires 13 scores per job")
        super().__init__(width=width, input_size=input_size, n_jobs=n_jobs,
                         n_actions=N_SCORES)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @classmethod
    def from_original(cls, original: OriginalGRUAgent, frozen_core: bool):
        """Copy the pretrained core only, without sharing parameter storage.

        The original action head is deliberately not transferred.  Both
        training conditions start with exactly the same zero split head.
        """
        if original.n_actions != 11:
            raise ValueError("source must be the original eleven-action GRU")
        reference = next(original.core.parameters())
        model = cls(width=original.width, input_size=original.input_size,
                    n_jobs=original.n_jobs).to(device=reference.device,
                                             dtype=reference.dtype)
        model.core.load_state_dict(original.core.state_dict(), strict=True)
        model.set_core_frozen(frozen_core)
        return model

    def set_core_frozen(self, frozen: bool):
        self.core.requires_grad_(not frozen)
        self.head.requires_grad_(True)
        return self

    choose = staticmethod(choose)

    def parameter_report(self):
        return dict(kind="plain_single_layer_gru_split_action", **self.config(),
                    native_action_codes=11, mode_scores=3, begin_scores=2,
                    finish_scores=8,
                    total_parameters=sum(p.numel() for p in self.parameters()),
                    trainable_parameters=sum(p.numel() for p in self.parameters()
                                             if p.requires_grad),
                    core_parameters=sum(p.numel() for p in self.core.parameters()),
                    head_parameters=sum(p.numel() for p in self.head.parameters()),
                    state_scalars=self.width, public_inputs_only=True,
                    core_frozen=not any(p.requires_grad for p in self.core.parameters()),
                    initial_head="all weights and biases exactly zero")


def split_loss(scores: torch.Tensor, mode_target: torch.Tensor,
               begin_target: torch.Tensor, finish_target: torch.Tensor,
               active: torch.Tensor, wait_weight: float = .25):
    """Fixed sum of weighted mode, conditional BEGIN, and dense content losses.

    Mode CE uses every row and divides by the sum of class weights, with
    weights [wait_weight, 1, 1].  BEGIN CE averages only teacher-BEGIN rows.
    FINISH-content CE averages *all active rows*, including processing WAIT
    and requests that will later be canceled/replaced.  No eligibility input
    is accepted.  Empty conditional masks produce differentiable exact zero.
    """
    if scores.ndim < 2 or scores.shape[-1] != N_SCORES:
        raise ValueError("split action scores must have final dimension 13")
    shape = scores.shape[:-1]
    if any(value.shape != shape for value in
           (mode_target, begin_target, finish_target, active)):
        raise ValueError("all targets and active mask must match score rows")
    if active.dtype != torch.bool:
        raise TypeError("active must be a boolean mask")
    if wait_weight <= 0:
        raise ValueError("WAIT class weight must be positive")
    flat = scores.reshape(-1, N_SCORES)
    mode_target = mode_target.reshape(-1)
    begin_target = begin_target.reshape(-1)
    finish_target = finish_target.reshape(-1)
    active = active.reshape(-1)
    begin_mask = mode_target == BEGIN_MODE
    mode_rows = mode_target.numel()
    begin_rows, finish_rows = begin_mask.sum(), active.sum()
    weights = scores.new_tensor([wait_weight, 1., 1.])
    if mode_rows:
        mode = F.cross_entropy(flat[:, MODE], mode_target, weight=weights,
                               reduction="mean")
        mode_weight_sum = weights[mode_target].sum()
    else:
        mode = flat[:, MODE].sum() * 0.
        mode_weight_sum = scores.new_zeros(())
    begin = (F.cross_entropy(flat[begin_mask, BEGIN], begin_target[begin_mask])
             if bool(begin_rows) else flat[:, BEGIN].sum() * 0.)
    finish = (F.cross_entropy(flat[active, FINISH], finish_target[active])
              if bool(finish_rows) else flat[:, FINISH].sum() * 0.)
    return dict(total=mode + begin + finish, mode=mode, begin=begin, finish=finish,
                mode_rows=mode_target.new_tensor(mode_rows), begin_rows=begin_rows,
                finish_rows=finish_rows, mode_weight_sum=mode_weight_sum)


# A short alias keeps the agreed experiment API explicit for callers.
loss = split_loss
