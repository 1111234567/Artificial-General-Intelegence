"""Small rule-token policies for comparing internal computation depth.

Only current inventory, goal and rules enter the network. The public simulator
supplies a current applicability mask; it does not filter future dead ends.
There is no recurrent memory across environment actions and no learned halt.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .world import Problem, State


def tensorize(
    problems: Sequence[Problem],
    states: Sequence[State] | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode public information, never problem metadata or solver labels."""
    if not problems:
        raise ValueError("cannot tensorize an empty batch")
    shape = (problems[0].n_actions, problems[0].n_items)
    if any((p.n_actions, p.n_items) != shape for p in problems):
        raise ValueError("all problems in a batch must have the same dimensions")
    if states is None:
        states = [p.initial for p in problems]
    if len(states) != len(problems):
        raise ValueError("one inventory per problem is required")
    consume = torch.tensor([p.consume for p in problems], dtype=torch.float32, device=device)
    produce = torch.tensor([p.produce for p in problems], dtype=torch.float32, device=device)
    inventory = torch.tensor(states, dtype=torch.float32, device=device)
    goal = torch.tensor([p.goal for p in problems], dtype=torch.float32, device=device)
    if inventory.shape != goal.shape or (inventory < 0).any():
        raise ValueError("inventory dimensions/counts are invalid")
    legal = (inventory[:, None, :] >= consume).all(dim=-1)
    features = torch.cat((consume, produce,
                          inventory[:, None, :].expand_as(consume),
                          goal[:, None, :].expand_as(consume),
                          legal.to(consume.dtype).unsqueeze(-1)), dim=-1)
    return features, legal


class Planner(nn.Module):
    """Shared or untied Transformer computation with identical input access.

    h starts at zero for EVERY decision. Each round receives the same static
    rule encoding and the previous hidden state. Detach changes gradients only,
    retaining a direct input-encoder path at every round. Rule order has no
    positional embedding. Item coordinates are randomly relabeled by the task;
    the network is not guaranteed to be equivariant to item relabeling.
    """

    def __init__(self, n_items: int = 40, width: int = 128, rounds: int = 4,
                 kind: str = "recurrent") -> None:
        super().__init__()
        if kind not in {"shallow", "recurrent", "untied", "detach"}:
            raise ValueError(f"unknown planner kind: {kind}")
        if rounds < 1 or width < 4 or width % 4:
            raise ValueError("rounds must be positive; width must be divisible by four")
        self.kind = kind
        self.rounds = 1 if kind == "shallow" else rounds
        self.width = width
        self.n_items = n_items
        # Create these before cores so reseeding shares their initialization
        # even when an untied model subsequently allocates additional cores.
        self.encoder = nn.Linear(4 * n_items + 1, width)
        self.input_norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.cores = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=width, nhead=4, dim_feedforward=2 * width,
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(self.rounds if kind == "untied" else 1)
        ])

    def forward(self, features: torch.Tensor, legal: torch.Tensor,
                rounds: int | None = None) -> torch.Tensor:
        count = self.rounds if rounds is None else rounds
        if count < 1 or (self.kind == "shallow" and count != 1):
            raise ValueError("invalid number of computation rounds")
        if self.kind == "untied" and count > len(self.cores):
            raise ValueError("untied model has no additional layers")
        if features.ndim != 3 or features.shape[-1] != 4 * self.n_items + 1:
            raise ValueError("features must be [batch, actions, 4*n_items+1]")
        if legal.shape != features.shape[:2] or legal.dtype != torch.bool:
            raise ValueError("legal must be a boolean [batch, actions] mask")
        x = self.encoder(features)
        h = torch.zeros_like(x)
        for index in range(count):
            if self.kind == "detach" and index:
                h = h.detach()
            core = self.cores[index] if self.kind == "untied" else self.cores[0]
            h = core(self.input_norm(h + x))
        logits = self.head(h).squeeze(-1)
        # Illegal rules still participate as context: a future prerequisite may
        # be currently unavailable. Only the final action choice is masked.
        return logits.masked_fill(~legal, -torch.inf)


def optimal_set_loss(logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """Negative log probability mass of ALL shortest-plan next actions."""
    if target_mask.shape != logits.shape or target_mask.dtype != torch.bool:
        raise ValueError("target_mask must be boolean and match logits")
    if not target_mask.any(dim=-1).all():
        raise ValueError("every training state needs at least one optimal action")
    selected = logits.masked_fill(~target_mask, -torch.inf)
    if not torch.isfinite(selected).any(dim=-1).all():
        raise ValueError("optimal targets must include a legal action")
    return (torch.logsumexp(logits, dim=-1) - torch.logsumexp(selected, dim=-1)).mean()
