"""Finite, uniform storage of observed features and their feedback labels.

Call ``add`` only after the real target has been revealed. The buffer neither
generates targets nor keeps a model/autograd graph. Independent random keys
make the retained history independent of how an input batch is partitioned.
"""
from __future__ import annotations

import torch
from torch import Tensor


class Reservoir:
    """Keep the highest independent random priorities from all seen examples.

    Every subset of ``capacity`` historical examples is equally likely. Replay
    is uniform with replacement, so a small buffer can supply a full batch.
    Two private CPU generators separate admission from replay, including when
    storage uses CUDA. Replay frequency therefore cannot change retention.
    """

    def __init__(self, capacity: int, feature_dim: int, device: str, seed: int):
        if capacity < 0:
            raise ValueError("capacity must be nonnegative")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.device = torch.device(device)
        self.seen = 0
        self._rng = torch.Generator(device="cpu").manual_seed(seed)
        self._sample_rng = torch.Generator(device="cpu").manual_seed((seed + 1) % (2**64))
        self._features = torch.empty((0, feature_dim), device=self.device)
        self._targets = torch.empty(0, dtype=torch.long, device=self.device)
        self._priorities = torch.empty(0, dtype=torch.float64, device=self.device)
        # Keep this counter on the storage device to avoid a CUDA sync per add.
        self._replacements = torch.zeros((), dtype=torch.long, device=self.device)

    def __len__(self) -> int:
        return self._targets.shape[0]

    @property
    def replacements(self) -> int:
        """Number of previously retained records evicted by later additions."""
        return int(self._replacements.item())

    @property
    def bytes(self) -> int:
        """Bytes occupied by stored features, targets and random priorities."""
        return sum(t.numel() * t.element_size()
                   for t in (self._features, self._targets, self._priorities))

    @torch.no_grad()
    def add(self, features: Tensor, targets: Tensor) -> None:
        """Copy a batch of real observations after its feedback is available."""
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f"features must have shape (batch, {self.feature_dim})")
        if targets.ndim != 1 or targets.shape[0] != features.shape[0]:
            raise ValueError("targets must have shape (batch,)")
        count = features.shape[0]
        if not count:
            return
        old_size = len(self)
        if old_size and (features.dtype != self._features.dtype or
                         targets.dtype != self._targets.dtype):
            raise ValueError("feature and target dtypes must stay consistent")

        incoming_features = features.detach().to(self.device)
        incoming_targets = targets.detach().to(self.device)
        incoming_priorities = torch.rand(
            count, dtype=torch.float64, generator=self._rng, device="cpu",
        ).to(self.device)
        priorities = torch.cat((self._priorities, incoming_priorities))
        kept_size = min(self.capacity, old_size + count)
        keep = priorities.topk(kept_size, sorted=True).indices
        if old_size:
            all_features = torch.cat((self._features, incoming_features))
            all_targets = torch.cat((self._targets, incoming_targets))
        else:
            all_features, all_targets = incoming_features, incoming_targets

        # index_select copies storage, including when every incoming row fits.
        self._features = all_features.index_select(0, keep)
        self._targets = all_targets.index_select(0, keep)
        self._priorities = priorities.index_select(0, keep)
        if old_size + count > self.capacity:
            self._replacements.add_(old_size - (keep < old_size).sum())
        self.seen += count

    @torch.no_grad()
    def sample(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """Return copied records, drawing only from the populated buffer."""
        if batch_size < 0:
            raise ValueError("batch_size must be nonnegative")
        if batch_size == 0:
            return self._features[:0].clone(), self._targets[:0].clone()
        if not len(self):
            raise ValueError("cannot sample an empty reservoir")
        indices = torch.randint(
            len(self), (batch_size,), generator=self._sample_rng, device="cpu",
        ).to(self.device)
        return (self._features.index_select(0, indices),
                self._targets.index_select(0, indices))

    @torch.no_grad()
    def state_dict(self) -> dict:
        """Return independent copies of the populated buffer and RNG state."""
        return {
            "capacity": self.capacity,
            "feature_dim": self.feature_dim,
            "features": self._features.clone(),
            "targets": self._targets.clone(),
            "priorities": self._priorities.clone(),
            "seen": self.seen,
            "replacements": self.replacements,
            "rng_state": self._rng.get_state().clone(),
            "sample_rng_state": self._sample_rng.get_state().clone(),
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict) -> None:
        """Restore a saved buffer onto this instance's storage device."""
        if (state["capacity"] != self.capacity or
                state["feature_dim"] != self.feature_dim):
            raise ValueError("saved reservoir dimensions do not match")
        features, targets, priorities = (
            state["features"], state["targets"], state["priorities"],
        )
        size = targets.shape[0]
        if (targets.ndim != 1 or size > self.capacity or
                features.shape != (size, self.feature_dim) or
                priorities.shape != (size,) or
                priorities.dtype != torch.float64 or state["seen"] < size):
            raise ValueError("invalid saved reservoir contents")
        rng = torch.Generator(device="cpu")
        rng.set_state(state["rng_state"].cpu())
        sample_rng = torch.Generator(device="cpu")
        sample_rng.set_state(state["sample_rng_state"].cpu())
        self._features = features.detach().to(self.device).clone()
        self._targets = targets.detach().to(self.device).clone()
        self._priorities = priorities.detach().to(self.device).clone()
        self.seen = int(state["seen"])
        self._replacements.fill_(state["replacements"])
        self._rng = rng
        self._sample_rng = sample_rng
