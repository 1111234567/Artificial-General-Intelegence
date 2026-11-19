"""Small symbol-association controls, not a full MemT reproduction.

The default eight public keys encode two contexts times four raw symbols.
Values supplied to ``write`` are the observed labels of public examples.
The model has no answer table, private mapping input, or direct key-to-answer
head: every query reads the same explicit, 64-scalar state matrix.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


KINDS = ("add", "delta", "recurrent")


class AssociationMemory(nn.Module):
    """Matched additive/delta memories and a larger recurrent control.

    For a unit key ``k``, public one-hot value ``v``, and matrix ``M``, the
    additive update is ``M + eta * k v^T``.  The delta update is
    ``M + eta * k (v - M^T k)^T``.  ``eta`` is a learned sigmoid scalar.
    The recurrent control uses a GRUCell on ``[k, v]`` and the flattened
    matrix as its hidden state.  All three read ``scale * M^T k`` with a
    learned positive softplus scale.  Seeds are controlled by the caller.
    """

    def __init__(self, kind="add", key_dim=16, n_keys=8, n_values=4):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        for name, number in (("key_dim", key_dim), ("n_keys", n_keys),
                             ("n_values", n_values)):
            if isinstance(number, bool) or not isinstance(number, int) or number < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.kind = kind
        self.key_dim = key_dim
        self.n_keys = n_keys
        self.n_values = n_values
        self.state_dim = key_dim * n_values
        self.key_embedding = nn.Embedding(n_keys, key_dim)
        if kind != "recurrent":
            self.eta_logit = nn.Parameter(torch.tensor(0.))
        self.read_scale_raw = nn.Parameter(torch.tensor(math.log(math.expm1(5.))))
        if kind == "recurrent":
            self.core = nn.GRUCell(key_dim + n_values, self.state_dim)

    @property
    def eta(self):
        """Differentiable update rate; the GRU control has no eta parameter."""
        return None if self.kind == "recurrent" else self.eta_logit.sigmoid()

    @property
    def read_scale(self):
        return F.softplus(self.read_scale_raw)

    def initial_state(self, batch, device=None):
        """Return zero state on the module's device/dtype unless overridden."""
        if isinstance(batch, bool) or not isinstance(batch, int) or batch < 0:
            raise ValueError("batch must be a nonnegative integer")
        reference = self.key_embedding.weight
        return torch.zeros(batch, self.state_dim,
                           device=reference.device if device is None else device,
                           dtype=reference.dtype)

    def _check_state_keys(self, state, keys):
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError(f"state must have shape [batch, {self.state_dim}]")
        if keys.ndim != 1 or keys.shape[0] != state.shape[0]:
            raise ValueError("keys must have shape [batch]")
        if keys.dtype != torch.long:
            raise TypeError("keys must be torch.long integer indices")
        if keys.device != state.device:
            raise ValueError("keys and state must be on the same device")
        if state.device != self.key_embedding.weight.device:
            raise ValueError("state and model must be on the same device")
        if state.dtype != self.key_embedding.weight.dtype:
            raise TypeError("state and model must have the same floating dtype")

    def _key(self, keys):
        return F.normalize(self.key_embedding(keys), dim=-1)

    def read(self, state, keys):
        """Read only the explicit state, yielding [batch, n_values] logits."""
        self._check_state_keys(state, keys)
        matrix = state.reshape(-1, self.key_dim, self.n_values)
        query = self._key(keys)
        return self.read_scale * torch.bmm(query[:, None, :], matrix)[:, 0, :]

    def _write_selected(self, state, keys, values):
        key = self._key(keys)
        value = F.one_hot(values, num_classes=self.n_values).to(dtype=state.dtype)
        if self.kind == "recurrent":
            return self.core(torch.cat((key, value), dim=-1), state)
        matrix = state.reshape(-1, self.key_dim, self.n_values)
        if self.kind == "delta":
            prediction = torch.bmm(key[:, None, :], matrix)[:, 0, :]
            value = value - prediction
        update = self.eta * key[:, :, None] * value[:, None, :]
        return (matrix + update).reshape_as(state)

    def write(self, state, keys, values, mask=None):
        """Write observed public examples, preserving every unselected lane.

        Only selected indices are looked up.  Masked lanes may therefore use
        sentinel key/value indices; they are not inputs to a state update.
        Neither this operation nor ``read`` mutates its input state.
        """
        self._check_state_keys(state, keys)
        if values.ndim != 1 or values.shape[0] != state.shape[0]:
            raise ValueError("values must have shape [batch]")
        if values.dtype != torch.long:
            raise TypeError("values must be torch.long integer indices")
        if values.device != state.device:
            raise ValueError("values and state must be on the same device")
        if mask is None:
            return self._write_selected(state, keys, values)
        if mask.shape != keys.shape or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean with shape [batch]")
        if mask.device != state.device:
            raise ValueError("mask and state must be on the same device")
        indices = mask.nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            return state
        updated = self._write_selected(state[indices], keys[indices], values[indices])
        return state.index_copy(0, indices, updated)

    def config(self):
        return dict(kind=self.kind, key_dim=self.key_dim, n_keys=self.n_keys,
                    n_values=self.n_values)

    def parameter_report(self):
        by_name = {name: value.numel() for name, value in self.named_parameters()}
        eta = self.eta
        return dict(**self.config(),
                    total_parameters=sum(by_name.values()),
                    trainable_parameters=sum(p.numel() for p in self.parameters()
                                             if p.requires_grad),
                    embedding_parameters=self.key_embedding.weight.numel(),
                    update_parameters=sum(size for name, size in by_name.items()
                                          if name == "eta_logit" or name.startswith("core.")),
                    read_scale_parameters=1,
                    state_scalars=self.state_dim,
                    parameters_by_name=by_name,
                    eta=None if eta is None else float(eta.detach().cpu()),
                    read_scale=float(self.read_scale.detach().cpu()))
