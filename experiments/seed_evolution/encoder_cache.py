"""Candidate-local reuse of frozen GRU traces during eager CPU CE training."""
from __future__ import annotations

from types import MethodType
from weakref import ref

import torch
from torch import nn


_MAX_STREAMS = 16


def _tensor_key(tensor):
    return (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tensor.shape,
            tensor.stride(), tensor.dtype, tensor.device, tensor._version)


def _cpu_float32(tensor):
    return (type(tensor) is torch.Tensor and tensor.device.type == "cpu"
            and tensor.dtype == torch.float32 and tensor.layout == torch.strided
            and not tensor.requires_grad and not tensor.is_inference()
            and not tensor.is_conj() and not tensor.is_neg())


class FrozenEncoderCache:
    """Cache one candidate's GRU outputs for the lifetime of its training call."""

    def __init__(self, model):
        encoder = model.encoder
        if type(encoder) is not nn.GRU or "forward" in vars(encoder):
            raise ValueError("the cache requires an unmodified nn.GRU encoder")
        self._model = ref(model)
        self._encoder = ref(encoder)
        self._entries = {}
        self._signature = None
        self._hits = self._misses = self._bypasses = self._peak_bytes = 0
        encoder.forward = MethodType(self._forward, encoder)

    @property
    def stats(self):
        return dict(hits=self._hits, misses=self._misses, bypasses=self._bypasses,
                    streams=len(self._entries), bytes=self._bytes(),
                    peak_bytes=self._peak_bytes)

    def _bytes(self):
        return sum(tensor.untyped_storage().nbytes()
                   for _, result in self._entries.values() for tensor in result)

    def close(self):
        encoder = self._encoder()
        if encoder is not None and getattr(encoder.forward, "__func__", None) == self._forward:
            del encoder.forward
        self._entries.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _forward(self, encoder, input, hx=None):
        model = self._model()
        if (model is None or not model.training or not encoder.training or encoder.dropout
                or not torch.is_grad_enabled() or torch.is_autocast_enabled("cpu")
                or torch.autograd.forward_ad._current_level >= 0
                or torch._C._are_functorch_transforms_active()
                or not _cpu_float32(input) or input.ndim != 3 or not input.numel()):
            self._bypasses += 1
            return nn.GRU.forward(encoder, input, hx)

        hidden_key = None
        if hx is not None:
            # The integer view also excludes negative zero and accepts only the
            # layout used by the policy's freshly initialized recurrent state.
            if (not _cpu_float32(hx) or not hx.is_contiguous() or hx.storage_offset()
                    or hx.ndim != 3 or torch.count_nonzero(hx.view(torch.int32)).item()):
                self._bypasses += 1
                return nn.GRU.forward(encoder, input, hx)
            hidden_key = (hx.shape, hx.stride())

        params = tuple(encoder.parameters())
        if any(p.requires_grad or p.is_inference() for p in params):
            self._bypasses += 1
            return nn.GRU.forward(encoder, input, hx)
        signature = (tuple((id(p), _tensor_key(p)) for p in params),
                     encoder.input_size, encoder.hidden_size, encoder.num_layers,
                     encoder.batch_first, encoder.bidirectional, encoder.bias,
                     torch.get_num_threads(), torch.get_float32_matmul_precision(),
                     torch.backends.mkldnn.enabled, torch.are_deterministic_algorithms_enabled())
        if signature != self._signature:
            self._entries.clear()
            self._signature = signature

        key = (_tensor_key(input), hidden_key)
        entry = self._entries.get(key)
        if entry is not None and entry[0]._version == input._version:
            self._hits += 1
            return tuple(tensor.clone() for tensor in entry[1])
        self._misses += 1
        result = nn.GRU.forward(encoder, input, hx)
        if len(self._entries) >= _MAX_STREAMS:
            return result
        # Retain the input view so a new allocation cannot reuse its address.
        # GRU outputs are dense. Cloning preserves shape, strides and dtype
        # while keeping both cached tensors private from downstream callers.
        self._entries[key] = (input, result)
        self._peak_bytes = max(self._peak_bytes, self._bytes())
        return tuple(tensor.clone() for tensor in result)


def attach_frozen_encoder_cache(model):
    """Attach the cache to one model and use the returned handle as a context.

    Use during CE with read-only training inputs and fixed backend settings.
    PyTorch in-place changes invalidate versioned tensors. Direct writes through
    raw storage, NumPy, or .data aliases must stay outside this context.
    """
    return FrozenEncoderCache(model)
