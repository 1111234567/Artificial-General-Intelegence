"""Opt-in, dtype-preserving snapshots and host transfers for captured metrics.

Use ``class Capture(PackedCaptureMixin, backend._RewardCapture)`` only in an
optional launcher. Existing evaluators and their captured rollout stay intact.
The dictionary interface is unchanged; its tensor values view owned buffers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .evolved_plasticity_run import unpack_scores as _unpack_scores
from .uncapped_cuda_backend import CUDAGraphUnavailable, _TAPE


@dataclass(frozen=True)
class _Field:
    name: str
    group: int
    shape: torch.Size
    offset: int
    size: int


class PackedResults(dict):
    """Metric dictionary whose same-dtype values share one owned buffer.

    Group by both dtype and device, so integer counts never pass through a
    floating-point representation. Separate snapshots have separate storage.
    """

    def __init__(self, buffers, layout):
        self.packed_buffers = tuple(buffers)
        self._layout = tuple(layout)
        fields = {
            field.name: self.packed_buffers[field.group]
            .narrow(0, field.offset, field.size).view(field.shape)
            for field in self._layout
        }
        super().__init__(fields)
        self._original_fields = fields

    def snapshot(self):
        """Clone each group once, keeping all returned metric views alive."""
        return PackedResults((buffer.clone() for buffer in self.packed_buffers), self._layout)

    def cpu_fields(self):
        """Copy each packed group to CPU once and return ordinary field views."""
        host = PackedResults(
            (buffer.detach().cpu() for buffer in self.packed_buffers), self._layout,
        )
        # Keep ordinary dict semantics if a caller adds, replaces or deletes a
        # field between evaluation and validation. Unmodified fields use packs.
        return {
            name: host[name] if (name in self._original_fields
                                 and value is self._original_fields[name]) else value
            for name, value in self.items()
        }


def pack_results(fields):
    """Pack tensor metrics without changing their order, shape, dtype or value.

    This copies once per (device, dtype) group. In a captured evaluator it is
    called only during construction, within the small output-packing graph.
    """
    groups, chunks, lengths, layout = {}, [], [], []
    for name, value in fields.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected a metric tensor in {name}")
        key = (value.device, value.dtype)
        group = groups.get(key)
        if group is None:
            group = groups[key] = len(chunks)
            chunks.append([])
            lengths.append(0)
        size = value.numel()
        layout.append(_Field(name, group, value.shape, lengths[group], size))
        chunks[group].append(value.reshape(-1))
        lengths[group] += size
    return PackedResults((torch.cat(group) for group in chunks), layout)


class PackedCaptureMixin:
    """Capture packing once, then clone one output buffer per dtype on replay.

    Place this mixin before the original capture class in the MRO. Its parent
    constructor must provide ``stream``, ``graph``, ``birth``, ``tape`` and the
    result tensor dictionary, as ``_RewardCapture`` does.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.packing_graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(self.packing_graph, stream=self.stream):
                self._packed_results = pack_results(self.results)
        except RuntimeError as error:
            raise CUDAGraphUnavailable(f"directed metric packing capture failed: {error}") from error

    def evaluate(self, birth, tape, caller_stream):
        with torch.cuda.stream(self.stream):
            for key, tensor in birth.items():
                self.birth[key].copy_(tensor)
            for key in _TAPE:
                getattr(self.tape, key).copy_(getattr(tape, key))
            self.graph.replay()
            self.packing_graph.replay()
            results = self._packed_results.snapshot()
        caller_stream.wait_stream(self.stream)
        for buffer in results.packed_buffers:
            buffer.record_stream(caller_stream)
        return results


def unpack_scores_packed(fields, size, unpack_scores=_unpack_scores):
    """Use grouped host transfers, then the original score validator unchanged.

    Ordinary dictionaries still take the reference path. The optional third
    argument allows a launcher to pass its saved validator explicitly.
    """
    if isinstance(fields, PackedResults):
        fields = fields.cpu_fields()
    return unpack_scores(fields, size)
