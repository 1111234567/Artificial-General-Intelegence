"""Opt-in tape transfer reuse for the packed CUDA Graph evaluator.

Use ``class Capture(CachedTapeCaptureMixin, backend._RewardCapture)`` in an
optional launcher. Each capture keeps the last CPU tensor and its mutation
version per tape field; a new capture starts with no transfer memo.
"""

from __future__ import annotations

import torch

from .uncapped_cuda_backend import _TAPE
from .uncapped_directed_packed import PackedCaptureMixin


class CachedTapeCaptureMixin(PackedCaptureMixin):
    """Reuse unchanged CPU tape fields while copying every batch's genomes.

    Object identity alone cannot detect public, in-place tensor edits. Include
    PyTorch's mutation version, and retain strong references so object IDs
    cannot be recycled. Unversioned tensors conservatively take the copy path.
    The memo belongs to this capture and disappears when the capture is evicted.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tape_transfer_cache = {}

    def evaluate(self, birth, tape, caller_stream):
        with torch.cuda.stream(self.stream):
            for key, tensor in birth.items():
                self.birth[key].copy_(tensor)
            for key in _TAPE:
                tensor = getattr(tape, key)
                version = None
                if tensor.device.type == "cpu":
                    try:
                        version = tensor._version
                    except RuntimeError:
                        # Inference tensors have no mutation counter. Identity
                        # cannot establish that their contents are unchanged.
                        pass
                previous = self._tape_transfer_cache.get(key)
                if (version is not None and previous is not None
                        and previous[0] is tensor and previous[1] == version):
                    continue
                getattr(self.tape, key).copy_(tensor)
                if version is None:
                    self._tape_transfer_cache.pop(key, None)
                else:
                    self._tape_transfer_cache[key] = (tensor, version)
            self.graph.replay()
            self.packing_graph.replay()
            results = self._packed_results.snapshot()
        caller_stream.wait_stream(self.stream)
        for buffer in results.packed_buffers:
            buffer.record_stream(caller_stream)
        return results
