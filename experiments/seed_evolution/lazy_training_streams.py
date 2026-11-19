"""Read the frozen CE streams on demand and keep one source per serial worker."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from threading import Lock

import torch

from experiments.learned_binding import common
from experiments.seed_evolution import core


_PATCH_LOCK = Lock()


class _TrainingStreams:
    def __init__(self, store):
        self._store = store

    def __getitem__(self, index):
        return self._store._get_stream(self, index)


class LazyTrainingStreams:
    """Reuse read-only stream tensors for successive candidates of one source."""

    def __init__(self):
        self._manifest = None
        self._source_seed = None
        self._rows = ()
        self._tensors = {}
        self._view = None
        self._file_reads = 0

    @property
    def stats(self):
        return dict(source_seed=self._source_seed, loaded_indices=sorted(self._tensors),
                    loaded_streams=len(self._tensors), file_reads=self._file_reads,
                    bytes=sum(tensor.numel() * tensor.element_size()
                              for pair in self._tensors.values() for tensor in pair))

    @contextmanager
    def install(self):
        """Install for one synchronous call to the original training function."""
        if not _PATCH_LOCK.acquire(blocking=False):
            raise RuntimeError("training stream loader is already installed in this worker")
        original = core._training_streams
        try:
            core._training_streams = self._for_source
            yield self
        finally:
            core._training_streams = original
            self._view = None
            _PATCH_LOCK.release()

    def _for_source(self, source_seed):
        source_seed = core._source_seed(source_seed)
        if self._view is not None:
            if source_seed != self._source_seed:
                raise RuntimeError("cannot change the training source within a candidate")
            return self._view
        if source_seed != self._source_seed:
            if self._manifest is None:
                self._manifest = json.loads(core.DATA_FREEZE.read_text(encoding="utf-8"))
            rows = sorted((row for row in self._manifest["streams"] if row["seed"] == source_seed),
                          key=lambda row: row["stream"])
            if [row["stream"] for row in rows] != list(range(common.SETTINGS["training_streams"])):
                raise ValueError(f"incomplete or duplicate v1 training streams for seed {source_seed}")
            self._tensors.clear()
            self._source_seed, self._rows = source_seed, rows
        self._view = _TrainingStreams(self)
        return self._view

    def _get_stream(self, view, index):
        if view is not self._view:
            raise RuntimeError("training streams can only be accessed within their candidate")
        if not 0 <= index < len(self._rows):
            raise IndexError(index)
        if index not in self._tensors:
            row = self._rows[index]
            path = core.ROOT / Path(row["path"].replace("\\", "/"))
            if common.sha(path) != row["sha256"]:
                raise ValueError(f"frozen v1 training stream hash mismatch: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            identity = payload["metadata"]
            if (identity["source_seed"], identity["stream_id"], identity["split"]) != (
                    self._source_seed, index, "train"):
                raise ValueError(f"training stream identity differs from manifest: {path}")
            obs, actions = payload["obs"], payload["actions"]
            ticks, batch = common.SETTINGS["train_ticks"], common.SETTINGS["train_batch"]
            if (obs.shape != (ticks, batch, 56) or obs.dtype != torch.float32
                    or actions.shape != (ticks, batch, 2) or actions.dtype != torch.long):
                raise ValueError(f"incompatible v1 training tensors: {path}")
            self._tensors[index] = obs, actions
            self._file_reads += 1
        return self._tensors[index]
