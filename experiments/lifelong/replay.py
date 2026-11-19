"""Finite replay of actual observed trajectories and their original feedback.

Only complete, nonoverlapping real-history blocks enter the uniform reservoir.
Admission and replay use independent private RNGs. Replay returns a detached
copy and never acts in a world, fabricates labels, or changes the live state.
The trainer admits each real chunk *after* its update and requests a historical
block ending no later than that update's live starting tick.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from numbers import Integral
import random
from typing import Any

import torch


FORMAT_VERSION = "lifelong-replay-v1"
_TENSOR_KEYS = ("observations", "targets", "actions")
_OPTIONAL_SEQUENCE_KEYS = ("family_context",)


def cpu_copy(value: Any) -> Any:
    """Recursively sever autograd/storage aliases before retaining experience."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    return deepcopy(value)


def _integer(name: str, value: int, *, lower: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < lower:
        raise ValueError(f"{name} must be an integer >= {lower}")
    return int(value)


def _seed(seed: int, namespace: str) -> int:
    return int.from_bytes(hashlib.sha256(
        f"{FORMAT_VERSION}/{seed}/{namespace}".encode()).digest()[:8], "little")


def block_hash(block: dict[str, Any]) -> str:
    """Hash an experience's causal data, independent of storage identities."""
    digest = hashlib.sha256()

    def put(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            digest.update(b"dict:")
            for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
                put(key)
                put(value[key])
        elif isinstance(value, (tuple, list)):
            digest.update(f"{type(value).__name__}:{len(value)}:".encode())
            for item in value:
                put(item)
        else:
            digest.update(f"{type(value).__name__}:{value!r};".encode())

    put({key: value for key, value in block.items() if key != "sha256"})
    return digest.hexdigest()


def validate_block(block: dict[str, Any], *, expected_ticks: int | None = None) -> None:
    """Validate copied observations/feedback without interpreting private truth."""
    if not isinstance(block, dict):
        raise ValueError("experience must be a block dictionary")
    required = {*_TENSOR_KEYS, "elapsed", "start_h", "start_tick", "end_tick"}
    if not required.issubset(block):
        raise ValueError(f"experience is missing fields: {sorted(required - block.keys())}")
    start = _integer("start_tick", block["start_tick"])
    end = _integer("end_tick", block["end_tick"], lower=1)
    ticks = end - start
    if ticks <= 0 or (expected_ticks is not None and ticks != expected_ticks):
        raise ValueError("experience must have the expected positive tick length")
    observation, target, action, hidden = (block[key] for key in
                                         ("observations", "targets", "actions", "start_h"))
    if (not isinstance(observation, torch.Tensor) or observation.ndim != 3 or
            observation.shape[0] != ticks or min(observation.shape) <= 0 or
            not observation.is_floating_point()):
        raise ValueError("observations must be floating [ticks, batch, inputs]")
    if (not isinstance(target, torch.Tensor) or target.ndim != 3 or
            target.shape[:2] != observation.shape[:2] or target.shape[2] <= 0 or
            target.dtype != torch.long):
        raise ValueError("targets must be int64 [ticks, batch, jobs]")
    if (not isinstance(action, torch.Tensor) or action.shape != target.shape or
            action.dtype != torch.long):
        raise ValueError("actions must have the same int64 shape as targets")
    if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 2 or
            hidden.shape[0] != observation.shape[1] or hidden.shape[1] <= 0 or
            not hidden.is_floating_point()):
        raise ValueError("start_h must be floating [batch, hidden]")
    try:
        elapsed = tuple(map(float, block["elapsed"]))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("elapsed must contain one finite nonnegative duration per tick") from error
    if len(elapsed) != ticks or any(not math.isfinite(dt) or dt < 0 for dt in elapsed):
        raise ValueError("elapsed must contain one finite nonnegative duration per tick")
    if not all(bool(torch.isfinite(value).all()) for value in (observation, hidden)):
        raise ValueError("observations and start_h must be finite")
    if bool((target < 0).any()) or bool((action < 0).any()):
        raise ValueError("actions and targets must be nonnegative codes")
    if "source_id" in block and (not isinstance(block["source_id"], str) or not block["source_id"]):
        raise ValueError("source_id must be a nonempty string")
    if "family_counts" in block:
        counts = block["family_counts"]
        if not isinstance(counts, dict):
            raise ValueError("family_counts must be a dictionary")
        for count in counts.values():
            _integer("family count", count)
    for key in _OPTIONAL_SEQUENCE_KEYS:
        if key in block:
            values = block[key]
            if (not isinstance(values, torch.Tensor) or values.shape != target.shape or
                    values.dtype != torch.long or bool(((values < -1) | (values >= 4)).any())):
                raise ValueError(f"{key} must be int64 [ticks, batch, jobs] with values -1..3")


def copied_block(block: dict[str, Any], *, expected_ticks: int | None = None) -> dict[str, Any]:
    validate_block(block, expected_ticks=expected_ticks)
    result = cpu_copy(block)
    result["start_tick"], result["end_tick"] = int(result["start_tick"]), int(result["end_tick"])
    result["elapsed"] = tuple(map(float, result["elapsed"]))
    result.setdefault("family_counts", {})
    result.setdefault("source_id", f"live:{result['start_tick']}:{result['end_tick']}")
    result.pop("sha256", None)
    return result


def combine_blocks(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Join consecutive real chunks, retaining the first chunk's actual state.

    This also constructs the recent-repeat window. It does not admit either
    source again and never substitutes recomputed actions or teacher labels.
    """
    left, right = copied_block(previous), copied_block(current)
    if left["end_tick"] != right["start_tick"]:
        raise ValueError("experience chunks must be consecutive")
    for key in _TENSOR_KEYS:
        if left[key].shape[1:] != right[key].shape[1:] or left[key].dtype != right[key].dtype:
            raise ValueError("consecutive chunks must have consistent tensor schemas")
    optional_keys = tuple(key for key in _OPTIONAL_SEQUENCE_KEYS if key in left or key in right)
    if any((key in left) != (key in right) for key in optional_keys):
        raise ValueError("consecutive chunks must have consistent optional sequence fields")
    if left["start_h"].shape != right["start_h"].shape or left["start_h"].dtype != right["start_h"].dtype:
        raise ValueError("consecutive chunks must have consistent hidden-state schemas")
    counts = dict(left["family_counts"])
    for family, count in right["family_counts"].items():
        counts[family] = counts.get(family, 0) + count
    result = {key: torch.cat((left[key], right[key]), dim=0) for key in _TENSOR_KEYS + optional_keys}
    result.update(start_h=left["start_h"], elapsed=left["elapsed"] + right["elapsed"],
                  start_tick=left["start_tick"], end_tick=right["end_tick"], family_counts=counts,
                  source_id=f"live:{left['start_tick']}:{right['end_tick']}",
                  source_chunks=tuple(left.get("source_chunks", (left["source_id"],))) +
                                tuple(right.get("source_chunks", (right["source_id"],))))
    validate_block(result)
    return result


class HistoryReservoir:
    """Uniform reservoir over completed nonoverlapping blocks of real history.

    add_chunk collects adjacent short chunks into block_ticks records, useful
    when the trainer updates twice per history block. add_block is the direct
    complete-block equivalent. Mixing the two while a partial block exists is
    rejected. Sampling is restricted to end_tick <= before_tick and does not
    affect admission decisions. Every returned/stored state is an owned copy.
    """

    def __init__(self, capacity: int = 32, block_ticks: int = 256, seed: int = 0):
        self.capacity = _integer("capacity", capacity)
        self.block_ticks = _integer("block_ticks", block_ticks, lower=1)
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValueError("seed must be an integer")
        self.admission_rng = random.Random(_seed(int(seed), "admission"))
        self.sample_rng = random.Random(_seed(int(seed), "sampling"))
        self.blocks_seen = 0
        self.chunks_seen = 0
        self.ticks_seen = 0
        self.samples_drawn = 0
        self.replacements = 0
        self.blocks: list[dict[str, Any]] = []
        self.partial: dict[str, Any] | None = None
        self.last_end_tick: int | None = None
        self._schema: dict[str, Any] | None = None
        # Retained contents are internally owned and never modified in place.
        # Public reads go through sample/state_dict; hashes need recomputation
        # only on admission, partial append, or checkpoint restoration.
        self._block_hashes: list[str] = []
        self._partial_hash: str | None = None

    def __len__(self) -> int:
        return len(self.blocks)

    @property
    def bytes(self) -> int:
        def tensor_bytes(value: Any) -> int:
            if isinstance(value, torch.Tensor):
                return value.numel() * value.element_size()
            if isinstance(value, dict):
                return sum(tensor_bytes(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return sum(tensor_bytes(item) for item in value)
            return 0
        return tensor_bytes(self.blocks) + tensor_bytes(self.partial)

    @staticmethod
    def _block_schema(block: dict[str, Any]) -> dict[str, Any]:
        keys = _TENSOR_KEYS + tuple(key for key in _OPTIONAL_SEQUENCE_KEYS if key in block)
        return {key: (tuple(block[key].shape[1:]), str(block[key].dtype)) for key in keys} | {
            "start_h": (tuple(block["start_h"].shape), str(block["start_h"].dtype))}

    def _check_admission(self, block: dict[str, Any]) -> None:
        if self.last_end_tick is None and block["start_tick"] != 0:
            raise ValueError("native real history must begin at birth tick zero")
        if self.last_end_tick is not None and block["start_tick"] != self.last_end_tick:
            raise ValueError("admit real history once in contiguous order; duplicate or skipped ticks")
        schema = self._block_schema(block)
        if self._schema is not None and schema != self._schema:
            raise ValueError("experience schema changed during the reservoir lifetime")

    def _record_input(self, block: dict[str, Any]) -> None:
        self._schema = self._block_schema(block)
        self.last_end_tick = block["end_tick"]
        self.ticks_seen += block["end_tick"] - block["start_tick"]
        self.chunks_seen += 1

    def _admit_complete(self, block: dict[str, Any]) -> None:
        self.blocks_seen += 1
        if len(self.blocks) < self.capacity:
            self.blocks.append(block)
            self._block_hashes.append(block_hash(block))
        else:
            index = self.admission_rng.randrange(self.blocks_seen)
            if index < self.capacity:
                self.blocks[index] = block
                self._block_hashes[index] = block_hash(block)
                self.replacements += 1

    def add_block(self, block: dict[str, Any]) -> None:
        """Admit one newly completed real block after the learner's update."""
        if self.partial is not None:
            raise ValueError("cannot directly admit a block while a partial block is pending")
        incoming = copied_block(block, expected_ticks=self.block_ticks)
        self._check_admission(incoming)
        self._record_input(incoming)
        self._admit_complete(incoming)

    def add_chunk(self, chunk: dict[str, Any]) -> bool:
        """Append an actual post-update chunk; return whether a block completed."""
        incoming = copied_block(chunk)
        self._check_admission(incoming)
        combined = incoming if self.partial is None else combine_blocks(self.partial, incoming)
        length = combined["end_tick"] - combined["start_tick"]
        if length > self.block_ticks:
            raise ValueError("chunk crosses a reservoir block boundary; split at that boundary")
        self._record_input(incoming)
        if length == self.block_ticks:
            self._admit_complete(combined)
            self.partial = None
            self._partial_hash = None
            return True
        self.partial = combined
        self._partial_hash = block_hash(combined)
        return False

    def eligible_count(self, before_tick: int) -> int:
        before_tick = _integer("before_tick", before_tick)
        return sum(block["end_tick"] <= before_tick for block in self.blocks)

    def sample(self, before_tick: int) -> dict[str, Any]:
        """Read a real block completed before the current live chunk began."""
        before_tick = _integer("before_tick", before_tick)
        eligible = [block for block in self.blocks if block["end_tick"] <= before_tick]
        if not eligible:
            raise ValueError("no eligible past history block before the current live chunk")
        selected = eligible[self.sample_rng.randrange(len(eligible))]
        self.samples_drawn += 1
        return cpu_copy(selected)

    def state_dict(self) -> dict[str, Any]:
        return {"format": FORMAT_VERSION, "capacity": self.capacity, "block_ticks": self.block_ticks,
                "blocks_seen": self.blocks_seen, "chunks_seen": self.chunks_seen,
                "ticks_seen": self.ticks_seen, "samples_drawn": self.samples_drawn,
                "replacements": self.replacements, "blocks": cpu_copy(self.blocks),
                "partial": cpu_copy(self.partial), "last_end_tick": self.last_end_tick,
                "schema": deepcopy(self._schema), "admission_rng_state": self.admission_rng.getstate(),
                "sample_rng_state": self.sample_rng.getstate()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore both private RNGs and unfinished real-history assembly."""
        fields = {"format", "capacity", "block_ticks", "blocks_seen", "chunks_seen", "ticks_seen",
                  "samples_drawn", "replacements", "blocks", "partial", "last_end_tick", "schema",
                  "admission_rng_state", "sample_rng_state"}
        if not isinstance(state, dict) or set(state) != fields or state.get("format") != FORMAT_VERSION:
            raise ValueError("unsupported or incomplete lifelong replay format")
        if state.get("capacity") != self.capacity or state.get("block_ticks") != self.block_ticks:
            raise ValueError("saved reservoir dimensions do not match")
        counters = {name: _integer(name, state[name]) for name in
                    ("blocks_seen", "chunks_seen", "ticks_seen", "samples_drawn", "replacements")}
        blocks = [copied_block(block, expected_ticks=self.block_ticks) for block in state["blocks"]]
        if len(blocks) != min(self.capacity, counters["blocks_seen"]):
            raise ValueError("invalid populated reservoir length")
        ranges = [(block["start_tick"], block["end_tick"]) for block in blocks]
        if len(set(ranges)) != len(ranges):
            raise ValueError("duplicate retained history block")
        partial = None if state["partial"] is None else copied_block(state["partial"])
        partial_ticks = 0 if partial is None else partial["end_tick"] - partial["start_tick"]
        if partial_ticks >= self.block_ticks:
            raise ValueError("partial history must be shorter than a full block")
        if counters["ticks_seen"] != counters["blocks_seen"] * self.block_ticks + partial_ticks:
            raise ValueError("history tick and complete-block counters disagree")
        if counters["chunks_seen"] > counters["ticks_seen"] or counters["chunks_seen"] < counters["blocks_seen"]:
            raise ValueError("invalid real-chunk counter")
        if counters["replacements"] > max(0, counters["blocks_seen"] - self.capacity):
            raise ValueError("invalid replacement counter")
        last_end = state["last_end_tick"]
        schema = deepcopy(state["schema"])
        if counters["ticks_seen"] == 0:
            if last_end is not None or schema is not None or partial is not None:
                raise ValueError("empty history cannot have a live endpoint/schema")
        else:
            last_end = _integer("last_end_tick", last_end, lower=1)
            if last_end != counters["ticks_seen"]:
                raise ValueError("history endpoint must equal observed ticks since birth")
            for block in blocks + ([] if partial is None else [partial]):
                if block["end_tick"] > last_end or self._block_schema(block) != schema:
                    raise ValueError("saved history schema or causal endpoint is invalid")
                if block["start_tick"] % self.block_ticks:
                    raise ValueError("saved block does not align with the nonoverlapping birth history")
            if partial is not None and partial["end_tick"] != last_end:
                raise ValueError("partial history must reach the current real-history endpoint")
        admission_rng, sample_rng = random.Random(), random.Random()
        admission_rng.setstate(state["admission_rng_state"])
        sample_rng.setstate(state["sample_rng_state"])
        for name, value in counters.items():
            setattr(self, name, value)
        self.blocks, self.partial, self.last_end_tick, self._schema = blocks, partial, last_end, schema
        self.admission_rng, self.sample_rng = admission_rng, sample_rng
        self._block_hashes = [block_hash(block) for block in blocks]
        self._partial_hash = None if partial is None else block_hash(partial)

    def retention_fingerprint(self) -> str:
        """Compositional full-content identity, without copying the full buffer.

        Every field of state_dict participates, with blocks and partial replaced
        by their cached content SHA-256s. This is a hash of a composition, not the
        canonical raw state digest used for full checkpoint audits. It changes
        with either RNG, all bookkeeping, and any admitted/retained content.
        Calling it consumes no RNG and never samples or admits experience.
        """
        metadata = {"format": FORMAT_VERSION, "capacity": self.capacity, "block_ticks": self.block_ticks,
                    "blocks_seen": self.blocks_seen, "chunks_seen": self.chunks_seen,
                    "ticks_seen": self.ticks_seen, "samples_drawn": self.samples_drawn,
                    "replacements": self.replacements, "blocks": tuple(self._block_hashes),
                    "partial": self._partial_hash, "last_end_tick": self.last_end_tick,
                    "schema": self._schema, "admission_rng_state": self.admission_rng.getstate(),
                    "sample_rng_state": self.sample_rng.getstate()}
        return block_hash(metadata)

    def summary(self) -> dict[str, Any]:
        counts: dict[Any, int] = {}
        for block in self.blocks:
            for family, count in block["family_counts"].items():
                counts[family] = counts.get(family, 0) + count
        return {"capacity_blocks": self.capacity, "block_ticks": self.block_ticks,
                "complete_blocks_seen": self.blocks_seen, "real_chunks_seen": self.chunks_seen,
                "real_ticks_seen": self.ticks_seen, "stored_blocks": len(self.blocks),
                "partial_ticks": 0 if self.partial is None else self.partial["end_tick"] - self.partial["start_tick"],
                "samples_drawn": self.samples_drawn, "replacements": self.replacements,
                "bytes": self.bytes, "retained_family_counts": counts,
                "stored_tick_ranges": [[block["start_tick"], block["end_tick"]] for block in self.blocks],
                "stored_source_ids": [block["source_id"] for block in self.blocks],
                "stored_sha256": list(self._block_hashes),
                "retention_fingerprint": self.retention_fingerprint(),
                "source": "Actual own-trajectory observations, actions, dt, start_h and original causal teacher labels.",
                "selection": "Uniform reservoir over completed nonoverlapping live blocks; independent admission/sample RNGs."}

