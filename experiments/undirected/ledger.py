"""Immutable compressed records of successful *real* live update blocks.

This is an I/O observer.  It never changes the learner, its RNGs, replay or
optimizer and stores no history of full RTRL sensitivities.  A stale checkpoint
may replay an already committed record only when its complete content agrees.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path

import torch

from experiments.continuity.run import state_digest as continuation_hash
from experiments.undirected.queue import atomic_json, digest, file_sha256


FORMAT = "undirected-live-record-v1"
BINDING_FORMAT = "undirected-live-ledger-binding-v1"
ENTRY_FORMAT = "undirected-live-ledger-entry-v1"


def content_hash(value):
    """Tensor-aware hash, compatible with the recorded-credit content hash."""
    result = hashlib.sha256()

    def add(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            result.update(f"tensor/{tensor.dtype}/{tuple(tensor.shape)}/".encode())
            result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            result.update(b"mapping[")
            for key in sorted(item):
                add(key)
                add(item[key])
            result.update(b"]")
        elif isinstance(item, (tuple, list)):
            result.update(b"tuple[" if isinstance(item, tuple) else b"list[")
            for child in item:
                add(child)
            result.update(b"]")
        else:
            result.update(f"{type(item).__name__}:{item!r};".encode())

    add(value)
    return result.hexdigest()


def _canonical(value):
    """Remove aliases/autograd and canonicalize insertion order for torch.save."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, Mapping):
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return tuple(_canonical(item) for item in value)
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return deepcopy(value)


def _json_normalized(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _validate_payload(payload, binding_hash=None):
    keys = {"format", "binding_hash", "update_age", "record", "live", "model_class", "model_config",
            "parameter_layout", "dynamic_names", "local_names", "method", "trace_binding", "source_hashes", "dependency_hashes"}
    if not isinstance(payload, Mapping) or set(payload) != keys or payload["format"] != FORMAT:
        raise ValueError("incomplete live ledger payload")
    if not _hash(payload["binding_hash"]) or (binding_hash is not None and payload["binding_hash"] != binding_hash):
        raise ValueError("foreign live ledger binding")
    age = _integer(payload["update_age"], "update age", 1)
    record, live, trace = payload["record"], payload["live"], payload["trace_binding"]
    record_keys = {"observations", "targets", "elapsed", "start_h", "end_h", "start_tick", "end_tick", "parameters", "buffers"}
    if not isinstance(record, Mapping) or set(record) != record_keys:
        raise ValueError("incomplete recorded parameter trajectory")
    if not isinstance(live, Mapping) or set(live) != {"actions", "family_context", "family_counts"}:
        raise ValueError("incomplete actual actions/context")
    start, end = _integer(record["start_tick"], "start tick"), _integer(record["end_tick"], "end tick", 1)
    ticks = end - start
    if ticks < 1 or start != (age - 1) * ticks or end != age * ticks:
        raise ValueError("ledger update/tick alignment differs")
    obs, targets, h0, h1 = (record[key] for key in ("observations", "targets", "start_h", "end_h"))
    if (not all(isinstance(value, torch.Tensor) for value in (obs, targets, h0, h1)) or obs.ndim != 3 or h0.ndim != 2
            or h1.shape != h0.shape or obs.shape[:2] != (ticks, len(h0)) or targets.ndim != 3 or targets.shape[:2] != obs.shape[:2]
            or targets.dtype != torch.long or not isinstance(record["elapsed"], tuple) or len(record["elapsed"]) != ticks):
        raise ValueError("invalid live ledger activity dimensions")
    for key in ("actions", "family_context"):
        if not isinstance(live[key], torch.Tensor) or live[key].shape != targets.shape or live[key].dtype != torch.long:
            raise ValueError(f"invalid actual live {key}")
    for value in (obs, h0, h1, *record["parameters"].values()):
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite or nonfloating live ledger value")
    if not isinstance(record["buffers"], Mapping):
        raise ValueError("invalid recorded buffers")
    layout = _json_normalized([(name, tuple(value.shape), str(value.dtype)) for name, value in record["parameters"].items()])
    # Parameter insertion order is not a derivative convention.  The explicit
    # dynamic_names order below is the flattened sensitivity layout.
    if sorted(payload["parameter_layout"]) != sorted(layout):
        raise ValueError("live ledger parameter layout differs")
    dynamic, local = payload["dynamic_names"], payload["local_names"]
    if (not isinstance(dynamic, list) or not dynamic or len(dynamic) != len(set(dynamic)) or not isinstance(local, list)
            or len(local) != len(set(local)) or set(dynamic) & set(local)
            or set(dynamic) | set(local) != set(record["parameters"])):
        raise ValueError("invalid dynamic/local ledger layout")
    if payload["model_class"] not in ("UndirectedModel", "PortGraphModel") or not isinstance(payload["model_config"], dict):
        raise ValueError("unknown ledger model")
    if (not isinstance(trace, Mapping) or trace.get("tick") != end or trace.get("update") != age
            or trace.get("credit_record_hash") != content_hash(record)
            or trace.get("snapshot_hash") != content_hash(record["parameters"])
            or trace.get("action_chunk_hash") != continuation_hash(live["actions"])
            or trace.get("live_hidden_hash") != continuation_hash(h1)):
        raise ValueError("ledger differs from the committed real live trace")
    causal = {key: record[key] for key in ("observations", "targets", "elapsed", "start_tick", "end_tick")}
    causal["family_context"] = live["family_context"]
    if trace.get("data_chunk_hash") != continuation_hash(causal):
        raise ValueError("ledger observations/targets/context differ from the real live trace")
    for key in ("source_hashes", "dependency_hashes"):
        if not isinstance(payload[key], Mapping) or not payload[key] or any(not _hash(value) for value in payload[key].values()):
            raise ValueError("incomplete live ledger source fingerprints")


def make_payload(trainer, binding_hash):
    """Read a complete real update; neither consume RNG nor modify any state."""
    if not trainer.updates or not trainer.credit.tape or trainer.last_chunk is None or not trainer.trace:
        raise ValueError("a newborn has no live block to record")
    record, chunk, point = trainer.credit.tape[-1], trainer.last_chunk, trainer.trace[-1]
    if (trainer.ticks != record["end_tick"] or trainer.updates != trainer.credit.updates
            or not torch.equal(record["end_h"], trainer.h.detach().cpu())):
        raise ValueError("ledger callback did not receive the actual completed live boundary")
    for name in ("observations", "targets", "elapsed", "start_h", "end_h", "start_tick", "end_tick"):
        if content_hash(record[name]) != content_hash(chunk[name]):
            raise ValueError("credit record and actual live chunk differ")
    payload = dict(format=FORMAT, binding_hash=binding_hash, update_age=trainer.updates, record=record,
        live={key: chunk[key] for key in ("actions", "family_context", "family_counts")},
        model_class=type(trainer.model).__name__, model_config=_json_normalized(trainer.model.config()),
        parameter_layout=_json_normalized([(name, tuple(parameter.shape), str(parameter.dtype))
                                          for name, parameter in trainer.model.named_parameters()]),
        dynamic_names=list(trainer.credit.dynamic_names), local_names=list(trainer.credit.local_names),
        method=trainer.credit.config.method,
        trace_binding=dict(tick=point["tick"], update=point["update"], credit_record_hash=point["credit"]["record_hash"],
            snapshot_hash=point["credit"]["snapshot_hash"], trajectory_hash=point["credit"]["trajectory_hash"],
            action_chunk_hash=point["action_chunk_hash"], data_chunk_hash=point["data_chunk_hash"], live_hidden_hash=point["live_hidden_hash"]),
        source_hashes=trainer.source_hashes, dependency_hashes=trainer.dependency_hashes)
    _validate_payload(payload, binding_hash)
    return _canonical(payload)


def _binding(folder, expected_binding=None):
    path = Path(folder) / "binding.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(value, dict) or set(value) != {"format", "binding", "binding_hash"}
            or value["format"] != BINDING_FORMAT or value["binding_hash"] != digest(value["binding"])
            or (expected_binding is not None and value["binding"] != _json_normalized(expected_binding))):
        raise ValueError("foreign live ledger binding file")
    return value


def load_record(folder, entry, expected_binding_hash=None):
    path = Path(folder) / entry["file"]
    if file_sha256(path) != entry["sha256"]:
        raise ValueError("compressed live record bytes changed")
    compressed = path.read_bytes()
    raw = gzip.decompress(compressed)
    if len(compressed) != entry["compressed_bytes"] or len(raw) != entry["uncompressed_bytes"]:
        raise ValueError("compressed live record sizes differ")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    _validate_payload(payload, expected_binding_hash)
    if (content_hash(payload) != entry["content_hash"] or payload["update_age"] != entry["update_age"]
            or payload["record"]["start_tick"] != entry["start_tick"] or payload["record"]["end_tick"] != entry["end_tick"]):
        raise ValueError("live record content differs from manifest")
    return payload


def read_ledger(folder, expected_binding=None, verify_files=True):
    folder = Path(folder).resolve()
    binding = _binding(folder, expected_binding)
    manifest = folder / "manifest.jsonl"
    raw = manifest.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("truncated append-only live manifest")
    entries, previous = [], binding["binding_hash"]
    for index, line in enumerate(raw.splitlines(), 1):
        entry = json.loads(line)
        keys = {"format", "update_age", "start_tick", "end_tick", "file", "content_hash", "sha256", "compressed_bytes",
                "uncompressed_bytes", "previous_entry_hash", "entry_hash"}
        if (not isinstance(entry, dict) or set(entry) != keys or entry["format"] != ENTRY_FORMAT
                or entry["update_age"] != index or entry["file"] != f"block_{index:06d}.pt.gz"
                or entry["previous_entry_hash"] != previous or entry["entry_hash"] != digest({key: value for key, value in entry.items() if key != "entry_hash"})
                or not _hash(entry["content_hash"]) or not _hash(entry["sha256"])
                or entry["start_tick"] != (entries[-1]["end_tick"] if entries else 0)
                or entry["end_tick"] <= entry["start_tick"]):
            raise ValueError("invalid or reordered append-only live manifest")
        if verify_files:
            payload = load_record(folder, entry, binding["binding_hash"])
            if entries:
                if payload["parameter_layout"] != prior["parameter_layout"] or payload["dynamic_names"] != prior["dynamic_names"]:
                    raise ValueError("live ledger parameter layout changed")
                if not torch.equal(payload["record"]["start_h"], prior["record"]["end_h"]):
                    raise ValueError("live ledger hidden trajectory was reset or replaced")
            prior = payload
        entries.append(entry)
        previous = entry["entry_hash"]
    return dict(binding=binding["binding"], binding_hash=binding["binding_hash"], entries=entries,
        binding_path=str((folder / "binding.json").resolve()), binding_sha256=file_sha256(folder / "binding.json"),
        manifest_path=str(manifest), manifest_sha256=file_sha256(manifest))


class LiveLedger:
    """Callable native.train observer, with idempotent stale-checkpoint replay."""
    def __init__(self, folder, binding):
        self.folder = Path(folder).resolve()
        normalized = _json_normalized(binding)
        if not self.folder.exists():
            self.folder.mkdir(parents=True)
        binding_path, manifest = self.folder / "binding.json", self.folder / "manifest.jsonl"
        if not binding_path.exists():
            if any(self.folder.iterdir()):
                raise ValueError("refusing an unbound existing live ledger")
            atomic_json(binding_path, dict(format=BINDING_FORMAT, binding=normalized, binding_hash=digest(normalized)))
            with manifest.open("xb") as stream:
                stream.flush()
                os.fsync(stream.fileno())
        self.state = read_ledger(self.folder, normalized)
        self.binding_hash = self.state["binding_hash"]
        self.entries = self.state["entries"]
        self._manifest_sha256 = self.state["manifest_sha256"]

    def _unchanged_manifest(self):
        if file_sha256(self.folder / "manifest.jsonl") != self._manifest_sha256:
            raise ValueError("live manifest changed outside its sole writer")
        if _binding(self.folder)["binding_hash"] != self.binding_hash:
            raise ValueError("live ledger binding changed")

    def append_payload(self, payload):
        self._unchanged_manifest()
        payload = _canonical(payload)
        _validate_payload(payload, self.binding_hash)
        age = payload["update_age"]
        value_hash = content_hash(payload)
        if age <= len(self.entries):
            entry = self.entries[age - 1]
            existing = load_record(self.folder, entry, self.binding_hash)
            if content_hash(existing) != value_hash:
                raise ValueError("replayed live block differs from its immutable existing record")
            return deepcopy(entry)
        if age != len(self.entries) + 1:
            raise ValueError("live ledger has a missing historical block; the latest checkpoint cannot reconstruct it")
        if self.entries:
            prior = load_record(self.folder, self.entries[-1], self.binding_hash)
            if (prior["parameter_layout"] != payload["parameter_layout"] or prior["dynamic_names"] != payload["dynamic_names"]
                    or not torch.equal(prior["record"]["end_h"], payload["record"]["start_h"])):
                raise ValueError("new record does not continue the actual recorded history")
        path = self.folder / f"block_{age:06d}.pt.gz"
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        raw = buffer.getvalue()
        compressed = gzip.compress(raw, compresslevel=6, mtime=0)
        if path.exists():
            # A crash after the atomic record rename but before manifest append
            # leaves an orphan.  Reuse only the exact deterministic record bytes.
            if path.read_bytes() != compressed:
                raise ValueError("foreign or different orphan live record")
        else:
            temporary = path.with_suffix(path.suffix + ".tmp")
            with temporary.open("wb") as stream:
                stream.write(compressed)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        entry = dict(format=ENTRY_FORMAT, update_age=age, start_tick=payload["record"]["start_tick"],
            end_tick=payload["record"]["end_tick"], file=path.name, content_hash=value_hash, sha256=file_sha256(path),
            compressed_bytes=len(compressed), uncompressed_bytes=len(raw),
            previous_entry_hash=self.entries[-1]["entry_hash"] if self.entries else self.binding_hash)
        entry["entry_hash"] = digest(entry)
        line = json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        with (self.folder / "manifest.jsonl").open("ab") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        self.entries.append(entry)
        self._manifest_sha256 = file_sha256(self.folder / "manifest.jsonl")
        return deepcopy(entry)

    def __call__(self, trainer):
        if not trainer.updates:
            if self.entries:
                first = load_record(self.folder, self.entries[0], self.binding_hash)
                if not torch.equal(first["record"]["start_h"], trainer.h.detach().cpu()):
                    raise ValueError("newborn does not match the existing ledger's actual start")
            return None
        if len(self.entries) < trainer.updates - 1:
            raise ValueError("past live records are missing; only the latest block is recoverable from this checkpoint")
        return self.append_payload(make_payload(trainer, self.binding_hash))

    def summary(self, *, updates=None, require_exact=False):
        state = read_ledger(self.folder, verify_files=True)
        if updates is not None and (len(state["entries"]) < updates or (require_exact and len(state["entries"]) != updates)):
            raise ValueError("ledger does not exactly cover the terminal's successful updates")
        return {key: state[key] for key in ("binding_hash", "binding_path", "binding_sha256", "manifest_path", "manifest_sha256")} | dict(
            records=len(state["entries"]), last_tick=state["entries"][-1]["end_tick"] if state["entries"] else 0,
            compressed_bytes=sum(entry["compressed_bytes"] for entry in state["entries"]),
            uncompressed_bytes=sum(entry["uncompressed_bytes"] for entry in state["entries"]))
