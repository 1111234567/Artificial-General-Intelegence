"""Opt-in, byte-bound restart cache for expensive read-only branch audits.

Only completed ``report.audit_branch`` results are cached.  Aggregation always
runs the current report's parent checks and final input-byte checks.  Cache
artifacts are gzip JSON, never pickle; their receipts detect changed bytes.
This is an integrity/restart mechanism for locally produced evidence, not an
authentication scheme for evidence supplied by an untrusted party.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time
import uuid

import numpy as np
import torch

from experiments.undirected import report, sensitivity_audit  # load producer dependencies before pinning
from experiments.undirected.queue import atomic_json, available_memory, file_sha256, verify_files


FORMAT = "undirected-branch-audit-cache-v1"
RECEIPT_FORMAT = "undirected-branch-audit-cache-receipt-v1"
ROOT = Path(__file__).resolve().parents[2]
GIB = 1024 ** 3
MAX_ARTIFACT_BYTES = 64 * 1024 ** 2


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _sha(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _pairs(rows):
    result = {}
    for key, value in rows:
        if key in result:
            raise ValueError("duplicate JSON key in audit cache")
        result[key] = value
    return result


def _json(raw):
    return json.loads(raw, object_pairs_hook=_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON {value}")))


def _encode(value):
    """Preserve tuples and integer dictionary keys without executable types."""
    if isinstance(value, dict):
        return {"dict": [[_encode(key), _encode(item)] for key, item in value.items()]}
    if isinstance(value, tuple):
        return {"tuple": [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if value is None or type(value) in (str, bool, int) or (type(value) is float and math.isfinite(value)):
        return value
    raise ValueError(f"unsupported audit cache value: {type(value).__name__}")


def _decode(value):
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"tuple"} and isinstance(value["tuple"], list):
            return tuple(_decode(item) for item in value["tuple"])
        if set(value) == {"dict"} and isinstance(value["dict"], list):
            result = {}
            for pair in value["dict"]:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError("invalid cached dictionary entry")
                key, item = map(_decode, pair)
                if type(key) not in (str, int, float, bool, tuple, type(None)) or key in result:
                    raise ValueError("invalid or duplicate cached dictionary key")
                result[key] = item
            return result
        raise ValueError("invalid audit cache type tag")
    if value is None or type(value) in (str, bool, int) or (type(value) is float and math.isfinite(value)):
        return value
    raise ValueError("invalid primitive audit cache value")


@contextmanager
def _lock(path):
    with Path(path).open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError(f"audit cache already has a live writer: {path}") from error
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _binding(protocol, mechanical, sensitivity_manifest):
    protocol = Path(protocol).resolve()
    plan = report._plan(protocol, mechanical)
    manifest = None if sensitivity_manifest is None else Path(sensitivity_manifest).resolve()
    supplied = {} if manifest is None else _json(manifest.read_bytes())
    identifiers = [row["id"] for row in plan["branches"]]
    if (not isinstance(supplied, dict) or set(supplied) - set(identifiers)
            or any(re.fullmatch(r"[A-Za-z0-9_-]+", name) is None for name in identifiers)):
        raise ValueError("foreign sensitivity inventory or unsafe branch identifier")
    checks, inputs = report.Checks(), {}
    report._manifests(protocol, plan, checks, inputs)
    # Deliberately conservative: exact producer files, including all non-test
    # experiment modules.  Source changes require a new cache or explicit review.
    sources = {str(path.resolve()): file_sha256(path) for path in (ROOT / "experiments").rglob("*.py")
               if not path.name.startswith("test_")}
    sources.update(plan["sources"])
    verify_files(sources)
    artifacts = {str(Path(path).resolve()): file_sha256(path) for path in supplied.values()}
    binding = dict(format=FORMAT, protocol=str(protocol), protocol_sha256=file_sha256(protocol),
                   mechanical=mechanical, branches=plan["branches"], source_files=sources,
                   command_manifests=inputs,
                   sensitivity_manifest=None if manifest is None else dict(path=str(manifest), sha256=file_sha256(manifest)),
                   sensitivity_artifacts=artifacts, supplied={key: str(Path(value).resolve()) for key, value in supplied.items()},
                   runtime=dict(python=sys.version, executable=str(Path(sys.executable).resolve()),
                                torch=torch.__version__, numpy=np.__version__, machine=platform.machine()))
    return plan, binding


def _verify_binding(binding):
    verify_files(binding["source_files"])
    verify_files(binding["command_manifests"])
    verify_files({binding["protocol"]: binding["protocol_sha256"], **binding["sensitivity_artifacts"]})
    if binding["sensitivity_manifest"] is not None:
        row = binding["sensitivity_manifest"]
        verify_files({row["path"]: row["sha256"]})


def _validate_result(result, item, binding, *, verify_inputs):
    required_keys = {"branch", "checkpoints", "per_seed", "probes", "trace", "graphs", "inputs", "checks"}
    if not isinstance(result, dict) or not required_keys <= set(result):
        raise ValueError("incomplete cached branch audit result")
    if any(result["branch"].get(key) != item[key] for key in ("id", "seed", *report.study.KEYS)):
        raise ValueError("cached branch identity differs")
    inputs, counts = result["inputs"], result["checks"]
    if (not isinstance(inputs, dict) or not isinstance(counts, dict) or not counts
            or any(type(value) is not int or value < 1 for value in counts.values())):
        raise ValueError("invalid cached fingerprints or check counts")
    folder = Path(item["output"])
    terminal_path = folder / "terminal.json"
    terminal = _json(terminal_path.read_bytes())
    expected = {binding["protocol"]: binding["protocol_sha256"], **report._json(binding["protocol"])["sources"],
                str(terminal_path.resolve()): file_sha256(terminal_path),
                str((folder / "snapshots.json").resolve()): file_sha256(folder / "snapshots.json"),
                terminal["checkpoint_path"]: terminal["checkpoint_sha256"],
                terminal["ledger"]["binding_path"]: terminal["ledger"]["binding_sha256"],
                terminal["ledger"]["manifest_path"]: terminal["ledger"]["manifest_sha256"]}
    expected.update({row["path"]: row["sha256"] for row in terminal["checkpoints"].values()})
    ledger_folder = Path(terminal["ledger"]["manifest_path"]).parent
    for line in Path(terminal["ledger"]["manifest_path"]).read_text(encoding="utf-8").splitlines():
        row = _json(line)
        expected[str((ledger_folder / row["file"]).resolve())] = row["sha256"]
    evaluation = Path(report._json(binding["protocol"])["raw_root"]) / "evaluation" / f"{item['id']}.json"
    expected[str(evaluation.resolve())] = file_sha256(evaluation)
    sensitivity = binding["supplied"].get(item["id"])
    if sensitivity:
        expected[sensitivity] = binding["sensitivity_artifacts"][sensitivity]
    if (terminal["branch"] != item or terminal["protocol_sha256"] != binding["protocol_sha256"]
            or result["branch"]["terminal_sha256"] != expected[str(terminal_path.resolve())]
            or any(inputs.get(path) != digest for path, digest in expected.items())):
        raise ValueError("cached audit input inventory differs from branch evidence")
    if verify_inputs:
        verify_files(inputs)


def _receipt_path(cache, identifier):
    return Path(cache) / f"{identifier}.receipt.json"


def _load(cache, item, binding, *, verify_inputs=True):
    receipt_path = _receipt_path(cache, item["id"])
    receipt = _json(receipt_path.read_bytes())
    expected_fields = {"format", "binding_sha256", "identifier", "artifact", "artifact_sha256", "result_sha256"}
    if (set(receipt) != expected_fields or receipt["format"] != RECEIPT_FORMAT
            or receipt["binding_sha256"] != _sha(binding) or receipt["identifier"] != item["id"]
            or re.fullmatch(r"[0-9a-f]{64}", receipt["artifact_sha256"]) is None
            or receipt["artifact"] != f"{item['id']}.{receipt['artifact_sha256']}.audit.json.gz"):
        raise ValueError("foreign or altered branch cache receipt")
    path = Path(cache) / receipt["artifact"]
    if path.stat().st_size > MAX_ARTIFACT_BYTES or file_sha256(path) != receipt["artifact_sha256"]:
        raise ValueError("cached branch artifact bytes changed")
    with gzip.open(path, "rb") as stream:
        raw = stream.read(MAX_ARTIFACT_BYTES + 1)
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise ValueError("cached branch artifact exceeds size limit")
    envelope = _json(raw)
    if (set(envelope) != {"format", "binding_sha256", "identifier", "result"}
            or envelope["format"] != FORMAT or envelope["binding_sha256"] != _sha(binding)
            or envelope["identifier"] != item["id"] or _sha(envelope["result"]) != receipt["result_sha256"]):
        raise ValueError("foreign or altered cached branch envelope")
    result = _decode(envelope["result"])
    _validate_result(result, item, binding, verify_inputs=verify_inputs)
    return result, {str(path.resolve()): receipt["artifact_sha256"],
                    str(receipt_path.resolve()): file_sha256(receipt_path)}


def _store(cache, item, binding, result):
    encoded = _encode(result)
    envelope = dict(format=FORMAT, binding_sha256=_sha(binding), identifier=item["id"], result=encoded)
    compressed = gzip.compress(_bytes(envelope), compresslevel=6, mtime=0)
    digest = hashlib.sha256(compressed).hexdigest()
    path = Path(cache) / f"{item['id']}.{digest}.audit.json.gz"
    temporary = Path(cache) / f".{item['id']}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(compressed)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            if file_sha256(path) != digest:
                raise ValueError("content-addressed cache path has different bytes")
        else:
            os.replace(temporary, path)
        atomic_json(_receipt_path(cache, item["id"]), dict(format=RECEIPT_FORMAT,
            binding_sha256=_sha(binding), identifier=item["id"], artifact=path.name,
            artifact_sha256=digest, result_sha256=_sha(encoded)))
    finally:
        if temporary.exists():
            temporary.unlink()


def _worker(cache, item, binding):
    torch.set_num_threads(1)
    with _lock(Path(cache) / f"{item['id']}.lock"):
        _verify_binding(binding)
        if _receipt_path(cache, item["id"]).exists():
            _load(cache, item, binding)
            return dict(id=item["id"], evidence="revalidated_cache")
        result = report.audit_branch(binding["protocol"], item["id"], mechanical=binding["mechanical"],
                                     sensitivity=binding["supplied"].get(item["id"]))
        _verify_binding(binding)
        _validate_result(result, item, binding, verify_inputs=True)
        _store(cache, item, binding, result)
        return dict(id=item["id"], evidence="fresh_branch_audit")


def run(protocol, output, cache, *, workers=1, mechanical=False, sensitivity_manifest=None,
        min_available_gib=4., worker_reserve_gib=1.5):
    if (type(workers) is not int or not 1 <= workers <= 16 or not math.isfinite(min_available_gib)
            or min_available_gib < 0 or not math.isfinite(worker_reserve_gib) or worker_reserve_gib <= 0):
        raise ValueError("invalid cache worker count or RAM limits")
    plan, binding = _binding(protocol, mechanical, sensitivity_manifest)
    cache, output = Path(cache).resolve(), Path(output).resolve()
    protected = [Path(row["output"]).resolve() for row in plan["branches"]]
    protected += [Path(path).resolve() for path in binding["source_files"]]
    if (cache == output or cache.is_relative_to(output) or output.is_relative_to(cache)
            or any(cache == path or cache.is_relative_to(path) or path.is_relative_to(cache) for path in protected)
            or any(output == path or output.is_relative_to(path) for path in protected)):
        raise ValueError("cache/report outputs must be separate from each other and frozen inputs")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("report output must be new or empty; retain prior artifacts and choose a new output")
    cache.mkdir(parents=True, exist_ok=True)
    with _lock(cache / "cache.lock"):
        binding_path = cache / "cache-binding.json"
        if binding_path.exists():
            if _json(binding_path.read_bytes()) != binding:
                raise ValueError("audit cache belongs to different sources/protocol/sensitivity/runtime; use a new cache")
        else:
            if any(path.name != "cache.lock" for path in cache.iterdir()):
                raise ValueError("refusing an unbound nonempty audit cache")
            atomic_json(binding_path, binding)
        floor, reserve = int(min_available_gib * GIB), int(worker_reserve_gib * GIB)
        free = available_memory()
        admitted = min(workers, max(0, (free - floor) // reserve))
        if admitted < 1:
            raise RuntimeError("insufficient available RAM for one audit worker plus reserve; cache unchanged")
        state = dict(format=FORMAT, binding_sha256=_sha(binding), requested_workers=workers,
                     admitted_workers=admitted, min_available_bytes=floor, worker_reserve_bytes=reserve,
                     initial_available_bytes=free, started=time.time(), complete=False, jobs={})
        status = cache / "cache-status.json"
        atomic_json(status, state)
        print(json.dumps(dict(stage="branch_audits", workers=admitted, available_gib=free / GIB)), flush=True)
        dispatched, futures = 0, {}
        try:
            with ProcessPoolExecutor(max_workers=admitted) as pool:
                while futures or dispatched < len(plan["branches"]):
                    while (dispatched < len(plan["branches"]) and len(futures) < admitted
                           and available_memory() >= floor):
                        item = plan["branches"][dispatched]
                        dispatched += 1
                        future = pool.submit(_worker, str(cache), item, binding)
                        futures[future] = item["id"]
                        state["jobs"][item["id"]] = dict(status="running")
                    if not futures and dispatched < len(plan["branches"]):
                        raise RuntimeError("RAM fell below dispatch floor; restart this command to resume completed branches")
                    completed, _ = wait(futures, timeout=1., return_when=FIRST_COMPLETED)
                    for future in completed:
                        identifier = futures.pop(future)
                        row = future.result()
                        state["jobs"][identifier] = dict(status="complete", **row)
                        print(json.dumps(dict(stage="branch_audits", completed=sum(
                            value["status"] == "complete" for value in state["jobs"].values()), **row)), flush=True)
                    atomic_json(status, state)
            _verify_binding(binding)
            results, artifacts = [], {str(binding_path): file_sha256(binding_path)}
            for item in plan["branches"]:
                # The same report build below rehashes every stored raw input;
                # workers already checked full input bytes before acceptance.
                result, files = _load(cache, item, binding, verify_inputs=False)
                result["inputs"].update(files)
                artifacts.update(files)
                results.append(result)
            results[0]["inputs"].update({**artifacts, **binding["source_files"], **binding["command_manifests"]})
            value = report.build_report(protocol, workers=1, mechanical=mechanical,
                sensitivity_results=binding["supplied"], cached_results=results)
            value["cache_provenance"] = dict(format=FORMAT, binding=str(binding_path),
                binding_sha256=file_sha256(binding_path), artifacts=artifacts,
                fresh_branch_audits=sum(row.get("evidence") == "fresh_branch_audit" for row in state["jobs"].values()),
                revalidated_cached_branches=sum(row.get("evidence") == "revalidated_cache" for row in state["jobs"].values()),
                scope="Locally produced JSON branch audit results; exact producer and all recorded raw input bytes revalidated; parent checks rerun.")
            header = report.write_artifacts(value, output)
            state.update(complete=True, finished=time.time(), report=str(output / "report.json"))
            atomic_json(status, state)
            print(json.dumps(dict(stage="complete", report=state["report"], counts=value["counts"])), flush=True)
            return header
        except BaseException as error:
            state.update(error=f"{type(error).__name__}: {error}", failed=time.time())
            atomic_json(status, state)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "output", "cache"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sensitivity-manifest")
    parser.add_argument("--mechanical", action="store_true")
    parser.add_argument("--min-available-gib", type=float, default=4.)
    parser.add_argument("--worker-reserve-gib", type=float, default=1.5)
    args = parser.parse_args(argv)
    run(args.protocol, args.output, args.cache, workers=args.workers, mechanical=args.mechanical,
        sensitivity_manifest=args.sensitivity_manifest, min_available_gib=args.min_available_gib,
        worker_reserve_gib=args.worker_reserve_gib)


if __name__ == "__main__":
    main()
