"""Supplement the execution amendment for generations that have no issued plan."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiments.seed_evolution import execution_runtime as runtime


NAME = "balanced-schedule.json"
_FORMAT = "seed-evolution-balanced-schedule-v1"
_BASE_BACKEND = runtime.backend_for


@dataclass(frozen=True)
class Schedule:
    start_generation: int
    hybrid_indices: tuple[int, ...]
    sha256: str

    def backend(self, cid, spec):
        backend = _BASE_BACKEND(cid, spec)
        generation, index = map(int, (cid[-6:-4], cid[-2:]))
        if generation < self.start_generation:
            return backend
        return "hybrid" if index in self.hybrid_indices else "cpu"


def _plan_hashes(root, before):
    return {
        f"plans/generation_{generation:02d}.json":
            runtime.sha(root / f"plans/generation_{generation:02d}.json")
        for generation in range(before)
    }


def _validate_indices(indices):
    if (not isinstance(indices, list) or any(type(index) is not int for index in indices)
            or sorted(set(indices)) != indices or not indices
            or indices[0] != 0 or indices[-1] >= 64):
        raise ValueError("invalid supplemental hybrid indices")


def load(root=runtime.OUT):
    root = Path(root)
    path = root / NAME
    record = runtime.read(path)
    spec = runtime.load_amendment(root)
    if record["format"] != _FORMAT or record["base_amendment_sha256"] != runtime.sha(root / runtime.AMENDMENT):
        raise ValueError("supplement does not match the base execution amendment")
    start = record["start_generation"]
    if type(start) is not int or not spec["start_generation"] < start < 32:
        raise ValueError("invalid supplemental generation boundary")
    indices = record["hybrid_indices"]
    _validate_indices(indices)
    if record["issued_plans"] != _plan_hashes(root, start):
        raise ValueError("a plan issued before the supplemental boundary has changed")
    return Schedule(start, tuple(indices), runtime.sha(path))


def prepare(hybrid_indices, root=runtime.OUT):
    root = Path(root)
    runner = runtime.runner
    path = root / NAME
    indices = list(hybrid_indices)
    _validate_indices(indices)
    with runner._lock(root / ".pipeline.lock"), runner._lock(root / ".run.lock"):
        if path.exists():
            raise FileExistsError(path)
        spec = runtime.load_amendment(root)
        issued = sorted(p.relative_to(root).as_posix() for p in (root / "plans").glob("generation_*.json"))
        start = len(issued)
        if not spec["start_generation"] < start < 32:
            raise ValueError("no unissued generation after the original execution boundary")
        plans = _plan_hashes(root, start)
        if list(plans) != issued:
            raise ValueError("issued plan inventory is not contiguous")
        for artifact in (root / "search").glob("rep_*/*/attempt.json"):
            if int(artifact.parent.name[-6:-4]) >= start:
                raise ValueError(f"an unissued generation already has an attempt: {artifact}")
        runner._write(path, dict(
            format=_FORMAT, created_utc=runner._utc(), start_generation=start,
            hybrid_indices=indices, issued_plans=plans,
            base_amendment_sha256=runtime.sha(root / runtime.AMENDMENT),
        ))
        return load(root)
