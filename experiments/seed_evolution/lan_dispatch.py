"""Stage assigned CPU search jobs and start the private LAN worker over SSH."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from uuid import uuid4

from experiments.seed_evolution import balanced_schedule, execution_runtime as runtime
from experiments.seed_evolution.audit import same_tree


runner = runtime.runner
_PARENT = re.compile(r"(?:shared|evolution)_g(\d{2})_i(\d{2})")


def _publish_file(destination, payload):
    """Publish through a same-directory rename without overwriting a prior copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if destination.read_bytes() != payload:
            raise ValueError(f"remote immutable input differs: {destination}")
        return

    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
        try:
            if os.name == "nt":
                # Windows rename fails when destination already exists.
                temporary.rename(destination)
            else:
                os.link(temporary, destination)
        except OSError:
            if not destination.is_file():
                raise
        if destination.read_bytes() != payload:
            raise ValueError(f"remote immutable input differs: {destination}")
    finally:
        temporary.unlink(missing_ok=True)


def _parent_files(rep, parent, generation):
    match = _PARENT.fullmatch(parent)
    if match is None or int(match[1]) >= generation or int(match[2]) >= runner.CONFIG["population"]:
        raise ValueError(f"invalid parent identity: {parent}")
    folder = runner._candidate_dir(rep, parent)
    candidate = (folder / "candidate.json").read_bytes()
    record = json.loads(candidate)
    if type(record["rep"]) is not int or (record["rep"], record["id"]) != (rep, parent):
        raise ValueError(f"parent record does not bind rep_{rep}/{parent}")
    initial = (folder / "initial.pt").read_bytes()
    if hashlib.sha256(initial).hexdigest() != record["artifacts"]["initial.pt"]:
        raise ValueError(f"parent initialization changed: rep_{rep}/{parent}")
    return folder.relative_to(runner.ROOT), initial, candidate


@dataclass(frozen=True)
class Shard:
    remote_run: Path
    manifest_path: Path
    plan_sha256: str
    jobs: tuple[dict, ...]
    command: tuple[str, ...]

    def start(self, stdout, stderr):
        """Start one SSH process; the caller owns polling and log handles."""
        return subprocess.Popen(self.command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                shell=False)

    def result_path(self, descriptor):
        if descriptor not in self.jobs:
            raise ValueError("requested result was not assigned to this shard")
        return self.remote_run / "search" / f"rep_{descriptor['rep']}" / descriptor["id"]


class ShardTransport:
    def __init__(self, remote_root: Path, ssh_args: list[str], remote_command_base: str, workers: int = 3):
        self.remote_root = Path(remote_root)
        if not self.remote_root.is_absolute():
            raise ValueError("remote root must be an absolute private bundle path")
        if self.remote_root == runner.ROOT.resolve():
            raise ValueError("remote bundle must be distinct from the active repository")
        if not ssh_args or any(not isinstance(arg, str) or not arg for arg in ssh_args):
            raise ValueError("SSH args must include the executable and connection destination")
        if not isinstance(remote_command_base, str) or not remote_command_base.strip():
            raise ValueError("remote command must name the bundle launcher")
        if type(workers) is not int or not 1 <= workers <= 8:
            raise ValueError("remote workers must be in 1..8")
        self.remote_run = self.remote_root / runner.OUT.relative_to(runner.ROOT)
        self.ssh_args = tuple(ssh_args)
        self.remote_command_base = remote_command_base.strip()
        self.workers = workers

    def prepare(self, jobs, plan_path, *, spec=None, schedule=None):
        jobs = tuple(dict(descriptor) for descriptor in jobs)
        if not jobs:
            raise ValueError("cannot stage an empty shard")
        if not self.remote_root.is_dir() or not (self.remote_root / "lan-start.py").is_file():
            raise FileNotFoundError(f"LAN worker bundle is unavailable: {self.remote_root}")
        plan_path = Path(plan_path).resolve(strict=True)
        plan_bytes = plan_path.read_bytes()
        plan = json.loads(plan_bytes)
        generation = plan["generation"]
        if type(generation) is not int or not 0 <= generation < runner.CONFIG["generations"]:
            raise ValueError("invalid issued generation")
        if plan_path != runner._plan_path(generation).resolve():
            raise ValueError("shard must use the issued plan in the active repository")

        if spec is None:
            spec = runtime.load_amendment()
        if schedule is None:
            schedule = balanced_schedule.load()
        plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
        planned = {(descriptor["rep"], descriptor["id"]): descriptor for descriptor in plan["candidates"]}
        if len(planned) != len(plan["candidates"]):
            raise ValueError("issued plan contains duplicate candidate identities")
        seen = set()
        parents = set()
        for descriptor in jobs:
            rep, cid = descriptor["rep"], descriptor["id"]
            identity = (rep, cid)
            if identity not in planned or identity in seen:
                raise ValueError(f"unplanned or repeated candidate in shard: {identity}")
            same_tree(descriptor, planned[identity], "shard descriptor differs from issued plan")
            if (descriptor["generation"] != generation or rep not in runner.CONFIG["source_seeds"]
                    or descriptor["source_seed"] != rep or schedule.backend(cid, spec) != "cpu"):
                raise ValueError(f"candidate is not assigned to the CPU backend: {identity}")
            seen.add(identity)
            if descriptor["parent_id"] is not None:
                parents.add((rep, descriptor["parent_id"]))

        for rep, parent in sorted(parents):
            relative, initial, candidate = _parent_files(rep, parent, generation)
            destination = self.remote_root / relative
            # A prior LAN attempt can leave different serialized bytes after local fallback.
            # Its scientific record remains separate from the primary host's parent snapshot.
            if any(path.is_file() and path.read_bytes() != payload for path, payload in (
                    (destination / "initial.pt", initial), (destination / "candidate.json", candidate))):
                archive = self.remote_run / "superseded" / f"rep_{rep}" / f"{parent}-{uuid4().hex}"
                archive.parent.mkdir(parents=True, exist_ok=True)
                destination.rename(archive)
            _publish_file(destination / "initial.pt", initial)
            _publish_file(destination / "candidate.json", candidate)
        remote_plan = self.remote_root / plan_path.relative_to(runner.ROOT)
        _publish_file(remote_plan, plan_bytes)

        manifest = dict(generation=generation, plan_sha256=plan_sha256, jobs=jobs)
        name = f"lan-shard-g{generation:02d}-{uuid4().hex}.json"
        manifest_path = self.remote_root / name
        _publish_file(manifest_path, runner._json_bytes(manifest))
        remote_command = f"{self.remote_command_base} --jobs {name} --workers {min(self.workers, len(jobs))}"
        return Shard(self.remote_run, manifest_path, plan_sha256, jobs, (*self.ssh_args, remote_command))
