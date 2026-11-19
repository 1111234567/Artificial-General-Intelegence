"""Package the stopped large seed-evolution run for a Debian CPU host."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import lzma
import os
from pathlib import Path, PurePosixPath
import re
import statistics
import tarfile
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parents[1]
RUN = "data/seed-evolution/2026-11-11-large-v1"
TOP = "seed-evolution-debian-cpu"
EXPERIMENTS = (
    "action_split", "associative_state", "continuity", "coverage_retention",
    "experience_transfer", "learn_remember", "learned_binding", "lifelong",
    "remaining_time", "seed_evolution", "teacher_timing", "timing_boundary",
)
CLOUD_SOURCES = tuple(f"experiments/seed_evolution/{name}.py" for name in (
    "cloud_cpu", "cloud_cpu_final", "cloud_cpu_audit",
))
CANDIDATE_ARTIFACTS = ("initial.pt", "inner64.pt", "score.json", "training.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def portable(name):
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    require(path.parts and not path.is_absolute() and ".." not in path.parts
            and ":" not in normalized and normalized == path.as_posix(),
            f"Invalid relative path: {name}")
    return normalized


def portable_hashes(entries):
    result = {portable(name): digest for name, digest in entries.items()}
    require(len(result) == len(entries), "Duplicate paths after normalization")
    return result


@dataclass(frozen=True)
class Member:
    source: Path | bytes
    digest: str | None = None
    executable: bool = False


def bind_sources(members, run, original_sources, amendment_sources):
    snapshot_manifest = (run / "source-snapshot-manifest.json").read_bytes()
    snapshot = json.loads(snapshot_manifest)
    require(snapshot["protocol_sha256"] == members[f"{RUN}/migration/original/protocol.json"].digest
            and portable_hashes(snapshot["files"]) == original_sources,
            "Frozen source snapshot does not match the original protocol")
    members[f"{RUN}/source-snapshot-manifest.json"] = Member(snapshot_manifest)

    for name in EXPERIMENTS:
        for path in sorted((ROOT / "experiments" / name).rglob("*.py")):
            if (path.name.startswith("test_") or path.name.endswith("_test.py")
                    or {"__pycache__", "tests", "test"}.intersection(path.parts)):
                continue
            members[path.relative_to(ROOT).as_posix()] = Member(path)

    bindings = dict(original_sources)
    for name, digest in amendment_sources.items():
        require(name not in bindings or bindings[name] == digest,
                f"Original protocol and amendment disagree on {name}")
        bindings[name] = digest
    for name, digest in bindings.items():
        path = ROOT / name
        if name in original_sources:
            path = run / "source-snapshot" / name
            frozen = Member(path, digest)
            members[f"{RUN}/source-snapshot/{name}"] = frozen
            members[name] = frozen
        else:
            members[name] = Member(path, digest)
    for name in CLOUD_SOURCES:
        data = (ROOT / name).read_bytes()
        bindings[name] = sha(data)
        members[name] = Member(data, bindings[name])

    bootstrap = ROOT / "scripts/cloud_cpu_bundle"
    require((bootstrap / "run.sh").is_file(), "Missing Debian launcher scripts/cloud_cpu_bundle/run.sh")
    for path in sorted(bootstrap.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        name = portable(path.name)
        require(name not in members, f"Duplicate bootstrap path: {name}")
        data = path.read_bytes()
        if path.suffix == ".sh":
            require(b"\r" not in data, f"Debian launcher requires LF line endings: {path}")
        bindings[name] = sha(data)
        members[name] = Member(data, bindings[name], executable=path.suffix == ".sh")
    return bindings


def freeze_inputs(members, protocol, original):
    origins = protocol["origins"]
    require(len(origins) == 3, "Expected three original executor checkpoints")
    for row in origins.values():
        name = row["path"] = portable(row["path"])
        members[name] = Member(ROOT / name, row["sha256"])
    require(len({row["path"] for row in origins.values()}) == 3, "Duplicate executor checkpoints")

    reference = protocol["data_freeze_ref"]
    name = reference["path"] = portable(reference["path"])
    freeze_data = (ROOT / name).read_bytes()
    require(sha(freeze_data) == reference["sha256"], "Original data-freeze manifest changed")
    original["data-freeze.json"] = freeze_data
    freeze = json.loads(freeze_data)
    streams = freeze["streams"]
    require(len(streams) == len({portable(row["path"]) for row in streams}) == 48,
            "Expected 48 distinct original training streams")
    for row in streams:
        stream = row["path"] = portable(row["path"])
        members[stream] = Member(ROOT / stream, row["sha256"])
    data = json_bytes(freeze)
    reference["sha256"] = sha(data)
    members[name] = Member(data, reference["sha256"])


def numbered_files(folder, digits=None):
    paths = {}
    for path in folder.glob("generation_*.json"):
        match = re.fullmatch(r"generation_(\d+)\.json", path.name)
        require(match is not None, f"Invalid generation filename: {path}")
        generation = int(match[1])
        suffix = str(generation) if digits is None else f"{generation:0{digits}d}"
        require(path.name == f"generation_{suffix}.json", f"Invalid generation filename: {path}")
        paths[generation] = path
    require(set(paths) == set(range(len(paths))), f"Noncontiguous generations in {folder}")
    return paths


def committed_candidate(members, run, descriptor):
    rep, cid = descriptor["rep"], descriptor["id"]
    require(type(rep) is int and rep in (0, 1, 2)
            and re.fullmatch(r"(?:shared|evolution|random)_g\d{2}_i\d{2}", cid),
            f"Invalid candidate identity in plan: {descriptor}")
    name = f"search/rep_{rep}/{cid}"
    folder = run / name
    try:
        marker = (folder / "candidate.json").read_bytes()
    except FileNotFoundError:
        return name, None
    record = json.loads(marker)
    require(all(record[key] == value for key, value in descriptor.items()),
            f"Candidate does not match its issued plan: {name}")
    artifacts = record["artifacts"]
    require(set(artifacts) == set(CANDIDATE_ARTIFACTS), f"Invalid candidate artifact list: {name}")
    prefix = f"{RUN}/{name}"
    members[f"{prefix}/candidate.json"] = Member(marker, sha(marker))
    for artifact in CANDIDATE_ARTIFACTS:
        members[f"{prefix}/{artifact}"] = Member(folder / artifact, artifacts[artifact])

    attempt = (folder / "attempt.json").read_bytes()
    require(json.loads(attempt)["attempt"] == record["attempt"], f"Candidate attempt differs: {name}")
    members[f"{prefix}/attempt.json"] = Member(attempt)
    training = (folder / "training.json").read_bytes()
    evaluation = (folder / "evaluation-time.json").read_bytes()
    for stage, data, artifact, hash_key in (
        ("train", training, "inner64.pt", "checkpoint_sha256"),
        ("eval", evaluation, "score.json", "score_sha256"),
    ):
        manifest = json.loads(data)
        require(manifest[hash_key] == artifacts[artifact]
                and manifest[f"{stage}_attempt"] == record[f"{stage}_attempt"]
                and manifest[f"{stage}_seconds"] == record[f"{stage}_seconds"],
                f"Candidate {stage} manifest differs: {name}")
    members[f"{prefix}/training.json"] = Member(training, artifacts["training.json"])
    members[f"{prefix}/evaluation-time.json"] = Member(evaluation)
    execution = folder / "execution.json"
    if execution.is_file():
        members[f"{prefix}/execution.json"] = Member(execution)
    return name, record


def add_history(members, run, protocol, amendment):
    config = protocol["config"]
    commits = numbered_files(run)
    plans = numbered_files(run / "plans", digits=2)
    cutover = len(commits)
    require(len(plans) == cutover + 1 <= config["generations"],
            "Expected a contiguous sealed prefix followed by one issued generation")
    migration = dict(cutover_generation=cutover, historical_candidates={}, historical_generations={},
                     historical_plans={}, discarded_inflight_candidate_dirs=[])
    records = {}
    distributions = []
    pending = 0
    seen = set()
    for generation in range(len(plans)):
        plan_file = plans[generation]
        plan_name = plan_file.relative_to(run).as_posix()
        data = plan_file.read_bytes()
        plan = json.loads(data)
        digest = sha(data)
        require(plan["generation"] == generation, f"Generation number differs: {plan_file}")
        members[f"{RUN}/{plan_name}"] = Member(data, digest)
        migration["historical_plans"][plan_name] = digest
        ids = [dict(rep=row["rep"], id=row["id"]) for row in plan["candidates"]]
        if generation < cutover:
            commit_file = commits[generation]
            data = commit_file.read_bytes()
            commit = json.loads(data)
            require(commit["generation"] == generation and commit["plan_sha256"] == digest
                    and commit["candidate_ids"] == ids, f"Sealed generation does not match plan: {commit_file}")
            digest = sha(data)
            members[f"{RUN}/{commit_file.name}"] = Member(data, digest)
            migration["historical_generations"][commit_file.name] = digest
        for descriptor in plan["candidates"]:
            name, record = committed_candidate(members, run, descriptor)
            require(name not in seen and descriptor["generation"] == generation,
                    f"Unexpected or repeated candidate in plan: {name}")
            seen.add(name)
            if record is None:
                require(generation == cutover, f"Sealed generation lacks a committed candidate: {name}")
                pending += 1
                if (run / name).is_dir():
                    migration["discarded_inflight_candidate_dirs"].append(name)
                continue
            marker = f"{name}/candidate.json"
            migration["historical_candidates"][marker] = members[f"{RUN}/{marker}"].digest
            records[descriptor["rep"], descriptor["id"]] = record["fitness"]
        for entry in plan["entries"]:
            ids = [(entry["rep"], cid) for cid in entry["new_ids"]]
            values = [records[key] for key in ids if key in records]
            row = dict(rep=entry["rep"], arm=entry["arm"], generation=generation,
                       n=len(values), planned=len(ids))
            if values:
                row.update(min=min(values), max=max(values), mean=statistics.fmean(values),
                           median=statistics.median(values))
            distributions.append(row)

    found = {path.relative_to(run).as_posix() for path in (run / "search").glob("rep_*/*/candidate.json")}
    require(found == set(migration["historical_candidates"]), "Unexpected committed candidate outside issued plans")
    for name, digest in portable_hashes(amendment["issued_plans"]).items():
        require(migration["historical_plans"].get(name) == digest, f"Originally issued plan changed: {name}")

    omitted = {}
    for name, digest in portable_hashes(amendment["existing_prefix_inventory"]).items():
        resolved = name
        if name.endswith("/.training.json.pending"):
            resolved = name.removesuffix(".training.json.pending") + "training.json"
        archived = members.get(f"{RUN}/{resolved}")
        if archived is None:
            pending_path = run / name
            require(sha(pending_path.read_bytes()) == digest, f"Old cutover artifact changed: {name}")
            omitted[name] = digest
        else:
            require(archived.digest == digest, f"Old cutover artifact changed: {name}")
    migration.update(historical_count=len(records), uncommitted_count=pending,
                     omitted_uncommitted_prefix_artifacts=omitted)
    progress = dict(phase="search", state="migrated", updated_utc=datetime.now(timezone.utc).isoformat(),
                    generation=cutover, completed_generations=cutover, completed_candidates=len(records),
                    total_candidates=protocol["unique_candidates"], generation_distributions=distributions)
    members[f"{RUN}/progress.json"] = Member(json_bytes(progress))
    return migration


def bundle_members(wheelhouse=None):
    run = ROOT / RUN
    original = {name: (run / name).read_bytes() for name in (
        "protocol.json", "execution-amendment.json", "progress.json",
    )}
    protocol = json.loads(original["protocol.json"])
    amendment = json.loads(original["execution-amendment.json"])
    require(amendment["format"] == "seed-evolution-execution-v1"
            and sha(original["protocol.json"]) == amendment["original_protocol_sha256"],
            "Original execution amendment does not bind the original protocol")

    members = {}
    freeze_inputs(members, protocol, original)
    original_hashes = {name: sha(data) for name, data in original.items()}
    for name, data in original.items():
        members[f"{RUN}/migration/original/{name}"] = Member(data, original_hashes[name])
    protocol["source_hashes"] = bind_sources(
        members, run, portable_hashes(protocol["source_hashes"]), portable_hashes(amendment["sources"]),
    )
    portable_protocol = json_bytes(protocol)
    protocol_digest = sha(portable_protocol)
    members[f"{RUN}/protocol.json"] = Member(portable_protocol, protocol_digest)

    history = add_history(members, run, protocol, amendment)
    migration = dict(format="seed-evolution-debian-cpu-migration-v1", original=original_hashes,
                     portable_protocol_sha256=protocol_digest, **history)
    members[f"{RUN}/migration.json"] = Member(json_bytes(migration))

    if wheelhouse is not None:
        wheels = sorted(wheelhouse.glob("*.whl"))
        require(wheels, f"No wheels found in {wheelhouse}")
        digests = {}
        sums = wheelhouse / "SHA256SUMS"
        if sums.is_file():
            data = sums.read_bytes()
            for line in data.decode("utf-8").splitlines():
                digest, name = line.split(maxsplit=1)
                name = portable(name.removeprefix("*"))
                require(re.fullmatch(r"[0-9a-f]{64}", digest) and name not in digests,
                        f"Invalid wheel checksum entry: {line}")
                digests[name] = digest
            require(digests.keys() == {path.name for path in wheels}, "Wheelhouse checksum inventory differs")
            members["wheelhouse/SHA256SUMS"] = Member(data)
        for path in wheels:
            members[f"wheelhouse/{portable(path.name)}"] = Member(path, digests.get(path.name))
        for path in sorted(wheelhouse.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS" and (
                    "sha256" in path.name.lower() or path.name == "wheelhouse-manifest.json"):
                members[f"wheelhouse/{portable(path.name)}"] = Member(path)
    return dict(sorted(members.items())), migration


def stream_digest(source):
    digest = hashlib.sha256()
    while block := source.read(1024 * 1024):
        digest.update(block)
    return digest.hexdigest()


class HashingReader:
    def __init__(self, source):
        self.source = source
        self.digest = hashlib.sha256()

    def read(self, size):
        data = self.source.read(size)
        self.digest.update(data)
        return data


def copy_members(members, tar=None):
    hashes = {}
    for name, member in members.items():
        info = tarfile.TarInfo(f"{TOP}/{name}")
        info.mode = 0o755 if member.executable else 0o644
        source = member.source
        with io.BytesIO(source) if isinstance(source, bytes) else source.open("rb") as handle:
            info.size = len(source) if isinstance(source, bytes) else os.fstat(handle.fileno()).st_size
            if tar is None:
                actual = stream_digest(handle)
            else:
                reader = HashingReader(handle)
                tar.addfile(info, reader)
                require(not handle.read(1), f"File grew while the stopped run was being packaged: {name}")
                actual = reader.digest.hexdigest()
        require(member.digest is None or actual == member.digest, f"Artifact SHA-256 differs: {name}")
        hashes[info.name] = actual
    return hashes


def verify_archive(path, hashes):
    remaining = set(hashes)
    with lzma.open(path, "rb") as compressed:
        with tarfile.open(fileobj=compressed, mode="r|", bufsize=1024 * 1024) as tar:
            for member in tar:
                name = member.name
                require(member.isfile() and name in remaining, f"Unexpected archive member: {name}")
                with tar.extractfile(member) as source:
                    require(stream_digest(source) == hashes[name], f"Archive member SHA-256 differs: {name}")
                remaining.remove(name)
        while compressed.read(1024 * 1024):
            pass
    require(not remaining, f"Archive omitted {len(remaining)} members")


def publish(members, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    checksum = output.with_name(output.name + ".sha256")
    require(not output.exists() and not checksum.exists(), f"Destination already exists: {output}")
    archive_tmp = None
    checksum_tmp = None
    try:
        with NamedTemporaryFile(dir=output.parent, prefix=output.name + ".", suffix=".part", delete=False) as file:
            archive_tmp = Path(file.name)
            with lzma.LZMAFile(file, "wb", preset=1) as compressor:
                with tarfile.open(fileobj=compressor, mode="w|", copybufsize=1024 * 1024) as tar:
                    hashes = copy_members(members, tar)
            file.flush()
            os.fsync(file.fileno())
        print(json.dumps(dict(event="archive-written", members=len(hashes), bytes=archive_tmp.stat().st_size)),
              flush=True)
        verify_archive(archive_tmp, hashes)
        with archive_tmp.open("rb") as file:
            digest = stream_digest(file)
        with NamedTemporaryFile(dir=output.parent, prefix=checksum.name + ".", suffix=".part", delete=False) as file:
            checksum_tmp = Path(file.name)
            file.write(f"{digest}  {output.name}\n".encode("utf-8"))
            file.flush()
            os.fsync(file.fileno())
        archive_tmp.replace(output)
        checksum_tmp.replace(checksum)
    finally:
        for path in (archive_tmp, checksum_tmp):
            if path is not None:
                path.unlink(missing_ok=True)
    return digest, checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--output", type=Path, metavar="BUNDLE.tar.xz")
    modes.add_argument("--check-only", action="store_true", help="Validate source files without writing an archive")
    parser.add_argument("--wheelhouse", type=Path, help="Optional Debian CPU wheels and checksum manifests")
    args = parser.parse_args()
    if args.output is not None:
        args.output = args.output.resolve()
        if not args.output.name.endswith(".tar.xz"):
            parser.error("--output must end in .tar.xz")
        if args.output.is_relative_to((ROOT / RUN).resolve()):
            parser.error("--output must be outside the original run directory")

    members, migration = bundle_members(args.wheelhouse)
    summary = dict(members=len(members), candidates=migration["historical_count"],
                   sealed_generations=migration["cutover_generation"],
                   issued_plans=len(migration["historical_plans"]),
                   uncommitted=migration["uncommitted_count"])
    print(json.dumps(dict(event="inventory", **summary)), flush=True)
    if args.check_only:
        copy_members(members)
        print(json.dumps(dict(event="source-verified", **summary)), flush=True)
        return
    digest, checksum = publish(members, args.output)
    print(json.dumps(dict(event="archive-ready", archive=str(args.output), checksum=str(checksum),
                          sha256=digest, bytes=args.output.stat().st_size, **summary)), flush=True)


if __name__ == "__main__":
    main()
