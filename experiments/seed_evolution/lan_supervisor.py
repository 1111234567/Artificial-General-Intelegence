"""Monitor the local LAN scheduler and resume recoverable failures."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data/seed-evolution/2026-11-11-large-v1"
MODULE = "experiments.seed_evolution.lan_runtime"
STATUS = "lan-manager-status.json"
_TRANSIENT = (
    "brokenprocesspool", "cuda out of memory", "outofmemoryerror",
    "cudnn_status_execution_failed_cudart", "cudnn_status_alloc_failed",
    "paging file is too small", "winerror 1455",
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _publish(path, value):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        for delay in (.02, .05, .1, .25):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                time.sleep(delay)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _lock(path):
    import msvcrt

    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError(f"lock is held: {path.name}") from error
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _argv(command):
    shell = ctypes.WinDLL("shell32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    shell.CommandLineToArgvW.argtypes = (ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int))
    shell.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
    kernel.LocalFree.argtypes = (ctypes.c_void_p,)
    kernel.LocalFree.restype = ctypes.c_void_p
    count = ctypes.c_int()
    result = shell.CommandLineToArgvW(command, ctypes.byref(count))
    if not result:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return [result[index] for index in range(count.value)]
    finally:
        kernel.LocalFree(result)


def _processes(pid):
    # The filter is constructed only after validating an integer PID.
    if type(pid) is not int or pid < 1:
        raise ValueError("invalid scheduler PID")
    command = (f"$p=@(Get-CimInstance Win32_Process -Filter 'ProcessId={pid} OR ParentProcessId={pid}' | "
               "Select-Object ProcessId,ParentProcessId,@{n='CreationDate';e={$_.CreationDate.ToUniversalTime().ToString('o')}},CommandLine); "
               "ConvertTo-Json -InputObject $p -Compress")
    output = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                            check=True, capture_output=True, text=True, timeout=15,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    return json.loads(output.stdout)


def _live(pid):
    """Check liveness without launching a shell for normal polls."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel.WaitForSingleObject.restype = ctypes.c_uint32
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel.OpenProcess(0x00100000, 0, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # ERROR_INVALID_PARAMETER for a PID with no process.
            return False
        raise ctypes.WinError(error)
    try:
        result = kernel.WaitForSingleObject(handle, 0)
        if result == 258:  # WAIT_TIMEOUT
            return True
        if result == 0:  # WAIT_OBJECT_0
            return False
        raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)


def _options(args):
    return {"--remote-root": str(args.remote_root), "--key": str(args.key),
            "--known-hosts": str(args.known_hosts), "--cpu-workers": str(args.cpu_workers),
            "--hybrid-workers": str(args.hybrid_workers), "--remote-workers": str(args.remote_workers),
            "--pairs": str(args.pairs), "--hybrid-start-free-mib": str(args.hybrid_start_free_mib)}


def _same_options(command, options):
    argv = _argv(command)
    try:
        module = argv.index("-m") + 1
        if argv[module] != MODULE:
            return False
    except (IndexError, ValueError):
        return False
    tail = argv[module + 1:]
    if len(tail) % 2:
        return False
    seen = dict(zip(tail[::2], tail[1::2]))
    defaults = {"--cpu-workers": "11", "--hybrid-workers": "4", "--remote-workers": "3",
                "--pairs": "28", "--hybrid-start-free-mib": "5600"}
    observed = defaults | seen
    if len(seen) != len(tail) // 2 or observed.keys() != options.keys():
        return False
    for name in ("--remote-root", "--key", "--known-hosts"):
        if os.path.normcase(os.path.normpath(observed[name])) != os.path.normcase(os.path.normpath(options[name])):
            return False
    return all(observed[name] == options[name] for name in defaults)


def _verify_identity(record, options):
    pid = record["pid"]
    processes = _processes(pid)
    selected = next((process for process in processes if process["ProcessId"] == pid), None)
    if selected is None:
        return False
    created = datetime.fromisoformat(selected["CreationDate"])
    published = datetime.fromisoformat(record["started_utc"])
    return (0 <= (published - created).total_seconds() <= 120
            and bool(selected["CommandLine"]) and _same_options(selected["CommandLine"], options))


def _validate_record(run, record):
    if (type(record.get("pid")) is not int or record["pid"] < 1
            or datetime.fromisoformat(record["started_utc"]).tzinfo is None):
        raise ValueError("invalid persisted scheduler identity")
    for name, field in (("execution-amendment.json", "amendment_sha"),
                        ("balanced-schedule.json", "balanced_schedule_sha256")):
        if record.get(field) != _sha(run / name):
            raise ValueError(f"persisted scheduler does not match current {name}")


def _completed(run, budget):
    marker = _read(run / "pipeline-completion.json")
    if marker is None:
        return False
    audit = _read(run / "audit.json")
    if (marker.get("full_audit_passed") is not True or audit is None or audit.get("mode") != "full"
            or audit.get("full_audit_passed") is not True
            or audit.get("candidates_provenance_checked") != budget
            or audit.get("heldout_ledgers_recounted") != 72 or audit.get("heldout_policy_replays") != 9):
        raise ValueError("pipeline completion has no matching full audit")
    for filename, key in (("execution-amendment.json", "execution_amendment_sha256"),
                          ("balanced-schedule.json", "balanced_schedule_sha256")):
        if marker.get(key) != audit.get(key) or marker[key] != _sha(run / filename):
            raise ValueError(f"pipeline completion evidence changed: {filename}")
    for filename, key in (("protocol.json", "protocol_sha256"), ("search.json", "search_sha256"),
                          ("final-freeze.json", "final_freeze_sha256"), ("test-completion.json", "test_completion_sha256")):
        if audit.get(key) != _sha(run / filename):
            raise ValueError(f"audited evidence changed: {filename}")
    return True


def _recoverable(run, record, progress):
    path = run / "execution-failure.json"
    failure = _read(path)
    if failure is None or progress.get("state") != "failed":
        return None
    started = datetime.fromisoformat(record["started_utc"]).timestamp()
    if path.stat().st_mtime < started or datetime.fromisoformat(progress["updated_utc"]).timestamp() < started:
        return None
    error = failure.get("error", "")
    # Ordinary artifact, validation and assertion errors need inspection.
    if progress.get("error") != error or not any(marker in error.lower() for marker in _TRANSIENT):
        return None
    return error


def _launch(run, options):
    name = f"lan-manager-restart-{uuid4().hex[:12]}"
    stdout = run / f"{name}.stdout.log"
    stderr = run / f"{name}.stderr.log"
    command = [sys.executable, "-u", "-m", MODULE]
    for key, value in options.items():
        command.extend((key, value))
    with stdout.open("xb") as log, stderr.open("xb") as errors:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=errors,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
    return child, {"pid": child.pid, "started_utc": _now(), "stdout": stdout.name, "stderr": stderr.name}


def _await_launch(run, child, launch, options, seconds):
    deadline = time.monotonic() + 120
    started = datetime.fromisoformat(launch["started_utc"]).timestamp()
    while time.monotonic() < deadline:
        record = _read(run / "execution-process.json")
        if record is not None and datetime.fromisoformat(record["started_utc"]).timestamp() >= started - 2:
            if _verify_identity(record, options):
                return record
        if child.poll() is not None:
            raise RuntimeError(f"replacement scheduler exited with code {child.returncode} before publishing its identity")
        time.sleep(min(seconds, 3))
    raise RuntimeError("replacement scheduler did not publish a verified identity")


def _await_previous_exit(run, record):
    deadline = time.monotonic() + 60
    while True:
        if _read(run / "execution-process.json") != record:
            raise RuntimeError("scheduler ownership changed while waiting to retry")
        if not _processes(record["pid"]):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("prior scheduler or its child processes are still present")
        time.sleep(3)
    with _lock(run / ".pipeline.lock"):
        if _read(run / "execution-process.json") != record:
            raise RuntimeError("scheduler ownership changed before relaunch")


def _committed_count(run):
    return sum(1 for _ in (run / "search").glob("rep_*/*/candidate.json"))


def _remote_progress(run, remote_root, progress):
    generation = progress.get("generation")
    if type(generation) is not int:
        return None
    plan = run / "plans" / f"generation_{generation:02d}.json"
    try:
        remote_root.stat()
        plan_sha = _sha(plan)
        remote_run = remote_root / run.relative_to(ROOT)
        manifests = sorted(remote_root.glob(f"lan-shard-g{generation:02d}-*.json"),
                           key=lambda path: path.stat().st_mtime_ns, reverse=True)
        for path in manifests:
            shard = _read(path)
            if (shard is None or shard.get("generation") != generation or shard.get("plan_sha256") != plan_sha):
                continue
            result = dict(generation=generation, manifest=path.name, assigned=len(shard["jobs"]),
                          remote_training=0, remote_scoring=0, remote_committed=0, local_imported=0)
            for job in shard["jobs"]:
                folder = Path("search") / f"rep_{job['rep']}" / job["id"]
                remote = remote_run / folder
                if (remote / "candidate.json").is_file():
                    result["remote_committed"] += 1
                elif (remote / "training.json").is_file():
                    result["remote_scoring"] += 1
                elif (remote / "attempt.json").is_file():
                    result["remote_training"] += 1
                if (run / folder / "lan-origin.json").is_file():
                    result["local_imported"] += 1
            result["remote_pending"] = result["assigned"] - sum(result[name] for name in (
                "remote_training", "remote_scoring", "remote_committed"))
            return result
    except (OSError, ValueError, KeyError) as error:
        return {"generation": generation, "error": str(error)}
    return None


def supervise(args, run=RUN):
    initial_options = _options(args)
    replacement_options = dict(initial_options)
    if args.restart_pairs is not None:
        replacement_options["--pairs"] = str(args.restart_pairs)
    budget = _read(run / "protocol.json")["unique_candidates"]
    status_path = run / STATUS

    with _lock(run / ".lan-manager.lock"):
        previous = _read(status_path) or {}
        progress = _read(run / "progress.json") or {}
        record = _read(run / "execution-process.json")
        committed = _committed_count(run)
        same_scope = (previous.get("scheduler_options") in (initial_options, replacement_options)
                      and previous.get("replacement_options", replacement_options) == replacement_options)
        options = previous["scheduler_options"] if same_scope else initial_options
        previous_count = previous.get("committed_high_water", 0) if same_scope else 0
        retries = previous.get("retries_without_progress", 0) if same_scope and committed <= previous_count else 0
        launch = previous.get("launch") if same_scope else None
        state = {"supervisor_pid": os.getpid(), "replacement_options": replacement_options}

        def publish(phase, error=None):
            state.update(state=phase, updated_utc=_now(), scheduler=record, progress=progress,
                         committed_high_water=committed, retries_without_progress=retries,
                         scheduler_options=options, launch=launch, error=error,
                         lan=_remote_progress(run, args.remote_root, progress) if phase == "monitoring" else state.get("lan"))
            _publish(status_path, state)

        try:
            if _completed(run, budget):
                publish("complete")
                return 0
            if record is None or (args.adopt_pid is not None and args.adopt_pid != record["pid"]):
                raise ValueError("no current scheduler matches the requested adoption")
            _validate_record(run, record)
            trusted = same_scope and previous.get("scheduler") == record
            if _live(record["pid"]):
                if not _verify_identity(record, options):
                    alternate = replacement_options if options == initial_options else initial_options
                    if alternate == options or not _verify_identity(record, alternate):
                        raise ValueError("current process is not the configured LAN scheduler")
                    options = alternate
            elif args.start_if_inactive:
                _await_previous_exit(run, record)
                options = initial_options
                publish("starting_scheduler")
                child, launch = _launch(run, options)
                publish("starting_scheduler")
                record = _await_launch(run, child, launch, options, args.poll_seconds)
                _validate_record(run, record)
            elif not trusted:
                raise ValueError("the current LAN scheduler is not live and has no prior manager attestation")
            publish("monitoring")

            while True:
                progress = _read(run / "progress.json") or {}
                if _completed(run, budget):
                    committed = budget
                    publish("complete")
                    return 0
                # During resume, published generation progress can lag the
                # authoritative candidate markers transferred over LAN.
                reported = progress.get("completed_candidates")
                if type(reported) is int and reported > committed:
                    actual = _committed_count(run)
                    if actual > committed:
                        committed, retries = actual, 0
                if _live(record["pid"]):
                    publish("monitoring")
                    time.sleep(args.poll_seconds)
                    continue
                if _read(run / "execution-process.json") != record:
                    raise RuntimeError("scheduler identity changed outside the manager")
                actual = _committed_count(run)
                if actual > committed:
                    committed, retries = actual, 0
                failure = _recoverable(run, record, progress)
                if failure is None:
                    raise RuntimeError("scheduler exited without a verified recoverable failure")
                if retries >= args.max_retries:
                    raise RuntimeError(f"retry budget exhausted without new committed candidates: {failure}")
                publish("waiting_to_retry", failure)
                time.sleep(args.retry_delay_seconds)
                if _live(record["pid"]) or _read(run / "execution-process.json") != record:
                    raise RuntimeError("scheduler ownership changed while waiting to retry")
                _await_previous_exit(run, record)
                retries += 1
                publish("starting_replacement", failure)
                child, launch = _launch(run, replacement_options)
                publish("starting_replacement", failure)
                record = _await_launch(run, child, launch, replacement_options, args.poll_seconds)
                _validate_record(run, record)
                options = replacement_options
        except KeyboardInterrupt:
            publish("stopped_monitoring")
            return 130
        except Exception as error:
            publish("needs_attention", str(error))
            print(str(error), file=sys.stderr, flush=True)
            return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-root", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--known-hosts", type=Path, required=True)
    parser.add_argument("--cpu-workers", type=int, default=11)
    parser.add_argument("--hybrid-workers", type=int, default=4)
    parser.add_argument("--remote-workers", type=int, default=3)
    parser.add_argument("--pairs", type=int, default=28)
    parser.add_argument("--restart-pairs", type=int, help="Override candidate pair count on relaunch only")
    parser.add_argument("--hybrid-start-free-mib", type=int, default=5600)
    parser.add_argument("--adopt-pid", type=int, help="Require the initial scheduler to have this PID")
    parser.add_argument("--start-if-inactive", action="store_true",
                        help="Start the configured scheduler if the recorded process and its children have exited")
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--retry-delay-seconds", type=float, default=30)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("LAN manager requires Windows")
    if (min(args.cpu_workers, args.hybrid_workers, args.remote_workers, args.pairs, args.hybrid_start_free_mib) < 1
            or (args.restart_pairs is not None and args.restart_pairs < 1)
            or not 1 <= args.poll_seconds <= 300 or not 0 <= args.retry_delay_seconds <= 300
            or not 0 <= args.max_retries <= 10):
        parser.error("invalid worker, polling or retry limits")
    return supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
