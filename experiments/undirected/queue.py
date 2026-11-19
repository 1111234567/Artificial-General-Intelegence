"""Finite source-bound queues with explicit scientific terminal outcomes.

An algorithm's numerical failure is a terminal *training outcome*: its wrapper
exits zero with a validated marker and its evaluation job can emit empty records.
Process, resource, source and implementation failures are execution failures and
remain pending for a later explicit queue resume.  A completed queue does not
mean every algorithm reached its endpoint or obtained a behavioral score.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


MANIFEST_FORMAT = "undirected-command-plan-v1"
TERMINAL_FORMAT = "undirected-branch-terminal-v1"
QUEUE_FORMAT = "undirected-queue-execution-v1"
KINDS = ("training", "baseline", "evaluation")


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    for attempt in range(8):
        try:
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(.02 * 2 ** attempt, .2))


save = atomic_json


def verify_files(files):
    if not isinstance(files, dict):
        raise ValueError("source inventory must be a mapping")
    for name, expected in files.items():
        path = Path(name)
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"frozen source/input changed: {name}")


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _bound_file(entry, parent, label):
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not _hash(entry.get("sha256")):
        raise ValueError(f"invalid terminal {label} binding")
    path = Path(entry["path"]).resolve()
    if not path.is_relative_to(parent.resolve()) or not path.is_file() or file_sha256(path) != entry["sha256"]:
        raise ValueError(f"terminal {label} bytes/path changed")
    return path


def terminal(job):
    """Read and validate a ready dependency, or return None if not yet written."""
    if not isinstance(job.get("terminal"), str):
        raise ValueError("job must declare its terminal dependency")
    marker = Path(job["terminal"])
    if not marker.exists():
        return None
    value = _json(marker)
    if not isinstance(value, dict) or value.get("format") != TERMINAL_FORMAT:
        raise ValueError("untrusted terminal format")
    branch = value.get("branch")
    if (not isinstance(branch, dict) or branch.get("id") != job.get("branch_id", job.get("id"))
            or ("branch" in job and branch != job["branch"])):
        raise ValueError("foreign terminal branch")
    if (not _hash(job.get("protocol_sha256")) or value.get("protocol_sha256") != job["protocol_sha256"]
            or value.get("target_ticks") != job.get("target_ticks")):
        raise ValueError("foreign terminal protocol or target")
    _integer(job.get("target_ticks"), "job target", 1)
    target = _integer(value.get("target_ticks"), "terminal target", 1)
    last = _integer(value.get("last_tick"), "terminal last tick")
    if not isinstance(value.get("checkpoints"), dict):
        raise ValueError("terminal must declare retained checkpoints")
    status = value.get("status")
    if status == "success":
        if last != target or value.get("completed") is not True or value.get("endpoint_comparable") is not True:
            raise ValueError("success terminal has incomplete progress")
        if value.get("failure") is not None:
            raise ValueError("success terminal cannot contain a failure binding")
    elif status == "numerical_failure":
        if last >= target or value.get("completed") is not False or value.get("endpoint_comparable") is not False:
            raise ValueError("numerical terminal has invalid endpoint semantics")
        _bound_file(value.get("failure"), marker.parent, "failure")
    else:
        raise ValueError("execution errors are not scientific terminal outcomes")
    _bound_file({"path": value.get("checkpoint_path"), "sha256": value.get("checkpoint_sha256")},
                marker.parent, "checkpoint")
    return value


def ready(job):
    return terminal(job) is not None


def available_memory():
    if os.name != "nt":
        return float("inf")

    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in (
                "total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]

    value = Memory()
    value.length = ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise OSError("memory status unavailable")
    return value.available


def live_pid(pid):
    if type(pid) is not int or pid < 1:
        return False
    if os.name == "nt":
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
        kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def terminate_tree(process):
    if process.poll() is None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _limits(value):
    if not isinstance(value, dict):
        raise ValueError("limits must be a mapping")
    if "max_workers" in value and "total" in value and value["max_workers"] != value["total"]:
        raise ValueError("conflicting max_workers and total limits")
    total = _integer(value.get("max_workers", value.get("total")), "max_workers", 1)
    training = _integer(value.get("training", total), "training workers", 1)
    evaluation = _integer(value.get("evaluation", value.get("cpu", total)), "evaluation workers", 1)
    memory = _integer(value.get("min_available_bytes", 0), "minimum available bytes")
    return {"max_workers": total, "training": training, "evaluation": evaluation, "min_available_bytes": memory}


def _load_plan(kind, path):
    path = Path(path).resolve()
    plan = _json(path)
    if not isinstance(plan, dict) or plan.get("format") != MANIFEST_FORMAT or plan.get("kind") != kind:
        raise ValueError("foreign command manifest format/kind")
    if not isinstance(plan.get("protocol"), str) or not isinstance(plan.get("file_hashes"), dict):
        raise ValueError("invalid command manifest protocol/source inventory")
    protocol = Path(plan["protocol"]).resolve()
    if (not protocol.is_file() or not _hash(plan.get("protocol_sha256"))
            or file_sha256(protocol) != plan["protocol_sha256"]
            or plan.get("file_hashes", {}).get(str(protocol)) != plan["protocol_sha256"]):
        raise ValueError("command manifest protocol binding changed")
    verify_files(plan["file_hashes"])
    if not isinstance(plan.get("cwd"), str) or not Path(plan["cwd"]).is_dir() or not isinstance(plan.get("jobs"), list):
        raise ValueError("invalid command manifest cwd/jobs")
    identifiers, jobs = set(), []
    protocol_value = _json(protocol)
    if not isinstance(protocol_value, dict):
        raise ValueError("protocol must be a mapping")
    target = protocol_value.get("target_ticks")
    for original in plan["jobs"]:
        if not isinstance(original, dict) or not isinstance(original.get("id"), str) or not original["id"] or original["id"] in identifiers:
            raise ValueError("invalid or duplicate command job identity")
        identifiers.add(original["id"])
        command, outputs = original.get("command"), original.get("outputs")
        if (not isinstance(command, list) or not command or any(not isinstance(arg, str) for arg in command)
                or not isinstance(outputs, list) or not outputs or any(not isinstance(name, str) or not Path(name).is_absolute() for name in outputs)
                or len(set(outputs)) != len(outputs) or not isinstance(original.get("log"), str)):
            raise ValueError("invalid command job argv/outputs/log")
        job = deepcopy(original)
        if kind in ("training", "evaluation"):
            job.setdefault("branch_id", job["id"])
            job.setdefault("protocol_sha256", plan["protocol_sha256"])
            job.setdefault("target_ticks", target)
            if "terminal" not in job and kind == "training":
                markers = [name for name in outputs if Path(name).name == "terminal.json"]
                if len(markers) == 1:
                    job["terminal"] = markers[0]
            if (not isinstance(job.get("terminal"), str) or not Path(job["terminal"]).is_absolute()
                    or job["protocol_sha256"] != plan["protocol_sha256"] or job["branch_id"] != job["id"]):
                raise ValueError("invalid command job terminal dependency")
            _integer(job.get("target_ticks"), "job target ticks", 1)
        jobs.append(job)
    return plan, jobs


def _output_hashes(job):
    if any(not Path(name).is_file() for name in job["outputs"]):
        raise ValueError("successful job outputs missing")
    return {name: file_sha256(name) for name in job["outputs"]}


def _verify_running_binding(state, plan):
    if digest(_json(state["plan"])) != state["plan_digest"]:
        raise ValueError("command manifest changed during execution")
    verify_files(plan["file_hashes"])


def execute(plans, folder, limits_path, *, poll_interval=1.):
    """Execute/resume all three finite manifests; refresh limits every poll."""
    if set(plans) != set(KINDS):
        raise ValueError("training, baseline and evaluation plans are all required")
    if isinstance(poll_interval, bool) or not isinstance(poll_interval, (int, float)) or not math.isfinite(poll_interval) or poll_interval <= 0:
        raise ValueError("positive finite poll interval required")
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    states, pending, common_protocol = {}, {}, None
    for kind in KINDS:
        path = Path(plans[kind]).resolve()
        plan, jobs = _load_plan(kind, path)
        binding = (plan["protocol"], plan["protocol_sha256"])
        if common_protocol is not None and binding != common_protocol:
            raise ValueError("queues belong to different protocols")
        common_protocol = binding
        state_path = folder / f"{kind}-execution.json"
        existed = state_path.exists()
        state = (_json(state_path) if existed else
                 {"format": QUEUE_FORMAT, "kind": kind, "plan": str(path), "plan_digest": digest(plan), "jobs": {}})
        if (state.get("format") != QUEUE_FORMAT or state.get("kind") != kind or state.get("plan") != str(path)
                or state.get("plan_digest") != digest(plan) or not isinstance(state.get("jobs"), dict)):
            raise ValueError("command manifest or execution binding changed")
        inventory = {job["id"]: job for job in jobs}
        if set(state["jobs"]) - set(inventory):
            raise ValueError("execution contains a foreign job")
        unfinished = []
        for job in jobs:
            row = state["jobs"].get(job["id"])
            if row is not None:
                if row.get("command_digest") != digest(job["command"]):
                    raise ValueError("recorded job command changed")
                if row.get("returncode") is None and live_pid(row.get("pid")):
                    raise RuntimeError(f"existing live job {job['id']}")
            if row is None or row.get("returncode") != 0:
                unfinished.append(job)
            else:
                if row.get("output_hashes") != _output_hashes(job):
                    raise ValueError("successful job output bytes changed")
                if kind in ("training", "evaluation"):
                    outcome = terminal(job)
                    if outcome is None or row.get("terminal_status") != outcome["status"]:
                        raise ValueError("successful job terminal changed")
        if existed and not state.get("complete"):
            index = 1
            while (folder / f"{kind}-execution-attempt-{index:03d}.json").exists():
                index += 1
            archive = folder / f"{kind}-execution-attempt-{index:03d}.json"
            archive.write_bytes(state_path.read_bytes())
            state.setdefault("resume_history", []).append({"archive": str(archive), "sha256": file_sha256(archive), "time": time.time()})
        for key in ("finished", "coordinator_error", "blocked_jobs"):
            state.pop(key, None)
        state["complete"] = False
        state["coordinator_sha256"] = file_sha256(__file__)
        states[kind] = (state, plan, jobs, state_path)
        pending[kind] = unfinished
        save(state_path, state)
    protected = {Path(path).resolve() for path in plans.values()} | {Path(limits_path).resolve()}
    for _, plan, _, state_path in states.values():
        protected.add(state_path)
        protected.update(Path(name).resolve() for name in plan["file_hashes"])
    output_owners = {}
    for kind, (_, _, jobs, _) in states.items():
        for job in jobs:
            for name in job["outputs"]:
                path = Path(name).resolve()
                if path in protected or path in output_owners:
                    raise ValueError("job output collides with a pinned file or another job")
                output_owners[path] = (kind, job["id"])
            if Path(job["log"]).resolve() in protected:
                raise ValueError("job log collides with a pinned file")
    active, failed, last_limits, normal_finish = {}, False, None, False
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    try:
        while any(pending.values()) or active:
            declared_limits = _json(limits_path)
            limits = _limits(declared_limits)
            if declared_limits != last_limits:
                for state, _, _, state_path in states.values():
                    state.setdefault("limits_history", []).append({"time": time.time(), "declared": declared_limits, "effective": limits})
                    save(state_path, state)
                last_limits = deepcopy(declared_limits)
            for kind in KINDS:
                state, plan, _, state_path = states[kind]
                for job in tuple(pending[kind]):
                    training_count = sum(key[0] == "training" for key in active)
                    evaluation_count = len(active) - training_count
                    if (failed or len(active) >= limits["max_workers"]
                            or (kind == "training" and training_count >= limits["training"])
                            or (kind != "training" and evaluation_count >= limits["evaluation"])):
                        break
                    if available_memory() < limits["min_available_bytes"]:
                        break
                    if kind == "evaluation" and not ready(job):
                        continue
                    _verify_running_binding(state, plan)
                    log_path = Path(job["log"])
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log = log_path.open("ab")
                    try:
                        process = subprocess.Popen(job["command"], cwd=plan["cwd"], env=env,
                            stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    except BaseException:
                        log.close()
                        raise
                    previous = state["jobs"].get(job["id"])
                    attempts = ([] if previous is None else previous.get("attempts", []) +
                                [deepcopy({name: value for name, value in previous.items() if name != "attempts"})])
                    state["jobs"][job["id"]] = {"started": time.time(), "pid": process.pid,
                        "returncode": None, "command_digest": digest(job["command"]), "log": str(log_path), "attempts": attempts}
                    active[(kind, job["id"])] = (job, process, log)
                    pending[kind].remove(job)
                    print(json.dumps({"started": job["id"], "kind": kind, "pid": process.pid}), flush=True)
                    save(state_path, state)
            for key, (job, process, log) in tuple(active.items()):
                raw_code = process.poll()
                if raw_code is None:
                    continue
                log.close()
                code, result = raw_code, {}
                missing = [name for name in job["outputs"] if not Path(name).is_file()]
                if raw_code == 0:
                    try:
                        result["output_hashes"] = _output_hashes(job)
                        if key[0] in ("training", "evaluation"):
                            outcome = terminal(job)
                            if outcome is None:
                                raise ValueError("job exited zero without a scientific terminal")
                            result["terminal_status"] = outcome["status"]
                            result["terminal_last_tick"] = outcome["last_tick"]
                            result["endpoint_comparable"] = outcome["endpoint_comparable"]
                    except Exception as error:
                        code = -1
                        result["output_validation_error"] = f"{type(error).__name__}: {error}"
                kind, identifier = key
                state, _, _, state_path = states[kind]
                state["jobs"][identifier].update(returncode=code, process_returncode=raw_code,
                    finished=time.time(), missing_outputs=missing, **result)
                failed |= code != 0
                del active[key]
                print(json.dumps({"finished": identifier, "kind": kind, "returncode": code,
                                  "terminal_status": result.get("terminal_status")}), flush=True)
                save(state_path, state)
            if failed and not active:
                break
            if not active and pending["evaluation"] and not pending["training"] and not pending["baseline"]:
                blocked = [job["id"] for job in pending["evaluation"] if not ready(job)]
                if blocked:
                    states["evaluation"][0]["blocked_jobs"] = blocked
                    states["evaluation"][0]["coordinator_error"] = "No remaining producer for terminal dependencies"
                    failed = True
                    break
            if any(pending.values()) or active:
                time.sleep(poll_interval)
        for state, plan, _, _ in states.values():
            _verify_running_binding(state, plan)
        normal_finish = not failed
    except BaseException as error:
        for state, _, _, _ in states.values():
            state["coordinator_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        for (kind, identifier), (_, process, log) in active.items():
            terminate_tree(process)
            log.close()
            states[kind][0]["jobs"][identifier].update(returncode=process.returncode or -2,
                process_returncode=process.returncode, finished=time.time(), interrupted=True)
        for kind, (state, _, jobs, state_path) in states.items():
            successful = all(state["jobs"].get(job["id"], {}).get("returncode") == 0 for job in jobs)
            state.update(finished=time.time(), complete=normal_finish and not pending[kind] and successful)
            state["terminal_counts"] = {status: sum(row.get("terminal_status") == status for row in state["jobs"].values())
                                        for status in ("success", "numerical_failure")}
            save(state_path, state)
    return 0 if all(value[0]["complete"] for value in states.values()) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training-plan", "evaluation-plan", "baseline-plan", "status-root", "limits"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    return execute({"training": args.training_plan, "evaluation": args.evaluation_plan,
                    "baseline": args.baseline_plan}, args.status_root, args.limits)


if __name__ == "__main__":
    raise SystemExit(main())
