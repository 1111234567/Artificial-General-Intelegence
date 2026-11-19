"""Separate, memory-gated scheduling of post-training sensitivity analyses.

This module never runs a learner, reads behavioral scores, changes a protocol,
or writes to a training directory.  Every declared branch receives an audit;
no-S methods and empty histories retain their explicit, non-calibrated scopes.
The coordinator is stdlib-only so it does not reserve a Torch worker's RAM.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from experiments.undirected.queue import (
    atomic_json, available_memory, digest, file_sha256, live_pid, terminate_tree, verify_files,
)


ROOT = Path(__file__).resolve().parents[2]
FORMAT = "undirected-independent-analysis-plan-v1"
STATE_FORMAT = "undirected-independent-analysis-execution-v1"
AUDIT_FORMAT = "undirected-sensitivity-audit-v1"
RUNTIME = ("undirected/run.py", "undirected/model.py", "undirected/credit.py", "undirected/plasticity.py",
           "continuity/run.py", "continuity/model.py", "continuity/world.py",
           "lifelong/world.py", "lifelong/schedule.py", "lifelong/replay.py")
ANALYSIS = ("analysis_batch.py", "test_analysis_batch.py", "sensitivity_audit.py", "test_sensitivity_audit.py")
WORKER_ENVIRONMENT = dict(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                          OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _integer(value, name, lower=0):
    if type(value) is not int or value < lower:
        raise ValueError(f"{name} must be an integer >= {lower}")
    return value


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def prepare(protocol, output, *, mechanical=False):
    """Only prepare analysis commands; never start them or prepare training."""
    protocol, output = Path(protocol).resolve(), Path(output).resolve()
    value = _json(protocol)
    if type(mechanical) is not bool or value.get("format") != "undirected-protocol-v1" or value.get("mechanical_test") is not mechanical:
        raise ValueError("explicit matching formal/mechanical analysis mode required")
    branches = value.get("branches")
    _integer(value.get("target_ticks"), "declared analysis target", 1)
    if (not isinstance(branches, list) or not branches or len({row["id"] for row in branches}) != len(branches)
            or (not mechanical and len(branches) != 42)):
        raise ValueError("incomplete declared analysis branch inventory")
    if any(not isinstance(row["id"], str) or re.fullmatch(r"[A-Za-z0-9_-]+", row["id"]) is None for row in branches):
        raise ValueError("unsafe analysis branch identifier")
    verify_files(value["sources"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("analysis preparation requires a new output folder")
    for item in branches:
        if output.is_relative_to(Path(item["output"]).resolve()):
            raise ValueError("analysis output must stay outside all training directories")
    sources = {**value["sources"], str(protocol): file_sha256(protocol)}
    sources.update({str((Path(__file__).parent / name).resolve()): file_sha256(Path(__file__).parent / name) for name in ANALYSIS})
    sha = file_sha256(protocol)
    python = value.get("python_executable", str((ROOT / ".venv/Scripts/python.exe").resolve()))
    if not Path(python).is_file():
        raise ValueError("declared project Python is unavailable")
    jobs = []
    for item in branches:
        terminal = str((Path(item["output"]) / "terminal.json").resolve())
        destination = str(output / "audits" / f"{item['id']}.json")
        jobs.append(dict(id=item["id"], branch=item, terminal=terminal, output=destination,
            log=str(output / "logs" / f"{item['id']}.log"),
            command=[python, "-m", "experiments.undirected.sensitivity_audit", "--terminal", terminal, "--output", destination]))
    plan = dict(format=FORMAT, analysis_only=True, protocol=str(protocol), protocol_sha256=sha,
        target_ticks=value["target_ticks"],
        mechanical_test=mechanical, cwd=str(ROOT), file_hashes=sources, runtime_source_paths=[str((ROOT / "experiments" / name).resolve()) for name in RUNTIME],
        worker_environment=WORKER_ENVIRONMENT,
        preparation_environment={name: os.environ.get(name) for name in WORKER_ENVIRONMENT},
        jobs=jobs, output_root=str(output), limits=str(output / "analysis-limits-v1.json"),
        scope="Independent JVP/primal analysis only. Numerical failure endpoints stay unavailable; no-S/empty histories are not S proofs.")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "analysis-plan-v1.json"
    atomic_json(path, plan)
    atomic_json(output / "analysis-limits-v1.json", dict(max_workers=1, min_available_bytes=4 * 1024**3))
    return path


def _load_plan(path):
    path = Path(path).resolve()
    plan = _json(path)
    if plan.get("format") != FORMAT or plan.get("analysis_only") is not True:
        raise ValueError("foreign analysis scheduler plan")
    verify_files(plan["file_hashes"])
    if plan.get("protocol_sha256") != file_sha256(plan["protocol"]):
        raise ValueError("training protocol changed")
    protocol = _json(plan["protocol"])
    if (protocol.get("branches") != [job["branch"] for job in plan["jobs"]]
            or protocol.get("mechanical_test") is not plan.get("mechanical_test") or plan.get("target_ticks") != protocol.get("target_ticks")
            or plan.get("cwd") != str(ROOT)
            or plan.get("worker_environment") != WORKER_ENVIRONMENT
            or plan.get("runtime_source_paths") != [str((ROOT / "experiments" / name).resolve()) for name in RUNTIME]):
        raise ValueError("analysis inventory differs from the unchanged training protocol")
    expected_sources = {**protocol["sources"], str(Path(plan["protocol"]).resolve()): file_sha256(plan["protocol"])}
    expected_sources.update({str((Path(__file__).parent / name).resolve()): file_sha256(Path(__file__).parent / name) for name in ANALYSIS})
    if plan["file_hashes"] != expected_sources:
        raise ValueError("analysis source inventory differs")
    root = Path(plan["output_root"]).resolve()
    if plan.get("limits") != str(root / "analysis-limits-v1.json"):
        raise ValueError("foreign analysis limits path")
    for job in plan["jobs"]:
        expected = str(root / "audits" / f"{job['id']}.json")
        terminal = str((Path(job["branch"]["output"]) / "terminal.json").resolve())
        prefix = [protocol.get("python_executable", str((ROOT / ".venv/Scripts/python.exe").resolve())),
                  "-m", "experiments.undirected.sensitivity_audit"]
        if (job["id"] != job["branch"]["id"] or job["terminal"] != terminal or job["output"] != expected
                or re.fullmatch(r"[A-Za-z0-9_-]+", job["id"]) is None
                or job.get("log") != str(root / "logs" / f"{job['id']}.log")
                or job["command"] != prefix + ["--terminal", terminal, "--output", expected]
                or Path(expected).is_relative_to(Path(job["branch"]["output"]).resolve())):
            raise ValueError("foreign analysis command/output")
    protected = {Path(name).resolve() for name in plan["file_hashes"]}
    mutable = [root / "analysis-execution-v1.json", root / "sensitivity-results-v1.json"]
    mutable.extend(Path(job[key]) for job in plan["jobs"] for key in ("output", "log"))
    if any(path.resolve() in protected for path in mutable):
        raise ValueError("analysis output collides with a frozen input/source")
    return plan


def _terminal(job, plan, *, verify_bytes=True):
    path = Path(job["terminal"])
    if not path.exists():
        return None
    value = _json(path)
    success = value.get("status") == "success"
    target, last = value.get("target_ticks"), value.get("last_tick")
    if (value.get("format") != "undirected-branch-terminal-v1" or value.get("branch") != job["branch"]
            or value.get("protocol_sha256") != plan["protocol_sha256"]
            or target != plan["target_ticks"]
            or type(last) is not int or type(target) is not int or not 0 <= last <= target
            or value.get("status") not in ("success", "numerical_failure")
            or value.get("completed") is not success or value.get("endpoint_comparable") is not success
            or (success and last != target) or (not success and last >= target)):
        raise ValueError("foreign or incomplete analysis terminal")
    checkpoint = Path(job["branch"]["output"]) / ("checkpoint_latest.pt" if success else "checkpoint_last_complete.pt")
    if (value.get("checkpoint_path") != str(checkpoint.resolve()) or not _hash(value.get("checkpoint_sha256"))
            or (verify_bytes and value["checkpoint_sha256"] != file_sha256(checkpoint))):
        raise ValueError("terminal reference checkpoint bytes changed")
    summary = value.get("ledger", {})
    folder = Path(job["branch"]["output"]) / "live_records"
    for key, name in (("binding", "binding.json"), ("manifest", "manifest.jsonl")):
        if (summary.get(key + "_path") != str((folder / name).resolve()) or not _hash(summary.get(key + "_sha256"))
                or (verify_bytes and summary[key + "_sha256"] != file_sha256(folder / name))):
            raise ValueError("terminal live-ledger fingerprints differ")
    _integer(summary.get("records"), "ledger record count")
    if summary.get("last_tick") != last:
        raise ValueError("terminal/ledger progress differs")
    return value


def validate_result(job, plan, terminal):
    """Scheduler scope/input validation; numerical comparisons belong to audit/report."""
    path = Path(job["output"])
    value = _json(path)
    records = terminal["ledger"]["records"]
    method = job["branch"]["credit"]
    if (value.get("format") != AUDIT_FORMAT or value.get("terminal_path") != job["terminal"]
            or value.get("terminal_sha256") != file_sha256(job["terminal"])
            or value.get("terminal_status") != terminal["status"] or value.get("endpoint_comparable") is not terminal["endpoint_comparable"]
            or value.get("method") != method or value.get("record_count") != records
            or value.get("manifest_sha256") != terminal["ledger"]["manifest_sha256"]
            or value.get("binding_hash") != terminal["ledger"]["binding_hash"]
            or (value.get("directions"), value.get("seed"), value.get("atol"), value.get("rtol")) != (2, 731, 1e-9, 1e-8)):
        raise ValueError("foreign independent audit artifact")
    if not records:
        expected = ("not_applicable_no_successful_live_blocks", False, "no_successful_live_history")
    elif method.startswith("rtrl"):
        expected = ("passed", True, "primal_history_and_projected_saved_S")
    else:
        expected = ("passed", False, "primal_history_only_no_saved_S_comparison")
    if (value.get("status"), value.get("sensitivity_calibrated"), value.get("validation_scope")) != expected:
        raise ValueError("independent audit claims the wrong sensitivity scope")
    fingerprints = value.get("input_fingerprints")
    if not isinstance(fingerprints, dict):
        raise ValueError("audit input fingerprints missing")
    required = {job["terminal"]: file_sha256(job["terminal"]), terminal["checkpoint_path"]: terminal["checkpoint_sha256"],
                terminal["ledger"]["binding_path"]: terminal["ledger"]["binding_sha256"],
                terminal["ledger"]["manifest_path"]: terminal["ledger"]["manifest_sha256"]}
    if records:
        required.update({entry["path"]: entry["sha256"] for entry in terminal["checkpoints"].values()})
        for line in Path(terminal["ledger"]["manifest_path"]).read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            required[str((Path(terminal["ledger"]["manifest_path"]).parent / entry["file"]).resolve())] = entry["sha256"]
        expected_sources = {str((ROOT / "experiments/undirected" / name).resolve()) for name in ("sensitivity_audit.py", "ledger.py", "model.py")}
        if set(value.get("analysis_sources", {})) != expected_sources:
            raise ValueError("independent audit analysis-source inventory differs")
        required.update({source: plan["file_hashes"][source] for source in plan["runtime_source_paths"]})
        required.update(value["analysis_sources"])
        if len(value.get("records", [])) != records or value.get("work") != dict(
                jvp_calls=2 * records, model_step_calls=2 * terminal["last_tick"], credit_engine_jacobian_calls=0, optimizer_steps=0, world_steps=0):
            raise ValueError("independent audit history/work inventory differs")
    if any(fingerprints.get(source) != expected for source, expected in required.items()):
        raise ValueError("independent audit omitted a bound input/source")
    verify_files(fingerprints)
    return value


def execute(plan_path, *, watch=False, poll_interval=2.):
    """Collect finished branches once, or explicitly watch for future terminals.

    Return 0 when all analyses are collected, 2 while terminals/resources are
    pending, and 1 after an analysis process error.  No failure is converted to
    a successful scientific result.
    """
    if type(watch) is not bool or isinstance(poll_interval, bool) or not isinstance(poll_interval, (int, float)) or not math.isfinite(poll_interval) or poll_interval <= 0:
        raise ValueError("explicit watch mode and positive poll interval required")
    plan_path = Path(plan_path).resolve()
    plan = _load_plan(plan_path)
    root = Path(plan["output_root"])
    state_path = root / "analysis-execution-v1.json"
    state = (_json(state_path) if state_path.exists() else dict(format=STATE_FORMAT, plan=str(plan_path), plan_hash=digest(plan), jobs={}))
    if state.get("format") != STATE_FORMAT or state.get("plan_hash") != digest(plan) or set(state.get("jobs", {})) - {job["id"] for job in plan["jobs"]}:
        raise ValueError("foreign analysis execution state")
    if "coordinator_error" in state:
        state.setdefault("previous_errors", []).append(state.pop("coordinator_error"))
    active, failed, last_limits, verified_complete, terminal_cache = {}, False, None, set(), {}
    env = os.environ.copy()
    env.update(plan["worker_environment"])
    try:
        while True:
            if digest(_json(plan_path)) != state["plan_hash"]:
                raise ValueError("analysis plan changed during execution")
            verify_files(plan["file_hashes"])
            declared = _json(plan["limits"])
            maximum = _integer(declared.get("max_workers"), "analysis max_workers", 1)
            reserve = _integer(declared.get("min_available_bytes", 0), "analysis memory reserve")
            if declared != last_limits:
                state.setdefault("limits_history", []).append(dict(time=time.time(), limits=declared))
                last_limits = deepcopy(declared)
            for job in plan["jobs"]:
                identifier = job["id"]
                if identifier in active or identifier in verified_complete:
                    continue
                previous = state["jobs"].get(identifier, {})
                if previous.get("status") == "running" and live_pid(previous.get("pid")):
                    raise RuntimeError("another coordinator still owns a live analysis worker")
                terminal_path = Path(job["terminal"])
                if not terminal_path.exists():
                    if identifier in terminal_cache or previous.get("terminal_sha256") is not None:
                        raise ValueError("an observed immutable terminal disappeared")
                    state["jobs"][identifier] = dict(status="waiting_terminal")
                    continue
                terminal_sha = file_sha256(terminal_path)
                prior_sha = terminal_cache.get(identifier, {}).get("sha256", previous.get("terminal_sha256"))
                if prior_sha is not None and prior_sha != terminal_sha:
                    raise ValueError("an observed immutable terminal changed")
                if identifier not in terminal_cache:
                    # Waiting jobs read only the small final marker.  Hash the
                    # large CP/manifest only at dispatch/result validation.
                    terminal_cache[identifier] = dict(sha256=terminal_sha, value=_terminal(job, plan, verify_bytes=False))
                terminal = terminal_cache[identifier]["value"]
                if Path(job["output"]).exists():
                    terminal = _terminal(job, plan)
                    value = validate_result(job, plan, terminal)
                    sha = file_sha256(job["output"])
                    if previous.get("status") == "complete" and previous.get("output_sha256") != sha:
                        raise ValueError("completed audit output bytes changed")
                    state["jobs"][identifier] = dict(status="complete", output=job["output"], output_sha256=sha,
                        terminal_sha256=file_sha256(job["terminal"]), terminal_status=terminal["status"],
                        endpoint_comparable=terminal["endpoint_comparable"], sensitivity_calibrated=value["sensitivity_calibrated"],
                        validation_scope=value["validation_scope"])
                    verified_complete.add(identifier)
                    continue
                if failed or len(active) >= maximum or available_memory() < reserve:
                    state["jobs"][identifier] = dict(status="waiting_resources", terminal_sha256=terminal_sha)
                    continue
                terminal = _terminal(job, plan)
                Path(job["output"]).parent.mkdir(parents=True, exist_ok=True)
                log_path = Path(job["log"])
                log_path.parent.mkdir(parents=True, exist_ok=True)
                stream = log_path.open("ab")
                try:
                    process = subprocess.Popen(job["command"], cwd=plan["cwd"], env=env, stdout=stream,
                                               stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                except BaseException:
                    stream.close()
                    raise
                active[identifier] = (job, process, stream)
                state["jobs"][identifier] = dict(status="running", pid=process.pid, started=time.time(), terminal_sha256=terminal_sha)
            for identifier, (job, process, stream) in tuple(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                state["jobs"][identifier].update(status="produced_unverified" if code == 0 else "analysis_execution_failure", returncode=code)
                failed |= code != 0
                del active[identifier]
            mapping = {identifier: row["output"] for identifier, row in state["jobs"].items() if row.get("status") == "complete"}
            collected = len(mapping) == len(plan["jobs"])
            state["complete"] = False
            state["stage"] = "final_input_validation" if collected else "running_or_waiting"
            state["validation_role"] = "Scheduler input/scope binding; audit/report own the numerical and scientific provenance checks."
            atomic_json(state_path, state)
            atomic_json(root / "sensitivity-results-v1.json", mapping)
            if collected:
                # Avoid rereading completed multi-GB histories every watch poll.
                # One final complete pass revalidates the immutable terminals,
                # ledgers, CPs and audit inputs before publishing complete=True.
                for job in plan["jobs"]:
                    row = state["jobs"][job["id"]]
                    if file_sha256(job["output"]) != row["output_sha256"] or file_sha256(job["terminal"]) != row["terminal_sha256"]:
                        raise ValueError("collected artifact or terminal changed before batch completion")
                    terminal = _terminal(job, plan)
                    validate_result(job, plan, terminal)
                verify_files(plan["file_hashes"])
                if digest(_json(plan_path)) != state["plan_hash"]:
                    raise ValueError("analysis plan changed during final validation")
                state.update(complete=True, stage="complete", finished=time.time())
                atomic_json(state_path, state)
                return 0
            if failed and not active:
                return 1
            if not active and not watch:
                # One more cheap pass validates outputs just produced this poll.
                if any(row.get("status") == "produced_unverified" for row in state["jobs"].values()):
                    continue
                return 2
            time.sleep(poll_interval)
    except BaseException as error:
        state["complete"] = False
        state["coordinator_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        for identifier, (_, process, stream) in active.items():
            terminate_tree(process)
            stream.close()
            state["jobs"][identifier].update(status="interrupted", returncode=process.returncode)
        atomic_json(state_path, state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--protocol", required=True)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--mechanical", action="store_true")
    run_parser = sub.add_parser("execute")
    run_parser.add_argument("--plan", required=True)
    run_parser.add_argument("--watch", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(prepare(args.protocol, args.output, mechanical=args.mechanical), flush=True)
        return 0
    return execute(args.plan, watch=args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
