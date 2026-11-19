"""Supervise an existing frozen queue and finish its separate analysis artifacts.

This Windows entry point never prepares or starts training. Run it in a detached
hidden process for durable completion; restart the same command after an ops
failure. Partial report/figure attempts are retained in separate directories.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from experiments.undirected import analysis_batch, queue


ROOT = Path(__file__).resolve().parents[2]
FORMAT = "undirected-finish-execution-v1"
ENVIRONMENT = analysis_batch.WORKER_ENVIRONMENT


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def sole_writer(path):
    """A kernel lock survives stale lock files and releases on process exit."""
    import msvcrt
    with Path(path).open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError("another finish coordinator holds this output lock") from error
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


class Producer:
    """Keep the actual Windows process handle, so PID reuse cannot fool us."""
    def __init__(self, pid):
        from ctypes import wintypes
        kernel = self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
        kernel.QueryFullProcessImageNameW.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD))
        kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        self.handle = kernel.OpenProcess(0x100000 | 0x1000, False, pid)
        if not self.handle:
            raise RuntimeError(f"producer PID {pid} is unavailable while queue is incomplete")
        try:
            times = [wintypes.FILETIME() for _ in range(4)]
            if not kernel.GetProcessTimes(self.handle, *[ctypes.byref(value) for value in times]):
                raise ctypes.WinError(ctypes.get_last_error())
            name, size = ctypes.create_unicode_buffer(32768), wintypes.DWORD(32768)
            if not kernel.QueryFullProcessImageNameW(self.handle, 0, name, ctypes.byref(size)):
                raise ctypes.WinError(ctypes.get_last_error())
            command = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                f"(Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}').CommandLine | ConvertTo-Json -Compress"],
                capture_output=True, text=True, check=True, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
            command = json.loads(command.stdout)
            if (not isinstance(command, str) or re.search(r"(?:^|\s)-m\s+experiments\.undirected\.queue(?:\s|$)", command) is None
                    or Path(name.value).resolve() != Path(sys.executable).resolve() or not self.alive()):
                raise RuntimeError("producer identity is not the live project Python frozen queue")
            shell = ctypes.WinDLL("shell32", use_last_error=True)
            shell.CommandLineToArgvW.argtypes = (wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int))
            shell.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
            count = ctypes.c_int()
            values = shell.CommandLineToArgvW(command, ctypes.byref(count))
            if not values:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                arguments = [values[index] for index in range(count.value)]
            finally:
                kernel.LocalFree.argtypes = (ctypes.c_void_p,)
                kernel.LocalFree(ctypes.cast(values, ctypes.c_void_p))
            self.identity = dict(pid=pid, creation_filetime=times[0].dwLowDateTime | times[0].dwHighDateTime << 32,
                                 executable=name.value, command=command, arguments=arguments)
        except BaseException:
            self.close()
            raise

    def alive(self):
        result = self.kernel.WaitForSingleObject(self.handle, 0)
        if result not in (0, 258):
            raise RuntimeError("cannot inspect producer process handle")
        return result == 258

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def queue_state(folder, plans, *, verify_outputs=False):
    """Read execution metadata, never behavioral records; validate bytes at end."""
    result = {}
    for kind, (path, plan, jobs) in plans.items():
        state_path = folder / f"{kind}-execution.json"
        state = read(state_path)
        if (state.get("format") != queue.QUEUE_FORMAT or state.get("kind") != kind
                or state.get("plan") != str(path) or state.get("plan_digest") != queue.digest(plan)
                or not isinstance(state.get("jobs"), dict) or set(state["jobs"]) - {job["id"] for job in jobs}):
            raise ValueError(f"{kind} queue execution binding changed")
        errors = [key for key, row in state["jobs"].items() if row.get("returncode") not in (None, 0)]
        if state.get("coordinator_error") or state.get("blocked_jobs") or errors:
            raise RuntimeError(f"{kind} producer execution failed: {state.get('coordinator_error') or errors or state['blocked_jobs']}")
        complete = state.get("complete") is True
        if complete and (set(state["jobs"]) != {job["id"] for job in jobs}
                         or any(row.get("returncode") != 0 for row in state["jobs"].values())):
            raise ValueError(f"{kind} queue falsely claims completion")
        counts = {status: sum(row.get("terminal_status") == status for row in state["jobs"].values())
                  for status in ("success", "numerical_failure")}
        if complete and kind != "baseline" and (sum(counts.values()) != len(jobs) or state.get("terminal_counts") != counts):
            raise ValueError(f"{kind} scientific terminal accounting differs")
        if complete and verify_outputs:
            for job in jobs:
                row = state["jobs"][job["id"]]
                if (row.get("command_digest") != queue.digest(job["command"])
                        or row.get("output_hashes") != queue._output_hashes(job)):
                    raise ValueError(f"completed {kind} output/command binding differs: {job['id']}")
                if kind != "baseline" and row.get("terminal_status") != queue.terminal(job)["status"]:
                    raise ValueError("queue terminal outcome differs")
        result[kind] = dict(complete=complete, declared_jobs=len(jobs), recorded_jobs=len(state["jobs"]),
                            terminal_counts=counts, execution_path=str(state_path), execution_sha256=queue.file_sha256(state_path))
    return result


def require_producer(snapshot, producer, reread):
    if all(value["complete"] for value in snapshot.values()):
        return True
    if producer is not None and producer.alive():
        return False
    # A producer publishes its final states immediately before its process exit.
    if not all(value["complete"] for value in reread().values()):
        raise RuntimeError("frozen queue producer died before verified 42/0/42 completion; analysis watch stopped")
    return True


def supervise(args):
    if os.name != "nt":
        raise RuntimeError("finish is a Windows-only operations entry point")
    protocol, analysis_path, folder = [Path(value).resolve() for value in (args.protocol, args.analysis_plan, args.output)]
    protocol_value = read(protocol)
    plan = analysis_batch._load_plan(analysis_path)
    if plan["protocol"] != str(protocol) or protocol_value.get("mechanical_test") is not False:
        raise ValueError("finish requires the exact formal protocol bound by the prepared analysis plan")
    branch_ids = {branch["id"] for branch in protocol_value["branches"]}
    if len(branch_ids) != 42:
        raise ValueError("formal finish requires all 42 declared branches")
    queue_root = Path(args.queue_root).resolve()
    protected_roots = [Path(branch["output"]).resolve() for branch in protocol_value["branches"]]
    protected_roots += [Path(plan["output_root"]).resolve()]
    if any(folder == root or folder.is_relative_to(root) or root.is_relative_to(folder) for root in protected_roots):
        raise ValueError("finish output must be separate from training and analysis outputs")
    plans = {}
    for kind in queue.KINDS:
        path = Path(read(queue_root / f"{kind}-execution.json")["plan"]).resolve()
        value, jobs = queue._load_plan(kind, path)
        if value["protocol"] != str(protocol) or (kind != "baseline" and {job["id"] for job in jobs} != branch_ids) or (kind == "baseline" and jobs):
            raise ValueError("producer command plans differ from the formal 42/0/42 inventory")
        plans[kind] = (path, value, jobs)
    sources = {str(path): queue.file_sha256(path) for path in
               (protocol, analysis_path, Path(__file__).resolve(), Path(__file__).with_name("report.py"),
                Path(__file__).with_name("plot.py"), Path(__file__).with_name("results.py"))}
    binding = dict(protocol=str(protocol), protocol_sha256=sources[str(protocol)], analysis_plan=str(analysis_path),
                   analysis_plan_sha256=sources[str(analysis_path)], queue_root=str(queue_root), sources=sources,
                   report_workers=args.report_workers)
    folder.mkdir(parents=True, exist_ok=True)
    with sole_writer(folder / "finish.lock"):
        status_path = folder / "finish-status.json"
        if not status_path.exists() and any(path.name != "finish.lock" for path in folder.iterdir()):
            raise FileExistsError("refusing an unbound nonempty finish output folder")
        state = read(status_path) if status_path.exists() else dict(format=FORMAT, binding=binding, stages={})
        if state.get("format") != FORMAT or state.get("binding") != binding:
            raise ValueError("finish output belongs to different inputs/sources/options; use a new output folder")
        previous = state.get("active_child")
        if previous and queue.live_pid(previous.get("pid")):
            raise RuntimeError("previous finish child remains live; refusing duplicate analysis/report/plot")
        state.update(complete=False, coordinator_pid=os.getpid(), started=time.time())
        state.pop("error", None)
        producer, child, stream = None, None, None

        def save():
            queue.atomic_json(status_path, state)

        def check_producer():
            snapshot = queue_state(queue_root, plans)
            state["queue"] = snapshot
            return require_producer(snapshot, producer, lambda: queue_state(queue_root, plans))

        def run_stage(name, command):
            nonlocal child, stream
            queue.verify_files(sources)
            stage = state["stages"].setdefault(name, {})
            attempt = stage.get("attempt", 0) + 1
            log = folder / f"{name}-{attempt:03d}.log"
            if log.exists():
                raise FileExistsError(f"refusing an unowned stage log collision: {log}")
            stage.update(attempt=attempt, status="running", command=command, log=str(log), started=time.time())
            stream = log.open("ab")
            child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, **ENVIRONMENT}, stdout=stream,
                                     stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            state["active_child"] = dict(stage=name, pid=child.pid)
            state["stage"] = name
            save()
            while child.poll() is None:
                check_producer()
                save()
                time.sleep(args.poll_seconds)
            code = child.returncode
            stream.close()
            child, stream = None, None
            state.pop("active_child", None)
            stage.update(status="command_succeeded" if code == 0 else "failed", returncode=code, finished=time.time())
            save()
            if code:
                raise RuntimeError(f"{name} exited {code}; see {log}")

        try:
            initial = queue_state(queue_root, plans)
            if not all(value["complete"] for value in initial.values()):
                producer = Producer(args.queue_pid)
                arguments = producer.identity["arguments"]
                expected = {"--status-root": queue_root, **{"--" + kind + "-plan": value[0] for kind, value in plans.items()}}
                for flag, expected_path in expected.items():
                    if arguments.count(flag) != 1:
                        raise RuntimeError("producer command omits or repeats a bound queue option")
                    raw = Path(arguments[arguments.index(flag) + 1])
                    if (raw if raw.is_absolute() else ROOT / raw).resolve() != expected_path:
                        raise RuntimeError("producer command targets a different frozen queue")
                if state.get("producer") not in (None, producer.identity):
                    raise RuntimeError("producer identity changed; use a new finish output folder")
                state["producer"] = producer.identity
            save()
            python = str(Path(sys.executable).resolve())
            # analysis_batch owns resumable audit validation and its resource limits.
            run_stage("analysis", [python, "-m", "experiments.undirected.analysis_batch", "execute", "--plan", str(analysis_path), "--watch"])
            audit_root = Path(plan["output_root"])
            audit_state = read(audit_root / "analysis-execution-v1.json")
            sensitivity = audit_root / "sensitivity-results-v1.json"
            if (audit_state.get("complete") is not True or audit_state.get("plan_hash") != queue.digest(plan)
                    or set(audit_state.get("jobs", {})) != branch_ids or set(read(sensitivity)) != branch_ids
                    or any(row.get("status") != "complete" for row in audit_state["jobs"].values())):
                raise ValueError("analysis child did not validate every declared branch")
            audit_artifacts = {row["output"]: row["output_sha256"] for row in audit_state["jobs"].values()}
            audit_artifacts.update({str(path): queue.file_sha256(path) for path in (sensitivity, audit_root / "analysis-execution-v1.json")})
            queue.verify_files(audit_artifacts)
            state["stages"]["analysis"].update(status="verified", manifest=str(sensitivity), artifacts=audit_artifacts,
                                                 branch_count=len(audit_state["jobs"]))
            state["stage"] = "waiting_frozen_queue_completion"
            while not check_producer():
                save()
                time.sleep(args.poll_seconds)
            state["queue"] = queue_state(queue_root, plans, verify_outputs=True)
            state["stages"]["producer"] = dict(status="verified", counts={kind: value["declared_jobs"] for kind, value in state["queue"].items()})
            save()
            report_stage = state["stages"].get("report", {})
            if report_stage.get("status") == "verified":
                report = Path(report_stage["report"])
                queue.verify_files(report_stage["artifacts"])
            else:
                attempt = report_stage.get("attempt", 0) + 1
                report_dir = folder / f"report-{attempt:03d}"
                if report_dir.exists():
                    raise FileExistsError(f"refusing report attempt collision: {report_dir}")
                report = report_dir / "report.json"
                run_stage("report", [python, "-m", "experiments.undirected.report", "--protocol", str(protocol),
                    "--output", str(report_dir), "--workers", str(args.report_workers), "--sensitivity-manifest", str(sensitivity)])
                header = read(report)
                if (header.get("audit", {}).get("status") != "passed" or header.get("counts", {}).get("accounted_branches") != 42
                        or header.get("independent_sensitivity", {}).get("pending_branch_ids") != []
                        or header.get("protocol_sha256") != sources[str(protocol)]):
                    raise ValueError("report did not account for all branches and required independent checks")
                artifacts = {str(path.resolve()): queue.file_sha256(path) for path in report_dir.iterdir() if path.is_file()}
                state["stages"]["report"].update(status="verified", report=str(report), artifacts=artifacts, counts=header["counts"])
            run_stage("plot_validate", [python, "-m", "experiments.undirected.plot", "validate", "--report", str(report)])
            state["stages"]["plot_validate"]["status"] = "verified"
            figure_stage = state["stages"].get("plot_render", {})
            if figure_stage.get("status") == "verified":
                queue.verify_files(figure_stage["artifacts"])
            else:
                figure_dir = folder / f"figures-{figure_stage.get('attempt', 0) + 1:03d}"
                if figure_dir.exists():
                    raise FileExistsError(f"refusing figure attempt collision: {figure_dir}")
                run_stage("plot_render", [python, "-m", "experiments.undirected.plot", "render", "--report", str(report), "--output", str(figure_dir)])
                manifest = read(figure_dir / "figure-manifest.json")
                if manifest.get("accounting_audit") != "passed" or manifest.get("protocol_sha256") != sources[str(protocol)] or not manifest.get("figures"):
                    raise ValueError("figure manifest lacks completed verified artifacts")
                state["stages"]["plot_render"].update(status="verified", manifest=str(figure_dir / "figure-manifest.json"),
                    figure_count=len(manifest["figures"]), artifacts={str(path.resolve()): queue.file_sha256(path) for path in figure_dir.rglob("*") if path.is_file()})
            result_stage = state["stages"].get("results", {})
            if result_stage.get("status") == "verified":
                queue.verify_files(result_stage["artifacts"])
            else:
                result_dir = folder / f"results-{result_stage.get('attempt', 0) + 1:03d}"
                if result_dir.exists():
                    raise FileExistsError(f"refusing results attempt collision: {result_dir}")
                result_dir.mkdir()
                result_file = result_dir / "results.zh.md"
                run_stage("results", [python, "-m", "experiments.undirected.results", "--report", str(report), "--output", str(result_file)])
                if not result_file.is_file() or result_file.stat().st_size == 0:
                    raise ValueError("human-readable results artifact is missing or empty")
                state["stages"]["results"].update(status="verified", output=str(result_file),
                                                   artifacts={str(result_file): queue.file_sha256(result_file)})
            queue.verify_files(sources)
            for name in ("analysis", "report", "plot_render", "results"):
                queue.verify_files(state["stages"][name]["artifacts"])
            state.update(complete=True, stage="complete", finished=time.time(),
                         completion_scope="All 42/0/42 frozen jobs and independent audits accounted; report and figure artifacts verified. No claim of favorable behavioral results.")
            save()
            print(json.dumps(dict(complete=True, status=str(status_path), report=str(report),
                                  figures=state["stages"]["plot_render"]["manifest"], results=state["stages"]["results"]["output"])), flush=True)
            return 0
        except BaseException as error:
            state.update(complete=False, error=f"{type(error).__name__}: {error}", failed=time.time())
            raise
        finally:
            if child is not None:
                queue.terminate_tree(child)
                state.pop("active_child", None)
            if stream is not None:
                stream.close()
            if producer is not None:
                producer.close()
            save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "analysis-plan", "queue-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--queue-pid", required=True, type=int)
    parser.add_argument("--report-workers", type=int, choices=range(1, 7), default=3)
    parser.add_argument("--poll-seconds", type=float, default=5.)
    args = parser.parse_args(argv)
    if args.queue_pid < 1 or not 0 < args.poll_seconds <= 60:
        parser.error("queue PID must be positive and poll seconds must be in (0,60]")
    return supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
