"""Serve a read-only dashboard for a seed-evolution scale search."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from urllib.parse import urlsplit


DEFAULT_RUN = Path(__file__).resolve().parents[2] / "data/seed-evolution/2026-11-15-scale-bf16-v1"
WEB_FILES = Path(__file__).resolve().with_name("webui")
STATIC = {
    "/": ("index.html", "text/html"),
    "/index.html": ("index.html", "text/html"),
    "/styles.css": ("styles.css", "text/css"),
    "/app.js": ("app.js", "text/javascript"),
}
ARMS = ("shared", "deep", "medium", "wide", "random")
ROUND_FILE = re.compile(r"round_([0-9]{3,})\.json")
CANDIDATE_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
RATE_MINUTES = 15
HISTORY_MINUTES = 60
SNAPSHOT_SECONDS = 4
GIB = 1024 ** 3


if os.name == "nt":
    import ctypes
    from ctypes import wintypes
    import msvcrt

    _kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    _filetime_ptr = ctypes.POINTER(wintypes.FILETIME)
    _kernel.CreateFileW.argtypes = (
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    )
    _kernel.CreateFileW.restype = wintypes.HANDLE
    _kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _kernel.OpenProcess.restype = wintypes.HANDLE
    _kernel.GetProcessTimes.argtypes = (wintypes.HANDLE, *(_filetime_ptr,) * 4)
    _kernel.GetSystemTimes.argtypes = (_filetime_ptr,) * 3
    _kernel.CloseHandle.argtypes = (wintypes.HANDLE,)

    class _MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", wintypes.DWORD), ("load", wintypes.DWORD),
            ("total", ctypes.c_uint64), ("available", ctypes.c_uint64),
            ("total_page", ctypes.c_uint64), ("available_page", ctypes.c_uint64),
            ("total_virtual", ctypes.c_uint64), ("available_virtual", ctypes.c_uint64),
            ("available_extended_virtual", ctypes.c_uint64),
        ]

    _kernel.GlobalMemoryStatusEx.argtypes = (ctypes.POINTER(_MemoryStatus),)

    def _filetime(value):
        return (value.dwHighDateTime << 32) | value.dwLowDateTime


def open_read(path):
    if os.name != "nt":
        return path.open("rb")
    # Permit the runner to replace progress.json even while a poll is reading it.
    generic_read, share_read_write_delete, open_existing = 0x80000000, 7, 3
    handle = _kernel.CreateFileW(str(path), generic_read, share_read_write_delete,
                                 None, open_existing, 0, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except OSError:
        _kernel.CloseHandle(handle)
        raise
    return os.fdopen(fd, "rb")


def object_or_empty(value):
    return value if isinstance(value, dict) else {}


def count(value):
    return value if type(value) is int and value >= 0 else None


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def timestamp(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except (ValueError, OSError, OverflowError):
        return None


def iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="seconds") if value is not None else None


def file_time(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def read_json(path):
    try:
        with open_read(path) as source:
            data = source.read(4 * 1024 * 1024 + 1)
        return json.loads(data) if len(data) <= 4 * 1024 * 1024 else None
    except (ValueError, OSError):
        return None


def tail(path, lines=30, byte_limit=16384):
    try:
        with open_read(path) as source:
            size = source.seek(0, os.SEEK_END)
            start = max(0, size - byte_limit)
            source.seek(start)
            text = source.read(byte_limit).decode("utf-8", errors="replace")
        rows = text.splitlines()
        if start and len(rows) > 1:
            rows = rows[1:]
        return rows[-lines:]
    except OSError:
        return []


def process_alive(pid, started):
    if not pid:
        return None
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return None
    query_limited_information = 0x1000
    handle = _kernel.OpenProcess(query_limited_information, False, pid)
    if not handle:
        return False if ctypes.get_last_error() == 87 else None
    try:
        created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
        if not _kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                       ctypes.byref(kernel), ctypes.byref(user)):
            return None
        creation_unix = (_filetime(created) - 116444736000000000) / 10_000_000
        return _filetime(exited) == 0 and (started is None or creation_unix <= started + 2)
    finally:
        _kernel.CloseHandle(handle)


def host_usage():
    if os.name == "nt":
        idle, kernel, user = (wintypes.FILETIME() for _ in range(3))
        cpu = None
        if _kernel.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            cpu = _filetime(kernel) + _filetime(user), _filetime(idle)
        memory = _MemoryStatus(length=ctypes.sizeof(_MemoryStatus))
        if _kernel.GlobalMemoryStatusEx(ctypes.byref(memory)):
            return cpu, (memory.total - memory.available) / GIB, memory.total / GIB
        return cpu, None, None
    try:
        with Path("/proc/stat").open() as source:
            stats = [int(field) for field in source.readline().split()[1:9]]
        cpu = sum(stats), stats[3] + stats[4]
    except (OSError, ValueError, IndexError):
        cpu = None
    try:
        with Path("/proc/meminfo").open() as source:
            stats = {key: int(value.split()[0]) for key, value in
                     (line.split(":", 1) for line in source)}
        total = stats["MemTotal"] / (1024 ** 2)
        return cpu, total - stats["MemAvailable"] / (1024 ** 2), total
    except (OSError, ValueError, IndexError, KeyError):
        return cpu, None, None


class Hardware:
    def __init__(self):
        self.cpu = None
        self.gpu = None
        self.gpu_at = -math.inf
        self.smi = shutil.which("nvidia-smi")

    def read(self, device):
        cpu, used, total = host_usage()
        load = None
        if self.cpu is not None and cpu is not None:
            elapsed = cpu[0] - self.cpu[0]
            if elapsed > 0:
                idle = cpu[1] - self.cpu[1]
                load = round(max(0, min(100, 100 * (1 - idle / elapsed))), 1)
        self.cpu = cpu
        now = time.monotonic()
        if now - self.gpu_at >= 12:
            self.gpu = self._read_gpu(device)
            self.gpu_at = now
        return dict(cpu_percent=load, ram_used_gib=used, ram_total_gib=total, gpu=self.gpu)

    def _read_gpu(self, device):
        if not self.smi:
            return None
        fields = ("name", "utilization", "used_mib", "total_mib", "temp_c", "power_w")
        try:
            answer = subprocess.run(
                [self.smi, "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, check=True, timeout=2, text=True, errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        gpus = [row for row in csv.reader(answer.stdout.splitlines(), skipinitialspace=True)
                if len(row) == len(fields)]
        if not gpus:
            return None
        row = next((row for row in gpus if row[0] == device), gpus[0])
        result = {"name": row[0]}
        for key, raw in zip(fields[1:], row[1:]):
            try:
                value = float(raw)
            except ValueError:
                value = None
            result[key] = value if finite(value) else None
        return result


def round_paths(folder):
    try:
        return {int(match[1]): path for path in folder.iterdir()
                if (match := ROUND_FILE.fullmatch(path.name))}
    except OSError:
        return {}


def score_rate(commits, now, started, completed, total, running):
    times = [when for when in commits if now - HISTORY_MINUTES * 60 <= when <= now]
    if not times:
        return dict(per_minute=None, window_minutes=RATE_MINUTES, eta_at=None, series=[])

    # Each chart point is the number of commits in a completed one-minute bin.
    bins = Counter(int(when // 60) for when in times)
    start = max(now - HISTORY_MINUTES * 60, started if started is not None else min(times))
    first_minute, this_minute = math.ceil(start / 60), int(now // 60)
    series = [dict(at=iso(minute * 60 + 60), per_minute=bins[minute])
              for minute in range(first_minute, this_minute)]
    cutoff = max(now - RATE_MINUTES * 60, start)
    elapsed = (now - cutoff) / 60
    recent = [when for when in times if when >= cutoff]
    rate = len(recent) / elapsed if elapsed >= 1 else None
    eta = None
    # Avoid projecting the rest of the search from its first score batch.
    if (running and rate and elapsed >= 5 and recent and max(recent) - min(recent) >= 60
            and completed is not None and total is not None and completed < total):
        seconds_left = (total - completed) / rate * 60
        if seconds_left < 365 * 86400:
            eta = iso(now + seconds_left)
    return dict(per_minute=round(rate, 2) if rate is not None else None,
                window_minutes=round(elapsed, 1), eta_at=eta, series=series)


class Dashboard:
    def __init__(self, run_dir):
        self.root = Path(run_dir).resolve()
        self.hardware = Hardware()
        self._sealed_json = {}
        self._candidate_scans = {}
        self._lock = threading.Lock()
        self._snapshot_at = -math.inf
        self._snapshot = None

    def snapshot(self):
        with self._lock:
            if time.monotonic() - self._snapshot_at >= SNAPSHOT_SECONDS:
                self._snapshot = self._read(time.time())
                self._snapshot_at = time.monotonic()
            return self._snapshot

    def _read_seal(self, path):
        # Plans and round seals are immutable once issued by the experiment.
        if path not in self._sealed_json:
            value = read_json(path)
            if value is not None:
                self._sealed_json[path] = value
            return value
        return self._sealed_json[path]

    def _scan_candidates(self, number, path, sealed):
        issued = object_or_empty(self._read_seal(path))
        parts = issued.get("plans")
        if issued.get("round") != number or not isinstance(parts, list):
            return {}
        # Reuse committed marker metadata and training timestamps. Recheck only
        # unfinished candidates until the round seal makes the scan immutable.
        old, was_sealed = self._candidate_scans.get(number, ({}, False))
        if was_sealed:
            return old
        current = {}
        for part in parts:
            part = object_or_empty(part)
            arm, rep = part.get("arm"), count(part.get("rep"))
            descriptors = part.get("candidates")
            if arm not in ARMS or rep is None or not isinstance(descriptors, list):
                continue
            for descriptor in descriptors:
                candidate = object_or_empty(descriptor).get("id")
                if not isinstance(candidate, str) or not CANDIDATE_ID.fullmatch(candidate):
                    continue
                key = rep, candidate
                _, trained_at, committed_at, fitness = old.get(key, (arm, None, None, None))
                if committed_at is None:
                    folder = self.root / "search" / f"rep_{rep}" / candidate
                    marker = folder / "candidate.json"
                    seen_at = file_time(marker)
                    if seen_at is not None:
                        row = object_or_empty(read_json(marker))
                        if (row.get("id") == candidate and row.get("rep") == rep
                                and row.get("arm") == arm and row.get("generation") == number
                                and finite(row.get("fitness"))):
                            committed_at, fitness = seen_at, row["fitness"]
                    if trained_at is None and committed_at is None and not sealed:
                        trained_at = file_time(folder / "training.json")
                current[key] = arm, trained_at, committed_at, fitness
        self._candidate_scans[number] = current, sealed
        return current

    def _search(self, plan, progress, now):
        arms = {arm: dict(id=arm, width=None, rounds_completed=0, rounds_total=None,
                          completed=0, total=None, best=None) for arm in ARMS}
        schedules = object_or_empty(plan.get("schedules"))
        repeats = count(plan.get("repeats"))
        shared = count(plan.get("shared_count"))
        if shared is not None:
            arms["shared"].update(width=shared, rounds_total=1, total=shared * repeats if repeats is not None else None)
        for arm in ARMS[1:]:
            schedule = object_or_empty(schedules.get(arm))
            generation, width = count(schedule.get("generations")), count(schedule.get("population"))
            arms[arm].update(width=width, rounds_total=generation,
                             total=generation * width * repeats if None not in (generation, width, repeats) else None)

        completed = trained = 0
        activity = []
        commits = []
        seals = {}
        for number, path in round_paths(self.root / "rounds").items():
            rows = self._read_seal(path)
            if not isinstance(rows, list) or not rows:
                continue
            seals[number] = file_time(path)
            counted_reps = {arm: set() for arm in ARMS}
            for row in rows:
                row = object_or_empty(row)
                arm, scored, rep = row.get("arm"), count(row.get("scored")), count(row.get("rep"))
                if arm not in ARMS or scored is None or rep is None or rep in counted_reps[arm]:
                    continue
                counted_reps[arm].add(rep)
                completed += scored
                trained += scored
                arms[arm]["completed"] += scored
                self._update_best(arms[arm], row.get("best_fitness"))
            for arm, reps in counted_reps.items():
                if reps and (repeats is None or len(reps) == repeats):
                    arms[arm]["rounds_completed"] += 1

        plans = round_paths(self.root / "plans")
        last_seal = max(seals, default=None)
        relevant = {number for number in plans if number not in seals or number == last_seal
                    or (seals[number] is not None and now - seals[number] <= HISTORY_MINUTES * 60)}
        self._candidate_scans = {number: cached for number, cached in self._candidate_scans.items()
                                 if number in relevant}
        for number in relevant:
            sealed = number in seals
            for arm, trained_at, committed_at, fitness in self._scan_candidates(number, plans[number], sealed).values():
                if committed_at is not None:
                    commits.append(committed_at)
                    if not sealed:
                        completed += 1
                        arms[arm]["completed"] += 1
                        self._update_best(arms[arm], fitness)
                if not sealed and (trained_at is not None or committed_at is not None):
                    trained += 1
                if trained_at is not None:
                    activity.append(trained_at)
        if 0 in seals:
            for arm in ARMS[1:]:
                self._update_best(arms[arm], arms["shared"]["best"])
        if not seals and not plans:
            completed = count(progress.get("completed_candidates"))
            trained = completed
        return arms, completed, trained, commits, max((*activity, *commits), default=None)

    @staticmethod
    def _update_best(arm, fitness):
        if finite(fitness) and (arm["best"] is None or fitness > arm["best"]):
            arm["best"] = fitness

    def _read(self, now):
        progress = object_or_empty(read_json(self.root / "progress.json"))
        process = object_or_empty(read_json(self.root / "process.json"))
        launch = object_or_empty(read_json(self.root / "launch.json"))
        protocol = object_or_empty(read_json(self.root / "protocol.json"))
        plan = object_or_empty(protocol.get("plan"))

        process_at = timestamp(process.get("started_utc"))
        started = timestamp(launch.get("started_utc")) or process_at
        progress_at = timestamp(progress.get("updated_utc"))
        pid = count(process.get("pid"))
        alive = process_alive(pid, process_at)
        current = not (alive and progress_at and process_at and progress_at < process_at)
        phase = progress.get("phase") if current else None
        arms, completed, trained, commits, artifact_at = self._search(plan, progress, now)
        progress_total = count(progress.get("total_candidates"))
        total = count(protocol.get("unique_candidates")) or progress_total
        if completed is None and protocol and not progress:
            completed = trained = 0

        status = "waiting"
        activity_at = max((value for value in (progress_at, process_at, artifact_at, started)
                           if value is not None), default=None)
        if phase in ("complete", "failed"):
            status = phase
        elif alive is False:
            status = "stopped"
        elif activity_at is not None:
            stale_after = 900 if phase in (None, "search") else 3600
            status = "stale" if now - activity_at >= stale_after else "running"

        committed = max(commits, default=None)
        max_round = max((row["rounds_total"] for arm, row in arms.items()
                         if arm != "shared" and row["rounds_total"] is not None), default=None)
        run = dict(id=self.root.name, phase=phase, status=status, started_at=iso(started),
                   finished_at=iso(progress_at) if status in ("complete", "failed") else None,
                   completed=completed, total=total, trained=trained, round=count(progress.get("round")),
                   round_complete=progress.get("round_complete") is True, max_round=max_round,
                   active_rep=count(progress.get("active_rep")), active_batch=count(progress.get("active_batch")),
                   gpu_batch=count(protocol.get("gpu_batch")) or count(progress.get("gpu_batch")),
                   cpu_workers=count(process.get("workers")) or count(progress.get("score_workers")),
                   pending_scores=count(progress.get("pending_scores")), final_completed=count(progress.get("completed_models")),
                   final_total=count(progress.get("total_models")), error=progress.get("error") if current else None,
                   progress_at=iso(progress_at), last_committed_at=iso(committed), pid=pid, process_alive=alive)
        return dict(
            updated_at=iso(now), run=run, arms=list(arms.values()),
            throughput=score_rate(commits, now, started, completed, total, phase == "search" and status == "running"),
            hardware=self.hardware.read(protocol.get("device")),
            logs={stream: tail(self.root / f"execution.{stream}.log") for stream in ("stdout", "stderr")},
        )


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._serve(send_body=True)

    def do_HEAD(self):
        self._serve(send_body=False)

    def log_message(self, *_args):
        pass

    def _serve(self, send_body):
        port = self.server.server_port
        allowed = (f"127.0.0.1:{port}", f"localhost:{port}")
        if self.request_version != "HTTP/1.0" and self.headers.get("Host", "").lower() not in allowed:
            self.send_error(400)
            return
        try:
            path = urlsplit(self.path).path
        except ValueError:
            self.send_error(400)
            return
        if path == "/api/status":
            content_type = "application/json"
            content = json.dumps(self.server.dashboard.snapshot(), ensure_ascii=False, allow_nan=False).encode("utf-8")
        elif path in STATIC:
            name, content_type = STATIC[path]
            target = WEB_FILES / name
            try:
                if target.resolve().parent != WEB_FILES:
                    raise FileNotFoundError(name)
                content = target.read_bytes()
            except OSError:
                self.send_error(404)
                return
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if send_body:
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")
    with ThreadingHTTPServer(("127.0.0.1", args.port), RequestHandler) as server:
        server.dashboard = Dashboard(args.run_dir)
        print(f"Seed evolution dashboard at http://127.0.0.1:{args.port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
