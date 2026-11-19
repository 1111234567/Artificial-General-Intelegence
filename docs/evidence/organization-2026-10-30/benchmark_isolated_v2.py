"""Operational CPU/CUDA remeasurement; suspend only this study, always resume.

No checkpoint, training source, model state or scientific protocol is changed.
The journal enables manual recovery if the process is forcibly terminated.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def worker(model_name, device, output):
    import torch
    from experiments.lifelong.run import NativeTrainer, NativeConfig
    from experiments.organization.run import NativeTrainer as OrganizationTrainer, NativeConfig as OrganizationConfig
    torch.set_num_threads(1)
    if device == "cuda":
        assert torch.cuda.is_available()
        torch.cuda.set_per_process_memory_fraction(.20)
    cls, config = (NativeTrainer, NativeConfig(seed=999, device=device, architecture="open_shrink",
        schedule="mixed", learning="history_replay", coverage="wide", shrink_step=0.)) if model_name == "native" else (
        OrganizationTrainer, OrganizationConfig(seed=999, device=device, field_mode=model_name))
    trainer = cls(config)
    initial = trainer.initial_parameter_hash
    trainer.run_updates(4)
    if device == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(3):
        start = time.perf_counter()
        trainer.run_updates(4)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) / 4)
    save(output, dict(model=model_name, device=device, threads=1, seed=999,
        batch_size=config.batch_size, width=config.width, chunk_ticks=config.chunk_ticks,
        warmup_updates=4, measured_updates=12, seconds_per_update_by_block=times,
        seconds_per_update=sum(times) / len(times), initial_parameter_hash=initial,
        finite=bool(torch.isfinite(trainer.h).all()), runtime=trainer.runtime,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else 0))


def main():
    import psutil
    journal = HERE / "isolated-benchmark-v2-process-journal.json"
    if journal.exists():
        raise FileExistsError("preserve the previous operational journal")
    targets = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        info = process.info
        command = " ".join(info["cmdline"] or [])
        if "python" not in (info["name"] or "").lower():
            continue
        training = any(f"-m experiments.organization.{module} train " in command
                       for module in ("study", "acquisition")) and "organization-2026-10-30" in command
        queue = "-m experiments.lifelong.queue " in command and any(name in command for name in (
            "organization-training-commands-v1.json", "acquisition-training-commands-v1.json"))
        if training or queue:
            targets.append(dict(pid=info["pid"], created=info["create_time"], command=command, queue=queue))
    targets.sort(key=lambda row: not row["queue"])
    record = dict(start_utc=datetime.now(timezone.utc).isoformat(), active=True, targets=targets,
                  suspended=[], resumed=[], rows=[])
    save(journal, record)
    started = time.perf_counter()
    try:
        for item in targets:
            process = psutil.Process(item["pid"])
            if process.create_time() != item["created"]:
                raise RuntimeError("PID identity changed")
            process.suspend()
            record["suspended"].append(item)
            save(journal, record)
        print(f"Suspended {len(record['suspended'])} owned launcher/worker processes; states remain resident.", flush=True)
        time.sleep(1.5)
        for model in ("native", "plain", "coupled"):
            for device in ("cpu", "cuda"):
                output = HERE / f"isolated-benchmark-v2-{model}-{device}.json"
                environment = os.environ.copy()
                environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                if device == "cpu":
                    environment["CUDA_VISIBLE_DEVICES"] = "-1"
                else:
                    environment.pop("CUDA_VISIBLE_DEVICES", None)
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", model,
                    "--device", device, "--output", str(output)], cwd=ROOT, env=environment,
                    check=True, timeout=120, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                row = json.loads(output.read_text(encoding="utf-8"))
                record["rows"].append(row)
                save(journal, record)
                print(json.dumps({k: row[k] for k in ("model", "device", "seconds_per_update")}), flush=True)
            pair = record["rows"][-2:]
            assert pair[0]["initial_parameter_hash"] == pair[1]["initial_parameter_hash"]
    finally:
        for item in reversed(record["suspended"]):
            try:
                process = psutil.Process(item["pid"])
                if process.create_time() == item["created"]:
                    process.resume()
                    record["resumed"].append(item["pid"])
            except psutil.NoSuchProcess:
                record.setdefault("exited_before_resume", []).append(item["pid"])
        record.update(active=False, end_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.perf_counter() - started,
            interpretation="Single model, one thread, same workload; only this study's processes suspended. Desktop and unrelated apps remained. Paused CUDA allocations stayed resident. Scientific training-time counters can include this operational pause.")
        save(journal, record)
        print(f"Resumed {len(record['resumed'])} processes; benchmark window {record['elapsed_seconds']:.2f}s.", flush=True)
    save(HERE / "isolated-throughput-v2.json", record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("native", "plain", "coupled"))
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.worker:
        worker(arguments.worker, arguments.device, arguments.output)
    else:
        main()
