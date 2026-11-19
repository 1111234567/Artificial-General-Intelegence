"""Paired CPU timing for fast reporting using archived candidate and history rows."""

from argparse import ArgumentParser
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

from experiments.seed_evolution.large_runner import _json_bytes

from . import local_rule_run as run
from .fast_training_host import _HistoryWriter, _ProgressSummary


def _training_rows(folder, batch_size):
    plan = run.read(folder / "plan.json")
    rows = []
    for start in range(0, len(plan), batch_size):
        entries = plan[start:start + batch_size]
        scores = run.read_batch(folder / "batches" / f"{start:04d}.json.gz", entries)
        rows.extend(run._training_rows([row["metrics"] for row in scores], entries))
    return rows


def _timings(exercise, reps):
    pairs = []
    for index in range(reps):
        order = (False, True) if index % 2 == 0 else (True, False)
        measured = {}
        for fast in order:
            measured[fast] = exercise(fast)
        pairs.append({"original_ms": measured[False] * 1000, "fast_ms": measured[True] * 1000})
    return {"pairs": pairs, "median_original_ms": median(row["original_ms"] for row in pairs),
            "median_fast_ms": median(row["fast_ms"] for row in pairs),
            "median_paired_speedup": median(row["original_ms"] / row["fast_ms"] for row in pairs)}


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=run.OUT)
    parser.add_argument("--generation", type=int, default=64)
    parser.add_argument("--arm", choices=run.ARMS, default="local")
    parser.add_argument("--report-rows", type=int, default=192)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if min(args.generation, args.report_rows, args.repetitions) < 1:
        parser.error("Generation, report interval, and repetitions must be positive")

    protocol = run.read(args.run / "protocol.json")
    batch_size = protocol["batch_size"]
    islands = protocol["islands"]
    folder = run._directory(args.run, args.arm, args.generation)
    rows = _training_rows(folder, batch_size)
    history = run.read(args.run / "history.json")
    history = [row for row in history if row["generation"] <= args.generation]
    endpoints = [*range(batch_size, len(rows), args.report_rows), len(rows), len(rows)]
    expected = [run._summary(rows[:end], islands) for end in endpoints]
    archived = run.read(folder / "summary.json")
    if any(archived[key] != value for key, value in expected[-1].items()):
        raise AssertionError("Original reporting did not reproduce the archived summary")

    def summary_exercise(fast):
        summarize = _ProgressSummary(run.rank) if fast else run._summary
        partial, reported = [], []
        before = perf_counter()
        for end in endpoints:
            partial.extend(rows[len(partial):end])
            reported.append(summarize(partial, islands))
        elapsed = perf_counter() - before
        if reported != expected:
            raise AssertionError("Training summaries differ")
        return elapsed

    with TemporaryDirectory(prefix="fast-training-host-") as folder:
        path = Path(folder) / "history.json"

        def history_exercise(fast):
            write = _HistoryWriter(run._write) if fast else run._write
            prefix, tail = history[:-3], history[-3:]
            write(path, prefix)
            before = perf_counter()
            for row in tail:
                prefix.append(row)
                write(path, prefix)
            elapsed = perf_counter() - before
            if path.read_bytes() != _json_bytes(history):
                raise AssertionError("History publication differs")
            return elapsed

        result = {"generation": args.generation, "arm": args.arm, "candidates": len(rows),
                  "islands": islands, "summary_calls": len(endpoints), "history_rows": len(history),
                  "history_writes_per_pair": 3, "original_reproduces_archive": True,
                  "summary": _timings(summary_exercise, args.repetitions),
                  "history": _timings(history_exercise, args.repetitions)}
    output = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
