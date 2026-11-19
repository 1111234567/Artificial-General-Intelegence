"""Optional host-only acceleration for the sealed local-rule training loop."""

from contextlib import contextmanager
from statistics import fmean
from unittest.mock import patch

from experiments.seed_evolution.large_runner import _atomic, _json_bytes

from . import local_rule_run as run


class _Group:
    def __init__(self):
        self.count = 0
        self.champion = None
        self.champion_rank = None
        self.columns = {}

    def append(self, row, rank):
        metrics = row["metrics"]
        if self.count == 0:
            self.columns = {key: [] for key in sorted(metrics) if type(metrics[key]) in (int, float)}
        for key, values in list(self.columns.items()):
            value = metrics.get(key)
            if type(value) in (float, int):
                values.append(value)
            else:
                del self.columns[key]
        if self.count == 0 or rank > self.champion_rank:
            self.champion = row
            self.champion_rank = rank
        self.count += 1

    def snapshot(self):
        if self.count == 0:
            raise ValueError("max() iterable argument is empty")
        return {"candidates": self.count, "champion": self.champion,
                "mean": {key: fmean(values) for key, values in self.columns.items()}}


class _ProgressSummary:
    """Index each training row once while the runner appends to a generation list."""

    def __init__(self, rank):
        self.rank = rank
        self.rows = None
        self.overall = _Group()
        self.groups = []

    def __call__(self, rows, islands):
        if rows is not self.rows or islands != len(self.groups) or len(rows) < self.overall.count:
            self.rows = rows
            self.overall = _Group()
            self.groups = [_Group() for _ in range(islands)]

        for i in range(self.overall.count, len(rows)):
            row = rows[i]
            group = self.groups[row["island"]]
            rank = self.rank(row)
            self.overall.append(row, rank)
            group.append(row, rank)
        return {**self.overall.snapshot(),
                "islands": {str(i): group.snapshot() for i, group in enumerate(self.groups) if group.count}}


class _HistoryWriter:
    """Reuse encoded append-only summaries and publish the complete JSON atomically."""

    def __init__(self, write):
        self.write = write
        self.history_path = None
        self.history = None
        self.encoded = []

    def __call__(self, path, value):
        if path.name != "history.json" or type(value) is not list:
            return self.write(path, value)
        if path != self.history_path or value is not self.history or len(value) < len(self.encoded):
            self.history_path, self.history = path, value
            self.encoded = []

        for i in range(len(self.encoded), len(value)):
            # The runner's serializer indents by two spaces at each level.
            row = _json_bytes(value[i]).removesuffix(b"\n")
            self.encoded.append(b"  " + row.replace(b"\n", b"\n  "))
        payload = b"[\n" + b",\n".join(self.encoded) + b"\n]\n" if self.encoded else b"[]\n"
        return _atomic(path, lambda temporary: temporary.write_bytes(payload))


@contextmanager
def installed():
    """Enable fast reporting only for a launch of the local-rule runner."""
    with (patch.object(run, "_summary", _ProgressSummary(run.rank)),
          patch.object(run, "_write", _HistoryWriter(run._write))):
        yield
