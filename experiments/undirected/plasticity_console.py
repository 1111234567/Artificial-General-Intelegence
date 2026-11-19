"""Serve a plasticity run's status and logs on localhost."""

from __future__ import annotations

import argparse
from functools import lru_cache
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
from time import monotonic
from urllib.parse import urlsplit
import zlib


DEFAULT_RUN = Path(__file__).resolve().parents[2] / "data/undirected/2026-11-16-evolved-plasticity-v1"
WEB_ROOT = Path(__file__).resolve().with_name("plasticity_webui")
STATIC = {
    "/": ("index.html", "text/html"),
    "/styles.css": ("styles.css", "text/css"),
    "/app.js": ("app.js", "text/javascript"),
    "/local-rule.js": ("local-rule.js", "text/javascript"),
    "/local-chart.js": ("local-chart.js", "text/javascript"),
    "/local-rule.css": ("local-rule.css", "text/css"),
    **{f"/local-rule-legacy/{name}": (f"local-rule-legacy/{name}", mime)
       for name, mime in (("styles.css", "text/css"), ("adapter.js", "text/javascript"),
                          ("audits.js", "text/javascript"), ("app.js", "text/javascript"))},
}
JSON_SOURCES = {"progress": dict, "history": list, "holdout": dict, "process": dict}
JSON_LIMIT = 8 * 1024 * 1024
HISTORY_LIMIT = 32 * 1024 * 1024
LOG_LIMIT = 32 * 1024
PRIVATE_AUDIT_LIMIT = 32 * 1024 * 1024
PRIVATE_HISTORY_READ_SECONDS = 2.0
PRIVATE_ARMS = ("downhill", "nondecreasing")
PRIVATE_CANDIDATE_ID = re.compile(r"g([0-9]+)-i([0-9]+)-c([0-9]+)")
PRIVATE_SEARCH_METRICS = (
    "strict", "completion", "strict_return", "selection_score", "symbol_late_mean",
    "symbol_late_a", "symbol_late_b", "symbol_first_return_mean", "ready_coverage",
    "premature_request_rate", "idle", "best_fitness", "mean_fitness", "mean_symbol_late",
)
PRIVATE_GRAPH_METRICS = (
    "symbol_accuracy", "initial_edges", "mean_edges", "final_edges", "average_edges",
    "active_edges", "final_reciprocity", "weight_updates", "births", "deaths",
    "changed_writes", "attempted_writes", "weight_change", "reward_events",
)


def open_read(path):
    if os.name != "nt":
        return path.open("rb")

    import _winapi
    import msvcrt

    # Allow delete sharing and close the file before decoding. On Windows the
    # writer also retries if a read overlaps its atomic replacement.
    share_read_write_delete = 7
    normal_attributes = 0x80
    handle = _winapi.CreateFile(
        str(path), _winapi.GENERIC_READ, share_read_write_delete, 0,
        _winapi.OPEN_EXISTING, normal_attributes, 0,
    )
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except OSError:
        _winapi.CloseHandle(handle)
        raise
    return os.fdopen(descriptor, "rb")


def finite_float(value):
    number = float(value)
    return number if math.isfinite(number) else None


def read_json(path, expected_type, *, limit=JSON_LIMIT, compressed=False):
    try:
        with open_read(path) as source:
            if compressed:
                with gzip.GzipFile(fileobj=source) as archive:
                    contents = archive.read(limit + 1)
            else:
                contents = source.read(limit + 1)
        if len(contents) > limit:
            return None, "too_large"
        result = json.loads(contents, parse_float=finite_float, parse_constant=lambda _: None)
        if not isinstance(result, expected_type):
            return None, "invalid"
        return result, None
    except FileNotFoundError:
        return None, None
    except (ValueError, OSError, EOFError, zlib.error):
        return None, "unreadable"


def read_log(path):
    try:
        with open_read(path) as source:
            end = source.seek(0, os.SEEK_END)
            offset = max(0, end - LOG_LIMIT)
            source.seek(offset)
            contents = source.read(LOG_LIMIT)
        if offset and b"\n" in contents:
            contents = contents.partition(b"\n")[2]
        return {"text": contents.decode("utf-8", errors="replace"), "truncated": bool(offset)}
    except FileNotFoundError:
        return {"text": "", "truncated": False}
    except OSError:
        return {"text": "", "truncated": False, "error": "unreadable"}


def process_active(process):
    pid = process.get("pid") if process else None
    if type(pid) is not int or not 0 < pid < 2**32:
        return None
    if os.name == "nt":
        import _winapi

        try:
            handle = _winapi.OpenProcess(_winapi.SYNCHRONIZE, False, pid)
        except OSError as error:
            return False if error.winerror == 87 else None  # ERROR_INVALID_PARAMETER
        try:
            return _winapi.WaitForSingleObject(handle, 0) == _winapi.WAIT_TIMEOUT
        finally:
            _winapi.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except (PermissionError, OverflowError):
        return None


def positive_integer(value):
    return type(value) is int and value > 0


def numeric_fields(source, keys=None):
    if not isinstance(source, dict):
        return {}
    return {key: source[key] for key in source if (keys is None or key in keys)
            and type(source[key]) in (int, float)}


LOCAL_PLOT_FIELDS = frozenset((
    "selection_score", "strict", "completion", "first_finish_correct", "ready_coverage", "idle",
    "premature_request_rate", "post_answer_actions", "pair_strict_rate", "pairs_both_strict", "eligible_pairs",
    "symbol_late_a", "symbol_late_b", "symbol_late_mean", "first_return_worst_context", "first_return_mean",
    "initial_edges", "mean_edges", "final_edges", "final_effective_edges", "final_reciprocity", "sparsity_bonus",
    "births", "deaths", "attempted_writes", "changed_writes", "weight_updates", "bias_updates",
    "weight_change", "bias_change", "final_weight_delta", "final_bias_delta", "final_gate_abs",
    "reward_events", "public_ack_events", "public_rej_events",
))


def local_rule_metrics(source, arm):
    if not isinstance(source, dict) or not source:
        return {}
    metrics = numeric_fields(source, LOCAL_PLOT_FIELDS)
    if arm == "control":
        # The original rule holds its birth biases fixed throughout each life.
        for field in ("bias_updates", "bias_change", "final_bias_delta"):
            if field not in source:
                metrics[field] = 0
    if "completion" in metrics:
        for rate, baseline in (("recovery_rate", "strict"), ("retry_recovery_rate", "first_finish_correct")):
            if baseline in metrics:
                metrics[rate] = metrics["completion"] - metrics[baseline]
    return metrics


def local_rule_summary(row, *, arm=None, generation=None, development_seed=None):
    if not isinstance(row, dict):
        return None
    stored_arm, stored_generation = row.get("arm"), row.get("generation")
    if ((arm is not None and stored_arm not in (None, arm))
            or (generation is not None and stored_generation not in (None, generation))):
        return None
    arm = stored_arm if arm is None else arm
    generation = stored_generation if generation is None else generation
    candidates = row.get("candidates")
    if not isinstance(arm, str) or not positive_integer(generation) or not positive_integer(candidates):
        return None
    champion = row.get("champion")
    metrics = local_rule_metrics(champion, arm)
    if isinstance(champion, dict) and isinstance(champion.get("id"), str):
        metrics["id"] = champion["id"]
        metrics.update(numeric_fields(champion, ("island",)))
    result = {"arm": arm, "generation": generation, "candidates": candidates,
              "sealed": stored_arm is not None and stored_generation is not None,
              "champion": metrics, "mean": local_rule_metrics(row.get("mean"), arm)}
    seed = row.get("development_seed", development_seed)
    if type(seed) is int:
        result["development_seed"] = seed
    return result


def cached_local_file(path, reader):
    try:
        info = path.stat()
    except FileNotFoundError:
        return None, None
    except OSError:
        return None, "unreadable"
    return reader(path, (info.st_mtime_ns, info.st_size))


@lru_cache(maxsize=2)
def local_rule_history(path, modified):
    rows, error = read_json(path, list, limit=HISTORY_LIMIT)
    if error or rows is None:
        return None, error
    return [summary for row in rows if (summary := local_rule_summary(row))], None


def compact_local_progress(progress):
    if not progress:
        return
    latest = progress.get("latest")
    current = None
    if (isinstance(progress.get("current_arm"), str) and positive_integer(progress.get("generation"))
            and isinstance(latest, dict) and latest.get("candidates") == progress.get("generation_completed")):
        current = local_rule_summary(latest, arm=progress["current_arm"], generation=progress["generation"],
                                     development_seed=progress.get("development_seed"))
    progress["latest"] = current
    for arm, branch in (progress.get("arms_progress") or {}).items():
        if isinstance(branch, dict):
            branch["latest"] = local_rule_summary(branch.get("latest"), arm=arm)


@lru_cache(maxsize=128)
def local_holdout_summary(path, modified):
    row, error = read_json(path, dict)
    if row is None:
        return None, error
    arms = row.get("arms")
    if not positive_integer(row.get("generation")) or not isinstance(arms, dict):
        return None, "invalid"
    result = {**numeric_fields(row, ("generation", "actors", "visits")), "final": row.get("final") is True,
              "arms": {arm: local_rule_metrics(values.get("mean"), arm)
                       for arm, values in arms.items() if isinstance(values, dict)}}
    if isinstance(row.get("seeds"), list):
        result["seed_count"] = len(row["seeds"])
    return result, None


@lru_cache(maxsize=16)
def local_receipt_summary(path, modified):
    row, error = read_json(path, dict)
    if row is None:
        return None, error
    if not positive_integer(row.get("generation")) or not isinstance(row.get("delays"), dict):
        return None, "invalid"
    result = {**numeric_fields(row, ("generation", "opportunities", "events")), "delays": row["delays"]}
    if isinstance(row.get("seeds"), list):
        result["seed_count"] = len(row["seeds"])
    return result, None


def local_rule_audits(run_dir, protocol, read_errors):
    holdouts = {}
    audit_dir = run_dir / "audits"
    for path in audit_dir.glob("*.json"):
        if path.stem.isdecimal():
            summary, error = cached_local_file(path, local_holdout_summary)
            if error:
                read_errors[f"audits/{path.name}"] = error
            if summary:
                holdouts[summary["generation"]] = summary
    holdout, error = cached_local_file(run_dir / "holdout.json", local_holdout_summary)
    if error:
        read_errors["holdout"] = error
    if holdout:
        holdouts.setdefault(holdout["generation"], holdout)
    holdout_history = [holdouts[generation] for generation in sorted(holdouts)]

    receipts, receipt_history = {}, []
    for generation in sorted(protocol.get("receipt_generations", ())):
        arms = {}
        for arm in protocol["arms"]:
            path = audit_dir / f"receipt-{generation:04d}-{arm}.json"
            summary, error = cached_local_file(path, local_receipt_summary)
            if error:
                read_errors[f"audits/{path.name}"] = error
            if summary and summary["generation"] == generation:
                arms[arm] = receipts[arm] = summary
        if arms:
            receipt_history.append({"generation": generation, "arms": arms})

    return {"holdout": holdout_history[-1] if holdout_history else {"generation": None, "arms": {}},
            "holdout_history": holdout_history, "receipts": receipts, "receipt_history": receipt_history}


def private_graph_metrics(row):
    metrics = numeric_fields(row, PRIVATE_GRAPH_METRICS)
    for canonical, alias in (("mean_edges", "average_edges"), ("final_edges", "active_edges")):
        value = metrics.get(canonical, metrics.get(alias))
        if value is not None:
            metrics[canonical] = metrics[alias] = value
    return metrics


@lru_cache(maxsize=16)
def journal_candidate(path, offset, ident):
    # Committed batches are immutable. Missing and malformed batches are retried.
    rows, error = read_json(path, list, compressed=True)
    if rows is None or len(rows) <= offset:
        raise ValueError(error or "missing candidate")
    row = rows[offset]
    if not isinstance(row, dict) or row.get("id") != ident:
        raise ValueError("candidate ID differs")
    return private_graph_metrics(row)


def private_candidate(branch_dir, protocol, generation, ident, committed):
    match = PRIVATE_CANDIDATE_ID.fullmatch(ident) if isinstance(ident, str) else None
    islands = protocol.get("islands")
    width = protocol.get("batch_size")
    if not (match and positive_integer(generation) and positive_integer(committed)
            and positive_integer(islands) and positive_integer(width)):
        return {}, None
    label_generation, island, lane = map(int, match.groups())
    index = lane * islands + island
    if label_generation + 1 != generation or island >= islands or index >= committed:
        return {}, None
    start, offset = divmod(index, width)
    journal = branch_dir / "generations" / f"{generation:04d}" / "batches" / f"{start * width:04d}.json.gz"
    try:
        return journal_candidate(journal, offset, ident), journal
    except ValueError:
        return {}, None


def private_history(rows, branch_dir, protocol, journals, read_seconds):
    # Return all search summaries promptly and fill uncached graph metrics newest first.
    deadline = monotonic() + read_seconds
    history = []
    for row in reversed(rows or ()):
        if not isinstance(row, dict) or not positive_integer(row.get("generation")):
            continue
        generation = row["generation"]
        summary = {"generation": generation, **numeric_fields(row, ("candidates", *PRIVATE_SEARCH_METRICS))}
        summary.update(private_graph_metrics(row))
        rule = row.get("selection_rule")
        if isinstance(rule, str):
            summary["selection_rule"] = rule
        ident = row.get("best_candidate")
        if isinstance(ident, str):
            summary["best_candidate"] = ident
            key = branch_dir, generation, ident
            metrics = journals.get(key)
            if metrics is None and monotonic() < deadline:
                metrics, journal = private_candidate(branch_dir, protocol, generation, ident,
                                                     row.get("candidates", protocol.get("population")))
                if journal:
                    journals[key] = metrics
            if metrics:
                summary.update(metrics)
        for key in ("inheritance_summary", "exploration_summary"):
            values = numeric_fields(row.get(key))
            if values:
                summary[key] = values
        history.append(summary)
    history.reverse()
    return history


@lru_cache(maxsize=32)
def private_audit(path):
    raw, error = read_json(path, dict, limit=PRIVATE_AUDIT_LIMIT)
    if raw is None:
        raise ValueError(error or "unreadable")
    if not positive_integer(raw.get("generation")) or not isinstance(raw.get("conditions"), dict):
        raise ValueError("invalid")
    conditions = {}
    for clock, condition in raw["conditions"].items():
        if not isinstance(condition, dict) or not isinstance(condition.get("modes"), dict):
            continue
        modes = {}
        for mode, report in condition["modes"].items():
            aggregate = report.get("aggregate") if isinstance(report, dict) else None
            if isinstance(aggregate, dict):
                means = numeric_fields(aggregate.get("means"))
                if means:
                    modes[mode] = {**means, **private_graph_metrics(means)}
        if modes:
            conditions[clock] = {"modes": modes, "timing": numeric_fields(condition.get("timing"))}
    if not conditions:
        raise ValueError("invalid")
    return {"generation": raw["generation"], "candidate": raw.get("candidate"),
            "actors": raw.get("actors"), "seeds": raw.get("seeds"), "conditions": conditions}


def latest_private_audit(branch_dir):
    try:
        paths = (path for path in (branch_dir / "audits").glob("*.json") if path.stem.isdecimal())
        path = max(paths, key=lambda item: int(item.stem), default=None)
        return (private_audit(path), None) if path else (None, None)
    except OSError:
        return None, "unreadable"
    except ValueError as error:
        return None, str(error)


def enrich_private_progress(progress, branch_dir, protocol, branch):
    generation = progress.get("generation")
    committed = progress.get("generation_completed")
    progress.pop("candidate_journal", None)
    counts = progress.get("arms_progress") or {}
    sealed = (counts.get(branch["id"]) or {}).get("sealed_generations")
    if positive_integer(sealed):
        checkpoint = branch_dir / "generations" / f"{sealed:04d}" / "elites.pt"
        if checkpoint.is_file():
            progress["checkpoint"] = checkpoint.relative_to(branch_dir.parents[1]).as_posix()

    # The producer reuses its status dict across arms and generations.
    if not positive_integer(committed) or progress.get("phase") in ("holdout", "complete"):
        for key in (*PRIVATE_SEARCH_METRICS, *PRIVATE_GRAPH_METRICS, "best_candidate"):
            progress.pop(key, None)
        if not positive_integer(committed):
            return
        row = next((row for row in reversed(branch["history"]) if row["generation"] == generation), {})
        progress.update(numeric_fields(row, PRIVATE_SEARCH_METRICS))
        if "best_candidate" in row:
            progress["best_candidate"] = row["best_candidate"]

    ident = progress.get("best_candidate")
    metrics, journal = private_candidate(branch_dir, protocol, generation, ident, committed)
    progress.update(metrics)
    if journal:
        progress["candidate_journal"] = journal.relative_to(branch_dir.parents[1]).as_posix()


def private_snapshot(run_dir, protocol, progress, read_errors, journals):
    current = None
    branches = []
    configured_arms = protocol.get("arms", PRIVATE_ARMS)
    arms = [arm for arm in PRIVATE_ARMS if arm in configured_arms] if isinstance(configured_arms, (list, tuple)) else []
    branch_read_seconds = PRIVATE_HISTORY_READ_SECONDS / max(1, protocol["pairs"] * len(arms))
    for pair in range(1, protocol["pairs"] + 1):
        for arm in arms:
            ident = f"{pair:02d}/{arm}"
            branch_dir = run_dir / f"pair-{pair:02d}" / arm
            history, history_error = read_json(branch_dir / "history.json", list, limit=HISTORY_LIMIT)
            holdout, holdout_error = latest_private_audit(branch_dir)
            if history_error:
                read_errors[f"{ident}/history"] = history_error
            if holdout_error:
                read_errors[f"{ident}/holdout"] = holdout_error
            branch = {"id": ident, "pair": pair, "arm": arm, "holdout": holdout,
                      "history": private_history(history, branch_dir, protocol, journals, branch_read_seconds)}
            branches.append(branch)
            if progress and progress.get("current_pair") == pair and progress.get("current_arm") == arm:
                current = ident
                enrich_private_progress(progress, branch_dir, protocol, branch)
    result = {
        "pairs": protocol["pairs"], "current": current, "branches": branches,
        **{key: protocol[key] for key in ("visits", "actors", "requests_per_visit")
           if positive_integer(protocol.get(key))},
    }
    sampling = protocol.get("context_sampling")
    if isinstance(sampling, str):
        result["context_sampling"] = sampling
    scoring = protocol.get("selection_score")
    if isinstance(scoring, dict) and isinstance(scoring.get("rule"), str):
        result["selection_rule"] = scoring["rule"]
    return result


class ConsoleServer(ThreadingHTTPServer):
    def __init__(self, run_dir, port):
        self.run_dir = run_dir
        self.protocol = None
        self.private_journals = {}
        super().__init__(("127.0.0.1", port), ConsoleHandler)

    def snapshot(self):
        result = {"run_name": self.run_dir.name, "read_errors": {}}
        if self.protocol is None:
            self.protocol, _ = read_json(self.run_dir / "protocol.json", dict)
        private = (self.protocol and self.protocol.get("task") == "private_symbol"
                   and positive_integer(self.protocol.get("pairs")))
        local = self.protocol and self.protocol.get("experiment_kind") == "local_rule_comparison"
        for name, source_type in JSON_SOURCES.items():
            if private and name in ("history", "holdout"):
                result[name] = None
                continue
            if local and name == "holdout":
                continue
            limit = HISTORY_LIMIT if name == "history" else JSON_LIMIT
            path = self.run_dir / f"{name}.json"
            if local and name == "history":
                value, error = cached_local_file(path, local_rule_history)
            else:
                value, error = read_json(path, source_type, limit=limit)
            result[name] = value
            if error:
                result["read_errors"][name] = error
        result["process_active"] = process_active(result["process"])
        if local:
            compact_local_progress(result["progress"])
            result.update(local_rule_audits(self.run_dir, self.protocol, result["read_errors"]))
            result["local_rule"] = {"arms": self.protocol["arms"],
                                    **numeric_fields(self.protocol, ("actors", "islands")),
                                    **numeric_fields(self.protocol.get("world"), ("requests_per_life",))}
        if private:
            result["private_symbol"] = private_snapshot(self.run_dir, self.protocol, result["progress"],
                                                        result["read_errors"], self.private_journals)
        return result


class ConsoleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        host, port = self.server.server_address
        if self.headers.get("Host") not in (f"{host}:{port}", f"localhost:{port}"):
            self.send_error(400)
            return

        path = urlsplit(self.path).path
        if path == "/api/snapshot":
            self.send_json(self.server.snapshot())
        elif path == "/api/logs":
            self.send_json({
                stream: read_log(self.server.run_dir / f"execution.{stream}.log")
                for stream in ("stdout", "stderr")
            })
        elif path in STATIC:
            filename, content_type = STATIC[path]
            if path == "/":
                protocol = self.server.protocol
                if protocol is None:
                    protocol, _ = read_json(self.server.run_dir / "protocol.json", dict)
                if protocol and protocol.get("experiment_kind") == "local_rule_comparison":
                    filename = "local-rule-legacy/index.html"
            self.send_payload((WEB_ROOT / filename).read_bytes(), content_type)
        else:
            self.send_error(404)

    def send_json(self, data):
        payload = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_payload(payload, "application/json")

    def send_payload(self, payload, content_type):
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (ConnectionResetError, BrokenPipeError):
            pass

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    with ConsoleServer(args.run_dir.resolve(), args.port) as server:
        host, port = server.server_address
        print(f"Plasticity console available at http://{host}:{port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
