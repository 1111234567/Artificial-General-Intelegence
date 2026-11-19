"""Independent content/timing artifact audit; never advances a world or trains.

The auditor reads saved records and recomputes their arithmetic without calling
the experiment summarizers. Source and input pins are checked against the exact
saved protocols. Probe verification uses an independent augmented least-squares
formulation rather than the executor's normal-equation implementation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import struct

for _thread_key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/content-timing/independent-audit.json"
FAMILY_MAPPINGS = ((0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 2, 3, 0))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, label):
    if not condition:
        raise AssertionError(label)


def equal(actual, expected, label, tolerance=1e-10):
    if actual is None or expected is None:
        require(actual is expected, f"{label}: {actual!r} != {expected!r}")
    elif isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        require(actual == expected or abs(actual - expected) <= tolerance, f"{label}: {actual!r} != {expected!r}")
    else:
        require(actual == expected, f"{label}: values differ")


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def semantic_digest(value):
    """Independent implementation of the frozen canonical digest format."""
    import numpy as np
    import torch

    digest = hashlib.sha256()
    def update(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(f"tensor/{tensor.dtype}/{tuple(tensor.shape)}/".encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(f"ndarray/{array.dtype}/{array.shape}/".encode())
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict{")
            for key in sorted(item, key=lambda candidate: (type(candidate).__name__, repr(candidate))):
                update(key)
                update(item[key])
            digest.update(b"}")
        elif isinstance(item, (list, tuple)):
            digest.update(b"list[" if isinstance(item, list) else b"tuple[")
            for member in item:
                update(member)
            digest.update(b"]")
        elif isinstance(item, float):
            digest.update(b"float/" + struct.pack("!d", item))
        elif isinstance(item, (str, int, bool)) or item is None:
            digest.update(type(item).__name__.encode() + b"/")
            digest.update(json.dumps(item, ensure_ascii=False).encode() + b";")
        else:
            raise TypeError(type(item).__name__)
    update(value)
    return digest.hexdigest()


def check_digest(value, key):
    equal(value[key], semantic_digest({name: item for name, item in value.items() if name != key}), key)


class Bindings:
    def __init__(self):
        self.expected = {}
        self.provenance = defaultdict(set)
        self.verified = {}

    def add(self, path, expected, where):
        item = Path(path)
        path = str((item if item.is_absolute() else ROOT / item).resolve())
        require(isinstance(expected, str) and len(expected) == 64,
                f"malformed SHA256 at {where}")
        if path in self.expected:
            equal(self.expected[path], expected, f"conflicting source/input pins: {path}")
        self.expected[path] = expected
        self.provenance[path].add(where)

    def mapping(self, values, where):
        for path, expected in values.items():
            self.add(path, expected, where)

    def discover(self, value, where):
        """Find explicit file records and absolute-path SHA256 maps only."""
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                self.add(value["path"], value["sha256"], where)
            for key, item in value.items():
                if (isinstance(item, str) and len(item) == 64
                        and all(c in "0123456789abcdef" for c in item)
                        and Path(str(key)).is_absolute()):
                    self.add(key, item, where)
                elif isinstance(item, (dict, list)):
                    self.discover(item, f"{where}/{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                self.discover(item, f"{where}/{index}")

    def verify(self):
        for path, expected in self.expected.items():
            actual = sha(path)
            equal(actual, expected, f"file binding {path}")
            self.verified[path] = dict(sha256=actual, bytes=Path(path).stat().st_size,
                                       declared_by=sorted(self.provenance[path]))
        return dict(audit="passed", unique_files=len(self.verified), files=self.verified)


def old_source_bindings(bindings):
    original_path = ROOT / "docs/evidence/undirected-2026-10-30/protocol-v1.json"
    coverage_path = ROOT / "data/coverage-exit/coverage-v1/protocol.json"
    readout_path = ROOT / "data/action-return/readout-v1/protocol.json"
    refit_path = ROOT / "data/action-return/refit-policy-v1/protocol.json"
    original = read(original_path)
    require(len(original["sources"]) == 26, "original native source closure must contain 26 files")
    bindings.mapping(original["sources"], str(original_path))
    bindings.mapping(read(coverage_path)["source_pins"], str(coverage_path))
    readout = read(readout_path)
    bindings.mapping(readout["original_source_pins"], str(readout_path))
    bindings.discover(readout["source"], str(readout_path))
    refit = read(refit_path)
    bindings.mapping(refit["source_files"], str(refit_path))
    for path in (original_path, coverage_path, readout_path, refit_path):
        bindings.add(path, sha(path), "audit protocol inventory")
    return dict(original_native_source_count=26,
                scope="Original native closure, coverage wrapper, BEGIN/FINISH/refit source closure; all additional actual inputs come from current study protocols.")


def load_tensor(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def matrix_from_predictions(targets, predictions, classes):
    import numpy as np

    return np.bincount(classes * targets + predictions,
                       minlength=classes * classes).reshape(classes, classes)


def probe_states(saved):
    """Independent row-table reconstruction, without importing probe.py."""
    import numpy as np

    rows, x = saved["rows"], saved["x24"].numpy()
    require(x.dtype == np.float64 and x.shape == (len(rows), 24), "probe feature shape/dtype")
    identities = sorted({row["anchor_id"] for row in rows})
    durations = sorted({row["duration"] for row in rows})
    ai = {identity: index for index, identity in enumerate(identities)}
    phases = ("cue", "midpoint", "first_legal")
    states = np.full((len(ai), len(durations), 4, 3, 24), np.nan)
    metadata = {}
    for index, row in enumerate(rows):
        require(FAMILY_MAPPINGS[row["family"]][row["raw_payload"]] == row["mapped_payload"],
                "raw-to-family target mapping")
        if row["stage"] not in phases:
            continue
        identity = row["anchor_id"]
        meta = (row["actor_lane"], row["job"], row["family"])
        require(identity not in metadata or metadata[identity] == meta, "anchor metadata changes")
        metadata[identity] = meta
        where = (ai[identity], durations.index(row["duration"]), row["raw_payload"], phases.index(row["stage"]))
        require(np.isnan(states[where]).all(), "duplicate paired probe phase")
        states[where] = x[index]
    require(np.isfinite(states).all(), "incomplete paired probe block")
    return identities, [metadata[identity] for identity in identities], durations, states


def audit_probe(bindings):
    import numpy as np

    folder = ROOT / "data/content-timing/probe-v1"
    path = folder / "protocol.json"
    plan, summary = read(path), read(folder / "summary.json")
    bindings.mapping(plan["inputs"], str(path))
    equal(summary["plan_sha256"], sha(ROOT / "experiments/content_timing/probe-plan.md"), "probe plan SHA")
    equal(summary["script_sha256"], sha(ROOT / "experiments/content_timing/probe.py"), "probe program SHA")
    equal(summary["matched_protocol_sha256"], sha(ROOT / "data/content-timing/matched-v1/protocol.json"), "probe matched protocol SHA")
    rows_verified, recomputed = 0, []
    for result in summary["seeds"]:
        seed = result["seed"]
        source = ROOT / result["source"]
        bindings.add(source, result["source_sha256"], "probe saved paired input")
        saved = load_tensor(source)
        equal(saved["protocol_hash"], result["matched_protocol_hash"], "probe matched semantic protocol hash")
        identities, meta, durations, states = probe_states(saved)
        equal(len(identities), result["anchors"], "probe anchor count")
        lanes = np.array([item[0] for item in meta])
        jobs = np.array([item[1] for item in meta])
        families = np.array([item[2] for item in meta])
        totals = defaultdict(Counter)
        folds = result["folds"]
        require(len(folds) == len(durations) * 2 * 2 * 5 * 2, "probe full fold inventory")
        for row in folds:
            matrix = np.array(row["confusion_true_by_pred"], dtype=np.int64)
            require(matrix.shape == (4, 4) and (matrix >= 0).all(), "probe confusion matrix")
            equal(int(matrix.sum()), row["total"], "probe matrix denominator")
            equal(int(matrix.trace()), row["correct"], "probe matrix numerator")
            require(np.all(matrix.sum(1) == row["test_anchors"]), "probe balanced labels")
            equal(row["null_correct"] * 4, row["total"], "probe within-anchor null")
            test = (jobs == row["job"]) & np.isin(lanes, row["test_lanes"])
            train = (jobs == row["job"]) & ~test
            equal(int(test.sum()), row["test_anchors"], "probe test anchors")
            equal(int(train.sum()), row["train_anchors"], "probe train anchors")
            require(set(families[test]) == set(families[train]) == set(range(4)), "probe family coverage")
            require(set(lanes[test]).isdisjoint(set(lanes[train])), "probe train/test actor disjointness")
            key = (row["duration"], row["train_phase"], row["test_phase"], row["label"])
            totals[key].update({name: row[name] for name in ("correct", "null_correct", "total")})
            rows_verified += 1
        require(len(result["pooled"]) == len(totals), "probe pooled inventory")
        for row in result["pooled"]:
            counts = totals[(row["duration"], row["train_phase"], row["test_phase"], row["label"])]
            for name in ("correct", "null_correct", "total"):
                equal(row[name], counts[name], f"probe pooled {name}")
            equal(row["accuracy"], rate(counts["correct"], counts["total"]), "probe pooled accuracy")
        # One late-stage mapped-content fit per seed, with the actor split
        # fixed independently of outcomes. np.linalg.lstsq uses an SVD-based
        # augmented system, unlike the executor's normal-equation solve.
        row = next(item for item in folds if item["duration"] == max(durations)
                   and item["job"] == 0 and item["test_lanes"] == [6, 7]
                   and item["train_phase"] == item["test_phase"] == "first_legal"
                   and item["label"] == "mapped")
        test = (jobs == 0) & np.isin(lanes, (6, 7))
        train = (jobs == 0) & ~test
        di = durations.index(max(durations))
        x = states[train, di, :, 2, :].reshape(-1, 24)
        xt = states[test, di, :, 2, :].reshape(-1, 24)
        targets = np.array(FAMILY_MAPPINGS)[families]
        y, yt = targets[train].reshape(-1), targets[test].reshape(-1)
        mean, scale = x.mean(0), x.std(0)
        scale[scale == 0] = 1
        z = (x - mean) / scale
        full = np.column_stack((np.ones(len(x)), z))
        penalty = np.zeros((24, 25))
        penalty[:, 1:] = np.eye(24) * np.sqrt(.001 * len(x))
        augmented = np.vstack((full, penalty))
        outcome = np.vstack((np.eye(4)[y], np.zeros((24, 4))))
        coefficients, _, rank, singular = np.linalg.lstsq(augmented, outcome, rcond=None)
        predictions = (np.column_stack((np.ones(len(xt)), (xt - mean) / scale)) @ coefficients).argmax(1)
        matrix = matrix_from_predictions(yt, predictions, 4)
        equal(matrix.tolist(), row["confusion_true_by_pred"], "independent SVD late-stage probe confusion")
        centers = xt.reshape(-1, 4, 24).mean(1)
        null = np.repeat((np.column_stack((np.ones(len(centers)), (centers - mean) / scale)) @ coefficients).argmax(1), 4)
        equal(int((null == yt).sum()), row["null_correct"], "independent SVD probe null")
        residual = full @ coefficients - np.eye(4)[y]
        objective = float(np.square(residual).sum() / len(x) + .001 * np.square(coefficients[1:]).sum())
        recomputed.append(dict(seed=seed, duration=max(durations), job=0, test_lanes=[6, 7],
                               phase="first_legal", label="mapped", method="independent augmented SVD least squares",
                               train_rows=len(x), test_rows=len(xt), rank=int(rank),
                               augmented_condition_number=float(singular[0] / singular[-1]),
                               objective=objective, confusion_true_by_pred=matrix.tolist(),
                               correct=int(matrix.trace()), total=len(yt), null_correct=int((null == yt).sum())))
    equal(sorted(row["seed"] for row in summary["seeds"]), [0, 1, 2], "probe independent seed inventory")
    bindings.add(path, sha(path), "probe protocol inventory")
    bindings.add(folder / "summary.json", sha(folder / "summary.json"), "audited probe summary")
    return dict(audit="passed", fold_count_rows_verified=rows_verified,
                independent_late_stage_svd_recomputations=recomputed,
                scope="Every confusion/count/rate and cross-actor split; an independent late-stage augmented-SVD refit per seed. No world actions, model learning, threshold selection or new research.")


NATURAL_PHASES = ("cue", "processing_mid", "first_legal")


def natural_statistics(rows):
    """Count saved distinct identities, keeping every paired denominator."""
    mature = [row for row in rows if row["matured"]]
    values = dict(issued=len(rows), censored=len(rows) - len(mature), matured=len(mature),
                  completed=sum(row["completed"] for row in mature),
                  clean_completed=sum(row["clean_completed"] for row in mature))
    status_counts = Counter(row["stage_status"] for row in mature)
    for status in ("never_began", "began_too_late", "began_no_observed_legal",
                   "no_observed_processing_tick", "cue_unobserved", "complete_three_stage"):
        values[status] = status_counts[status]
    equal(sum(status_counts.values()), len(mature), "natural stage-status partition")
    for phase in NATURAL_PHASES:
        samples = [row[phase] for row in mature if row[phase] is not None]
        values[phase + "_n"] = len(samples)
        for measure in ("correct", "version_correct", "payload_correct"):
            correct = sum(sample[measure] for sample in samples)
            values[phase + "_" + measure + "_count"] = correct
            values[phase + "_" + measure + "_rate"] = rate(correct, len(samples))
        values[phase + "_mean_correct_rank"] = statistics.mean(point["correct_rank"] for point in samples) if samples else None
        values[phase + "_mean_margin"] = statistics.mean(point["margin"] for point in samples) if samples else None
        actions = Counter("wait" if point["action"] == 0 else "begin" if point["action"] in (1, 2)
                          else "finish" for point in samples)
        for action in ("wait", "begin", "finish"):
            values[phase + "_actual_" + action] = actions[action]
        values[phase + "_prior_rejected_requests"] = sum(point["prior_request_actions"].get("rejected", 0) > 0 for point in samples)
    for early, late in (("cue", "processing_mid"), ("cue", "first_legal"), ("processing_mid", "first_legal")):
        pairs = [(row[early], row[late]) for row in mature if row[early] is not None and row[late] is not None]
        prefix = early + "_to_" + late
        values[prefix + "_n"] = len(pairs)
        for measure in ("correct", "version_correct", "payload_correct"):
            transitions = Counter((bool(first[measure]), bool(last[measure])) for first, last in pairs)
            for key, transition in (("both_correct", (True, True)), ("lost", (True, False)),
                                    ("gained", (False, True)), ("both_wrong", (False, False))):
                values[prefix + "_" + measure + "_" + key] = transitions[transition]
            values[prefix + "_" + measure + "_" + early + "_rate"] = rate(sum(first[measure] for first, _ in pairs), len(pairs))
            values[prefix + "_" + measure + "_" + late + "_rate"] = rate(sum(last[measure] for _, last in pairs), len(pairs))
    triples = Counter("".join(str(int(row[phase]["correct"])) for phase in NATURAL_PHASES)
                      for row in mature if all(row[phase] is not None for phase in NATURAL_PHASES))
    for pattern in range(8):
        name = format(pattern, "03b")
        values["three_stage_correct_" + name] = triples[name]
    equal(sum(triples.values()), status_counts["complete_three_stage"], "natural triple population")
    return values


def check_natural_fork(rows, raw, trace, old_record):
    import numpy as np

    logits, obs = raw["logits"].numpy(), raw["observations"].numpy().reshape(2048, 8, 2, 28)
    before = {key: tensor.numpy() for key, tensor in raw["before"].items()}
    require(logits.shape == (2048, 8, 2, 11) and logits.dtype == np.float64, "natural raw logits")
    require(obs.dtype in (np.float32, np.float64) and np.isfinite(obs).all()
            and np.isfinite(logits).all(), "natural raw observations/finite scores")
    require(len(trace) == 2048, "natural trace length")
    ticks, times = np.array([point["tick"] for point in trace]), np.array([point["time"] for point in trace])
    actions = np.array([point["executed_action"] for point in trace])
    receipts = np.array([point["result"] for point in trace])
    require(np.array_equal(logits.argmax(-1), actions), "natural actual eleven-way argmax")
    require(np.array_equal(before["event_kind"], np.array([point["event_kind"] for point in trace])), "natural event alignment")
    require(np.array_equal(before["request_id"], np.array([point["request_id"] for point in trace])), "natural request alignment")
    require(np.array_equal(before["active"], np.array([point["active_before_action"] for point in trace])), "natural active alignment")
    expected_primary = set()
    regime = old_record["record"]["evaluation_regime_id"]
    commands = np.isin(before["event_kind"], (1, 2)) & before["event_eligible"] & (before["event_regime_id"] == regime)
    for index, lane, job in np.argwhere(commands):
        if lane >= 4:
            expected_primary.add((int(lane), int(job), int(before["request_id"][index, lane, job])))
    actual_primary = {(row["lane"], row["job"], row["request_id"]) for row in rows}
    equal(actual_primary, expected_primary, "natural complete eligible primary issue table")
    point_count = 0
    for row in rows:
        lane, job, identity = row["lane"], row["job"], row["request_id"]
        same = before["request_id"][:, lane, job] == identity
        active = same & before["active"][:, lane, job]
        started = active & before["started"][:, lane, job]
        begin = active & np.isin(actions[:, lane, job], (1, 2)) & (receipts[:, lane, job] == 1)
        begin_indices = np.flatnonzero(begin)
        require(len(begin_indices) <= 1, "natural multiple accepted BEGINs for one identity")
        begin_time = float(times[begin_indices[0]]) if len(begin_indices) else None
        equal(row["accepted_begin_time"], begin_time, "natural accepted BEGIN time")
        duration = row["duration"]
        cue_indices = np.flatnonzero(same & commands[:, lane, job])
        require(len(cue_indices) == 1, "natural one cue per eligible request")
        cue = int(cue_indices[0])
        for name, key in (("raw_payload", "raw_payload"), ("mapped_payload", "payload"),
                          ("version", "version"), ("request_family", "current_family")):
            equal(row[name], int(before[key][cue, lane, job]), "natural request " + name)
        equal(row["mapped_payload"], FAMILY_MAPPINGS[row["request_family"]][row["raw_payload"]], "natural mapping")
        equal(row["target"], 4 * row["version"] + row["mapped_payload"], "natural exact content target")
        ready = before["ready_time"][:, lane, job]
        processing = np.flatnonzero(started & (times < ready - 1e-6))
        legal = np.flatnonzero(started & (times >= ready - 1e-6))
        mid = None if not len(processing) else int(processing[np.argmin(abs(times[processing] - (begin_time + duration / 2)))])
        first = None if not len(legal) else int(legal[0])
        equal(row["processing_observations"], len(processing), "natural all actual processing observations")
        expected_status = ("never_began" if begin_time is None else
                           ("began_too_late" if begin_time + duration > row["deadline"] + 1e-6 else "began_no_observed_legal")
                           if first is None else "no_observed_processing_tick" if mid is None else "complete_three_stage")
        equal(row["stage_status"], expected_status, "natural independently derived missing reason")
        mature = np.flatnonzero(before["matured_mask"][:, lane, job] &
                                (before["matured_request_id"][:, lane, job] == identity))
        equal(row["matured"], bool(len(mature)), "natural actual matured identity")
        if len(mature):
            require(len(mature) == 1, "natural duplicate maturity")
            equal(row["completed"], bool(before["matured_completed"][mature[0], lane, job]), "natural completion")
            equal(row["clean_completed"], bool(before["matured_clean_completed"][mature[0], lane, job]), "natural strict completion")
        for phase, index in (("cue", cue), ("processing_mid", mid), ("first_legal", first)):
            point = row[phase]
            equal(point is None, index is None, "natural phase availability")
            if index is None:
                continue
            scores = logits[index, lane, job, 3:]
            order = np.lexsort((np.arange(8), -scores)).tolist()
            choice, target = order[0], row["target"]
            equal(point["tick"], int(ticks[index]), "natural phase tick")
            equal(point["time"], float(times[index]), "natural phase time")
            equal(point["finish_scores"], scores.tolist(), "natural raw FINISH score vector")
            equal(point["finish_order"], order, "natural exact stable score ranking")
            equal(point["choice"], choice, "natural FINISH choice")
            equal(point["target"], target, "natural phase target")
            equal(point["correct_rank"], order.index(target) + 1, "natural correct target rank")
            equal(point["correct"], choice == target, "natural exact correctness")
            equal(point["version_correct"], choice // 4 == target // 4, "natural version correctness")
            equal(point["payload_correct"], choice % 4 == target % 4, "natural payload correctness")
            equal(point["margin"], float(scores[target] - np.delete(scores, target).max()), "natural correct margin")
            equal(point["action"], int(actions[index, lane, job]), "natural phase actual action")
            equal(point["result"], int(receipts[index, lane, job]), "natural phase actual receipt")
            public = obs[index, lane, job]
            previous = int(public[11:22].argmax()) if public[11:22].any() else 0
            equal(point["previous_public_action"], previous, "natural previous public action")
            equal(point["previous_public_accepted"], bool(public[22]), "natural public accepted feedback")
            equal(point["previous_public_rejected"], bool(public[23]), "natural public rejected feedback")
            past = active & (np.arange(2048) < index)
            prior = point["prior_request_actions"]
            for name, mask in (("positions", np.ones(2048, dtype=bool)),
                               ("begin", np.isin(actions[:, lane, job], (1, 2))),
                               ("finish", actions[:, lane, job] >= 3),
                               ("accepted", receipts[:, lane, job] == 1),
                               ("rejected", receipts[:, lane, job] == -1),
                               ("accepted_begin", begin),
                               ("accepted_finish", (actions[:, lane, job] >= 3) & (receipts[:, lane, job] == 1))):
                equal(prior.get(name, 0), int((past & mask).sum()), "natural pre-stage action history " + name)
            if begin_time is not None and phase != "cue":
                equal(point["processing_age"], float(times[index] - begin_time), "natural processing origin")
                equal(point["processing_fraction"], float((times[index] - begin_time) / duration), "natural processing fraction")
            if phase == "first_legal":
                require(times[index] <= before["deadline"][index, lane, job] + 1e-6, "natural legal deadline")
            point_count += 1
    mature_rows = [row for row in rows if row["matured"]]
    identities = sorted((row["case_id"], row["job"], row["request_id"]) for row in mature_rows)
    fingerprint = hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest()
    equal(fingerprint, old_record["shadow"]["heldout"]["primary_request_set_sha256"], "natural old matured request set")
    equal(len(mature_rows), old_record["shadow"]["heldout"]["counts"]["primary_matured"], "natural old matured denominator")
    return point_count


def audit_natural(bindings):
    folder = ROOT / "data/content-timing/natural-v1"
    plan, summary = read(folder / "protocol.json"), read(folder / "summary.json")
    check_digest(plan, "protocol_hash")
    check_digest(summary, "summary_hash")
    bindings.discover(plan, "natural protocol")
    equal(plan["protocol_hash"], summary["protocol_hash"], "natural summary protocol")
    total_requests = total_samples = total_forks = 0
    all_rows = []
    for seed in range(3):
        path = folder / f"seed_{seed:03d}/result.json"
        result = read(path)
        check_digest(result, "result_hash")
        equal(result["protocol_hash"], plan["protocol_hash"], "natural result protocol")
        require(result["source_unchanged"], "natural source state unchanged")
        bindings.mapping(result["bindings"], "natural result artifacts")
        bindings.add(path, summary["results"][str(path)], "natural summary result")
        rows = [json.loads(line) for line in (path.parent / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
        old = read(ROOT / f"data/action-return/refit-policy-v1/seed_{seed:03d}/result.json")
        require(len(result["records"]) == 8, "natural eight forks per seed")
        for record in result["records"]:
            condition, family = record["condition"], record["family"]
            anchor = next(item for item in old["records"] if item["arm"] == "own_policy"
                          and item["condition"] == condition and item["family"] == family)
            equal(record["record"], anchor["record"], "natural whole archived evaluator equality")
            equal(record["shadow"], anchor["shadow"], "natural whole archived shadow equality")
            stem = path.parent / f"{condition}_family_{family}"
            raw = load_tensor(str(stem) + ".raw.pt")
            trace = json.loads(gzip.decompress(Path(str(stem) + ".trace.json.gz").read_bytes()))
            subset = [row for row in rows if row["condition"] == condition and row["fork_family"] == family]
            total_samples += check_natural_fork(subset, raw, trace, anchor)
            total_forks += 1
        all_rows.extend(rows)
        total_requests += len(rows)
    table_rows = 0
    for key in ("per_seed", "per_fork"):
        for row in summary[key]:
            subset = [r for r in all_rows if r["seed"] == row["seed"] and r["condition"] == row["condition"]
                      and (key != "per_fork" or r["fork_family"] == row["family"])]
            computed = natural_statistics(subset)
            for field, value in computed.items():
                equal(row[field], value, "natural table " + field)
            table_rows += 1
    for condition, values in summary["seed_rates"].items():
        selected = [row for row in summary["per_seed"] if row["condition"] == condition]
        require(len(selected) == 3, "natural three independent seed rates")
        for field, stats in values.items():
            equal(stats["mean"], statistics.mean(row[field] for row in selected), "natural three-seed mean")
            equal(stats["sample_sd"], statistics.stdev(row[field] for row in selected), "natural three-seed sample SD")
    bindings.add(folder / "protocol.json", sha(folder / "protocol.json"), "natural protocol inventory")
    bindings.add(folder / "summary.json", sha(folder / "summary.json"), "audited natural summary")
    return dict(audit="passed", forks=total_forks, issued_requests_reconstructed=total_requests,
                stage_samples_recomputed=total_samples, aggregate_table_rows_verified=table_rows,
                checks=["All original evaluator/shadow fields equal archived own-policy results",
                        "Eligible primary issue inventory and matured identities reconstructed from raw pre-action metadata",
                        "Every cue, closest processing midpoint and first legal tick selected independently",
                        "Every FINISH rank, margin, exact/version/payload correctness and pre-stage receipt history recomputed",
                        "Every marginal, matched transition, missing-status partition and three-seed mean/sample SD recomputed"])


MATCHED_STAGES = ("cue", "midpoint", "common_age2", "first_legal", "teacher_first", "common_short_teacher")


def audit_matched(bindings):
    import numpy as np

    folder = ROOT / "data/content-timing/matched-v1"
    plan, summary = read(folder / "protocol.json"), read(folder / "summary.json")
    check_digest(plan, "protocol_hash")
    equal(summary["protocol_hash"], plan["protocol_hash"], "matched summary protocol")
    bindings.discover(plan, "matched protocol")
    for source in plan["sources"].values():
        bindings.add(source["model"], source["model_sha256"], "matched deployed model")
        bindings.add(source["parent_result"], source["parent_result_sha256"], "matched historical result")
    paired = read(folder / "paired_scores.json")
    timing = read(folder / "matched_timing.json")
    equal(paired["protocol_hash"], plan["protocol_hash"], "matched paired table protocol")
    equal(timing["protocol_hash"], plan["protocol_hash"], "matched timing table protocol")
    bindings.add(folder / "paired_scores.json", summary["paired_scores_sha256"], "matched paired score table")
    bindings.add(folder / "matched_timing.json", summary["matched_timing_sha256"], "matched timing table")
    bindings.mapping(summary["bindings"], "matched summary artifacts")
    all_data, anchor_total, raw_total, score_checks, summary_rows = {}, 0, 0, 0, 0
    invariant_cue_anchors = 0
    for seed in range(3):
        path = folder / f"seed_{seed:03d}/result.json"
        result = read(path)
        check_digest(result, "result_hash")
        equal(result["protocol_hash"], plan["protocol_hash"], "matched result protocol")
        require(result["source_unchanged"], "matched source endpoint unchanged")
        bindings.add(result["records_path"], result["records_sha256"], "matched saved rows")
        data = load_tensor(result["records_path"])
        equal(data["protocol_hash"], plan["protocol_hash"], "matched raw protocol")
        rows = data["rows"]
        scores, x = data["scores11"].numpy(), data["x24"].numpy()
        offsets = data["finish_offsets8"].numpy()
        require(scores.dtype == x.dtype == offsets.dtype == np.float64, "matched FP64 samples")
        require(scores.shape == (len(rows), 11) and x.shape == (len(rows), 24), "matched sample shapes")
        require(np.isfinite(scores).all() and np.isfinite(x).all() and np.isfinite(offsets).all(), "matched finite states/scores")
        require(np.array_equal(scores[:, 3:], data["finish_scores8"].numpy()), "matched FINISH slice")
        model = load_tensor(plan["sources"][str(seed)]["model"])
        indices = model["begin_head.indices"].tolist()
        equal(indices, data["non_input_indices"], "matched feature node order")
        require(len(indices) == len(set(indices)) == 24, "matched 24 distinct non-input nodes")
        equal(data["source_controller_hash"], semantic_digest(model), "matched controller state hash")
        input_ports = next(value.tolist() for key, value in model.items() if key.endswith("input_ports"))
        equal(indices, sorted(set(range(80)) - set(input_ports)), "matched excludes all56 input ports")
        anchors = {}
        for background in result["background"]:
            bindings.add(background["path"], background["sha256"], "matched real continuous background")
            history = load_tensor(background["path"])
            equal(len(history["anchors"]), 48, "matched background anchor quota")
            equal(history["source_controller_hash"], data["source_controller_hash"], "matched background controller")
            equal(history["observations"].shape[0], history["ticks"], "matched background steps")
            quota = Counter()
            for anchor in history["anchors"]:
                identity = anchor["anchor_id"]
                require(identity not in anchors, "matched duplicate anchor identity")
                anchors[identity] = anchor
                quota[(anchor["actor_lane"], anchor["job"], anchor["version"])] += 1
                equal(anchor["source_fingerprint"], semantic_digest(dict(world=anchor["world"], h_before=anchor["h_before"], observation=anchor["obs"])), "matched actual anchor fingerprint")
                require(np.array_equal(anchor["obs"].numpy(), history["observations"][anchor["source_index"]].numpy()), "matched source observation at anchor")
                require(anchor["next_external_time"] > anchor["cue_time"] + 8 + anchor["window"] + 1e-6, "matched unmodified scheduler exclusion window")
                tensors = anchor["world"]["tensors"]
                i, j = anchor["actor_lane"], anchor["job"]
                equal(int(tensors["request_id"][i, j]), anchor["request_id"], "matched real request identity")
                equal(int(tensors["version"][i, j]), anchor["version"], "matched real version")
                require(bool(tensors["event_eligible"][i, j]), "matched real eligible anchor")
            require(len(quota) == 16 and set(quota.values()) == {3}, "matched balanced lane/job/version quota")
        equal(len(anchors), result["anchors"], "matched anchor inventory")
        equal(len(rows), len(anchors) * 2 * 4 * 6, "matched complete factorial row count")
        groups = defaultdict(dict)
        marginals = defaultdict(list)
        for index, row in enumerate(rows):
            equal(row["row_id"], index, "matched tensor row index")
            anchor = anchors[row["anchor_id"]]
            for field in ("seed", "family", "actor_lane", "actor_id", "job", "case_id", "request_id", "cue_kind", "version", "source_raw_payload", "source_fingerprint"):
                equal(row[field], anchor[field], "matched unaltered anchor metadata " + field)
            duration = row["duration"]
            require(duration in (2., 8.), "matched duration arm")
            equal(row["duration_arm"], "short" if duration == 2 else "long", "matched duration label")
            equal(row["mapped_payload"], FAMILY_MAPPINGS[row["family"]][row["raw_payload"]], "matched mapped payload")
            target = 4 * row["version"] + row["mapped_payload"]
            equal(row["target8"], target, "matched eight-class target")
            rank = np.lexsort((np.arange(8), -scores[index, 3:]))
            equal(data["rank8"][index].tolist(), rank.tolist(), "matched stable ranking")
            equal(row["conditional_choice8"], int(rank[0]), "matched conditional winner")
            equal(row["content_correct"], rank[0] == target, "matched exact correctness")
            equal(row["version_correct"], rank[0] // 4 == row["version"], "matched version correctness")
            equal(row["payload_correct"], rank[0] % 4 == row["mapped_payload"], "matched payload correctness")
            equal(row["model_action"], int(scores[index].argmax()), "matched measured full argmax")
            equal(row["finish_vs_nonfinish_margin"], float(scores[index, 3:].max() - scores[index, :3].max()), "matched FINISH gate margin")
            age = {"cue": 0., "midpoint": duration / 2, "common_age2": 2., "first_legal": duration,
                   "teacher_first": math.ceil((duration + anchor["window"] / 2 - 1e-6) / .25) * .25,
                   "common_short_teacher": math.ceil((2 + anchor["window"] / 2 - 1e-6) / .25) * .25}[row["stage"]]
            equal(row["age"], age, "matched independently specified stage age")
            equal(row["now"], anchor["cue_time"] + age, "matched absolute stage time")
            equal(row["tick"], anchor["cue_tick"] + round(age / .25), "matched stage tick")
            equal(row["issued_time"], anchor["cue_time"], "matched issue time")
            equal(row["deadline"], anchor["cue_time"] + duration + anchor["window"], "matched manipulated deadline")
            require(row["active"] and row["target_external_event_free"], "matched still-live unmodified request")
            equal(row["started"], age > 0, "matched actual locked-BEGIN processing state")
            equal(row["ready_time"], anchor["cue_time"] + duration if age > 0 else float("inf"), "matched actual ready time")
            equal(row["legal"], age >= duration - 1e-6, "matched actual legal stage")
            equal(row["teacher_finish_expected"], age >= (duration + anchor["window"] / 2) - 1e-6, "matched teacher timing separate from first legal")
            if row["stage"] == "first_legal":
                require(not row["previous_observation_legal"] and row["previous_observation_time"] < row["ready_time"] - 1e-6, "matched first observed legal boundary")
            group = (row["anchor_id"], duration, row["stage"])
            require(row["raw_payload"] not in groups[group], "matched duplicate content cell")
            groups[group][row["raw_payload"]] = index
            marginals[(row["family"], row["duration_arm"], row["stage"])].append(index)
        require(all(set(members) == set(range(4)) for members in groups.values()), "matched every content group is balanced")
        for identity in anchors:
            cue_indices = [index for index, row in enumerate(rows)
                           if row["anchor_id"] == identity and row["stage"] == "cue"]
            equal(len(cue_indices), 8, "matched complete cue content/duration inventory")
            cue_states, cue_scores = x[cue_indices], scores[cue_indices]
            require(np.array_equal(cue_states, np.broadcast_to(cue_states[0], cue_states.shape)),
                    "matched structural immediate non-input cue invariance")
            require(np.array_equal(cue_scores, np.broadcast_to(cue_scores[0], cue_scores.shape)),
                    "matched structural immediate action-score cue invariance")
            invariant_cue_anchors += 1
        for job in range(2):
            selected = np.array([row["job"] == job for row in rows])
            predicted_offsets = ((x[selected] - model[f"j{job}_mean"].numpy()) / model[f"j{job}_std"].numpy()) @ model[f"j{job}_coef"].numpy().T + model[f"j{job}_bias"].numpy()
            require(np.allclose(predicted_offsets, offsets[selected], atol=1e-10, rtol=0), "matched saved affine offsets from frozen coefficients")
            score_checks += int(selected.sum())
        for trace in result["traces"]:
            anchor = anchors[trace["anchor_id"]]
            for key in ("public_observation_equality", "non_target_source_observation_equality", "target_begin_accepted", "target_external_event_free", "all_model_argmax_actions_measurement_only"):
                require(trace[key], "matched recorded execution assertion " + key)
            equal(trace["first_legal_ticks"], [8] * 4 + [32] * 4, "matched all actual first-legal ticks")
            expected_teacher = [math.ceil((duration + anchor["window"] / 2 - 1e-6) / .25) for duration in (2., 8.) for _ in range(4)]
            equal(trace["first_teacher_ticks"], expected_teacher, "matched all actual teacher-first ticks")
            scheduler = semantic_digest(anchor["world"]["scheduler"])
            require(len(trace["variants"]) == 8, "matched eight treatment hashes")
            for variant in trace["variants"]:
                equal(variant["scheduler_hash"], scheduler, "matched initial scheduler unchanged")
        for row in (item for item in summary["rows"] if item["seed"] == seed):
            members = [rows[index] for index in marginals[(row["family"], row["duration_arm"], row["stage"])]]
            equal(row["n"], len(members), "matched summary denominator")
            for key in ("content_correct", "version_correct", "payload_correct", "legal", "teacher_finish_expected"):
                equal(row[key], sum(item[key] for item in members), "matched summary " + key)
            actual = Counter("WAIT" if item["model_action"] == 0 else "BEGIN" if item["model_action"] < 3 else "FINISH" for item in members)
            equal(row["model_action_counts"], dict(actual), "matched measured action distribution")
            equal(row["mean_finish_vs_nonfinish_margin"], statistics.mean(item["finish_vs_nonfinish_margin"] for item in members), "matched mean gate margin")
            summary_rows += 1
        all_data[seed] = data
        anchor_total += len(anchors)
        raw_total += len(rows)
    equal(summary_rows, 3 * 4 * 2 * 6, "matched complete summary row inventory")
    pair_totals = defaultdict(list)
    seen_pairs = set()
    for pair in paired["rows"]:
        data = all_data[pair["seed"]]
        a, z = pair["row_a"], pair["row_b"]
        first, last = data["rows"][a], data["rows"][z]
        key = (pair["seed"], pair["anchor_id"], pair["duration_arm"], pair["stage"], pair["raw_a"], pair["raw_b"])
        require(key not in seen_pairs, "matched duplicate symmetric pair")
        seen_pairs.add(key)
        for field in ("anchor_id", "duration_arm", "stage", "family", "actor_lane", "job", "version"):
            equal(first[field], last[field], "matched pair same background " + field)
            equal(pair[field], first[field], "matched pair identity " + field)
        equal(first["raw_payload"], pair["raw_a"], "matched first raw symbol")
        equal(last["raw_payload"], pair["raw_b"], "matched second raw symbol")
        require(pair["raw_a"] < pair["raw_b"], "matched unordered symbol pair")
        scores = data["finish_scores8"].numpy()
        ta, tz = first["target8"], last["target8"]
        ma, mz = float(scores[a, ta] - scores[a, tz]), float(scores[z, ta] - scores[z, tz])
        delta = ma - mz
        for name, value in (("target_a", ta), ("target_b", tz), ("relative_score_a", ma),
                            ("relative_score_b", mz), ("matched_difference", delta),
                            ("symmetric_pair_order_correct", delta > 0),
                            ("both_targets_outrank_alternative", ma > 0 and mz < 0),
                            ("conditional_argmax_distinct", first["conditional_choice8"] != last["conditional_choice8"]),
                            ("both_argmax_correct", first["content_correct"] and last["content_correct"])):
            equal(pair[name], value, "matched independent pair " + name)
        pair_totals[(pair["seed"], pair["duration_arm"], pair["stage"])].append(pair)
    equal(len(seen_pairs), anchor_total * 2 * 6 * 6, "matched complete six-pair inventory")
    for row in summary["paired"]:
        members = pair_totals[(row["seed"], row["duration_arm"], row["stage"])]
        equal(row["pairs"], len(members), "matched pair denominator")
        equal(row["mean_matched_difference"], statistics.mean(item["matched_difference"] for item in members), "matched mean paired differential")
        for output, raw_name in (("pair_order_correct", "symmetric_pair_order_correct"),
                                 ("both_targets_outrank_alternative", "both_targets_outrank_alternative"),
                                 ("conditional_argmax_distinct", "conditional_argmax_distinct"), ("both_argmax_correct", "both_argmax_correct")):
            equal(row[output], sum(item[raw_name] for item in members), "matched paired summary " + output)
    seen_timing = set()
    for row in timing["rows"]:
        data = all_data[row["seed"]]
        first, last = data["rows"][row["short_row"]], data["rows"][row["long_row"]]
        key = (row["seed"], row["anchor_id"], row["raw_payload"], row["stage"])
        require(key not in seen_timing, "matched duplicate common-age comparison")
        seen_timing.add(key)
        for field in ("anchor_id", "raw_payload", "stage", "age"):
            equal(first[field], last[field], "matched common-age identity " + field)
            equal(row[field], first[field], "matched timing row " + field)
        require(first["legal"] and not last["legal"], "matched common-age short legal versus long processing")
        for prefix, point in (("short", first), ("long", last)):
            for output, raw_name in (("action", "model_action"), ("legal", "legal"), ("teacher_action", "teacher_action")):
                equal(row[prefix + "_" + output], point[raw_name], "matched common-age actual quantity")
        equal(row["short_minus_long_finish_margin"], first["finish_vs_nonfinish_margin"] - last["finish_vs_nonfinish_margin"], "matched common-age gate contrast")
    equal(len(seen_timing), anchor_total * 4 * 2, "matched complete common-age timing inventory")
    bindings.add(folder / "protocol.json", sha(folder / "protocol.json"), "matched protocol inventory")
    bindings.add(folder / "summary.json", sha(folder / "summary.json"), "audited matched summary")
    return dict(audit="passed", independent_seeds=3, anchors=anchor_total, raw_stage_rows=raw_total,
                affine_score_rows_recomputed=score_checks, summary_rows_verified=summary_rows,
                symmetric_pairs_recomputed=len(seen_pairs), common_age_pairs_recomputed=len(seen_timing),
                exactly_invariant_immediate_cue_anchors=invariant_cue_anchors,
                scope="Saved actual anchor fingerprints/quotas and scheduler exclusions, all raw sample fields, frozen affine scores, complete factorial/pair inventories and summary arithmetic. Recorded execution-equality flags are verified as saved assertions; the audit does not rerun world dynamics.")


def transfer_confusion(predictions):
    import numpy as np

    target = np.array([row["target"] for row in predictions], dtype=np.int64)
    choice = np.array([row["choice"] for row in predictions], dtype=np.int64)
    matrix = matrix_from_predictions(target, choice, 4)
    denominators, correct = matrix.sum(1), matrix.diagonal()
    recalls = [float(a / b) if b else None for a, b in zip(correct, denominators)]
    return dict(correct=int(correct.sum()), total=len(predictions),
                accuracy=rate(int(correct.sum()), len(predictions)),
                confusion_true_by_pred=matrix.tolist(), class_denominators=denominators.tolist(),
                class_correct=correct.tolist(), class_recall=recalls,
                macro_recall=statistics.mean(value for value in recalls if value is not None) if predictions else None,
                observed_classes=int((denominators > 0).sum()),
                majority_class_count=int(denominators.max()),
                majority_class_proportion=rate(int(denominators.max()), len(predictions)))


def audit_transfer(bindings):
    """Audit identities and saved predictions; do not repeat continuous replay."""
    import numpy as np

    folder = ROOT / "data/content-timing/transfer-v1"
    plan, summary = read(folder / "protocol.json"), read(folder / "summary.json")
    check_digest(plan, "protocol_hash")
    check_digest(summary, "summary_hash")
    equal(summary["protocol_hash"], plan["protocol_hash"], "transfer summary protocol")
    amendment_path = folder / "replay_correction.json"
    amendment = read(amendment_path)
    check_digest(amendment, "amendment_hash")
    equal(amendment["protocol_hash"], plan["protocol_hash"], "transfer replay serialization correction protocol")
    bindings.discover(amendment, "transfer serialization-only correction")
    bindings.add(amendment_path, sha(amendment_path), "transfer correction inventory")
    bindings.discover(plan, "transfer protocol")
    bindings.mapping(summary["results"], "transfer summary results")
    predictions_verified = table_rows = replay_forks = 0
    combined = {"per_seed": [], "per_fold": [], "per_family": []}
    for seed in range(3):
        path = folder / f"seed_{seed:03d}/result.json"
        result = read(path)
        check_digest(result, "result_hash")
        equal(result["protocol_hash"], plan["protocol_hash"], "transfer result protocol")
        bindings.discover(result, "transfer seed result")
        replay = read(result["replay_result"]["path"])
        check_digest(replay, "result_hash")
        equal(replay["serialization_correction_hash"], amendment["amendment_hash"], "transfer corrected replay provenance")
        require(replay["source_unchanged"] and len(replay["checks"]) == 8, "transfer unchanged source/eight replays")
        bindings.mapping(replay["artifacts"], "transfer continuous features")
        for check in replay["checks"]:
            require(check["steps"] == 2048 and check["all11_logits_exact"] and check["all_actions_exact"]
                    and check["every_hidden_hash_exact"], "transfer recorded exact continuous replay")
            feature_path = path.parent / f"{check['condition']}_family_{check['family']}.features.pt"
            features = load_tensor(feature_path)
            equal(features["protocol_hash"], plan["protocol_hash"], "transfer feature protocol")
            equal(features["natural_raw_sha256"], check["source_raw_sha256"], "transfer raw/feature binding")
            require(features["x24"].shape == (2048, 8, 24) and str(features["x24"].dtype) == "torch.float64", "transfer continuous FP64 feature shape")
            replay_forks += 1
        natural = [json.loads(line) for line in (ROOT / f"data/content-timing/natural-v1/seed_{seed:03d}/requests.jsonl").read_text(encoding="utf-8").splitlines()]
        natural = [row for row in natural if row["matured"]]
        by_identity = {(row["condition"], row["fork_family"], row["case_id"], row["job"], row["request_id"]): row for row in natural}
        predictions = [json.loads(line) for line in Path(result["prediction_artifact"]["path"]).read_text(encoding="utf-8").splitlines()]
        seen = set()
        for prediction in predictions:
            key = (prediction["condition"], prediction["fork_family"], prediction["case_id"], prediction["job"], prediction["request_id"])
            require(key in by_identity, "transfer natural matured request identity")
            row = by_identity[key]
            phase, label = prediction["phase"], prediction["label"]
            require(phase in NATURAL_PHASES and label in ("raw", "mapped") and row[phase] is not None, "transfer available stage/label")
            complete = all(row[item] is not None for item in NATURAL_PHASES)
            equal(prediction["complete_three_stage"], complete, "transfer fixed common request population")
            equal(prediction["lane"], row["lane"], "transfer actual actor lane")
            equal(prediction["actor_id"], row["actor_id"], "transfer actual actor identity")
            equal(prediction["request_family"], row["request_family"], "transfer actual request family")
            expected_fold = [4, 5] if row["lane"] in (4, 5) else [6, 7]
            equal(prediction["test_lanes"], expected_fold, "transfer actor-heldout fold")
            equal(prediction["duration"], 2 if row["condition"] == "short" else 8, "transfer fixed duration mapping")
            target = row["raw_payload"] if label == "raw" else row["mapped_payload"]
            equal(prediction["target"], target, "transfer scorer-only target label")
            scores = np.array(prediction["scores"])
            require(scores.shape == (4,) and np.isfinite(scores).all(), "transfer finite four-score output")
            equal(prediction["choice"], int(scores.argmax()), "transfer fixed score argmax")
            full_key = key + (phase, label)
            require(full_key not in seen, "transfer no duplicate request across actor folds")
            seen.add(full_key)
        expected = {key + (phase, label) for key, row in by_identity.items()
                    for phase in NATURAL_PHASES if row[phase] is not None for label in ("raw", "mapped")}
        equal(seen, expected, "transfer exact all-available natural inventory")
        matched = load_tensor(result["matched_artifact"]["path"])
        _, meta, durations, states = probe_states(matched)
        lanes, jobs, families = (np.array([item[index] for item in meta]) for index in range(3))
        for fit in result["fits"]:
            training = (jobs == fit["job"]) & ~np.isin(lanes, fit["test_lanes"])
            require(set(lanes[training]).isdisjoint(fit["test_lanes"]), "transfer no training actor leakage")
            require(set(families[training]) == set(range(4)), "transfer all training families")
            phase = NATURAL_PHASES.index(fit["phase"])
            x = states[training, durations.index(fit["duration"]), :, phase, :].reshape(-1, 24)
            targets = np.broadcast_to(np.arange(4), (len(meta), 4)) if fit["label"] == "raw" else np.array(FAMILY_MAPPINGS)[families]
            y = targets[training].reshape(-1)
            equal(fit["training_anchors"], int(training.sum()), "transfer training anchors")
            equal(fit["training_positions"], len(x), "transfer training positions")
            equal(fit["training_class_counts"], np.bincount(y, minlength=4).tolist(), "transfer training class balance")
            equal(fit["training_feature_hash"], hashlib.sha256(x.tobytes()).hexdigest(), "transfer exact matched train features")
            equal(fit["training_target_hash"], hashlib.sha256(y.tobytes()).hexdigest(), "transfer exact matched train labels")
        for source_key, summary_key in (("folds", "per_fold"), ("per_seed", "per_seed"), ("per_family", "per_family")):
            for row in result[source_key]:
                selected = [p for p in predictions if p["condition"] == row["condition"] and p["phase"] == row["phase"]
                            and p["label"] == row["label"] and (row["population"] == "all_available" or p["complete_three_stage"])
                            and (source_key != "folds" or (p["job"] == row["job"] and p["test_lanes"] == row["test_lanes"]))
                            and (source_key != "per_family" or p["fork_family"] == row["family"])]
                for name, value in transfer_confusion(selected).items():
                    equal(row[name], value, "transfer independent confusion/statistic " + name)
                table_rows += 1
            combined[summary_key].extend(result[source_key])
        predictions_verified += len(predictions)
    for key, expected in combined.items():
        equal(summary[key], expected, "transfer complete summary " + key)
    for row in summary["aggregate"]:
        selected = [item for item in combined["per_seed"] if all(item[field] == row[field] for field in ("condition", "phase", "label", "population"))]
        require(len(selected) == 3, "transfer aggregate three independent seeds")
        for metric in ("accuracy", "macro_recall", "majority_class_proportion"):
            equal(row[metric + "_mean"], statistics.mean(item[metric] for item in selected), "transfer seed mean")
            equal(row[metric + "_sample_sd"], statistics.stdev(item[metric] for item in selected), "transfer seed sample SD")
    bindings.add(folder / "protocol.json", sha(folder / "protocol.json"), "transfer protocol inventory")
    bindings.add(folder / "summary.json", sha(folder / "summary.json"), "audited transfer summary")
    return dict(audit="passed", distinct_stage_label_predictions_verified=predictions_verified,
                confusion_table_rows_recomputed=table_rows, saved_continuous_replay_forks_checked=replay_forks,
                replay_serialization_correction_verified=True,
                scope="Every natural matured identity, available phase, raw/mapped label, actor fold, complete-three-stage population, score argmax, matched training-feature/label hash, confusion table and three-seed summary. Saved exact-replay flags/bindings checked; no new replay or readout fitting.")


def audit_onset(bindings):
    import numpy as np

    folder = ROOT / "data/content-timing/onset-v1"
    plan, summary = read(folder / "protocol.json"), read(folder / "summary.json")
    check_digest(plan, "protocol_hash")
    equal(summary["protocol_hash"], plan["protocol_hash"], "onset summary protocol")
    bindings.mapping(plan["pins"], "onset frozen source/input pins")
    bindings.discover(plan["parent_protocol"], "onset parent protocol")
    groups_verified = score_rows = fold_rows = 0
    for seed_summary in summary["seeds"]:
        seed = seed_summary["seed"]
        path = folder / f"seed_{seed:03d}/result.json"
        result, data = read(path), load_tensor(path.parent / "records.pt")
        check_digest(result, "result_hash")
        equal(result["protocol_hash"], plan["protocol_hash"], "onset seed protocol")
        equal(data["protocol_hash"], plan["protocol_hash"], "onset raw protocol")
        for key, value in seed_summary.items():
            equal(result[key], value, "onset summary/result " + key)
        bindings.add(path.parent / "records.pt", result["raw_sha256"], "onset saved samples")
        bindings.add(path, sha(path), "onset audited result")
        require(result["controller_unchanged"] and result["cue_bitwise_alignment_all"], "onset saved execution checks")
        matched = load_tensor(ROOT / f"data/content-timing/matched-v1/seed_{seed:03d}/records.pt")
        x, h = data["x24"].numpy(), data["h80"].numpy()
        scores, cue, obs = (data[key].numpy() for key in ("scores11", "cue_x24", "model_obs56"))
        rows, indices = data["rows"], data["non_input_indices"]
        require(x.shape == cue.shape == (768, 24) and h.shape == (768, 80), "onset full sample dimensions")
        require(x.dtype == scores.dtype == h.dtype == np.float64, "onset FP64 states/scores")
        require(np.array_equal(x, h[:, indices]), "onset non-input feature slice")
        require(np.array_equal(scores[:, 3:], data["finish_scores8"].numpy()), "onset FINISH score slice")
        grouped = defaultdict(dict)
        for index, row in enumerate(rows):
            old_index = row["matched_cue_row_id"]
            old_row = matched["rows"][old_index]
            for field in ("anchor_id", "raw_payload", "mapped_payload", "target8", "version", "family", "job", "actor_lane", "case_id", "request_id", "duration", "treatment_fingerprint"):
                equal(row[field], old_row[field], "onset unchanged real treatment " + field)
            require(old_row["stage"] == "cue" and old_row["duration"] == 8., "onset fixed source duration/stage")
            require(np.array_equal(cue[index], matched["x24"][old_index].numpy()), "onset exact matched cue-state alignment")
            require(np.array_equal(data["cue_scores11"][index].numpy(), matched["scores11"][old_index].numpy()), "onset exact matched cue-score alignment")
            equal(row["stage"], "onset1", "onset phase")
            equal(row["age"], .25, "onset single-tick age")
            equal(row["tick"], old_row["tick"] + 1, "onset absolute tick")
            equal(row["now"], old_row["now"] + .25, "onset absolute time")
            require(row["active"] and row["started"] and not row["legal"], "onset actual processing status")
            equal(row["ready_time"], row["issued_time"] + 8., "onset actual accepted BEGIN readiness")
            target_obs = obs[index].reshape(2, 28)[row["job"]]
            require(not target_obs[:11].any() and not target_obs[24:].any(), "onset no repeated target cue")
            action = target_obs[11:22]
            require(action.sum() == 1 and action[1 + row["version"]] == 1
                    and target_obs[22] == 1 and target_obs[23] == 0, "onset fixed accepted-BEGIN public receipt")
            choice = int(scores[index, 3:].argmax())
            equal(row["conditional_choice8"], choice, "onset conditional FINISH winner")
            equal(row["model_action"], int(scores[index].argmax()), "onset full argmax")
            for key, value in (("content_correct", choice == row["target8"]),
                               ("version_correct", choice // 4 == row["version"]),
                               ("payload_correct", choice % 4 == row["mapped_payload"])):
                equal(row[key], value, "onset measured " + key)
            equal(row["finish_vs_nonfinish_margin"], float(scores[index, 3:].max() - scores[index, :3].max()), "onset measured FINISH gate margin")
            require(row["raw_payload"] not in grouped[row["anchor_id"]], "onset duplicate raw variant")
            grouped[row["anchor_id"]][row["raw_payload"]] = index
        equal(len(grouped), 192, "onset unchanged anchor population")
        propagation = {row["anchor_id"]: row for row in result["propagation"]}
        counts = Counter()
        for identity, members in grouped.items():
            require(set(members) == set(range(4)), "onset complete balanced content block")
            selected = [members[raw] for raw in range(4)]
            states, prior = x[selected], cue[selected]
            require(np.array_equal(prior, np.broadcast_to(prior[0], prior.shape)), "onset invariant immediate cue state")
            require(np.array_equal(obs[selected], np.broadcast_to(obs[selected[0]], obs[selected].shape)), "onset identical post-cue public observations")
            differences = states[:, None] - states[None, :]
            distances = np.sqrt(np.square(differences).sum(-1))
            point = propagation[identity]
            nonzero, distinct = bool(np.any(differences != 0)), len({tuple(vector) for vector in states}) == 4
            equal(point["onset_nonzero"], nonzero, "onset first-propagation flag")
            equal(point["onset_all_four_distinct"], distinct, "onset all-four distinction")
            equal(point["max_pair_l2"], float(distances.max()), "onset propagation L2 scale")
            equal(point["min_pair_l2"], float(distances[np.triu_indices(4, 1)].min()), "onset minimum pair L2 scale")
            equal(point["max_abs_difference"], float(abs(differences).max()), "onset propagation max absolute scale")
            counts["cue"] += 1
            counts["nonzero"] += nonzero
            counts["distinct"] += distinct
            groups_verified += 1
        for field, counter in (("groups_cue_invariant", "cue"), ("groups_onset_nonzero", "nonzero"), ("groups_onset_four_distinct", "distinct")):
            equal(result[field], counts[counter], "onset aggregate propagation counts")
        for row in result["probe"]["folds"]:
            matrix = np.array(row["confusion_true_by_pred"])
            require(matrix.shape == (4, 4) and np.all(matrix >= 0), "onset probe confusion")
            equal(int(matrix.trace()), row["correct"], "onset fold correct numerator")
            equal(int(matrix.sum()), row["total"], "onset fold denominator")
            require(np.all(matrix.sum(1) == row["test_anchors"]), "onset fold balanced target labels")
            equal(row["null_correct"] * 4, row["total"], "onset fold exactly25percent paired null")
            fold_rows += 1
        for row in result["probe"]["pooled"]:
            folds = [item for item in result["probe"]["folds"] if item["label"] == row["label"]]
            for field in ("correct", "total", "null_correct"):
                equal(row[field], sum(item[field] for item in folds), "onset pooled probe " + field)
            equal(row["accuracy"], row["correct"] / row["total"], "onset pooled probe accuracy")
        score_rows += len(rows)
    bindings.add(folder / "protocol.json", sha(folder / "protocol.json"), "onset protocol inventory")
    bindings.add(folder / "summary.json", sha(folder / "summary.json"), "audited onset summary")
    return dict(audit="passed", original_anchor_groups_recomputed=groups_verified, raw_rows_verified=score_rows,
                confusion_folds_verified=fold_rows,
                scope="Exact original cue-state/score correspondence, unchanged request/treatment population, actual one-tick metadata and fixed BEGIN receipts, full non-input-state slices, every within-anchor propagation flag/distance and saved probe-confusion arithmetic. No dynamics replay or new probe fitting.")


def audit_natural_probe(bindings):
    import numpy as np

    folder = ROOT / "data/content-timing/natural-probe-v1"
    plan, summary = read(folder / "protocol.json"), read(folder / "summary.json")
    equal(summary["protocol_sha256"], sha(folder / "protocol.json"), "natural calibration protocol SHA")
    bindings.mapping(plan["inputs"], "natural calibration source/input pins")
    fold_count = summary_count = 0
    for result in summary["seeds"]:
        seed = result["seed"]
        replay_path = ROOT / f"data/content-timing/transfer-v1/seed_{seed:03d}/replay.json"
        equal(result["replay_sha256"], sha(replay_path), "natural calibration continuous-feature replay binding")
        natural = [json.loads(line) for line in (ROOT / f"data/content-timing/natural-v1/seed_{seed:03d}/requests.jsonl").read_text(encoding="utf-8").splitlines()]
        natural = [row for row in natural if row["matured"] and row["stage_status"] == "complete_three_stage"]
        groups = defaultdict(list)
        for row in result["folds"]:
            chosen = [point for point in natural if point["condition"] == row["condition"]]
            test = [point for point in chosen if point["job"] == row["job"] and point["lane"] in row["heldout"]]
            train = [point for point in chosen if point["job"] == row["job"] and point["lane"] not in row["heldout"]]
            equal(row["n_train"], len(train), "natural calibration actor-disjoint train count")
            require(set(point["lane"] for point in train).isdisjoint(row["heldout"]), "natural calibration no actor leakage")
            label = "raw_payload" if row["label"] == "raw" else "mapped_payload"
            train_labels, test_labels = [point[label] for point in train], [point[label] for point in test]
            majority = int(np.bincount(train_labels, minlength=4).argmax())
            equal(row["train_majority"], majority, "natural calibration training-only majority class")
            equal(row["baseline_correct"], test_labels.count(majority), "natural calibration heldout majority baseline")
            matrix = np.array(row["confusion"], dtype=np.int64)
            require(matrix.shape == (4, 4) and np.all(matrix >= 0), "natural calibration confusion shape")
            equal(matrix.sum(1).tolist(), np.bincount(test_labels, minlength=4).tolist(), "natural calibration actual target denominators")
            equal(int(matrix.sum()), len(test), "natural calibration complete heldout request denominator")
            groups[(row["condition"], row["phase"], row["label"])].append(row)
            fold_count += 1
        require(len(result["folds"]) == 48 and len(result["summary"]) == 12, "natural calibration full condition/phase/label/fold inventory")
        for row in result["summary"]:
            folds = groups[(row["condition"], row["phase"], row["label"])]
            require(len(folds) == 4, "natural calibration four job/actor folds")
            matrix = sum((np.array(item["confusion"], dtype=np.int64) for item in folds), np.zeros((4, 4), dtype=np.int64))
            equal(row["confusion"], matrix.tolist(), "natural calibration summed confusion")
            equal(row["correct"], int(matrix.trace()), "natural calibration correct numerator")
            equal(row["n"], int(matrix.sum()), "natural calibration common three-stage denominator")
            equal(row["accuracy"], float(matrix.trace() / matrix.sum()), "natural calibration accuracy")
            equal(row["macro"], float(np.mean(matrix.diagonal() / matrix.sum(1))), "natural calibration macro recall")
            equal(row["majority_baseline"], sum(item["baseline_correct"] for item in folds) / row["n"], "natural calibration pooled training-majority baseline")
            summary_count += 1
    bindings.add(folder / "protocol.json", sha(folder / "protocol.json"), "natural calibration protocol inventory")
    bindings.add(folder / "summary.json", sha(folder / "summary.json"), "audited natural calibration summary")
    return dict(audit="passed", actor_disjoint_fold_confusions_verified=fold_count,
                seed_condition_phase_label_summaries_verified=summary_count,
                scope="Frozen source/data and continuous-feature bindings; whole-actor train/test populations, true-class denominators, training-only majority baseline, every saved confusion matrix's arithmetic and pooled accuracy/macro recall. Does not refit the natural calibration probe or establish policy use.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-only", action="store_true")
    parser.add_argument("--study", choices=("all", "natural", "matched", "probe"), default="all")
    parser.add_argument("--include-transfer", action="store_true")
    parser.add_argument("--include-supplements", action="store_true")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    bindings = Bindings()
    value = dict(format="content-timing-independent-audit-v1",
                 created_utc=datetime.now(timezone.utc).isoformat(),
                 verifier=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
                 old_source_closure=old_source_bindings(bindings), studies={})
    if not args.old_only:
        if args.study in ("natural", "all"):
            value["studies"]["natural"] = audit_natural(bindings)
        if args.study in ("matched", "all"):
            value["studies"]["matched"] = audit_matched(bindings)
        if args.study in ("probe", "all"):
            value["studies"]["probe"] = audit_probe(bindings)
        if args.include_transfer:
            value["studies"]["transfer"] = audit_transfer(bindings)
        if args.include_supplements:
            value["studies"]["onset"] = audit_onset(bindings)
            value["studies"]["natural_probe"] = audit_natural_probe(bindings)
    value["bindings"] = bindings.verify()
    value["audit"] = "passed_old_sources_only" if args.old_only else "passed"
    value["scope"] = ("all_three_primary_studies_plus_transfer" if args.include_transfer else "all_three_primary_studies") if args.study == "all" and not args.old_only else "partial_audit"
    if args.study == "all" and args.include_transfer and args.include_supplements and not args.old_only:
        value["scope"] = "three_primary_studies_plus_transfer_onset_and_natural_calibration"
    value["interpretation_limits"] = [
        "The three inherited training seeds are the independent replication units; requests, jobs, variants, families, phases and folds are nested observations.",
        "Natural delayed scores may use the model's own attempted-action/result feedback and policy-dependent survival; they do not isolate passive memory retention.",
        "Matched locked-BEGIN/WAIT forks isolate within-anchor raw-payload variation. Fixed version is repeated by the BEGIN receipt; this is not an isolated version-memory test.",
        "Controlled duration changes preserve the actual scheduler and select uninterrupted windows; they do not describe a newly sampled natural task regime.",
        "Actor lanes are held out only from the external action-head fitting, not from prior backbone training; probe actor folds keep each complete anchor's variants together.",
        "Final deployed FINISH scores come from the externally supervised controller; probe or pair discrimination is not evidence of native port learning or autonomous complete behavior.",
        "Immediate cue non-input states/scores are exactly invariant because each step flows first and then adds the pulse only on input nodes. The 25-percent balanced cue probe result is structural and does not establish failed cue encoding.",
        "Stored controlled execution-equality assertions are source-reviewed and artifact-bound; this independent audit does not advance a world to reproduce them.",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    print(json.dumps(dict(audit=value["audit"], output=str(OUT), bound_files=len(bindings.verified))))


if __name__ == "__main__":
    main()
