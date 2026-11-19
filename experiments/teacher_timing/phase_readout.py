"""Posthoc fixed first-legal versus teacher-FINISH content comparison.

This program only reads already frozen facts and native logits. It performs
no model inference, optimization, label input, time gating or position sweep.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/teacher-timing/phase-readout-v1"
PLAN = OUT / "protocol.json"
COMMON = ROOT / "data/teacher-timing/common-v1"
SEEDS = (0, 1, 2)
POLICIES = ("earliest_action", "earliest_joint", "midpoint_action", "midpoint_joint")
PHASES = ("first_legal", "teacher_FINISH")
EXPECTED_REQUESTS = {0: 1338, 1: 1363, 2: 1356}
WARM, EPS = 512, 1e-6


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def protocol_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def raw_path(seed):
    return COMMON / f"seed_{seed:03d}" / "teacher_midpoint_long.npz"


def predictions_path(seed, policy):
    return COMMON / f"seed_{seed:03d}" / f"history_midpoint__{policy}.npz"


def declare():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    common = read(COMMON / "protocol.json")
    inputs = [COMMON / name for name in ("protocol.json", "histories.freeze.json", "completion.json")]
    inputs += [ROOT / "data/teacher-timing/eval-v1/completion.json",
        ROOT / "data/teacher-timing/train-v1/globalfreeze.json"]
    for seed in SEEDS:
        path = raw_path(seed)
        inputs += [path, path.with_suffix(".json")]
        for policy in POLICIES:
            pred = predictions_path(seed, policy)
            inputs += [pred, pred.with_suffix(".json")]
    inputs += [ROOT / name for name in common["checkpoints"]]
    sources = [Path(__file__), Path(__file__).with_name("phase_readout_test.py")]
    # Existing producer hashes are preserved separately from this new analysis.
    prior_sources = {str(ROOT / name): digest for name, digest in common["source_sha256"].items()}
    for path, digest in prior_sources.items():
        assert sha(path) == digest, path
    plan = dict(format="teacher-timing-fixed-phase-posthoc-v1",
        created_utc=datetime.now(timezone.utc).isoformat(), posthoc=True,
        declaration_timing="After all main autonomous and crossed-common-history outcomes were known; before this fixed-position paired count was computed.",
        known_observation="Root has observed better autonomous strict completion for midpoint-trained policies, but lower conditional content accuracy at first legal on common histories. This analysis asks whether the latter changes by the later actual teacher FINISH phase; it cannot explain autonomous success causally.",
        seeds=list(SEEDS), policies=list(POLICIES), test_teacher="midpoint", condition="long",
        phases=list(PHASES), warm=WARM, expected_requests_by_seed=EXPECTED_REQUESTS,
        expected_requests_total=sum(EXPECTED_REQUESTS.values()),
        denominator="Exactly all4057 postwarm NEW/REPLACE externally eligible matured requests in the existing midpoint teacher reports:1338/1363/1356. No selection by tested-model completion, prediction, content correctness or available boundary pair. Missing fixed position fails the analysis rather than dropping a request.",
        first_legal="Earliest archived current-state row for the same active started request with finite ready_time, now>=ready_time-1e-6 and now<=deadline+1e-6.",
        teacher_FINISH="The unique archived accepted native FINISH action for the same active started request, observed before that action; assert native action3+4*version+payload, receipt1 and legal clock. Preserve identical request id/version/payload and active-started status throughout the two positions.",
        readout="Use only existing native_logits; native11 argmax unfiltered with native tie rule, and conditional8way argmax of logits3:11. Count correct content and native WAIT/correctFINISH/wrongFINISH/BEGIN separately at both fixed positions.",
        paired="For conditional8way correctness and native11 correctFINISH separately: firstwrong->latercorrect, firstcorrect->laterwrong, bothcorrect, neither. Report every seed-policy row and all4 aggregated policy rows without picking a favorable one.",
        limits="Read-only same-teacher-history diagnostic after observing main results; neither tested model acts in these trajectories. No new model run, training, label input, action gate, time prediction, phase/threshold sweep or autonomous completion inference.",
        source_sha256={str(path): sha(path) for path in sources},
        preserved_producer_source_sha256=prior_sources,
        input_sha256={str(path): sha(path) for path in dict.fromkeys(inputs)})
    # Normalize integer seed keys into the same JSON representation used later.
    plan = json.loads(json.dumps(plan))
    plan["protocol_hash"] = protocol_digest(plan)
    write_new(PLAN, plan)
    return plan


def checked():
    plan = read(PLAN)
    assert protocol_digest({key: value for key, value in plan.items() if key != "protocol_hash"}) == plan["protocol_hash"]
    for table in ("source_sha256", "preserved_producer_source_sha256", "input_sha256"):
        for path, digest in plan[table].items():
            assert sha(path) == digest, path
    return plan


def select_positions(facts, report, *, warm=WARM, expected=None):
    """Take the fixed positions for every report request; never filter failures."""
    requests = report["requests"]
    if expected is not None:
        assert len(requests) == expected
    identities, positions, targets, versions, payloads = [], [], [], [], []
    seen = set()
    for request in requests:
        lane, job, rid = (int(request[key]) for key in ("lane", "job", "rid"))
        actor, issued, matured = (int(request[key]) for key in ("actor", "issued", "mature_tick"))
        assert (actor, job, rid) not in seen
        seen.add((actor, job, rid))
        assert actor == int(facts["actor_ids"][lane]) and issued >= warm
        assert int(facts["event_kind"][issued, lane, job]) in (1, 2)
        assert int(facts["event_id"][issued, lane, job]) == rid
        assert bool(facts["event_eligible"][issued, lane, job])
        assert bool(facts["mature"][matured, lane, job])
        assert int(facts["mature_id"][matured, lane, job]) == rid
        match = facts["rid"][:, lane, job] == rid
        active = match & facts["active"][:, lane, job] & facts["started"][:, lane, job]
        legal = active & np.isfinite(facts["ready"][:, lane, job])
        legal &= facts["now"] >= facts["ready"][:, lane, job] - EPS
        legal &= facts["now"] <= facts["deadline"][:, lane, job] + EPS
        legal_ticks = np.flatnonzero(legal)
        assert len(legal_ticks), (actor, job, rid, "missing first legal")
        first = int(legal_ticks[0])
        accepted_finish = match & (facts["actions"][:, lane, job] >= 3) & (facts["receipts"][:, lane, job] == 1)
        finish_ticks = np.flatnonzero(accepted_finish)
        assert len(finish_ticks) == 1, (actor, job, rid, "accepted FINISH count", len(finish_ticks))
        later = int(finish_ticks[0])
        assert first <= later and legal[later]
        version, payload = int(facts["version"][first, lane, job]), int(facts["payload"][first, lane, job])
        segment = slice(first, later+1)
        assert active[segment].all()
        assert (facts["version"][segment, lane, job] == version).all()
        assert (facts["payload"][segment, lane, job] == payload).all()
        target = 4*version + payload
        assert int(facts["actions"][later, lane, job]) == 3 + target
        assert int(facts["receipts"][later, lane, job]) == 1
        assert int(facts["event_version"][issued, lane, job]) == version
        assert int(facts["event_payload"][issued, lane, job]) == payload
        identities.append((actor, lane, job, rid, issued, matured))
        positions.append((first, later)); targets.append(target)
        versions.append(version); payloads.append(payload)
    return dict(identity=np.asarray(identities, dtype=np.int64),
        ticks=np.asarray(positions, dtype=np.int64), target_content=np.asarray(targets, dtype=np.int64),
        version=np.asarray(versions, dtype=np.int64), payload=np.asarray(payloads, dtype=np.int64))


def paired_counts(correct):
    assert correct.dtype == np.bool_ and correct.ndim == 2 and correct.shape[1] == 2
    first, later = correct[:, 0], correct[:, 1]
    result = dict(firstwrong_latercorrect=int((~first & later).sum()),
        firstcorrect_laterwrong=int((first & ~later).sum()),
        bothcorrect=int((first & later).sum()), neither=int((~first & ~later).sum()))
    assert sum(result.values()) == len(correct)
    return result


def summarize_logits(selected, targets):
    assert selected.shape == (len(targets), 2, 11) and np.isfinite(selected).all()
    native = selected.argmax(axis=-1)
    content = selected[..., 3:11].argmax(axis=-1)
    content_correct = content == targets[:, None]
    finish_correct = native == targets[:, None]+3
    phases = {}
    for phase, index in zip(PHASES, range(2)):
        phases[phase] = dict(requests=len(targets), conditional8_content_correct=int(content_correct[:, index].sum()),
            native11_correct_FINISH=int(finish_correct[:, index].sum()),
            native11_WAIT=int((native[:, index] == 0).sum()),
            native11_wrong_FINISH=int(((native[:, index] >= 3) & ~finish_correct[:, index]).sum()),
            native11_BEGIN=int(((native[:, index] == 1) | (native[:, index] == 2)).sum()))
        assert sum(phases[phase][key] for key in ("native11_correct_FINISH", "native11_WAIT", "native11_wrong_FINISH", "native11_BEGIN")) == len(targets)
    return dict(requests=len(targets), phases=phases,
        paired_conditional8_content=paired_counts(content_correct),
        paired_native11_correct_FINISH=paired_counts(finish_correct)), dict(
            native11_argmax=native, conditional8_argmax=content,
            conditional8_correct=content_correct, native11_correct_FINISH=finish_correct)


def aggregate(rows):
    result = dict(requests=sum(row["requests"] for row in rows), phases={},
        paired_conditional8_content={}, paired_native11_correct_FINISH={})
    for phase in PHASES:
        result["phases"][phase] = {key: sum(row["phases"][phase][key] for row in rows)
            for key in rows[0]["phases"][phase]}
    for group in ("paired_conditional8_content", "paired_native11_correct_FINISH"):
        result[group] = {key: sum(row[group][key] for row in rows) for key in rows[0][group]}
    return result


def run():
    plan = checked()
    results, evidence_files = [], {}
    for seed in SEEDS:
        path = raw_path(seed)
        with np.load(path, allow_pickle=False) as archive:
            facts = dict(archive)
        selection = select_positions(facts, read(path.with_suffix(".json")), expected=EXPECTED_REQUESTS[seed])
        first, later = selection["ticks"].T
        actor, lane, job, rid, issued, matured = selection["identity"].T
        evidence = dict(selection)
        evidence["phase_seconds_after_ready"] = facts["now"][selection["ticks"]] - facts["ready"][selection["ticks"], lane[:, None], job[:, None]]
        for policy in POLICIES:
            with np.load(predictions_path(seed, policy), allow_pickle=False) as archive:
                logits = archive["native_logits"]
            selected = logits[selection["ticks"], lane[:, None], job[:, None]]
            summary, details = summarize_logits(selected, selection["target_content"])
            summary.update(seed=seed, policy=policy)
            results.append(summary)
            for key, value in details.items():
                evidence[f"{policy}__{key}"] = value
        evidence_path = OUT / f"seed_{seed:03d}_paired_requests.npz"
        with evidence_path.open("xb") as stream:
            np.savez_compressed(stream, **evidence)
        evidence_files[str(evidence_path)] = sha(evidence_path)
    totals = {policy: aggregate([row for row in results if row["policy"] == policy]) for policy in POLICIES}
    assert len(results) == 12 and all(row["requests"] == 4057 for row in totals.values())
    # Verify every original raw/source/prediction binding again after counting.
    checked()
    result = dict(format="teacher-timing-fixed-phase-posthoc-results-v1", posthoc=True,
        protocol_hash=plan["protocol_hash"], protocol_sha256=sha(PLAN),
        rows=results, pooled_by_policy=totals, evidence_files=evidence_files,
        all_original_inputs_and_sources_unchanged=True, no_new_model_runs=True,
        interpretation="Paired content/readout counts on one actual midpoint-teacher background, after seeing main outcomes; fixed teacher FINISH is not each tested model's own action time or its autonomous trajectory.")
    write_new(OUT / "results.json", result)
    with (OUT / "per_seed.csv").open("x", newline="", encoding="utf-8") as stream:
        fields = ["seed", "policy", "requests"] + [f"{phase}__{key}" for phase in PHASES
            for key in ("conditional8_content_correct", "native11_correct_FINISH", "native11_WAIT", "native11_wrong_FINISH", "native11_BEGIN")] + [f"paired_content__{key}" for key in results[0]["paired_conditional8_content"]]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in results:
            flat = {key: row[key] for key in ("seed", "policy", "requests")}
            for phase in PHASES:
                flat.update({f"{phase}__{key}": row["phases"][phase][key] for key in row["phases"][phase] if key != "requests"})
            flat.update({f"paired_content__{key}": value for key, value in row["paired_conditional8_content"].items()})
            writer.writerow(flat)
    print(json.dumps(totals, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "run"))
    args = parser.parse_args()
    declare() if args.command == "declare" else run()


if __name__ == "__main__":
    main()
