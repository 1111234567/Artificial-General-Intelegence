"""Fixed paired memories from the already learned, actual midpoint history.

Selection uses completed past teacher experiences, never policy predictions.
The full public windows remain native; only the named slot's 96 postcue rows
train, including its real inactive WAIT tail. Numerical replay anchors are deliberately absent: each update must
reconstruct them with its current model from the original public history.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from experiments.continuity.run import state_digest
from experiments.experience_transfer.train_transfer import SEEDS as SEED_SPECS
from experiments.teacher_timing import data as old

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/episodic-review/bank-v1"
PLAN = OUT / "protocol.json"
FORMAT = "episodic-review-paired-past-teacher-bank-v1"
SEEDS = (0, 1, 2)
FAMILIES = ("A", "B")
WINDOW, PRELUDE, PAIRS, PAIRS_PER_RAW = 128, 32, 256, 64
WARM, TOTAL, DT, TIME_SCALE = 512, 33280, .25, 16.
REPLAY_UPDATES, REPLAY_BATCH, REPLAY_EVERY = 32, 32, 8
SELECTION_SEED_BASE, SCHEDULE_SEED_BASE = 936000, 937000
PAIR_KEY = ("target_slot", "raw", "version", "cue_kind", "supervised_length", "first_legal_offset")
ARRAY_FIELDS = ("obs", "actions", "time_target", "time_mask", "action_mask")
VECTOR_FIELDS = ("anchor_start", "actor_lane", "actor_id", "target_slot", "rid", "raw", "family",
    "version", "cue_kind", "cue_tick", "finish_tick", "supervised_length", "first_legal_offset", "pair_id")
FACT_FIELDS = ("obs", "actions", "receipts", "active", "started", "rid", "raw", "family", "version",
    "payload", "duration", "ready", "deadline", "issued_time", "event_kind", "event_id", "event_raw",
    "event_family", "event_version", "event_payload", "event_duration", "event_window", "now", "tick",
    "actor_ids", "time_target", "time_mask")
sha, write_new = old.sha, old.write_new


def bank_path(seed):
    if int(seed) not in SEEDS:
        raise ValueError(seed)
    return OUT / f"seed_{int(seed):03d}" / "bank.npz"


def metadata_path(seed):
    return bank_path(seed).with_suffix(".json")


def source_obs_path(seed):
    return old.raw_path(seed, "midpoint")


def load_history(seed):
    """Original public observations only; no private facts or cached hidden h."""
    with np.load(source_obs_path(seed), allow_pickle=False) as archive:
        return archive["obs"]


def load_source_facts(seed):
    with np.load(source_obs_path(seed), allow_pickle=False) as archive:
        return {key: archive[key] for key in FACT_FIELDS}


def load_bank(seed):
    with np.load(bank_path(seed), allow_pickle=False) as archive:
        result = {family: {key: archive[f"{family}__{key}"] for key in (*ARRAY_FIELDS, *VECTOR_FIELDS)}
            for family in FAMILIES}
        result["schedule"] = archive["schedule"]
    return result


def source_inputs(seed):
    return [source_obs_path(seed), old.metadata_path(seed, "midpoint"), old.states_path(seed, "midpoint"),
        old.audit_path(seed, "midpoint"), old.PLAN,
        old.OUT / "audit_completion.json", old.OUT / "completion.json"]


def input_paths(seed):
    return [PLAN, bank_path(seed), metadata_path(seed), *source_inputs(seed)]


def sources():
    return list(dict.fromkeys([Path(__file__).resolve(), Path(__file__).with_name("bank_test.py"),
        Path(old.__file__).resolve(), ROOT / "experiments/action_split/data.py",
        ROOT / "experiments/experience_transfer/train_transfer.py", ROOT / "experiments/continuity/run.py",
        ROOT / "experiments/continuity/world.py", ROOT / "experiments/lifelong/world.py"]))


def replay_schedule(seed):
    """Four independently shuffled complete passes; A/B use identical pair IDs."""
    rng = np.random.default_rng(SCHEDULE_SEED_BASE + int(seed))
    return np.concatenate([rng.permutation(PAIRS) for _ in range(4)]).reshape(REPLAY_UPDATES, REPLAY_BATCH)


def candidates(facts, seed):
    """Completed past requests, matched by actual unique command identity.

    Neither eligibility nor eventual cancellation/maturity flags are read.
    A canceled/replaced/nonfinished command simply is not a completed memory.
    """
    actions, receipts = facts["actions"], facts["receipts"]
    n = len(actions)
    accepted = (actions >= 3) & (receipts == 1)
    finishes = defaultdict(list)
    for tick, lane, slot in zip(*np.nonzero(accepted)):
        identity = int(lane), int(slot), int(facts["rid"][tick, lane, slot])
        finishes[identity].append(int(tick))
    cues = {SEED_SPECS[int(seed)]["history_cue"]: "A", SEED_SPECS[int(seed)]["B_cue"]: "B"}
    result = {family: defaultdict(list) for family in FAMILIES}
    command_ids = set()
    for tick, lane, slot in zip(*np.nonzero((facts["event_kind"] == 1) | (facts["event_kind"] == 2))):
        tick, lane, slot = int(tick), int(lane), int(slot)
        rid = int(facts["event_id"][tick, lane, slot])
        identity = lane, slot, rid
        assert identity not in command_ids, ("duplicate actual command", identity)
        command_ids.add(identity)
        start = tick - PRELUDE
        if tick < WARM or start < 0 or start + WINDOW > n:
            continue
        if len(finishes.get(identity, ())) != 1:
            continue
        finish = finishes[identity][0]
        if finish < tick or finish >= start + WINDOW:
            continue
        # A cue without an immediate accepted BEGIN is not this library's
        # complete, independently executable cue-to-FINISH experience.
        if actions[tick, lane, slot] != 1 + facts["event_version"][tick, lane, slot] or receipts[tick, lane, slot] != 1:
            continue
        family = cues[int(facts["event_family"][tick, lane, slot])]
        duration = float(facts["event_duration"][tick, lane, slot])
        row = dict(anchor_start=start, actor_lane=lane, actor_id=int(facts["actor_ids"][lane]),
            target_slot=slot, rid=rid, raw=int(facts["event_raw"][tick, lane, slot]),
            family=int(facts["event_family"][tick, lane, slot]), version=int(facts["event_version"][tick, lane, slot]),
            cue_kind=int(facts["event_kind"][tick, lane, slot]), cue_tick=tick, finish_tick=finish,
            supervised_length=finish-tick+1,
            first_legal_offset=int(np.ceil(duration / DT - 1e-6)))
        result[family][tuple(row[key] for key in PAIR_KEY)].append(row)
    return result


def select_pairs(facts, seed):
    """Fixed random ranking within matched strata, then exactly 64/raw."""
    available = candidates(facts, seed)
    rng = np.random.default_rng(SELECTION_SEED_BASE + int(seed))
    by_raw = defaultdict(list)
    for key in sorted(set(available["A"]) & set(available["B"])):
        left, right = available["A"][key], available["B"][key]
        left = [left[i] for i in rng.permutation(len(left))]
        right = [right[i] for i in rng.permutation(len(right))]
        for a, b in zip(left, right):
            by_raw[key[1]].append((float(rng.random()), a, b))
    selected = []
    for raw in range(4):
        ranked = sorted(by_raw[raw], key=lambda pair: pair[0])
        assert len(ranked) >= PAIRS_PER_RAW, (seed, raw, len(ranked))
        selected.extend(ranked[:PAIRS_PER_RAW])
    # Bank order is a separate fixed random ranking, shared across A and B.
    selected.sort(key=lambda pair: pair[0])
    rows = {family: [] for family in FAMILIES}
    for pair_id, (_, a, b) in enumerate(selected):
        for family, row in (("A", a), ("B", b)):
            rows[family].append({**row, "pair_id": pair_id})
    report = dict(candidate_counts={family: sum(map(len, group.values())) for family, group in available.items()},
        matched_available_per_raw={str(raw): len(by_raw[raw]) for raw in range(4)}, selected_pairs=PAIRS)
    return rows, report


def request_ledger(facts, row, seed, family):
    """Independently check a selected identity and its actual accepted actions.

    Clocks are reconstructed from the public command and accepted BEGIN,
    rather than taking archived ready/time labels as the source of truth.
    """
    cue, finish = int(row["cue_tick"]), int(row["finish_tick"])
    lane, slot, rid = (int(row[key]) for key in ("actor_lane", "target_slot", "rid"))
    ix = cue, lane, slot
    version, raw, context = (int(facts[key][ix]) for key in ("event_version", "event_raw", "event_family"))
    expected_context = SEED_SPECS[int(seed)]["history_cue" if family == "A" else "B_cue"]
    payload = SEED_SPECS[int(seed)][family][raw]
    assert context == expected_context == row["family"]
    assert raw == row["raw"] and version == row["version"]
    assert int(facts["event_kind"][ix]) == row["cue_kind"] in (1, 2)
    assert int(facts["event_id"][ix]) == rid
    assert cue >= WARM and int(facts["tick"][cue]) == cue
    assert row["anchor_start"] == cue - PRELUDE >= 0
    assert row["anchor_start"] + WINDOW <= len(facts["actions"])
    assert row["supervised_length"] == finish-cue+1
    assert finish < row["anchor_start"] + WINDOW
    assert facts["issued_time"][ix] == facts["now"][cue]
    assert not bool(facts["started"][ix]) and bool(facts["active"][ix])
    assert int(facts["event_payload"][ix]) == payload
    duration = float(facts["event_duration"][ix])
    deadline = float(facts["now"][cue]) + duration + float(facts["event_window"][ix])
    ready = float(facts["now"][cue]) + duration
    assert row["first_legal_offset"] == int(np.ceil(duration / DT - 1e-6))
    ticks = np.arange(cue, finish+1)
    idx = ticks, lane, slot
    for key, value in (("rid", rid), ("family", context), ("raw", raw), ("version", version),
                       ("payload", payload), ("duration", duration), ("deadline", deadline)):
        assert np.all(facts[key][idx] == value), (rid, key)
    assert not np.any(facts["event_kind"][cue+1:finish+1, lane, slot]), "interrupted selected request"
    assert np.all(facts["active"][idx])
    assert not facts["started"][ix] and np.all(facts["started"][cue+1:finish+1, lane, slot])
    assert np.all(facts["ready"][cue+1:finish+1, lane, slot] == ready)
    assert np.isinf(facts["ready"][ix])
    expected = np.zeros(len(ticks), np.int64)
    expected[0], expected[-1] = 1+version, 3+4*version+payload
    np.testing.assert_array_equal(facts["actions"][idx], expected)
    np.testing.assert_array_equal(facts["receipts"][idx], (expected != 0).astype(np.int64))
    # Exactly one accepted BEGIN and one accepted correct FINISH for this
    # request, even if a corrupted archive repeats an action outside the span.
    same = facts["rid"][:, lane, slot] == rid
    nonwait = same & (facts["actions"][:, lane, slot] != 0)
    np.testing.assert_array_equal(np.flatnonzero(nonwait), [cue, finish])
    now = facts["now"][finish]
    assert ready-1e-6 <= now <= deadline+1e-6
    assert now >= .5*(ready+deadline)-1e-6 or now+DT > deadline+1e-6
    if finish > cue+1:
        previous = facts["now"][finish-1]
        assert previous < .5*(ready+deadline)-1e-6 and previous+DT <= deadline+1e-6
    tail = slice(finish+1, row["anchor_start"]+WINDOW)
    assert not np.any(facts["event_kind"][tail, lane, slot]), "new external command in selected inactive tail"
    assert not np.any(facts["active"][tail, lane, slot]), "active selected post-FINISH tail"
    assert not np.any(facts["started"][tail, lane, slot]), "started selected post-FINISH tail"
    assert not np.any(facts["actions"][tail, lane, slot]), "non-WAIT selected post-FINISH tail"
    assert not np.any(facts["receipts"][tail, lane, slot]), "action receipt in selected inactive tail"
    mask = np.arange(WINDOW) > PRELUDE
    mask &= np.arange(WINDOW) <= finish-row["anchor_start"]
    target = np.zeros(WINDOW, np.float32)
    target[mask] = ((ready-facts["now"][row["anchor_start"]:row["anchor_start"]+WINDOW][mask])/TIME_SCALE).astype(np.float32)
    return mask, target


def materialize(facts, rows, seed):
    result = {"schedule": replay_schedule(seed)}
    for family in FAMILIES:
        selected = rows[family]
        group = {key: np.asarray([row[key] for row in selected], np.int64) for key in VECTOR_FIELDS}
        indices = group["anchor_start"][None, :] + np.arange(WINDOW)[:, None]
        lanes = group["actor_lane"][None, :]
        group["obs"] = facts["obs"][indices, lanes].copy()
        group["actions"] = facts["actions"][indices, lanes].copy()
        group["action_mask"] = np.zeros((WINDOW, PAIRS, 2), bool)
        group["time_mask"] = np.zeros_like(group["action_mask"])
        group["time_target"] = np.zeros((WINDOW, PAIRS, 2), np.float32)
        for pair_id, row in enumerate(selected):
            mask, target = request_ledger(facts, row, seed, family)
            slot = row["target_slot"]
            group["action_mask"][PRELUDE:, pair_id, slot] = True
            group["time_mask"][:, pair_id, slot] = mask
            group["time_target"][:, pair_id, slot] = target
            native_mask = facts["time_mask"][indices[:, pair_id], row["actor_lane"]]
            np.testing.assert_array_equal(group["time_mask"][:, pair_id], native_mask & group["action_mask"][:, pair_id])
            native_target = facts["time_target"][indices[:, pair_id], row["actor_lane"]]
            np.testing.assert_array_equal(group["time_target"][:, pair_id],
                np.where(group["time_mask"][:, pair_id], native_target, np.float32(0)))
        result[family] = group
    validate(result, seed)
    return result


def count_rows(group):
    mask, actions = group["action_mask"], group["actions"]
    time_mask, target = group["time_mask"], group["time_target"]
    selected = actions[mask]
    wait = int(np.count_nonzero(selected == 0))
    return dict(windows=PAIRS, observation_lane_ticks=WINDOW*PAIRS, action_rows=int(mask.sum()),
        WAIT=wait, BEGIN=int(np.count_nonzero((selected == 1) | (selected == 2))),
        FINISH=int(np.count_nonzero(selected >= 3)), action_weight_sum=float(mask.sum()-.75*wait),
        action_codes={str(code): int(np.count_nonzero(selected == code)) for code in range(11)},
        time_rows=int(time_mask.sum()), time_positive=int((time_mask & (target > 0)).sum()),
        time_zero=int((time_mask & (target == 0)).sum()), time_negative=int((time_mask & (target < 0)).sum()),
        raw_counts={str(raw): int(np.count_nonzero(group["raw"] == raw)) for raw in range(4)},
        cue_kind_counts={str(kind): int(np.count_nonzero(group["cue_kind"] == kind)) for kind in (1, 2)},
        per_window_time_rows=(time_mask.sum(axis=(0, 2))).tolist())


def validate(bank, seed):
    np.testing.assert_array_equal(bank["schedule"], replay_schedule(seed))
    for cycle in bank["schedule"].reshape(4, PAIRS):
        np.testing.assert_array_equal(np.sort(cycle), np.arange(PAIRS))
    for family in FAMILIES:
        group = bank[family]
        assert group["obs"].shape == (WINDOW, PAIRS, 56) and group["obs"].dtype == np.float32
        for key in ARRAY_FIELDS[1:]:
            assert group[key].shape == (WINDOW, PAIRS, 2), key
        assert group["actions"].dtype == np.int64 and group["time_target"].dtype == np.float32
        assert group["action_mask"].dtype == group["time_mask"].dtype == np.bool_
        for key in VECTOR_FIELDS:
            assert group[key].shape == (PAIRS,) and group[key].dtype == np.int64, key
        np.testing.assert_array_equal(group["pair_id"], np.arange(PAIRS))
        identities = list(zip(group["actor_lane"], group["target_slot"], group["rid"]))
        assert len(set(identities)) == PAIRS
        assert np.all(group["cue_tick"] >= WARM)
        assert np.all(group["anchor_start"] == group["cue_tick"]-PRELUDE)
        assert np.all(group["anchor_start"]+WINDOW <= TOTAL)
        assert np.all(group["finish_tick"]-group["cue_tick"]+1 == group["supervised_length"])
        assert np.all(group["time_mask"] <= group["action_mask"])
        assert not np.any(group["time_target"][~group["time_mask"]])
        for pair_id in range(PAIRS):
            slot, length = group["target_slot"][pair_id], group["supervised_length"][pair_id]
            expected = np.zeros((WINDOW, 2), bool)
            expected[PRELUDE:, slot] = True
            np.testing.assert_array_equal(group["action_mask"][:, pair_id], expected)
            expected[PRELUDE+length:, slot] = False
            expected[PRELUDE, slot] = False
            np.testing.assert_array_equal(group["time_mask"][:, pair_id], expected)
            labels = group["actions"][:, pair_id][group["action_mask"][:, pair_id]]
            assert labels[0] == 1+group["version"][pair_id]
            payload = SEED_SPECS[int(seed)][family][group["raw"][pair_id]]
            assert labels[length-1] == 3+4*group["version"][pair_id]+payload
            assert np.all(labels[1:length-1] == 0)
            assert np.all(labels[length:] == 0)
        counts = count_rows(group)
        assert list(counts["raw_counts"].values()) == [PAIRS_PER_RAW]*4
        assert counts["BEGIN"] == counts["FINISH"] == PAIRS
    for key in (*PAIR_KEY, "action_mask", "time_mask", "pair_id"):
        np.testing.assert_array_equal(bank["A"][key], bank["B"][key], err_msg=key)
    np.testing.assert_array_equal(np.sign(bank["A"]["time_target"]), np.sign(bank["B"]["time_target"]))
    return True


def declare():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    old.checked()
    inputs = {}
    for seed in SEEDS:
        metadata = json.loads(old.metadata_path(seed, "midpoint").read_text(encoding="utf-8"))
        audit = json.loads(old.audit_path(seed, "midpoint").read_text(encoding="utf-8"))
        assert sha(source_obs_path(seed)) == metadata["stream_sha256"] == audit["stream_sha256"]
        assert audit["native_replay_ticks"] == TOTAL and audit["observations_actions_receipts"] == "exact native replay every tick"
        inputs[str(seed)] = {str(path): sha(path) for path in source_inputs(seed)}
    plan = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(), seeds=list(SEEDS),
        source_sha256={str(path): sha(path) for path in sources()}, input_sha256=inputs,
        teacher="actual executed midpoint teacher; immutable pre-withdraw archive only",
        source_stream=dict(total_ticks=TOTAL, warm_ticks=WARM, learned_ticks=TOTAL-WARM, batch=32, public_width=56),
        selection_seed_base=SELECTION_SEED_BASE, schedule_seed_base=SCHEDULE_SEED_BASE,
        pairs=PAIRS, pairs_per_raw=PAIRS_PER_RAW, window=WINDOW, prelude=PRELUDE, pair_key=list(PAIR_KEY),
        family_specs={str(seed): SEED_SPECS[seed] for seed in SEEDS},
        selection="For each unique actual NEW/REPLACE identity issued at tick>=512: accepted BEGIN exactly at cue, one accepted correct native FINISH, entire128 window in old archive. Match strata, shuffle each family's candidates, pair without replacement, random-rank matched pairs and select64/raw; final shared random-rank bank order. No policy instantiated or model-prediction selection.",
        past_outcome="This is an intentional library of already completed past teacher experiences. Canceled, replaced-before-completion, and other noncomplete commands are excluded. The outcome is known before continuation training; no future continuation/test events or future-eligibility flags select labels.",
        action_supervision="Only target slot at all96 postcue positions32..127; all prelude and other-slot rows masked. Native accepted BEGIN, processing WAIT, correct FINISH, and actual inactive post-FINISH WAIT tail are supervised. Assert no external event or active task in tail. Native full-window labels remain archived but only action_mask contributes. WAIT=.25, BEGIN/FINISH=1 remains trainer objective.",
        time_supervision="action_mask & current active & started & finite reconstructed ready; signed float32(ready-now)/16. No eligibility/outcome filter within selected episode; exactly L-1 rows. first_legal_offset pairing also matches positive/negative row signs.",
        replay_state="No hidden/anchor values stored. At every replay update, reconstruct current-model h from h0 on old public observations [0:anchor_start] for each selected actor lane, detach then run selected128 window. Replay h never overwrites live h.",
        schedule=dict(updates=REPLAY_UPDATES, every_nth_update=REPLAY_EVERY, batch=REPLAY_BATCH,
            construction="four independently shuffled permutations of256 pair IDs, concatenated and reshaped32x32; A/B share indices", presentations_per_pair=4),
        native_provenance="Pinned old raw stream, metadata, initial/final world state and already completed full native replay audit. Every selected request is separately rechecked by accepted-BEGIN command ledger and actual correct FINISH receipt.",
        no_hidden_arrays=True, inputs="public56 observations only; clocks and request facts only labels/audit metadata")
    plan["protocol_hash"] = state_digest(plan)
    write_new(PLAN, plan)
    return plan


def checked():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert state_digest({key: value for key, value in plan.items() if key != "protocol_hash"}) == plan["protocol_hash"]
    for path, digest in plan["source_sha256"].items():
        assert sha(path) == digest, "frozen bank source changed: "+path
    for files in plan["input_sha256"].values():
        for path, digest in files.items():
            assert sha(path) == digest, "frozen bank input changed: "+path
    return plan


def build(seed):
    plan = checked()
    if bank_path(seed).exists() or metadata_path(seed).exists():
        raise FileExistsError(bank_path(seed))
    facts = load_source_facts(seed)
    rows, selection = select_pairs(facts, seed)
    bank = materialize(facts, rows, seed)
    arrays = {f"{family}__{key}": value for family in FAMILIES for key, value in bank[family].items()}
    arrays["schedule"] = bank["schedule"]
    path = bank_path(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    restored = load_bank(seed)
    for family in FAMILIES:
        for key in bank[family]:
            np.testing.assert_array_equal(bank[family][key], restored[family][key])
    report = dict(format=FORMAT, seed=int(seed), protocol_hash=plan["protocol_hash"],
        bank_path=str(path), bank_sha256=sha(path), source_obs_path=str(source_obs_path(seed)),
        source_sha256=plan["source_sha256"], input_sha256=plan["input_sha256"][str(seed)],
        selection=selection, counts={family: count_rows(bank[family]) for family in FAMILIES},
        selected_requests=rows, native_ledger=dict(passed=True, checked_requests=2*PAIRS,
            accepted_BEGIN=2*PAIRS, accepted_correct_FINISH=2*PAIRS,
            private_ready_used_to_construct_labels=False, eligibility_flags_used=False,
            past_completed_outcome_used_for_library_selection=True),
        pair_checks=dict(action_masks_exact=True, time_masks_exact=True,
            WAIT_BEGIN_FINISH_counts_exact=True, action_weight_sum_exact=True,
            time_positive_zero_negative_positions_exact=True, unique_requests_within_family=True),
        schedule_sha256=old.array_hash({"schedule": bank["schedule"]}, ("schedule",)),
        replay_indices=bank["schedule"].tolist(), stored_hidden_anchors=False)
    write_new(metadata_path(seed), report)
    print(json.dumps(dict(event="bank_built", seed=int(seed), bank_sha256=report["bank_sha256"],
        counts=report["counts"], selection=selection)), flush=True)
    return report


def freeze():
    plan = checked()
    reports = {}
    for seed in SEEDS:
        report = json.loads(metadata_path(seed).read_text(encoding="utf-8"))
        assert report["protocol_hash"] == plan["protocol_hash"]
        assert sha(bank_path(seed)) == report["bank_sha256"]
        validate(load_bank(seed), seed)
        reports[str(seed)] = dict(bank_path=str(bank_path(seed)), bank_sha256=sha(bank_path(seed)),
            metadata_path=str(metadata_path(seed)), metadata_sha256=sha(metadata_path(seed)))
    result = dict(format=FORMAT, complete=True, seeds=list(SEEDS), protocol_hash=plan["protocol_hash"],
        banks=reports, training_started=False)
    write_new(OUT / "completion.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "build", "all", "freeze"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.command == "declare":
        print(json.dumps(dict(event="bank_declared", protocol_hash=declare()["protocol_hash"])), flush=True)
    elif args.command == "build":
        if args.seed is None:
            parser.error("build requires --seed")
        build(args.seed)
    elif args.command == "freeze":
        freeze()
    else:
        for seed in SEEDS:
            build(seed)
        freeze()


if __name__ == "__main__":
    main()
