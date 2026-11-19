"""Request-level A replay bank from the last four genuine own-action histories.

Only public observations and a public-history reconstruction of h are policy
inputs. Oracle labels and request identities are loss/audit metadata. A sample
always has 128 observations and exactly three target-slot supervision rows.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from experiments.experience_transfer import train_transfer as source

OUT = ROOT / "data/review-selection/bank-v1"
ROUNDS = (8, 9, 10, 11)
PREFIX, SUFFIX = 32, 96
PER_ROUND_RAW = 16
FORMAT = "review-selection-real-A-request-bank-v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load(seed, root=OUT):
    """Return time-major obs/labels/mask, initial anchor, and request metadata."""
    return torch.load(Path(root) / f"seed_{seed:03d}" / "bank.pt", map_location="cpu", weights_only=False)


def stable_priority(seed, row):
    # Fixed before any B/review evaluation; never inspect completion or loss.
    key = f"A-bank-v1:{seed}:{row['source_round']}:{row['lane']}:{row['slot']}:{row['request_id']}"
    return hashlib.sha256(key.encode()).hexdigest()


def public_hidden(model, obs, anchor):
    """Frozen A endpoint, source's actual numeric anchor, public observations only."""
    with torch.no_grad():
        values, _ = model.core(obs, anchor[None])
    return torch.cat((anchor[None], values), dim=0)


def scan(seed, index, path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    assert data["teacher_probability"] == 0 and not data["teacher_lanes"].any()
    assert data["labels"] is not None and tuple(data["obs"].shape) == (1024, 64, 56)
    spec = source.SEEDS[seed]
    candidates, live = [], {}
    stats = Counter()
    per_raw = {str(raw): Counter() for raw in range(4)}
    with source.mapping_scope(source.mapping_table(seed, "related")):
        world = source.worlds.ContextWorld.from_state_dict(data["initial_world"], device="cpu")
        for tick in range(len(data["obs"])):
            obs = world.observe()
            labels = world.oracle("midpoint")
            assert torch.equal(obs, data["obs"][tick]), (path, tick, "public obs")
            assert torch.equal(labels, data["labels"][tick]), (path, tick, "oracle labels")
            for lane, slot in (world.event_kind == source.worlds.NEW).nonzero().tolist():
                assert int(world.event_family[lane, slot]) == spec["history_cue"]
                raw = int(world.raw_event_payload[lane, slot])
                row = dict(source_round=index, lane=lane, actor_id=int(world.actor_ids[lane]), slot=slot,
                           request_id=int(world.event_request_id[lane, slot]), cue_tick=tick,
                           world_cue_tick=world.tick, raw=raw, version=int(world.version[lane, slot]),
                           context=spec["history_cue"], duration=float(world.duration[lane, slot]),
                           window=float(world.event_window[lane, slot]), wait_ticks=[], finish_tick=None,
                           actual_cue_action=int(data["actions"][tick, lane, slot]),
                           A_target=spec["A"][raw], B_target=spec["B"][raw], C_target=spec["C"][raw])
                assert int(labels[lane, slot]) == 1 + row["version"]
                live[lane, slot] = row
                stats["NEW_requests"] += 1
                per_raw[str(raw)]["NEW_requests"] += 1
            for key, row in tuple(live.items()):
                lane, slot = key
                if int(world.request_id[lane, slot]) != row["request_id"]:
                    del live[key]
                    continue
                active, started = bool(world.active[lane, slot]), bool(world.started[lane, slot])
                label = int(labels[lane, slot])
                if active and started and label == 0 and world.now < float(world.ready_time[lane, slot]) - 1e-6:
                    row["wait_ticks"].append(tick)
                if row["finish_tick"] is None and label >= 3:
                    assert active and started
                    assert world.now >= float(world.ready_time[lane, slot]) - 1e-6
                    assert world.now <= float(world.deadline[lane, slot]) + 1e-6
                    assert label == 3 + 4 * row["version"] + row["A_target"]
                    row["finish_tick"] = tick
                    stats["requests_with_legal_teacher_FINISH"] += 1
                    per_raw[str(row["raw"])]["requests_with_legal_teacher_FINISH"] += 1
                    if row["wait_ticks"]:
                        stats["requests_with_processing_WAIT_and_FINISH"] += 1
                        per_raw[str(row["raw"])]["requests_with_processing_WAIT_and_FINISH"] += 1
                        start = row["cue_tick"] - PREFIX
                        stop = row["cue_tick"] + SUFFIX
                        if start >= 0 and stop <= len(data["obs"]) and tick < stop:
                            row["start_tick"], row["stop_tick"] = start, stop
                            row["wait_tick"] = row["wait_ticks"][len(row["wait_ticks"]) // 2]
                            row["conflict"] = row["raw"] in spec["conflict_raw"]
                            row["priority_sha256"] = stable_priority(seed, row)
                            candidates.append(dict(row))
                            stats["fixed_window_eligible"] += 1
                            per_raw[str(row["raw"])]["fixed_window_eligible"] += 1
            world.act(data["actions"][tick])
    return data, candidates, dict(total=dict(stats), raw={key: dict(v) for key, v in per_raw.items()})


def build(seed):
    begin = time.perf_counter()
    folder = OUT / f"seed_{seed:03d}"
    folder.mkdir(parents=True, exist_ok=False)
    src_folder = source.OUT / f"seed_{seed:03d}" / "related"
    checkpoint = src_folder / "B000.pt"
    model, _, _, cp = source.load_checkpoint(checkpoint, "cpu")
    model.eval()
    selected, statistics, sources = [], [], {}
    all_obs, all_actions, all_labels, all_masks, all_request_masks, all_anchor = [], [], [], [], [], []
    for index in ROUNDS:
        path = src_folder / f"trajectory_history_{index:03d}.pt"
        source_sha = sha(path)
        data, candidates, stats = scan(seed, index, path)
        by_raw = {raw: sorted((r for r in candidates if r["raw"] == raw), key=lambda r: r["priority_sha256"])
                  for raw in range(4)}
        assert all(len(v) >= PER_ROUND_RAW for v in by_raw.values()), {k:len(v) for k,v in by_raw.items()}
        rows = [row for raw in range(4) for row in by_raw[raw][:PER_ROUND_RAW]]
        hs = public_hidden(model, data["obs"], data["anchor"])
        for row in rows:
            lane, slot = row["lane"], row["slot"]
            start, stop = row["start_tick"], row["stop_tick"]
            positions = [row["cue_tick"] - start, row["wait_tick"] - start, row["finish_tick"] - start]
            labels = torch.zeros(PREFIX + SUFFIX, 2, dtype=torch.long)
            labels[:, slot] = data["labels"][start:stop, lane, slot]
            mask = torch.zeros_like(labels, dtype=torch.bool)
            mask[positions, slot] = True
            request_mask = torch.zeros_like(mask)
            request_mask[PREFIX:positions[-1]+1, slot] = True
            assert labels[positions[0], slot] in (1, 2) and labels[positions[1], slot] == 0
            assert labels[positions[-1], slot] >= 3 and mask.sum() == 3
            row.pop("wait_ticks")
            row.update(sample_index=len(selected), positions=positions,
                       source=str(path.relative_to(ROOT)), source_sha256=source_sha,
                       id=f"seed{seed}/history{index:03d}/actor{row['actor_id']}/slot{slot}/request{row['request_id']}")
            selected.append(row)
            all_obs.append(data["obs"][start:stop, lane].clone())
            all_actions.append(data["actions"][start:stop, lane].clone())
            all_labels.append(labels)
            all_masks.append(mask)
            all_request_masks.append(request_mask)
            all_anchor.append(hs[start, lane].clone())
        statistics.append(dict(source_round=index, selected=len(rows), **stats))
        sources[str(path.relative_to(ROOT))] = source_sha
    n = len(selected)
    bank = dict(format=FORMAT, seed=seed, obs=torch.stack(all_obs, 1), actions=torch.stack(all_actions, 1),
                labels=torch.stack(all_labels, 1), mask=torch.stack(all_masks, 1),
                request_mask=torch.stack(all_request_masks, 1), anchor=torch.stack(all_anchor),
                raw=torch.tensor([r["raw"] for r in selected], dtype=torch.long),
                conflict=torch.tensor([r["conflict"] for r in selected], dtype=torch.bool),
                target_slot=torch.tensor([r["slot"] for r in selected], dtype=torch.long),
                positions=torch.tensor([r["positions"] for r in selected], dtype=torch.long),
                ids=[r["id"] for r in selected], metadata=selected)
    assert n == 256 and bank["obs"].shape == (128, n, 56)
    assert torch.equal(bank["mask"].sum((0, 2)), torch.full((n,), 3))
    for raw in range(4):
        assert int((bank["raw"] == raw).sum()) == 64
    torch.save(bank, folder / "bank.pt")
    write(folder / "metadata.json", selected)
    counts = Counter()
    for row in statistics:
        counts.update(row["total"])
    manifest = dict(format=FORMAT, seed=seed, created_utc=datetime.now(timezone.utc).isoformat(),
        prefix_ticks=PREFIX, suffix_ticks=SUFFIX, sequence_ticks=128, samples=n, samples_per_raw=64,
        source_rounds=list(ROUNDS), source_history_actual_policy="own argmax, teacher_probability=0",
        fixed_supervision="Exactly one teacher BEGIN at NEW, one processing WAIT, first legal midpoint teacher FINISH; target slot only, three rows per example.",
        input="Native public 56-float observation, including both slots and actual action receipts. Private state and labels never model inputs.",
        anchor="Frozen A endpoint B000 GRU, starting each source trajectory at its actual archived numeric anchor; replay original public observations up to sample start. Prefix32 is then part of the gradient sequence.",
        label_weight="Main three-row mask is intended for equal per-row unweighted CE. request_mask is an audit/optional alternative and has unequal per-sample row counts.",
        candidate_sampling="Bank strata: each of four source rounds x four raws retains first16 by fixed SHA256 priority. No prediction loss, B outcome or completion result enters priority or eligibility.",
        selection_bias="Restricted to NEW requests with a real processing WAIT and legal teacher FINISH opportunity in the own-action stream, and complete128tick window inside a source round. This excludes canceled/replaced/unstarted/too-late starts, and is not all historical A experience. Actual completion and errors do not filter candidates.",
        mapping=source.SEEDS[seed], total_source_counts=dict(counts), per_source_counts=statistics,
        retained_actual_cue_action_counts=dict(Counter(str(r["actual_cue_action"]) for r in selected)),
        actual_supervised_positions=3*n, optional_request_positions=int(bank["request_mask"].sum()),
        checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=sha(checkpoint),
        sources=sources, implementation_sha256=sha(__file__), bank_sha256=sha(folder / "bank.pt"),
        metadata_sha256=sha(folder / "metadata.json"), elapsed_seconds=time.perf_counter()-begin)
    write(folder / "manifest.json", manifest)
    print(json.dumps(dict(seed=seed, samples=n, source_counts=dict(counts), seconds=time.perf_counter()-begin)), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=range(3))
    args = parser.parse_args()
    seeds = range(3) if args.seed is None else (args.seed,)
    for seed in seeds:
        build(seed)


if __name__ == "__main__":
    main()
