"""Independent native-world and per-tick-GRU validation of the A request bank."""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from experiments.experience_transfer import train_transfer as source
from experiments.review_selection import bank as builder


def audit(seed):
    begin = time.perf_counter()
    folder = builder.OUT / f"seed_{seed:03d}"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    data = builder.load(seed)
    assert manifest["bank_sha256"] == builder.sha(folder / "bank.pt")
    assert manifest["metadata_sha256"] == builder.sha(folder / "metadata.json")
    assert manifest["implementation_sha256"] == builder.sha(builder.__file__)
    for path, digest in manifest["sources"].items():
        assert builder.sha(ROOT / path) == digest
    checkpoint = ROOT / manifest["checkpoint"]
    assert builder.sha(checkpoint) == manifest["checkpoint_sha256"]
    model, _, _, _ = source.load_checkpoint(checkpoint, "cpu")
    model.eval()
    spec = source.SEEDS[seed]
    a, b, c = spec["A"], spec["B"], spec["C"]
    conflict = [raw for raw in range(4) if a[raw] != b[raw]]
    assert conflict == spec["conflict_raw"]
    assert all(a[raw] == c[raw] != b[raw] for raw in conflict)
    assert sum(x == y for x, y in zip(a, b)) == 2
    assert sum(x == y for x, y in zip(c, b)) == 0
    assert data["obs"].shape == (128, 256, 56) and data["obs"].dtype == torch.float32
    assert data["anchor"].shape == (256, 192)
    assert data["labels"].shape == data["mask"].shape == (128, 256, 2)
    assert torch.equal(data["mask"].sum((0, 2)), torch.full((256,), 3))
    assert len(set(data["ids"])) == 256
    assert torch.bincount(data["raw"]).tolist() == [64] * 4
    assert int(data["conflict"].sum()) == 128
    by_source = defaultdict(list)
    for i, meta in enumerate(data["metadata"]):
        by_source[meta["source"]].append(i)
        slot = meta["slot"]
        assert int(data["target_slot"][i]) == slot
        assert int(data["raw"][i]) == meta["raw"]
        assert bool(data["conflict"][i]) == (meta["raw"] in conflict)
        assert data["positions"][i].tolist() == meta["positions"]
        assert data["ids"][i] == meta["id"]
        assert meta["priority_sha256"] == builder.stable_priority(seed, meta)
        expected_mask = torch.zeros(128, 2, dtype=torch.bool)
        expected_mask[meta["positions"], slot] = True
        assert torch.equal(data["mask"][:, i], expected_mask)
        assert not data["mask"][:, i, 1-slot].any()
        assert not data["labels"][:, i, 1-slot].any()
        expected_request_mask = torch.zeros(128, 2, dtype=torch.bool)
        expected_request_mask[32:meta["positions"][-1]+1, slot] = True
        assert torch.equal(data["request_mask"][:, i], expected_request_mask)
    total = Counter()
    anchor_max_abs = 0.
    source_checks = []
    for src_path, indices in sorted(by_source.items()):
        original = torch.load(ROOT / src_path, map_location="cpu", weights_only=False)
        assert original["teacher_probability"] == 0 and not original["teacher_lanes"].any()
        starts, targets = defaultdict(list), defaultdict(list)
        for i in indices:
            meta = data["metadata"][i]
            lane, slot, start, stop = meta["lane"], meta["slot"], meta["start_tick"], meta["stop_tick"]
            assert stop-start == 128 and meta["cue_tick"]-start == 32
            assert torch.equal(data["obs"][:, i], original["obs"][start:stop, lane])
            assert torch.equal(data["actions"][:, i], original["actions"][start:stop, lane])
            assert torch.equal(data["labels"][:, i, slot], original["labels"][start:stop, lane, slot])
            starts[start].append(i)
            for role, tick in zip(("BEGIN", "WAIT", "FINISH"), (meta["cue_tick"], meta["wait_tick"], meta["finish_tick"])):
                targets[tick].append((i, role))
        h = original["anchor"].clone()
        with source.mapping_scope(source.mapping_table(seed, "related")), torch.no_grad():
            world = source.worlds.ContextWorld.from_state_dict(original["initial_world"], device="cpu")
            for tick in range(len(original["obs"])):
                for i in starts[tick]:
                    lane = data["metadata"][i]["lane"]
                    difference = float((h[lane]-data["anchor"][i]).abs().max())
                    anchor_max_abs = max(anchor_max_abs, difference)
                    assert torch.equal(h[lane], data["anchor"][i]), (src_path, tick, difference)
                obs = world.observe()
                labels = world.oracle("midpoint")
                assert torch.equal(obs, original["obs"][tick])
                assert torch.equal(labels, original["labels"][tick])
                for i, role in targets[tick]:
                    meta = data["metadata"][i]
                    lane, slot = meta["lane"], meta["slot"]
                    label = int(labels[lane, slot])
                    assert int(world.current_family[lane, slot]) == spec["history_cue"]
                    assert int(world.request_id[lane, slot]) == meta["request_id"]
                    assert int(world.raw_payload[lane, slot]) == meta["raw"]
                    assert int(world.version[lane, slot]) == meta["version"]
                    if role == "BEGIN":
                        assert world.event_kind[lane, slot] == source.worlds.NEW
                        assert label == 1 + meta["version"]
                        assert obs[lane].reshape(2, 28)[slot, 5:9].argmax() == meta["raw"]
                        assert obs[lane].reshape(2, 28)[slot, 24:28].argmax() == meta["context"]
                    elif role == "WAIT":
                        assert label == 0 and world.active[lane, slot] and world.started[lane, slot]
                        assert world.now < float(world.ready_time[lane, slot])-1e-6
                    else:
                        assert label == 3 + 4 * meta["version"] + a[meta["raw"]]
                        assert world.active[lane, slot] and world.started[lane, slot]
                        assert float(world.ready_time[lane, slot])-1e-6 <= world.now <= float(world.deadline[lane, slot])+1e-6
                    total[role] += 1
                # Independent step-wise reconstruction, not builder's single GRU call.
                _, h = model.step(obs, h)
                world.act(original["actions"][tick])
                total["native_world_ticks"] += 1
                total["native_obs_label_rows"] += labels.numel()
        source_checks.append(dict(source=src_path, samples=len(indices), raw_counts=dict(Counter(str(data['metadata'][i]['raw']) for i in indices)),
                                  own_actions=True, public_observations_exact=True, oracle_labels_exact=True))
    assert total["BEGIN"] == total["WAIT"] == total["FINISH"] == 256
    result = dict(seed=seed, passed=True, checks=dict(total), source_checks=source_checks,
                  mapping_conflict_raw=conflict, A_C_equal_and_A_B_differ_on_conflict=True,
                  fixed_shape_and_three_rows_per_example=True, unrelated_slot_never_supervised=True,
                  all_source_obs_actions_labels_and_request_identity_exact=True,
                  anchor_rebuilt_from_public_obs_only=True, anchor_independent_stepwise_max_abs=anchor_max_abs,
                  anchor_tolerance=dict(atol=0., rtol=0.), anchor_independent_stepwise_exact=True,
                  limitation="Checks archived own actions against the real world, not the unsaved historical policy weights' exact argmax. Frozen A B000 anchor reconstruction is independently repeated per tick.",
                  bank_sha256=builder.sha(folder / "bank.pt"), audit_implementation_sha256=builder.sha(__file__),
                  elapsed_seconds=time.perf_counter()-begin)
    builder.write(folder / "audit.json", result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=range(3))
    args = parser.parse_args()
    for seed in range(3) if args.seed is None else (args.seed,):
        audit(seed)
