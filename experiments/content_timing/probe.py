"""Cross-actor linear probes for paired, continuous FP64 state snapshots."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

from experiments.lifelong.world import FAMILY_MAPPINGS


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/content-timing"
PLAN = Path(__file__).with_name("probe-plan.md")
PHASES = ("cue", "mid", "first_legal")
OUTPUT = DATA / "probe-v1"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fit_scores(train, targets, test):
    """Unpenalized intercept and standardized, regularized slopes."""
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0] = 1
    x = (train - mean) / scale
    y = np.eye(4)[targets]
    base = y.mean(axis=0)
    ridge = .001 * len(x) * np.eye(x.shape[1])
    weights = np.linalg.solve(x.T @ x + ridge, x.T @ (y - base))
    return (test - mean) / scale @ weights + base


def evaluate(anchors, states, durations, raw_symbols):
    """states[anchor, duration, raw, phase, node]; all variants stay together."""
    n = len(anchors)
    assert states.dtype == np.float64 and np.isfinite(states).all()
    assert states.shape == (n, len(durations), 4, 3, 24) and list(raw_symbols) == list(range(4))
    jobs = np.array([a["job"] for a in anchors])
    lanes = np.array([a["lane"] for a in anchors])
    families = np.array([a["family"] for a in anchors])
    assert set(lanes) == {4, 5, 6, 7} and set(jobs) == {0, 1}
    assert len({a["identity"] for a in anchors}) == n
    raw = np.broadcast_to(np.arange(4), (n, 4))
    mapped = np.array(FAMILY_MAPPINGS)[families]
    records = []
    for di, duration in enumerate(durations):
        for job in range(2):
            for heldout in ((4, 5), (6, 7)):
                test = (jobs == job) & np.isin(lanes, heldout)
                train = (jobs == job) & ~test
                assert set(families[test]) == set(families[train]) == set(range(4))
                for train_phase, test_phase in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2)]:
                    x = states[train, di, :, train_phase, :].reshape(-1, 24)
                    xt = states[test, di, :, test_phase, :].reshape(-1, 24)
                    # One identical state per block deliberately removes within-anchor information.
                    centers = xt.reshape(-1, 4, 24).mean(axis=1)
                    for label, target in (("raw", raw), ("mapped", mapped)):
                        y, yt = target[train].ravel(), target[test].ravel()
                        scores = fit_scores(x, y, np.concatenate([xt, centers]))
                        pred = scores[:len(xt)].argmax(-1)
                        null = np.repeat(scores[len(xt):].argmax(-1), 4)
                        null_correct = int((null == yt).sum())
                        assert null_correct * 4 == len(yt)
                        matrix = np.bincount(4 * yt + pred, minlength=16).reshape(4, 4)
                        records.append(dict(duration=duration, job=job, test_lanes=list(heldout),
                            train_phase=PHASES[train_phase], test_phase=PHASES[test_phase], label=label,
                            train_anchors=int(train.sum()), test_anchors=int(test.sum()),
                            correct=int(matrix.trace()), total=len(yt), null_correct=null_correct,
                            confusion_true_by_pred=matrix.tolist()))
    totals = defaultdict(Counter)
    for r in records:
        key = (r["duration"], r["train_phase"], r["test_phase"], r["label"])
        totals[key].update({k:r[k] for k in ("correct", "null_correct", "total")})
    by_seed = [dict(duration=d, train_phase=a, test_phase=b, label=y, **v,
                    accuracy=v["correct"]/v["total"]) for (d, a, b, y), v in totals.items()]
    return records, by_seed


def load_matched(path):
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=False)
    rows = saved["rows"]
    features = saved["x24"].numpy()
    assert features.shape == (len(rows), 24) and features.dtype == np.float64
    source_phases = ("cue", "midpoint", "first_legal")
    identities = sorted({r["anchor_id"] for r in rows})
    durations = sorted({r["duration"] for r in rows})
    anchor_indices = {key: i for i, key in enumerate(identities)}
    states = np.full((len(identities), len(durations), 4, 3, 24), np.nan, dtype=np.float64)
    anchors = [None] * len(identities)
    for index, r in enumerate(rows):
        if r["stage"] not in source_phases:
            continue
        a = anchor_indices[r["anchor_id"]]
        di = durations.index(r["duration"])
        raw = r["raw_payload"]
        pi = source_phases.index(r["stage"])
        assert FAMILY_MAPPINGS[r["family"]][raw] == r["mapped_payload"]
        assert np.isnan(states[a, di, raw, pi]).all(), "duplicate phase in a paired cell"
        states[a, di, raw, pi] = features[index]
        meta = dict(identity=r["anchor_id"], lane=r["actor_lane"], job=r["job"], family=r["family"])
        assert anchors[a] is None or anchors[a] == meta
        anchors[a] = meta
    assert len(anchors) > 0 and np.isfinite(states).all(), "missing member of factorial block"
    return anchors, states, durations, saved["protocol_hash"]


def pinned_inputs():
    paths = [PLAN, Path(__file__), ROOT/"experiments/lifelong/world.py"]
    for seed in range(3):
        folder = ROOT/"data/action-return/refit-policy-v1"/f"seed_{seed:03d}"
        paths.extend([folder/"model_own_policy.pt", folder/"result.json"])
    return {str(path.relative_to(ROOT)):sha(path) for path in paths}


def declare():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT/"protocol.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(format="content-timing-probe-declaration-v1",
                       created_utc=datetime.now(timezone.utc).isoformat(),
                       analysis_plan=PLAN.read_text(encoding="utf-8"), inputs=pinned_inputs()),
                  stream, ensure_ascii=False, indent=2)


def main():
    matched = DATA / "matched-v1"
    plan = json.loads((OUTPUT/"protocol.json").read_text(encoding="utf-8"))
    if plan["inputs"] != pinned_inputs():
        raise ValueError("probe or inherited sources changed after declaration")
    seeds = []
    for seed in range(3):
        path = matched / f"seed_{seed:03d}" / "records.pt"
        anchors, states, durations, protocol_hash = load_matched(path)
        folds, rates = evaluate(anchors, states, durations, list(range(4)))
        seeds.append(dict(seed=seed, matched_protocol_hash=protocol_hash, source=str(path.relative_to(ROOT)),
                          source_sha256=sha(path), anchors=len(anchors), folds=folds, pooled=rates))
    result = dict(format="content-timing-cross-actor-probe-v1", plan_sha256=sha(PLAN),
                  script_sha256=sha(Path(__file__)), matched_protocol_sha256=sha(matched/"protocol.json"),
                  created_utc=datetime.now(timezone.utc).isoformat(), seeds=seeds)
    path = OUTPUT / "summary.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(path)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "declare":
        declare()
    elif sys.argv[1:]:
        raise SystemExit("usage: python -m experiments.content_timing.probe [declare]")
    else:
        main()
