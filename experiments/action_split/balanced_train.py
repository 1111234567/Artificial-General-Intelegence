"""One post-hoc mode-class normalization control, with the first run retained.

Only mode CE changes: average the three within-class mean CEs. Frozen base
training code is reused through temporary process-local bindings. The origin,
zero split heads, public stream, conditional losses, Adam and budget are fixed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from experiments.action_split import data, train as base
from experiments.action_split.model import split_loss as original_loss
from experiments.continuity.run import state_digest

OUT = ROOT / "data/action-split/balanced-v1"
PLAN = OUT / "protocol.json"
FORMAT = "action-split-balanced-training-v1"
PARENT_OUT, PARENT_PLAN = base.OUT, base.PLAN
PARENT_SOURCES = tuple(base.sources())
PARENT_COUNTS = base.summarize_rows
ARMS, SEEDS, SNAPSHOTS = base.ARMS, base.SEEDS, base.SNAPSHOTS
WARM, BLOCK, UPDATES = base.WARM, base.BLOCK, base.UPDATES


def balanced_loss(scores, mode_target, begin_target, finish_target, active,
                  wait_weight=.25):
    """Three class means each contribute one third; conditional terms fixed."""
    if wait_weight != .25:
        raise ValueError("the normalization control has no tunable class weights")
    flat_mode, labels = scores.reshape(-1, 13)[:, :3], mode_target.reshape(-1)
    masks = [labels == value for value in (0, 1, 2)]
    if not all(bool(mask.any()) for mask in masks):
        raise ValueError("every control training block must contain all three mode classes")
    result = original_loss(scores, mode_target, begin_target, finish_target, active)
    means = [F.cross_entropy(flat_mode[mask], labels[mask]) for mask in masks]
    mode = (means[0] + means[1] + means[2]) / 3.
    result.update(mode=mode, total=mode + result["begin"] + result["finish"],
                  mode_weight_sum=scores.new_tensor(3.))
    return result


def summarize_rows(stream, start, stop):
    counts = PARENT_COUNTS(stream, start, stop)
    if not all(counts[key] > 0 for key in ("mode_WAIT_rows", "mode_BEGIN_rows", "mode_FINISH_rows")):
        raise ValueError("every control training block must contain all three mode classes")
    # Inverse class-count row weights sum to 3; division by 3 averages
    # the class means. Raw supervision counts retain their original meaning.
    counts["mode_weight_sum"] = 3.
    return counts


def checkpoint_path(seed, arm, step=256):
    assert int(seed) in SEEDS and arm in ARMS and int(step) in SNAPSHOTS
    return OUT / f"seed_{int(seed):03d}" / arm / f"step{int(step):03d}.pt"


@contextmanager
def bindings():
    updates = dict(OUT=OUT, PLAN=PLAN, FORMAT=FORMAT, checked=checked,
                   split_loss=balanced_loss, summarize_rows=summarize_rows)
    saved = {name: getattr(base, name) for name in updates}
    try:
        for name, value in updates.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(base, name, value)


def verify_parent():
    plan = data.read(PARENT_PLAN)
    assert state_digest({k: v for k, v in plan.items() if k != "protocol_hash"}) == plan["protocol_hash"]
    for path, expected in plan["source_sha256"].items():
        assert base.sha(path) == expected, path
    for items in plan["input_sha256"].values():
        for path, expected in items.items():
            assert base.sha(path) == expected, path
    return plan


def declare(extra_sources=()):
    if PLAN.exists():
        raise FileExistsError(PLAN)
    base.deterministic(0)
    assert torch.cuda.is_available()
    assert Path(sys.executable).resolve() == (ROOT / ".venv/Scripts/python.exe").resolve()
    data.checked()
    parent = verify_parent()
    diagnostic = ROOT / "data/action-split/diagnostic-v1"
    diagnosis = data.read(diagnostic / "summary.json")
    failed_teacher = {}
    for kind in ("training_teacher", "heldout_teacher"):
        reports = [r for r in diagnosis["reports"] if r["arm"] == "full" and r["history"] == kind]
        assert len(reports) == 3
        failed_teacher[kind] = dict(legal_rows=sum(r["groups"]["legal"]["rows"] for r in reports),
            FINISH_mode_argmax=sum(r["groups"]["legal"]["mode_FINISH_argmax"] for r in reports))
        assert failed_teacher[kind]["FINISH_mode_argmax"] == 0
    upstream = [PARENT_PLAN, PARENT_OUT / "globalfreeze.json",
        ROOT / "data/action-split/eval-v1/protocol.json", ROOT / "data/action-split/eval-v1/freeze.json",
        ROOT / "data/action-split/eval-v1/completion.json", diagnostic / "protocol.json", diagnostic / "summary.json"]
    expected_counts, minimum_counts = {}, {}
    for seed in SEEDS:
        stream = data.load_stream(seed, keys=base.STREAM_FIELDS)
        base.validate_stream(stream, WARM, BLOCK, UPDATES)
        blocks = [summarize_rows(stream, WARM + step * BLOCK, WARM + (step + 1) * BLOCK)
                  for step in range(UPDATES)]
        expected_counts[str(seed)] = {key: sum(row[key] for row in blocks) for key in blocks[0]}
        minimum_counts[str(seed)] = {key: min(row[key] for row in blocks)
            for key in ("mode_WAIT_rows", "mode_BEGIN_rows", "mode_FINISH_rows")}
        for key, value in parent["expected_training_rows"][str(seed)].items():
            if key != "mode_weight_sum":
                assert expected_counts[str(seed)][key] == value
    sources = [Path(__file__), Path(__file__).with_name("balanced_test.py"), *PARENT_SOURCES,
               *map(Path, extra_sources)]
    plan = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
        design="One explicitly post-hoc class-normalization control, chosen after retaining the failed original mode-loss run and diagnosing low FINISH recall even on teacher histories. No sweep.",
        known_parent_results=dict(primary_full_strict_completion=0, diagnostic_full=failed_teacher,
            note="Known results motivated this single control; they are not confirmatory evidence for class imbalance as the sole cause."),
        seeds=list(SEEDS), arms=list(ARMS), snapshots=list(SNAPSHOTS), runtime=base.runtime(),
        source_sha256={str(path.resolve()): base.sha(path) for path in sources},
        input_sha256=parent["input_sha256"], upstream_sha256={str(path): base.sha(path) for path in upstream},
        data_protocol_sha256=parent["data_protocol_sha256"], data_protocol_hash=parent["data_protocol_hash"],
        parent_training_protocol_hash=parent["protocol_hash"],
        initial_model=parent["initial_model"], source_policy=parent["source_policy"],
        initialization="Original seed-matched midpoint B008 core only plus exactly-zero new heads; NOT a continuation of the failed step256 control.",
        optimizer=parent["optimizer"], stream=parent["stream"],
        loss=dict(mode="(mean CE on WAIT rows + mean CE on BEGIN rows + mean CE on FINISH rows) / 3, independently within each128tick block; all3 classes required, no empty-class fallback",
            begin=parent["loss"]["begin"], finish=parent["loss"]["finish"],
            coefficients=dict(mode=1., begin=1., finish=1.), eligible_filter=False,
            class_weights="inverse actual within-block class counts, followed by division by3; fixed formula, no tuning"),
        counters=dict(raw_supervision_counts="All mode/BEGIN/FINISH content row counts identical to parent",
            mode_weight_sum="Sum of inverse-class-count row weights =3 per update,768 total; a reduction denominator counter, not additional supervision"),
        expected_training_rows=expected_counts, minimum_class_rows_per_block=minimum_counts,
        implementation="Frozen base train.py reused only through process-local OUT/PLAN/FORMAT/checked/loss/row-counter bindings. All parent sources and artifacts unchanged.",
        checkpoint=parent["checkpoint"],
        no_adaptation="Exactly six fixed continuations: frozen/full x3seeds, same stream/order/256updates; no LR, coefficient, threshold, sampling or epoch search. All6 endpoints freeze before own-action scoring. Original negative results retained.")
    plan["protocol_hash"] = state_digest(plan)
    base.write_new(PLAN, plan)
    print(json.dumps(dict(event="balanced_declared", path=str(PLAN), protocol_hash=plan["protocol_hash"])), flush=True)
    return plan


def checked():
    plan = data.read(PLAN)
    assert state_digest({k: v for k, v in plan.items() if k != "protocol_hash"}) == plan["protocol_hash"]
    assert base.sha(data.PLAN) == plan["data_protocol_sha256"]
    data.checked()
    for section in ("source_sha256", "upstream_sha256"):
        for path, expected in plan[section].items():
            assert base.sha(path) == expected, path
    for items in plan["input_sha256"].values():
        for path, expected in items.items():
            assert base.sha(path) == expected, path
    return plan


def run(seed):
    checked()
    with bindings():
        base.run(seed)


def freeze_endpoints():
    checked()
    with bindings():
        return base.freeze_endpoints()


def all_seeds():
    checked()
    workers = []
    for seed in SEEDS:
        log = (OUT / f"seed_{seed:03d}.log").open("x", encoding="utf-8")
        process = subprocess.Popen([sys.executable, "-u", "-m", "experiments.action_split.balanced_train", "run", "--seed", str(seed)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        workers.append((seed, process, log))
    results = []
    for seed, process, log in workers:
        results.append(dict(seed=seed, exit_code=process.wait()))
        log.close()
    base.write_new(OUT / "exit_codes.json", results)
    if any(result["exit_code"] for result in results):
        raise RuntimeError(results)
    freeze_endpoints()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "run", "all", "freeze"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--extra-source", action="append", default=[])
    args = parser.parse_args()
    if args.command == "declare":
        declare(args.extra_source)
    elif args.command == "all":
        all_seeds()
    elif args.command == "freeze":
        freeze_endpoints()
    else:
        if args.seed is None:
            parser.error("run requires --seed")
        run(args.seed)


if __name__ == "__main__":
    main()
