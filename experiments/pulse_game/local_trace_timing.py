"""Fork an identical post-taste pulse prefix into several equal-clock readouts."""

from argparse import ArgumentParser
from dataclasses import replace
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
import torch

from . import edit_receipt_graph, edit_receipt_train, local_trace_train
from . import local_trace_graph as local
from . import revisit_choice_graph as old
from . import revisit_choice_probe as probe
from . import revisit_choice_train as archived


ROOT = Path(__file__).resolve().parents[2]
OLD_PATH = ROOT / "data/revisit-choice-20261119"
LOCAL_PATH = ROOT / "data/local-trace-20261119/local"
OUT = ROOT / "docs/evidence/local-trace-2026-11-19/timing.json"
PHASES = (
    "pre_taste", "post_taste", "old_proposal_4", "neutral_4",
    "old_physical_5", "old_disabled_5", "local_write_5", "neutral_5",
    "old_receipt_6", "old_zero_receipt_6", "old_disabled_zero_6", "local_write_6", "neutral_6",
)
TICKS = (-3, 0, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6)
RETURNS = PHASES[-5:]
PHASE_PAIRS = (
    ("old_proposal_4", "neutral_4"), ("old_physical_5", "old_disabled_5"), ("local_write_5", "neutral_5"),
    ("old_receipt_6", "old_zero_receipt_6"), ("old_zero_receipt_6", "old_disabled_zero_6"),
    ("old_disabled_zero_6", "neutral_6"), ("local_write_6", "neutral_6"), ("old_receipt_6", "neutral_6"),
)
RETURN_METRICS = ("actual", "expected", "same_side_actual", "swapped_side_actual")


def digest(path):
    with Path(path).open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def _prefix_error(old_value, local_value):
    torch.testing.assert_close(old_value, local_value, rtol=2e-6, atol=1e-7)
    return float((old_value - local_value).abs().max())


@torch.no_grad()
def fork(graph, rule, tape, delay, *, keep_connected=True):
    """Apply both writers to the same birth and physical action/taste on a held-out menu."""
    left, right, draws = tape.left, -tape.left, tape.draws
    first = old.act(old.start(graph, len(left)), left, right, draws[:, 0])
    local_first = local.act(rule, local.start(graph, len(left)), left, right, draws[:, 0])
    torch.testing.assert_close(first.side, local_first.side, rtol=0, atol=0)
    taste = tape.first_good * first.side
    old_post = old.taste(first, left, right, taste)
    local_post = local.observe(rule, local_first, left, right, taste)
    error = max(_prefix_error(first.state.hidden, local_first.state.hidden),
                _prefix_error(first.motor_logit, local_first.motor_logit),
                _prefix_error(old_post.state.hidden, local_post.state.hidden))

    proposals = old.propose(old_post, draws[:, 1:5].reshape(-1, 2, 2))
    choice = old.choose(proposals, draws[:, 5])
    chosen = proposals.hidden[torch.arange(len(left)), choice.index.clamp_max(1)]
    proposal_4 = torch.where((choice.index != old.WITHHOLD)[:, None], chosen, proposals.withheld_hidden)
    physical = old.commit(old_post, proposals, choice, keep_connected=keep_connected)
    zero_receipt = old.commit(old_post, proposals, choice, keep_connected=keep_connected, receipt_sensing="zero")
    disabled = old.commit(old_post, proposals, choice, keep_connected=keep_connected,
                          intervention="disabled", receipt_sensing="zero")
    local_edit = local.commit(rule, local_post, draws[:, 1:5])

    physical_5 = old.wait(old.State(physical.state.graph, proposal_4), 1)
    disabled_5 = old.wait(old.State(disabled.state.graph, proposal_4), 1)
    neutral_4 = old.wait(old_post.state, 4)
    neutral_5 = old.wait(neutral_4, 1)
    neutral_6 = old.wait(neutral_5, 1)
    local_5 = old.wait(old.State(local_edit.state.graph, neutral_4.hidden), 1)
    local_6 = old.State(local_edit.state.graph, local_edit.state.hidden)
    _prefix_error(old.wait(physical_5, 1).hidden, zero_receipt.state.hidden)
    _prefix_error(old.wait(disabled_5, 1).hidden, disabled.state.hidden)
    _prefix_error(old.wait(local_5, 1).hidden, local_6.hidden)

    final = (physical.state, zero_receipt.state, disabled.state, local_6, neutral_6)
    torch.testing.assert_close(final[0].graph.weights, final[1].graph.weights, rtol=0, atol=0)
    torch.testing.assert_close(final[0].graph.bias, final[1].graph.bias, rtol=0, atol=0)
    assert torch.equal(final[0].graph.present, final[1].graph.present)
    hidden = torch.stack((old_post.contact_hidden, old_post.state.hidden, proposal_4, neutral_4.hidden,
                          physical_5.hidden, disabled_5.hidden, local_5.hidden, neutral_5.hidden,
                          *(state.hidden for state in final)), dim=1)
    layout = torch.where(tape.swap, -1., 1.)
    return_left = left * layout[:, None]
    revisit = [old.act(old.wait(state, delay), return_left, -return_left, draws[:, 6]) for state in final]
    return {
        "hidden": hidden.numpy(), "first_side": first.side.numpy(), "swap": tape.swap.numpy(),
        "targets": torch.stack((taste, tape.first_good, tape.first_good * layout), dim=1).numpy(),
        "return_side": torch.stack([a.side for a in revisit], dim=1).numpy(),
        "return_logit": torch.stack([a.motor_logit for a in revisit], dim=1).numpy(),
        "prefix_error": error,
    }


def split(graph, rule, tape, delay, batch, keep_connected):
    rows = []
    for start in range(0, len(tape.left), batch):
        section = slice(start, start + batch)
        selected = probe.Worlds(*(values[section] for values in vars(tape).values()))
        rows.append(fork(graph, rule, selected, delay, keep_connected=keep_connected))
    out = {name: np.concatenate([row[name] for row in rows]) for name in rows[0] if name != "prefix_error"}
    out["prefix_error"] = max(row["prefix_error"] for row in rows)
    out["weight"], out["cell_count"] = probe.balanced_weights(out["first_side"], out["swap"], out["targets"][:, 0])
    return out


def diagnose(graph, rule, birth, seed, cal_pairs, test_pairs, delay, batch, ridge, keep_connected):
    calibration = split(graph, rule, probe.worlds(seed, birth, cal_pairs, calibration=True), delay, batch, keep_connected)
    test = split(graph, rule, probe.worlds(seed, birth, test_pairs, calibration=False), delay, batch, keep_connected)
    x, y, weight = (calibration[name] for name in ("hidden", "targets", "weight"))
    head = probe.fit_readout(x, y, weight, ridge)
    shuffle = np.random.default_rng([seed, birth, probe.SHUFFLE_DOMAIN]).choice((-1, 1), size=cal_pairs)
    null = probe.fit_readout(x, y * shuffle.repeat(4)[:, None], weight, ridge)

    def score(readout):
        return probe.accuracy(probe.score_readout(test["hidden"], readout), test["targets"], test["weight"])

    target = test["targets"][:, 2, None]
    actual = test["return_side"] == target
    expected = .5 * (1 + np.tanh(.5 * test["return_logit"] * target))
    cells = [test["weight"] * (test["swap"] == swapped) for swapped in (False, True)]
    natural = np.stack((test["weight"] @ actual, test["weight"] @ expected,
                        *((weight / weight.sum()) @ actual for weight in cells)), axis=1)
    return {"linear": score(head), "shuffled_linear": score(null), "natural": natural,
            "prefix_error": max(calibration["prefix_error"], test["prefix_error"])}, {
        "calibration": calibration["cell_count"].tolist(), "heldout": test["cell_count"].tolist(),
    }


def _interval(values, draw):
    interval = np.quantile(values[draw].mean(axis=1), [.025, .975], axis=0)
    return {"mean": values.mean(axis=0).tolist(), "ci95": interval.T.tolist()}


def summarize(records, draws):
    linear = np.stack([record["linear"] for record in records])
    natural = np.stack([record["natural"] for record in records])
    return {
        "linear": dict(zip(PHASES, linear.mean(0).tolist(), strict=True)),
        "shuffled_linear": dict(zip(PHASES, np.mean([r["shuffled_linear"] for r in records], axis=0).tolist(), strict=True)),
        "natural": {name: _interval(natural[:, idx], draws) for idx, name in enumerate(RETURNS)},
        "linear_contrasts": {f"{a} minus {b}": _interval(linear[:, PHASES.index(a)] - linear[:, PHASES.index(b)], draws)
                             for a, b in PHASE_PAIRS},
        "natural_contrasts": {f"{a} minus {b}": _interval(natural[:, RETURNS.index(a)] - natural[:, RETURNS.index(b)], draws)
                              for a, b in PHASE_PAIRS if a in RETURNS and b in RETURNS},
        "max_prefix_abs_delta": max(record["prefix_error"] for record in records),
    }


def run(old_path, local_path, out, *, local_step=None, seed=20261127, cal_pairs=2048,
        test_pairs=4096, delay=4, batch=768, ridge=.1, bootstrap=20_000):
    if min(cal_pairs, test_pairs) < 2 or min(batch, bootstrap) < 1 or delay < 0 or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("Require valid calibration, holdout, clock and ridge settings")
    old_path, local_path, out = (Path(path) for path in (old_path, local_path, out))
    old_plan = json.loads((old_path / "run.json").read_text(encoding="utf8"))
    local_plan = json.loads((local_path / "run.json").read_text(encoding="utf8"))
    old_step = old_plan["config"]["updates"]
    if local_step is None:
        local_step = local_plan["config"]["updates"]
    births = old_plan["births"]
    if set(births) != set(local_plan["births"]) or local_step not in local_plan["checkpoints"]:
        raise ValueError("Expected matching independent roots and a listed local checkpoint")
    old_files = {r: edit_receipt_train.checkpoint_path(old_path, "ranked", r, old_step) for r in births}
    local_files = {r: local_trace_train.checkpoint_path(local_path, "odor", "active", r, local_step) for r in births}
    for file in (*old_files.values(), *local_files.values()):
        if not file.is_file():
            raise FileNotFoundError(file)

    torch.set_num_threads(1)
    picks = np.random.default_rng([seed, 0x71F4]).integers(0, len(births), size=(bootstrap, len(births)))
    results = {
        "schema": 1, "seed": seed, "births": births, "calibration_worlds_per_birth": cal_pairs,
        "heldout_worlds_per_birth": test_pairs, "lanes_per_world": 4, "ridge": ridge,
        "wait_after_tick_6": delay, "bootstrap_draws": bootstrap, "interval_unit": "paired independent birth",
        "ticks_from_taste": dict(zip(PHASES, TICKS, strict=True)), "linear_targets": probe.TARGETS,
        "natural_metrics": RETURN_METRICS, "natural_return_tick": 6 + delay + local.REACTION_TICKS,
        "phases": {
            "old_proposal_4": "Selected candidate activity or native withhold activity before any write.",
            "old_physical_5": "One neutral step through old writer's committed graph.",
            "old_disabled_5": "Same selected activity and clock with physical commit suppressed.",
            "old_receipt_6": "One step through the physically committed graph with actual old edit receipt.",
            "old_zero_receipt_6": "Same physically committed graph and activity with zero receipt pulse.",
            "old_disabled_zero_6": "Same selected activity with disabled write and zero receipt pulse.",
            "local_write_5_and_6": "Local rule writes after four neutral steps, followed by neutral activity.",
            "neutral": "Keep the birth graph and advance through neutral inputs.",
        },
        "source_sha256": {Path(file).name: digest(file) for file in (
            __file__, old.__file__, edit_receipt_graph.__file__, old.fw.__file__, local.__file__,
            probe.__file__, archived.__file__, local_trace_train.__file__,
        )},
        "origins": {},
    }
    for origin, native, step, files in (("archived_ranked_odor", "old", old_step, old_files),
                                        ("local_active_odor", "local", local_step, local_files)):
        rows, cells = [], []
        for birth in births:
            trained_local, rule = local_trace_train.load_checkpoint(local_path, "odor", "active", birth, local_step)
            graph = archived.load_checkpoint(old_path, "ranked", birth, old_step) if native == "old" else trained_local
            row, count = diagnose(graph, rule, birth, seed, cal_pairs, test_pairs, delay, batch, ridge,
                                  old_plan["config"]["keep_connected"])
            rows.append(row)
            cells.append(count)
            print(f"timing {origin} birth={birth} prefix_abs_delta={row['prefix_error']:.3g}", flush=True)
        results["origins"][origin] = {
            "checkpoint_update": step, "native_writer": native,
            "cross_applied_writer": "local" if native == "old" else "old",
            "checkpoints_sha256": {str(r): digest(files[r]) for r in births},
            "local_rule_checkpoint_update": local_step,
            "local_rule_checkpoints_sha256": {str(r): digest(local_files[r]) for r in births},
            "by_birth": {str(birth): {**{key: np.asarray(value).tolist() for key, value in row.items()}, "cells": count}
                         for birth, row, count in zip(births, rows, cells, strict=True)},
            "summary": summarize(rows, picks),
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf8")
    return results


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=OLD_PATH)
    parser.add_argument("--local", type=Path, default=LOCAL_PATH)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--local-step", type=int)
    parser.add_argument("--seed", type=int, default=20261127)
    parser.add_argument("--calibration-pairs", type=int, default=2048)
    parser.add_argument("--test-pairs", type=int, default=4096)
    args = parser.parse_args()
    run(args.old, args.local, args.out, local_step=args.local_step, seed=args.seed,
        cal_pairs=args.calibration_pairs, test_pairs=args.test_pairs)


if __name__ == "__main__":
    main()
