"""Matched lifetime runs after all outer-loop parameter learning has stopped."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import self_write_core as core


EVIDENCE = Path("docs/evidence/self-writing-2026-11-19/core")
DATA = Path("data/pulse-game/self-writing-20261119")
REGIMES = ("stationary", "changing")
ROOTS = tuple(range(101, 109))
WORLDS = 24
CONTACTS = 192
WORLD_SEED = 83231001


def load_graph(path):
    with np.load(path) as data:
        return core.Graph(data["w"], data["b"], data["mask"])


def worlds(root, count=WORLDS, contacts=CONTACTS, seed=WORLD_SEED):
    """The same cue order, presentations, initial labels and first shock in both regimes."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, root]))
    basis = np.linalg.qr(rng.normal(size=(count, 3, 3)))[0].transpose(0, 2, 1)
    initial_preference = rng.choice([-1, 1], size=(count, 3))
    category = np.stack([rng.permutation(np.tile(np.arange(3), contacts // 3))
                         for _ in range(count)], axis=1)
    orientation = rng.choice([-1, 1], size=(contacts, count))
    left = (1.35 * basis[np.arange(count), category] * orientation[:, :, None]
            + rng.normal(0.0, 0.035, size=(contacts, count, 3))).astype(np.float32)
    right = (-1.35 * basis[np.arange(count), category] * orientation[:, :, None]
             + rng.normal(0.0, 0.035, size=(contacts, count, 3))).astype(np.float32)
    initial_side = (-initial_preference[np.arange(count), category] * orientation).astype(np.int8)
    reported_stationary = initial_side.copy()
    noisy_time, noisy_lane = np.where(rng.random(initial_side.shape) < 0.22)
    late = noisy_time > 48
    reported_stationary[noisy_time[late], noisy_lane[late]] *= -1
    reported_stationary[48] *= -1
    changing = initial_side.copy()
    changing[48:96] *= -1
    changing[144:] *= -1
    keys = rng.integers(2**32, size=contacts, dtype=np.uint64)
    shuffles = np.stack([np.roll(np.arange(count), rng.integers(1, count))
                         for _ in range(contacts)])
    return {
        "left": left, "right": right, "category": category, "basis": basis,
        "initial_preference": initial_preference, "orientation": orientation,
        "hidden": {"stationary": initial_side, "changing": changing},
        "reported": {"stationary": reported_stationary, "changing": changing},
        "keys": keys, "shuffles": shuffles,
    }


def quiet_probe(graph, h, world, preference_flip):
    basis = (1.35 * world["basis"]).astype(np.float32)
    lane_scores = []
    for cat in range(basis.shape[1]):
        good = world["initial_preference"][:, cat] * preference_flip
        for order in (-1, 1):
            cues = order * basis[:, cat]
            evaluated = core.advance(graph, h, core.sense(cues, -cues), core.MOTOR_TICKS)
            side = np.where(core.motor(evaluated) > 0, 1, -1)
            lane_scores.append(side == -good * order)
    return np.mean(lane_scores, axis=0)


def stack_log(log):
    return {name: np.stack([row[name] for row in log]) for name in log[0]}


def run_world(birth, world, regime, arm, view="own", *, keep_log=True, snapshots=True,
              edit_scale=1.0, sparse=False, step_fn=core.step):
    """No optimizer. The runtime only advances dynamics and applies emitted writes."""
    left, right = world["left"], world["right"]
    count = left.shape[1]
    graph = core.repeated_graph(birth, count)
    h = core.rest(graph)
    audit_rows = []
    edit_rows = []
    checkpoints = {}
    for time in range(len(left)):
        truth = world["hidden"][regime][time]
        good_reported = world["reported"][regime][time]
        tastes = np.stack((-good_reported, good_reported), axis=1).astype(np.float32)
        prior = graph
        graph, h, side, tasted, edit, receipt = step_fn(
            graph, h, left[time], right[time], tastes, world["keys"][time],
            arm=arm, view=view, shuffle_indices=world["shuffles"][time], edit_scale=edit_scale, sparse=sparse,
        )
        if keep_log:
            edit_rows.append({
                "proposed_edge_delta": edit.edge_delta, "proposed_bias_delta": edit.bias_delta,
                "effective_edge_delta": graph.w - prior.w, "effective_bias_delta": receipt.bias_delta,
                "executed_tuning_edge_delta": receipt.edge_delta,
                "birth_address": edit.birth.astype(np.int16), "retire_address": edit.retire.astype(np.int16),
                "birth_probability": edit.birth_probability, "retire_probability": edit.retire_probability,
                "proposed_birth_value": edit.birth_value,
                "retired_value": receipt.retired_value,
                "birth_status": receipt.birth_status, "retire_status": receipt.retire_status,
            })
        audit_rows.append({
            "side": side.astype(np.int8), "actual_taste": tasted.astype(np.int8),
            "correct": side == truth, "reported_correct": side == good_reported,
            "edge_count": receipt.edge_count.astype(np.int16),
            "births": receipt.birth_status == 1, "retirements": receipt.retire_status == 1,
            "writer_births": (receipt.birth_status == 1) & (edit.birth >= core.N * core.WRITER.start),
            "writer_retirements": ((receipt.retire_status == 1)
                                   & (edit.retire >= core.N * core.WRITER.start)),
            "writer_edge_updates": np.count_nonzero(receipt.edge_delta[:, core.WRITER], axis=(1, 2)),
            "normal_edge_updates": np.count_nonzero(receipt.edge_delta[:, :core.WRITER.start], axis=(1, 2)),
            "writer_bias_updates": np.count_nonzero(receipt.bias_delta[:, core.WRITER], axis=1),
            "normal_bias_updates": np.count_nonzero(receipt.bias_delta[:, :core.WRITER.start], axis=1),
        })
        if snapshots and (time + 1) % 48 == 0:
            phase = (time // 48) % 2
            flip = -1 if regime == "changing" and phase else 1
            label = f"contact_{time + 1}"
            checkpoints.update({
                f"{label}_w": graph.w.copy(), f"{label}_b": graph.b.copy(),
                f"{label}_mask": graph.mask.copy(), f"{label}_h": h.copy(),
                f"{label}_quiet_rest": quiet_probe(graph, core.rest(graph), world, flip),
                f"{label}_quiet_live": quiet_probe(graph, h, world, flip),
            })
    return graph, h, stack_log(audit_rows), stack_log(edit_rows) if keep_log else {}, checkpoints


def study(root, checkpoint, count, contact_count, save, edit_scale=1.0, data_path=DATA,
          view="own", sparse=False):
    birth = load_graph(checkpoint)
    world = worlds(root, count, contact_count)
    run_path = data_path / str(root)
    if save:
        run_path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(run_path / "physical-worlds.npz", left=world["left"], right=world["right"],
                            category=world["category"], basis=world["basis"],
                            initial_preference=world["initial_preference"], orientation=world["orientation"],
                            stationary_truth=world["hidden"]["stationary"],
                            changing_truth=world["hidden"]["changing"],
                            stationary_report=world["reported"]["stationary"],
                            changing_report=world["reported"]["changing"],
                            keys=world["keys"], shuffles=world["shuffles"])
    summaries = {}
    audit_by_arm = {}
    for arm in core.ARMS:
        for regime in REGIMES:
            endpoint, h, audit, edits, checkpoints = run_world(
                birth, world, regime, arm, view=view, keep_log=save, snapshots=True,
                edit_scale=edit_scale, sparse=sparse)
            key = f"{regime}-{arm}"
            summaries[key] = summarize(birth, endpoint, audit, checkpoints)
            audit_by_arm[(arm, regime)] = audit
            if save:
                np.savez_compressed(run_path / f"{key}-receipts.npz", **audit, **edits)
                np.savez_compressed(run_path / f"{key}-snapshots.npz", w=endpoint.w, b=endpoint.b,
                                    mask=endpoint.mask, h=h, **checkpoints)
        stationary = audit_by_arm[(arm, "stationary")]
        changing = audit_by_arm[(arm, "changing")]
        for field in ("side", "actual_taste", "edge_count"):
            if not np.array_equal(stationary[field][:49], changing[field][:49]):
                raise AssertionError(f"First surprise mismatch for {arm=} {field=}")
    return summaries


def summarize(birth, endpoint, audit, checkpoints):
    correct = audit["correct"]
    counts = audit["edge_count"]
    return {
        "baseline_terminal": float(correct[24:48].mean()),
        "shock_early": float(np.mean([correct[start:start + 3].mean()
                                     for start in (48, 96, 144)])),
        "shock_late": float(np.mean([correct[start + 24:start + 48].mean()
                                    for start in (48, 96, 144)])),
        "first_shock_early": float(correct[48:51].mean()),
        "terminal": float(correct[-24:].mean()),
        "terminal_quiet_rest": float(checkpoints["contact_192_quiet_rest"].mean()),
        "terminal_quiet_live": float(checkpoints["contact_192_quiet_live"].mean()),
        "closest_edge_count": [int(counts.min()), int(counts.max())],
        "mean_edge_count_endpoint": float(counts[-1].mean()),
        "all_strongly_connected": all(map(core.connected, endpoint.mask)),
        "same_budget_contacts": int(correct.shape[0]),
        "unchanged_writer_mask_lanes": int(np.all(endpoint.mask[:, core.WRITER]
                                                 == birth.mask[:, core.WRITER], axis=(1, 2)).sum()),
        "support_hamming_mean": float((endpoint.mask != birth.mask).sum(axis=(1, 2)).mean()),
        "writer_abs_weight_change_mean": float(np.abs(endpoint.w[:, core.WRITER]
                                                       - birth.w[:, core.WRITER]).mean()),
        "writer_abs_bias_change_mean": float(np.abs(endpoint.b[:, core.WRITER]
                                                     - birth.b[:, core.WRITER]).mean()),
        "positive_clipped_weight_fraction": float(np.mean(endpoint.w == core.LIMIT)),
        "negative_clipped_weight_fraction": float(np.mean(endpoint.w == -core.LIMIT)),
        **{field: int(audit[field].sum()) for field in
           ("births", "retirements", "writer_births", "writer_retirements",
            "writer_edge_updates", "normal_edge_updates", "writer_bias_updates", "normal_bias_updates")},
    }


def aggregate(rows):
    """Exploit matched birth roots for regime and intervention contrasts."""
    rng = np.random.default_rng(9119138)
    keys = tuple(next(iter(rows.values())))
    scalar_fields = ("terminal", "terminal_quiet_rest", "terminal_quiet_live",
                     "baseline_terminal", "first_shock_early", "shock_early", "shock_late")
    sample = rng.integers(0, len(rows), size=(10006, len(rows)))

    def block(values):
        values = np.asarray(values)
        interval = np.quantile(values[sample].mean(axis=1), [0.025, 0.975])
        return {"mean": float(values.mean()), "paired_root_95_interval": interval.tolist()}

    table = {key: {field: block([row[key][field] for row in rows.values()])
                   for field in scalar_fields} for key in keys}
    contrasts = {}
    for other in ("primitive", "none", "shuffle"):
        for field in scalar_fields:
            by_regime = {r: np.array([row[f"{r}-active"][field] - row[f"{r}-{other}"][field]
                                      for row in rows.values()]) for r in REGIMES}
            contrasts[f"active_minus_{other}.{field}"] = {
                **{r: block(v) for r, v in by_regime.items()},
                "interaction_changing_minus_stationary": block(by_regime["changing"]
                                                                 - by_regime["stationary"]),
            }
    return {"regime_arm": table, "contrasts": contrasts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=int, nargs="+", default=ROOTS)
    parser.add_argument("--worlds", type=int, default=WORLDS)
    parser.add_argument("--prefix", default="bootstrap")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--edit-scale", type=float, choices=(1.0, 0.12), default=1.0)
    args = parser.parse_args()
    variant = "lifetime" if args.edit_scale == 1.0 else "lifetime-reduced-amplitude"
    data_path = DATA if args.edit_scale == 1.0 else DATA / "reduced-amplitude"
    rows = {}
    for root in args.roots:
        checkpoint = EVIDENCE / f"{args.prefix}-own-{root}.npz"
        rows[root] = study(root, checkpoint, args.worlds, CONTACTS, save=not args.pilot,
                           edit_scale=args.edit_scale, data_path=data_path)
        print(f"{root} " + json.dumps({
            key: [round(row["terminal"], 4), row["births"], row["retirements"],
                  round(row["writer_abs_weight_change_mean"], 4)] for key, row in rows[root].items()
        }), flush=True)
    if not args.pilot:
        files = sorted(data_path.glob("*/*.npz"))
        manifest = {str(path.as_posix()): hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
                    for path in files}
        (EVIDENCE / f"{variant}-summary.json").write_text(
            json.dumps({"edit_scale": args.edit_scale, "roots": rows, **aggregate(rows), "files_sha256": manifest}, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
