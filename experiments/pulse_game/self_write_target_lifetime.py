"""Matched eight-contact and long lifetime tests without an external optimizer."""
import argparse
import hashlib
import json

import numpy as np

from . import self_write_core as core
from . import self_write_core_lifetime as life
from . import self_write_target as target


WORLD_SEED = 92110419
PAIR_WORLDS = 96
PAIR_CONTACTS = 8
DATA = life.DATA / "target-value"
ARMS = {
    "own": ("own", "own", "active"),
    "own_masked": ("own", "blind", "active"),
    "blind": ("blind", "blind", "active"),
    "primitive": ("own", "own", "primitive"),
    "none": ("own", "own", "none"),
}


def pairs(rng, count):
    odors = rng.normal(size=(count, 3))
    odors = (1.35 * odors / np.linalg.norm(odors, axis=1, keepdims=True)).astype(np.float32)
    good = rng.choice(np.array([-1, 1], dtype=np.int8), size=count)
    return odors, good


def quiet_pair(graph, odor, good):
    results = []
    for sign in (-1, 1):
        cue = sign * odor
        h = core.advance(graph, core.rest(graph), core.sense(cue, -cue), core.MOTOR_TICKS)
        choice = np.where(core.motor(h) > 0, 1, -1)
        results.append(choice == -sign * good)
    return np.stack(results)


def run_simple(birth, root, view, arm):
    rng = np.random.default_rng(np.random.SeedSequence([target.PAIR_SEED, root]))
    odor, good = pairs(rng, PAIR_WORLDS)
    orientations = rng.choice(np.array([-1, 1], dtype=np.int8),
                              size=(PAIR_CONTACTS, PAIR_WORLDS))
    keys = rng.integers(2**32, size=PAIR_CONTACTS, dtype=np.uint64)
    graph = core.repeated_graph(birth, PAIR_WORLDS)
    h = core.rest(graph)
    logs = []
    silent = [quiet_pair(graph, odor, good)]
    for index, orientation in enumerate(orientations):
        cue = orientation[:, None] * odor
        expected = -orientation * good
        tastes = np.stack((-expected, expected), axis=1).astype(np.float32)
        prior = graph
        graph, h, side, tasted, edit, receipt = target.step(
            graph, h, cue, -cue, tastes, keys[index], arm=arm, view=view)
        logs.append({
            "side": side, "taste": tasted, "physical_correct": side == expected,
            "proposed_edge_delta": edit.edge_delta, "proposed_bias_delta": edit.bias_delta,
            "executed_tuning_edge_delta": receipt.edge_delta,
            "effective_edge_delta": graph.w - prior.w, "effective_bias_delta": receipt.bias_delta,
            "birth_address": edit.birth, "retire_address": edit.retire,
            "birth_status": receipt.birth_status, "retire_status": receipt.retire_status,
            "birth_probability": edit.birth_probability, "retire_probability": edit.retire_probability,
            "proposed_birth_value": edit.birth_value, "retired_value": receipt.retired_value,
        })
        silent.append(quiet_pair(graph, odor, good))
    return {
        "odor": odor, "good": good, "orientations": orientations, "keys": keys,
        "quiet": np.stack(silent), "w": graph.w, "b": graph.b, "mask": graph.mask, "h": h,
        **life.stack_log(logs),
    }


def endpoint_pair(birth, endpoint, root, view, arm):
    count = len(endpoint.w)
    rng = np.random.default_rng(np.random.SeedSequence([target.PAIR_SEED, root, 6]))
    odors, good = pairs(rng, count)
    sign = rng.choice(np.array([-1, 1], dtype=np.int8), size=count)
    key = rng.integers(2**32, dtype=np.uint64)
    expected = -sign * good
    tastes = np.stack((-expected, expected), axis=1).astype(np.float32)
    cue = odors * sign[:, None]
    result = {}
    for label, graph in (("birth", core.repeated_graph(birth, count)), ("mature", endpoint)):
        updated, _, _, _, _, _ = target.step(
            graph, core.rest(graph), cue, -cue, tastes, key, arm=arm, view=view)
        result[f"{label}_before"] = quiet_pair(graph, odors, good)
        result[f"{label}_after"] = quiet_pair(updated, odors, good)
    return result


def summarize_deployed(roots):
    rng = np.random.default_rng(92110422)
    samples = rng.integers(0, len(roots), size=(10006, len(roots)))

    def metric(values):
        values = np.asarray(values)
        ci = np.quantile(values[samples].mean(axis=1), [0.025, 0.975])
        return {"mean": float(values.mean()), "paired_root_95_interval": ci.tolist()}

    by_arm = {}
    for arm in ARMS:
        by_arm[arm] = {
            "simple_quiet_curve": [metric([run[arm]["simple_quiet_curve"][index] for run in roots.values()])
                                   for index in range(PAIR_CONTACTS + 1)],
            **{regime: {
                field: metric([run[arm][regime][field] for run in roots.values()])
                for field in ("terminal", "terminal_quiet_rest", "terminal_quiet_live",
                              "baseline_terminal", "first_shock_early", "shock_late",
                              "fresh_pair_mature_before", "fresh_pair_mature_after", "fresh_pair_birth_before",
                              "fresh_pair_birth_after", "fresh_pair_mature_gain", "fresh_pair_gain_change",
                              "positive_clipped_weight_fraction", "negative_clipped_weight_fraction",
                              "writer_abs_weight_change_mean")
            } for regime in life.REGIMES},
        }
    differences = {}
    for arm in ("own_masked", "blind", "primitive", "none"):
        key = f"own_minus_{arm}"
        differences[key] = {
            "simple_quiet_contact_8": metric([run["own"]["simple_quiet_curve"][-1]
                                             - run[arm]["simple_quiet_curve"][-1] for run in roots.values()])
        }
        for field in ("terminal", "terminal_quiet_rest", "fresh_pair_mature_gain"):
            contrast = {
                regime: np.array([run["own"][regime][field] - run[arm][regime][field]
                                  for run in roots.values()]) for regime in life.REGIMES
            }
            differences[key][field] = {**{regime: metric(value) for regime, value in contrast.items()},
                                       "interaction": metric(contrast["changing"] - contrast["stationary"])}
    return {"condition": by_arm, "contrasts": differences}


def study(root, folder):
    with_bootstrap = {
        view: life.load_graph(target.EVIDENCE / f"bootstrap-{view}-{root}.npz")
        for view in ("own", "blind")
    }
    world = life.worlds(root, seed=WORLD_SEED)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        folder / "world.npz", left=world["left"], right=world["right"], basis=world["basis"],
        category=world["category"], orientation=world["orientation"], keys=world["keys"],
        shuffles=world["shuffles"], initial_preference=world["initial_preference"],
        **{f"{regime}_{field}": world[field][regime]
           for regime in life.REGIMES for field in ("hidden", "reported")},
    )
    results = {}
    for label, (training, view, arm) in ARMS.items():
        birth = with_bootstrap[training]
        simple = run_simple(birth, root, view, arm)
        np.savez_compressed(folder / f"{label}-simple.npz", **simple)
        results[label] = {"simple_quiet_curve": simple["quiet"].mean(axis=(1, 2)).tolist()}
        arms = {}
        for regime in life.REGIMES:
            endpoint, h, audit, edits, snapshots = life.run_world(
                birth, world, regime, arm, view=view, step_fn=target.step,
            )
            arms[regime] = audit
            fresh = endpoint_pair(birth, endpoint, root, view, arm)
            results[label][regime] = life.summarize(birth, endpoint, audit, snapshots)
            results[label][regime].update({f"fresh_pair_{key}": float(values.mean())
                                           for key, values in fresh.items()})
            results[label][regime]["fresh_pair_mature_gain"] = float(
                (fresh["mature_after"].astype(int) - fresh["mature_before"]).mean())
            results[label][regime]["fresh_pair_gain_change"] = float(
                results[label][regime]["fresh_pair_mature_gain"] -
                (fresh["birth_after"].astype(int) - fresh["birth_before"]).mean())
            np.savez_compressed(folder / f"{label}-{regime}-receipts.npz", **audit, **edits)
            np.savez_compressed(folder / f"{label}-{regime}-snapshots.npz",
                                w=endpoint.w, b=endpoint.b, mask=endpoint.mask, h=h,
                                **snapshots, **{f"fresh_pair_{key}": arr for key, arr in fresh.items()})
        for field in ("side", "actual_taste", "edge_count"):
            np.testing.assert_array_equal(arms["stationary"][field][:49], arms["changing"][field][:49])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=int, nargs="+", default=target.ROOTS)
    args = parser.parse_args()
    rows = {}
    for root in args.roots:
        rows[root] = study(root, DATA / str(root))
        print(root, {
            key: [round(row["simple_quiet_curve"][-1], 4),
                  *[round(row[regime]["terminal"], 4) for regime in life.REGIMES]]
            for key, row in rows[root].items()
        }, flush=True)
    summary = target.EVIDENCE / "lifetime-summary.json"
    sources = [*sorted(target.EVIDENCE.glob("bootstrap-*.npz")),
               *sorted(DATA.glob("*/*.npz"))]
    hashes = {}
    for path in sources:
        with path.open("rb") as reader:
            hashes[path.as_posix()] = hashlib.file_digest(reader, "sha256").hexdigest()
    summary.write_text(json.dumps({"roots": rows, **summarize_deployed(rows), "files_sha256": hashes},
                                  indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
