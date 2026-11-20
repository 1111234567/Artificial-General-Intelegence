"""Replay held-out multi-object retention on frozen pulse graphs."""

from argparse import ArgumentParser
from dataclasses import dataclass
import json
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch

from . import local_trace_eval as prior
from . import local_trace_graph as native
from . import local_trace_train as training
from . import revisit_choice_graph as legacy


TAPE_DOMAIN = 0x1E3B1
TASKS = ("odor", "feeder_novel", "feeder_fixed_anchor")
BRANCHES = ("write", "exposure_no_write", "neutral")
CAPACITIES = (0, 1, 2, 4, 8, 16)
CONTACT_TICKS = (native.REACTION_TICKS + native.CONTACT_TICKS + native.TASTE_TICKS
                 + native.PRE_WRITE_TICKS + native.POST_WRITE_TICKS)
GRAPH_FIELDS = ("weights", "bias", "present")


@dataclass(frozen=True)
class Tape:
    task: str
    worlds: int
    directions: torch.Tensor
    law: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    right_value: torch.Tensor
    constant_value: torch.Tensor
    motor_draw: torch.Tensor
    edit_draw: torch.Tensor
    legacy_draw: torch.Tensor

    @property
    def lanes(self):
        return len(self.law)

    @property
    def contacts(self):
        return self.motor_draw.shape[-1] - 1

    @property
    def max_k(self):
        return self.law.shape[1] - 1


def make_tape(task, seed, worlds, max_k):
    """Share geometry/draws across a crossed design of A law and B-vector sign."""
    if task not in TASKS or worlds < 1 or max_k < 0:
        raise ValueError("Choose a known task, positive world count, and nonnegative K")
    odor = task == "odor"
    contacts = 1 if odor else 2
    streams = np.random.SeedSequence([TAPE_DOMAIN, seed, int(not odor)]).spawn(5)
    geometry, laws, sides, motors, edits = (np.random.default_rng(s) for s in streams)
    objects = max_k + 1
    # Draw object-major to make the tape prefix independent of the requested maximum K.
    directions = np.stack([prior.directions(geometry, worlds, away_from_fixed=not odor)
                           for _ in range(objects)], axis=1)
    if task == "feeder_fixed_anchor":
        directions[:, 0] = (1.35, 0, 0)
    base_law = prior.signs(laws, (objects, worlds)).T
    base_law[:, 0] = 1
    law = np.repeat(base_law, 4, axis=0)
    law[:, 0] *= np.tile([-1, 1, -1, 1], worlds)
    law[:, 1:] *= np.tile([1, 1, -1, -1], worlds)[:, None]

    orientation = np.repeat(prior.signs(sides, (objects, worlds, 2)).transpose(1, 0, 2), 4, axis=0)
    first, probe = orientation[:, :, 0], orientation[:, :, 1]
    mark = np.repeat(directions, 4, axis=0)[:, :, None, :]
    if odor:
        orientation = np.stack((first, probe), axis=2)
        left = mark * orientation[..., None]
        right = -left
        right_value = law[:, :, None] * orientation
        constant_value = np.zeros_like(right_value)
    else:
        zero = np.zeros_like(first)
        right = mark * np.stack((first, -first, probe), axis=2)[..., None]
        left = right * np.array([1, 1, -1], dtype=np.float32)[None, None, :, None]
        right_value = np.stack((zero, zero, probe * law), axis=2)
        constant_value = np.stack((first * law, -first * law, zero), axis=2)

    draws = motors.random((objects, worlds, contacts + 1, 2), dtype=np.float32).transpose(1, 0, 2, 3)
    edit_draw = edits.random((objects, worlds, contacts, native.EDIT_DRAWS), dtype=np.float32)
    tensors = (directions, law, left, right, right_value, constant_value,
               np.repeat(draws[..., 0], 4, axis=0), np.repeat(edit_draw.transpose(1, 0, 2, 3), 4, axis=0),
               np.repeat(draws[:, :, :contacts, 1], 4, axis=0))
    return Tape(task, worlds, *(torch.from_numpy(value) for value in tensors))


def _act(rule, state, left, right, draw):
    return legacy.act(state, left, right, draw) if rule is None else native.act(rule, state, left, right, draw)


def _wait(rule, state, ticks):
    return legacy.wait(state, ticks) if rule is None else native.wait(rule, state, ticks)


def _same_graph(left, right):
    return all(torch.equal(getattr(left, field), getattr(right, field)) for field in GRAPH_FIELDS)


def _study(rule, state, tape, obj, *, write):
    contacts = []
    for c in range(tape.contacts):
        left, right = tape.left[:, obj, c], tape.right[:, obj, c]
        action = _act(rule, state, left, right, tape.motor_draw[:, obj, c])
        taste = tape.constant_value[:, obj, c] + action.side * tape.right_value[:, obj, c]
        draws = tape.edit_draw[:, obj, c]
        if rule is None:
            experience = legacy.taste(action, left, right, taste)
            proposals = legacy.propose(experience, draws.reshape(-1, 2, 2))
            choice = legacy.choose(proposals, tape.legacy_draw[:, obj, c])
            result = legacy.commit(experience, proposals, choice, intervention="active" if write else "disabled")
        else:
            experience = native.observe(rule, action, left, right, taste)
            result = native.commit(rule, experience, draws, intervention="active" if write else "sham")
        contacts.append(torch.stack((action.side, taste, action.motor_logit)))
        state = result.state
    return state, torch.stack(contacts, dim=1)


def _probe(rule, snapshot, tape, obj):
    # Actions are counterfactual siblings of the same snapshot. Their states are discarded.
    return _act(rule, snapshot, tape.left[:, obj, -1], tape.right[:, obj, -1], tape.motor_draw[:, obj, -1])


def _snapshot(state, lanes):
    arrays = {field: getattr(state.graph, field)[:lanes] for field in GRAPH_FIELDS}
    arrays["hidden"] = state.hidden[:lanes]
    if isinstance(state, native.State):
        arrays.update(sender_trace=state.sender_trace[:lanes], receiver_trace=state.receiver_trace[:lanes])
    return arrays


@torch.no_grad()
def evaluate(graph, rule, tape, capacities=CAPACITIES, *, probe_wait=None, snapshot_worlds=1):
    """Evaluate one historical or newly fitted graph. Raw axes are listed in run()."""
    capacities = tuple(sorted(capacities))
    if (not capacities or min(capacities) < 0 or max(capacities) > tape.max_k
            or len(capacities) != len(set(capacities))):
        raise ValueError("Specify distinct capacities available on this tape")
    if not 0 <= snapshot_worlds <= tape.worlds:
        raise ValueError("Snapshot count must fit inside the tape")
    probe_wait = (4 if tape.task == "odor" else 0) if probe_wait is None else probe_wait
    prior.validate_inputs(graph, (probe_wait,))
    if rule is None:
        initial = legacy.start(graph, tape.lanes)
    else:
        initial = native.start(graph, tape.lanes)
        rule = prior.birth_rule(rule)

    anchor, first = _study(rule, initial, tape, 0, write=True)
    states = dict.fromkeys(BRANCHES, anchor)
    common = dict.fromkeys(BRANCHES, first)
    contacts = {branch: [common[branch]] for branch in BRANCHES}
    k_max = max(capacities)
    rows = (len(BRANCHES), len(capacities), k_max + 1, tape.lanes)
    side = np.zeros(rows, dtype=np.int8)
    logit = np.full(rows, np.nan, dtype=np.float32)
    parameter_distances = []
    snapshots = []
    cut = 0
    for k in range(k_max + 1):
        if k > 0:
            for branch in BRANCHES[:-1]:
                states[branch], contact = _study(rule, states[branch], tape, k, write=branch == "write")
                contacts[branch].append(contact)
            states["neutral"] = _wait(rule, states["neutral"], CONTACT_TICKS * tape.contacts)
            contacts["neutral"].append(torch.full_like(first, torch.nan))
        if k != capacities[cut]:
            continue
        cut_distances, cut_snapshots = [], []
        for branch_index, (branch, state) in enumerate(states.items()):
            if branch != "write" and not _same_graph(state.graph, anchor.graph):
                raise AssertionError(f"{branch} changed parameters after A study")
            snapshot = _wait(rule, state, probe_wait)
            for obj in range(k + 1):
                act = _probe(rule, snapshot, tape, obj)
                side[branch_index, cut, obj] = prior.as_array(act.side)
                logit[branch_index, cut, obj] = prior.as_array(act.motor_logit)
            current, original = state.graph, anchor.graph
            cut_distances.append(torch.stack((prior.distance(current.weights, original.weights),
                                              prior.distance(current.bias, original.bias),
                                              (current.present != original.present).flatten(1).sum(1)), 1))
            if snapshot_worlds:
                cut_snapshots.append(_snapshot(snapshot, snapshot_worlds * 4))
        parameter_distances.append(torch.stack(cut_distances))
        snapshots.append(cut_snapshots)
        cut += 1

    raw = {key: prior.as_array(value) for key, value in vars(tape).items() if isinstance(value, torch.Tensor)}
    raw.update(capacities=np.asarray(capacities), probe_wait=np.asarray(probe_wait), side=side, logit=logit,
               contact=prior.as_array(torch.stack([torch.stack(rows) for rows in contacts.values()])),
               parameter_distance_from_post_A=prior.as_array(torch.stack(parameter_distances, dim=1)))
    if snapshot_worlds:
        for key in snapshots[0][0]:
            raw[f"snapshot_{key}"] = prior.as_array(torch.stack([
                torch.stack([state[key] for state in cut_states]) for cut_states in snapshots], dim=1))
    _audit(raw)
    return raw


def _audit(raw):
    target = raw["right_value"][:, :, -1]
    if not (np.array_equal(target[::2, 0], -target[1::2, 0])
            and np.array_equal(target[::2, 1:], target[1::2, 1:])
            and np.array_equal(target[0::4, 1:], -target[2::4, 1:])
            and np.array_equal(target[1::4, 1:], -target[3::4, 1:])):
        raise AssertionError("Expected crossed complementary laws without a visible law cue")
    first = raw["contact"][0, 0, 0, 0]
    if not (np.array_equal(first[::2], first[1::2]) and np.all(raw["contact"][:, 0, 0, 0] == first)):
        raise AssertionError("Matched laws or branches changed the first choice before feedback")
    neutral = raw["side"][BRANCHES.index("neutral")]
    if not (np.array_equal(neutral[..., 0::4], neutral[..., 2::4])
            and np.array_equal(neutral[..., 1::4], neutral[..., 3::4])):
        raise AssertionError("Neutral choices changed with unavailable B outcome signs")
    if raw["capacities"][0] == 0 and not np.all(raw["side"][:, 0] == raw["side"][0, 0]):
        raise AssertionError("K=0 branches must agree before any branch intervention")


def _metrics(raw, branch, cut, k):
    targets = raw["right_value"][:, :k + 1, -1].T
    side = raw["side"][branch, cut, :k + 1]
    correct = side == targets
    anchor = correct[0]
    expected = .5 * (1 + np.tanh(.5 * raw["logit"][branch, cut, 0] * targets[0]))
    law = raw["law"][:, :k + 1].T
    metrics = {"A": anchor, "A_expected": expected,
               "A_law_negative": anchor[law[0] < 0], "A_law_positive": anchor[law[0] > 0],
               "A_both_laws": anchor[::2] & anchor[1::2], "A_and_all_studied": correct.all(axis=0)}
    if k:
        earlier, later = correct[1:], law[1:]
        metrics.update(B_mean=earlier.flatten(), B_latest=earlier[-1], all_B=earlier.all(axis=0),
                       B_law_negative=earlier[later < 0], B_law_positive=earlier[later > 0],
                       A_and_B=(anchor[None, :] & earlier).flatten(), A_and_latest_B=anchor & earlier[-1])
    rates = {key: float(value.mean()) for key, value in metrics.items()}
    counts = {key: value.size for key, value in metrics.items()}
    rates["A_weaker_law"] = min(rates["A_law_negative"], rates["A_law_positive"])
    counts["A_weaker_law"] = counts["A_law_negative"]
    if k:
        rates["B_weaker_law"] = min(rates["B_law_negative"], rates["B_law_positive"])
        rates["B_weakest_object"] = float(earlier.mean(axis=1).min())
        rates["weakest_studied_object_law"] = min(
            min(obj[sign < 0].mean(), obj[sign > 0].mean()) for obj, sign in zip(correct, law))
        counts["B_weaker_law"] = counts["B_law_negative"]
        counts["B_weakest_object"] = anchor.size
        counts["weakest_studied_object_law"] = counts["A_law_negative"]
    return rates, counts


def summarize(rows, intervals):
    capacities = rows[0]["capacities"]
    summary = {}
    for cut, k in enumerate(capacities):
        branch_metrics, branch_counts = {}, {}
        for b, branch in enumerate(BRANCHES):
            rates, counts = zip(*(_metrics(raw, b, cut, k) for raw in rows))
            branch_metrics[branch] = {key: np.array([value[key] for value in rates]) for key in rates[0]}
            branch_counts[branch] = {key: [value[key] for value in counts] for key in rates[0]}
        block = summary[str(k)] = {
            "branches": {branch: {key: intervals.report(value, branch_counts[branch][key])
                                   for key, value in metrics.items()} for branch, metrics in branch_metrics.items()},
            "write_minus_control": {
                branch: {key: intervals.report(values - branch_metrics[branch][key])
                         for key, values in branch_metrics["write"].items()} for branch in BRANCHES[1:]},
        }
        if capacities[0] == 0:
            baseline = summary["0"]["branches"]["write"]["A"]["birth_values"]
            ref = np.array(list(baseline.values()))
            block["A_retention_difference_from_K0"] = {
                branch: intervals.report(metrics["A"] - ref) for branch, metrics in branch_metrics.items()}
    return summary


def run(sources, out, *, tasks=TASKS, source_task=None, arm="active", step=None, seed=871831,
        worlds=192, capacities=CAPACITIES, bootstrap_draws=10_000, snapshot_worlds=1, probe_wait=None):
    """Replay compatible run directories. New studies can supply their checkpoint task key."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifests = {label: prior.load_manifest(Path(path)) for label, path in sources.items()}
    if not manifests:
        raise ValueError("Supply at least one checkpoint run")
    first_manifest = next(iter(manifests.values()))
    births = first_manifest["births"]
    if len(set(births)) != len(births) or any(set(m["births"]) != set(births) for m in manifests.values()):
        raise ValueError("Paired sources need matching independent birth IDs")
    step = max(first_manifest["checkpoints"]) if step is None else step
    if "local" in manifests and "legacy" in manifests:
        prior.verify_same_budget(manifests["local"], manifests["legacy"], [step], [arm])
    intervals = prior.BirthIntervals(births, bootstrap_draws, seed)
    report = {
        "seed": seed, "births": births, "worlds_per_birth": worlds, "lanes_per_world": 4,
        "capacities": list(capacities), "tasks": list(tasks), "branches": list(BRANCHES),
        "bootstrap_draws": bootstrap_draws, "interval_unit": "equal-weight birth, paired differences before resampling",
        "stage": step, "arm": arm, "ticks_per_contact": CONTACT_TICKS,
        "probe_wait_by_task": {task: (4 if task == "odor" else 0) if probe_wait is None else probe_wait for task in tasks},
        "source_sha256": {Path(p).name: prior.digest(Path(p))
                          for p in (__file__, prior.__file__, native.__file__, legacy.__file__, training.__file__)},
        "fit_manifest_sha256": {label: prior.digest(Path(path) / "run.json") for label, path in sources.items()},
        "checkpoint_sha256": {}, "raw_sha256": {}, "results": {},
        "protocol": {
            "law_design": "Within a world A law is crossed with the sign inversion of one independently sampled B law vector.",
            "trials": "Each birth has the same held-out worlds. Within-world law lanes are matched, and K values share a prefix.",
            "positions": "Each object's study and query orientations are sampled independently. Feeder demonstrations expose both antipodes.",
            "geometry": "Each object receives an independent physical antipodal pair. Novel feeder axes avoid the training axis."
                        " Fixed-anchor feeder holds only A at the training axis.",
            "fork": "A always studied with writes. For B, write and exposure branches receive their own real tastes with identical"
                    " decision clocks. The neutral branch advances the same number of ticks with no B sensory stimulus.",
            "probe": "All queries for a given K start from the same post-B state after a common wait. No query delivers taste or"
                     " commits a write. Query actions use shared draws across treatments and independent draws across objects.",
            "A_retention_difference_from_K0": "A accuracy at K minus A accuracy after its study and probe wait alone.",
            "B_weakest_object": "Minimum of B accuracies by serial position within each birth.",
            "weakest_studied_object_law": "Minimum accuracy over studied serial positions and private laws within each birth.",
        },
        "raw_axes": {
            "left/right/right_value/constant_value/motor_draw": ["lane", "object", "contact or terminal"],
            "side/logit": ["branch", "capacity", "object", "lane"],
            "contact": ["branch", "object", "field (side, taste, logit)", "contact", "lane"],
            "parameter_distance_from_post_A": ["branch", "capacity", "lane", "field (weights L1, bias L1, changed topology bits)"],
            "snapshot_*": ["branch", "capacity", "lane in first sampled worlds", "state field axes"],
        },
        "snapshot_worlds_per_birth": snapshot_worlds,
    }
    for label, source in sources.items():
        source = Path(source)
        manifest = manifests[label]
        report["results"][label] = {}
        for task in tasks:
            checkpoint_task = source_task or ("odor" if task == "odor" else "feeder")
            tape = make_tape(task, seed, worlds, max(capacities))
            rows = []
            for birth in births:
                graph, rule, filename, fingerprint = prior.verified_checkpoint(
                    training, source, manifest, checkpoint_task, arm, birth, step)
                archive_name = (Path("checkpoints") / label / filename).as_posix()
                target = out / archive_name
                target.parent.mkdir(parents=True, exist_ok=True)
                copyfile(source / filename, target)
                if prior.digest(target) != fingerprint:
                    raise AssertionError("Archived checkpoint does not match verified input")
                report["checkpoint_sha256"][archive_name] = fingerprint
                raw = evaluate(graph, rule, tape, capacities, probe_wait=probe_wait, snapshot_worlds=snapshot_worlds)
                name = f"raw/{label}-{task}-{birth}.npz"
                (out / name).parent.mkdir(exist_ok=True)
                np.savez_compressed(out / name, **raw)
                report["raw_sha256"][name] = prior.digest(out / name)
                rows.append(raw)
            report["results"][label][task] = summarize(rows, intervals)
            print(f"replayed {label} {task} across {len(births)} births and {worlds} worlds each", flush=True)
    if "local" in sources and "legacy" in sources:
        report["local_minus_legacy"] = {
            task: prior.compare_rates(report["results"]["local"][task], report["results"]["legacy"][task],
                                      intervals, tuple((branch, branch) for branch in BRANCHES)) for task in tasks}
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--legacy-run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--tasks", choices=TASKS, nargs="+", default=TASKS)
    parser.add_argument("--source-task")
    parser.add_argument("--arm", default="active")
    parser.add_argument("--step", type=int)
    parser.add_argument("--seed", type=int, default=871831)
    parser.add_argument("--worlds", type=int, default=192)
    parser.add_argument("--capacities", type=int, nargs="+", default=CAPACITIES)
    parser.add_argument("--snapshot-worlds", type=int, default=1)
    parser.add_argument("--probe-wait", type=int)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    sources = {"local": args.run}
    if args.legacy_run is not None:
        sources["legacy"] = args.legacy_run
    run(sources, args.out, tasks=args.tasks, source_task=args.source_task, arm=args.arm, step=args.step,
        seed=args.seed, worlds=args.worlds, capacities=args.capacities, snapshot_worlds=args.snapshot_worlds,
        probe_wait=args.probe_wait)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        copyfile(args.out / "summary.json", args.summary)


if __name__ == "__main__":
    main()
