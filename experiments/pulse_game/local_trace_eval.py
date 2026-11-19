"""Paired holdout tests for sensory feedback, local writes, and live state exchange."""

from argparse import ArgumentParser
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import local_trace_graph as native
from . import revisit_choice_graph as legacy


EVAL_DOMAIN = 0x1E3A1
DONOR_DOMAIN = 0x1E3A2
BOOTSTRAP_DOMAIN = 0x1E3A3
FEEDBACK_BRANCHES = ("intact", "zero", "flipped", "donor", "sham")
PRECOMMIT = ("precommit_hidden", "precommit_trace", "precommit_graph")
READOUT = ("readout_hidden", "readout_trace", "readout_graph")
SHUFFLES = ("shuffle", "shuffle_values")
BRANCHES = (*FEEDBACK_BRANCHES, *SHUFFLES, *PRECOMMIT, *READOUT)
LEGACY_BRANCHES = ("intact", "zero", "flipped", "no_write")
TASKS = ("odor", "feeder", "feeder_novel")
RECEIPT_FIELDS = ("proposed_birth", "proposed_retirement", "birth", "retired",
                  "birth_weight", "retired_weight")


@dataclass(frozen=True)
class Tape:
    # Each world contains multiple matched lane pairs. Adjacent lanes carry opposite laws.
    task: str
    worlds: int
    left: torch.Tensor
    right: torch.Tensor
    right_value: torch.Tensor
    constant_value: torch.Tensor
    draw: torch.Tensor
    edit_draw: torch.Tensor
    legacy_choice_draw: torch.Tensor
    shuffle_draw: torch.Tensor
    law: torch.Tensor
    swap: torch.Tensor
    first_mark: torch.Tensor
    donor_left: torch.Tensor
    donor_right: torch.Tensor
    donor_right_value: torch.Tensor
    donor_constant_value: torch.Tensor
    donor_draw: torch.Tensor
    donor_edit_draw: torch.Tensor

    @property
    def lanes(self):
        return len(self.left)

    @property
    def contacts(self):
        return self.left.shape[1] - 1

    @property
    def twin(self):
        return torch.arange(self.lanes) ^ 1

    def pack(self):
        tensors = {name: as_array(value) for name, value in vars(self).items()
                   if isinstance(value, torch.Tensor) and name != "shuffle_draw"}
        width = self.lanes // self.worlds
        tensors["shuffle_draw_per_world"] = as_array(self.shuffle_draw[::width])
        tensors["twin"] = as_array(self.twin)
        return tensors


def as_array(tensor):
    return tensor.detach().cpu().numpy()


def directions(rng, worlds, *, away_from_fixed=False):
    def sample(n):
        x = rng.normal(size=(n, 3))
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    x = sample(worlds)
    if away_from_fixed:
        close = np.abs(x[:, 0]) > .5
        while close.any():
            x[close] = sample(int(close.sum()))
            close = np.abs(x[:, 0]) > .5
    return (1.35 * x).astype(np.float32)


def repeat_worlds(values, lanes):
    return np.repeat(values, lanes, axis=0)


def signs(rng, shape):
    return (2 * rng.integers(0, 2, size=shape) - 1).astype(np.float32)


def make_tape(task, seed, worlds):
    if task not in TASKS or worlds < 1:
        raise ValueError("Choose a known task and a positive number of worlds")
    # Fixed and novel feeder variants have matching laws, layouts, and random draws.
    domain = int(task != "odor")
    rng = np.random.default_rng([EVAL_DOMAIN, seed, domain, 0])
    geometry = np.random.default_rng([EVAL_DOMAIN, seed, domain, 1])
    donor = np.random.default_rng([DONOR_DOMAIN, seed, domain, 0])
    donor_geometry = np.random.default_rng([DONOR_DOMAIN, seed, domain, 1])

    if task == "odor":
        contacts, width = 1, 4
        law = np.tile(np.array([-1, 1, -1, 1], dtype=np.float32), worlds)
        swap = np.tile([False, False, True, True], worlds)
        first_mark = repeat_worlds(signs(rng, worlds), width)
        orientation = np.stack((first_mark, np.where(swap, -first_mark, first_mark)), axis=1)
        left = repeat_worlds(directions(geometry, worlds), width)[:, None] * orientation[..., None]
        right = -left
        right_value = orientation * law[:, None]
        constant = np.zeros_like(right_value)

        donor_mark = signs(donor, worlds)
        donor_left = (directions(donor_geometry, worlds) * donor_mark[:, None])[:, None]
        donor_right = -donor_left
        donor_right_value = (donor_mark * signs(donor, worlds))[:, None]
        donor_constant = np.zeros_like(donor_right_value)
    else:
        contacts, width = 2, 8
        law = np.tile(np.array([-1, 1], dtype=np.float32), worlds * 4)
        first_mark = np.tile(np.repeat(np.array([-1, 1], dtype=np.float32), 4), worlds)
        terminal = np.tile(np.repeat(np.array([-1, 1], dtype=np.float32), 2), worlds * 2)
        swap = np.zeros_like(law, dtype=bool)
        if task == "feeder_novel":
            mark = directions(geometry, worlds, away_from_fixed=True)
            donor_mark = directions(donor_geometry, worlds, away_from_fixed=True)
        else:
            mark = donor_mark = np.tile(np.array([1.35, 0, 0], dtype=np.float32), (worlds, 1))
        mark = repeat_worlds(mark, width)
        right = mark[:, None] * np.stack((first_mark, -first_mark, terminal), 1)[..., None]
        left = right * np.array([1, 1, -1], dtype=np.float32)[None, :, None]
        zero = np.zeros_like(law)
        constant = np.stack((first_mark * law, -first_mark * law, zero), 1)
        right_value = np.stack((zero, zero, terminal * law), 1)

        donor_order = signs(donor, worlds)
        donor_gains = donor_order * signs(donor, worlds)
        donor_left = donor_mark[:, None] * np.stack((donor_order, -donor_order), 1)[..., None]
        donor_right = donor_left.copy()
        donor_constant = np.stack((donor_gains, -donor_gains), 1)
        donor_right_value = np.zeros_like(donor_constant)

    draw = repeat_worlds(rng.random((worlds, contacts + 1), dtype=np.float32), width)
    edit = repeat_worlds(rng.random((worlds, contacts, 4), dtype=np.float32), width)
    shuffle = repeat_worlds(rng.random((worlds, contacts, native.SHUFFLE_DRAWS), dtype=np.float32), width)
    legacy_choice = repeat_worlds(rng.random((worlds, contacts), dtype=np.float32), width)
    donor_draw = donor.random((worlds, contacts), dtype=np.float32)
    donor_edit_draw = donor.random((worlds, contacts, 4), dtype=np.float32)
    return Tape(task, worlds, *(torch.from_numpy(values.copy()) for values in (
        left, right, right_value, constant, draw, edit, legacy_choice, shuffle, law, swap, first_mark,
        donor_left, donor_right, donor_right_value, donor_constant, donor_draw, donor_edit_draw,
    )))


def capture_receipt(receipt, *, record_delta=False):
    row = {key: as_array(getattr(receipt, key)) for key in RECEIPT_FIELDS}
    weight = as_array(receipt.weight_delta).reshape(len(receipt.birth), -1)
    bias = as_array(receipt.bias_delta)
    row["weight_write_count"] = np.count_nonzero(weight, axis=1).astype(np.int16)
    row["bias_write_count"] = np.count_nonzero(bias, axis=1).astype(np.int16)
    row["write_count"] = row["weight_write_count"] + row["bias_write_count"]
    row["write_l1"] = np.abs(weight).sum(1) + np.abs(bias).sum(1)
    if record_delta:
        row["delta"] = np.concatenate((weight, bias), axis=1)
    return row


def distance(a, b):
    return (a - b).abs().flatten(1).sum(1)


def state_distance(a, b):
    return as_array(torch.stack((distance(a.hidden, b.hidden),
                                 distance(a.sender_trace, b.sender_trace),
                                 distance(a.receiver_trace, b.receiver_trace),
                                 distance(a.graph.weights, b.graph.weights),
                                 distance(a.graph.bias, b.graph.bias),
                                 (a.graph.present != b.graph.present).flatten(1).sum(1)), 1))


def outcome(tape, action, t):
    return tape.constant_value[:, t] + action.side * tape.right_value[:, t]


def donor_receipts(graph, rule, tape):
    state = native.start(graph, batch_size=tape.worlds)
    values = []
    for t in range(tape.contacts):
        left, right = tape.donor_left[:, t], tape.donor_right[:, t]
        action = native.act(rule, state, left, right, tape.donor_draw[:, t])
        values.append(tape.donor_constant_value[:, t] + action.side * tape.donor_right_value[:, t])
        if t < tape.contacts - 1:
            experience = native.observe(rule, action, left, right, values[-1])
            state = native.commit(rule, experience, tape.donor_edit_draw[:, t]).state
    return torch.stack(values, dim=1).repeat_interleave(tape.lanes // tape.worlds, dim=0)


def birth_rule(rule):
    """Normalize a single loaded Rule or a homogeneous expanded view to one row."""
    options = rule.expand(1) if isinstance(rule, native.Rule) else rule
    if not vars(options):
        raise ValueError("Expected a local rule")
    values = {}
    for name, tensor in vars(options).items():
        row = tensor[:1]
        if not len(row) or not torch.equal(tensor, row.expand_as(tensor)):
            raise ValueError("Evaluation takes one independent rule birth at a time")
        values[name] = row
    return replace(options, **values)


def validate_inputs(graph, delays):
    if not delays or any(d < 0 for d in delays):
        raise ValueError("Evaluation needs at least one nonnegative delay")
    if len(graph.bias) != 1:
        raise ValueError("Evaluation takes one independent graph birth at a time")


@torch.no_grad()
def evaluate(graph, rule, tape, delays):
    """Run only natural actions. All intervention branches share their world tape."""
    validate_inputs(graph, delays)
    rule = birth_rule(rule)
    donor = donor_receipts(graph, rule, tape)
    twin = tape.twin
    states, records, experiences = {}, {}, {}
    for branch in FEEDBACK_BRANCHES:
        state = native.start(graph, batch_size=tape.lanes)
        records[branch] = []
        for t in range(tape.contacts):
            left, right = tape.left[:, t], tape.right[:, t]
            action = native.act(rule, state, left, right, tape.draw[:, t])
            earned = outcome(tape, action, t)
            if branch == "zero":
                delivered = torch.zeros_like(earned)
            elif branch == "flipped":
                delivered = -earned
            elif branch == "donor":
                delivered = donor[:, t]
            else:
                delivered = earned
            experience = native.observe(rule, action, left, right, delivered)
            edit = native.commit(rule, experience, tape.edit_draw[:, t],
                                 intervention="sham" if branch == "sham" else "active")
            row = dict(contact_side=as_array(action.side).astype(np.int8),
                       contact_logit=as_array(action.motor_logit),
                       actual_feedback=as_array(earned).astype(np.int8),
                       delivered_feedback=as_array(delivered).astype(np.int8),
                       **capture_receipt(edit.receipt, record_delta=branch == "intact" and t == tape.contacts - 1))
            records[branch].append(row)
            state = edit.state
            if branch == "intact":
                experiences[t] = experience
        states[branch] = state

    last = tape.contacts - 1
    experience = experiences[last]
    before_write = experience.state
    before_write_twin = before_write.subset(twin)
    active_reference = records["intact"][-1]
    final_deltas = [active_reference.pop("delta")]
    for branch in (*SHUFFLES, *PRECOMMIT):
        if branch in SHUFFLES:
            edit = native.commit(rule, experience, tape.edit_draw[:, last],
                                 intervention=branch, shuffle=tape.shuffle_draw[:, last])
        else:
            component = branch.removeprefix("precommit_")
            swapped = native.swap(before_write, before_write_twin, component)
            edit = native.commit(rule, replace(experience, state=swapped), tape.edit_draw[:, last])
        states[branch] = edit.state
        ledger = capture_receipt(edit.receipt, record_delta=branch in SHUFFLES)
        if branch in SHUFFLES:
            final_deltas.append(ledger.pop("delta"))
        records[branch] = [*records["intact"][:-1], {**active_reference, **ledger}]
    for branch in READOUT:
        records[branch] = records["intact"]

    readout = {branch: [] for branch in BRANCHES}
    readout_distance = []
    for delay in delays:
        waited = {branch: native.wait(rule, state, delay) for branch, state in states.items()}
        intact = waited["intact"]
        other = intact.subset(twin)
        readout_distance.append(state_distance(intact, other))
        for branch in READOUT:
            waited[branch] = native.swap(intact, other, branch.removeprefix("readout_"))
        for branch in BRANCHES:
            action = native.act(rule, waited[branch], tape.left[:, -1], tape.right[:, -1], tape.draw[:, -1])
            readout[branch].append((as_array(action.side).astype(np.int8), as_array(action.motor_logit)))

    result = tape.pack()
    result["delays"] = np.asarray(delays, dtype=np.int16)
    result["donor_feedback"] = as_array(donor).astype(np.int8)
    result["precommit_twin_distance"] = state_distance(before_write, before_write_twin)
    result["readout_twin_distance"] = np.stack(readout_distance)
    result["sham_parameter_distance"] = state_distance(states["sham"], native.start(graph, tape.lanes))[:, 3:]
    result["side"] = np.array([[pair[0] for pair in readout[branch]] for branch in BRANCHES])
    result["logit"] = np.array([[pair[1] for pair in readout[branch]] for branch in BRANCHES])
    result.update({name: np.array([[row[name] for row in records[branch]] for branch in BRANCHES])
                   for name in records["intact"][0]})
    result["final_delta"] = np.stack(final_deltas)
    return result


@torch.no_grad()
def evaluate_legacy(graph, tape, delays):
    """Evaluate the addressed writer on the same physical tape and motor draws."""
    validate_inputs(graph, delays)
    expanded = graph.repeat(tape.lanes)
    start = legacy.State(expanded, torch.zeros_like(expanded.bias))
    states, records = {}, {}
    for branch in LEGACY_BRANCHES:
        state = start
        records[branch] = []
        for t in range(tape.contacts):
            left, right = tape.left[:, t], tape.right[:, t]
            action = legacy.act(state, left, right, tape.draw[:, t])
            earned = outcome(tape, action, t)
            delivered = {"zero": torch.zeros_like(earned), "flipped": -earned}.get(branch, earned)
            experienced = legacy.taste(action, left, right, delivered)
            proposed = legacy.propose(experienced, tape.edit_draw[:, t].reshape(-1, 2, 2))
            decision = legacy.choose(proposed, tape.legacy_choice_draw[:, t])
            result = legacy.commit(experienced, proposed, decision,
                                   intervention="disabled" if branch == "no_write" else "active")
            records[branch].append(dict(
                contact_side=as_array(action.side).astype(np.int8),
                contact_logit=as_array(action.motor_logit),
                actual_feedback=as_array(earned).astype(np.int8),
                delivered_feedback=as_array(delivered).astype(np.int8),
                edit_address=as_array(result.command.address).astype(np.int16),
                edit_operation=as_array(result.command.op).astype(np.int8),
                edit_status=as_array(result.raw_receipt.status).astype(np.int8),
            ))
            state = result.state
        states[branch] = state
    acts = [[legacy.act(legacy.wait(states[branch], delay), tape.left[:, -1], tape.right[:, -1], tape.draw[:, -1])
             for delay in delays] for branch in LEGACY_BRANCHES]
    result = tape.pack()
    result["delays"] = np.asarray(delays, dtype=np.int16)
    result["side"] = np.array([[as_array(act.side).astype(np.int8) for act in row] for row in acts])
    result["logit"] = np.array([[as_array(act.motor_logit) for act in row] for row in acts])
    result.update({name: np.array([[row[name] for row in records[branch]] for branch in LEGACY_BRANCHES])
                   for name in records["intact"][0]})
    frozen = states["no_write"].graph
    result["no_write_graph_unchanged"] = np.asarray(all(torch.equal(getattr(frozen, attr), getattr(expanded, attr))
                                                       for attr in ("weights", "bias", "present")))
    return result


def branch_metrics(raw, task, i, delay):
    side, logit = raw["side"][i, delay], raw["logit"][i, delay]
    truth = raw["right_value"][:, -1]
    correct = side == truth
    twin = raw["twin"]
    result = {
        "correct": correct,
        "expected_correct": .5 * (1 + np.tanh(.5 * logit * truth)),
        "both_twins_correct": correct[::2] & correct[twin[::2]],
        "twins_choose_different": side[::2] != side[twin[::2]],
        "choice_changes_vs_intact": side != raw["side"][0, delay],
        "law_negative": correct[raw["law"] < 0],
        "law_positive": correct[raw["law"] > 0],
    }
    if task == "odor":
        first = raw["contact_side"][0, 0]
        tasted = raw["actual_feedback"][0, 0]
        swap = raw["swap"]
        repeat = side == np.where(swap, -first, first)
        result.update(positive_repeat=repeat[tasted > 0], negative_switch=~repeat[tasted < 0],
                      correct_same=correct[~swap], correct_swap=correct[swap],
                      positive_repeat_same=repeat[(tasted > 0) & ~swap],
                      positive_repeat_swap=repeat[(tasted > 0) & swap],
                      negative_switch_same=~repeat[(tasted < 0) & ~swap],
                      negative_switch_swap=~repeat[(tasted < 0) & swap],
                      motor_repeat=side == first)
    return result


def audit_matching(raw, branches):
    """Check that the evaluator varies private laws only through delivered feedback."""
    intact, zero, flipped = (branches.index(b) for b in ("intact", "zero", "flipped"))
    twin = raw["twin"]
    side, logit = raw["side"], raw["logit"]
    first = raw["contact_side"][:, 0]
    truth = raw["right_value"][:, -1]
    if not (np.array_equal(twin[twin], np.arange(len(twin))) and np.all(truth == -truth[twin])):
        raise AssertionError("Complementary twins do not have opposite target sides")
    if not (np.all(first == first[0]) and np.all(first == first[:, twin])):
        raise AssertionError("Matched first actions diverged before any feedback")
    controls = [zero]
    if "donor" in branches:
        donor = branches.index("donor")
        seen = raw["donor_feedback"]
        if not (np.array_equal(seen, seen[twin]) and np.array_equal(raw["delivered_feedback"][donor].T, seen)):
            raise AssertionError("Donor receipt was not identically delivered to its two recipient twins")
        controls.append(donor)
    for branch in controls:
        if not (np.array_equal(side[branch], side[branch][:, twin])
                and np.array_equal(logit[branch], logit[branch][:, twin])):
            raise AssertionError("Control branch leaked its recipient's hidden law")
    if not (np.array_equal(side[flipped], side[intact][:, twin])
            and np.array_equal(logit[flipped], logit[intact][:, twin])):
        raise AssertionError("Flipped-feedback action did not reproduce the opposite twin")
    return {"matched_twin_pairs": len(twin) // 2, "donor_receipts": int(raw["donor_feedback"].size // 2)
            if "donor" in branches else 0}


def audit(raw):
    """Check matching and preserve quantitative local intervention diagnostics per birth."""
    matching = audit_matching(raw, BRANCHES)
    intact, sham, trace_swap = (BRANCHES.index(b) for b in ("intact", "sham", "readout_trace"))
    side, logit = raw["side"], raw["logit"]
    if not (np.array_equal(side[intact], side[trace_swap])
            and np.array_equal(logit[intact], logit[trace_swap])):
        raise AssertionError("A trace swap at readout changed the action without a write")
    if (raw["sham_parameter_distance"].any() or (raw["birth"][sham] != -1).any()
            or (raw["retired"][sham] != -1).any() or raw["write_count"][sham].any()):
        raise AssertionError("Sham changed graph parameters or topology")

    counts = {branch: {
        "writes": int(raw["write_count"][i].sum()),
        "weight_writes": int(raw["weight_write_count"][i].sum()),
        "bias_writes": int(raw["bias_write_count"][i].sum()),
        "births": int((raw["birth"][i] >= 0).sum()),
        "retirements": int((raw["retired"][i] >= 0).sum()),
    } for i, branch in enumerate(BRANCHES)}
    distances = ("hidden", "sender_trace", "receiver_trace", "weights", "bias", "topology")
    return {**matching, "counts_across_contacts": counts,
            "shuffle_last_commit": shuffle_audit(raw, "shuffle"),
            "shuffle_values_last_commit": shuffle_audit(raw, "shuffle_values"),
            "precommit_twin_difference_lanes": dict(zip(distances, np.count_nonzero(raw["precommit_twin_distance"], axis=0).tolist())),
            "readout_twin_difference_lanes": {
                str(delay): dict(zip(distances, np.count_nonzero(raw["readout_twin_distance"][i], axis=0).tolist()))
                for i, delay in enumerate(raw["delays"])
            }}


def shuffle_audit(raw, branch):
    intact, shuffle = BRANCHES.index("intact"), BRANCHES.index(branch)
    native_delta, shuffled_delta = raw["final_delta"][[0, 1 + SHUFFLES.index(branch)]]
    edges = native.NODES ** 2
    native_count = ((native_delta[:, :edges] != 0).sum(1), (native_delta[:, edges:] != 0).sum(1))
    shuffled_count = ((shuffled_delta[:, :edges] != 0).sum(1), (shuffled_delta[:, edges:] != 0).sum(1))
    mismatch = np.concatenate((np.sort(native_delta[:, :edges], axis=1) - np.sort(shuffled_delta[:, :edges], axis=1),
                               np.sort(native_delta[:, edges:], axis=1) - np.sort(shuffled_delta[:, edges:], axis=1)), 1)
    active_address = native_delta != 0
    shuffled_address = shuffled_delta != 0
    # Address masks can be identical even when the same nonzero payloads moved inside that mask.
    changed_values = np.abs(native_delta - shuffled_delta) > 2e-6
    changes = {
        "lanes": len(native_delta),
        "weight_count_mismatch_lanes": int((native_count[0] != shuffled_count[0]).sum()),
        "bias_count_mismatch_lanes": int((native_count[1] != shuffled_count[1]).sum()),
        "additive_payload_mismatch_lanes": int((np.abs(mismatch) > 2e-6).any(1).sum()),
        "max_additive_payload_abs_difference": float(np.max(np.abs(mismatch))),
        "different_additive_addresses_lanes": int((active_address != shuffled_address).any(1).sum()),
        "different_additive_values_lanes": int(changed_values.any(1).sum()),
    }
    for address, payload in (("birth", "birth_weight"), ("retired", "retired_weight")):
        initial = raw[address][intact, -1]
        altered = raw[address][shuffle, -1]
        common = (initial >= 0) & (altered >= 0)
        delta = raw[payload][intact, -1, common] - raw[payload][shuffle, -1, common]
        changes[f"{address}_count_mismatch_lanes"] = int(((initial >= 0) != (altered >= 0)).sum())
        changes[f"{address}_matched_write_lanes"] = int(common.sum())
        changes[f"{address}_relocated_lanes"] = int((common & (initial != altered)).sum())
        changes[f"{address}_payload_mismatch_lanes"] = int((np.abs(delta) > 2e-6).sum())
        changes[f"{address}_max_payload_abs_difference"] = float(np.max(np.abs(delta), initial=0))
    if branch == "shuffle_values" and any(changes[f"{event}_relocated_lanes"] or changes[f"{event}_count_mismatch_lanes"]
                                          for event in ("birth", "retired")):
        raise AssertionError("Value shuffle changed the matched structural edits")
    return changes


def audit_legacy(raw):
    if not raw["no_write_graph_unchanged"].item():
        raise AssertionError("Legacy no-write branch changed graph parameters")
    return audit_matching(raw, LEGACY_BRANCHES)


class BirthIntervals:
    def __init__(self, births, draws, seed):
        if not births or draws < 1:
            raise ValueError("At least one birth and one interval resample are required")
        self.births = list(births)
        rng = np.random.default_rng([BOOTSTRAP_DOMAIN, seed])
        self.indices = rng.integers(0, len(births), size=(draws, len(births)))

    def report(self, values, counts=None):
        x = np.asarray(values, dtype=float)
        if x.shape != (len(self.births),) or not np.isfinite(x).all():
            raise ValueError("Each birth must contribute one finite estimate")
        result = {"mean": float(x.mean()), "birth_values": dict(zip(map(str, self.births), x.tolist())),
                  "ci95": np.quantile(x[self.indices].mean(1), [.025, .975]).tolist()}
        if counts is not None:
            result["trials_by_birth"] = dict(zip(map(str, self.births), map(int, counts)))
        return result


def summarize(rows, task, delays, intervals, branches=BRANCHES):
    results = {}
    for d, delay in enumerate(delays):
        rates, counts = {}, {}
        for i, branch in enumerate(branches):
            metrics = [branch_metrics(row, task, i, d) for row in rows]
            rates[branch] = {key: np.array([cell[key].mean() for cell in metrics]) for key in metrics[0]}
            counts[branch] = {key: [len(cell[key]) for cell in metrics] for key in metrics[0]}
            if task.startswith("feeder"):
                rates[branch]["weaker_law"] = np.minimum(rates[branch]["law_negative"], rates[branch]["law_positive"])
                counts[branch]["weaker_law"] = counts[branch]["law_negative"]
        results[str(delay)] = {
            "branches": {branch: {key: intervals.report(values, counts[branch][key])
                                   for key, values in rates[branch].items()}
                         for branch in branches},
            "intact_minus_branch": {branch: {key: intervals.report(rates["intact"][key] - rates[branch][key])
                                              for key in rates[branch] if key != "choice_changes_vs_intact"}
                                    for branch in branches if branch != "intact"},
        }
    return results


def compare_rates(left, right, intervals, branch_pairs):
    result = {}
    for delay, block in left.items():
        result[delay] = {}
        for left_branch, right_branch in branch_pairs:
            left_estimates = block["branches"][left_branch]
            right_estimates = right[delay]["branches"][right_branch]
            result[delay][left_branch] = {}
            for key in (key for key in left_estimates if key in right_estimates and key != "choice_changes_vs_intact"):
                a, b = left_estimates[key]["birth_values"], right_estimates[key]["birth_values"]
                result[delay][left_branch][key] = intervals.report([a[str(r)] - b[str(r)] for r in intervals.births])
    return result


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def load_manifest(path):
    return json.loads((path / "run.json").read_text(encoding="utf-8"))


def controller_for(manifest):
    return manifest.get("controller", manifest.get("config", {}).get("controller", "local"))


def verify_same_budget(manifest, control, steps, arms):
    if controller_for(manifest) != "local" or controller_for(control) != "legacy":
        raise ValueError("Paired controller comparison needs --run local and --legacy-run legacy")
    if set(manifest["births"]) != set(control["births"]) or set(steps) - set(control["checkpoints"]):
        raise ValueError("Controller runs differ in requested births or checkpoints")
    if set(arms) - set(control["arms"]):
        raise ValueError("Legacy controller run lacks a requested arm")
    a, b = manifest.get("config", {}), control.get("config", {})
    for key in ("pairs", "min_wait", "max_wait", "world_seed", "lr", "grad_clip"):
        if key not in a or key not in b or a[key] != b[key]:
            raise ValueError(f"Controller run budgets differ on {key}")


def verified_checkpoint(training, path, manifest, task, arm, birth, step):
    filename = training.checkpoint_path(path, task, arm, birth, step)
    name = filename.relative_to(path).as_posix()
    before = digest(filename)
    expected = manifest.get("checkpoint_sha256", {}).get(name)
    if expected is not None and before != expected:
        raise ValueError(f"Checkpoint SHA-256 mismatch at {filename}")
    graph, rule = training.load_checkpoint(path, task, arm, birth, step)
    if digest(filename) != before:
        raise ValueError(f"Checkpoint changed while loading at {filename}")
    if (rule is None) != (controller_for(manifest) == "legacy"):
        raise ValueError(f"Checkpoint does not use the declared controller at {filename}")
    return graph, rule, name, before


def evaluate_stage(training, source, manifest, step, task, tape, arms, intervals, delays, out, protocol):
    controller = controller_for(manifest)
    train_task = "feeder" if task == "feeder_novel" else task
    results, checks = {}, {}
    branches = LEGACY_BRANCHES if controller == "legacy" else BRANCHES
    for arm in arms:
        rows = []
        checks[arm] = {}
        for birth in intervals.births:
            graph, rule, path, fingerprint = verified_checkpoint(training, source, manifest, train_task, arm, birth, step)
            protocol["checkpoint_sha256"][controller][path] = fingerprint
            if controller == "legacy":
                raw = evaluate_legacy(graph, tape, delays)
                checks[arm][str(birth)] = audit_legacy(raw)
            else:
                raw = evaluate(graph, rule, tape, delays)
                checks[arm][str(birth)] = audit(raw)
            name = f"{task}-{arm}-{birth}-{step}.npz" if controller == "local" else f"legacy-{task}-{arm}-{birth}-{step}.npz"
            np.savez_compressed(out / name, **raw)
            protocol["traces"][name] = digest(out / name)
            rows.append(raw)
        results[arm] = summarize(rows, task, delays, intervals, branches)
    return results, checks


def run(run_path, out, *, pairs=128, seed=871319, delays=(0, 4, 16), stages=("last",),
        bootstrap_draws=10_000, tasks=TASKS, arms=None, legacy_run=None):
    from . import local_trace_train as training

    source_manifest = run_path / "run.json"
    manifest = load_manifest(run_path)
    births = manifest["births"]
    arms = manifest["arms"] if arms is None else list(arms)
    controller = controller_for(manifest)
    if controller not in {"local", "legacy"} or len(births) != len(set(births)):
        raise ValueError("Expected a known controller and unique birth IDs")
    checkpoint_stages = manifest["checkpoints"]
    steps = sorted({max(checkpoint_stages) if s == "last" else int(s) for s in stages})
    if (set(steps) - set(checkpoint_stages) or set(arms) - set(manifest["arms"])
            or set(tasks) - set(TASKS)):
        raise ValueError("Requested an unlisted checkpoint, arm, or task")
    if not delays or min(delays) < 0 or len(set(delays)) != len(delays):
        raise ValueError("Specify distinct nonnegative delays")
    comparison = load_manifest(legacy_run) if legacy_run is not None else None
    if comparison is not None:
        verify_same_budget(manifest, comparison, steps, arms)
    out.mkdir(parents=True, exist_ok=True)
    intervals = BirthIntervals(births, bootstrap_draws, seed)
    protocol = {
        "births": births, "stages": steps, "arms": arms, "tasks": list(tasks), "seed": seed,
        "worlds_per_birth": pairs, "delays": list(delays), "bootstrap_draws": bootstrap_draws,
        "interval_unit": "equal-weight birth, paired differences before resampling",
        "controller": controller, "branches": list(LEGACY_BRANCHES if controller == "legacy" else BRANCHES),
        "audits": {}, "traces": {}, "results": {}, "active_minus_independent": {},
        "fit_manifest_sha256": digest(source_manifest),
        "checkpoint_sha256": {controller: {}},
        "source_sha256": {Path(p).name: digest(Path(p)) for p in (__file__, native.__file__, legacy.__file__, training.__file__)},
        "raw_axes": {"act": ["branch", "delay", "lane"], "contact": ["branch", "contact", "lane"],
                     "final_delta": ["intact/shuffle/shuffle_values", "lane", "address"],
                     "distance_columns": ["hidden", "sender_trace", "receiver_trace", "weights", "bias", "topology"]},
        "branch_phase": {"sham": "all contacts, proposals sampled and clocks matched, writes suppressed",
                         "shuffle": "final contact only, same intact snapshot/draws before write",
                         "shuffle_values": "final contact only, relocate numeric payloads with original turnover preserved",
                         "precommit": "opposite twin after final contact observation, before local commit",
                         "readout": "opposite twin after common delay, immediately before autonomous action",
                         "legacy_no_write": "all contacts, candidate selection and disabled-execution sensory receipt retained"},
        "feeder_novel": "novel physical antipodal mark pair with both marks first demonstrated",
    }
    if comparison is not None:
        protocol.update(legacy_results={}, legacy_audits={}, local_minus_legacy={},
                        legacy_fit_manifest_sha256=digest(legacy_run / "run.json"), legacy_branches=list(LEGACY_BRANCHES))
        protocol["checkpoint_sha256"]["legacy"] = {}
    for step in steps:
        stage = str(step)
        protocol["results"][stage] = {}
        protocol["audits"][stage] = {}
        protocol["active_minus_independent"][stage] = {}
        if comparison is not None:
            for key in ("legacy_results", "legacy_audits", "local_minus_legacy"):
                protocol[key][stage] = {}
        for task in tasks:
            tape = make_tape(task, seed, pairs)
            task_results, task_checks = evaluate_stage(training, run_path, manifest, step, task, tape, arms,
                                                       intervals, delays, out, protocol)
            protocol["results"][stage][task] = task_results
            protocol["audits"][stage][task] = task_checks
            if {"active", "independent"}.issubset(arms):
                contrast = compare_rates(task_results["active"], task_results["independent"], intervals,
                                         (("intact", "intact"),))
                protocol["active_minus_independent"][stage][task] = {d: p["intact"] for d, p in contrast.items()}
            if comparison is not None:
                other_results, other_checks = evaluate_stage(training, legacy_run, comparison, step, task, tape, arms,
                                                            intervals, delays, out, protocol)
                protocol["legacy_results"][stage][task] = other_results
                protocol["legacy_audits"][stage][task] = other_checks
                common_branches = (("intact", "intact"), ("zero", "zero"), ("flipped", "flipped"), ("sham", "no_write"))
                protocol["local_minus_legacy"][stage][task] = {
                    arm: compare_rates(task_results[arm], other_results[arm], intervals, common_branches) for arm in arms
                }
            print(f"evaluated {task} checkpoint {step} across {len(births)} births", flush=True)
    (out / "summary.json").write_text(json.dumps(protocol, indent=2, allow_nan=False), encoding="utf-8")
    return protocol


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--legacy-run", type=Path, help="optional matched addressed-writer fit for paired comparisons")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=128, help="independent world roots per birth")
    parser.add_argument("--seed", type=int, default=871319)
    parser.add_argument("--delays", type=int, nargs="+", default=[0, 4, 16])
    parser.add_argument("--steps", nargs="+", default=["last"])
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--arms", nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.run, args.out, pairs=args.pairs, seed=args.seed, delays=args.delays, stages=args.steps,
        bootstrap_draws=args.bootstrap_draws, tasks=args.tasks, arms=args.arms, legacy_run=args.legacy_run)


if __name__ == "__main__":
    main()
