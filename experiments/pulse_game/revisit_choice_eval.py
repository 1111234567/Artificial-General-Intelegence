"""Held-out, matched first-feedback and edit forks for the odor revisit task."""

from argparse import ArgumentParser
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import edit_receipt_graph as receipt_graph
from . import revisit_choice_graph as native
from . import self_param_writer as fw


HOLDOUT_DOMAIN = 0x4D71A
DONOR_DOMAIN = 0x4D71B
BOOTSTRAP_DOMAIN = 0x4D71C
OBSERVED_TASTES = (-1, 0, 1)
ODORS = ("same", "novel")
BRANCHES = (
    "natural", "disabled", "receipt_matched_disabled", "reset_h",
    "receipt_matched_disabled_reset_h", "proposal_0", "proposal_1", "skip",
)
FEEDBACK = ("true", "flipped", "zero", "yoked")
FORCED = tuple(BRANCHES.index(name) for name in ("proposal_0", "proposal_1", "skip"))


@dataclass(frozen=True)
class Heldout:
    first_left: torch.Tensor
    novel_left: torch.Tensor
    first_side: torch.Tensor
    swap: torch.Tensor
    donor_taste: torch.Tensor
    first_draw: torch.Tensor
    proposal_draws: torch.Tensor
    gate_draw: torch.Tensor
    second_draw: torch.Tensor
    random_gate: torch.Tensor

    @property
    def lanes(self):
        return len(self.first_side)

    def second_left(self, odor):
        source = self.first_left if odor == "same" else self.novel_left
        return source * (1 - 2 * self.swap[:, None].float())


def make_holdout(seed, birth, pairs):
    """Use unrelated RNG streams for recipient geometry/draws and donor signs."""
    if pairs < 1:
        raise ValueError("pairs must be positive")
    rng = np.random.default_rng([seed, birth, HOLDOUT_DOMAIN])
    donor_rng = np.random.default_rng([seed, birth, DONOR_DOMAIN])

    def directions(n):
        draws = rng.normal(size=(n, 3))
        return draws / np.linalg.norm(draws, axis=-1, keepdims=True)

    first, novel = directions(pairs), directions(pairs)
    close = np.abs((first * novel).sum(axis=-1)) > .75
    while close.any():
        novel[close] = directions(close.sum())
        close = np.abs((first * novel).sum(axis=-1)) > .75

    def cross(values):
        return torch.from_numpy(np.repeat(values, 4, axis=0))

    first_side = torch.tensor([-1, -1, 1, 1], dtype=torch.float32).repeat(pairs)
    swap = torch.tensor([False, True, False, True]).repeat(pairs)
    return Heldout(
        cross((1.35 * first).astype(np.float32)), cross((1.35 * novel).astype(np.float32)),
        first_side, swap, cross(donor_rng.choice((-1, 1), size=pairs).astype(np.int8)),
        cross(rng.random(pairs, dtype=np.float32)),
        cross(rng.random((pairs, 2, 2), dtype=np.float32)),
        cross(rng.random(pairs, dtype=np.float32)), cross(rng.random(pairs, dtype=np.float32)),
        cross(rng.integers(0, 2, size=pairs, dtype=np.int8)),
    )


def clear_hidden(result):
    state = result.state
    return replace(result, state=native.State(state.graph, torch.zeros_like(state.hidden)))


@torch.no_grad()
def evaluate(graph, holdout, delays, *, keep_connected=True):
    """Execute three sensory streams with common draws. Scoring labels never enter here."""
    lanes = holdout.lanes
    shape = (len(OBSERVED_TASTES), len(ODORS), len(delays), len(BRANCHES), lanes)
    arrays = {
        "motor_side": np.empty(shape, dtype=np.int8),
        "motor_logit": np.empty(shape, dtype=np.float32),
        "receipt_status": np.empty((len(OBSERVED_TASTES), len(BRANCHES), lanes), dtype=np.int8),
        "gate_index": np.empty((len(OBSERVED_TASTES), lanes), dtype=np.int8),
        "gate_logits": np.empty((len(OBSERVED_TASTES), lanes, 3), dtype=np.float32),
        "benefit": np.empty((len(OBSERVED_TASTES), lanes, 2), dtype=np.float32),
        "predicted_effect": np.empty((len(OBSERVED_TASTES), lanes, 2, 3), dtype=np.float32),
        "address": np.empty((len(OBSERVED_TASTES), lanes, 2), dtype=np.int16),
        "operation": np.empty((len(OBSERVED_TASTES), lanes, 2), dtype=np.int8),
    }
    arrays.update({name: tensor.cpu().numpy() for name, tensor in vars(holdout).items()})
    start = native.start(graph, lanes)
    left = holdout.first_left
    first_action = native.act(start, left, -left, holdout.first_draw)
    arrays["first_logit"] = first_action.motor_logit.cpu().numpy()
    arrays["first_natural_side"] = first_action.side.cpu().numpy().astype(np.int8)
    for f, observed_taste in enumerate(OBSERVED_TASTES):
        # observe accepts the right-side outcome. This converts the observed contact taste
        # into that public-world convention and holds the first contact action fixed.
        outcome = holdout.first_side * observed_taste
        experience = native.observe(start, left, -left, outcome, holdout.first_draw,
                                    side_override=holdout.first_side)
        proposals = native.propose(experience, holdout.proposal_draws)
        decision = native.choose(proposals, holdout.gate_draw)

        def commit(selection, **kwargs):
            return native.commit(experience, proposals, selection, keep_connected=keep_connected, **kwargs)

        natural = commit(decision)
        disabled = commit(decision, intervention="disabled")
        receipt_matched = commit(decision, intervention="disabled", receipt_override=natural.sensed_receipt)
        forced = tuple(commit(native.choose(proposals, holdout.gate_draw, override=index))
                       for index in range(3))
        results = (natural, disabled, receipt_matched, clear_hidden(natural),
                   clear_hidden(receipt_matched), *forced)
        arrays["gate_index"][f] = decision.index.cpu().numpy()
        arrays["gate_logits"][f] = proposals.logits.cpu().numpy()
        arrays["benefit"][f] = proposals.benefit.cpu().numpy()
        arrays["predicted_effect"][f] = proposals.effect_prediction.cpu().numpy()
        arrays["address"][f] = proposals.command.address.cpu().numpy()
        arrays["operation"][f] = proposals.command.op.cpu().numpy()
        arrays["first_actual_side"] = experience.side.cpu().numpy().astype(np.int8)

        for b, result in enumerate(results):
            arrays["receipt_status"][f, b] = result.receipt.status.cpu().numpy()
            for d, delay in enumerate(delays):
                state = native.wait(result.state, delay)
                for o, odor in enumerate(ODORS):
                    revisit_left = holdout.second_left(odor)
                    action = native.act(state, revisit_left, -revisit_left, holdout.second_draw)
                    arrays["motor_side"][f, o, d, b] = action.side.cpu().numpy()
                    arrays["motor_logit"][f, o, d, b] = action.motor_logit.cpu().numpy()
    return arrays


def observed_indices(condition, sign, donor_taste):
    """Map scored conditions to previously executed sensory streams."""
    if condition == "yoked":
        return donor_taste.astype(np.int64) + 1
    taste = {"true": sign, "flipped": -sign, "zero": 0}[condition]
    return np.full(len(donor_taste), taste + 1, dtype=np.int64)


def logistic(logit):
    return .5 * (1 + np.tanh(.5 * logit))


def selection(values, index):
    return values[index, np.arange(len(index))]


def order_accuracy(predicted, observed, *, threshold):
    informative = np.abs(observed) > threshold
    decisions = np.where(predicted == 0, .5, predicted * observed > 0)
    return np.where(informative, decisions, np.nan)


def score_world(raw, odor, delay, feedback, sign):
    """Score a counterfactual label offline. For novelty this is a reference slot."""
    stream = observed_indices(feedback, sign, raw["donor_taste"])
    lanes = np.arange(len(stream))
    index = (stream, odor, delay, slice(None), lanes)
    side, logit = raw["motor_side"][index].T, raw["motor_logit"][index].T
    repeat_side = raw["first_actual_side"] * (1 - 2 * raw["swap"].astype(np.int8))
    target_side = repeat_side * sign
    correct = (side == target_side).astype(np.float64)
    expected = logistic(logit * target_side)
    status = raw["receipt_status"][stream, :, lanes].T
    gate = raw["gate_index"][stream, lanes]

    forced_correct, forced_expected = correct[list(FORCED)], expected[list(FORCED)]
    selected_random = raw["random_gate"]
    selected_rate_matched = np.where(gate == 2, 2, selected_random)
    oracle_index = forced_expected.argmax(axis=0)
    predicted = raw["benefit"][stream, lanes].T
    gate_order = raw["gate_logits"][stream, lanes].T
    measured_gap = forced_expected[0] - forced_expected[1]
    realized_gap = forced_correct[0] - forced_correct[1]
    predicted_gap = predicted[0] - predicted[1]
    gate_gap = gate_order[0] - gate_order[1]
    gate_vs_skip = gate_order[:2] - gate_order[2]
    realized_vs_skip = forced_correct[:2] - forced_correct[2]

    by_branch = {}
    for i, branch in enumerate(BRANCHES):
        by_branch[branch] = {
            "actual": correct[i], "p_correct": expected[i],
            "applied_rate": status[i] == fw.Status.APPLIED,
            "action_changed_vs_skip": side[i] != side[FORCED[-1]],
        }
    return {
        "branches": by_branch,
        "identity_repeat": side[0] == repeat_side,
        "motor_repeat": side[0] == raw["first_actual_side"],
        "natural_first": raw["first_actual_side"] == raw["first_natural_side"],
        "ranking": {
            "withhold_rate": gate == 2,
            "unequal_candidate_rate": np.abs(measured_gap) > 1e-6,
            "candidate_p_gap": np.abs(measured_gap),
            "candidate_action_disagreement": realized_gap != 0,
            "candidate_vs_skip_action_disagreement": realized_vs_skip != 0,
            "benefit_order_accuracy": order_accuracy(predicted_gap, measured_gap, threshold=1e-6),
            "gate_order_accuracy": order_accuracy(gate_gap, measured_gap, threshold=1e-6),
            "benefit_order_accuracy_realized": order_accuracy(predicted_gap, realized_gap, threshold=0),
            "gate_order_accuracy_realized": order_accuracy(gate_gap, realized_gap, threshold=0),
            "benefit_vs_skip_accuracy_realized": order_accuracy(predicted, realized_vs_skip, threshold=0),
            "gate_vs_skip_accuracy_realized": order_accuracy(gate_vs_skip, realized_vs_skip, threshold=0),
            "uniform_proposals_actual": selection(forced_correct, selected_random),
            "uniform_proposals_mean_actual": forced_correct[:2].mean(axis=0),
            "uniform_proposals_p_correct": forced_expected[:2].mean(axis=0),
            "uniform_all_mean_actual": forced_correct.mean(axis=0),
            "uniform_all_p_correct": forced_expected.mean(axis=0),
            "rate_matched_random_actual": selection(forced_correct, selected_rate_matched),
            "rate_matched_random_mean_actual": np.where(gate == 2, forced_correct[2], forced_correct[:2].mean(axis=0)),
            "rate_matched_random_p_correct": np.where(gate == 2, forced_expected[2], forced_expected[:2].mean(axis=0)),
            "oracle_actual": selection(forced_correct, oracle_index),
            "oracle_p_correct": forced_expected.max(axis=0),
            "oracle_headroom_over_skip": forced_expected.max(axis=0) - forced_expected[2],
            "oracle_headroom_over_natural": forced_expected.max(axis=0) - expected[0],
        },
    }


def mean(values):
    finite = np.isfinite(values)
    return float(np.mean(values[finite])) if finite.any() else np.nan


class BirthBootstrap:
    """Keep each birth intact, sharing the resampling draws for paired contrasts."""

    def __init__(self, births, seed, draws):
        self.births = list(births)
        self.seed = seed
        if not self.births or draws < 1:
            raise ValueError("at least one birth and one bootstrap draw are required")
        rng = np.random.default_rng([seed, BOOTSTRAP_DOMAIN])
        self.indices = rng.integers(0, len(self.births), size=(draws, len(self.births)))

    def __call__(self, values):
        values = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(values)
        count = int(finite.sum())
        interval = None
        if count > 1:
            # For conditional metrics, sample the births with an observable denominator.
            valid = values[finite]
            indices = self.indices
            if not finite.all():
                rng = np.random.default_rng([self.seed, BOOTSTRAP_DOMAIN, *np.flatnonzero(finite)])
                indices = rng.integers(0, count, size=(len(self.indices), count))
            interval = np.quantile(valid[indices].mean(axis=1), [.025, .975]).tolist()
        return {
            "mean": float(values[finite].mean()) if count else None,
            "ci95": interval, "n_births": count,
            "birth_values": [float(x) if ok else None for x, ok in zip(values, finite, strict=True)],
        }


def summarize_group(raw_by_birth, bootstrap, odor, delay):
    labels = {feedback: [(score_world(raw, odor, delay, feedback, 1),
                          score_world(raw, odor, delay, feedback, -1)) for raw in raw_by_birth]
              for feedback in FEEDBACK}

    def summarize(fn, feedback="true"):
        return bootstrap([mean(np.concatenate([fn(pos), fn(neg)])) for pos, neg in labels[feedback]])

    def contrast(left, right, feedback="true"):
        return summarize(lambda x: left(x).astype(float) - right(x), feedback)

    def branch_stat(result, branch, metric):
        return result["branches"][branch][metric]

    behavior = {}
    causal = {}
    for feedback in FEEDBACK:
        behavior[feedback] = {
            branch: {metric: summarize(lambda x: branch_stat(x, branch, metric), feedback)
                     for metric in ("actual", "p_correct", "applied_rate", "action_changed_vs_skip")}
            for branch in BRANCHES
        }
        behavior[feedback]["natural_repeat"] = {
            "motor": summarize(lambda x: x["motor_repeat"], feedback),
            "identity" if ODORS[odor] == "same" else "reference_slot": summarize(lambda x: x["identity_repeat"], feedback),
        }
        behavior[feedback]["natural_first"] = {
            branch: {metric: summarize(lambda x: branch_stat(x, branch, metric)[x["natural_first"]], feedback)
                     for metric in ("actual", "p_correct")}
            for branch in ("natural", "receipt_matched_disabled", "skip")
        }
        # Positive/negative accuracy is evaluated in odor coordinates. In the swap
        # stratum a correct positive repeat necessarily changes physical motor side.
        by_side = {}
        for first in (-1, 1):
            for swapped in (False, True):
                cells = []
                for raw, (pos, neg) in zip(raw_by_birth, labels[feedback], strict=True):
                    mask = (raw["first_actual_side"] == first) & (raw["swap"] == swapped)
                    cells.append((mean(pos["identity_repeat"][mask]), mean(~neg["identity_repeat"][mask])))
                by_side[f"first_{first:+d}_swap_{int(swapped)}"] = {
                    "positive_repeat": bootstrap([x[0] for x in cells]),
                    "negative_switch": bootstrap([x[1] for x in cells]),
                }
        behavior[feedback]["natural_side_swap"] = by_side
        if feedback != "true":
            causal[f"true_minus_{feedback}"] = {
                branch: {metric: bootstrap([
                    mean(np.concatenate([branch_stat(true, branch, metric) - branch_stat(control, branch, metric)
                                         for true, control in zip(paired, comparison, strict=True)]))
                    for paired, comparison in zip(labels["true"], labels[feedback], strict=True)])
                    for metric in ("actual", "p_correct")}
                for branch in BRANCHES
            }

    lesions = {}
    for left, right in (("natural", "receipt_matched_disabled"), ("natural", "disabled"),
                        ("natural", "reset_h"), ("reset_h", "receipt_matched_disabled_reset_h"),
                        ("receipt_matched_disabled", "receipt_matched_disabled_reset_h"), ("natural", "skip")):
        lesions[f"{left}_minus_{right}"] = {
            metric: contrast(lambda x: branch_stat(x, left, metric), lambda x: branch_stat(x, right, metric))
            for metric in ("actual", "p_correct")
        }
    lesions["natural_first_natural_minus_receipt_matched_disabled"] = {
        metric: summarize(lambda x: (branch_stat(x, "natural", metric)
                                     - branch_stat(x, "receipt_matched_disabled", metric))[x["natural_first"]])
        for metric in ("actual", "p_correct")
    }

    write_specific_feedback = {}
    for feedback in FEEDBACK[1:]:
        for metric in ("actual", "p_correct"):
            for left, right in (("natural", "receipt_matched_disabled"),
                                ("reset_h", "receipt_matched_disabled_reset_h")):
                def effect(x):
                    return branch_stat(x, left, metric) - branch_stat(x, right, metric)

                values = [mean(np.concatenate([effect(true) - effect(control)
                                               for true, control in zip(paired, comparison, strict=True)]))
                          for paired, comparison in zip(labels["true"], labels[feedback], strict=True)]
                name = f"true_minus_{feedback}__{left}_minus_{right}"
                write_specific_feedback.setdefault(name, {})[metric] = bootstrap(values)

    ranking = {name: summarize(lambda x: x["ranking"][name]) for name in labels["true"][0][0]["ranking"]}
    for comparator in ("uniform_proposals", "rate_matched_random", "uniform_all", "oracle"):
        actual_key = "oracle_actual" if comparator == "oracle" else f"{comparator}_mean_actual"
        ranking[f"natural_minus_{comparator}"] = {
            "actual": contrast(lambda x: x["branches"]["natural"]["actual"], lambda x: x["ranking"][actual_key]),
            "p_correct": contrast(lambda x: x["branches"]["natural"]["p_correct"],
                                  lambda x: x["ranking"][f"{comparator}_p_correct"]),
        }
        if comparator in ("uniform_proposals", "rate_matched_random"):
            ranking[f"natural_minus_{comparator}"]["sampled_actual"] = contrast(
                lambda x: x["branches"]["natural"]["actual"], lambda x: x["ranking"][f"{comparator}_actual"])
    return {"behavior": behavior, "feedback_contrasts": causal, "write_activity_contrasts": lesions,
            "feedback_by_write_contrasts": write_specific_feedback, "proposal_ranking": ranking}


def paired_arms(groups, arms, steps, delays, bootstrap):
    """Paired training-arm differences for the primary same-odor endpoints."""
    comparison = {}
    if "ranked" not in arms:
        return comparison
    ranked = groups["ranked"]
    for arm in arms:
        if arm == "ranked":
            continue
        arm_comparison = comparison[f"ranked_minus_{arm}"] = {}
        for step in steps:
            step_comparison = arm_comparison[str(step)] = {}
            for delay in delays:
                a, b = (data[str(step)]["same"][str(delay)] for data in (ranked, groups[arm]))
                endpoints = {
                    "true_natural_actual": (a["behavior"]["true"]["natural"]["actual"],
                                            b["behavior"]["true"]["natural"]["actual"]),
                    "true_natural_p_correct": (a["behavior"]["true"]["natural"]["p_correct"],
                                               b["behavior"]["true"]["natural"]["p_correct"]),
                    "true_natural_first_actual": tuple(
                        x["behavior"]["true"]["natural_first"]["natural"]["actual"] for x in (a, b)),
                    "natural_minus_receipt_matched_disabled": tuple(
                        x["write_activity_contrasts"]["natural_minus_receipt_matched_disabled"]["actual"] for x in (a, b)),
                    "true_minus_flipped_natural": tuple(x["feedback_contrasts"]["true_minus_flipped"]["natural"]["actual"]
                                                         for x in (a, b)),
                    "natural_minus_rate_matched_random": tuple(
                        x["proposal_ranking"]["natural_minus_rate_matched_random"]["actual"] for x in (a, b)),
                }
                step_comparison[str(delay)] = {
                    endpoint: bootstrap(np.asarray(left["birth_values"], dtype=float) - right["birth_values"])
                    for endpoint, (left, right) in endpoints.items()
                }
    return comparison


def paired_odors(arm_groups, steps, delays, bootstrap):
    comparison = {}
    for step in steps:
        dose = comparison[str(step)] = {}
        for delay in delays:
            same, novel = (arm_groups[str(step)][odor][str(delay)] for odor in ODORS)
            dose[str(delay)] = {}
            for feedback in FEEDBACK[1:]:
                name = f"true_minus_{feedback}"
                dose[str(delay)][name] = {
                    metric: bootstrap(np.asarray(same["feedback_contrasts"][name]["natural"][metric]["birth_values"])
                                      - novel["feedback_contrasts"][name]["natural"][metric]["birth_values"])
                    for metric in ("actual", "p_correct")
                }
    return comparison


def run(input_path, output, trials_output, *, arms=None, steps=("all",), holdout_seed=20261121,
        pairs=128, delays=(0, 1, 4, 16), bootstrap_draws=2000):
    from . import revisit_choice_train as training

    manifest = input_path / "run.json" if input_path.is_dir() else input_path
    training_run = json.loads(manifest.read_text(encoding="utf-8"))
    births = training_run["births"]
    arms = list(arms) if arms else training_run["arms"]
    available = [int(step) for step in training_run["checkpoints"]]
    if tuple(steps) == ("all",):
        steps = available
    elif tuple(steps) == ("last",):
        steps = [max(available)]
    else:
        steps = [int(step) for step in steps]
    if any(step not in available for step in steps):
        raise ValueError(f"Requested {steps}, available checkpoints are {available}")
    if any(arm not in training_run["arms"] for arm in arms):
        raise ValueError(f"Requested {arms}, available arms are {training_run['arms']}")
    if len(set(arms)) != len(arms) or len(set(steps)) != len(steps):
        raise ValueError("arms and steps must each be unique")
    if not delays or min(delays) < 0 or len(set(delays)) != len(delays):
        raise ValueError("delays must contain unique nonnegative ticks")
    if trials_output.resolve() == output.resolve():
        raise ValueError("summary and trial trace destinations must be distinct")

    bootstrap = BirthBootstrap(births, holdout_seed, bootstrap_draws)
    holdouts = [make_holdout(holdout_seed, birth, pairs) for birth in births]
    keep_connected = training_run["config"].get("keep_connected", True)
    report = {
        "schema_version": 1,
        "training_manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
        "evaluator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "holdout_seed": holdout_seed, "holdout_domain": HOLDOUT_DOMAIN, "donor_domain": DONOR_DOMAIN,
        "pairs_per_birth": pairs, "lanes_per_birth": 4 * pairs,
        "births": births, "arms": arms, "steps": steps, "delays": list(delays),
        "probability_intervals": {"method": "paired training-birth percentile bootstrap", "draws": bootstrap_draws},
        "trials_file": (trials_output.name if trials_output.parent.resolve() == output.parent.resolve()
                        else str(trials_output.resolve())),
        "design": {
            "first_action": "forced both sides for each pair, held fixed across feedback variants",
            "natural_first": "exact replay slice matching the native first motor draw, from the same forced-side pair",
            "second_action": "native stochastic action before second feedback, with a shared draw across branches",
            "truth": "score both private contact outcomes after the rollout, which receives sign through the feedback stream",
            "yoked": "independent donor sign held fixed across both private truth signs",
            "novel": "score the new antipodal odor pair with |cosine| <= .75 using a balanced reference slot",
            "reset_h": "zero recurrent activation after first receipt processing, before the neutral delay",
            "receipt_matched_disabled": "disable physical writes and replay natural receipt features for the same selection",
            "random": "sampled comparators replay the realized forced action for an independent held-out index, and mean comparators average forced actions",
            "oracle": "diagnostic upper bound that selects among both proposals and skip using expected correctness from the forced branches",
        },
        "groups": {},
    }
    traces = {"birth_names": np.asarray(births), "arm_names": np.asarray(arms), "steps": np.asarray(steps),
              "delays": np.asarray(delays), "observed_tastes": np.asarray(OBSERVED_TASTES),
              "odor_names": np.asarray(ODORS), "branch_names": np.asarray(BRANCHES)}
    for name in vars(holdouts[0]):
        traces[name] = np.stack([getattr(tape, name).cpu().numpy() for tape in holdouts])
    layout = {"motor_side": ["observed_taste", "odor", "delay", "branch", "lane"],
              "motor_logit": ["observed_taste", "odor", "delay", "branch", "lane"],
              "receipt_status": ["observed_taste", "branch", "lane"],
              "gate_index": ["observed_taste", "lane"], "gate_logits": ["observed_taste", "lane", "choice"],
              "benefit": ["observed_taste", "lane", "proposal"],
              "predicted_effect": ["observed_taste", "lane", "proposal", "effect_component"],
              "address": ["observed_taste", "lane", "proposal"], "operation": ["observed_taste", "lane", "proposal"],
              "first_logit": ["lane"], "first_actual_side": ["lane"], "first_natural_side": ["lane"]}
    report["trial_axes"] = {"tape_prefix": ["birth"], "trace_prefix": ["arm", "step", "birth"], "trace": layout}
    report["receipt_codes"] = {member.name: int(member) for member in fw.Status}
    report["receipt_codes"]["WITHHELD"] = receipt_graph.WITHHELD
    for a, arm in enumerate(arms):
        arm_groups = report["groups"][arm] = {}
        for s, step in enumerate(steps):
            raw = [evaluate(training.load_checkpoint(manifest, arm, birth, step), tape, delays,
                            keep_connected=keep_connected) for birth, tape in zip(births, holdouts, strict=True)]
            for name in layout:
                values = np.stack([item[name] for item in raw])
                if name not in traces:
                    traces[name] = np.empty((len(arms), len(steps), *values.shape), dtype=values.dtype)
                traces[name][a, s] = values
            arm_groups[str(step)] = {
                odor: {str(delay): summarize_group(raw, bootstrap, o, d) for d, delay in enumerate(delays)}
                for o, odor in enumerate(ODORS)
            }
            print(f"Evaluated {arm} checkpoint {step} for {len(births)} births", flush=True)
    report["paired_arms"] = paired_arms(report["groups"], arms, steps, delays, bootstrap)
    report["paired_odors"] = {arm: paired_odors(report["groups"][arm], steps, delays, bootstrap) for arm in arms}
    trials_output.parent.mkdir(parents=True, exist_ok=True)
    with trials_output.open("wb") as handle:
        np.savez_compressed(handle, **traces)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Training run.json or its directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials-output", type=Path, help="Compact NPZ trial trace")
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--steps", nargs="+", default=["all"], help="all, last, or checkpoint numbers")
    parser.add_argument("--holdout-seed", type=int, default=20261121)
    parser.add_argument("--pairs", type=int, default=128)
    parser.add_argument("--delays", nargs="+", type=int, default=[0, 1, 4, 16])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    trials = args.trials_output or args.output.with_name(f"{args.output.stem}-trials.npz")
    run(args.input, args.output, trials, arms=args.arms, steps=args.steps, holdout_seed=args.holdout_seed,
        pairs=args.pairs, delays=args.delays, bootstrap_draws=args.bootstrap)


if __name__ == "__main__":
    main()
