"""Paired fresh-world interventions on the autonomous graph editor."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from . import self_write_core as core


ORIENTATION = np.array([-1, 1], dtype=np.int8)
WRITER_ROWS = core.WRITER
PLAN = (
    Path(__file__).resolve().parents[2]
    / "docs" / "evidence" / "self-writing-2026-11-19" / "transfer" / "preregistered.json"
)


@dataclass(frozen=True)
class ContactTape:
    """The receiver gets only the current odors, its side, and the resulting taste."""

    odors: np.ndarray
    orientations: np.ndarray
    cues: np.ndarray
    targets: np.ndarray
    taste_flipped: np.ndarray
    scripted_arms: np.ndarray

    @property
    def lives(self):
        return self.cues.shape[0]

    @property
    def contacts(self):
        return self.cues.shape[1]

    def present(self, event):
        lanes = np.arange(self.lives)
        return self.odors[lanes, self.cues[:, event], 0], self.odors[lanes, self.cues[:, event], 1]

    def target(self, event):
        return self.targets[np.arange(self.lives), event, self.cues[:, event]]

    def taste(self, event, action):
        signed = np.where(action == self.target(event), 1.0, -1.0)
        return np.where(self.taste_flipped[:, event], -signed, signed).astype(np.float32)


def build_tape(seed, lives, odors, rounds, phase_length, noise, *, complementary):
    """Make the plants, private target assignments and noise independently of an actor."""

    rng = np.random.default_rng(seed)
    repeats = 2 if complementary else 1
    if lives % repeats or rounds % phase_length:
        raise ValueError("An incomplete complement pair or phase would change exposure counts")
    pairs = lives // repeats
    directions = rng.uniform(-1.0, 1.0, (pairs, odors, 3)).astype(np.float32)
    first_odors = np.stack((directions, -directions), axis=2)
    public_odors = np.repeat(first_odors, repeats, axis=0)
    orientations = rng.choice(ORIENTATION, (pairs, odors))
    orientations = np.repeat(orientations, repeats, axis=0)
    if complementary:
        orientations[1::2] *= -1

    cue_rounds = rng.random((pairs, rounds, odors)).argsort(axis=-1)
    phases = rounds // phase_length
    phase_events = phase_length * odors
    contacts = rounds * odors
    cues = cue_rounds.reshape(pairs, contacts)
    slips = rng.random((pairs, contacts)) < noise
    scripted = rng.choice(ORIENTATION, (pairs, contacts))
    if complementary:
        for switch in range(phase_events, contacts, phase_events):
            bridge_cue = cues[:, switch - 1]
            first_round = cues[:, switch : switch + odors]
            bridge_offset = np.argmax(first_round == bridge_cue[:, None], axis=1)
            first_round[np.arange(pairs), bridge_offset] = first_round[:, 0]
            first_round[:, 0] = bridge_cue
            confirmed = orientations[::repeats][np.arange(pairs), bridge_cue]
            confirmed = confirmed * (-1) ** (switch // phase_events - 1)
            scripted[:, switch - 1] = confirmed
            scripted[:, switch] = confirmed
            slips[:, switch - 1 : switch + 1] = False

    targets = orientations[:, None, :] * np.repeat((-1) ** np.arange(phases), phase_events)[None, :, None]
    scripted = np.repeat(scripted, repeats, axis=0)
    if complementary:
        scripted[1::2] *= -1
    return ContactTape(
        public_odors,
        orientations,
        np.repeat(cues, repeats, axis=0),
        targets.astype(np.int8),
        np.repeat(slips, repeats, axis=0),
        scripted,
    )


def repeat_graph(graph, copies):
    return core.Graph(*(np.repeat(getattr(graph, name), copies, axis=0) for name in ("w", "b", "mask")))


def writer_delta(before, after):
    delta_w = after.w[:, WRITER_ROWS] - before.w[:, WRITER_ROWS]
    delta_b = after.b[:, WRITER_ROWS] - before.b[:, WRITER_ROWS]
    edge_changes = (after.mask[:, WRITER_ROWS] != before.mask[:, WRITER_ROWS]).sum(axis=(1, 2))
    return np.sqrt((delta_w ** 2).sum(axis=(1, 2)) + (delta_b ** 2).sum(axis=1)), edge_changes


def scramble_writer(base, intact, seed):
    """Keep each writer's support and permute increments across compatible inputs."""

    rng = np.random.default_rng(seed)
    result = intact.clone()
    for lane in range(base.b.shape[0]):
        for row in range(WRITER_ROWS.start, WRITER_ROWS.stop):
            before = base.mask[lane, row]
            after = intact.mask[lane, row]
            for active_before in (False, True):
                columns = np.flatnonzero(after & (before == active_before))
                result.w[lane, row, columns] = bounded_shuffle(
                    base.w[lane, row, columns], intact.w[lane, row, columns], rng,
                )
        result.b[lane, WRITER_ROWS] = bounded_shuffle(base.b[lane, WRITER_ROWS], intact.b[lane, WRITER_ROWS], rng)
    np.testing.assert_allclose(writer_delta(base, result)[0], writer_delta(base, intact)[0], rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(result.mask, intact.mask)
    return result


def bounded_shuffle(base, intact, rng):
    delta = intact - base
    if len(delta) > 1 and np.any(delta != delta[0]):
        for _ in range(48):
            candidate = base + rng.permutation(delta)
            if np.all(np.abs(candidate) <= core.LIMIT) and np.any(candidate != intact):
                return candidate
    return intact


def validate_transplant(base, intact, donor):
    for field in ("w", "b", "mask"):
        original, moved, learned = (getattr(graph, field) for graph in (base, intact, donor))
        np.testing.assert_array_equal(moved[:, :WRITER_ROWS.start], original[:, :WRITER_ROWS.start])
        np.testing.assert_allclose(moved[:, WRITER_ROWS], learned[:, WRITER_ROWS], rtol=1e-6, atol=1e-7)


def predict(graph, hidden, left, right):
    return core.advance(graph, hidden, core.sense(left, right), ticks=core.MOTOR_TICKS)


def right_score(hidden):
    return 1.0 / (1.0 + np.exp(-core.motor(hidden)))


def write_step(graph, hidden, left, right, action, taste, key, *, view, arm="active"):
    before, after = core.contact_states(graph, hidden, left, right, action, taste)
    edits = core.proposal(graph, before, after, key, view=view)
    next_graph, receipt = core.apply(graph, edits, arm=arm)
    return next_graph, after, receipt


def train_donors(base, tape, key_root, view):
    graph = base.clone()
    hidden = core.rest(base)
    tastes = np.empty_like(tape.cues, dtype=np.float32)
    actions = np.empty_like(tape.scripted_arms)
    for event in range(tape.contacts):
        left, right = tape.present(event)
        hidden = predict(graph, hidden, left, right)
        action = np.where(core.motor(hidden) > 0, 1, -1)
        taste = tape.taste(event, action)
        actions[:, event], tastes[:, event] = action, taste
        graph, hidden, _ = write_step(graph, hidden, left, right, action, taste, key_root + event, view=view)
    delta, support = writer_delta(base, graph)
    return graph, {
        "writer_delta_l2": delta.tolist(),
        "writer_support_changes": support.tolist(),
        "positive_receipts": (tastes > 0).sum(axis=1).tolist(),
        "right_choices": (actions > 0).sum(axis=1).tolist(),
    }


def measure_queries(graph, hidden, tape, target_event):
    probabilities = []
    correct = []
    for cue in range(tape.odors.shape[1]):
        next_hidden = predict(graph, hidden, tape.odors[:, cue, 0], tape.odors[:, cue, 1])
        target_right = tape.targets[:, target_event, cue] > 0
        chance_right = right_score(next_hidden)
        probabilities.append(np.where(target_right, chance_right, 1.0 - chance_right))
        hard_right = core.motor(next_hidden) > 0
        correct.append(hard_right == target_right)
    return np.mean(probabilities, axis=0), np.mean(correct, axis=0)


def assess(initial, tape, key_root, phase_contacts, *, policy, view, arm):
    graph = initial.clone()
    hidden = core.rest(graph)
    phases = tape.contacts // phase_contacts
    curve_shape = (tape.lives, phases, phase_contacts + 1)
    correct_prob = np.empty(curve_shape, dtype=np.float32)
    hard_correct = np.empty(curve_shape, dtype=np.float32)
    actions = np.empty_like(tape.scripted_arms)
    tastes = np.empty_like(tape.cues, dtype=np.int8)
    pre_act_right = np.empty_like(tape.cues, dtype=np.float32)
    for event in range(tape.contacts):
        phase, steps_before = divmod(event, phase_contacts)
        if steps_before == 0:
            correct_prob[:, phase, 0], hard_correct[:, phase, 0] = measure_queries(graph, hidden, tape, event)
        left, right = tape.present(event)
        hidden = predict(graph, hidden, left, right)
        pre_act_right[:, event] = right_score(hidden)
        action = tape.scripted_arms[:, event]
        if policy == "free_choice":
            action = np.where(core.motor(hidden) > 0, 1, -1)
        taste = tape.taste(event, action)
        actions[:, event], tastes[:, event] = action, taste
        graph, hidden, _ = write_step(graph, hidden, left, right, action, taste, key_root + event, view=view, arm=arm)
        correct_prob[:, phase, steps_before + 1], hard_correct[:, phase, steps_before + 1] = measure_queries(
            graph, hidden, tape, event,
        )
    return {
        "query_score": correct_prob,
        "query_hard": hard_correct,
        "chosen_actions": actions,
        "received_tastes": tastes,
        "contact_right_score": pre_act_right,
        "actual_choice_correct": actions == tape.targets[
            np.arange(tape.lives)[:, None], np.arange(tape.contacts)[None, :], tape.cues,
        ],
    }


def flatten_body_edits(before, after):
    weights = after.w[:, :WRITER_ROWS.start] - before.w[:, :WRITER_ROWS.start]
    bias = after.b[:, :WRITER_ROWS.start] - before.b[:, :WRITER_ROWS.start]
    return np.concatenate((weights.reshape(len(weights), -1), bias), axis=1)


def edit_witness(base, intact, scrambled, tape, view, key):
    """Replay the first forced contact before any outcome-dependent choice can differ."""

    left, right = tape.present(0)
    side = tape.scripted_arms[:, 0]
    signed_taste = tape.taste(0, side)
    zero_taste = np.zeros_like(signed_taste)
    rest = core.rest(base)
    preact = predict(base, rest, left, right)
    pre_common, post_common = core.contact_states(base, preact, left, right, side, signed_taste)

    def deliver(graph, taste):
        pre, post = core.contact_states(graph, preact, left, right, side, taste)
        edit = core.proposal(graph, pre, post, key, view=view)
        changed, _ = core.apply(graph, edit)
        return flatten_body_edits(graph, changed), post[:, WRITER_ROWS]

    edited_base, writer_base = deliver(base, signed_taste)
    edited_intact, writer_intact = deliver(intact, signed_taste)
    edited_sham, _ = deliver(scrambled, signed_taste)
    blank_base, _ = deliver(base, zero_taste)
    blank_intact, _ = deliver(intact, zero_taste)

    base_candidate = core.proposal(base, pre_common, post_common, key, view=view)
    intact_candidate = core.proposal(intact, pre_common, post_common, key, view=view)
    common_edge = (intact_candidate.edge_delta - base_candidate.edge_delta)[:, :WRITER_ROWS.start]
    common_bias = (intact_candidate.bias_delta - base_candidate.bias_delta)[:, :WRITER_ROWS.start]

    return {
        "accepted_body_edit_delta_l2": np.linalg.norm(edited_intact - edited_base, axis=1),
        "accepted_body_edit_delta_vs_scramble_l2": np.linalg.norm(edited_intact - edited_sham, axis=1),
        "blank_body_edit_delta_l2": np.linalg.norm(blank_intact - blank_base, axis=1),
        "taste_specific_body_edit_difference_l2": np.linalg.norm(
            (edited_intact - blank_intact) - (edited_base - blank_base), axis=1,
        ),
        "writer_activity_delta_l2": np.linalg.norm(writer_intact - writer_base, axis=1),
        "same_activity_body_proposal_delta_l2": np.sqrt(
            (common_edge ** 2).sum(axis=(1, 2)) + (common_bias ** 2).sum(axis=1),
        ),
        "first_right_score_abs_delta": np.abs(right_score(predict(intact, rest, left, right)) - right_score(preact)),
        "first_receipt": signed_taste,
    }


def bootstrap_interval(per_donor, draws):
    simulated = per_donor[draws].mean(axis=1)
    return {
        "mean": float(per_donor.mean()),
        "ci95": np.quantile(simulated, [0.025, 0.975]).tolist(),
        "donor_means": per_donor.tolist(),
    }


def summarize(arms, comparisons, donors, draws, phase_contacts):
    groups = {}
    for arm, values in arms.items():
        score = values["query_score"]
        hard = values["query_hard"]
        groups[arm] = {
            "query_score_by_phase_and_contact": score.mean(axis=0).tolist(),
            "hard_accuracy_by_phase_and_contact": hard.mean(axis=0).tolist(),
            "gain_contacts_1_to_12": (score[:, :, 1:13].mean(axis=2) - score[:, :, 0]).mean(axis=0).tolist(),
            "actual_contact_correct_by_phase": values["actual_choice_correct"].reshape(
                len(score), -1, phase_contacts,
            ).mean(axis=(0, 2)).tolist(),
            "positive_tastes_by_phase": (values["received_tastes"] > 0).reshape(
                len(score), -1, phase_contacts,
            ).sum(axis=2).mean(axis=0).tolist(),
        }
    paired = {}
    for comparison in comparisons:
        main, control = comparison.split(" - ")
        delta = arms[main]["query_score"] - arms[control]["query_score"]
        endpoint = delta[:, :, 1:13].mean(axis=2) - delta[:, :, 0]
        per_donor_phase = endpoint.reshape(donors, -1, endpoint.shape[1]).mean(axis=1)
        paired[comparison] = [bootstrap_interval(per_donor_phase[:, phase], draws) for phase in range(endpoint.shape[1])]
    return {"conditions": groups, "paired_gain": paired}


def run_assay(checkpoint, plan_path, output, *, view):
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    with np.load(checkpoint, allow_pickle=False) as saved:
        boot = core.Graph(saved["w"], saved["b"], saved["mask"])
    if len(boot.w) != 1:
        raise ValueError("All lives must start from one shared bootstrap graph")

    arm_policy = {
        "naive": "active",
        "writer_only": "active",
        "scrambled_writer": "active",
        "writer_protected": "primitive",
        "intact_frozen_writer": "primitive",
        "writer_only_no_writes": "none",
    }
    donor_count = plan["donors_per_regime"]
    world_pairs = plan["fresh_world_pairs_per_donor"]
    lives_per_donor = 2 * world_pairs
    odors = plan["odors_per_recipient"]
    phase_rounds = plan["contacts_per_odor_per_phase"]
    phase_contacts = odors * phase_rounds
    recipient_rounds = plan["phases"] * phase_rounds
    donor_rounds = plan["donor_rounds"]
    base = core.repeated_graph(boot, donor_count)
    donors_by_regime = {}
    donor_odors = set()
    raw = {}
    results = {}
    summary = {
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
        "plan_sha256": sha256(plan_bytes).hexdigest(),
        "core_sha256": sha256(Path(core.__file__).read_bytes()).hexdigest(),
        "view": view,
        "results": results,
    }

    for regime_id, (regime, parameters) in enumerate(plan["donor_regimes"].items()):
        donor_seed = plan["donor_root"] + regime_id
        tape = build_tape(
            donor_seed, donor_count, odors, donor_rounds,
            parameters["switch_every_rounds"] or donor_rounds, parameters["noise"], complementary=False,
        )
        donor_odors.update(map(tuple, tape.odors.reshape(-1, tape.odors.shape[-1])))
        donor, report = train_donors(base, tape, donor_seed + 5000, view)
        intact = core.transplant_writer(base, donor, base)
        validate_transplant(base, intact, donor)
        scrambled = scramble_writer(base, intact, donor_seed + 9000)
        report["checkpoint_to_intact_delta_verified"] = True
        report["scramble_delta_from_intact_l2"] = writer_delta(intact, scrambled)[0].tolist()
        report["intact_connected"] = [core.connected(mask) for mask in intact.mask]
        donors_by_regime[regime] = (intact, scrambled)
        results[regime] = {"donor_seed": donor_seed, "donors": report, "recipients": {}}

    for regime_id, (regime, (intact, scrambled)) in enumerate(donors_by_regime.items()):
        initials = {
            "naive": base,
            "writer_only": intact,
            "scrambled_writer": scrambled,
            "writer_protected": base,
            "intact_frozen_writer": intact,
            "writer_only_no_writes": intact,
        }
        initial_graphs = {name: repeat_graph(graph, lives_per_donor) for name, graph in initials.items()}
        resampling = np.random.default_rng(plan["uncertainty_seed"] + regime_id).integers(
            0, donor_count, (plan["cluster_bootstrap_draws"], donor_count),
        )
        recipient_seed = plan["recipient_root"] + regime_id
        key_root = recipient_seed + 6000
        for noise in plan["recipient_noise"]:
            noise_report = {}
            results[regime]["recipients"][str(noise)] = noise_report
            tape = build_tape(
                recipient_seed, donor_count * lives_per_donor, odors, recipient_rounds,
                phase_rounds, noise, complementary=True,
            )
            fresh_odor_set = set(map(tuple, tape.odors.reshape(-1, tape.odors.shape[-1])))
            if not donor_odors.isdisjoint(fresh_odor_set):
                raise RuntimeError("A novel-world sensory vector was already used on a donor")
            cohort = f"{regime}|{noise}"
            for name in ("odors", "cues", "orientations", "taste_flipped", "scripted_arms"):
                raw[f"{cohort}|tape|{name}"] = getattr(tape, name)
            witness = edit_witness(
                initial_graphs["naive"], initial_graphs["writer_only"], initial_graphs["scrambled_writer"],
                tape, view, key_root,
            )
            noise_report["next_edit_witness"] = {
                name: {"mean": float(array.mean()), "nonzero": int(np.count_nonzero(array > 1e-7))}
                for name, array in witness.items() if name != "first_receipt"
            }
            for name, array in witness.items():
                raw[f"{cohort}|witness|{name}"] = array

            for policy in plan["recipient_policy_modes"]:
                arms = {}
                for condition in plan["recipient_conditions"]:
                    values = assess(
                        initial_graphs[condition], tape, key_root, phase_contacts,
                        policy=policy, view=view, arm=arm_policy[condition],
                    )
                    arms[condition] = values
                    for name, array in values.items():
                        raw[f"{cohort}|{policy}|{condition}|{name}"] = array
                for values in arms.values():
                    np.testing.assert_allclose(values["query_score"][:, 0, 0].mean(), 0.5, atol=1e-7, rtol=0)
                    if policy == "matched_contact":
                        np.testing.assert_array_equal(values["chosen_actions"], arms["naive"]["chosen_actions"])
                        np.testing.assert_array_equal(values["received_tastes"], arms["naive"]["received_tastes"])
                        np.testing.assert_array_equal(values["received_tastes"][:, phase_contacts - 1], 1)
                        np.testing.assert_array_equal(values["received_tastes"][:, phase_contacts], -1)
                noise_report[policy] = summarize(arms, plan["primary_contrasts"], donor_count, resampling, phase_contacts)
                print(f"Finished {cohort}|{policy}", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "transfer_traces.npz"
    np.savez_compressed(raw_path, **raw)
    summary["traces_sha256"] = sha256(raw_path.read_bytes()).hexdigest()
    (output / "transfer_results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--output", type=Path, default=PLAN.parent)
    parser.add_argument("--view", choices=("own", "blind"), default="own")
    args = parser.parse_args()
    run_assay(args.checkpoint, args.plan, args.output, view=args.view)


if __name__ == "__main__":
    main()
