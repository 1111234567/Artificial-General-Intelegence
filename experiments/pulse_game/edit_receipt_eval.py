"""Evaluate self-edit checkpoints on paired held-out lifetimes and edit forks."""

from argparse import ArgumentParser
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic, sleep

import numpy as np
import torch

from . import edit_receipt_graph as native
from . import edit_receipt_train as training
from . import self_param_writer as fw


DATA = Path(__file__).resolve().parents[2] / "data" / "edit-receipt-20261119"
SCENARIOS = ("stable_noise", "reversal")
HISTORIES = ("ABA", "CBA", "B_only")
WRITE_MODES = {"full": "active", "no_write": "disabled"}
SPLIT_DOMAINS = {"deployment": 0xEA19, "return": 0xEB19, "fork_holdout": 0xEC19,
                 "fork_validation": 0xED19, "yoked_donor": 0xEF19}


@dataclass(frozen=True)
class Settings:
    seed: int = 909719
    worlds: int = 32
    holdout_worlds: int = 32
    contacts: int = 60
    channels: int = 3
    phase_contacts: int = 24
    reversal_at: int | None = None
    noise: float = .20
    holdout_noise: float = .15
    forks: int = 6
    early: int = 4
    late: int = 8
    bootstrap_draws: int = 2000


@dataclass(frozen=True)
class Tape:
    left: torch.Tensor
    truth: torch.Tensor
    outcome: torch.Tensor
    draws: torch.Tensor
    cue: torch.Tensor

    @property
    def length(self):
        return self.truth.shape[1]

    def contact(self, t, *, silent=False):
        left = self.left[:, t]
        outcome = self.outcome[:, t]
        return left, -left, torch.zeros_like(outcome) if silent else outcome, self.draws[:, t]

    def window(self, lo, hi):
        return Tape(*(x[:, lo:hi] for x in (self.left, self.truth, self.outcome, self.draws, self.cue)))

    def revalue(self, multiplier):
        return Tape(self.left, self.truth * multiplier, self.outcome * multiplier, self.draws, self.cue)

    def interleave(self, donor):
        return Tape(*(torch.stack((a, b), dim=1).flatten(0, 1)
                      for a, b in zip(vars(self).values(), vars(donor).values(), strict=True)))


def make_bank(rng, worlds, channels):
    odor = rng.normal(size=(worlds, channels, 3))
    odor /= np.linalg.norm(odor, axis=-1, keepdims=True)
    for channel in range(1, channels):
        close = np.any(np.abs((odor[:, :channel] * odor[:, channel:channel + 1]).sum(-1)) > .75, axis=1)
        while close.any():
            candidates = rng.normal(size=(int(close.sum()), 3))
            odor[close, channel] = candidates / np.linalg.norm(candidates, axis=-1, keepdims=True)
            close = np.any(np.abs((odor[:, :channel] * odor[:, channel:channel + 1]).sum(-1)) > .75, axis=1)
    signs = rng.choice((-1, 1), size=(worlds, channels))
    return (1.35 * odor).astype(np.float32), signs


def make_tape(rng, odor, signs, contacts, noise):
    worlds, channels = signs.shape
    cycles = (contacts + channels - 1) // channels
    cue = rng.random((worlds, cycles, channels)).argsort(axis=-1).reshape(worlds, -1)[:, :contacts]
    swap = rng.choice((-1, 1), size=(worlds, contacts))
    slip = np.where(rng.random((worlds, contacts)) < noise, -1, 1)
    truth = np.take_along_axis(signs, cue, axis=1) * swap
    left = odor[np.arange(worlds)[:, None], cue] * swap[..., None]
    draws = rng.random((worlds, contacts, 4), dtype=np.float32)
    return Tape(torch.from_numpy(left.astype(np.float32)), torch.from_numpy(truth.astype(np.float32)),
                torch.from_numpy((truth * slip).astype(np.float32)), torch.from_numpy(draws),
                torch.from_numpy(cue.astype(np.int16)))


def environment_tapes(settings, birth, split):
    rng = np.random.default_rng([settings.seed, int(birth), SPLIT_DOMAINS[split]])
    if split == "return":
        first_bank = make_bank(rng, settings.worlds, settings.channels)
        other_bank = make_bank(rng, settings.worlds, settings.channels)
        phase, noise = settings.phase_contacts, settings.noise
        a_first = make_tape(rng, *first_bank, phase, noise)
        return {
            "A_first": a_first, "C_first": a_first.revalue(-1),
            "B": make_tape(rng, *other_bank, phase, noise),
            "A_return": make_tape(rng, *first_bank, phase, noise),
        }
    holdout = split == "fork_holdout"
    worlds = settings.holdout_worlds if holdout else settings.worlds
    noise = settings.holdout_noise if holdout else settings.noise
    stable = make_tape(rng, *make_bank(rng, worlds, settings.channels), settings.contacts, noise)
    signs = torch.ones(settings.contacts)
    signs[reversal_at(settings):] = -1
    return {"stable_noise": stable, "reversal": stable.revalue(signs)}


def reversal_at(settings):
    return settings.contacts // 2 if settings.reversal_at is None else settings.reversal_at


def policy_settings(arm, config):
    return {"sensing": training.arm_sensing(arm), "gate": "learned",
            "keep_connected": config.get("keep_connected", True)}


def propose(state, tape, t, policy, *, silent=False):
    return native.propose(state, *tape.contact(t, silent=silent), stochastic=True, sensing=policy["sensing"])


def commit(state, proposal, draws, policy, *, intervention="active", accept_override=None):
    return native.commit(
        state, proposal, draws, gate=policy["gate"], stochastic_gate=True,
        accept_override=accept_override, sensing=policy["sensing"], intervention=intervention,
        keep_connected=policy["keep_connected"],
    )


def observation(proposal, result, truth):
    return {
        "side": proposal.side.to(torch.int8),
        "taste": proposal.taste.to(torch.int8),
        "correct": proposal.side == truth,
        "p_correct": (proposal.motor_logit * truth).sigmoid(),
        "motor_logit": proposal.motor_logit,
        "address": proposal.command.address.to(torch.int16),
        "op": proposal.command.op.to(torch.int8),
        "accepted": result.accepted,
        "applied": result.receipt.status == fw.Status.APPLIED,
        "status": result.receipt.status.to(torch.int8),
        "predicted_benefit": proposal.benefit,
        "predicted_effect": proposal.effect_prediction,
        "receipt_effect": result.receipt.effect_target,
    }


def fork_contact(state, proposal, ordinary, tape, t, policy, horizon):
    """Force both decisions for one proposal and resume ordinary live policies."""
    draws = tape.draws[:, t]
    take = commit(state, proposal, draws, policy, accept_override=True)
    skip = commit(state, proposal, draws, policy, accept_override=False)
    branches = training.combine_states(take.state, skip.state)
    take_sides, skip_sides = [], []
    take_logits, skip_logits = [], []
    for future in range(t + 1, t + 1 + horizon):
        left, right, outcome, draws = [torch.cat((x, x)) for x in tape.contact(future)]
        choice = native.propose(branches, left, right, outcome, draws, stochastic=True, sensing=policy["sensing"])
        a, b = choice.side.chunk(2)
        a_logit, b_logit = choice.motor_logit.chunk(2)
        take_sides.append(a)
        skip_sides.append(b)
        take_logits.append(a_logit)
        skip_logits.append(b_logit)
        if future < t + horizon:
            branches = commit(branches, choice, draws, policy).state
    a, b = torch.stack(take_sides, dim=1), torch.stack(skip_sides, dim=1)
    truth = tape.truth[:, t + 1:t + 1 + horizon]
    outcomes = tape.outcome[:, t + 1:t + 1 + horizon]
    delta = (a == truth).to(torch.float32) - (b == truth).to(torch.float32)
    a_logit, b_logit = torch.stack(take_logits, dim=1), torch.stack(skip_logits, dim=1)
    return {
        "predicted_benefit": proposal.benefit,
        "predicted_effect": proposal.effect_prediction,
        "forced_receipt_effect": take.receipt.effect_target,
        "skip_receipt_effect": skip.receipt.effect_target,
        "taken_status": take.receipt.status.to(torch.int8),
        "skipped_status": skip.receipt.status.to(torch.int8),
        "applied": take.receipt.status == fw.Status.APPLIED,
        "naturally_accepted": ordinary.accepted,
        "naturally_applied": ordinary.receipt.status == fw.Status.APPLIED,
        "take_side": a.to(torch.int8),
        "skip_side": b.to(torch.int8),
        "future_truth": truth.to(torch.int8),
        "future_outcome": outcomes.to(torch.int8),
        "action_changed": a != b,
        "delta_correct": delta.mean(dim=1),
        "delta_positive_taste": ((a - b) * outcomes).mean(dim=1) / 2,
        "delta_p_correct": ((a_logit * truth).sigmoid() - (b_logit * truth).sigmoid()).mean(dim=1),
        "delta_p_positive_taste": ((a_logit * outcomes).sigmoid() - (b_logit * outcomes).sigmoid()).mean(dim=1),
    }


def pack(records):
    return {name: torch.stack(values, dim=1).detach().cpu().numpy()
            for name, values in records.items()}


def add(records, row):
    for name, values in row.items():
        records.setdefault(name, []).append(values)


def deploy(initial, tape, policy, *, intervention="active", silent=False, fork_times=(), horizon=0,
           record=True):
    state = initial
    observations, forks = {}, {}
    times = set(fork_times)
    stop = tape.length if record or not times else max(times) + 1
    for t in range(stop):
        proposal = propose(state, tape, t, policy, silent=silent)
        result = commit(state, proposal, tape.draws[:, t], policy, intervention=intervention)
        if t in times:
            add(forks, fork_contact(state, proposal, result, tape, t, policy, horizon))
        if record:
            add(observations, observation(proposal, result, tape.truth[:, t]))
        state = result.state
    return state, pack(observations), pack(forks)


def prefix(records, stem, values):
    records.update({f"{stem}/{key}": val for key, val in values.items()})


def join(*observations):
    return {name: np.concatenate([part[name] for part in observations], axis=1)
            for name in observations[0]}


def independent_motor_probes(state, tape, count):
    """Each motor action starts at the same post-B state, before any A feedback."""
    rows = {}
    for t in range(count):
        left, right, _, draws = tape.contact(t)
        _, logit = native.motor(state, left, right)
        side = (draws[:, 2] < logit.sigmoid()).to(torch.int8) * 2 - 1
        add(rows, {"side": side, "motor_logit": logit, "correct": side == tape.truth[:, t]})
    return pack(rows)


def writer_graft(recipient, donor):
    """Copy all writer-receiver rows and writer biases into the cold recipient."""
    dst, src = recipient.graph, donor.graph
    weights = torch.cat((dst.weights[:, :fw.BODY], src.weights[:, fw.BODY:]), dim=1)
    bias = torch.cat((dst.bias[:, :fw.BODY], src.bias[:, fw.BODY:]), dim=1)
    support = torch.cat((dst.present[:, :fw.BODY], src.present[:, fw.BODY:]), dim=1)
    valid = torch.ones(len(support), dtype=torch.bool, device=support.device)
    changed = (support != dst.present).flatten(1).any(dim=1)
    for lane in torch.where(changed)[0].tolist():
        valid[lane] = fw.strongly_connected(support[lane].cpu().numpy())
    graph = fw.Graph(
        torch.where(valid[:, None, None], weights, dst.weights),
        torch.where(valid[:, None], bias, dst.bias),
        torch.where(valid[:, None, None], support, dst.present),
    )
    return native.State(graph, recipient.hidden), valid.cpu().numpy()


def graph_distance(before, after):
    old, new = before.graph, after.graph
    weights = (new.weights - old.weights)[:, fw.BODY:].flatten(1)
    bias = (new.bias - old.bias)[:, fw.BODY:]
    return {
        "writer_norm": torch.cat((weights, bias), dim=1).norm(dim=1).cpu().numpy(),
        "writer_support_changes": (old.present[:, fw.BODY:] != new.present[:, fw.BODY:]).sum((1, 2)).cpu().numpy(),
    }


def sampled_forks(length, count, horizon):
    available = length - horizon
    if count == 0:
        return np.array([], dtype=np.int64)
    if available <= 0:
        raise ValueError("The contact tape needs at least one pre-fork and H continuation contacts")
    return np.unique(np.linspace(0, available - 1, min(count, available)).round().astype(np.int64))


@torch.no_grad()
def evaluate_fit(graph, birth_graph, settings, policy, tapes, horizon):
    data = {}
    start = native.start(graph, settings.worlds)
    birth_start = native.start(birth_graph, settings.worlds)
    early = settings.early
    holdout_start = native.start(graph, settings.holdout_worlds)
    times = sampled_forks(settings.contacts, settings.forks, horizon)

    checkpoint_graft, valid = writer_graft(birth_start, start)
    prefix(data, "checkpoint_writer", {"valid": valid, **graph_distance(birth_start, checkpoint_graft)})
    for condition in SCENARIOS:
        tape = tapes["deployment"][condition]
        for mode, intervention in WRITE_MODES.items():
            fork_times = times if mode == "full" else ()
            _, records, forks = deploy(start, tape, policy, intervention=intervention,
                                       fork_times=fork_times, horizon=horizon)
            prefix(data, f"deployment/{condition}/{mode}", records)
            if forks:
                prefix(data, f"fork/{condition}", {"time": times, **forks})
            if graph is birth_graph:
                graft_records = records
            else:
                _, graft_records, _ = deploy(checkpoint_graft, tape, policy, intervention=intervention)
            prefix(data, f"checkpoint_writer/{condition}/{mode}", graft_records)
        if policy["sensing"] == "own":
            coupled = tape.interleave(tapes["yoked_donor"][condition])
            _, yoked, _ = deploy(native.start(graph, settings.worlds * 2), coupled,
                                 {**policy, "sensing": "yoked"})
            prefix(data, f"deployment/{condition}/yoked", {name: values[::2] for name, values in yoked.items()})
        if len(times):
            _, _, forks = deploy(holdout_start, tapes["fork_holdout"][condition], policy,
                                  fork_times=times, horizon=horizon, record=False)
            prefix(data, f"fork_holdout/{condition}", {"time": times, **forks})
    if len(times):
        validation_tape = tapes["fork_validation"]["mixed"]
        validation_times = sampled_forks(validation_tape.length, settings.forks, horizon)
        _, _, forks = deploy(holdout_start, validation_tape, policy, fork_times=validation_times,
                              horizon=horizon, record=False)
        prefix(data, "fork_validation/mixed", {"time": validation_times, **forks})

    exposure = tapes["return"]
    return_tape = exposure["A_return"]
    donors = {}
    for name in HISTORIES:
        first = {"ABA": "A_first", "CBA": "C_first", "B_only": None}[name]
        if first is None:
            state, segments = start, []
        else:
            state, records, _ = deploy(start, exposure[first], policy)
            segments = [records]
        donor, b_records, _ = deploy(state, exposure["B"], policy)
        donors[name] = donor
        _, a_records, _ = deploy(donor, return_tape, policy)
        prefix(data, f"return/{name}", join(*segments, b_records, a_records))
        _, silent, _ = deploy(donor, return_tape.window(0, early), policy, intervention="disabled", silent=True)
        pre_feedback = independent_motor_probes(donor, return_tape, early)
        np.testing.assert_array_equal(silent["side"][:, 0], a_records["side"][:, 0])
        np.testing.assert_array_equal(pre_feedback["side"][:, 0], a_records["side"][:, 0])
        prefix(data, f"return_silent/{name}", silent)
        prefix(data, f"return_pre_feedback/{name}", pre_feedback)
        prefix(data, f"return_state/{name}", graph_distance(start, donor))

    for name, donor in (("A_writer", donors["ABA"]), ("C_writer", donors["CBA"]),
                        ("B_writer", donors["B_only"]), ("fresh_writer", start)):
        graft, valid = writer_graft(start, donor)
        prefix(data, f"transplant/{name}", {"valid": valid, **graph_distance(start, graft)})
        for mode, intervention in WRITE_MODES.items():
            _, records, _ = deploy(graft, return_tape, policy, intervention=intervention)
            prefix(data, f"transplant/{name}/{mode}", records)
    return data


def make_tapes(settings, birth, config):
    tapes = {split: environment_tapes(settings, birth, split) for split in
             ("deployment", "return", "fork_holdout", "yoked_donor")}
    validation = training.make_worlds(
        [settings.seed, birth, SPLIT_DOMAINS["fork_validation"]], settings.holdout_worlds,
        config["contacts"], noise=config["noise"], layout="mixed",
    )
    cue_unavailable = torch.full_like(validation.truth, -1, dtype=torch.int16)
    tapes["fork_validation"] = {"mixed": Tape(validation.x, validation.truth, validation.outcome,
                                               validation.draws, cue_unavailable)}
    return tapes


def save_tapes(out, settings, births, config):
    files = {}
    for birth in births:
        records = {}
        for split, phases in make_tapes(settings, birth, config).items():
            for phase, tape in phases.items():
                prefix(records, f"{split}/{phase}", {name: val.numpy() for name, val in vars(tape).items()})
        path = Path(f"birth_{birth}/worlds.npz")
        training.write_arrays(out / path, **records)
        files[str(birth)] = {"file": path.as_posix(), "sha256": digest(out / path)}
    return files


def fit_file(out, arm, birth, step):
    return out / f"birth_{birth}/{arm}/checkpoint_{step}.npz"


def evaluate_checkpoint(run_dir, out, settings, config, identity):
    torch.set_num_threads(1)
    birth, arm, step = (identity[name] for name in ("birth", "arm", "step"))
    birth_graph = training.load_checkpoint(run_dir, arm, birth, 0)
    graph = birth_graph if step == 0 else training.load_checkpoint(run_dir, arm, birth, step)
    tapes = make_tapes(settings, birth, config)
    records = evaluate_fit(graph, birth_graph, settings, policy_settings(arm, config), tapes, config["horizon"])
    path = fit_file(out, arm, birth, step)
    training.write_arrays(path, **records)
    result = {**identity, "file": path.relative_to(out).as_posix(), "sha256": digest(path)}
    write_json(path.with_suffix(".json"), result)
    print(f"evaluated birth={birth} arm={arm} step={step}", flush=True)
    return result


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def boot_pairs(births, worlds, draws, seed):
    rng = np.random.default_rng([seed, 0xEE19])
    return (rng.integers(births, size=(draws, births, 1)), rng.integers(worlds, size=(draws, births, worlds)))


def paired_stats(values, indices):
    values = np.asarray(values, dtype=np.float64)
    reps = values[indices].mean(axis=(1, 2))
    return {"mean": float(values.mean()), "ci95": np.quantile(reps, [.025, .975]).tolist(),
            "by_birth": values.mean(axis=1).tolist()}


def windows(settings, scenario):
    first = {"contact_1": (0, 1), "early": (0, settings.early),
             "late": (settings.contacts - settings.late, settings.contacts)}
    if scenario == "reversal":
        flip = reversal_at(settings)
        first.update({"before_flip": (flip - settings.late, flip), "after_flip_1": (flip, flip + 1),
                      "after_flip_early": (flip, flip + settings.early)})
    return first


def paired_accuracy_scores(arrays, windows_to_use, indices):
    return {window: {name: paired_stats(a[:, :, lo:hi].mean(axis=2), indices)
                     for name, a in arrays.items()} for window, (lo, hi) in windows_to_use.items()}


def score_windows(out, settings, births, arms, checkpoints):
    indices = boot_pairs(len(births), settings.worlds, settings.bootstrap_draws, settings.seed)
    phase = settings.phase_contacts
    return_windows = {"contact_1": (0, 1), "early": (0, settings.early),
                      "late": (phase - settings.late, phase)}
    by_step, birth_accuracy = {}, {}
    for step in checkpoints:
        result = {"arms": {}, "arm_contrasts": {}}
        performance = {}
        for arm in arms:
            files = [np.load(fit_file(out, arm, birth, step), allow_pickle=False) for birth in births]
            try:
                def read(key):
                    return np.stack([file[key] for file in files]).astype(float)

                arm_result = {"deployment": {}, "checkpoint_writer": {}}
                performance[arm] = {}
                for scenario in SCENARIOS:
                    actual = {}
                    graft = {}
                    for mode in WRITE_MODES:
                        actual[mode] = read(f"deployment/{scenario}/{mode}/correct")
                        graft[mode] = read(f"checkpoint_writer/{scenario}/{mode}/correct")
                    write_effect = actual["full"] - actual["no_write"]
                    performance[arm][scenario] = (actual["full"], write_effect)
                    outcomes = {**actual, "full_minus_no_write": write_effect}
                    outcomes["side_changed_by_disabling_writes"] = (
                        read(f"deployment/{scenario}/full/side") != read(f"deployment/{scenario}/no_write/side"))
                    if training.arm_sensing(arm) == "own":
                        yoked = read(f"deployment/{scenario}/yoked/correct")
                        outcomes.update({"yoked": yoked, "own_minus_yoked": actual["full"] - yoked})
                    writer_outcomes = {**graft, "full_minus_no_write": graft["full"] - graft["no_write"]}
                    if step == 0:
                        birth_accuracy[arm, scenario] = actual
                    base = birth_accuracy.get((arm, scenario))
                    if base is not None:
                        for mode in WRITE_MODES:
                            outcomes[f"{mode}_minus_birth"] = actual[mode] - base[mode]
                            writer_outcomes[f"{mode}_minus_birth"] = graft[mode] - base[mode]
                        outcomes["write_effect_minus_birth"] = outcomes["full_minus_birth"] - outcomes["no_write_minus_birth"]
                        writer_outcomes["write_effect_minus_birth"] = (
                            writer_outcomes["full_minus_birth"] - writer_outcomes["no_write_minus_birth"])
                    deployment = paired_accuracy_scores(outcomes, windows(settings, scenario), indices)
                    deployment["late_minus_early"] = {
                        name: paired_stats(values[:, :, -settings.late:].mean(axis=2)
                                           - values[:, :, :settings.early].mean(axis=2), indices)
                        for name, values in outcomes.items()
                    }
                    if scenario == "reversal":
                        flip = reversal_at(settings)
                        deployment["recovery_late_minus_after_flip_early"] = {
                            name: paired_stats(values[:, :, -settings.late:].mean(axis=2)
                                               - values[:, :, flip:flip + settings.early].mean(axis=2), indices)
                            for name, values in outcomes.items()
                        }
                    arm_result["deployment"][scenario] = deployment
                    arm_result["checkpoint_writer"][scenario] = paired_accuracy_scores(
                        writer_outcomes, windows(settings, scenario), indices)

                for group in ("return", "return_silent", "return_pre_feedback"):
                    paths = {name: read(f"{group}/{name}/correct") for name in HISTORIES}
                    if group == "return":
                        paths = {name: values[:, :, -phase:] for name, values in paths.items()}
                        probes = return_windows
                    else:
                        probes = {name: return_windows[name] for name in ("contact_1", "early")}
                    paths.update({"ABA_minus_CBA": paths["ABA"] - paths["CBA"],
                                  "ABA_minus_B_only": paths["ABA"] - paths["B_only"]})
                    arm_result[group] = paired_accuracy_scores(paths, probes, indices)
                arm_result["transplant"] = {}
                transplant = {}
                for mode in WRITE_MODES:
                    grafts = {name: read(f"transplant/{name}/{mode}/correct") for name in
                              ("A_writer", "C_writer", "B_writer", "fresh_writer")}
                    differences = {f"A_writer_minus_{control}": grafts["A_writer"] - grafts[control]
                                   for control in ("C_writer", "B_writer", "fresh_writer")}
                    transplant[mode] = differences
                    arm_result["transplant"][mode] = paired_accuracy_scores(
                        {**grafts, **differences}, return_windows, indices)
                arm_result["transplant"]["write_dependent_advantage"] = paired_accuracy_scores(
                    {name: arr - transplant["no_write"][name] for name, arr in transplant["full"].items()},
                    return_windows, indices)
                arm_result["graft_valid_rate"] = {
                    label: float(read(f"{label}/valid").mean())
                    for label in ("checkpoint_writer", "transplant/A_writer", "transplant/C_writer",
                                  "transplant/B_writer")
                }
                arm_result["forks"] = summarize_forks(read, settings)
                result["arms"][arm] = arm_result
            finally:
                for file in files:
                    file.close()

        for lhs, rhs in (("causal", "sensing"), ("causal", "shuffled"), ("sensing", "baseline"),
                         ("causal", "baseline")):
            if lhs not in performance or rhs not in performance:
                continue
            result["arm_contrasts"][f"{lhs}_minus_{rhs}"] = {}
            for scenario in SCENARIOS:
                a, a_write = performance[lhs][scenario]
                b, b_write = performance[rhs][scenario]
                result["arm_contrasts"][f"{lhs}_minus_{rhs}"][scenario] = paired_accuracy_scores(
                    {"full": a - b, "no_write": (a - a_write) - (b - b_write),
                     "write_effect": a_write - b_write}, windows(settings, scenario), indices)
        by_step[str(step)] = result
    return by_step


def calibrate(prediction, observed, *, constant=None):
    """Summarize a fixed forecast and its zero-effect comparator."""
    count = int(observed.size)
    nonzero = np.abs(observed) > 1e-7
    stats = {"count": count, "nonzero_count": int(nonzero.sum())}
    if not count:
        return stats
    errors = prediction - observed
    stats.update({
        "prediction_mean": float(prediction.mean()), "target_mean": float(observed.mean()),
        "target_abs_mean": float(np.abs(observed).mean()), "bias": float(errors.mean()),
        "mae": float(np.abs(errors).mean()), "mse": float((errors ** 2).mean()),
        "zero_baseline_mse": float((observed ** 2).mean()),
        "sign_accuracy_on_nonzero": float(np.mean(np.sign(prediction[nonzero]) == np.sign(observed[nonzero])))
        if nonzero.any() else None,
        "always_positive_on_nonzero": float(np.mean(observed[nonzero] > 0)) if nonzero.any() else None,
        "always_negative_on_nonzero": float(np.mean(observed[nonzero] < 0)) if nonzero.any() else None,
    })
    if constant is not None:
        stats.update({"development_constant": constant, "development_constant_mse":
                      float(((observed - constant) ** 2).mean())})
    variance = prediction.var()
    if variance > 1e-14:
        covariance = np.mean((prediction - prediction.mean()) * (observed - observed.mean()))
        stats["calibration_slope"] = float(covariance / variance)
        if observed.var() > 1e-14:
            stats["pearson_r"] = float(covariance / np.sqrt(variance * observed.var()))
    edges = [-1., -.25, -.05, -.01, 0., .01, .05, .25, 1.]
    slots = np.searchsorted(edges[1:-1], prediction, side="right")
    stats["calibration_bins"] = [
        {"range": [lo, hi], "count": int(which.sum()), "prediction": float(prediction[which].mean()),
         "observed": float(observed[which].mean())}
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)) if (which := slots == i).any()
    ]
    return stats


def summarize_forks(read, settings):
    if not settings.forks:
        return {}
    stats, constants = {}, {}
    targets = ("delta_positive_taste", "delta_p_positive_taste", "delta_correct", "delta_p_correct")
    subsets = {"all": "all", "applicable": "applied", "naturally_applied": "naturally_applied"}
    for split, worlds in (("fork", SCENARIOS), ("fork_holdout", SCENARIOS), ("fork_validation", ("mixed",))):
        stats[split] = {}
        for scenario in worlds:
            path = f"{split}/{scenario}"
            predicted = read(f"{path}/predicted_benefit")
            changed = read(f"{path}/action_changed").astype(bool)
            world = {}
            for name, key in subsets.items():
                include = np.ones_like(predicted, dtype=bool) if key == "all" else read(f"{path}/{key}").astype(bool)
                subset = {
                    "action_changed_count": int(changed.any(axis=-1)[include].sum()),
                    "action_changed_fraction": float(changed.any(axis=-1)[include].mean()) if include.any() else None,
                    "action_changed_per_contact": float(changed[include].mean()) if include.any() else None,
                }
                for metric in targets:
                    delta = read(f"{path}/{metric}")[include]
                    constant_key = (scenario, name, metric)
                    if split == "fork" and delta.size:
                        constants[constant_key] = float(delta.mean())
                    baseline = constants.get(constant_key) if split == "fork_holdout" else None
                    subset[metric] = calibrate(predicted[include], delta, constant=baseline)
                world[name] = subset
            local_pred = read(f"{path}/predicted_effect")
            local_actual = read(f"{path}/forced_receipt_effect")
            applicable = read(f"{path}/applied").astype(bool)
            world["local_receipt_applied"] = {
                name: calibrate(local_pred[..., i][applicable], local_actual[..., i][applicable])
                for i, name in enumerate(("local_activation", "value", "topology"))
            }
            stats[split][scenario] = world
    return stats


def write_json(path, record):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf8")
    tmp.replace(path)


def validate(settings, config):
    if min(settings.worlds, settings.holdout_worlds, settings.contacts, settings.phase_contacts,
           settings.early, settings.late, settings.bootstrap_draws) < 1:
        raise ValueError("World counts, contacts, analysis windows and bootstrap draws must be positive")
    if not 1 <= settings.channels <= 4 or not 0 <= settings.noise <= 1 or not 0 <= settings.holdout_noise <= 1:
        raise ValueError("Require 1–4 separable odor channels and noise rates in [0, 1]")
    flip = reversal_at(settings)
    if (settings.early > min(settings.contacts, settings.phase_contacts, settings.contacts - flip)
            or settings.late > min(flip, settings.phase_contacts, settings.contacts - flip)):
        raise ValueError("The early and late windows must fit the selected stable, reversal and return phases")
    if settings.seed < 0 or settings.forks < 0 or (settings.forks and config["horizon"] >= settings.contacts):
        raise ValueError("Require nonnegative seed/forks and a future horizon shorter than the evaluation tape")


def select(requested, available, what):
    if requested is None:
        return available
    unknown = set(requested) - set(available)
    if unknown:
        raise ValueError(f"Unknown {what} in {sorted(unknown)}")
    return [value for value in available if value in requested]


def load_training_manifest(path, follow, poll_seconds, idle_timeout):
    started = monotonic()
    while True:
        try:
            content = path.read_bytes()
            return json.loads(content), sha256(content).hexdigest()
        except (FileNotFoundError, json.JSONDecodeError):
            if not follow:
                raise
            if idle_timeout is not None and monotonic() - started >= idle_timeout:
                raise TimeoutError(f"Timed out waiting for a complete training manifest at {path}") from None
            sleep(poll_seconds)


def checkpoint_identity(run_dir, protocol_hash, follow, key):
    birth, arm, step = key
    paths = [training.checkpoint_path(run_dir, arm, birth, stage) for stage in (0, step)]
    for file in paths:
        if not file.is_file():
            if follow:
                return None
            raise FileNotFoundError(f"Required training checkpoint is missing: {file}")
    birth_hash = digest(paths[0])
    return {"birth": birth, "arm": arm, "step": step, "protocol_sha256": protocol_hash,
            "birth_checkpoint_sha256": birth_hash, "checkpoint_sha256": birth_hash if step == 0 else digest(paths[1])}


def reusable_fit(out, identity):
    path = fit_file(out, identity["arm"], identity["birth"], identity["step"])
    try:
        metadata = json.loads(path.with_suffix(".json").read_bytes())
        if all(metadata.get(key) == value for key, value in identity.items()) and digest(path) == metadata.get("sha256"):
            return metadata
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return None


def evaluate_checkpoints(run_dir, out, settings, config, keys, protocol_hash, *, workers, follow, poll_seconds,
                         idle_timeout):
    results, running = {}, {}
    pending = list(keys)
    last_progress = monotonic()
    if workers == 1 and not follow:
        for key in keys:
            identity = checkpoint_identity(run_dir, protocol_hash, follow, key)
            results[key] = reusable_fit(out, identity) or evaluate_checkpoint(run_dir, out, settings, config, identity)
        return [results[key] for key in keys]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        while pending or running:
            still_pending = []
            for index, key in enumerate(pending):
                if len(running) >= workers:
                    still_pending.extend(pending[index:])
                    break
                identity = checkpoint_identity(run_dir, protocol_hash, follow, key)
                if identity is None:
                    still_pending.append(key)
                    continue
                existing = reusable_fit(out, identity)
                if existing is None:
                    job = pool.submit(evaluate_checkpoint, run_dir, out, settings, config, identity)
                    running[job] = key
                else:
                    results[key] = existing
                    last_progress = monotonic()
            pending = still_pending
            if running:
                completed, _ = wait(running, timeout=poll_seconds, return_when=FIRST_COMPLETED)
                for job in completed:
                    results[running.pop(job)] = job.result()
                    last_progress = monotonic()
            elif pending:
                if idle_timeout is not None and monotonic() - last_progress >= idle_timeout:
                    birth, arm, step = pending[0]
                    raise TimeoutError(f"Waiting for {len(pending)} checkpoints, beginning with birth={birth}, "
                                       f"arm={arm}, step={step}")
                sleep(poll_seconds)
    return [results[key] for key in keys]


def run(run_dir, out, settings, *, births=None, arms=None, checkpoints=None, workers=1, follow=False,
        poll_seconds=5., idle_timeout=None):
    run_dir, out = Path(run_dir).resolve(), Path(out).resolve()
    if run_dir == out or workers < 1 or not 0 < poll_seconds <= 60 or (idle_timeout is not None and idle_timeout <= 0):
        raise ValueError("Choose a separate evaluation directory, at least one worker and positive wait limits")
    training_manifest = run_dir / "run.json"
    plan, plan_hash = load_training_manifest(training_manifest, follow, poll_seconds, idle_timeout)
    config = plan["config"]
    validate(settings, config)
    births = select(births, plan["births"], "births")
    arms = select(arms, plan["arms"], "arms")
    checkpoints = select(checkpoints, plan["checkpoints"], "checkpoints")
    for birth in births:
        for arm in arms:
            for step in {0, *checkpoints}:
                file = training.checkpoint_path(run_dir, arm, birth, step)
                if not follow and not file.is_file():
                    raise FileNotFoundError(f"Required training checkpoint is missing: {file}")
                if fit_file(out, arm, birth, step).resolve() == file.resolve():
                    raise ValueError(f"Evaluation would overwrite training checkpoint {file}")

    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out / "protocol.json"
    old_protocol = json.loads(protocol_path.read_bytes()) if protocol_path.exists() else None
    tapes = old_protocol["tapes"] if old_protocol is not None else save_tapes(out, settings, births, config)
    source_files = [Path(module.__file__) for module in (native, training, fw)] + [Path(__file__)]
    sources = {path.name: digest(path) for path in source_files}
    source_mismatch = [name for name, expected in plan.get("sources_sha256", {}).items()
                       if sources.get(name) != expected]
    manifest = {
        "schema_version": 1, "settings": asdict(settings), "training_config": config,
        "training_manifest": str(training_manifest), "training_manifest_sha256": plan_hash,
        "sources_sha256": sources, "source_mismatch_with_training_manifest": source_mismatch,
        "software": {"numpy": np.__version__, "torch": torch.__version__},
        "births": births, "arms": arms, "checkpoints": checkpoints, "tapes": tapes,
        "split_key": [settings.seed, "birth", "domain"], "split_domains": SPLIT_DOMAINS,
        "reversal_at_zero_based": reversal_at(settings),
        "deployment_windows_zero_based_half_open": {s: windows(settings, s) for s in SCENARIOS},
        "fork_horizon": config["horizon"], "teacher_target": config.get("credit_target", "taste"),
        "world_and_protocol": {
            "deployment": "Stable and reversal share input and all exogenous draws. The clean and noisy taste "
                          "signs invert together at reversal_at_zero_based. The policy sees only the actual taste.",
            "disabled": "Blocks physical graph writes and preserves recurrent sensing of the actual disabled receipt.",
            "yoked": "All own-sensing arms are also deployed with the native yoked pre/post local sensing from a "
                     "paired independently drawn donor world. Both lanes start at the same checkpoint. Only the "
                     "primary world's outcomes are scored, with its original tape and draws.",
            "return": "ABA and CBA use one odor bank in the first phase, opposite hidden taste assignments, "
                      "and the same exogenous draws. B uses new odors and is shared exactly across histories. "
                      "A return uses the first A assignments on new presentations and is shared exactly. "
                      "B_only starts at the same cold checkpoint and skips the first phase.",
            "return_silent": "Serial return controls suppress A taste and physical writes across the short prefix.",
            "return_pre_feedback": "Each early A motor sample forks from the same unmodified post-B state, "
                                   "with no contacts, taste or edits from earlier A probes.",
            "transplant": "After B, copy only the writer-receiver weight/support rows and writer biases into the "
                          "same cold checkpoint. Checkpoint_writer uses the trained checkpoint as donor and "
                          "its matched step-0 cold body as recipient. Donor activity is never copied. Disconnected "
                          "grafts fall back to the recipient and remain identifiable in valid.",
            "fork": "At predetermined contacts of a live deployment, force or suppress the same proposal. "
                    "Both branches sample their future decisions with identical exogenous draws and continue "
                    "normal writes. Fork_holdout uses wholly independent worlds and a different slip rate. "
                    "Fork_validation uses independently sampled worlds with the training contact count, "
                    "two-cue stimulus distribution, layout mix and slip rate.",
            "statistics": "Score whole worlds within birth, then average births with equal weight. The paired "
                          "hierarchical bootstrap resamples births and worlds within each resampled birth; "
                          "intervals are pointwise. Calibration fits no model to holdout: zero effect and "
                          "the constant mean of the corresponding deployment-fork split are comparators.",
        },
        "arrays": {
            "tape": "[world, contact, ...]. Draw order is address, operation, motor, gate. "
                    "Truth/outcome are the clean/noisy right-side taste signs.",
            "deployment": "[world, contact, ...], actual signed side, delivered taste, correctness, "
                          "one-step p_correct, writer address/op, accepted/applied/status and predicted/actual effects.",
            "return": "[world, contact, ...]. ABA/CBA have three contiguous phase_contacts segments, "
                      "B_only has two. The final phase is always A return.",
            "transplant": "[world, A-return contact, ...]; valid and writer distances are [world]. "
                          "Checkpoint_writer score trajectories use the deployment contact axis.",
            "fork": "[world, fork contact, ...]. The time vector is zero-based. Future side/truth/action_changed "
                    "arrays have an additional horizon axis. Targets are horizon mean take-minus-skip in "
                    "positive delivered taste/clean correctness, or their per-branch sigmoid probabilities.",
        },
    }
    if old_protocol is not None:
        if json.loads(json.dumps(manifest)) != old_protocol:
            raise ValueError(f"Existing protocol differs from requested evaluation: {protocol_path}")
        for item in tapes.values():
            tape = out / item["file"]
            if digest(tape) != item["sha256"]:
                raise ValueError(f"Evaluation tape mismatch at {tape}")
    else:
        write_json(protocol_path, manifest)
    protocol_hash = digest(protocol_path)
    keys = [(birth, arm, step) for step in checkpoints for birth in births for arm in arms]
    outputs = evaluate_checkpoints(run_dir, out, settings, config, keys, protocol_hash, workers=workers, follow=follow,
                                   poll_seconds=poll_seconds, idle_timeout=idle_timeout)
    scores = score_windows(out, settings, births, arms, checkpoints)
    write_json(out / "summary.json", {
        "protocol_sha256": protocol_hash, "fits": outputs, "steps": scores,
    })
    return scores


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, help="Evaluation output directory. Defaults to RUN/evaluation.")
    parser.add_argument("--births", type=int, nargs="+")
    parser.add_argument("--arms", choices=training.ARMS, nargs="+")
    parser.add_argument("--checkpoints", "--steps", dest="checkpoints", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--follow", action="store_true", help="Evaluate available checkpoints as training saves them.")
    parser.add_argument("--poll-seconds", type=float, default=5.)
    parser.add_argument("--idle-timeout", type=float, help="Fail after this many idle seconds waiting for training.")
    for name, default in asdict(Settings()).items():
        kind = float if isinstance(default, float) else int
        parser.add_argument(f"--{name.replace('_', '-')}", type=kind, default=default)
    args = vars(parser.parse_args())
    run_dir, out = args.pop("run"), args.pop("out")
    selections = {name: args.pop(name) for name in
                  ("births", "arms", "checkpoints", "workers", "follow", "poll_seconds", "idle_timeout")}
    run(run_dir, out or (run_dir / "evaluation"), Settings(**args), **selections)


if __name__ == "__main__":
    main()
