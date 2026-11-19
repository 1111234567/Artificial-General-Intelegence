"""Train and test paired scalar-reward writers with controlled self-readback."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch import nn

from . import self_write_view as env


EVIDENCE = Path("docs/evidence/self-writing-2026-11-19/view")
TEST_GROUPS = tuple(range(873100, 873116))
TEST_BATCH = 128
DEV_SEED = 285197
TRAIN_SEEDS = tuple(range(192300, 192308))
BOOTSTRAP_BATCHES = 1000
BOOTSTRAP_BATCH = 192
PANEL = env.MODES + ("swap_to_current_only", "swap_to_blind", "swap_to_yoked", "swap_to_stale",
                     "random_command", "random_valid_edit", "oracle")


class Writer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(env.FEATURES, 48), nn.SiLU(), nn.Linear(48, 48), nn.SiLU())
        self.commands = nn.Linear(48, env.STRIDE)
        self.idle = nn.Linear(48, 1)

    def forward(self, observation):
        tokens = self.encode(observation)
        pooled = tokens.mean(dim=1)
        return torch.cat((self.commands(tokens).flatten(1), self.idle(pooled)), dim=1)


def utility(error, initial_error, attempted):
    # Relative improvement is meaningful even when different perturbations have
    # different absolute force. The floor sets the scale of sham penalties.
    improvement = np.clip((initial_error - error) / np.maximum(initial_error, 0.00025), -1, 1)
    return np.where(improvement < 0, 0.25 * improvement, improvement) - 0.02 * attempted


def measurements(trial):
    target = env.run_body(trial.reference, env.bank_drive(trial))
    damage = env.motor_error(trial.damaged, trial, target)
    return target, damage


def trial_arrays(trial):
    return {"reference_exists": trial.reference.exists, "reference_values": trial.reference.values,
            "damaged_exists": trial.damaged.exists, "damaged_values": trial.damaged.values,
            "drive": trial.drive, "disturbance": trial.disturbance, "inverse": trial.inverse, "yoke": trial.yoke}


def trial_from_arrays(arrays):
    return env.Trial(env.Body(arrays["reference_exists"], arrays["reference_values"]),
                     env.Body(arrays["damaged_exists"], arrays["damaged_values"]), arrays["drive"],
                     arrays["disturbance"], arrays["inverse"], arrays["yoke"])


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(directory):
    directory.mkdir(parents=True, exist_ok=True)
    holdout = directory / "holdout.npz"
    manifest = directory / "seal.json"
    if holdout.exists() or manifest.exists():
        raise FileExistsError("Holdout already exists")
    blocks = [trial_arrays(env.random_trial(seed, TEST_BATCH)) for seed in TEST_GROUPS]
    np.savez_compressed(holdout, **{key: np.stack([record[key] for record in blocks]) for key in blocks[0]})
    manifest.write_text(json.dumps({"holdout_sha256": sha256(holdout), "block_seeds": TEST_GROUPS,
                                    "trials_per_block": TEST_BATCH, "development_seed": DEV_SEED,
                                    "actor_seeds": TRAIN_SEEDS}, indent=2) + "\n", encoding="utf-8")


def train(seed, batches, batch, modes, *, report=False):
    writers = {}
    optimizers = {}
    for mode in modes:
        torch.manual_seed(seed)
        writers[mode] = Writer()
        optimizers[mode] = torch.optim.Adam(writers[mode].parameters(), lr=0.0018)
    rng = np.random.default_rng(seed)
    history = []
    totals = np.zeros((len(modes), 4))
    for step in range(batches):
        trial = env.random_trial(int(rng.integers(1 << 62)), batch)
        target, damage = measurements(trial)
        public = env.public_signals(trial)
        sampled = rng.integers(env.ACTION_COUNT, size=batch)
        explore = rng.random(batch) < (0.95 - 0.25 * step / batches)
        for index, mode in enumerate(modes):
            values = writers[mode](torch.from_numpy(env.local_features(trial, mode, public)))
            actions = np.where(explore, sampled, values.detach().argmax(dim=1).numpy())
            receipt = env.apply(trial.damaged, actions)
            error = env.motor_error(receipt.body, trial, target)
            reward = utility(error, damage, actions != env.NOOP).astype(np.float32)
            outcome = torch.from_numpy(reward)
            chosen = values.gather(1, torch.from_numpy(actions[:, None]))[:, 0]
            objective = (chosen - outcome).square().mean()
            optimizers[mode].zero_grad(set_to_none=True)
            objective.backward()
            nn.utils.clip_grad_norm_(writers[mode].parameters(), 1.0)
            optimizers[mode].step()
            totals[index] += (reward.mean(), receipt.changed.mean(), (actions == trial.inverse).mean(),
                              (actions == env.NOOP).mean())
        if (step + 1) % 50 == 0 or step + 1 == batches:
            count = (step % 50) + 1
            history.append({"step": step + 1, "modes": {mode: (totals[i] / count).tolist()
                                                        for i, mode in enumerate(modes)}})
            if report:
                print(json.dumps(history[-1]), flush=True)
            totals.fill(0)
    return writers, history


def oracle_controls(trial, target, damage, seed):
    size = len(trial.drive)
    rng = np.random.default_rng(seed)
    potentials = np.empty((size, env.ACTION_COUNT), dtype=np.float32)
    eligible = np.empty_like(potentials, dtype=bool)
    for action in range(env.ACTION_COUNT):
        receipt = env.apply(trial.damaged, np.full(size, action))
        error = env.motor_error(receipt.body, trial, target)
        potentials[:, action] = utility(error, damage, action != env.NOOP)
        eligible[:, action] = receipt.valid
    valid_edit = eligible.copy()
    valid_edit[:, env.NOOP] = False
    random_ranking = rng.random(potentials.shape)
    return {"random_command": rng.integers(env.ACTION_COUNT, size=size),
            "random_valid_edit": np.argmax(np.where(valid_edit, random_ranking, -1), axis=1),
            "oracle": np.argmax(np.where(eligible, potentials, -np.inf), axis=1)}, potentials, eligible


def score(trial, target, damage, actions, oracle, eligible, donor_ranking):
    receipt = env.apply(trial.damaged, actions)
    error = env.motor_error(receipt.body, trial, target)
    undo_actions = env.inverse_action(trial.damaged, actions, receipt)
    reverted = env.apply(receipt.body, undo_actions)
    undo_error = env.motor_error(reverted.body, trial, target)
    undone = env.exact_rows(reverted.body, trial.damaged)
    if not undone.all():
        raise AssertionError("Undo failed to restore actual pre-action body")

    slots = np.arange(env.SLOTS)
    donor_actions = slots[None, :] * env.STRIDE + np.maximum(receipt.operation[:, None], 0)
    donors = np.take_along_axis(eligible, donor_actions, axis=1)
    donors &= receipt.changed[:, None] & (slots[None, :] != receipt.slot[:, None])
    donors &= (slots[None, :] < env.EDGES) == (receipt.slot[:, None] < env.EDGES)
    same_remove_dose = trial.damaged.values == -receipt.delta[:, None]
    donors &= (receipt.operation[:, None] != env.REMOVE) | same_remove_dose
    candidate = np.argmax(np.where(donors, donor_ranking, -1), axis=1)
    retarget_available = donors.any(axis=1)
    rows = np.arange(len(actions))
    retarget_actions = np.where(retarget_available, donor_actions[rows, candidate], env.NOOP)
    retarget = env.apply(reverted.body, retarget_actions)
    if not np.array_equal(retarget.changed, retarget_available):
        raise AssertionError("A matched retarget failed to execute")
    np.testing.assert_array_equal(retarget.delta[retarget_available], receipt.delta[retarget_available])
    retarget_error = env.motor_error(retarget.body, trial, target)
    intended_slot, intended_operation = env.decode(trial.inverse)
    disturbed = trial.disturbance != env.NOOP
    expected_delta = trial.reference.values[rows, intended_slot] - trial.damaged.values[rows, intended_slot]
    benefit = damage - error
    return {"action": actions, "valid": receipt.valid, "changed": receipt.changed, "slot": receipt.slot,
            "operation": receipt.operation, "delta": receipt.delta, "error": error, "initial_error": damage,
            "benefit": benefit, "restored": env.exact_rows(receipt.body, trial.reference),
            "intended_slot": disturbed & receipt.changed & (receipt.slot == intended_slot),
            "intended_operation": disturbed & receipt.changed & (receipt.operation == intended_operation),
            "signed_dose": disturbed & receipt.changed & (receipt.delta == expected_delta),
            "direction": disturbed & receipt.changed & (np.sign(receipt.delta) == np.sign(expected_delta)),
            "intended_edge": disturbed & (intended_slot < env.EDGES),
            "intended_bias": disturbed & (intended_slot >= env.EDGES),
            "edge_written": receipt.changed & (receipt.slot < env.EDGES),
            "bias_written": receipt.changed & (receipt.slot >= env.EDGES),
            "regret": oracle - utility(error, damage, actions != env.NOOP),
            "undone": undone, "undo_error": undo_error, "undo_action": undo_actions,
            "retarget_action": retarget_actions, "retarget_available": retarget_available,
            "retarget_error": retarget_error}


def choose(writer, trial, mode, public):
    with torch.inference_mode():
        scores = writer(torch.from_numpy(env.local_features(trial, mode, public)))
        return scores.argmax(dim=1).numpy()


def evaluate(writers, trial, seed):
    target, damage = measurements(trial)
    public = env.public_signals(trial)
    choices, values, eligible = oracle_controls(trial, target, damage, seed)
    for mode, writer in writers.items():
        choices[mode] = choose(writer, trial, mode, public)
    if "veridical" in writers:
        for mode in env.MODES[1:]:
            choices[f"swap_to_{mode}"] = choose(writers["veridical"], trial, mode, public)
    optimum = values[np.arange(len(damage)), choices["oracle"]]
    rng = np.random.default_rng(seed + 1)
    ranking = rng.random((len(damage), env.SLOTS))
    return {mode: score(trial, target, damage, actions, optimum, eligible, ranking)
            for mode, actions in choices.items()}


def mean(values, mask=None):
    if mask is None:
        return float(values.mean())
    count = np.count_nonzero(mask)
    return float(values[mask].mean()) if count else None


def metrics(record, disturbed, kinds):
    damaged = np.broadcast_to(disturbed, record["action"].shape)
    attempted = record["action"] != env.NOOP
    branches = record["retarget_available"] & damaged
    accepted = record["changed"] & damaged
    return {"valid_given_attempt": mean(record["valid"], attempted),
            "abstain_rate": mean(~attempted), "no_effect_rate": mean(~record["changed"]),
            "invalid_rate": mean(~record["valid"]), "changed_rate": mean(record["changed"]),
            "damaged_changed_rate": mean(record["changed"], damaged),
            "damaged_exact_recovery": mean(record["restored"], damaged),
            "sham_preserved": mean(record["restored"], ~damaged),
            "target_hit": mean(record["intended_slot"], damaged),
            "opcode_hit": mean(record["intended_operation"], damaged),
            "dose_match": mean(record["signed_dose"], damaged),
            "direction_match": mean(record["direction"], damaged),
            "edge_target_hit": mean(record["intended_slot"], record["intended_edge"]),
            "bias_target_hit": mean(record["intended_slot"], record["intended_bias"]),
            "edge_written_count": int(record["edge_written"].sum()),
            "bias_written_count": int(record["bias_written"].sum()),
            "delivered_dose_counts": {str(dose): int((record["delta"][record["changed"]] == dose).sum())
                                      for dose in env.DELTAS.tolist()},
            "exact_by_disturbance": {kind: mean(record["restored"], np.broadcast_to(mask, damaged.shape))
                                     for kind, mask in kinds.items()},
            "additional_damage_mse_sham": mean(record["error"], ~damaged),
            "damaged_initial_mse": mean(record["initial_error"], damaged),
            "damaged_final_mse": mean(record["error"], damaged),
            "damaged_benefit_mse": mean(record["benefit"], damaged),
            "damaged_improved": mean(record["benefit"] > 0, damaged),
            "damaged_utility_regret": mean(record["regret"], damaged),
            "causal_undo_count": int(accepted.sum()), "causal_undo_exact": mean(record["undone"], accepted),
            "causal_undo_added_error": mean(record["undo_error"] - record["error"], accepted),
            "matched_retarget_count": int(branches.sum()),
            "matched_retarget_added_error": mean(record["retarget_error"] - record["error"], branches)}


def paired_interval(first, second, mask):
    """Resample paired training seeds and held-out batch clusters independently."""
    delta = (first.astype(np.float64) - second) * mask[None, :, :]
    count = mask.sum(axis=1)
    numerators = delta.sum(axis=2)
    rng = np.random.default_rng(821571)
    samples = 2000
    seeds, groups = numerators.shape
    seed_draws = rng.multinomial(seeds, np.full(seeds, 1 / seeds), size=samples)
    group_draws = rng.multinomial(groups, np.full(groups, 1 / groups), size=samples)
    boot = np.einsum("ks,sg,kg->k", seed_draws, numerators, group_draws) / (seeds * (group_draws @ count))
    return {"difference": float(numerators.sum() / (seeds * count.sum())),
            "ci95": np.quantile(boot, [0.025, 0.975]).tolist()}


def study(directory):
    start = time.perf_counter()
    manifest = json.loads((directory / "seal.json").read_text(encoding="utf-8"))
    holdout = directory / "holdout.npz"
    if sha256(holdout) != manifest["holdout_sha256"]:
        raise ValueError("Holdout hash mismatch")
    outputs = [directory / name for name in ("boot.json", "weights.npz", "rounds.npz", "summary.json")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Study outputs already exist")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    histories = {}
    writer_bank = []
    parameters = {}
    source_hashes = {Path(path).name: sha256(Path(path)) for path in (__file__, env.__file__)}
    for seed in TRAIN_SEEDS:
        writers, history = train(seed, BOOTSTRAP_BATCHES, BOOTSTRAP_BATCH, env.MODES)
        writer_bank.append(writers)
        histories[str(seed)] = history
        for mode, writer in writers.items():
            for name, value in writer.state_dict().items():
                parameters[f"{seed}__{mode}__{name}"] = value.detach().numpy()
        print(f"Frozen bootstrap seed {seed} after {time.perf_counter() - start:.1f}s", flush=True)
    boot, weights, rounds, summary = outputs
    boot.write_text(json.dumps({"history_columns": ["reward", "changed", "exact_command", "abstain"],
                                "history": histories, "batches": BOOTSTRAP_BATCHES,
                                "trials_per_batch": BOOTSTRAP_BATCH, "source_sha256": source_hashes,
                                "parameters_per_writer": sum(p.numel() for p in writer.parameters())},
                               indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(weights, **parameters)

    with np.load(holdout) as sealed:
        raw = {key: sealed[key] for key in sealed.files}
    records = [{mode: [] for mode in PANEL} for _ in TRAIN_SEEDS]
    for block in range(len(TEST_GROUPS)):
        trial = trial_from_arrays({key: value[block] for key, value in raw.items()})
        for index, writers in enumerate(writer_bank):
            panel = evaluate(writers, trial, TEST_GROUPS[block] + 7000000)
            for mode, result in panel.items():
                records[index][mode].append(result)
    combined = {mode: {key: np.asarray([[block[key] for block in row[mode]] for row in records])
                       for key in records[0][mode][0]}
                for mode in PANEL}
    np.savez_compressed(rounds, **{f"{mode}__{key}": value for mode, record in combined.items()
                                  for key, value in record.items()})
    disturbed = raw["disturbance"] != env.NOOP
    slot, op = env.decode(raw["disturbance"])
    kinds = {"edge_shift": disturbed & (slot < env.EDGES) & (op < 4),
             "edge_add": disturbed & (op >= 4) & (op < env.REMOVE),
             "edge_remove": disturbed & (op == env.REMOVE),
             "bias_shift": disturbed & (slot >= env.EDGES)}
    differences = {}
    for base in ("veridical", "current_only"):
        for compare in (m for m in PANEL if m != base):
            differences[f"{base}_minus_{compare}"] = {
                key: paired_interval(combined[base][key], combined[compare][key], disturbed)
                for key in ("restored", "benefit", "regret")}
    results = {"metrics": {mode: metrics(record, disturbed, kinds) for mode, record in combined.items()},
               "paired_differences_on_disturbed_trials": differences,
               "counts_per_seed": {"disturbed": int(disturbed.sum()), "sham": int((~disturbed).sum()),
                                   **{name: int(mask.sum()) for name, mask in kinds.items()}},
               "elapsed_seconds": time.perf_counter() - start,
               "platform": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                            "system": platform.platform(), "device": "cpu"},
               "sha256": {path.name: sha256(path) for path in (holdout, boot, weights, rounds)}}
    summary.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("seal", "run"))
    parser.add_argument("--directory", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    (seal if args.phase == "seal" else study)(args.directory)
