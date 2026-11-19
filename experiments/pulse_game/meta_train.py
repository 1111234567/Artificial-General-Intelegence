"""Train and evaluate a recurrent policy using only earned receipts as labels."""

from dataclasses import dataclass, replace
import argparse
import json
from pathlib import Path
import time

import torch
from torch import Tensor
from torch.nn import functional as F

from .meta_env import ACTION_TICKS, TICKS, BatchTape, independent_context_map
from .meta_model import GatedPolicy, ModelConfig, Policy, TemporalPolicy, create_policy, load_policy, policy_state_dict, recurrent_terms


@dataclass
class Rollout:
    # Each column is one life. Bool arrays retain the denominators for paired audit.
    hit: Tensor
    contacted: Tensor
    known: Tensor
    swapped: Tensor
    opposite_context_seen: Tensor
    conflicting_context_seen: Tensor
    educated: Tensor
    loss: Tensor | None = None

    def summary(self) -> dict:
        groups = {
            "overall": torch.ones_like(self.known),
            "first": ~self.known,
            "revisit": self.known,
            "swapped": self.known & self.swapped,
            "other_context_seen": self.known & self.opposite_context_seen,
            "conflict_seen": self.known & self.conflicting_context_seen,
            "educated": self.educated,
        }
        result = {}
        for group, mask in groups.items():
            count, hits, contact = map(int, torch.stack((mask.sum(), (self.hit & mask).sum(), (self.contacted & mask).sum())).tolist())
            result[group] = {"trials": count, "hits": hits, "contacts": contact, "success": hits / count if count else None}
        return result


def rollout(
    policy: Policy | TemporalPolicy | GatedPolicy,
    tape: BatchTape,
    *,
    training: bool = False,
    forced: bool = False,
    explore: float = 0.0,
    seed: int = 1,
    write_enabled: bool = True,
    read_enabled: bool = True,
    flip_receipt: bool = False,
    reset_slow_each_trial: bool = False,
    reset_fast_each_trial: bool = False,
    disable_recurrence: bool = False,
    memory_loss_weight: float = 0.0,
    receipt_loss_weight: float = 0.0,
    gate_loss_weight: float = 0.0,
    cue_ticks: int = TICKS,
    trace_loss_weight: float = 0.0,
    key_separation_weight: float = 0.0,
    write_cost: float = 0.0,
    receipt_dropout: float = 0.0,
    receipt_cutoff_probability: float = 0.0,
    receipt_cutoff_trial: int = 12,
    random_cue_ticks: tuple[int, ...] = (),
    reward_objective: bool = False,
    value_objective: bool = False,
    contact_bonus: float = 0.25,
    entropy_weight: float = 0.015,
) -> Rollout:
    """`forced` uses a random constant L/R script independent of each hidden map.

    All other runs choose pulses from the policy. Loss labels come from prior actual
    contacts in the same life. The seen tables are optimizer/audit bookkeeping and
    are never passed to the policy or used to select an action.
    """
    if not training and (forced or explore or random_cue_ticks):
        raise ValueError("Evaluation requires an unforced deterministic policy rollout")
    if (reward_objective or value_objective) and forced:
        raise ValueError("The reward objective must sample its own actions")
    if reward_objective and value_objective:
        raise ValueError("Choose one outer objective")
    if not 1 <= cue_ticks <= TICKS or any(not 1 <= ticks <= TICKS for ticks in random_cue_ticks):
        raise ValueError("cue_ticks must be within the trial")
    if not 0 <= receipt_dropout <= 1 or not 0 <= receipt_cutoff_probability <= 1 or receipt_cutoff_trial < 0:
        raise ValueError("Receipt mask probabilities must be 0..1 and cutoff trial must be nonnegative")
    temporal = policy.config.memory_kind == "temporal"
    explicit = policy.config.memory_kind == "explicit"
    batch, trials = tape.ctx.shape
    device = tape.ctx.device
    rows = torch.arange(batch, device=device)
    seen = torch.full((batch, 8), -1, device=device, dtype=torch.long)
    previous_layout = torch.full_like(seen, -1)
    state = policy.initial_state(batch)
    recurrent = recurrent_terms(policy, not disable_recurrence)
    results = []
    all_margins, all_labels, all_masks = [], [], []
    memory_logits, known_preferences, memory_masks = [], [], []
    receipt_logits, revealed_preferences, receipt_masks = [], [], []
    gate_logits, gate_targets = [], []
    trace_start, trace_slots, trace_drift, drift_masks = [], [], [], []
    write_gains = []
    returns, taken, entropies = [], [], []
    action_values, value_masks = [], []
    has_dropouts = bool(receipt_dropout or receipt_cutoff_probability)
    visible_receipts = torch.ones((trials, batch), device=device, dtype=torch.bool)
    cue_exposure = None
    if training:
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn((trials, batch), generator=generator, device=device) * explore
        forced_pulses = 2 * torch.randint(2, (trials, batch), generator=generator, device=device) - 1
        if receipt_dropout:
            visible_receipts &= torch.rand((trials, batch), device=device, generator=generator) >= receipt_dropout
        if receipt_cutoff_probability:
            always_visible = torch.rand((batch,), device=device, generator=generator) >= receipt_cutoff_probability
            visible_receipts[receipt_cutoff_trial:] &= always_visible
        if random_cue_ticks:
            options = torch.tensor(random_cue_ticks, device=device)
            choice = torch.randint(len(random_cue_ticks), (trials, batch), device=device, generator=generator)
            cue_exposure = options[choice]
        if reward_objective:
            action_draws = torch.rand((trials, ACTION_TICKS, batch), generator=generator, device=device)

    for index in range(trials):
        if reset_slow_each_trial:
            state = replace(state, activity=torch.zeros_like(state.activity))
        if reset_fast_each_trial:
            state = replace(state, fast=torch.zeros_like(state.fast))
        trial = tape.trial(index)
        token = trial.ctx * 4 + trial.cue
        known_good = seen[rows, token]
        known = known_good >= 0
        swapped = (previous_layout[rows, token] != trial.left_id) & known
        opposite_good = seen[rows, (1 - trial.ctx) * 4 + trial.cue]
        opposite_seen = opposite_good >= 0
        conflict = opposite_seen & (opposite_good != known_good)
        educated = previous_layout[rows, token] >= 0
        # The current supervised label requires a prior receipt.
        desired = (known_good != trial.left_id).to(policy.recurrent.dtype)
        static = trial.static_observation(device, policy.recurrent.dtype)
        key = policy.encode_key(static[:, :6]) if explicit else None
        blank = torch.zeros_like(static[:, :6])
        hidden_key = policy.encode_key(blank) if explicit and (cue_ticks < TICKS or cue_exposure is not None) else None
        vehicle = trial.initial()
        trial_logp, trial_entropy = [], []
        trial_value, trial_value_mask = [], []
        for tick in range(TICKS):
            obs = vehicle.observe(trial, static, tick)
            step_key = key
            if cue_exposure is not None:
                visible = tick < cue_exposure[index, :, None]
                obs = torch.cat((torch.where(visible, obs[:, :6], blank), obs[:, 6:]), dim=1)
                if explicit:
                    step_key = torch.where(visible, key, hidden_key)
            elif tick >= cue_ticks:
                obs = torch.cat((blank, obs[:, 6:]), dim=1)
                step_key = hidden_key
            if has_dropouts:
                visible_reward = obs[:, 13:15] * visible_receipts[index, :, None]
                obs = torch.cat((obs[:, :13], visible_reward, obs[:, 15:]), dim=1)
            if flip_receipt:
                obs = torch.cat((obs[:, :13], obs[:, [14, 13]], obs[:, 15:]), dim=1)
            logits, state = policy(
                obs, state, key=step_key, recurrent=recurrent,
                write_enabled=write_enabled, read_enabled=read_enabled,
            )
            if training and memory_loss_weight and tick == 0:
                memory_logits.append(policy.decode_preference(state, key))
                known_preferences.append(known_good.clamp_min(0).to(logits.dtype))
                memory_masks.append(known)
            if training and temporal and (gate_loss_weight or receipt_loss_weight):
                received = (obs[:, 13:15].sum(dim=1) > 0) & (obs[:, 15:17].sum(dim=1) > 0)
                if gate_loss_weight:
                    gate_logits.append(state.association.gates)
                    gate_targets.append(torch.stack((obs[:, 17] > 0, received), dim=1))
                if receipt_loss_weight:
                    receipt_logits.append(policy.preference_probe(state.association.value).squeeze(1))
                    revealed = ((obs[:, 13] > 0) & (obs[:, 16] > 0)) | ((obs[:, 14] > 0) & (obs[:, 15] > 0))
                    revealed_preferences.append(revealed.to(logits.dtype))
                    receipt_masks.append(received)
            if training and (trace_loss_weight or key_separation_weight):
                normalized_trace = F.normalize(state.trace, dim=1, eps=1e-6)
                if tick == 0:
                    opening_address = normalized_trace
                    trace_start.append(opening_address)
                    trace_slots.append(token)
                if tick > 0:
                    trace_drift.append(1 - (opening_address * normalized_trace).sum(dim=1))
                    drift_masks.append((obs[:, 13:15].sum(dim=1) > 0).to(logits.dtype))
            if training and write_cost:
                write_gains.append((0.25 * state.association.gates[:, 1] + 0.5).clamp(0, 1).mean())
            if tick >= ACTION_TICKS:
                continue
            margin = logits[:, 1] - logits[:, 0]
            if training and reward_objective:
                right = action_draws[index, tick] < margin.detach().sigmoid()
            elif training:
                right = margin.detach() + noise[index] > 0
            else:
                right = margin > 0
            pulse = forced_pulses[index] if forced else 2 * right.to(torch.long) - 1
            if training and reward_objective:
                trial_logp.append(F.logsigmoid(margin * pulse) * vehicle.active)
                entropy = F.softplus(margin) - margin * margin.sigmoid()
                trial_entropy.append(entropy * vehicle.active)
            elif training and value_objective:
                trial_value.append(torch.where(pulse > 0, logits[:, 1], logits[:, 0]))
                trial_value_mask.append(vehicle.active)
            elif training:
                if tick == 0:
                    exploratory_label = (pulse > 0).to(desired.dtype)
                all_margins.append(margin)
                all_labels.append(torch.where(known, desired, exploratory_label))
                # Encourage the exploration action to reach a target, using no map label.
                # The fixed script supplies observations only, so skip it as a loss label.
                novelty_weight = 0.0 if forced else 0.12
                all_masks.append(vehicle.active * torch.where(known, 1.0, novelty_weight))
            vehicle = vehicle.step(trial, pulse, tick)

        contacted = vehicle.last_contact >= 0
        receipt_visible = contacted & visible_receipts[index]
        hit = contacted & (vehicle.last_contact == trial.good_id)
        results.append(torch.stack((hit, contacted, known, swapped, opposite_seen, conflict, educated)))
        if training and (reward_objective or value_objective):
            event_score = torch.where(hit, 1.0, -1.0) * receipt_visible
            returns.append(event_score + contact_bonus * contacted)
        if training and reward_objective:
            taken.append(torch.stack(trial_logp).sum(dim=0))
            entropies.append(torch.stack(trial_entropy).sum(dim=0))
        if training and value_objective:
            action_values.append(torch.stack(trial_value))
            value_masks.append(torch.stack(trial_value_mask))
        # Both good and bad real contacts identify the public two-target preference.
        revealed_good = torch.where(hit, vehicle.last_contact, 1 - vehicle.last_contact)
        if training and receipt_loss_weight and not temporal:
            # This is a training-only probe of the four actually observed receipt channels.
            sensed = torch.stack((hit, contacted & ~hit, vehicle.last_contact == 0, vehicle.last_contact == 1), dim=1)
            learned_value = policy.receipt(sensed.to(policy.recurrent.dtype))
            receipt_logits.append(policy.preference_probe(learned_value).squeeze(1))
            revealed_preferences.append(revealed_good.clamp(0, 1).to(logits.dtype))
            receipt_masks.append(receipt_visible)
        seen[rows, token] = torch.where(receipt_visible, revealed_good, known_good)
        previous_layout[rows, token] = torch.where(receipt_visible, trial.left_id, previous_layout[rows, token])

    loss = None
    if training:
        if reward_objective:
            score = torch.stack(returns)
            advantage = score - score.mean()
            loss = (-(advantage * torch.stack(taken)).mean() - entropy_weight * torch.stack(entropies).mean()) / ACTION_TICKS
        elif value_objective:
            predictions = torch.stack(action_values)
            score = torch.stack(returns)[:, None, :].expand_as(predictions)
            mask = torch.stack(value_masks)
            error = F.smooth_l1_loss(predictions, score, reduction="none")
            loss = (mask * error).sum() / mask.sum().clamp_min(1)
        else:
            loss = weighted_bce(all_margins, all_labels, all_masks)
        if memory_loss_weight:
            loss = loss + memory_loss_weight * weighted_bce(memory_logits, known_preferences, memory_masks)
        if receipt_loss_weight:
            loss = loss + receipt_loss_weight * weighted_bce(receipt_logits, revealed_preferences, receipt_masks)
        if gate_loss_weight:
            loss = loss + gate_loss_weight * balanced_gate_bce(gate_logits, gate_targets)
        if trace_loss_weight or key_separation_weight:
            stable, spaced = address_consistency(trace_start, trace_slots, trace_drift, drift_masks)
            loss = loss + trace_loss_weight * stable + key_separation_weight * spaced
        if write_cost:
            loss = loss + write_cost * torch.stack(write_gains).mean()
    return Rollout(*torch.stack(results, dim=1).unbind(), loss=loss)


def weighted_bce(logits, labels, masks):
    mask = torch.stack(masks)
    loss = F.binary_cross_entropy_with_logits(torch.stack(logits), torch.stack(labels), reduction="none")
    return (mask * loss).sum() / mask.sum().clamp_min(1)


def balanced_gate_bce(logits, targets):
    logits = torch.stack(logits)
    positive = torch.stack(targets)
    negative = ~positive
    loss = F.binary_cross_entropy_with_logits(logits, positive.to(logits.dtype), reduction="none")
    hit_loss = (positive * loss).sum(dim=(0, 1)) / positive.sum(dim=(0, 1)).clamp_min(1)
    idle_loss = (negative * loss).sum(dim=(0, 1)) / negative.sum(dim=(0, 1)).clamp_min(1)
    return 0.5 * (hit_loss + idle_loss).mean()


def address_consistency(starts, slots, drift, masks):
    keys = torch.cat(starts)
    tokens = torch.cat(slots)
    groups = F.one_hot(tokens, 8).to(keys.dtype)
    reference = F.normalize(groups.T @ keys, dim=1, eps=1e-6)
    consistency = (1 - (keys * reference[tokens]).sum(dim=1)).mean()
    off_diagonal = ~torch.eye(8, device=keys.device, dtype=torch.bool)
    separation = ((reference @ reference.T).square() * off_diagonal).sum() / 56
    mask = torch.stack(masks)
    retained = (mask * torch.stack(drift)).sum() / mask.sum().clamp_min(1)
    return consistency + retained, separation


def make_batch(base: int, lives: int, trials: int, device, *, independent_contexts=False,
               distance_jitter=0.0, wind_scale=0.0) -> BatchTape:
    from . import world

    tape = world.make_tape(range(base, base + lives), trials, distance_jitter=distance_jitter, wind_scale=wind_scale)
    if independent_contexts:
        tape = independent_context_map(tape)
    return BatchTape.from_tape(tape, device)


def save_checkpoint(path: Path, policy: Policy, optimizer, step: int, base: int, best_success: float):
    temporary = path.with_suffix(".tmp")
    torch.save({**policy_state_dict(policy), "optimizer": optimizer.state_dict(), "step": step,
                "seed_base": base, "dev_best_success": best_success}, temporary)
    temporary.replace(path)


@torch.no_grad()
def evaluate(policy, args, base: int):
    model_device = next(policy.parameters()).device
    tape = make_batch(base, args.eval_lives, args.eval_trials, model_device,
                      independent_contexts=args.independent_contexts, distance_jitter=args.distance_jitter,
                      wind_scale=args.wind_scale)
    return rollout(policy, tape, cue_ticks=args.eval_cue_ticks or args.cue_ticks).summary()


def train(args):
    if args.seed >= 1_000_000:
        raise ValueError("Use a seed below 1,000,000 to keep training and development namespaces disjoint")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    latest = directory / "last.pt"
    settings_path = directory / "settings.jsonl"
    if latest.exists() and not args.resume:
        raise ValueError(f"Use --resume to continue an existing run: {latest}")
    if args.resume and args.initialize:
        raise ValueError("Choose either a resume or an initialization checkpoint")
    if (args.reward_objective or args.value_objective) and args.forced_steps:
        raise ValueError("The reward objective requires --forced-steps 0")
    if args.reward_objective and args.value_objective:
        raise ValueError("Choose only one of --reward-objective and --value-objective")
    fast_enabled = not args.slow_only and args.memory_kind != "gated"
    if not fast_enabled and args.receipt_loss:
        raise ValueError("The fast-value receipt probe requires fast synapses")
    if (args.gate_loss or args.trace_loss or args.key_separation or args.write_cost) and args.memory_kind != "temporal":
        raise ValueError("Gate and trace training require the temporal policy")
    config = ModelConfig(hidden=args.hidden, degree=args.degree, keys=args.keys, values=args.values,
                         fast_enabled=fast_enabled, aux_probe=bool(args.memory_loss or args.receipt_loss),
                         memory_kind=args.memory_kind, association_hidden=args.association_hidden)
    start = 0
    dev_best = -1.0
    base = 10_000_000_000 + args.seed * 100_000_000
    if args.lives * args.steps > 100_000_000:
        raise ValueError("Training life namespace would exceed its reserved interval")
    if args.resume:
        saved = torch.load(latest, map_location=device, weights_only=True)
        if base != saved["seed_base"]:
            raise ValueError("Resume requires the original training seed")
        if config != ModelConfig(**saved["config"]):
            raise ValueError("Resume requires the original model configuration")
        with settings_path.open(encoding="utf-8") as log:
            prior = json.loads(log.readline())
        if prior["lives"] != args.lives or prior["trials"] != args.trials:
            raise ValueError("Resume requires the original batch dimensions to preserve life seeds")
        start = saved["step"]
        dev_best = saved["dev_best_success"]
        policy = load_policy(latest, device)
    elif args.initialize:
        policy = load_policy(args.initialize, device)
        if policy.config != config:
            raise ValueError("Initialization checkpoint uses a different model configuration")
    else:
        policy = create_policy(config).to(device)
    policy.train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.resume:
        optimizer.load_state_dict(saved["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr

    settings = vars(args).copy()
    settings.update({"steps_completed_at_start": start, "train_seed_base": base, "dev_seed_base": args.dev_seed})
    with settings_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(settings) + "\n")
    elapsed = time.perf_counter()
    for step in range(start + 1, args.steps + 1):
        if (directory / "STOP").exists():
            break
        trial_base = base + (step - 1) * args.lives
        batch = make_batch(trial_base, args.lives, args.trials, device, independent_contexts=args.independent_contexts,
                           distance_jitter=args.distance_jitter, wind_scale=args.wind_scale)
        forced = step <= args.forced_steps
        optimizer.zero_grad(set_to_none=True)
        run = rollout(policy, batch, training=True, forced=forced, explore=args.explore, seed=trial_base,
                      memory_loss_weight=args.memory_loss, receipt_loss_weight=args.receipt_loss,
                      gate_loss_weight=args.gate_loss, cue_ticks=args.cue_ticks, trace_loss_weight=args.trace_loss,
                      key_separation_weight=args.key_separation,
                      write_cost=args.write_cost, receipt_dropout=args.receipt_dropout,
                      receipt_cutoff_probability=args.receipt_cutoff_probability, receipt_cutoff_trial=args.receipt_cutoff_trial,
                      random_cue_ticks=args.random_cue_ticks, reward_objective=args.reward_objective,
                      value_objective=args.value_objective,
                      contact_bonus=args.contact_bonus, entropy_weight=args.entropy_weight)
        run.loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 3.0)
        optimizer.step()

        checkpoint = (step % args.save_every == 0 or step == args.steps or step == args.forced_steps
                      or (directory / "STOP").exists())
        evaluate_now = step % args.eval_every == 0 or step == args.steps or step == args.forced_steps
        if evaluate_now:
            report = evaluate(policy, args, args.dev_seed)
            metric = report["revisit"]["success"]
            if metric is not None and metric > dev_best:
                dev_best = metric
                save_checkpoint(directory / "best.pt", policy, optimizer, step, base, dev_best)
        if checkpoint or evaluate_now or step == start + 1:
            save_checkpoint(latest, policy, optimizer, step, base, dev_best)
        if checkpoint or evaluate_now or step == start + 1:
            train_summary = run.summary()
            record = {
                "step": step, "training_history": ("policy_reinforce" if args.reward_objective else
                                                  "policy_fitted_value" if args.value_objective else
                                                  "forced_random" if forced else "policy_exploration"),
                "outer_loss": float(run.loss.detach()), "elapsed_s": round(time.perf_counter() - elapsed, 3),
                "train": train_summary, "dev_policy": report if evaluate_now else None,
            }
            with (directory / "history.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps({"step": step, "time_s": record["elapsed_s"], "stage": record["training_history"],
                              "loss": record["outer_loss"], "train_revisit": train_summary["revisit"]["success"],
                              "dev_revisit": report["revisit"]["success"] if evaluate_now else None}), flush=True)


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", required=True)
    cli.add_argument("--device", default="cuda")
    cli.add_argument("--cpu-threads", type=int, default=2)
    cli.add_argument("--steps", type=int, default=600)
    cli.add_argument("--lives", type=int, default=96)
    cli.add_argument("--trials", type=int, default=24)
    cli.add_argument("--forced-steps", type=int, default=64)
    cli.add_argument("--explore", type=float, default=0.3)
    cli.add_argument("--lr", type=float, default=0.003)
    cli.add_argument("--seed", type=int, default=1119)
    cli.add_argument("--dev-seed", type=int, default=1_119_240_000)
    cli.add_argument("--eval-every", type=int, default=20)
    cli.add_argument("--eval-lives", type=int, default=128)
    cli.add_argument("--eval-trials", type=int, default=64)
    cli.add_argument("--save-every", type=int, default=10)
    cli.add_argument("--hidden", type=int, default=32)
    cli.add_argument("--degree", type=int, default=6)
    cli.add_argument("--keys", type=int, default=16)
    cli.add_argument("--values", type=int, default=8)
    cli.add_argument("--association-hidden", type=int, default=0)
    cli.add_argument("--memory-loss", type=float, default=0.0)
    cli.add_argument("--receipt-loss", type=float, default=0.0)
    cli.add_argument("--gate-loss", type=float, default=0.0)
    cli.add_argument("--trace-loss", type=float, default=0.0)
    cli.add_argument("--key-separation", type=float, default=0.0)
    cli.add_argument("--write-cost", type=float, default=0.0)
    cli.add_argument("--memory-kind", choices=("explicit", "temporal", "gated"), default="explicit")
    cli.add_argument("--cue-ticks", type=int, default=TICKS)
    cli.add_argument("--random-cue-ticks", type=int, nargs="+", default=())
    cli.add_argument("--eval-cue-ticks", type=int)
    cli.add_argument("--independent-contexts", action="store_true")
    cli.add_argument("--distance-jitter", type=float, default=0.0)
    cli.add_argument("--wind-scale", type=float, default=0.0)
    cli.add_argument("--receipt-dropout", type=float, default=0.0)
    cli.add_argument("--receipt-cutoff-probability", type=float, default=0.0)
    cli.add_argument("--receipt-cutoff-trial", type=int, default=12)
    cli.add_argument("--reward-objective", action="store_true")
    cli.add_argument("--value-objective", action="store_true")
    cli.add_argument("--contact-bonus", type=float, default=0.25)
    cli.add_argument("--entropy-weight", type=float, default=0.015)
    cli.add_argument("--slow-only", action="store_true")
    cli.add_argument("--initialize", help="Start a new run from model weights only")
    cli.add_argument("--resume", action="store_true")
    return cli


if __name__ == "__main__":
    train(parser().parse_args())
