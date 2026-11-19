"""Train a learned-address Scout on closed teacher and student histories."""

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from . import world
from .meta_scout_env import map_parity, randomly_change_one_context_rule, scaled_geometry, varied_sensitive_count
from .meta_scout_model import PULSES, ScoutMetaPolicy, ScoutModelConfig, ScoutRawAgent, ScoutState, load_scout_model, model_payload
from .scout_world import SCOUT_HIT, SCOUT_OBS_SIZE, SCOUT_SIDE, ScoutHistoryPolicy, ScoutWorld, make_scout_tape, rollout_scout


@dataclass
class Examples:
    obs: np.ndarray
    targets: np.ndarray
    actions: np.ndarray
    metrics: dict
    action_bias: np.ndarray | None = None


@dataclass
class Targets:
    context: Tensor
    had_context: Tensor
    cue: Tensor
    known: Tensor
    context_map: Tensor
    receipt: Tensor
    inferred: Tensor


def make_batch(base, lives, trials, *, parity=None, switch_probability=0.0, sensitive_count=None):
    tape = make_scout_tape(range(base, base + lives), trials, distance_jitter=0.22, wind_scale=0.12)
    if sensitive_count is not None:
        tape = varied_sensitive_count(tape, sensitive_count)
    if parity is not None:
        tape = map_parity(tape, parity)
    if switch_probability:
        tape = randomly_change_one_context_rule(tape, switch_probability)
    return world.swap_sides(tape, first_trial=trials // 2)


def action_ids(pulses):
    left, right = pulses[..., 0], pulses[..., 1]
    return np.where(left == right, 0, np.where(left, 1, 2))


def metrics(result, tape):
    sensitive = result.sensitive_trial
    different_destination = result.scout_side != (tape.left_id != tape.good_id)
    must_reverse = sensitive & different_destination
    direct_opportunity = ~sensitive & different_destination
    late = np.arange(tape.trials)[None, :] >= tape.trials - min(16, tape.trials)

    def count(condition, selection):
        return {"hits": int((condition & selection).sum()), "trials": int(selection.sum())}

    return {
        "rewarded": int(result.correct.sum()), "trials": int(result.correct.size), "accuracy": result.accuracy,
        "contact": result.contact_rate, "crossed": result.probe_rate,
        "probed_then_opposite": result.reverse_rate,
        "late_accuracy": float(result.correct[:, -16:].mean()),
        "sensitive_late_accuracy": count(result.correct, sensitive & late),
        "insensitive_late_accuracy": count(result.correct, ~sensitive & late),
        "sensitive_probed_late": count(result.probed, sensitive & late),
        "insensitive_probed_late": count(result.probed, ~sensitive & late),
        "must_reverse": count(result.reversed & result.correct, must_reverse),
        "must_reverse_late": count(result.reversed & result.correct, must_reverse & late),
        "correct_direct_late": count(~result.probed & result.correct, direct_opportunity & late),
        "mean_ticks": float(np.where(result.timed_out, world.MOVEMENT_TICKS, result.latency_ticks).mean()),
    }


def collect(tape, *, seed, student=None, teacher_mix=0.0, temperature=0.0, teacher_targets=True,
            receipt_dropout=0.0, receipt_cutoff_probability=0.0, receipt_cutoff_trial=18, coherent=False,
            post_scout_noise=0.0):
    """Teacher targets see the same public stream that led to the recorded action."""
    if not 0 <= receipt_dropout <= 1 or not 0 <= receipt_cutoff_probability <= 1 or receipt_cutoff_trial < 0:
        raise ValueError("Receipt mask probabilities must be 0..1 and cutoff trial must be nonnegative")
    dropout = bool(receipt_dropout or receipt_cutoff_probability)
    teacher = ScoutHistoryPolicy(seed) if teacher_targets else None
    if student is None and teacher is None:
        raise ValueError("A reward rollout requires an acting student")
    if student is None and not dropout:
        history = rollout_scout(teacher, tape, record=True)
        actions = action_ids(history.pulses)
        return Examples(history.obs, actions, actions, metrics(history, tape))

    rng = np.random.default_rng(seed ^ 0x5C3410)
    use_teacher = rng.random((tape.n, tape.trials)) < teacher_mix
    stickiness = rng.gumbel(size=(tape.n, tape.trials, 3)) if temperature and coherent else None
    keep_receipt = rng.random((tape.n, tape.trials)) >= receipt_dropout if dropout else None
    if dropout and receipt_cutoff_probability:
        reveal_through_life = rng.random(tape.n) >= receipt_cutoff_probability
        keep_receipt[:, receipt_cutoff_trial:] &= reveal_through_life[:, None]
    env = ScoutWorld(tape, record=True)
    if teacher is not None:
        teacher.reset(tape.n)
    if student is not None:
        student.reset(tape.n)
    recommendations = np.empty((tape.n, tape.trials, world.TICKS), dtype=np.int64)
    seen_observations = np.empty((*recommendations.shape, SCOUT_OBS_SIZE), dtype=np.float32) if dropout else None
    action_bias = None
    if post_scout_noise:
        sample = np.random.default_rng(seed ^ 0x7A2E8838).normal(0, post_scout_noise, (tape.n, tape.trials)).astype(np.float32)
        trial_bias = np.stack((np.zeros_like(sample), sample, -sample), axis=2)
        action_bias = np.empty((*recommendations.shape, 3), dtype=np.float32)
        post_probe = np.zeros(tape.n, dtype=bool)
    obs = env.observe()
    while obs is not None:
        trial, tick = env.trial_index, env.tick_index
        if dropout:
            obs = obs.copy()
            obs[~keep_receipt[:, trial], world.REWARD_POS:world.REWARD_NEG + 1] = 0
            seen_observations[:, trial, tick] = obs
        if action_bias is not None:
            if tick == 0:
                post_probe.fill(False)
            post_probe |= obs[:, SCOUT_HIT] > 0
            bias = trial_bias[:, trial] * post_probe[:, None]
            action_bias[:, trial, tick] = bias
        if student is None:
            student_action = None
        elif temperature:
            logits = student.logits(obs)
            if stickiness is not None:
                selected = (logits.cpu().numpy() / temperature + stickiness[:, trial]).argmax(axis=1)
            else:
                scores = logits / temperature
                if action_bias is not None:
                    scores = scores + torch.as_tensor(bias, device=logits.device)
                probabilities = scores.softmax(dim=1).cpu().numpy()
                selected = (rng.random(tape.n)[:, None] > np.cumsum(probabilities[:, :2], axis=1)).sum(axis=1)
            student_action = PULSES[selected]
        else:
            student_action = student.act(obs)
        if teacher is None:
            pulse = student_action
            recommendations[:, trial, tick] = action_ids(student_action)
        else:
            teacher_action = teacher.act(obs)
            recommendations[:, trial, tick] = action_ids(teacher_action)
            pulse = teacher_action if student is None else np.where(use_teacher[:, trial, None], teacher_action, student_action)
        obs = env.step(pulse)
    history = env.result()
    return Examples(seen_observations if dropout else history.obs, recommendations,
                    action_ids(history.pulses), metrics(history, tape), action_bias)


def targets_from_public(obs: np.ndarray, device):
    """Recover only labels identifiable from already delivered public observations."""
    batch, trials, ticks, _ = obs.shape
    seen = np.full((batch, 4, 2), -1, dtype=np.int8)
    prefix = np.empty((batch, trials, 2), dtype=np.int8)
    rows = np.arange(batch)
    cue = obs[:, :, 0, world.CUE].argmax(axis=-1)
    visible = obs[..., world.CTX].sum(axis=-1) == 1
    c = (1 + obs[..., world.CTX].argmax(axis=-1)) * visible
    context = np.maximum.accumulate(c, axis=2)
    contacted = obs[..., world.CONTACT_ID].sum(axis=-1) == 1
    rewarded = obs[..., world.REWARD_POS] > 0
    arrived = contacted & (rewarded | (obs[..., world.REWARD_NEG] > 0))
    target = obs[..., world.CONTACT_ID].argmax(axis=-1)
    inferred = np.where(rewarded, target, 1 - target)

    for trial in range(trials):
        index = cue[:, trial]
        prefix[:, trial] = seen[rows, index]
        observed_context_and_receipt = arrived[:, trial] & (context[:, trial] > 0)
        r, at = np.nonzero(observed_context_and_receipt)
        seen[r, index[r], context[r, trial, at] - 1] = inferred[r, trial, at]

    def on_device(array):
        return torch.as_tensor(array, device=device)

    return Targets(on_device(context), on_device(visible), on_device(cue), on_device(prefix >= 0),
                   on_device(prefix.clip(0, 1)).float(), on_device(arrived), on_device(inferred).float())


def masked_mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


def binary_loss(prediction, supervision, mask):
    loss = F.binary_cross_entropy_with_logits(prediction, supervision, reduction="none")
    return masked_mean(loss, mask)


def sequence(model, obs, fork_targets=None):
    """Run one life per row with gradients through activity, key trace and fast weights."""
    batch, trials, ticks, width = obs.shape
    inputs = obs.reshape(batch, trials * ticks, width).transpose(0, 1)
    drive = model.drive(inputs)
    update = model.update(inputs).sigmoid()
    state = model.initial(batch)
    weights = model.prepared()
    logits, reads, payloads, gates, addresses, states = [], [], [], [], [], []
    forks, fork_labels = [], []
    for time, (x, d, u) in enumerate(zip(inputs.unbind(), drive.unbind(), update.unbind())):
        if fork_targets is not None:
            at = fork_targets[:, time] >= 0
            if torch.any(at):
                observed = x[at]
                alternative = torch.cat((observed[:, [1, 0]], observed[:, 2:]), dim=1)
                before = ScoutState(state.activity[at], state.fast[at], state.trace[at])
                forks.append(model(alternative, before, weights).logits)
                fork_labels.append(fork_targets[at, time])
        step = model(x, state, weights, (d, u))
        state = step.state
        logits.append(step.logits)
        reads.append(step.read)
        payloads.append(step.payload)
        gates.append(step.gate_logits)
        addresses.append(step.key)
        states.append(state.activity)

    def batch_major(items):
        return torch.stack(items, dim=1).reshape(batch, trials, ticks, -1)

    outputs = [batch_major(item) for item in (logits, reads, payloads, gates, addresses, states)]
    if fork_targets is not None:
        counterfactual = F.cross_entropy(torch.cat(forks), torch.cat(fork_labels)) if forks else obs.new_zeros(())
        outputs.append(counterfactual)
    return outputs


def self_labels(obs, labels):
    """Outer motor targets from a life the student already physically executed."""
    chosen_context = (labels.context - 1).clamp_min(0)[..., None]
    prefix_known = labels.known[:, :, None, :].expand(-1, -1, world.TICKS, -1)
    prefix_map = labels.context_map[:, :, None, :].expand_as(prefix_known)
    known_context = prefix_known.gather(-1, chosen_context).squeeze(-1) & (labels.context > 0)
    left = obs[..., world.LEFT_ID].argmax(dim=-1)
    goal = prefix_map.gather(-1, chosen_context).squeeze(-1)
    target_direction = 1 + (goal != left).long()
    agree = labels.known.all(dim=2) & (labels.context_map[:, :, 0] == labels.context_map[:, :, 1])
    direct = 1 + (labels.context_map[:, :, 0, None] != left).long()
    toward_beacon = 1 + obs[..., SCOUT_SIDE].argmax(dim=-1)
    before_context = labels.context == 0
    preliminary = torch.where(agree[..., None], direct, toward_beacon)

    own_contact = obs[..., world.CONTACT_ID].sum(dim=2)
    endpoint_seen = own_contact.sum(dim=-1) > 0
    exploration_direction = 1 + (own_contact.argmax(dim=-1)[..., None] != left).long()
    after_context = torch.where(known_context, target_direction, exploration_direction)
    active = obs[..., world.ACTIVE] > 0
    output = torch.where(active, torch.where(before_context, preliminary, after_context), 0)
    weight = torch.where(before_context | known_context, 1.0, 0.2 * endpoint_seen[..., None])
    return output, torch.where(active, weight, 0.08)


def learning_objective(model, batch, device, args):
    obs = torch.as_tensor(batch.obs, device=device)
    actions = torch.as_tensor(batch.targets, device=device)
    labels = targets_from_public(batch.obs, device)
    fork_targets = None
    if args.counterfactual_context_loss:
        opposite = (2 - labels.context).clamp(0, 1)[..., None]
        past_map = labels.context_map[:, :, None, :].expand(-1, -1, world.TICKS, -1)
        alternate_id = past_map.gather(-1, opposite).squeeze(-1)
        both_known = labels.known.all(dim=-1)[..., None]
        seen_now = labels.had_context & (obs[..., SCOUT_HIT] > 0) & (obs[..., world.ACTIVE] > 0)
        left = obs[..., world.LEFT_ID].argmax(dim=-1)
        fork_targets = torch.where(seen_now & both_known, 1 + (alternate_id != left).long(), -1).flatten(1)
    logits, memory, payload, gates, keys, states, *counterfactual = sequence(model, obs, fork_targets)
    active = obs[..., world.ACTIVE]
    reverse_weight = 1 + 0.5 * (labels.context > 0)
    decision = (obs[..., world.TRIAL_START] > 0) | (obs[..., SCOUT_HIT] > 0)
    decision_weight = 1 + args.decision_weight * decision
    if args.self_labeled:
        actions, action_mask = self_labels(obs, labels)
    else:
        action_mask = 0.08 + 0.92 * active
    action_loss = F.cross_entropy(logits.flatten(0, 2), actions.flatten(), reduction="none").reshape_as(active)
    total = masked_mean(action_loss, action_mask * reverse_weight * decision_weight)
    terms = {"action": total.detach()}
    if counterfactual:
        fork_loss = counterfactual[0]
        total = total + args.counterfactual_context_loss * fork_loss
        terms["counterfactual_context"] = fork_loss.detach()

    if args.context_loss:
        contexts = model.context_probe(states)
        loss = F.cross_entropy(contexts.flatten(0, 2), labels.context.flatten(), reduction="none").reshape_as(active)
        context_loss = masked_mean(loss, 0.1 + 0.9 * active)
        total = total + args.context_loss * context_loss
        terms["context"] = context_loss.detach()

    if model.config.fast_enabled:
        # Supervision is supplied only here. The readout-motor variant reuses predictions, never labels.
        if args.memory_loss:
            selected = memory[:, :, [0, 4, 8], :]
            prefer = model.preference_probe(selected)
            known = labels.known[:, :, None, :].expand_as(prefer)
            values = labels.context_map[:, :, None, :].expand_as(prefer)
            map_loss = binary_loss(prefer, values, known)
            seen_loss = F.binary_cross_entropy_with_logits(model.seen_probe(selected), known.to(prefer.dtype))
            total = total + args.memory_loss * (map_loss + 0.25 * seen_loss)
            terms.update(memory=map_loss.detach(), seen=seen_loss.detach())
        if args.receipt_loss:
            info = model.preference_probe(payload)
            chosen = info.gather(-1, (labels.context - 1).clamp_min(0)[..., None]).squeeze(-1)
            payload_loss = binary_loss(chosen, labels.inferred, labels.receipt & (labels.context > 0))
            total = total + args.receipt_loss * payload_loss
            terms["receipt"] = payload_loss.detach()
        if args.gate_loss:
            scores = gates[..., 1]
            loss = F.binary_cross_entropy_with_logits(scores, labels.receipt.to(scores.dtype), reduction="none")
            gate_loss = 0.5 * (masked_mean(loss, labels.receipt) + masked_mean(loss, ~labels.receipt))
            total = total + args.gate_loss * gate_loss
            terms["gate"] = gate_loss.detach()
        if args.write_cost:
            gain = (0.25 * gates[..., 1] + 0.5).clamp(0, 1).mean()
            total = total + args.write_cost * gain
            terms["write_gain"] = gain.detach()
        if args.key_loss:
            initial_key = keys[:, :, 0, :]
            tokens = F.one_hot(labels.cue, 4).to(keys.dtype).flatten(0, 1)
            prototypes = F.normalize(tokens.T @ initial_key.flatten(0, 1), dim=1, eps=1e-6)
            targets = prototypes[labels.cue][:, :, None, :]
            consistency = 1 - (keys * targets).sum(dim=-1)
            fit = consistency[:, :, 0].mean() + masked_mean(consistency, labels.receipt)
            distinguishable = (prototypes @ prototypes.T).square()
            overlap = (distinguishable.sum() - distinguishable.diag().sum()) / 12
            key_loss = fit + 0.5 * overlap
            total = total + args.key_loss * key_loss
            terms["address"] = key_loss.detach()
    if args.observed_goal_loss:
        chosen_context = (labels.context - 1).clamp_min(0)[..., None]
        prefix_known = labels.known[:, :, None, :].expand(-1, -1, world.TICKS, -1)
        prefix_map = labels.context_map[:, :, None, :].expand_as(prefix_known)
        known_context = prefix_known.gather(-1, chosen_context).squeeze(-1) & (labels.context > 0)
        goal = prefix_map.gather(-1, chosen_context).squeeze(-1)
        left = obs[..., world.LEFT_ID].argmax(dim=-1)
        desired_pulse = 1 + (goal != left).long()
        goal_loss = F.cross_entropy(logits.flatten(0, 2), desired_pulse.flatten(), reduction="none").reshape_as(active)
        goal_loss = masked_mean(goal_loss, known_context * active * (1 + 2 * (obs[..., SCOUT_HIT] > 0)))
        total = total + args.observed_goal_loss * goal_loss
        terms["post_scout_goal"] = goal_loss.detach()
    return total, terms


def reward_objective(model, batch, device, args, body_fraction=0.0):
    """Actor–critic from the signed public outcome and observed travel only."""
    obs = torch.as_tensor(batch.obs, device=device)
    actions = torch.as_tensor(batch.actions, device=device)
    logits, memory, _, _, _, states = sequence(model, obs)
    active = obs[..., world.ACTIVE]
    signed = obs[..., world.REWARD_POS] - obs[..., world.REWARD_NEG]
    rewards = signed + args.contact_bonus * signed.abs() - args.travel_cost * active - args.probe_cost * obs[..., SCOUT_HIT]
    progress = None
    if args.value_progress_weight:
        progress = value_progress_rewards(obs, memory.detach())
        rewards = rewards + args.value_progress_weight * progress
    if body_fraction:
        physical = observed_body_rewards(obs, contact_bonus=args.body_contact_bonus,
                                         scout_bonus=args.body_scout_bonus, reverse_bonus=args.body_reverse_bonus,
                                         travel_cost=args.travel_cost)
        rewards = torch.lerp(rewards, physical, body_fraction)
    whole_return = discounted_public_returns(rewards, args.discount, args.trial_discount)

    value = model.critic(states).squeeze(-1)
    policy_scores = logits / args.student_temperature
    if batch.action_bias is not None:
        policy_scores = policy_scores + torch.as_tensor(batch.action_bias, device=device)
    logp = policy_scores.log_softmax(dim=-1)
    chosen = logp.gather(-1, actions[..., None]).squeeze(-1)
    advantage = (whole_return - value).detach().clamp(-3, 3)
    actor = -masked_mean(advantage * chosen, active)
    critic = masked_mean(F.smooth_l1_loss(value, whole_return, reduction="none"), active)
    entropy = masked_mean(-(logp.exp() * logp).sum(dim=-1), active)
    total = actor + 0.5 * critic - args.entropy_cost * entropy
    metrics = {"actor": actor.detach(), "critic": critic.detach(), "entropy": entropy.detach()}
    if progress is not None:
        metrics["value_conditioned_progress"] = masked_mean(progress, active)
    if args.outcome_weight:
        if model.config.indexed_outcomes:
            predictions = 3.0 * memory
        else:
            predictions = model.endpoint_probe(torch.cat((states, memory), dim=-1))
        prediction_loss = tried_endpoint_loss(predictions, obs, before_context_weight=args.outcome_before_context_weight)
        total = total + args.outcome_weight * prediction_loss
        metrics["outcome_prediction"] = prediction_loss.detach()
    if args.body_prediction_weight:
        prediction = model.body_predictor(torch.cat((states, F.one_hot(actions, 3).to(states.dtype)), dim=-1))
        motion, events = actual_body_prediction_loss(prediction, obs)
        total = total + args.body_prediction_weight * (motion + events)
        metrics.update(next_motion=motion.detach(), later_physical_events=events.detach())
    return total, metrics


def discounted_public_returns(rewards, tick_discount, trial_discount):
    """Discount normal ticks and optionally carry returns across trials of the same life."""
    ticks = rewards.shape[2]
    future = torch.zeros_like(rewards[:, :, 0])
    within = []
    for index in reversed(range(ticks)):
        future = rewards[:, :, index] + tick_discount * future
        within.append(future)
    basic = torch.stack(list(reversed(within)), dim=2)
    if not trial_discount:
        return basic
    remaining = torch.arange(ticks - 1, -1, -1, device=rewards.device, dtype=rewards.dtype)
    future_weight = (tick_discount ** remaining) * trial_discount
    cross_trial = torch.zeros_like(rewards[:, 0, 0])
    values = []
    for own in reversed(basic.unbind(dim=1)):
        adjusted = own + cross_trial[:, None] * future_weight
        values.append(adjusted)
        cross_trial = adjusted[:, 0]
    return torch.stack(list(reversed(values)), dim=1)


def value_progress_rewards(obs, read):
    """Use the learner's at-sensor value difference and the actual next displacement."""
    revealed = (obs[..., SCOUT_HIT] > 0) & (obs[..., world.CTX].sum(dim=-1) == 1)
    delta = revealed.cumsum(dim=2)
    opened = revealed & (delta == 1)
    identity_left = obs[..., world.LEFT_ID].argmax(dim=-1, keepdim=True)
    left = read.gather(-1, identity_left).squeeze(-1).clamp(-1, 1)
    right = read.gather(-1, 1 - identity_left).squeeze(-1).clamp(-1, 1)
    signed_value = (opened * (right - left)).sum(dim=2, keepdim=True) * 0.5
    position = obs[..., world.POSITION]
    displacement = F.pad(position[:, :, 1:] - position[:, :, :-1], (0, 1))
    distance = obs[..., world.LEFT_DISTANCE:world.RIGHT_DISTANCE + 1].mean(dim=-1).clamp_min(0.01)
    return signed_value * displacement * (delta > 0) * obs[..., world.ACTIVE] / distance


def observed_body_rewards(obs, *, contact_bonus, scout_bonus, reverse_bonus, travel_cost):
    contact = obs[..., world.CONTACT_ID].sum(dim=-1) > 0
    scout = obs[..., SCOUT_HIT] > 0
    probed = scout.cummax(dim=2).values
    side = obs[..., SCOUT_SIDE].argmax(dim=-1)
    reached_right = obs[..., world.POSITION] > 0
    crossed_to_other_end = contact & probed & (reached_right != side)
    return contact_bonus * contact + scout_bonus * scout + reverse_bonus * crossed_to_other_end - travel_cost * obs[..., world.ACTIVE]


def actual_body_prediction_targets(obs):
    pos_vel = obs[..., [world.POSITION, world.VELOCITY]]
    next_pos_vel = F.pad(pos_vel[:, :, 1:], (0, 0, 0, 1))
    events = torch.stack((obs[..., SCOUT_HIT], obs[..., world.CONTACT_ID].sum(dim=-1)), dim=-1)
    padded = F.pad(events, (0, 0, 0, 3))
    next_event = padded.unfold(2, 3, 1)[:, :, 1:].amax(dim=-1)
    return next_pos_vel, next_event


def actual_body_prediction_loss(prediction, obs):
    """Use observed next position and velocity, and actual events in the next three ticks."""
    mask = obs[..., world.ACTIVE]
    next_pos_vel, next_event = actual_body_prediction_targets(obs)
    error = F.smooth_l1_loss(prediction[..., :2], next_pos_vel, reduction="none").mean(dim=-1)
    motion = masked_mean(error, mask)
    event_loss = F.binary_cross_entropy_with_logits(prediction[..., 2:], next_event, reduction="none")
    event_weight = 1 + next_event
    events = masked_mean((event_loss * event_weight).mean(dim=-1), mask)
    return motion, events


def tried_endpoint_loss(predictions, obs, *, before_context_weight=0.25):
    contacts = obs[..., world.CONTACT_ID].sum(dim=2)
    tried = contacts.argmax(dim=-1)[..., None, None].expand(-1, -1, world.TICKS, 1)
    arrived = predictions.gather(dim=-1, index=tried).squeeze(-1)
    positive = (obs[..., world.REWARD_POS].sum(dim=2) > 0)[..., None]
    scored = ((obs[..., world.REWARD_POS] + obs[..., world.REWARD_NEG]).sum(dim=2) > 0)[..., None]
    scored &= contacts.sum(dim=-1, keepdim=True) > 0
    observed_ctx = (obs[..., world.CTX].sum(dim=-1) > 0).cummax(dim=2).values
    mask = scored * obs[..., world.ACTIVE] * torch.where(observed_ctx, 1.0, before_context_weight)
    return binary_loss(arrived, positive.expand_as(arrived).to(arrived.dtype), mask)


def save(path, model, optimizer, step, best):
    temporary = path.with_suffix(".tmp")
    torch.save({**model_payload(model), "optimizer": optimizer.state_dict(), "step": step, "best": best}, temporary)
    temporary.replace(path)


def score(metrics):
    late = metrics["must_reverse_late"]
    return metrics["accuracy"] + 0.45 * late["hits"] / max(1, late["trials"])


def train(args):
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    latest = directory / "last.pt"
    if latest.exists() and not args.resume:
        raise ValueError("Use --resume for an existing run")
    if sum(map(bool, (args.resume, args.initialize, args.initialize_shared))) > 1:
        raise ValueError("Choose one checkpoint initialization mode")
    if args.reward_finetune and (args.forced_steps or args.mixed_through or not args.student_temperature > 0):
        raise ValueError("Reward training requires student-only actions sampled at positive temperature")
    if args.self_labeled and (args.forced_steps or args.mixed_through or args.reward_finetune):
        raise ValueError("Self-label training uses student actions and no teacher")
    if args.outcome_weight and (not (args.endpoint_value or args.indexed_outcomes) or not args.reward_finetune):
        raise ValueError("The tried-endpoint loss requires its head and outcome-based rollouts")
    if args.indexed_outcomes and (not args.reward_finetune or args.endpoint_value):
        raise ValueError("The indexed-outcome control uses reward training and its direct readout")
    if args.post_scout_noise and (not args.reward_finetune or args.post_scout_noise < 0):
        raise ValueError("Post-scout exploration requires a nonnegative scale in reward training")
    if args.body_prediction_weight and (not args.body_model or not args.reward_finetune):
        raise ValueError("The body forecast requires the predictive head and outcome training")
    if args.body_warmup_steps and not args.reward_finetune:
        raise ValueError("Physical curriculum uses only public physical outcomes")
    if not 0 <= args.trial_discount <= 1 or (args.trial_discount and not args.reward_finetune):
        raise ValueError("Cross-trial discount is available for reward training and must be in 0..1")
    if args.freeze_representation and not args.reward_finetune:
        raise ValueError("The frozen-representation control uses outcome training")
    if args.value_progress_weight and (not args.indexed_outcomes or not args.freeze_representation or args.value_progress_weight < 0):
        raise ValueError("Value-conditioned progress requires frozen indexed values and a nonnegative weight")
    configuration = ScoutModelConfig(hidden=args.hidden, degree=args.degree, keys=args.keys,
                                     values=args.values, projection=args.projection, fast_enabled=not args.no_fast,
                                     readout_motor=args.readout_motor, endpoint_value=args.endpoint_value,
                                     indexed_outcomes=args.indexed_outcomes, indexed_min_write=args.indexed_min_write,
                                     body_model=args.body_model)
    start, best = 0, -1.0
    if args.resume:
        old = torch.load(latest, map_location=device, weights_only=True)
        start, best = old["step"], old["best"]
        first_settings = json.loads((directory / "settings.jsonl").open(encoding="utf-8").readline())
        for key in ("seed", "lives", "trials", "train_seed", "map_parity"):
            if getattr(args, key) != first_settings.get(key):
                raise ValueError(f"Resume requires the original {key}")
        if args.freeze_representation != first_settings.get("freeze_representation", False):
            raise ValueError("Resume requires the original representation freeze")
        model = load_scout_model(latest, device)
    elif args.initialize:
        model = load_scout_model(args.initialize, device)
    elif args.initialize_shared:
        model = ScoutMetaPolicy(configuration).to(device)
        source = load_scout_model(args.initialize_shared, device)
        if replace(source.config, readout_motor=configuration.readout_motor) != configuration:
            raise ValueError("Shared initialization permits only a change to the motor head")
        loaded = model.load_state_dict({key: value for key, value in source.state_dict().items()
                                        if not key.startswith("motor.")}, strict=False)
        if loaded.unexpected_keys or any(not key.startswith("motor.") for key in loaded.missing_keys):
            raise ValueError("Unable to load the shared pretrained parameters")
    else:
        model = ScoutMetaPolicy(configuration).to(device)
    if model.config != configuration:
        raise ValueError("The model configuration must match its checkpoint")
    if args.freeze_representation:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(("motor.", "critic.")))

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    if args.resume:
        optimizer.load_state_dict(old["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
    validation = make_batch(args.dev_seed, args.eval_lives, args.eval_trials, parity=args.map_parity,
                            sensitive_count=args.eval_sensitive_count)
    with (directory / "settings.jsonl").open("a", encoding="utf-8") as settings:
        settings.write(json.dumps({**vars(args), "initial_step": start}) + "\n")

    start_time = perf_counter()
    for step in range(start + 1, args.steps + 1):
        if (directory / "STOP").exists():
            break
        seed = args.train_seed + (step - 1) * args.lives
        tape = make_batch(seed, args.lives, args.trials, parity=args.map_parity, switch_probability=args.rule_switch,
                           sensitive_count=args.sensitive_count)
        curriculum = 0.0
        if args.body_warmup_steps:
            curriculum = 1 - min(1.0, max(0.0, (step - args.body_warmup_steps) / max(1, args.body_restore_steps)))
        if curriculum:
            scale = 1.0 - (1.0 - args.body_scale) * curriculum
            beacon = 0.35 - (0.35 - args.body_beacon) * curriculum
            tape = scaled_geometry(tape, scale, beacon)
        warm = step <= args.forced_steps
        student = None if warm else ScoutRawAgent(model)
        teacher_mix = args.teacher_mix if step <= args.mixed_through else 0.0
        reported_stage = ("public_self_labels" if args.self_labeled else "reward_policy_history" if args.reward_finetune else "teacher_history" if warm else
                          "mixed_history" if teacher_mix else "student_history")
        if curriculum:
            reported_stage = "physical_returns" if curriculum == 1 else "physical_reward_transition"
        receipt_dropout = 1.0 if curriculum == 1.0 else args.receipt_dropout
        examples = collect(tape, seed=seed ^ 1770293, student=student, teacher_mix=teacher_mix,
                           temperature=args.student_temperature, teacher_targets=not (args.reward_finetune or args.self_labeled),
                           receipt_dropout=receipt_dropout, receipt_cutoff_probability=args.receipt_cutoff_probability,
                           receipt_cutoff_trial=args.receipt_cutoff_trial, coherent=args.self_labeled,
                           post_scout_noise=args.post_scout_noise)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if args.reward_finetune:
            loss, components = reward_objective(model, examples, device, args, body_fraction=curriculum)
        else:
            loss, components = learning_objective(model, examples, device, args)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()

        stage_boundaries = (args.forced_steps, args.mixed_through, args.body_warmup_steps,
                            args.body_warmup_steps + args.body_restore_steps, args.steps)
        scheduled = step % args.eval_every == 0 or step in stage_boundaries
        if scheduled:
            trial = rollout_scout(ScoutRawAgent(model), validation)
            development = metrics(trial, validation)
            metric = score(development)
            if metric > best:
                best = metric
                save(directory / "best.pt", model, optimizer, step, best)
        else:
            development = None
        if scheduled or step % args.save_every == 0 or step == start + 1 or (directory / "STOP").exists():
            save(latest, model, optimizer, step, best)
            record = {"step": step, "stage": reported_stage, "s": round(perf_counter() - start_time, 3),
                      "physical_reward_fraction": curriculum,
                      "loss": float(loss.detach()), "parts": {key: float(value) for key, value in components.items()},
                      "training_history_metrics": examples.metrics, "development_policy": development}
            with (directory / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps({"step": step, "stage": reported_stage, "s": record["s"], "loss": record["loss"],
                              "validation": development, "parts": record["parts"]}), flush=True)


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", required=True)
    cli.add_argument("--seed", type=int, default=2211)
    cli.add_argument("--train-seed", type=int, default=4_211_000_000)
    cli.add_argument("--dev-seed", type=int, default=4_220_000_000)
    cli.add_argument("--device", default="cpu")
    cli.add_argument("--cpu-threads", type=int, default=2)
    cli.add_argument("--steps", type=int, default=420)
    cli.add_argument("--lives", type=int, default=64)
    cli.add_argument("--trials", type=int, default=40)
    cli.add_argument("--eval-lives", type=int, default=192)
    cli.add_argument("--eval-trials", type=int, default=64)
    cli.add_argument("--map-parity", type=int, choices=(0, 1))
    cli.add_argument("--sensitive-count", type=int, choices=range(5))
    cli.add_argument("--eval-sensitive-count", type=int, choices=range(5))
    cli.add_argument("--rule-switch", type=float, default=0.0)
    cli.add_argument("--save-every", type=int, default=16)
    cli.add_argument("--eval-every", type=int, default=32)
    cli.add_argument("--lr", type=float, default=0.002)
    cli.add_argument("--forced-steps", type=int, default=128)
    cli.add_argument("--mixed-through", type=int, default=192)
    cli.add_argument("--teacher-mix", type=float, default=0.125)
    cli.add_argument("--student-temperature", type=float, default=0.0)
    cli.add_argument("--self-labeled", action="store_true")
    cli.add_argument("--reward-finetune", action="store_true")
    cli.add_argument("--freeze-representation", action="store_true")
    cli.add_argument("--value-progress-weight", type=float, default=0.0)
    cli.add_argument("--travel-cost", type=float, default=0.008)
    cli.add_argument("--probe-cost", type=float, default=0.015)
    cli.add_argument("--contact-bonus", type=float, default=0.1)
    cli.add_argument("--entropy-cost", type=float, default=0.012)
    cli.add_argument("--discount", type=float, default=0.995)
    cli.add_argument("--trial-discount", type=float, default=0.0)
    cli.add_argument("--outcome-weight", type=float, default=0.0)
    cli.add_argument("--outcome-before-context-weight", type=float, default=0.25)
    cli.add_argument("--post-scout-noise", type=float, default=0.0)
    cli.add_argument("--body-model", action="store_true")
    cli.add_argument("--body-prediction-weight", type=float, default=0.0)
    cli.add_argument("--body-warmup-steps", type=int, default=0)
    cli.add_argument("--body-restore-steps", type=int, default=0)
    cli.add_argument("--body-scale", type=float, default=0.62)
    cli.add_argument("--body-beacon", type=float, default=0.22)
    cli.add_argument("--body-contact-bonus", type=float, default=0.4)
    cli.add_argument("--body-scout-bonus", type=float, default=0.4)
    cli.add_argument("--body-reverse-bonus", type=float, default=1.2)
    cli.add_argument("--memory-loss", type=float, default=1.0)
    cli.add_argument("--decision-weight", type=float, default=0.0)
    cli.add_argument("--observed-goal-loss", type=float, default=0.0)
    cli.add_argument("--counterfactual-context-loss", type=float, default=0.0)
    cli.add_argument("--context-loss", type=float, default=0.25)
    cli.add_argument("--receipt-loss", type=float, default=0.45)
    cli.add_argument("--gate-loss", type=float, default=0.55)
    cli.add_argument("--write-cost", type=float, default=0.0)
    cli.add_argument("--receipt-dropout", type=float, default=0.0)
    cli.add_argument("--receipt-cutoff-probability", type=float, default=0.0)
    cli.add_argument("--receipt-cutoff-trial", type=int, default=18)
    cli.add_argument("--key-loss", type=float, default=0.2)
    cli.add_argument("--hidden", type=int, default=56)
    cli.add_argument("--degree", type=int, default=10)
    cli.add_argument("--keys", type=int, default=12)
    cli.add_argument("--values", type=int, default=16)
    cli.add_argument("--projection", type=int, default=48)
    cli.add_argument("--no-fast", action="store_true")
    cli.add_argument("--readout-motor", action="store_true")
    cli.add_argument("--endpoint-value", action="store_true")
    cli.add_argument("--indexed-outcomes", action="store_true")
    cli.add_argument("--indexed-min-write", type=float, default=0.0)
    cli.add_argument("--resume", action="store_true")
    cli.add_argument("--initialize")
    cli.add_argument("--initialize-shared")
    return cli


if __name__ == "__main__":
    train(parser().parse_args())
