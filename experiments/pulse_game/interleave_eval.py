"""Independent, closed-loop holdout for interleaved odor and mark learning."""

from argparse import ArgumentParser
from copy import copy
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch

from . import local_trace_graph as local, mixture_world as food, revisit_choice_graph as legacy, world
from . import self_param_writer as writer


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/interleaved-learning-20261120"
EVIDENCE = ROOT / "docs/evidence/interleaved-learning-2026-11-20"
EVAL_DOMAIN = 0x7EA141
DONOR_DOMAIN = 0x7EA142
BOOTSTRAP_DOMAIN = 0x7EA143
REPLICATION_DOMAIN = 0x7EA151
REPLICATION_DONOR_DOMAIN = 0x7EA152
CUES = np.array([[0., 1.1, 0.], [0., -.55, .95262794], [0., -.55, -.95262794]], np.float32)
MARK_AXIS = np.array([1.35, 0., 0.], np.float32)
SENSOR_SCALE = np.float32(2.75)
A, B, C, MARK = range(4)
HISTORIES = ("aba", "cba", "flipped", "b_only", "prime_zero", "prime_donor")
PROFILES = ("familiar", "independent_odors", "new_layout", "novel_marks", "multi_pair")
DOSES = (0, 1, 2, 4)
PRIMARY = ("A", "B", "both", "mark")
SHUFFLE_AUDIT = ("matched_weights", "matched_bias", "relocated_birth", "relocated_retirement", "matched_retirement")


@dataclass(frozen=True)
class Arena:
    menu: food.Foods
    actual_ears: np.ndarray
    roots: np.ndarray
    maps: np.ndarray
    mark_law: np.ndarray
    contexts: np.ndarray
    item_pair: np.ndarray
    exposed: np.ndarray
    donor: np.ndarray
    motor_draw: np.ndarray
    edit_draw: np.ndarray
    shuffle_draw: np.ndarray
    gap: np.ndarray
    scored: dict[str, np.ndarray]
    reversal_at: int | None
    profile: str

    @property
    def good_side(self):
        return self.menu.probability.argmax(axis=2)


@dataclass(frozen=True)
class Measurement:
    side: np.ndarray
    tick: np.ndarray
    reward: np.ndarray
    observed: np.ndarray
    pulses: np.ndarray
    write_count: np.ndarray
    graph_digest: str | None
    shuffle_audit: np.ndarray | None
    probes: dict[str, world.Rollout]


def _rng(root, stream, domain=EVAL_DOMAIN):
    return np.random.default_rng(np.random.SeedSequence([int(root), domain, stream]))


def world_domains(split):
    if split == "replication":
        return REPLICATION_DOMAIN, REPLICATION_DONOR_DOMAIN
    return EVAL_DOMAIN, DONOR_DOMAIN


def evaluation_roots(split, count):
    """Disjoint numerical ranges for pilot, main and optional replication roots."""
    bounds = {"pilot": (2_000_000_000, 2_100_000_000), "main": (2_100_000_000, 2_200_000_000),
              "replication": (2_200_000_000, 2_300_000_000)}
    low, high = bounds[split]
    domain, _ = world_domains(split)
    return tuple(int(v) for v in _rng(low, 45, domain).choice(high - low, count, replace=False) + low)


def _direction(rng):
    x = rng.normal(size=3).astype(np.float32)
    return 1.35 * x / np.linalg.norm(x)


def _pair(rng, independent):
    x = _direction(rng)
    if not independent:
        return np.stack((x, -x))
    y = _direction(rng)
    while abs(float(x @ y) / 1.35**2) > .7:
        y = _direction(rng)
    return np.stack((x, y))


def _new_mark(rng):
    x = _direction(rng)
    while abs(float(x[0]) / 1.35) > .5:
        x = _direction(rng)
    return x


def _mappings(profile):
    if profile == "multi_pair":
        a = np.array(list(product((0, 1), repeat=3)), np.int8)
        return np.stack((a, 1 - a), axis=1)
    return np.array(list(product((0, 1), repeat=2)), np.int8).reshape(4, 2, 1)


def _history_schedule(history, pair_count, rounds):
    first = {"cba": C, "b_only": B}.get(history, A)
    # Closely interleaved marks early in life, followed by a conflicting context streak.
    sequence = [(MARK, 0, "demo"), (first, 0, "teach"), (MARK, 1, "demo"), (MARK, 0, "teach")]
    for i in range(rounds * pair_count - 1):
        sequence.append((first, (i + 1) % pair_count, "teach"))
        if i == 0:
            sequence.append((MARK, 1, "teach"))
    sequence.extend((B, i % pair_count, "teach") for i in range(rounds * pair_count))
    return sequence


def _reversal_schedule(rounds):
    sequence = _history_schedule("aba", 1, rounds)
    # Hold A's same cue and same repeated identities on both sides of the change.
    sequence.extend((A, 0, "teach") for _ in range(rounds))
    sequence.extend((A, 0, f"dose{dose}") for dose in range(5))
    return sequence


def _physical(root, length, profile, domain):
    tape_seed = int(np.random.SeedSequence([root, domain, 917]).generate_state(1)[0])
    # Novel layout changes endpoint asymmetry and current strength, preserving reachability.
    if profile == "new_layout":
        tape = world.make_tape([tape_seed], length, distance_jitter=.36, wind_scale=.16)
        distances = tape.distances[0].copy()
        near = _rng(root, 201, domain).integers(0, 2, size=length)
        distances[np.arange(length), near] *= .89
        distances[np.arange(length), 1 - near] *= 1.07
        return tape_seed, distances, tape.wind[0]
    tape = world.make_tape([tape_seed], length, distance_jitter=.13, wind_scale=.07)
    return tape_seed, tape.distances[0], tape.wind[0]


def make_arena(roots, profile="familiar", case="aba", rounds=4, *, domain=EVAL_DOMAIN, donor_domain=DONOR_DOMAIN):
    """Sample physical roots once and fully cross the same exogenous draws with private laws."""
    if profile not in PROFILES or rounds < 2 or len(roots) != len(set(roots)) or not roots:
        raise ValueError("Choose an evaluation profile, distinct roots and at least two rounds")
    maps = np.repeat(_mappings(profile), 2, axis=0)
    width, _, pair_count = maps.shape
    mark_law = np.tile(np.array([0, 1], np.int8), width // 2)
    reversal = case in {"stable", "natural", *(f"aligned_{d}" for d in DOSES)}
    if case not in HISTORIES and not reversal:
        raise ValueError(f"Unknown heldout case {case}")
    if reversal and pair_count != 1:
        raise ValueError("Single pairing is required to count feedback on the reversed mapping")
    sequence = _reversal_schedule(rounds) if reversal else _history_schedule(case, pair_count, rounds)
    length = len(sequence) + (0 if reversal else 3)
    reversal_at = length - 5 if reversal else None
    count = len(roots) * width
    contexts = np.empty((count, length), np.int8)
    item_pair = np.empty_like(contexts)
    ears = np.empty((count, length, 2, 3), np.float32)
    realized = np.empty((count, length, 2), np.int8)
    exposed = np.ones((count, length), bool)
    donor = np.zeros((count, length), np.int8)
    scored = {key: np.empty(count, np.int16) for key in (tuple(f"dose{d}" for d in DOSES) if reversal else ("A", "B", "mark"))}
    distances = np.empty((count, length, 2), np.float32)
    wind = np.empty((count, length, world.MOVEMENT_TICKS), np.float32)
    motor = np.empty((count, length), np.float32)
    edit = np.empty((count, length, 6), np.float32)
    shuffle = np.empty((count, length, local.SHUFFLE_DRAWS), np.float32)
    gap = np.empty((count, length), np.int8)
    seeds = []
    for i, root in enumerate(roots):
        rows = slice(i * width, (i + 1) * width)
        geom, design, pos = (_rng(root, j, domain) for j in range(3))
        seed, distances[rows], wind[rows] = _physical(root, length, profile, domain)
        seeds.extend([seed] * width)
        ambient = CUES[geom.permutation(3)]
        odors = np.stack([_pair(geom, profile == "independent_odors") for _ in range(pair_count)])
        c_odors = np.stack([_pair(geom, profile == "independent_odors") for _ in range(pair_count)])
        mark = _new_mark(_rng(root, 10, domain)) if profile == "novel_marks" else MARK_AXIS
        marks = np.stack((mark, -mark))
        c_law = _rng(root, 12, domain).integers(0, 2, size=pair_count)
        motor[rows] = _rng(root, 3, domain).random(length, dtype=np.float32)
        edit[rows] = _rng(root, 4, domain).random((length, 6), dtype=np.float32)
        shuffle[rows] = _rng(root, 5, domain).random((length, local.SHUFFLE_DRAWS), dtype=np.float32)
        gap[rows] = _rng(root, 6, domain).integers(0, 9, size=length)
        left_item = pos.integers(0, 2, size=length)
        probe_pair = design.integers(0, pair_count)
        visits = list(sequence)
        if not reversal:
            order = design.permutation([A, B, MARK])
            visits.extend((context, 0 if context == MARK else probe_pair, "probe") for context in order)
            if profile == "new_layout":
                # Use the ABA schedule's swap in every history, even where the receiver has no A exposure.
                nominal = _history_schedule("aba", pair_count, rounds)
                for j, (context, pair, _) in enumerate(visits[len(sequence):], len(sequence)):
                    earlier = [k for k, (c, p, role) in enumerate(nominal)
                               if c == context and p == pair and role == "teach"]
                    left_item[j] = 1 - left_item[earlier[-1]] if earlier else left_item[j]

        # Independent donor's private law and uniform motor choices, sampled without the receiver map or policy.
        donor_rng = _rng(root, 30, donor_domain)
        donor_law = donor_rng.integers(0, 2, size=(3, pair_count))
        donor_left = donor_rng.integers(0, 2, size=length)
        donor_motor = donor_rng.integers(0, 2, size=length)
        for j, (context, pair, role) in enumerate(visits):
            contexts[rows, j] = context
            item_pair[rows, j] = pair
            both_items = [left_item[j], 1 - left_item[j]]
            if context == MARK:
                mark_items = [pair, pair] if role == "demo" else both_items
                ears[rows, j] = marks[mark_items]
                realized[rows, j] = np.where(np.array(mark_items) == mark_law[:, None], 1, -1)
            else:
                odors_here = c_odors if context == C else odors
                ears[rows, j] = odors_here[pair, both_items] + ambient[context]
                good = np.full(width, c_law[pair]) if context == C else maps[:, context, pair]
                if case == "flipped" and context == A and role == "teach":
                    good = 1 - good
                if reversal and j >= reversal_at and case != "stable":
                    good = 1 - good
                realized[rows, j] = np.where(np.array(both_items) == good[:, None], 1, -1)
                if case == "prime_donor" and context == A and role == "teach":
                    contacted = donor_left[j] ^ donor_motor[j]
                    donor[rows, j] = 1 if contacted == donor_law[context, pair] else -1
                if case == "prime_zero" and context == A and role == "teach":
                    exposed[rows, j] = False
            if role == "probe":
                scored[{A: "A", B: "B", MARK: "mark"}[context]][rows] = j
                exposed[rows, j] = False
            if reversal and role in scored:
                scored[role][rows] = j

    if reversal and case.startswith("aligned_"):
        dose = int(case.rsplit("_", 1)[1])
        exposed[:, reversal_at:reversal_at + 5] = False
        exposed[:, reversal_at + 4 - dose:reversal_at + 4] = True
        scored = {f"dose{dose}": scored["dose4"]}
    elif reversal:
        # The final choice has no feedback. Earlier choices are scored before their own receipts.
        exposed[:, -1] = False

    base = world.make_tape(seeds, length)
    base = replace(base, ctx=np.minimum(contexts, 1), distances=distances, wind=wind)
    chemicals = np.empty((count, length, 2, 4), np.float32)
    if np.max(np.abs(ears)) > SENSOR_SCALE:
        raise RuntimeError("The physical receptor scale would clip an ear")
    chemicals[..., :3] = (1 + ears / SENSOR_SCALE) * .5
    chemicals[..., 3] = .5
    menu = food.Foods(base, chemicals, (realized > 0).astype(np.float32), realized,
                      np.empty(0), "interleaved_holdout", len(sequence))
    return Arena(menu, ears, np.repeat(np.array(roots), width), np.tile(maps, (len(roots), 1, 1)),
                 np.tile(mark_law, len(roots)), contexts, item_pair, exposed, donor, motor, edit, shuffle,
                 gap, scored, reversal_at, profile)


class _Body(food.Body):
    def __init__(self, arena, **kwargs):
        self.arena = arena
        super().__init__(arena.menu, **kwargs)

    def _load_trial(self):
        super()._load_trial()
        # Bilateral ear offsets carry the public context. Never expose private context indices.
        self._background[:, food.CTX] = 0

    def _shown_reward(self, receipt):
        signed = super()._shown_reward(receipt)
        j = self.trial_index
        shown = np.where(self.arena.donor[:, j] != 0, self.arena.donor[:, j] * receipt, signed)
        return np.where(self.arena.exposed[:, j], shown, 0).astype(np.int8)


def ears_from_observation(obs):
    both = obs[:, food.LEFT.start:food.RIGHT.stop].reshape(len(obs), 2, 4)
    return (2 * both[..., :3] - 1) * SENSOR_SCALE


def _steer(side, obs):
    active = obs[:, world.ACTIVE] != 0
    return np.column_stack((active & (side == 0), active & (side == 1)))


class _PublicPolicy:
    """Deterministic pair lookup trained solely from own visible contact and taste."""

    def __init__(self, motor_draw, *, learn=True):
        self.motor_draw = motor_draw
        self.learn = learn

    def reset(self, n):
        self.trial = -1
        self.known_pairs = [{} for _ in range(n)]
        self.known_items = [{} for _ in range(n)]
        self.side = np.zeros(n, dtype=np.int8)

    @staticmethod
    def _keys(obs):
        # The transmitted float32 observations are exact across revisits and side swaps.
        chemicals = obs[:, food.LEFT.start:food.RIGHT.stop].reshape(len(obs), 2, 4)
        return [(bytes(left), bytes(right)) for left, right in chemicals]

    def act(self, obs):
        if obs[0, world.TRIAL_START]:
            self.trial += 1
            self.side[:] = self.motor_draw[:, self.trial] < .5
            if self.learn:
                self.items = self._keys(obs)
                for row, (left, right) in enumerate(self.items):
                    pair = tuple(sorted((left, right)))
                    good = self.known_pairs[row].get(pair)
                    if good is not None:
                        self.side[row] = int(right == good)
                    else:
                        knowledge = self.known_items[row]
                        sl, sr = knowledge.get(left, 0), knowledge.get(right, 0)
                        if sl != sr:
                            self.side[row] = int(sl < sr)
        if self.learn:
            taste = obs[:, world.REWARD_POS] - obs[:, world.REWARD_NEG]
            for row in np.flatnonzero((taste != 0) & obs[:, world.CONTACT_ID].any(axis=1)):
                left, right = self.items[row]
                side = int(obs[row, world.CONTACT_ID].argmax())
                if left == right:
                    self.known_items[row][left] = int(taste[row])
                else:
                    items = (left, right)
                    good = items[side if taste[row] > 0 else 1 - side]
                    self.known_pairs[row][tuple(sorted(items))] = good
        return _steer(self.side, obs)


class _GraphPolicy:
    """Drive the physical body toward the native graph's sampled motor output."""

    def __init__(self, graph, rule, arena, *, intervention="active", stochastic=True):
        if intervention == "shuffle" and rule is None:
            raise ValueError("Address shuffle uses the local writer's native dose-matched control")
        self.graph, self.rule, self.intervention, self.stochastic = graph, rule, intervention, stochastic
        self.motor_draw = torch.from_numpy(arena.motor_draw)
        self.edit_draw = torch.from_numpy(arena.edit_draw)
        self.shuffle_draw = torch.from_numpy(arena.shuffle_draw)
        self.gap = torch.from_numpy(arena.gap)

    def reset(self, n):
        self.trial = -1
        self.state = (local.start if self.rule else legacy.start)(self.graph, n)
        if self.rule is not None:
            self.expanded = self.rule.expand(n)
        self.write_count = np.zeros(self.motor_draw.shape, np.int16)
        self.shuffle_audit = np.full((*self.motor_draw.shape, len(SHUFFLE_AUDIT)), -1, np.int8) if self.intervention == "shuffle" else None
        self.side = np.zeros(n, np.int8)
        self.wait = torch.zeros(n, dtype=torch.int64)

    def _advance_gap(self):
        self.state = (local.wait(self.expanded, self.state, self.wait) if self.rule else
                      legacy.wait(self.state, self.wait))

    def _start_trial(self, obs):
        if self.trial >= 0:
            self._advance_gap()
        self.trial += 1
        self.wait = self.gap[:, self.trial].to(torch.int64)
        self.ears = torch.from_numpy(ears_from_observation(obs))
        left, right = self.ears.unbind(dim=1)
        draw = self.motor_draw[:, self.trial]
        if self.rule:
            self.action = local.act(self.expanded, self.state, left, right, draw, stochastic=self.stochastic)
        else:
            self.action = legacy.act(self.state, left, right, draw, stochastic=self.stochastic)
        self.state = self.action.state
        self.side = (self.action.side.numpy() > 0).astype(np.int8)

    def _local_contact(self, rows, sides, taste):
        index = torch.from_numpy(rows)
        left, right = self.ears[index].unbind(dim=1)
        rule = replace(self.expanded, **{key: values[index] for key, values in vars(self.expanded).items()})
        landed = 2 * sides - 1
        action = replace(self.action, state=self.action.state.subset(index), side=landed,
                         motor_logit=self.action.motor_logit[index], motor_log_prob=self.action.motor_log_prob[index])
        experience = local.observe(rule, action, left, right, taste)
        # A silent physical contact carries its side while its graph is held fixed.
        mode = self.intervention if taste.count_nonzero() else "disabled"
        edit = self.edit_draw[index, self.trial, :local.EDIT_DRAWS]
        result = local.commit(rule, experience, edit, intervention=mode,
                              shuffle=self.shuffle_draw[index, self.trial] if mode == "shuffle" else None)
        receipt = result.receipt
        changed = (receipt.weight_delta.count_nonzero(dim=(1, 2)) + receipt.bias_delta.count_nonzero(dim=1)
                   + (receipt.birth >= 0) + (receipt.retired >= 0))
        self.write_count[rows, self.trial] = changed.numpy()
        if receipt.match is not None:
            match = receipt.match
            self.shuffle_audit[rows, self.trial] = torch.stack((match.weight_matches, match.bias_matches,
                                                               match.birth_relocated, match.retirement_relocated,
                                                               match.retirement_matches), dim=1).numpy()
        return result.state

    def _legacy_contact(self, rows, sides, taste):
        index = torch.from_numpy(rows)
        left, right = self.ears[index].unbind(dim=1)
        action = replace(self.action.subset(index), side=2 * sides - 1)
        experience = legacy.taste(action, left, right, taste)
        draws = self.edit_draw[index, self.trial]
        candidates = legacy.propose(experience, draws[:, :4].reshape(len(rows), 2, 2))
        decision = legacy.choose(candidates, draws[:, 4])
        mode = self.intervention if taste.count_nonzero() else "disabled"
        result = legacy.commit(experience, candidates, decision, intervention=mode)
        self.write_count[rows, self.trial] = (result.raw_receipt.status == writer.Status.APPLIED).numpy()
        return result.state

    def _merge(self, rows, subset):
        index = torch.from_numpy(rows)
        graph = writer.Graph(*(getattr(self.state.graph, key).index_copy(0, index, getattr(subset.graph, key))
                               for key in ("weights", "bias", "present")))
        names = ("hidden", "sender_trace", "receiver_trace") if self.rule else ("hidden",)
        self.state = type(self.state)(graph, *(getattr(self.state, key).index_copy(0, index, getattr(subset, key))
                                               for key in names))

    @torch.no_grad()
    def act(self, obs):
        if obs[0, world.TRIAL_START]:
            self._start_trial(obs)
        received = obs[:, world.CONTACT_ID].any(axis=1)
        valence = obs[:, world.REWARD_POS] - obs[:, world.REWARD_NEG]
        # Separate lane-wise silent and signed contacts. Position/side comes from physical receipt.
        for mask in (received & (valence != 0), received & (valence == 0)):
            rows = np.flatnonzero(mask)
            if len(rows):
                sides = torch.from_numpy(obs[rows, world.CONTACT_ID].argmax(axis=1)).float()
                taste = torch.from_numpy(valence[rows])
                operation = self._local_contact if self.rule else self._legacy_contact
                self._merge(rows, operation(rows, sides, taste))
        return _steer(self.side, obs)


def _single_probe(arena, indices):
    rows = np.arange(len(arena.roots))

    def take(array):
        return array[rows, indices][:, None]

    body = arena.menu.body
    single_body = replace(body, **{key: take(getattr(body, key))
                                  for key in ("ctx", "cue", "left_id", "good_id", "distances", "wind")})
    menu = replace(arena.menu, body=single_body, chemicals=take(arena.menu.chemicals),
                   probability=take(arena.menu.probability), realized=take(arena.menu.realized), counter=0)
    selection = {key: take(getattr(arena, key)) for key in ("actual_ears", "contexts", "item_pair", "exposed",
                                                            "donor", "motor_draw", "edit_draw", "shuffle_draw", "gap")}
    return replace(arena, menu=menu, scored={}, **selection)


@torch.no_grad()
def _probe_fork(arena, common_prefix, indices):
    probe = _single_probe(arena, indices)
    if probe.exposed.any():
        raise AssertionError("A same-snapshot probe must suppress feedback")
    # Graph states are immutable dataclasses. Each fork advances an independent view
    # of exactly the same graph, traces and activity after the common B prefix.
    actor = copy(common_prefix)
    actor.trial = -1
    actor.side = actor.side.copy()
    actor.motor_draw = probe.motor_draw
    if isinstance(actor, _GraphPolicy):
        actor._advance_gap()
        actor.motor_draw = torch.from_numpy(probe.motor_draw)
        actor.edit_draw = torch.from_numpy(probe.edit_draw)
        actor.shuffle_draw = torch.from_numpy(probe.shuffle_draw)
        actor.gap = torch.from_numpy(probe.gap)
        actor.write_count = np.zeros(probe.motor_draw.shape, np.int16)
        actor.shuffle_audit = None
    body = _Body(probe)
    obs = body.observe()
    while obs is not None:
        obs = body.step(actor.act(obs))
    measured = body.result()
    if measured.observed_reward.any() or isinstance(actor, _GraphPolicy) and actor.write_count.any():
        raise AssertionError("An isolated probe observed feedback or wrote to its graph")
    return measured


def rollout(arena, policy, *, record=False):
    body = _Body(arena, record=record)
    policy.reset(arena.menu.body.n)
    obs = body.observe()
    probes = {}
    while obs is not None:
        if arena.reversal_at is None and body.trial_index == arena.menu.counter and body.tick_index == 0:
            probes = {label: _probe_fork(arena, policy, indices) for label, indices in arena.scored.items()}
        obs = body.step(policy.act(obs))
    result = body.result()
    digest = None
    if isinstance(policy, _GraphPolicy):
        graph = policy.state.graph
        digest = sha256(b"".join(t.detach().contiguous().numpy().tobytes()
                                  for t in (graph.weights, graph.bias, graph.present))).hexdigest()
    counts = getattr(policy, "write_count", np.zeros(result.contact_side.shape, np.int16))
    return Measurement(result.contact_side, result.contact_tick, result.reward, result.observed_reward,
                       result.motor_pulses, counts, digest, getattr(policy, "shuffle_audit", None), probes), result.obs


def payload(arena, result):
    """Raw, per-lane observations and fitted outputs with no silent scoring exclusions."""
    counts = np.count_nonzero(result.observed, axis=2)
    hits = result.side >= 0
    if np.any(counts > 1) or np.any(counts[~hits]) or np.any(counts[~arena.exposed]):
        raise AssertionError("A reward was exposed without one enabled physical contact")
    if np.any(result.write_count[(counts == 0)]):
        raise AssertionError("An unlicensed graph write occurred at a silent contact")
    rows = np.arange(len(arena.roots))
    data = dict(roots=arena.roots, maps=arena.maps, mark_law=arena.mark_law, contexts=arena.contexts, pair=arena.item_pair,
                distances=arena.menu.body.distances, wind=arena.menu.body.wind, ears=arena.actual_ears,
                donor=arena.donor, enabled_feedback=arena.exposed, motor_draw=arena.motor_draw,
                edit_draw=arena.edit_draw, gap=arena.gap, side=result.side, contact_tick=result.tick,
                realized_reward=result.reward, observed_reward=result.observed.sum(axis=2),
                pulses=result.pulses, writes=result.write_count, good_side=arena.good_side,
                graph_digest=np.asarray(result.graph_digest or ""))
    for label, indices in arena.scored.items():
        data[f"index/{label}"] = indices
        chosen = result.side[rows, indices]
        if label in result.probes:
            readout = result.probes[label]
            data[f"serial_side/{label}"] = chosen
            data[f"probe_tick/{label}"] = readout.contact_tick[:, 0]
            data[f"probe_pulses/{label}"] = readout.motor_pulses[:, 0]
            chosen = readout.contact_side[:, 0]
            first = indices == arena.menu.counter
            if not np.array_equal(chosen[first], data[f"serial_side/{label}"][first]):
                raise AssertionError("An isolated probe disagreed with the same request run first in sequence")
        data[f"hit/{label}"] = chosen >= 0
        data[f"side/{label}"] = chosen
        data[f"right/{label}"] = arena.good_side[rows, indices] == 1
        data[f"success/{label}"] = chosen == arena.good_side[rows, indices]
        if arena.reversal_at is None:
            if np.any(counts[rows, indices]):
                raise AssertionError("A silent return score included feedback")
        else:
            before = (np.arange(counts.shape[1]) >= arena.reversal_at) & (np.arange(counts.shape[1]) < indices[:, None])
            data[f"changed_feedback_before/{label}"] = np.sum(counts * before, axis=1)
    if arena.reversal_at is None:
        data["success/both"] = data["success/A"] & data["success/B"]
        data["hit/both"] = data["hit/A"] & data["hit/B"]
        data["success/both_modalities"] = data["success/both"] & data["success/mark"]
    if result.shuffle_audit is not None:
        data["shuffle_audit"] = result.shuffle_audit
    return data


def _atomic_json(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(contents, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _atomic_npz(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **data)
    tmp.replace(path)


def _specs(args):
    entries = [dict(name=p, birth=0, kind=p, intervention="active") for p in ("uniform", "public_lookup")]
    for birth in args.births:
        if args.old:
            for controller, task, step in (("local", "odor", 8000), ("local", "feeder", 8000),
                                           ("legacy", "odor", 8000), ("legacy", "feeder", 8000)):
                entries.append(dict(name=f"old_{controller}_{task}_{step}", birth=birth, kind="old",
                                    controller=controller, task=task, step=step, intervention="active"))
        if args.new:
            for name in args.arms:
                for mode in (("active", "disabled", "shuffle") if name == "local_active" else ("active",)):
                    suffix = "" if mode == "active" else f"_{mode}"
                    entries.append(dict(name=f"{name}{suffix}", birth=birth, kind="new",
                                        arm=name, step=args.step, intervention=mode))
    return entries


def _checkpoint_model(spec, args):
    if spec["kind"] == "old":
        from . import local_trace_train

        base = local_trace_train.DEFAULT_OUTPUT / spec["controller"]
        path = local_trace_train.checkpoint_path(base, spec["task"], "active", spec["birth"], spec["step"])
        return local_trace_train.load_checkpoint(base, spec["task"], "active", spec["birth"], spec["step"]), path
    if spec["kind"] == "new":
        from . import interleave_train

        path = args.new / spec["arm"] / f"step-{spec['step']:06d}.pt"
        return interleave_train.load_checkpoint(args.new, spec["arm"], spec["birth"], spec["step"]), path
    return None, None


def _file_digest(path):
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol(args, roots, specs, profiles, cases):
    domain, donor_domain = world_domains(args.split)
    return dict(split=args.split, rounds_per_context_pair=args.rounds, roots=roots,
                profiles=profiles, cases=cases, specs=specs, stochastic=not args.greedy,
                domain=domain, donor_domain=donor_domain,
                source_sha256=_file_digest(Path(__file__)),
                cue_bank=CUES.tolist(), mark_axis=MARK_AXIS.tolist(),
                physical="A native motor choice drives sustained left/right pulses in the existing mixture body. "
                "Its actual contact supplies side plus signed taste to native sensory ports; all silent contacts "
                "retain side while forcing physical writes off. Movement-step states stay in the body adapter.",
                laws="Single-pair profiles fully cross independent A/B binary laws and both mark laws per root. "
                "The three-pair stress crosses all 8 A maps with both mark laws and sets B to the inverse "
                "for each pair. C has newly drawn odors "
                "and unrelated laws. Flipped keeps its prime sensory streams and inverts only earlier A taste. "
                "B-only substitutes B for A while holding length constant. Donor signs share neither private "
                "law nor choices with the receiver, and appear at the receiver's actual receipt event.",
                scoring="A, B and marks are measured in distinct silent physical forks of the same pre-probe "
                "snapshot. The randomly ordered silent sequence is retained as a secondary measurement. "
                "Missed contacts count as incorrect. Joint A and B means both forked physical contacts were correct. "
                "The reversal's natural row exposes feedback immediately after each of the first four changed "
                "requests; aligned rows mask the earlier changes so all doses share the final physical layout.",
                selection="These evaluation roots are held out from model and checkpoint selection.")


def execute(args):
    roots = evaluation_roots(args.split, args.roots)
    domain, donor_domain = world_domains(args.split)
    profiles = tuple(args.profiles)
    cases = tuple(args.cases)
    specs = _specs(args)
    models = {}
    for spec in specs:
        key = (spec["name"], spec["birth"])
        models[key], source = _checkpoint_model(spec, args)
        if source is not None:
            spec["checkpoint_sha256"] = _file_digest(source)
    output = RAW / args.out
    evidence = EVIDENCE / args.out
    protocol = _protocol(args, roots, specs, profiles, cases)
    manifest = output / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != json.loads(json.dumps(protocol)):
        raise ValueError(f"Output name {args.out!r} already has a different frozen protocol")
    _atomic_json(manifest, protocol)

    # Complete the central ABA outcome across all geometries before expanding the guards.
    for case in cases:
        for profile in profiles:
            if profile == "multi_pair" and case not in HISTORIES:
                continue
            arena = make_arena(roots, profile, case, rounds=args.rounds, domain=domain, donor_domain=donor_domain)
            for spec in specs:
                key = (spec["name"], spec["birth"])
                target = output / profile / case / f"{key[0]}-{key[1]}.npz"
                if target.is_file():
                    continue
                if spec["kind"] in ("uniform", "public_lookup"):
                    policy = _PublicPolicy(arena.motor_draw, learn=spec["kind"] == "public_lookup")
                else:
                    graph, rule = models[key]
                    policy = _GraphPolicy(graph, rule, arena, intervention=spec["intervention"], stochastic=not args.greedy)
                result, _ = rollout(arena, policy)
                _atomic_npz(target, payload(arena, result))
            print(profile, case, flush=True)
    report = summarize(output, protocol)
    _atomic_json(output / "summary.json", report)
    _atomic_json(evidence / "summary.json", compact_report(report))
    _atomic_json(evidence / "protocol.json", protocol)
    print(f"Summary: {evidence / 'summary.json'}", flush=True)


def _law_code(maps):
    if maps.shape[2] == 1:
        return np.array([f"A{a}B{b}" for a, b in maps[:, :, 0]])
    return np.array(["A" + "".join(str(bit) for bit in a) for a in maps[:, A]])


def _statistics(rows, metric, keep=None):
    """Aggregate roots equally and retain a within-birth worst mapping for each measure."""
    code = _law_code(rows[0]["maps"])
    if "mark_law" in rows[0] and metric.endswith(("/mark", "/both_modalities")):
        marks = np.array([f"mark{law}" for law in rows[0]["mark_law"]])
        code = marks if metric.endswith("/mark") else np.char.add(np.char.add(code, "_"), marks)
    if keep is None:
        keep = np.ones(len(code), bool)
    per_birth = {}
    for row in rows:
        values = row[metric][keep]
        strata = {mapping: float(values[code[keep] == mapping].mean()) for mapping in sorted(set(code[keep]))}
        per_birth[str(row["birth"])] = dict(mean=float(values.mean()), worst_mapping=min(strata.values()), by_map=strata)
    return dict(mean=float(np.mean([b["mean"] for b in per_birth.values()])),
                worst_within_birth=float(min(b["worst_mapping"] for b in per_birth.values())),
                by_birth=per_birth)


def _comparison(left, right, metric, draws, keep=None):
    pairs = list(zip(left, right, strict=True))
    if not pairs:
        return None
    for a, b in pairs:
        if (a["birth"] != b["birth"] or any(not np.array_equal(a[key], b[key]) for key in ("roots", "maps"))
                or not np.array_equal(a.get("mark_law"), b.get("mark_law"))):
            raise ValueError("Paired contrast must match both birth and world roots, retaining all law lanes")
    roots = left[0]["roots"]
    width = int(np.count_nonzero(roots == roots[0]))
    if keep is None:
        keep = np.ones(len(roots), bool)
    differences = np.stack([(a[metric].astype(float) - b[metric]).reshape(-1, width)
                            [:, keep[:width]].mean(axis=1) for a, b in pairs])
    births, world_count = differences.shape
    rng = _rng(births, world_count, BOOTSTRAP_DOMAIN)
    birth_samples = rng.integers(births, size=(draws, births))
    root_samples = rng.integers(world_count, size=(draws, world_count))
    replicated = differences[birth_samples[:, :, None], root_samples[:, None, :]].mean(axis=(1, 2))
    return dict(paired_difference=float(differences.mean()), pointwise_95=np.quantile(replicated, [.025, .975]).tolist(),
                by_birth={str(row["birth"]): float(differences[i].mean()) for i, row in enumerate(left)},
                births=births, roots=world_count)


def summarize(output, protocol):
    data = {}
    for profile in protocol["profiles"]:
        for case in protocol["cases"]:
            for spec in protocol["specs"]:
                birth, model = spec["birth"], spec["name"]
                path = output / profile / case / f"{model}-{birth}.npz"
                if not path.exists():
                    continue
                with np.load(path, allow_pickle=False) as raw:
                    kept = ("roots", "maps", "mark_law", "side", "motor_draw", "edit_draw", "writes", "observed_reward",
                            "good_side", "shuffle_audit")
                    prefixes = ("hit/", "success/", "side/", "right/", "changed_feedback_before/")
                    row = {name: raw[name] for name in raw.files if name in kept or name.startswith(prefixes)}
                    row["birth"] = birth
                    data.setdefault((profile, case, model), []).append(row)
    report = dict(split=protocol["split"], models={}, paired={}, audit={}, rounds=protocol["rounds_per_context_pair"],
                  n_roots=len(protocol["roots"]))
    for (profile, case, model), rows in data.items():
        group = report["models"].setdefault(model, {}).setdefault(profile, {})
        group[case] = cell = {}
        tally = rows[0]["maps"]
        conflict = np.any(tally[:, A] != tally[:, B], axis=1)
        for key in rows[0]:
            if key.startswith("success/") or key.startswith("hit/"):
                cell[key] = {"all": _statistics(rows, key)}
                if conflict.any():
                    cell[key]["conflict"] = _statistics(rows, key, conflict)
                if (~conflict).any():
                    cell[key]["agree"] = _statistics(rows, key, ~conflict)
            if key.startswith("changed_feedback_before/"):
                found, counts = np.unique(np.concatenate([row[key] for row in rows]), return_counts=True)
                cell[key] = {str(int(x)): int(n) for x, n in zip(found, counts, strict=True)}
        cell["writes_per_request"] = float(np.concatenate([row["writes"] for row in rows]).mean())
        cell["contact_all"] = float((np.concatenate([row["side"] for row in rows]) >= 0).mean())
        if "shuffle_audit" in rows[0]:
            audit = np.concatenate([row["shuffle_audit"].reshape(-1, len(SHUFFLE_AUDIT)) for row in rows])
            revealed = audit[:, 0] >= 0
            cell["address_shuffle"] = {name: float(column[revealed].mean()) if revealed.any() else None
                                       for name, column in zip(SHUFFLE_AUDIT, audit.T, strict=True)}
        if profile == "familiar" and case == "aba" and model in ("local_active", "public_lookup"):
            cell["goal_right_marginal"] = {key.removeprefix("right/"): float(value.mean())
                                           for key, value in rows[0].items() if key.startswith("right/")}

    for (profile, case, model), rows in data.items():
        if case == "aba":
            probes = report["paired"].setdefault(model, {}).setdefault(profile, {})
            comparisons = [(other, model) for other in HISTORIES[1:]]
            if model == "local_active":
                comparisons.extend(("aba", other) for other in ("local_active_disabled", "local_active_shuffle", "local_independent", "legacy_active"))
            for other_case, other_model in comparisons:
                other = data.get((profile, other_case, other_model))
                if other is None:
                    continue
                for left, right in zip(rows, other, strict=True):
                    for draw in ("motor_draw", "edit_draw"):
                        if not np.array_equal(left[draw], right[draw]):
                            raise AssertionError("Paired control lost shared exogenous draws")
                name = other_case if model == other_model else other_model
                conflict = np.any(rows[0]["maps"][:, A] != rows[0]["maps"][:, B], axis=1)
                probes[name] = {metric: dict(all=_comparison(rows, other, f"success/{metric}", 2500),
                                              conflict=_comparison(rows, other, f"success/{metric}", 2500, conflict))
                                for metric in PRIMARY}
                if name == "flipped":
                    for metric in ("A", "B"):
                        changed_action = np.concatenate([(a[f"side/{metric}"] != b[f"side/{metric}"])
                                                         & a[f"hit/{metric}"] & b[f"hit/{metric}"]
                                                         for a, b in zip(rows, other, strict=True)])
                        probes[name][metric]["opposite_physical_side_fraction"] = float(changed_action.mean())
        if case == "natural":
            stable = data.get((profile, "stable", model))
            if stable is None:
                continue
            same_first = all(np.array_equal(a["side/dose0"], b["side/dose0"]) for a, b in zip(rows, stable, strict=True))
            if not same_first:
                raise AssertionError("An unannounced reversal changed behavior before the first changed receipt")
            report["audit"].setdefault(model, {})[profile] = {"hidden_reversal_zero_feedback_identical_actions": True}
            probes = report["paired"].setdefault(model, {}).setdefault(profile, {})
            probes["natural_reversal_minus_stable"] = {f"dose{d}": _comparison(rows, stable, f"success/dose{d}", 2500)
                                                       for d in DOSES}
            aligned_zero = data.get((profile, "aligned_0", model))
            if aligned_zero is not None:
                aligned = {}
                for dose in DOSES[1:]:
                    branch = data.get((profile, f"aligned_{dose}", model))
                    if branch is None:
                        continue
                    zero_at_dose = [{**row, f"success/dose{dose}": row["success/dose0"]} for row in aligned_zero]
                    aligned[f"dose{dose}"] = _comparison(branch, zero_at_dose, f"success/dose{dose}", 2500)
                probes["aligned_reversal_improvement_over_zero"] = aligned
    return report


def compact_report(report):
    """Retain outcome magnitudes, worst maps and paired evidence alongside the full raw summary."""
    result = {key: value for key, value in report.items() if key not in ("models", "paired")}
    result["models"] = models = {}
    result["paired"] = paired = {}
    result["statistics"] = ("Matched roots and initial graphs are the sampling units. Worst values minimize over map "
                            "within birth and then over births. Paired intervals are pointwise crossed bootstrap draws.")
    for model, profiles in report["models"].items():
        models[model] = model_cells = {}
        for profile, cases in profiles.items():
            model_cells[profile] = profile_cells = {}
            for case, cells in cases.items():
                profile_cells[case] = selected = {}
                for key, value in cells.items():
                    if key.startswith("success/"):
                        selected[key] = {stratum: {m: bit[m] for m in ("mean", "worst_within_birth")}
                                         for stratum, bit in value.items() if stratum != "agree"}
                    elif (key in {"contact_all", "writes_per_request", "address_shuffle", "goal_right_marginal"}
                          or key.startswith("changed_feedback_before/")):
                        selected[key] = value
    for model, profiles in report["paired"].items():
        paired[model] = model_cells = {}
        for profile, cases in profiles.items():
            model_cells[profile] = profile_cells = {}
            for case, metrics in cases.items():
                profile_cells[case] = selected = {}
                for key, value in metrics.items():
                    if key.startswith("dose"):
                        selected[key] = {k: value[k] for k in ("paired_difference", "pointwise_95")}
                    else:
                        selected[key] = {k: {stat: value[k][stat] for stat in ("paired_difference", "pointwise_95")}
                                         for k in ("all", "conflict")}
                        if "opposite_physical_side_fraction" in value:
                            selected[key]["opposite_physical_side_fraction"] = value["opposite_physical_side_fraction"]
    return result


def joint_by_birth(output, *, model="local_active"):
    """Read actual conflict outcomes from the official familiar same-state ABA probes."""
    manifest = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    specs = [spec for spec in manifest["specs"] if spec["name"] == model]
    if not specs:
        raise ValueError(f"No registered {model} arm in {output}")
    case = output / "familiar" / "aba"
    with np.load(case / "uniform-0.npz", allow_pickle=False) as reference:
        matched = {key: reference[key] for key in ("roots", "maps", "mark_law", "motor_draw", "edit_draw")}
        conflict = np.any(matched["maps"][:, A] != matched["maps"][:, B], axis=1)
        uniform = float(reference["success/both_modalities"][conflict].mean())
    records = []
    for spec in specs:
        birth = spec["birth"]
        with np.load(case / f"{model}-{birth}.npz", allow_pickle=False) as raw:
            if not all(np.array_equal(raw[key], value) for key, value in matched.items()):
                raise ValueError(f"Uniform comparison changed worlds, laws or exogenous draws for birth {birth}")
            serial = np.logical_and.reduce([raw[f"serial_side/{label}"] == raw[f"right/{label}"]
                                            for label in ("A", "B", "mark")])
            scores = {label: float(raw[f"success/{label}"][conflict].mean())
                      for label in (*PRIMARY, "both_modalities")}
            records.append(dict(birth=birth, trials=int(conflict.sum()), scores=scores, uniform_joint=uniform,
                                serial_joint=float(serial[conflict].mean())))
    return records


def joint_table(runs):
    """Format matched-world conflict results for fixed endpoints across training families."""
    batches = [(label, joint_by_birth(output)) for label, output in runs]
    trial_counts = sorted({row["trials"] for _, rows in batches for row in rows})
    counts = "、".join(str(n) for n in trial_counts)
    lines = ["# 逐出生的同快照联合成功率", "",
             f"熟悉气味的 A→B 留出协议，保留 A、B 口味冲突的世界。每个出生有 {counts} 条实地冲突轨迹。"
             "A、B 与标记均按各自静默分叉的实际接触侧计分。均匀基准使用同一批世界、规律和运动随机数。"
             "连续三项使用同一物理生命的一段随机排序静默尾部。", "",
             "| 训练系列 | 出生 | A | B | A∧B | 标记 | A∧B∧标记 | 均匀三项 | 差（百分点） | 连续三项 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, rows in batches:
        for row in rows:
            scores = row["scores"]
            probabilities = [scores[k] for k in (*PRIMARY, "both_modalities")] + [row["uniform_joint"]]
            cells = [label, str(row["birth"]), *(f"{p:.2%}" for p in probabilities),
                     f"{100 * (scores['both_modalities'] - row['uniform_joint']):+.2f}", f"{row['serial_joint']:.2%}"]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("pilot", "main", "replication"), default="pilot")
    parser.add_argument("--roots", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--profiles", choices=PROFILES, default=PROFILES, nargs="+")
    parser.add_argument("--cases", choices=(*HISTORIES, "stable", "natural", *(f"aligned_{d}" for d in DOSES)),
                        default=(*HISTORIES, "stable", "natural", *(f"aligned_{d}" for d in DOSES)), nargs="+")
    parser.add_argument("--old", action="store_true")
    parser.add_argument("--new", type=Path)
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--births", type=int, nargs="+", default=[731, 732, 733, 734, 735, 736])
    parser.add_argument("--arms", nargs="+", default=["local_active", "local_independent", "legacy_active"])
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    execute(args)


if __name__ == "__main__":
    main()
