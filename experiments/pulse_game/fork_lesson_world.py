"""Two tasted objects, two marked doors, and one delayed consequence."""

from dataclasses import dataclass, replace

import numpy as np
import torch
from torch.nn import functional as F

from .native_rehearsal_trial import LIMIT, REGION, State, TrialGraph


PORTS = 10
CUE_SCALE = 1.35
MIN_DIRECTION_X = 0.45
SUPPORT_GAP_TICKS = 2
APPROACH_TICKS = 3
TRAVERSE_TICKS = 2
TASTE_TICKS = 2
SUPPORT_TICKS = 2 * (SUPPORT_GAP_TICKS + APPROACH_TICKS + TRAVERSE_TICKS + TASTE_TICKS)
CUE_TICKS = 3
BLANK_TICKS = 6
DELAY_TICKS = 6


class ForkGraph(TrialGraph):
    """The main region forecasts and steers through its own native units."""

    def __init__(self, birth, kind="trial"):
        super().__init__(birth, kind)

    def forecast(self, state_or_activity):
        activity = state_or_activity.h if isinstance(state_or_activity, State) else state_or_activity
        return activity[..., 12:14]

    def tick(self, s, x, *, feedback_override=None, disable_feedback=False, candidate_override=None,
             read_override=None, birth_only=False, adoption_off=False, outbound_override=None,
             disable_outbound=False, return_outgoing=False):
        # Keep the parent's shared intra-region weights and its live candidate/adoption rule.
        # Express each physical bridge as a separate term so it can be replaced at its receiver.
        n = REGION
        primary, shadow = s.h.split(n, dim=1)
        main_weights = self.weight * self.mask[:n]
        intra = main_weights[:, :n]
        main_fast = s.fast[:, :n, :n]

        def fast_current(weights, activity):
            return torch.bmm(weights, activity[:, :, None]).squeeze(2)

        back = F.linear(shadow, main_weights[:, n:]) + fast_current(s.fast[:, :n, n:], shadow)
        outward = (F.linear(primary, self.return_weight * self.mask[n:, :n])
                   + fast_current(s.fast[:, n:, :n], primary))
        if feedback_override is not None or disable_feedback:
            back = torch.zeros_like(back) if feedback_override is None else feedback_override
        if outbound_override is not None or disable_outbound:
            outward = torch.zeros_like(outward) if outbound_override is None else outbound_override

        projected = s.candidate if candidate_override is None else candidate_override
        enabled = self.read_candidate if read_override is None else read_override
        estimated_fast = (main_fast * (not birth_only) + projected * enabled).clamp(-LIMIT, LIMIT)
        primary_current = F.linear(primary, intra) + fast_current(main_fast, primary) + back
        shadow_current = F.linear(shadow, intra) + fast_current(estimated_fast, shadow) + outward
        current = torch.cat((primary_current, shadow_current), dim=1)
        target = torch.tanh(current + self.bias + F.pad(x, (0, self.width - PORTS)))
        h = s.h + self.leak_logit.sigmoid() * (target - s.h)
        trace = s.trace + self.keep_logit.neg().sigmoid() * (s.h - s.trace)

        shadow_now = h[:, n:]
        receiver = self.adopt_bias + self.adopt_receiver * shadow_now
        sender = self.adopt_sender * shadow_now
        adoption = (receiver[:, :, None] + sender[:, None, :]).sigmoid()
        accepted = s.candidate * adoption * (not adoption_off)
        mod = F.linear(h, self.gate * self.mask).tanh()
        post = (h + self.post_bias).tanh() * mod * self.edit_rate.sigmoid()
        new = post[:, :, None] * trace[:, None, :] * self.mask
        proposed = torch.cat((torch.cat((accepted, new[:, :n, n:]), dim=2),
                              F.pad(new[:, n:, :n], (0, n))), dim=1)
        fast = (s.fast + proposed).clamp(-LIMIT, LIMIT)
        updated = State(h, fast, trace, new[:, :n, :n])
        return (updated, back, outward) if return_outgoing else (updated, back)


@dataclass(frozen=True)
class Episode:
    vectors: torch.Tensor
    values: torch.Tensor
    first: torch.Tensor
    door_order: torch.Tensor
    flips: torch.Tensor
    draws: torch.Tensor
    approach_sides: torch.Tensor


@dataclass(frozen=True)
class Visit:
    object_id: torch.Tensor
    cue: torch.Tensor
    side: torch.Tensor
    payoff: torch.Tensor
    taste: torch.Tensor
    inputs: torch.Tensor


@dataclass(frozen=True)
class Choice:
    logit: torch.Tensor
    side: torch.Tensor
    immediate: torch.Tensor
    score: torch.Tensor


@dataclass(frozen=True)
class Outcome:
    payoff: torch.Tensor
    score: torch.Tensor
    taste: torch.Tensor


def scenes(seed, count, *, device="cpu"):
    """Draw two recognizable directions and randomize each role and private value."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(MIN_DIRECTION_X, 1., size=(count, 2))
    x[:, 1] *= -1
    azimuth = rng.uniform(0., 2 * np.pi, size=(count, 2))
    radius = np.sqrt(1 - x ** 2)
    vectors = CUE_SCALE * np.stack((x, radius * np.cos(azimuth), radius * np.sin(azimuth)), axis=-1)
    first_door = rng.integers(2, size=count)

    def tensor(value, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=device)

    return Episode(
        vectors=tensor(vectors),
        values=tensor(rng.choice((-1., 1.), size=(count, 2))),
        first=tensor(rng.integers(2, size=count), torch.long),
        door_order=tensor(np.stack((first_door, 1 - first_door), axis=1), torch.long),
        flips=tensor(rng.choice((-1., 1.), size=(count, 2))),
        draws=tensor(rng.random(size=count)),
        approach_sides=tensor(rng.integers(2, size=(count, 2)), torch.long),
    )


def _pulse(vector):
    return vector.new_zeros((len(vector), PORTS))


def _taste(payoff):
    return torch.stack((payoff.relu(), (-payoff).relu()), dim=1)


def _contact(model, state, e, obj, side):
    rows = torch.arange(len(obj), device=obj.device)
    view = _pulse(e.vectors)
    view[:, :3] = e.vectors[rows, obj]
    state = model.advance(state, view, APPROACH_TICKS)

    # A single open guide rail leads to the object. The body starts at position zero
    # and must reach the target on the rail before the world emits contact or taste.
    position = torch.zeros_like(side)
    target = (2 * side - 1) * TRAVERSE_TICKS
    contacts = []
    for _ in range(TRAVERSE_TICKS):
        position = position + (target - position).sign()
        touched = position == target
        touch = view.clone()
        touch[:, 6:8] = F.one_hot(side, num_classes=2) * touched[:, None]
        state = model.advance(state, touch, 1)
        contacts.append(touch)

    payoff = e.values[rows, obj] * touched
    tastes = touch.clone()
    tastes[:, 8:10] = _taste(payoff)
    state = model.advance(state, tastes, TASTE_TICKS)
    inputs = torch.cat((view[None].expand(APPROACH_TICKS, -1, -1), torch.stack(contacts),
                        tastes[None].expand(TASTE_TICKS, -1, -1)))
    return state, Visit(obj, view[:, :3], side, payoff, tastes[:, 8:10], inputs)


def support(model, e, *, return_receipts=False):
    """Investigate the two objects in their encountered order from a fresh body state."""
    state = model.start(len(e.first))
    visits = []
    for rank, obj in enumerate((e.first, 1 - e.first)):
        state = model.advance(state, _pulse(e.vectors), SUPPORT_GAP_TICKS)
        state, visit = _contact(model, state, e, obj, e.approach_sides[:, rank])
        visits.append(visit)
    return (state, tuple(visits)) if return_receipts else state


def door_cues(e):
    """In the public menu, ports 0:3 and 3:6 show left and right object directions."""
    rows = torch.arange(len(e.first), device=e.first.device)
    vectors = e.vectors[rows[:, None], e.door_order]
    return torch.cat((vectors.flatten(start_dim=1), e.flips, e.flips.new_zeros((len(rows), 2))), dim=1)


def cue(model, state, e, *, show=True, cut=False, cut_outbound=False):
    pulse = door_cues(e) if show else _pulse(e.vectors)
    for _ in range(CUE_TICKS):
        state, _ = model.tick(state, pulse, disable_feedback=cut, disable_outbound=cut_outbound)
    return state


def choice_window(model, state, *, ticks=BLANK_TICKS, override=None, cut=False,
                  outbound_override=None, cut_outbound=False, return_outgoing=False):
    """Run on silence, with optional tick-aligned replacements at either bridge receiver."""
    batch = len(state.h)
    for incoming in (override, outbound_override):
        if incoming is not None and incoming.shape != (ticks, batch, REGION):
            raise ValueError(f"Bridge replacement must have shape {(ticks, batch, REGION)}")
    if ticks < 1:
        raise ValueError("Choice window must contain at least one tick")
    zero = _pulse(state.h)
    activities, returns, outgoing = [], [], []
    for tick in range(ticks):
        state, back, outward = model.tick(
            state, zero, feedback_override=None if override is None else override[tick],
            disable_feedback=cut, outbound_override=None if outbound_override is None else outbound_override[tick],
            disable_outbound=cut_outbound, return_outgoing=True,
        )
        activities.append(state.h)
        returns.append(back)
        if return_outgoing:
            outgoing.append(outward)
    results = (state, torch.stack(activities), torch.stack(returns))
    return (*results, torch.stack(outgoing)) if return_outgoing else results


def choose(model, state, e, *, greedy=False, side=None, exploration=0.):
    """The main motor ports commit the body to one side with zero immediate reward."""
    if not 0. <= exploration <= 1.:
        raise ValueError("Exploration must be a probability")
    logit = model.logit(state)
    probability = (1 - exploration) * logit.sigmoid() + exploration / 2
    if side is None:
        side = (logit > 0 if greedy else e.draws < probability.detach()).long()
    else:
        side = torch.as_tensor(side, dtype=torch.long, device=logit.device).expand_as(logit)
        if ((side < 0) | (side > 1)).any():
            raise ValueError("Door side must be 0 (left) or 1 (right)")
    score = torch.where(side.bool(), probability.log(), torch.log1p(-probability))
    return Choice(logit, side, torch.zeros_like(logit), score)


def commit(model, state, e, choice):
    """Cross the selected door, wait on either route, then taste its consequence."""
    threshold = _pulse(state.h)
    threshold[:, 6:8] = F.one_hot(choice.side, num_classes=2)
    state = model.advance(state, threshold, 1)
    state = model.advance(state, _pulse(state.h), DELAY_TICKS)

    rows = torch.arange(len(choice.side), device=choice.side.device)
    selected = e.door_order[rows, choice.side]
    payoff = e.values[rows, selected] * e.flips[rows, choice.side]
    terminal = _pulse(state.h)
    terminal[:, 8:10] = _taste(payoff)
    state = model.advance(state, terminal, TASTE_TICKS)
    return state, Outcome(payoff, choice.score, terminal[:, 8:10])


def oracle(e):
    """Offline reference for verifying forced traversals and episode distributions."""
    return e.values.gather(1, e.door_order) * e.flips


def swap_doors(e):
    return replace(e, door_order=e.door_order.flip(1), flips=e.flips.flip(1))
