"""Keep a modular circuit alive across physical requests and edit its support causally."""

from dataclasses import dataclass

import numpy as np

from .free_graph_model import strongly_connected


@dataclass(frozen=True)
class Schedule:
    growth_every: int
    retirement_every: int
    decay: float


WIDTH = 32


def expit(values):
    return 1 / (1 + np.exp(-np.clip(values, -40, 40)))


class Population:
    """Batched lives. Deliberately has no access to the physical environment or task IDs."""

    def __init__(self, parameters, masks, schedule, random_seeds):
        self.parameters = parameters
        self.mask = masks.copy()
        self.initial = masks.copy()
        self.schedule = schedule
        self.rng = [np.random.default_rng(seed) for seed in random_seeds]
        self.state = np.zeros((len(masks), WIDTH), dtype=np.float32)
        self.receive_count = np.zeros(len(masks), dtype=np.int32)
        self.grow_scores = np.zeros_like(masks, dtype=np.float32)
        self.retire_scores = np.zeros_like(self.grow_scores)
        self.no_loops = ~np.eye(WIDTH, dtype=bool)
        self.refresh_matrices()

    def refresh_matrices(self):
        self.degree = self.mask.sum(axis=-1, dtype=np.float32)
        self.weight = self.parameters["gain"] / np.sqrt(self.degree)
        self.matrix = self.mask * self.weight[..., None]

    def begin_request(self):
        self.arc_trace = np.zeros_like(self.grow_scores)
        self.common_trace = np.zeros_like(self.state)
        self.logit_sum = np.zeros(len(self.state), dtype=np.float32)
        self.trace_ticks = np.zeros_like(self.logit_sum)

    def act(self, public_input, motor_active=None):
        p = self.parameters
        drive = np.einsum("bi,bij->bj", public_input, p["drive"]) + p["bias"]
        candidate_input, gate_input = np.split(drive, 2, axis=-1)
        incoming = np.matmul(self.matrix, self.state[..., None])[..., 0]
        candidate = np.tanh(candidate_input + incoming)
        gate = expit(gate_input)
        prior = self.state
        self.state = (1 - gate) * candidate + gate * prior
        logit = (p["readout"] * self.state).sum(axis=1) + p["offset"]
        if motor_active is not None:
            derivative = (1 - gate) * (1 - candidate ** 2) * p["readout"] * motor_active[:, None]
            self.arc_trace += derivative[..., None] * prior[:, None, :]
            self.common_trace += derivative * incoming
            self.logit_sum += motor_active * logit
            self.trace_ticks += motor_active
        return logit >= 0

    def finish_request(self, contacted, shown):
        """Use a signed public receipt matched to its own realized side, or do nothing."""
        observed = (contacted >= 0) & (shown != 0)
        if np.any(observed & (self.trace_ticks == 0)):
            raise ValueError("A contact receipt requires a preceding movement phase")
        side = 2 * contacted - 1
        ticks = self.trace_ticks.clip(1)
        label_direction = side * shown
        credit = label_direction * expit(-label_direction * (self.logit_sum / ticks)) / ticks
        d = self.degree
        gain = self.parameters["gain"]
        higher = np.sqrt(d + 1)
        lower = np.sqrt(np.maximum(1, d - 1))
        add = (gain / higher)[..., None] * self.arc_trace
        add += ((np.sqrt(d) / higher - 1) * self.common_trace)[..., None]
        remove = (-gain / lower)[..., None] * self.arc_trace
        remove += ((np.sqrt(d) / lower - 1) * self.common_trace)[..., None]
        for score, change in ((self.grow_scores, add), (self.retire_scores, remove)):
            decay = self.schedule.decay
            score[observed] = decay * score[observed] + (1 - decay) * (credit[:, None, None] * change)[observed]
        self.receive_count += observed
        growth = observed & (self.receive_count % self.schedule.growth_every == 0)
        retirement = observed & (self.receive_count % self.schedule.retirement_every == 0)
        return self._edit(growth, retirement, random=False)

    def random_edit(self, opportunities):
        """Replay a just-completed donor edit schedule with randomized endpoints."""
        return self._edit(opportunities[:, 0], opportunities[:, 1], random=True)

    def _edit(self, growth, retirement, *, random):
        events = np.full((len(self.state), 2), -1, dtype=np.int16)
        for lane in np.flatnonzero(growth | retirement):
            mask = self.mask[lane]
            if growth[lane]:
                allowed = self.no_loops & ~mask
                if random:
                    available = np.flatnonzero(allowed)
                    pick = self.rng[lane].choice(available) if len(available) else -1
                else:
                    scores = np.where(allowed, self.grow_scores[lane], -np.inf)
                    pick = scores.argmax()
                    if scores.flat[pick] <= 0:
                        pick = -1
                if pick >= 0:
                    mask.flat[pick] = True
                    events[lane, 0] = pick
            if retirement[lane]:
                # Check minimum degree first, then the whole graph on the candidate removal.
                allowed = mask & (mask.sum(axis=0) > 1)[None, :] & (mask.sum(axis=1) > 1)[:, None]
                choices = np.flatnonzero(allowed)
                if random:
                    choices = self.rng[lane].permutation(choices)
                else:
                    scores = self.retire_scores[lane].ravel()[choices]
                    choices = choices[np.argsort(-scores, kind="stable")]
                    if not len(scores) or scores.max() <= 0:
                        choices = choices[:0]
                    else:
                        choices = choices[self.retire_scores[lane].ravel()[choices] > 0]
                # On joint boundaries, wait a request before pruning any new arc.
                for pick in choices:
                    if pick == events[lane, 0]:
                        continue
                    mask.flat[pick] = False
                    if strongly_connected(WIDTH, zip(*np.nonzero(mask.T), strict=True)):
                        events[lane, 1] = pick
                        break
                    mask.flat[pick] = True
            # Old scores for edited targets use a different incoming normalization.
            for pick in events[lane]:
                if pick >= 0:
                    self.grow_scores[lane, pick // WIDTH] = 0
                    self.retire_scores[lane, pick // WIDTH] = 0
        if (events >= 0).any():
            self.refresh_matrices()
        return events
