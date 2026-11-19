"""A fast relation matrix trained by feedback on physically reached odors.

The inherited front and actuator run at every tick. The readout predicts one
continuous relation: a displacement of the last tasted sensory vector and a
signed change in taste. Both foods are evaluated against this same prediction.
Each life starts with zero relation matrices. The shared outer rule has 32
parameters and gates the two rank-one updates made from adjacent contacts.
"""

from dataclasses import dataclass

import numpy as np

from . import context_local, habitat_deploy, lifelong_agent, mixture_world as food, world


_DIM = food.DIM
_SLOTS = 6


def _unit(activity):
    return activity / np.maximum(np.linalg.norm(activity, axis=-1, keepdims=True), 1e-8)


def _logistic(x):
    return 1. / (1. + np.exp(-np.clip(x, -20., 20.)))


@dataclass(frozen=True, eq=False)
class Rule:
    """Shared outer parameters for the 4×4×2 feedback network and two gains."""

    weights: np.ndarray

    def __post_init__(self):
        weights = np.array(self.weights, dtype=np.float32, copy=True)
        if weights.shape != (32,) or not np.isfinite(weights).all():
            raise ValueError("Expected 32 finite outer parameters")
        weights.setflags(write=False)
        object.__setattr__(self, "weights", weights)

    @classmethod
    def from_vector(cls, weights):
        return cls(weights)

    @staticmethod
    def default_vector():
        weights = np.random.default_rng(49321).normal(0., .12, 32).astype(np.float32)
        weights[28:] = [.4, -.1, .65, 0.]
        return weights

    @staticmethod
    def mutation_scales():
        scales = np.full(32, .25, dtype=np.float32)
        scales[-4:] = .35
        return scales

    def vector(self):
        return self.weights.copy()

    def write_rates(self, pre_activity):
        w = self.weights
        hidden = np.tanh(pre_activity @ w[:16].reshape(4, 4) + w[16:20])
        rate = _logistic(hidden @ w[20:28].reshape(4, 2) + w[28:30])
        return 1.1 * rate[:, 0], 1.25 * rate[:, 1]

    @property
    def prior_strength(self):
        return float(_logistic(self.weights[30]))

    @property
    def read_scale(self):
        return float(.25 + 1.5 * _logistic(self.weights[31]))


class RelationLearner(habitat_deploy.Forager):
    """Use only the pulsed sensory ports, reached side, and received taste.

    A prototype table addresses the fast tensor using the public background.
    Every block stores a displacement map and one taste coupling. The maps of
    the other addresses supply a graded covariance prior to a new write. This
    allows experience to transfer without imposing a rank or a transformation.
    The last real contact per address is a bounded eligibility trace. A trace
    is eligible only on the immediately following request. Silence never writes.
    """

    def __init__(self, front, seeds, *, motion, rule, matrix_writes=True, share_prior=True):
        super().__init__(front, {}, seeds, start=-1, motor=motion)
        self.rule = rule
        self.matrix_writes = matrix_writes
        self._prior_strength = rule.prior_strength if share_prior else 0.
        self._read_scale = rule.read_scale
        receptors = front["sensors"][:_DIM]
        ground, _ = front["ground"]
        # Normalize the units of the frozen receptor decoder without rotating
        # them to any world-defined directions. Only the published front enters.
        _, scales, axes = np.linalg.svd(receptors @ front["matrix"] @ ground, full_matrices=False)
        self._whitening = ((axes.T / np.maximum(scales, 1e-6)) @ axes).astype(np.float32)

    def reset(self, n):
        if n != len(self.seeds):
            raise ValueError("One seeded actuator per life is required")
        self.n = n
        self.it = 0
        self.round = -1
        self.rng = [np.random.default_rng(np.random.SeedSequence([s, 745930])) for s in self.seeds]
        self.noise = np.empty((0, n, 2), dtype=np.float32)
        self.h = np.zeros((n, 2, self.perception["matrix"].shape[0]), dtype=np.float32)
        self.odor_background = np.zeros_like(self.h)
        self.odor_pending = np.zeros(n, dtype=bool)
        self.motor = np.zeros((n, 2), dtype=np.float32)
        self.mv = np.zeros_like(self.motor)
        self.places = context_local.Addresses(n, _SLOTS, .09)
        self.matrix = np.zeros((n, _SLOTS, _DIM + 1, _DIM + 1), dtype=np.float32)
        self.last_odor = np.zeros((n, _SLOTS, _DIM), dtype=np.float32)
        self.last_taste = np.zeros((n, _SLOTS), dtype=np.float32)
        self.last_round = np.full((n, _SLOTS), -2, dtype=np.int32)
        self.odor = np.zeros((n, 2, _DIM), dtype=np.float32)
        self.odor_seen = np.zeros(n, dtype=bool)
        self.context_seen = np.zeros(n, dtype=bool)
        self.prediction = np.zeros((n, 2), dtype=np.float32)
        self.receipts = np.zeros(n, dtype=np.int32)
        self.matrix_updates = np.zeros(n, dtype=np.int32)

    def snapshot_matrix(self):
        """Return an independent copy of the fast synapses for every life."""
        return self.matrix.copy()

    def restore_matrix(self, snapshot):
        """Overwrite only fast synapses, including source maps for transfer."""
        snapshot = np.asarray(snapshot, dtype=self.matrix.dtype)
        if snapshot.shape != self.matrix.shape or not np.isfinite(snapshot).all():
            raise ValueError("Incompatible fast matrix snapshot")
        np.copyto(self.matrix, snapshot)

    def _learn(self, obs):
        taste = obs[:, world.REWARD_POS] - obs[:, world.REWARD_NEG]
        contact = obs[:, world.CONTACT_ID]
        rows = np.flatnonzero((taste != 0) & contact.any(axis=1) & self.odor_seen & self.context_seen)
        if not rows.size:
            return
        slots = self.places.current[rows]
        sides = contact[rows].argmax(axis=1)
        sensed = self.odor[rows, sides]
        signed = taste[rows]
        eligible = self.last_round[rows, slots] == self.round - 1
        if self.matrix_writes and eligible.any():
            self._write(rows[eligible], slots[eligible], sensed[eligible], signed[eligible], sides[eligible])
        self.last_odor[rows, slots] = sensed
        self.last_taste[rows, slots] = signed
        self.last_round[rows, slots] = self.round
        self.receipts[rows] += 1

    def _write(self, rows, slots, sensed, taste, sides):
        previous = self.last_odor[rows, slots]
        sign = self.last_taste[rows, slots]
        current = self.matrix[rows, slots]
        maps = self.matrix[rows, :, :_DIM, :_DIM]
        other = (np.arange(_SLOTS)[None, :] != slots[:, None])[:, :, None, None]
        sources = maps * other
        covariance = np.einsum("nsji,nsjk->nik", sources, sources, optimize=False)
        trace = np.trace(covariance, axis1=1, axis2=2)
        prior_weight = self._prior_strength * trace / (.25 + trace)
        metric = (1. - prior_weight)[:, None, None] * np.eye(_DIM, dtype=np.float32)
        metric += (_DIM * self._prior_strength / (.25 + trace))[:, None, None] * covariance

        displacement = sensed - previous
        predicted = previous + np.einsum("ni,nij->nj", previous, current[:, :_DIM, :_DIM], optimize=False)
        error = taste - self.prediction[rows, sides]
        adjusted = np.einsum("ni,nij->nj", displacement, metric, optimize=False)
        inputs = np.stack(((previous * sensed).sum(axis=1), (_unit(predicted) * sensed).sum(axis=1),
                           (_unit(displacement) * _unit(adjusted)).sum(axis=1), error * sign), axis=1)
        map_rate, taste_rate = self.rule.write_rates(inputs)
        source = np.einsum("ni,nij->nj", previous, metric, optimize=False)
        norm = np.maximum((source * previous).sum(axis=1), .2)
        delta = adjusted - (predicted - previous)
        paid = (map_rate / norm)[:, None] * source
        current[:, :_DIM, :_DIM] *= .98
        current[:, :_DIM, :_DIM] += paid[:, :, None] * delta[:, None, :]
        np.clip(current[:, :_DIM, :_DIM], -2., 2., out=current[:, :_DIM, :_DIM])
        # Local signed error drives the value attached to this relation. Use
        # the saved prediction for the reached side, even across blank ticks.
        current[:, _DIM, _DIM] = np.clip(.82 * current[:, _DIM, _DIM] + taste_rate * error * sign, -1., 1.)
        self.matrix[rows, slots] = current
        self.matrix_updates[rows] += 1

    def _sense(self, obs):
        front = self.perception
        context = obs[:, food.CTX]
        scents = obs[:, food.LEFT.start:food.RIGHT.stop].reshape(self.n, 2, _DIM) * 2. - 1.
        recurrent = front["matrix"]
        background_drive = (context @ front["sensors"][_DIM:])[:, None, :]
        free_activity = self.h @ recurrent + background_drive
        self.h = free_activity + scents @ front["sensors"][:_DIM]

        ctx_rows = np.flatnonzero(context.any(axis=1))
        if ctx_rows.size:
            self.places.register(ctx_rows, context[ctx_rows])
            self.context_seen[ctx_rows] = True
        ready = np.flatnonzero(self.odor_pending)
        if ready.size:
            # Read after one recurrent step. Subtract the graph's response to
            # its previous state and background alone to isolate this pulse.
            ground, _ = front["ground"]
            reference = self.odor_background[ready] @ recurrent + background_drive[ready]
            self.odor[ready] = _unit(((self.h[ready] - reference) @ ground) @ self._whitening)
            self.odor_seen[ready] = True
        self.odor_pending[:] = scents.any(axis=(1, 2))
        self.odor_background[self.odor_pending] = free_activity[self.odor_pending]

    def _read(self, active):
        rows = np.flatnonzero(self.odor_seen & self.context_seen & active)
        if not rows.size:
            return
        slots = self.places.current[rows]
        previous = self.last_odor[rows, slots]
        matrix = self.matrix[rows, slots]
        predicted = previous + np.einsum("ni,nij->nj", previous, matrix[:, :_DIM, :_DIM], optimize=False)
        overlap = np.einsum("nsi,ni->ns", self.odor[rows], _unit(predicted), optimize=False)
        relation = matrix[:, _DIM, _DIM] * self.last_taste[rows, slots]
        relation *= (self.last_round[rows, slots] == self.round - 1)
        affinity = .5 + .5 * np.clip(overlap, -1., 1.)
        self.prediction[rows] = np.clip(self._read_scale * relation[:, None] * affinity, -1., 1.)

    def act(self, obs):
        if np.all(obs[:, world.TRIAL_START]):
            self.round += 1
            self.odor_seen.fill(False)
            self.odor_pending.fill(False)
            self.context_seen.fill(False)
            self.prediction.fill(0.)
        self._sense(obs)
        self._learn(obs)
        active = obs[:, world.ACTIVE]
        self._read(active > 0)

        tonic, gain, leak, noise, excite, oppose, confidence = self.motion
        diff = self.prediction - self.prediction[:, ::-1]
        drive = (tonic + confidence * np.abs(diff)) * active[:, None] - .85 * (1. - active[:, None])
        drive += gain * np.tanh(diff) + noise * self._noise() + excite * self.motor - oppose * self.motor[:, ::-1]
        self.mv = np.maximum(-1., leak * self.mv + drive)
        self.motor = (self.mv >= 1.).astype(np.float32)
        self.mv *= 1. - self.motor
        return self.motor


def birth(seeds, rule=None, *, mode="adaptive", checkpoint=lifelong_agent.DEFAULT):
    if mode == "local-selective":
        return lifelong_agent.new_lives(seeds, checkpoint, address="local-selective")
    if mode not in ("adaptive", "frozen", "local-only"):
        raise ValueError(f"Unknown relation learner mode {mode!r}")
    if rule is None:
        rule = Rule.from_vector(Rule.default_vector())
    front, _, motion, _ = lifelong_agent.load(checkpoint)
    return RelationLearner(front, seeds, motion=motion, rule=rule,
                           matrix_writes=mode != "frozen", share_prior=mode != "local-only")
