"""Evidence-conditioned strength for the frozen relation learner's covariance."""

from dataclasses import dataclass

import numpy as np

from .relation_learner import RelationLearner, _DIM, _SLOTS, _logistic, _unit


IDENTITY = np.eye(_DIM, dtype=np.float32)
FEATURES = ("compatibility", "directional_energy", "displacement_energy",
            "sensory_prediction", "signed_surprise")


@dataclass(frozen=True, eq=False)
class Gate:
    """The only trained change is a scalar multiplier of existing sharing."""

    mode: str
    weights: tuple = ()

    def __post_init__(self):
        lengths = {"always": 0, "local": 0, "explicit": 0, "constant": 1, "learned": 9}
        if self.mode not in lengths or len(self.weights) != lengths[self.mode]:
            raise ValueError("Unknown gate mode or incorrect number of weights")
        if not np.isfinite(self.weights).all():
            raise ValueError("Weights must be finite")
        if self.mode == "constant" and not 0 <= self.weights[0] <= 1:
            raise ValueError("A constant gate must be between zero and one")

    def amount(self, evidence, memory):
        if self.mode == "always":
            return np.ones(len(evidence), dtype=np.float32)
        if self.mode == "local":
            return np.zeros(len(evidence), dtype=np.float32)
        if self.mode == "constant":
            return np.full(len(evidence), self.weights[0], dtype=np.float32)
        if self.mode == "explicit":
            # Generic angular agreement with the prior's action on this change.
            return np.clip((memory[:, 0] - .65) / .25, 0., 1.)
        w = np.asarray(self.weights, dtype=np.float32)
        current = evidence - np.array([.8, 1., .5, .5, .5], dtype=np.float32)
        recent = memory[:, [0, 1, 4]] - np.array([.8, 1., .5], dtype=np.float32)
        return _logistic(w[0] + current @ w[1:6] + recent @ w[6:9])


def covariance_of_others(matrix, rows, slots):
    maps = matrix[rows, :, :_DIM, :_DIM]
    other = (np.arange(_SLOTS)[None, :] != slots[:, None])[:, :, None, None]
    sources = maps * other
    return np.einsum("nsji,nsjk->nik", sources, sources, optimize=False)


def read_prediction(rule, matrix, previous, odor, last_taste, adjacent):
    predicted = previous + np.einsum("ni,nij->nj", previous, matrix[:, :_DIM, :_DIM], optimize=False)
    overlap = np.einsum("nsi,ni->ns", odor, _unit(predicted), optimize=False)
    relation = matrix[:, _DIM, _DIM] * last_taste
    relation *= adjacent
    affinity = .5 + .5 * np.clip(overlap, -1., 1.)
    return np.clip(rule.read_scale * relation[:, None] * affinity, -1., 1.)


def update(rule, gate, current, covariance, previous, sensed, sign, taste, prediction, memory, count,
           override=None):
    """One write from a signed, adjacent, same-address pair and prior forecasts."""
    trace = np.trace(covariance, axis1=1, axis2=2)
    displacement = sensed - previous
    predicted = previous + np.einsum("ni,nij->nj", previous, current[:, :_DIM, :_DIM], optimize=False)
    error = taste - prediction
    prior_direction = np.einsum("ni,nij->nj", displacement, covariance, optimize=False)
    energy = np.sum(displacement ** 2, axis=1)
    supported = np.sum(displacement * prior_direction, axis=1)
    agreement = supported / np.maximum(np.sqrt(energy) * np.linalg.norm(prior_direction, axis=1), 1e-8)
    directional = _DIM * supported / np.maximum(energy * trace, 1e-8)
    sensory_match = (_unit(predicted) * sensed).sum(axis=1)
    evidence = np.stack((agreement, directional, .25 * energy,
                         .5 * (1. + sensory_match), .5 * error * sign), axis=1)
    memory = np.where(count[:, None] == 0, evidence, .65 * memory + .35 * evidence)
    amount = gate.amount(evidence, memory)
    if override is not None:
        amount = np.where(np.isfinite(override), override, amount)

    # Keep both branches of the previously frozen metric and the entire
    # downstream write identical. Only its existing prior strength is scaled.
    strength = np.asarray(rule.prior_strength * amount, dtype=np.float32)
    prior_weight = strength * trace / (.25 + trace)
    metric = (1. - prior_weight)[:, None, None] * IDENTITY
    metric += (_DIM * strength / (.25 + trace))[:, None, None] * covariance
    adjusted = np.einsum("ni,nij->nj", displacement, metric, optimize=False)
    inputs = np.stack(((previous * sensed).sum(axis=1), sensory_match,
                       (_unit(displacement) * _unit(adjusted)).sum(axis=1), error * sign), axis=1)
    map_rate, taste_rate = rule.write_rates(inputs)
    source = np.einsum("ni,nij->nj", previous, metric, optimize=False)
    norm = np.maximum((source * previous).sum(axis=1), .2)
    delta = adjusted - (predicted - previous)
    paid = (map_rate / norm)[:, None] * source
    current[:, :_DIM, :_DIM] *= .98
    current[:, :_DIM, :_DIM] += paid[:, :, None] * delta[:, None, :]
    np.clip(current[:, :_DIM, :_DIM], -2., 2., out=current[:, :_DIM, :_DIM])
    current[:, _DIM, _DIM] = np.clip(.82 * current[:, _DIM, _DIM] + taste_rate * error * sign, -1., 1.)
    return current, memory, amount, evidence


class TrustLearner(RelationLearner):
    """Live gate driven exclusively through the inherited sensory/contact path."""

    def __init__(self, front, seeds, *, motion, rule, gate, schedule=None):
        super().__init__(front, seeds, motion=motion, rule=rule)
        self.gate = gate
        self.schedule = {} if schedule is None else schedule

    def reset(self, n):
        super().reset(n)
        self.evidence = np.zeros((n, _SLOTS, len(FEATURES)), dtype=np.float32)
        self.evidence_count = np.zeros((n, _SLOTS), dtype=np.int32)
        self.trust = np.full((n, _SLOTS), np.nan, dtype=np.float32)
        self.gate_trace = {}

    def _write(self, rows, slots, sensed, taste, sides):
        covariance = covariance_of_others(self.matrix, rows, slots)
        override = self.schedule.get(self.round)
        if override is not None:
            override = override[rows]
        result = update(self.rule, self.gate, self.matrix[rows, slots], covariance,
                        self.last_odor[rows, slots], sensed, self.last_taste[rows, slots], taste,
                        self.prediction[rows, sides], self.evidence[rows, slots],
                        self.evidence_count[rows, slots], override)
        matrix, memory, amounts, evidence = result
        self.matrix[rows, slots] = matrix
        self.evidence[rows, slots] = memory
        self.evidence_count[rows, slots] += 1
        self.trust[rows, slots] = amounts
        self.matrix_updates[rows] += 1
        if self.round not in self.gate_trace:
            self.gate_trace[self.round] = (np.full(self.n, np.nan, np.float32),
                                           np.full(self.n, np.nan, np.float32),
                                           np.zeros(self.n, bool))
        weight, compatibility, mask = self.gate_trace[self.round]
        weight[rows], compatibility[rows], mask[rows] = amounts, evidence[:, 0], True
