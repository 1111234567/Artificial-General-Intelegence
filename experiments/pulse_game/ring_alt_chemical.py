"""Lifetime differentiation of unlabeled olfactory receptor populations."""

from dataclasses import dataclass, replace

import numpy as np

from .ring_world import ACTIVE, CTX, CUE, FRUIT, NEGATIVE, NEW, POSITIVE, REACH, TASTED


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)


@dataclass(frozen=True)
class ChemSpec:
    fruit_bank: str = "growing"
    cue_bank: str = "raw"
    capacity: int = 8
    cue_capacity: int = 8
    width: float = .22
    cue_width: float | None = None
    novelty: float = .97
    receptor_rate: float = .20
    retire_age: int = 4
    units: int = 64
    cov_rate: float = .14
    fruit_cov_rate: float = 0.
    rate: float = .15
    outcomes: str = "negative"
    exploration: float = .12
    bite_tolerance: float = .07

    def __post_init__(self):
        options = {"raw", "fixed", "competitive", "growing", "growth-fixed-writes", "reassign-fixed"}
        if self.fruit_bank not in options or self.cue_bank not in options:
            raise ValueError("unsupported receptor organization")
        if not 1 <= self.capacity <= 32 or not 1 <= self.cue_capacity <= 32:
            raise ValueError("receptor capacity must be between 1 and 32")
        if self.outcomes not in {"negative", "signed", "frozen"}:
            raise ValueError("unsupported outcome channel")
        if min(self.width, self.cue_width if self.cue_width is not None else self.width) < 0:
            raise ValueError("population width must be nonnegative")
        if not 0 < self.novelty <= 1 or not 0 <= self.receptor_rate <= 1 or self.fruit_cov_rate < 0:
            raise ValueError("invalid population response constants")


class ReceptorPopulation:
    """Local normalized competitive learning, with optional recruitment.

    A slot is a budget for one sensory branch. Slots are initially empty in
    growing mode and have no prescribed odor, cue, context, or reward label.
    Graded responses may overlap, so a bite can tag more than one output branch.
    """

    def __init__(self, mode, capacity, spec, rng):
        self.mode, self.capacity, self.spec = mode, capacity, spec
        self.dimension = 4 if mode == "raw" else capacity
        self.seed_weights = unit(rng.gamma(1.6, 1., (capacity, 4))).astype(np.float32)

    def reset(self, n):
        self.prototype = np.tile(self.seed_weights, (n, 1, 1))
        self.alive = np.full((n, self.capacity), self.mode not in {"growing", "growth-fixed-writes"}, dtype=bool)
        self.last = np.full((n, self.capacity), -20, dtype=np.int16)
        self.usage = np.zeros((n, self.capacity), dtype=np.float32)
        self.births = np.zeros(n, dtype=np.int32)
        self.retirements = np.zeros(n, dtype=np.int32)
        self.centroid_writes = np.zeros(n, dtype=np.int32)
        self.centroid_l2 = np.zeros(n, dtype=np.float64)
        self.response_participation = np.zeros(n, dtype=np.float64)
        self.response_count = np.zeros(n, dtype=np.int32)
        self.presentations = np.zeros(n, dtype=np.int32)

    def match(self, trace):
        if self.mode == "raw":
            return trace / np.maximum(trace.sum(axis=1, keepdims=True), 1e-8)
        sim = self.similarity(self.prototype, trace)
        np.copyto(sim, -4., where=~self.alive)
        if self.spec.width == 0:
            code = np.zeros_like(sim)
            code[np.arange(len(sim)), sim.argmax(axis=1)] = 1.
            code *= self.alive
        else:
            cut = sim.max(axis=1, keepdims=True) - self.spec.width
            code = np.maximum(sim - cut, 0.) * self.alive
            code /= np.maximum(code.sum(axis=1, keepdims=True), 1e-8)
        return code * (trace > 0).any(axis=1, keepdims=True)

    def similarity(self, incoming, trace):
        return np.einsum("nck,nk->nc", incoming, trace, optimize=False)

    def observe(self, trace, valid, trial, *, adapt=True):
        """Make one local contact update and return every replaced branch."""
        ids = np.flatnonzero(valid)
        if not ids.size:
            return ids, ids
        self.presentations[ids] += 1
        if self.mode == "raw":
            return ids[:0], ids[:0]
        samples = trace[ids]
        live = self.alive[ids]
        sim = self.similarity(self.prototype[ids], samples)
        sim[~live] = -4.
        winner = (sim - .045 * self.usage[ids]).argmax(axis=1)
        novelty = sim.max(axis=1) < self.spec.novelty
        known = live.any(1) & ~novelty
        self.last[ids[known], winner[known]] = trial
        if not adapt or self.mode == "fixed":
            return ids[:0], ids[:0]

        edit = np.zeros(len(ids), dtype=bool)
        structural = self.mode in {"growing", "growth-fixed-writes", "reassign-fixed"}
        if structural:
            inactive = ~live
            vacant = inactive.any(1)
            # Evaluate only currently unseen branches for retirement; the
            # matched branch just received the latest physical presentation.
            inactive_age = trial - self.last[ids]
            stale = inactive_age > self.spec.retire_age
            stale &= live
            edit = novelty & (vacant | stale.any(axis=1))
            target = np.where(vacant, inactive.argmax(axis=1), inactive_age.argmax(axis=1))
            winner[edit] = target[edit]
            if edit.any():
                replaced = ids[edit]
                branch = winner[edit]
                occupied = self.alive[replaced, branch]
                self.births[replaced] += ~occupied
                self.retirements[replaced] += occupied
                self.alive[replaced, branch] = True
                start = self.prototype[replaced, branch].copy()
                if self.mode != "growth-fixed-writes":
                    self.prototype[replaced, branch] = samples[edit]
                else:
                    # Recruitment retains the same numerical movement rule as
                    # every ordinary competitive edit; only activity changes.
                    self.prototype[replaced, branch] = unit(start + self.spec.receptor_rate * (samples[edit] - start))
                self.centroid_l2[replaced] += np.linalg.norm(self.prototype[replaced, branch] - start, axis=1)

        ordinary = ~edit & (~novelty if structural else True)
        rows = ids[ordinary]
        slot = winner[ordinary]
        current = self.prototype[rows, slot]
        revised = unit(current + self.spec.receptor_rate * (samples[ordinary] - current))
        self.prototype[rows, slot] = revised
        self.centroid_l2[rows] += np.linalg.norm(revised - current, axis=1)
        updated = edit | ordinary
        self.last[ids[updated], winner[updated]] = trial
        self.usage[ids] *= .97
        self.usage[ids[updated], winner[updated]] += .03
        self.centroid_writes[ids[updated]] += 1
        return ids[edit], winner[edit]

    def measure(self, code, seen):
        ids = np.flatnonzero(seen)
        if ids.size:
            mass = np.square(code[ids]).sum(axis=1)
            diversity = np.divide(1., mass, out=np.zeros_like(mass), where=mass > 1e-6)
            self.response_participation[ids] += diversity
            self.response_count[ids] += 1

    def summary(self):
        active = np.full(len(self.alive), self.dimension) if self.mode == "raw" else self.alive.sum(axis=1)
        return {"active": active.tolist(), "births": self.births.tolist(),
                "retirements": self.retirements.tolist(), "prototype_writes": self.centroid_writes.tolist(),
                "prototype_l2": self.centroid_l2.tolist(), "presentations": self.presentations.tolist(),
                "participation": np.divide(self.response_participation, self.response_count,
                                           where=self.response_count > 0, out=np.zeros_like(self.response_participation)).tolist()}


class ChemicalForager:
    """One continuous LIF life, accepting fruit by its current odor alone."""

    def __init__(self, spec=ChemSpec(), *, birth_seed=0, action_seeds=None):
        self.spec, self.birth_seed = spec, int(birth_seed)
        self.action_seeds = None if action_seeds is None else tuple(action_seeds)
        generators = np.random.SeedSequence(birth_seed).spawn(4)
        fruit_rng, cue_rng, rng, link_rng = [np.random.default_rng(seed) for seed in generators]
        h = spec.units
        self.fruit = ReceptorPopulation(spec.fruit_bank, spec.capacity, spec, fruit_rng)
        cue_spec = spec if spec.cue_width is None else replace(spec, width=spec.cue_width)
        self.cue = ReceptorPopulation(spec.cue_bank, spec.cue_capacity, cue_spec, cue_rng)
        aff = rng.normal(.24, .66, (h, 34)).astype(np.float32)
        self.aff = (aff * (rng.random(aff.shape) < .72))[:, :self.cue.dimension + 2].copy()
        self.threshold = rng.uniform(.92, 1.08, h).astype(np.float32)
        self.bias = rng.uniform(.04, .10, h).astype(np.float32)
        mask = rng.random((h, h)) < .065
        np.fill_diagonal(mask, False)
        signs = np.where(rng.random(h) < .24, -1.6, 1.).astype(np.float32)
        self.recurrent = rng.uniform(.02, .07, (h, h)).astype(np.float32) * mask * signs
        self.links = (link_rng.random((32, h)) < .76)[:self.fruit.dimension]

    def reset(self, n):
        self.fruit.reset(n)
        self.cue.reset(n)
        h = self.spec.units
        self.v = np.zeros((n, h), np.float32)
        self.spike = np.zeros_like(self.v)
        self.trace = np.zeros_like(self.v)
        self.feature = np.zeros_like(self.v)
        self.mean = np.full_like(self.v, .16)
        self.direction = np.zeros_like(self.v)
        self.p = np.tile(np.eye(h, dtype=np.float32), (n, 1, 1)) if self.spec.cov_rate else None
        self.weights = np.zeros((n, self.fruit.dimension, h), dtype=np.float32)
        self.fruit_p = (np.tile(np.eye(self.fruit.dimension, dtype=np.float32), (n, 1, 1))
                        if self.spec.fruit_cov_rate else None)
        self.motor = np.zeros((n, 2), np.float32)
        self.reached_before = np.zeros(n, dtype=bool)
        self.peak = np.full(n, -np.inf, dtype=np.float32)
        self.eligibility = np.zeros_like(self.v)
        self.post = np.zeros((n, self.fruit.dimension), dtype=np.float32)
        self.sampled_odor = np.zeros((n, 4), dtype=np.float32)
        self.prediction = np.zeros(n, dtype=np.float32)
        self.random_direction = np.zeros((n, 4), dtype=np.float32)
        self.receipts = np.zeros((n, 2), dtype=np.int32)
        self.weight_updates = np.zeros((n, 2), dtype=np.int32)
        self.replaced_weight_l2 = np.zeros(n, dtype=np.float64)
        self.misaligned_receipts = np.zeros(n, dtype=np.int32)
        self.spike_count = np.zeros(n, dtype=np.int64)
        self.bite_relax = np.zeros(n, dtype=np.int32)
        self.weight_l2_sum = np.zeros(n, dtype=np.float64)
        self.trial = -1
        self.clock = -1
        self.adapt_receptors = True
        self.adapt_association = True
        self.force_phase = False
        if self.action_seeds is None:
            seeds = np.random.SeedSequence((self.birth_seed, 1257)).spawn(n)
        else:
            seeds = self.action_seeds
        if len(seeds) != n:
            raise ValueError("one action seed required per life")
        self.rng = [np.random.default_rng(seed) for seed in seeds]

    def _start(self):
        self.trial += 1
        self.clock = 0
        self.peak.fill(-np.inf)
        self.reached_before.fill(False)
        draws = np.stack([r.random(5, dtype=np.float32) for r in self.rng])
        strength = np.where(draws[:, 0] < self.spec.exploration, 4., .006).astype(np.float32)
        self.random_direction = (draws[:, 1:] - .5) * strength[:, None]

    def _receipt(self, obs):
        odor = unit(obs[:, TASTED])
        actual = (odor > 0).any(axis=1)
        aligned = (odor * self.sampled_odor).sum(axis=1) > .83
        self.misaligned_receipts += actual & ~aligned
        signed = obs[:, POSITIVE] - obs[:, NEGATIVE]
        self.receipts[:, 0] += aligned & (signed > 0)
        self.receipts[:, 1] += aligned & (signed < 0)
        mode = self.spec.outcomes
        if mode == "frozen" or not self.adapt_association:
            return
        if mode == "negative":
            signed = np.minimum(signed, 0.)
        used = np.flatnonzero(aligned & (signed != 0))
        if not used.size:
            return
        error = signed[used] - self.prediction[used]
        dw = (self.spec.rate * error)[:, None, None] * self.post[used, :, None] * self.eligibility[used, None, :]
        dw *= self.links
        before = self.weights[used]
        after = np.clip(before + dw, -.85, .85)
        self.weights[used] = after
        self.weight_l2_sum[used] += np.linalg.norm((after - before).reshape(len(used), -1), axis=1)
        self.weight_updates[used, 0] += signed[used] > 0
        self.weight_updates[used, 1] += signed[used] < 0

    def _snapshot(self):
        x = self.trace - self.mean
        self.feature[:] = x
        self.direction[:] = x
        if self.p is not None:
            z = np.einsum("nij,nj->ni", self.p, x, optimize=False)
            factor = np.linalg.norm(x, axis=1) / np.maximum(np.linalg.norm(z, axis=1), 1e-8)
            self.direction[:] = factor[:, None] * z
            if self.adapt_association:
                rate = self.spec.cov_rate
                energy = np.maximum((x * z).sum(axis=1), 0.)
                self.p -= (rate / (1 + rate * energy))[:, None, None] * z[:, :, None] * z[:, None, :]
        if self.adapt_association:
            self.mean += .02 * x

    def _update_fruit_geometry(self, post, entered, erased_lives, erased_branches):
        if self.fruit_p is None or not self.adapt_association:
            return
        self.fruit_p[erased_lives, erased_branches, :] = 0.
        self.fruit_p[erased_lives, :, erased_branches] = 0.
        self.fruit_p[erased_lives, erased_branches, erased_branches] = 1.
        sampled = np.flatnonzero(entered)
        if sampled.size:
            mean = post[sampled]
            z = np.einsum("nij,nj->ni", self.fruit_p[sampled], mean, optimize=False)
            rate = self.spec.fruit_cov_rate
            step = rate / (1 + rate * np.maximum((mean * z).sum(axis=1), 0.))
            self.fruit_p[sampled] -= step[:, None, None] * z[:, :, None] * z[:, None, :]

    def _post_credit(self, post, fire):
        post_body = post[fire]
        if self.fruit_p is not None and fire.any():
            z = np.einsum("nij,nj->ni", self.fruit_p[fire], post_body, optimize=False)
            factor = np.linalg.norm(post_body, axis=1) / np.maximum(np.linalg.norm(z, axis=1), 1e-8)
            post_body = factor[:, None] * z
        return post_body

    def act(self, obs):
        self.clock += 1
        if obs[0, NEW]:
            self._start()
        self._receipt(obs)
        cue_odor = unit(obs[:, CUE])
        if self.clock == 2:
            self.cue.observe(cue_odor, (cue_odor > 0).any(axis=1), self.trial, adapt=self.adapt_receptors)
        cue_code = self.cue.match(cue_odor)
        self.v *= .53
        self.v += np.column_stack((obs[:, CTX], cue_code)) @ self.aff.T
        self.v += self.bias + self.spike @ self.recurrent.T
        np.greater_equal(self.v, self.threshold, out=self.spike)
        self.v[self.spike != 0] = 0
        self.trace *= .73
        self.trace += .27 * self.spike
        self.spike_count += self.spike.sum(axis=1, dtype=np.int64)
        if self.clock == 6:
            self._snapshot()
        fruit_odor = unit(obs[:, FRUIT])
        active = obs[:, ACTIVE] > 0
        reach = active & (obs[:, REACH] > 0) & (fruit_odor > 0).any(axis=1)
        entered = reach & ~self.reached_before
        erase_life, erase_slot = self.fruit.observe(fruit_odor, entered, self.trial, adapt=self.adapt_receptors)
        if erase_life.size:
            # Existing effective outgoing synapses leave with their branch.
            prior = self.weights[erase_life, erase_slot]
            self.replaced_weight_l2[erase_life] += np.linalg.norm(prior, axis=1)
            self.weights[erase_life, erase_slot] = 0.
        self.reached_before[:] = reach

        post = self.fruit.match(fruit_odor)
        self.fruit.measure(post, entered)
        self._update_fruit_geometry(post, entered, erase_life, erase_slot)
        projected = np.einsum("nfh,nh->nf", self.weights, self.feature, optimize=False)
        prediction = np.tanh(2.5 * (projected * post).sum(axis=1))
        score = prediction + (fruit_odor * self.random_direction).sum(axis=1)

        if 6 <= self.clock < 21:
            np.maximum(self.peak, np.where(reach, score, -np.inf), out=self.peak)
        relax = .30 * max(0, self.clock - 32)
        accepted = score + self.spec.bite_tolerance + relax >= self.peak
        if self.force_phase:
            accepted[:] = True
        drive = reach & accepted & (self.clock >= 21)
        self.motor *= .43
        self.motor[:, 0] += 1.23 * active
        self.motor[:, 1] += 1.23 * drive
        pulses = self.motor >= 1.
        self.motor[pulses] = 0.
        fire = pulses[:, 1]
        self.bite_relax += fire & (self.clock > 32)
        self.sampled_odor.fill(0.)
        self.sampled_odor[fire] = fruit_odor[fire]
        self.eligibility[fire] = self.direction[fire]
        self.post[fire] = self._post_credit(post, fire)
        self.prediction[fire] = prediction[fire]
        return np.column_stack((np.zeros(len(obs), dtype=bool), pulses))

    def stats(self):
        return {"fruit": self.fruit.summary(), "cue": self.cue.summary(), "spikes": self.spike_count.tolist(),
                "weight_updates": self.weight_updates.tolist(), "misaligned": self.misaligned_receipts.tolist(),
                "receipts": self.receipts.tolist(), "retired_weight_l2": self.replaced_weight_l2.tolist(),
                "fallback_bites": self.bite_relax.tolist(), "association_write_l2": self.weight_l2_sum.tolist(),
                "recurrent_edges": int(np.count_nonzero(self.recurrent))}
