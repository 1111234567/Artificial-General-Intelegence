"""Paired, closed-loop transform transfer with contact-gated teaching tastes."""

from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np

from . import mixture_flash, mixture_world, world


DOSES = (0, 2, 4, 8, 16)
HISTORIES = ("related", "unrelated")
IDENTITY = np.eye(4)


@dataclass(frozen=True)
class Stage:
    menu: mixture_world.Foods
    appearance: np.ndarray
    odor_tick: np.ndarray
    context_tick: np.ndarray
    truth: np.ndarray
    taste: np.ndarray

    def slice(self, start, stop):
        part = np.s_[:, start:stop]
        tape = self.menu.body
        body = replace(tape, **{name: getattr(tape, name)[part] for name in
                               ("ctx", "cue", "left_id", "good_id", "distances", "wind")})
        menu = replace(self.menu, body=body, chemicals=self.menu.chemicals[part],
                       probability=self.menu.probability[part], realized=self.menu.realized[part])
        return Stage(menu, self.appearance[part], self.odor_tick[part], self.context_tick[part],
                     self.truth[part], self.taste[part])


@dataclass(frozen=True)
class Protocol:
    roots: tuple[int, ...]
    seeds: tuple[int, ...]
    row_root: np.ndarray
    parity: np.ndarray
    partition: str
    doses: tuple[int, ...]
    history: dict[str, Stage]
    teaching: Stage
    probes: dict[str, Stage]
    nonce: int


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _rng(root, nonce, stream):
    return np.random.default_rng(np.random.SeedSequence([root, nonce, stream]))


def _transform(anchors, matrix):
    return anchors @ matrix.T


def _teaching(anchors, matrix, signs, jitter, rng):
    transformed = _transform(anchors, matrix)
    odors = np.stack((anchors, transformed), axis=1).reshape(-1, 4)
    if jitter:
        odors = _unit(odors + rng.normal(0., jitter, size=odors.shape))
    return np.repeat(odors[:, None, :], 2, axis=1), signs


def _queries(anchors, matrix, signs, jitter, rng):
    transformed = _transform(anchors, matrix)
    # Equal angular distance from the anchor. An identity or nearest-anchor
    # heuristic cannot distinguish the two candidates.
    cosine = (anchors * transformed).sum(axis=1, keepdims=True).clip(-1., 1.)
    direction = rng.normal(size=anchors.shape)
    direction -= (direction * anchors).sum(axis=1, keepdims=True) * anchors
    decoy = cosine * anchors + np.sqrt(1. - cosine**2) * _unit(direction)
    related = rng.integers(2, size=len(anchors))
    odors = np.empty((len(anchors), 2, 2, 4))
    odors[:, 0] = anchors[:, None, :]
    odors[np.arange(len(anchors)), 1, related] = transformed
    odors[np.arange(len(anchors)), 1, 1 - related] = decoy
    if jitter:
        anchor_noise = rng.normal(0., jitter, size=anchors.shape)
        odors[:, 0] = _unit(anchors + anchor_noise)[:, None, :]
        odors[:, 1] = _unit(odors[:, 1] + rng.normal(0., jitter, size=odors[:, 1].shape))
    return odors.reshape(-1, 2, 4), signs, related


def _stage(roots, seeds, row_root, tags, parity, designs, habitat, probe, nonce, stream):
    pairs = len(designs[0][0]) // 2
    trials = 3 * pairs
    geometry_seeds = tuple(int(_rng(root, nonce, stream).integers(2**63)) for root in roots)
    geometry = world.make_tape(geometry_seeds, trials, distance_jitter=.13, wind_scale=.07)
    odor_tick, context_tick = mixture_flash.timing(geometry_seeds, trials)
    # World geometry, sensory timing, odor and layouts are identical across
    # sign twins. The actor sees the root seed only as its motor noise seed.
    body = replace(geometry, seeds=seeds, ctx=np.full((len(seeds), trials), habitat, dtype=np.int8),
                   **{name: getattr(geometry, name)[row_root] for name in
                      ("cue", "left_id", "good_id", "good_map", "distances", "wind")})
    odors = np.empty((len(seeds), trials, 2, 4), dtype=np.float32)
    signs = np.zeros((len(seeds), trials), dtype=np.int8)
    pair_odors = np.asarray([item[0] for item in designs], dtype=np.float32)[row_root]
    pair_signs = np.asarray([item[1] for item in designs], dtype=np.int8)[row_root]
    gap_rng = [_rng(root, nonce, stream + 900) for root in roots]
    gap_odors = np.asarray([_unit(rng.normal(size=(pairs, 4))) for rng in gap_rng], dtype=np.float32)
    odors[:, 0::3] = gap_odors[row_root, :, None, :]
    odors[:, 1::3], odors[:, 2::3] = pair_odors[:, ::2], pair_odors[:, 1::2]
    signs[:, 0::3] = np.asarray([rng.choice((-1, 1), size=pairs) for rng in gap_rng], dtype=np.int8)[row_root]
    signs[:, 1::3], signs[:, 2::3] = pair_signs[:, ::2], pair_signs[:, 1::2]
    realized = np.repeat(signs[:, :, None], 2, axis=2)
    linked = parity[:, habitat, None]
    truth = np.full(realized.shape[:2], -1, dtype=np.int8)
    taste = np.ones(realized.shape[:2], dtype=bool)
    taste[:, 0::3] = False
    if probe:
        related = np.asarray([item[2] for item in designs], dtype=np.int8)[row_root]
        good = np.where(signs[:, 1::3] * linked > 0, related, 1 - related)
        truth[:, 2::3] = good
        realized[:, 2::3] = np.where(np.arange(2) == good[:, :, None], 1, -1)
        taste[:, 2::3] = False
    else:
        realized[:, 2::3] *= linked[:, :, None]
    appearance = np.broadcast_to(tags[row_root, habitat, None, :], (len(seeds), trials, 2))
    menu = mixture_world.Foods(body, .5 * (odors + 1.), (realized > 0).astype(np.float32), realized,
                               np.empty(0), "visible_odor_transform", 0)
    return Stage(menu, appearance, odor_tick[row_root], context_tick[row_root], truth, taste)


def make(roots, *, partition="develop", history_pairs=12, probes=3, doses=DOSES,
         counterbalance=True, jitter=0., nonce=31242086):
    """Build matched histories without including any rule, mate or phase ID in observations.

    A reflects both dimensions of a private plane. Development B reflects one
    line in that plane. Held-out B rotates the plane by a signed quarter turn.
    Related/unrelated histories differ only in whether the A plane is also B's.
    Tastant parity in A and B is independently counterbalanced within root.
    """
    roots = tuple(map(int, roots))
    doses = tuple(map(int, doses))
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("supply distinct physical roots")
    if partition not in ("develop", "heldout"):
        raise ValueError("partition must be 'develop' or 'heldout'")
    if history_pairs < 2 or probes < 1 or jitter < 0.:
        raise ValueError("need at least two history pairs, one probe and nonnegative jitter")
    if not doses or doses[0] != 0 or any(d < 0 or d % 2 for d in doses) or sorted(set(doses)) != list(doses):
        raise ValueError("doses must start at zero and increase in complete two-receipt pairs")
    # This order keeps complete sign strata adjacent to their physical root.
    twins = np.array([(1, 1), (-1, -1), (1, -1), (-1, 1)], dtype=np.int8)
    if counterbalance:
        row_root = np.repeat(np.arange(len(roots)), len(twins))
        parity = np.tile(twins, (len(roots), 1))
    else:
        row_root = np.arange(len(roots))
        parity = np.asarray([_rng(root, nonce, 991).choice((-1, 1), size=2) for root in roots], dtype=np.int8)
    seeds = tuple(roots[i] for i in row_root)
    designs = {key: [] for key in ("related", "unrelated", "teach", "new", "old_related", "old_unrelated")}
    tags = []
    for root in roots:
        geometry = _rng(root, nonce, 631)
        shared = np.linalg.qr(geometry.normal(size=(4, 2)))[0]
        independent = np.linalg.qr(geometry.normal(size=(4, 2)))[0]
        line = shared @ _unit(geometry.normal(size=2))
        sense = geometry.choice((-1, 1))
        if partition == "develop":
            b = IDENTITY - 2. * np.outer(line, line)
        else:
            p, q = shared.T
            b = IDENTITY - shared @ shared.T + sense * (np.outer(q, p) - np.outer(p, q))
        a = IDENTITY - 2. * shared @ shared.T
        u = IDENTITY - 2. * independent @ independent.T
        low = geometry.uniform(-.15, .4)
        angles = np.array([low, low + geometry.uniform(.75, 1.)])[geometry.permutation(2)]
        tags.append(np.column_stack((np.cos(angles), np.sin(angles))))
        for key, count, matrix, stream, query in (
            ("related", history_pairs, a, 101, False), ("unrelated", history_pairs, u, 101, False),
            ("teach", max(doses[-1] // 2, 1), b, 102, False), ("new", probes, b, 103, True),
            ("old_related", probes, a, 104, True), ("old_unrelated", probes, u, 104, True),
        ):
            rng = _rng(root, nonce, stream)
            anchors = _unit(rng.normal(size=(count, 4)))
            signs = rng.choice((-1, 1), size=count)
            generator = _queries if query else _teaching
            designs[key].append(generator(anchors, matrix, np.repeat(signs, 2), jitter, rng))
    tags = np.asarray(tags, dtype=np.float32)

    def stage(key, habitat, probe, stream):
        return _stage(roots, seeds, row_root, tags, parity, designs[key], habitat, probe, nonce, stream)

    return Protocol(roots, seeds, row_root, parity, partition, doses,
                    {key: stage(key, 0, False, 20) for key in HISTORIES}, stage("teach", 1, False, 21),
                    {"new": stage("new", 1, True, 22),
                     **{"old_" + key: stage("old_" + key, 0, True, 23) for key in HISTORIES}}, nonce)


class Body(mixture_world.Body):
    """Same single-tick public ports and body physics as the released habitat."""

    def __init__(self, stage, *, donor_taste=None, record=False):
        self.stage = stage
        self.donor_taste = donor_taste
        super().__init__(stage.menu, record=record)

    def observe(self):
        obs = super().observe()
        trial, tick = self.trial_index, self.tick_index
        obs[self.stage.odor_tick[:, trial] != tick, mixture_world.LEFT.start:mixture_world.RIGHT.stop] = .5
        obs[:, mixture_world.CTX] = self.stage.appearance[:, trial] * (self.stage.context_tick[:, trial] == tick)[:, None]
        return obs

    def _shown_reward(self, receipt):
        trial = self.trial_index
        if self.donor_taste is not None:
            return self.donor_taste[:, trial] * receipt
        return super()._shown_reward(receipt) * self.stage.taste[:, trial]


def _play(actor, stage, donor_taste=None, record=False):
    sim = Body(stage, donor_taste=donor_taste, record=record)
    obs = sim.observe()
    while obs is not None:
        if actor is None:
            # Privileged geometry-only control, never handed to a learner.
            good = stage.truth[:, sim.trial_index]
            side = np.where(good >= 0, good, sim.trial_index % 2)
            pulses = (side[:, None] == np.arange(2)) & obs[:, world.ACTIVE, None].astype(bool)
        else:
            pulses = actor.act(obs)
        obs = sim.step(pulses)
    return sim.result()


def _probe(actor, stage):
    # Every novel anchor and its silent forced choice receive a fresh copy
    # of the same post-dose actor. Anchor taste remains real and can write.
    outcomes = {name: [] for name in ("success", "contact", "anchor", "shown", "side", "tick")}
    for start in range(0, stage.menu.body.trials, 3):
        pair = stage.slice(start, start + 3)
        res = _play(deepcopy(actor), pair)
        outcomes["success"].append(res.contact_side[:, 2] == pair.truth[:, 2])
        outcomes["contact"].append(res.contact_side[:, 2] >= 0)
        outcomes["anchor"].append((res.observed_reward[:, 1] != 0).any(1))
        outcomes["shown"].append((res.observed_reward[:, 2] != 0).sum(1))
        outcomes["side"].append(res.contact_side[:, 2])
        outcomes["tick"].append(res.contact_tick[:, 2])
    return {name: np.stack(values, axis=1) for name, values in outcomes.items()}


def _exposure(target, prefix, observed, contact):
    target[prefix + "_receipts"] = (observed != 0).sum(1)
    target[prefix + "_positive"] = (observed > 0).sum(1)
    target[prefix + "_negative"] = (observed < 0).sum(1)
    target[prefix + "_contacts"] = (contact >= 0).sum(1)
    target[prefix + "_left"] = (contact == 0).sum(1)
    target[prefix + "_right"] = (contact == 1).sum(1)
    target[prefix + "_complete_pairs"] = (observed.reshape(len(observed), -1, 2) != 0).all(2).sum(1)


def _metrics(actor, protocol, history, hist, shown, true, side, fillers, pre_a):
    new = _probe(actor, protocol.probes["new"])
    old = _probe(actor, protocol.probes["old_" + history])
    result = dict(joint=(new["success"] & old["success"]).mean(1),
                  joint_contact=(new["contact"] & old["contact"]).mean(1),
                  pre_A=old["success"].mean(1) if pre_a is None else pre_a)
    for name, probe in (("new", new), ("old", old)):
        result[name] = probe["success"].mean(1)
        result[name + "_first"] = probe["success"][:, 0]
        result[name + "_contact"] = probe["contact"].mean(1)
        result[name + "_timeout"] = (~probe["contact"]).mean(1)
        result[name + "_anchor_receipts"] = probe["anchor"].sum(1)
        result[name + "_silent_receipts"] = probe["shown"].sum(1)
        result[name + "_first_side"] = probe["side"][:, 0]
        result[name + "_first_tick"] = probe["tick"][:, 0]
    signed_slots = np.arange(hist.contact_side.shape[1]) % 3 != 0
    _exposure(result, "history", hist.observed_reward[:, signed_slots].sum(2), hist.contact_side[:, signed_slots])
    result["history_total_contacts"] = (hist.contact_side >= 0).sum(1)
    _exposure(result, "b", shown, side)
    result["b_total_contacts"] = result["b_contacts"] + fillers
    result["b_true_positive"] = (true > 0).sum(1)
    result["b_true_negative"] = (true < 0).sum(1)
    result["b_true_receipts"] = (true != 0).sum(1)
    return result


def _shuffled(protocol, shown):
    if len(protocol.roots) < 2:
        raise ValueError("reward shuffle requires at least two physical roots")
    rng = np.random.default_rng(np.random.SeedSequence([protocol.nonce, 7781]))
    output = np.empty_like(shown)
    rows = np.arange(len(protocol.seeds)).reshape(len(protocol.roots), -1)
    for trial in range(shown.shape[1]):
        order = rng.permutation(len(rows))
        donors = np.empty_like(order)
        donors[order] = np.roll(order, 1)
        for root, donor in enumerate(donors):
            output[rows[root], trial] = shown[rng.permutation(rows[donor]), trial]
    return output


def _series(protocol, actor, history, hist, *, rollback=False, donor_taste=None):
    post_history = actor.snapshot_matrix() if rollback else None
    intact, reversed_matrix = {}, {}
    n = len(protocol.seeds)
    shown = np.empty((n, protocol.doses[-1]), dtype=np.int8)
    true, side = np.empty_like(shown), np.empty_like(shown)
    fillers = np.zeros(n, dtype=np.int32)
    old, previous = None, 0
    for dose in protocol.doses:
        if dose > previous:
            span = np.s_[:, previous:dose]
            stage = protocol.teaching.slice(3 * previous // 2, 3 * dose // 2)
            signed_slots = np.arange(stage.menu.body.trials) % 3 != 0
            donor = None
            if donor_taste is not None:
                donor = np.zeros_like(stage.taste, dtype=np.int8)
                donor[:, signed_slots] = donor_taste[span]
            res = _play(actor, stage, donor)
            shown[span] = res.observed_reward[:, signed_slots].sum(2)
            true[span], side[span] = res.reward[:, signed_slots], res.contact_side[:, signed_slots]
            fillers += (res.contact_side[:, ~signed_slots] >= 0).sum(1)
        args = protocol, history, hist, shown[:, :dose], true[:, :dose], side[:, :dose], fillers, old
        metrics = _metrics(actor, *args)
        old = metrics["pre_A"]
        intact[str(dose)] = metrics
        if rollback:
            if dose == 0:
                reversed_matrix[str(dose)] = {k: v.copy() for k, v in metrics.items()}
            else:
                restored = deepcopy(actor)
                restored.restore_matrix(post_history)
                reversed_matrix[str(dose)] = _metrics(restored, *args)
        previous = dose
    return intact, reversed_matrix, shown


def evaluate(protocol, factory, *, rollback=False, shuffle=False):
    """Return per-life paired physical results keyed by history, variant and dose."""
    result, histories = {}, {}
    for condition in HISTORIES:
        actor = factory(protocol.seeds) if factory is not None else None
        if actor is not None:
            actor.reset(len(protocol.seeds))
        hist = _play(actor, protocol.history[condition])
        before = deepcopy(actor) if shuffle else None
        intact, reversed_matrix, receipts = _series(protocol, actor, condition, hist, rollback=rollback)
        result[condition] = {"intact": intact}
        histories[condition] = hist.contact_side
        if rollback:
            result[condition]["rollback"] = reversed_matrix
        if shuffle:
            shuffled, _, _ = _series(protocol, before, condition, hist, donor_taste=_shuffled(protocol, receipts))
            result[condition]["shuffled"] = shuffled
    same = (histories["related"] == histories["unrelated"]).all(1)
    for variants in result.values():
        for doses in variants.values():
            for values in doses.values():
                values["matched_history_sides"] = same
    return result


def physical_oracle(protocol):
    """Privileged action and geometry check using hidden targets and full pulse drive."""
    return evaluate(protocol, None)


def _outer(left, right):
    return left[:, :, None] * right[:, None, :]


def _basis(scatter, rank):
    return np.linalg.eigh(scatter)[1][:, :, -rank:]


class _PublicPairOracle:
    """Diagnostic given the transform class and cadence, with no world data access."""

    def __init__(self, partition):
        self.partition = partition

    def reset(self, n):
        self.round = -1
        self.first_context = np.zeros((n, 2))
        self.current = np.zeros(n, dtype=np.intp)
        self.context = np.zeros(n, dtype=bool)
        self.odor = np.zeros((n, 2, 4))
        self.scent = np.zeros(n, dtype=bool)
        self.anchor = np.zeros((n, 4))
        self.anchor_sign = np.zeros(n)
        self.parity = np.zeros((n, 2))
        self.pairs = np.zeros((n, 2), dtype=np.int32)
        self.scatter = np.zeros((n, 2, 4, 4))
        self.cross = np.zeros((n, 4, 4))
        self.a_plane = np.zeros((n, 4, 4))
        self.a_basis = np.zeros((n, 4, 2))
        self.fit = np.broadcast_to(IDENTITY, (n, 2, 4, 4)).copy()

    def _estimate(self, rows, habitat, x, y):
        delta = y - x
        self.scatter[rows, habitat] += _outer(delta, delta)
        self.pairs[rows, habitat] += 1
        scatter = self.scatter[rows, habitat]
        if habitat == 0:
            basis = _basis(scatter, 2)
            plane = basis @ basis.transpose(0, 2, 1)
            self.a_plane[rows], self.a_basis[rows] = plane, basis
            return IDENTITY - 2. * plane
        if self.partition == "develop":
            basis = _basis(scatter, 1)
            return IDENTITY - 2. * (basis @ basis.transpose(0, 2, 1))

        self.cross[rows] += _outer(y, x)
        a_plane = self.a_plane[rows]
        mass = np.trace(scatter, axis1=1, axis2=2)
        defect = mass - np.sum(a_plane * scatter, axis=(1, 2))
        share = (defect <= .01 * mass) & (self.pairs[rows, 0] >= 2)
        basis = np.where(share[:, None, None], self.a_basis[rows], _basis(scatter, 2))
        plane = basis @ basis.transpose(0, 2, 1)
        p, q = basis.transpose(2, 0, 1)
        spin = _outer(q, p) - _outer(p, q)
        orientation = np.where(np.sum(spin * self.cross[rows], axis=(1, 2)) >= 0., 1., -1.)
        inferred = IDENTITY - plane + orientation[:, None, None] * spin

        # A single pair is underdetermined if its displacement lies outside
        # the old plane. Use the minimum-angle orthogonal map until two pairs.
        c = (x * y).sum(1).clip(-1., 1.)[:, None]
        v = _unit(y - c * x)
        partial = (IDENTITY + (c - 1.)[:, :, None] * (_outer(x, x) + _outer(v, v))
                   + np.sqrt(1. - c**2)[:, :, None] * (_outer(v, x) - _outer(x, v)))
        return np.where((share | (self.pairs[rows, 1] > 1))[:, None, None], inferred, partial)

    def act(self, obs):
        if obs[0, world.TRIAL_START]:
            self.round += 1
            self.context.fill(False)
            self.scent.fill(False)
            if self.round % 3 == 0:
                self.anchor_sign.fill(0.)
        context = obs[:, mixture_world.CTX]
        rows = np.flatnonzero(context.any(1))
        if self.round == 0:
            self.first_context[rows] = context[rows]
        self.current[rows] = ((context[rows] * self.first_context[rows]).sum(1) < .9)
        self.context[rows] = True

        scents = obs[:, mixture_world.LEFT.start:mixture_world.RIGHT.stop].reshape(len(context), 2, 4) * 2. - 1.
        rows = np.flatnonzero(scents.any(axis=(1, 2)))
        self.scent[rows] = True
        self.odor[rows] = _unit(scents[rows])
        feedback = obs[:, world.REWARD_POS] - obs[:, world.REWARD_NEG]
        contact = obs[:, world.CONTACT_ID]
        found = (feedback != 0) & contact.any(1) & self.scent & self.context
        slot = self.round % 3
        if slot == 1:
            rows = np.flatnonzero(found)
            self.anchor[rows] = self.odor[rows, contact[rows].argmax(1)]
            self.anchor_sign[rows] = feedback[rows]
        elif slot == 2:
            for habitat in (0, 1):
                rows = np.flatnonzero(found & (self.anchor_sign != 0) & (self.current == habitat))
                if rows.size:
                    self.parity[rows, habitat] += feedback[rows] * self.anchor_sign[rows]
                    y = self.odor[rows, contact[rows].argmax(1)]
                    self.fit[rows, habitat] = self._estimate(rows, habitat, self.anchor[rows], y)

        ready = self.scent & self.context & obs[:, world.ACTIVE].astype(bool)
        chosen = np.full(len(context), self.round % 2, dtype=np.int8)
        if slot == 2:
            rows = np.flatnonzero(ready & (self.anchor_sign != 0))
            sites = self.current[rows]
            expected = np.einsum("nij,nj->ni", self.fit[rows, sites], self.anchor[rows])
            near = np.einsum("nsi,ni->ns", self.odor[rows], expected).argmax(1)
            sign = self.anchor_sign[rows] * self.parity[rows, sites]
            chosen[rows] = np.where(sign > 0, near, np.where(sign < 0, 1 - near, chosen[rows]))
        return ready[:, None] & (chosen[:, None] == np.arange(2))


def public_oracle(protocol):
    """Contact-driven geometric reference given pair cadence and transform class.

    Uses the same policy for both histories, and makes its decisions from the
    observed odor, context, reached side and signed taste. The structure-aware
    fit may assume a shared plane if the actual B displacements support it.
    """
    return evaluate(protocol, lambda _seeds: _PublicPairOracle(protocol.partition))
