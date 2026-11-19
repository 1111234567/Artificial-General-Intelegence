"""Assign retention credit from actual receipts that arrived after an edge was born."""

import numpy as np

from .coupled_birth import Population as CoupledPopulation, WIDTH, expit, strongly_connected


ARMS = ("instant", "later", "shuffled")
AUDIT_INDEX_FIELDS = ("lane", "trial", "arc", "birth_receipt")
AUDIT_VALUE_FIELDS = ("later_score", "policy_score", "past_instant_score", "heldout_effect", "effective_weight")
WRITE_GROUPS = ("initial_generation", "prior_growth", "new_growth")
WRITE_FIELDS = ("received", "changed", "absolute_step_sum")


class Population(CoupledPopulation):
    def __init__(self, parameters, masks, schedule, random_seeds, *, arm, **kwargs):
        if arm not in ARMS or schedule.retirement_every != 6:
            raise ValueError("Expected a registered arm and a six-receipt retirement schedule")
        super().__init__(parameters, masks, schedule, random_seeds, **kwargs)
        self.arm = arm
        self.future_score = np.zeros_like(self.raw)
        self.future_observations = np.zeros_like(self.newborn_receipt)
        self.shuffle = [np.random.default_rng(
            np.random.SeedSequence(seed.entropy, spawn_key=(*seed.spawn_key, 1))
            if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence([seed, 1])
        ) for seed in random_seeds]
        self.evidence_index = []
        self.evidence_values = []
        self.evidence_active = []
        self.writes = []
        self.edge_counts = []

    def marginal_removal(self, lanes):
        """Return pre-update loss improvement for each occupied coordinate silenced in isolation."""
        masks = self.mask[lanes].reshape(len(lanes), WIDTH * WIDTH)
        columns = np.flatnonzero(masks).reshape(-1) % (WIDTH * WIDTH)
        counts = masks.sum(axis=-1)
        offsets = np.concatenate(([0], np.cumsum(counts[:-1])))
        examples = np.repeat(np.arange(len(lanes)), counts)
        variants = np.arange(len(columns)) - np.repeat(offsets, counts)
        target, source = np.divmod(columns, WIDTH)
        coefficients = self.matrix[lanes[examples], target, source]
        size = int(counts.max()) + 1  # The last variant replays the intact circuit in the same batch.
        initial = self.history[0][1][lanes]
        state = np.broadcast_to(initial[:, None, :], (len(lanes), size, WIDTH)).copy()
        motor = self.parameters["readout"][lanes, None, :]
        offset = self.parameters["offset"][lanes, None]
        matrix = self.matrix[lanes].transpose(0, 2, 1)
        total = np.zeros((len(lanes), size), dtype=np.float64)
        for signal, _, _, _, _, active in self.history:
            drive = np.einsum("bi,bij->bj", signal[lanes], self.parameters["drive"][lanes])
            candidate, leak = np.split(drive + self.parameters["bias"][lanes], 2, axis=-1)
            gate = expit(leak)[:, None, :]
            incoming = np.matmul(state, matrix)
            incoming[examples, variants, target] -= coefficients * state[examples, variants, source]
            state = (1 - gate) * np.tanh(candidate[:, None, :] + incoming) + gate * state
            if active is not None:
                total += active[lanes, None] * ((state * motor).sum(axis=-1) + offset)
        direction = self.receipt_direction[lanes, None]
        loss = np.logaddexp(0., -direction * total / self.trace_ticks[lanes, None])
        effect = (loss[examples, -1] - loss[examples, variants]).astype(np.float32)
        return lanes[examples], target, source, effect

    def finish_request(self, contacted, shown, *, donor=None):
        observed = (contacted >= 0) & (shown != 0)
        if np.any(observed & (self.trace_ticks == 0)):
            raise ValueError("A contact receipt requires a preceding movement phase")
        self.pre_mask = self.mask.copy()
        self.receipt_direction = (2 * contacted - 1) * shown
        self.receipt_writes = np.zeros((len(self.state), len(WRITE_GROUPS), len(WRITE_FIELDS)), dtype=np.float32)
        self.current_observed = observed
        next_receipt = self.receive_count + 1
        slot = next_receipt % 6
        sample = observed & np.isin(slot, (4, 5))
        heldout = observed & (slot == 0)
        scored = np.flatnonzero(sample | heldout)
        if len(scored):
            lane, target, source, effect = self.marginal_removal(scored)
            arc = lane, target, source
            old = self.newborn_receipt[arc] < next_receipt[lane]
            update = sample[lane] & old
            update_arc = lane[update], target[update], source[update]
            # The receipt that creates an edge was evaluated with that edge dormant.
            self.future_score[update_arc] = (
                self.schedule.decay * self.future_score[update_arc] + (1 - self.schedule.decay) * effect[update]
            )
            self.future_observations[update_arc] += 1
        self.policy_score = self.future_score
        mature = self.future_observations >= 2
        if self.arm == "shuffled" and heldout.any():
            self.policy_score = self.future_score.copy()
            for life in np.flatnonzero(heldout):
                for grown in (False, True):
                    group = self.mask[life] & mature[life] & ((self.newborn_trial[life] >= 0) == grown)
                    self.policy_score[life][group] = self.shuffle[life].permutation(self.future_score[life][group])
        if heldout.any():
            # Its label is for the forecast audit only. Retirement ranks using the older samples.
            eligible = heldout[lane] & mature[arc]
            audit_arc = lane[eligible], target[eligible], source[eligible]
            birth = self.newborn_receipt[audit_arc]
            ranked_score = self.retire_scores if self.arm == "instant" else self.policy_score
            self.evidence_index.append(np.column_stack((
                audit_arc[0], np.full(len(birth), self.trial), audit_arc[1] * WIDTH + audit_arc[2], birth
            )).astype(np.int16))
            self.evidence_values.append(np.column_stack((
                self.future_score[audit_arc], ranked_score[audit_arc], self.retire_scores[audit_arc],
                effect[eligible], self.raw[audit_arc]
            )).astype(np.float32))
            self.evidence_active.append(self.future_observations[audit_arc].copy())
        result = super().finish_request(contacted, shown, donor=donor)
        self.writes.append(self.receipt_writes)
        self.edge_counts.append(self.mask.sum(axis=(1, 2)).astype(np.int16))
        return result

    def _update_weights(self, credit, *, gradients=None):
        before = self.raw.copy()
        super()._update_weights(credit, gradients=gradients)
        step = np.abs(self.raw - before)
        justified = self.mask & self.current_observed[:, None, None]
        if np.any((step != 0) & ~justified):
            raise RuntimeError("An inactive or unobserved edge received a weight update")
        for index, group in enumerate((
            self.pre_mask & (self.newborn_trial < 0), self.pre_mask & (self.newborn_trial >= 0), ~self.pre_mask,
        )):
            trained = justified & group
            self.receipt_writes[:, index, 0] = trained.sum(axis=(1, 2))
            self.receipt_writes[:, index, 1] = (trained & (step != 0)).sum(axis=(1, 2))
            self.receipt_writes[:, index, 2] = np.where(trained, step, 0.).sum(axis=(1, 2))

    def _retire(self, retire, events, records):
        if self.arm == "instant":
            super()._retire(retire, events, records)
            return
        for lane in np.flatnonzero(retire):
            mask = self.mask[lane]
            scores = self.policy_score[lane]
            mature = self.future_observations[lane] >= 2
            allowed = (mask & mature & (scores > 0)
                       & (mask.sum(axis=0) > 1)[None, :] & (mask.sum(axis=1) > 1)[:, None])
            picks = np.flatnonzero(allowed)
            for pick in picks[np.argsort(-scores.ravel()[picks], kind="stable")]:
                target, source = divmod(int(pick), WIDTH)
                mask[target, source] = False
                if strongly_connected(WIDTH, zip(*np.nonzero(mask.T), strict=True)):
                    events[lane, 1] = pick
                    records[lane, 1] = self.newborn_records(lane, target, source, self.degree[lane, target])
                    self._reset_arc(lane, target, source)
                    break
                mask[target, source] = True
        if (events[:, 1] >= 0).any():
            self.refresh_matrices()

    def _reset_arc(self, lane, target, source):
        super()._reset_arc(lane, target, source)
        self.future_score[lane, target, source] = 0
        self.future_observations[lane, target, source] = 0
