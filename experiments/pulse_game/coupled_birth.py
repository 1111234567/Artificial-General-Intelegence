"""Joint edge birth and weight learning from the same physical receipt."""

import numpy as np

from .free_graph_model import strongly_connected
from .joint_live_model import Population as JointPopulation, WIDTH, expit


AUDIT_FIELDS = ("ascent", "birth_weight", "predicted_loss_reduction", "exact_loss_reduction")


class Population(JointPopulation):
    """Give a dormant edge its own first update, then train it like every active edge."""

    def __init__(self, parameters, masks, schedule, random_seeds, *, learning_rate,
                 pin_newborn=False, birth_scale=1., independent_sign=False, audit=False):
        if learning_rate <= 0 or birth_scale <= 0 or pin_newborn:
            raise ValueError("Coupled births require positive online learning and trainable new edges")
        self.birth_scale = np.float32(birth_scale)
        self.independent_sign = independent_sign
        self.audit = audit
        self.birth_checks = []
        super().__init__(parameters, masks, schedule, random_seeds, learning_rate=learning_rate)
        # Convert the checkpoint once, then keep effective edge weights as the parameters.
        self.raw /= np.sqrt(self.degree)[..., None]
        self.refresh_matrices()

    def refresh_matrices(self):
        self.degree = self.mask.sum(axis=-1, dtype=np.float32)
        self.matrix = np.where(self.mask, self.raw, np.float32(0.))

    def begin_request(self, *, learn_edges=False):
        # This rule uses the full recurrent derivative, which makes the legacy local trace redundant.
        super().begin_request()

    def finish_request(self, contacted, shown, *, donor=None):
        if donor is not None:
            raise ValueError("A coupled life uses its own receipts")
        observed = (contacted >= 0) & (shown != 0)
        if np.any(observed & (self.trace_ticks == 0)):
            raise ValueError("A contact receipt requires a preceding movement phase")
        events = np.full((len(self.state), 2), -1, dtype=np.int16)
        records = self._empty_events()
        audit = np.full((len(self.state), len(AUDIT_FIELDS)), np.nan, dtype=np.float32)
        if self.audit:
            self.birth_checks.append(audit)
        if not observed.any():
            return events, records

        direction = (2 * contacted - 1) * shown
        ticks = self.trace_ticks.clip(1)
        mean_logit = self.logit_sum / ticks
        credit = np.where(observed, direction * expit(-direction * mean_logit) / ticks, 0.)
        ascent, *other = self.gradients(credit, dense_effective=True)
        self.receive_count += observed

        # Removing effective w gives approximate loss change -dL/dw * w.
        decay = self.schedule.decay
        self.retire_scores[observed] = (decay * self.retire_scores[observed]
                                        - (1 - decay) * (ascent * self.raw)[observed])
        grow = observed & (self.receive_count % self.schedule.growth_every == 0)
        grow_lanes, target, source = self._growth_candidates(grow, ascent)
        arc = grow_lanes, target, source
        proposed = ascent[arc] * self.birth_scale
        if self.independent_sign:
            signs = np.array([2 * self.rng[lane].integers(2) - 1 for lane in grow_lanes], dtype=np.float32)
            proposed = np.abs(proposed) * signs

        recurrent = ascent * self.mask
        recurrent[arc] = proposed
        if self.audit and len(grow_lanes):
            old_matrix = self.matrix[grow_lanes].copy()
            old_params = {key: value[grow_lanes].copy() for key, value in self.parameters.items()}
        # The dormant coordinate starts at zero and participates in the ordinary receipt update.
        self.mask[arc] = True
        self._update_weights(credit, gradients=(recurrent, *other))

        events[grow_lanes, 0] = target * WIDTH + source
        self.newborn_raw[arc] = self.raw[arc]
        self.newborn_trial[arc] = self.trial
        self.newborn_receipt[arc] = self.receive_count[grow_lanes]
        self.newborn_indegree[arc] = self.degree[grow_lanes, target]
        records[grow_lanes, 0] = self.newborn_records(*arc, self.degree[grow_lanes, target])
        if self.audit and len(grow_lanes):
            # Evaluation only. Keep the original within-request inputs, movement mask, carried state,
            # and all other pre-update coefficients fixed while replaying this one perturbation.
            added = self.raw[arc]
            old_matrix[np.arange(len(grow_lanes)), target, source] = added
            replay = self._mean_logit(grow_lanes, old_matrix, old_params) / ticks[grow_lanes]
            labels = direction[grow_lanes]
            old_loss = np.logaddexp(0., -labels * mean_logit[grow_lanes])
            new_loss = np.logaddexp(0., -labels * replay)
            audit[grow_lanes] = np.column_stack((ascent[arc], added, ascent[arc] * added, old_loss - new_loss))

        retire = observed & (self.receive_count % self.schedule.retirement_every == 0)
        self._retire(retire, events, records)
        return events, records

    def _growth_candidates(self, grow, ascent):
        lanes = np.flatnonzero(grow)
        allowed = self.no_loops & ~self.mask[lanes]
        # The receipt step predicts a decrease proportional to the squared sensitivity.
        score = np.where(allowed, np.abs(ascent[lanes]), 0.).reshape(len(lanes), WIDTH * WIDTH)
        pick = score.argmax(axis=1)
        has_gain = score[np.arange(len(lanes)), pick] > 0
        target, source = np.divmod(pick[has_gain], WIDTH)
        return lanes[has_gain], target, source

    def _mean_logit(self, lanes, matrix, params):
        state = self.history[0][1][lanes]
        total = np.zeros(len(lanes), dtype=state.dtype)
        for public_input, _, _, _, _, motor_active in self.history:
            drive = np.einsum("bi,bij->bj", public_input[lanes], params["drive"]) + params["bias"]
            candidate, leak = np.split(drive, 2, axis=-1)
            gate = expit(leak)
            incoming = np.matmul(matrix, state[..., None])[..., 0]
            state = (1 - gate) * np.tanh(candidate + incoming) + gate * state
            if motor_active is not None:
                logit = (state * params["readout"]).sum(axis=1) + params["offset"]
                total += motor_active[lanes] * logit
        return total

    def _retire(self, retire, events, records):
        for lane in np.flatnonzero(retire):
            mask = self.mask[lane]
            scores = self.retire_scores[lane]
            # A newborn must have two chances to adjust to its later receipts before removal.
            old_enough = ((self.newborn_trial[lane] < 0)
                          | (self.receive_count[lane] - self.newborn_receipt[lane] >= 2))
            allowed = (mask & old_enough & (scores > 0)
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
        arc = lane, target, source
        self.retire_scores[arc] = self.raw[arc] = self.newborn_raw[arc] = self.newborn_path[arc] = 0
        self.newborn_trial[arc] = self.newborn_first_change[arc] = -1
        self.newborn_receipt[arc] = self.newborn_indegree[arc] = self.newborn_changes[arc] = 0
