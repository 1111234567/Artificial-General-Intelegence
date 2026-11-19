"""Receipt-only weight updates and uncapped structural editing in one continuous circuit."""

import numpy as np

from .live_module_edits import Population as EdgePopulation, WIDTH, expit


WEIGHT_NAMES = ("raw", "drive", "bias", "readout", "offset")
BIRTH_FIELDS = ("trial", "receipt", "initial_raw", "initial_indegree", "end_raw",
                "end_indegree", "later_receipts", "changed_receipts", "first_change", "path_raw")
GRADIENT_CLIP = 4.


class Population(EdgePopulation):
    """One trainable raw value per active arc, with the original receiver normalization."""

    def __init__(self, parameters, masks, schedule, random_seeds, *, learning_rate=0., pin_newborn=False):
        self.learning_rate = learning_rate
        self.pin_newborn = pin_newborn
        self.raw = np.where(masks, parameters["gain"][..., None], np.float32(0))
        self.newborn_trial = np.full(masks.shape, -1, dtype=np.int16)
        self.newborn_receipt = np.zeros(masks.shape, dtype=np.int16)
        self.newborn_raw = np.zeros(masks.shape, dtype=np.float32)
        self.newborn_indegree = np.zeros(masks.shape, dtype=np.int16)
        self.newborn_changes = np.zeros(masks.shape, dtype=np.int16)
        self.newborn_first_change = np.full(masks.shape, -1, dtype=np.int16)
        self.newborn_path = np.zeros(masks.shape, dtype=np.float32)
        self.weight_step = np.zeros((len(masks), len(WEIGHT_NAMES)), dtype=np.float32)
        self.input_step = np.zeros(parameters["drive"].shape[:2], dtype=np.float32)
        self.trial = -1
        super().__init__({name: value.copy() for name, value in parameters.items()}, masks, schedule, random_seeds)

    def refresh_matrices(self):
        self.degree = self.mask.sum(axis=-1, dtype=np.float32)
        self.matrix = np.where(self.mask, self.raw / np.sqrt(self.degree)[..., None], np.float32(0))

    def begin_request(self, *, learn_edges=False):
        super().begin_request()
        self.learn_edges = learn_edges
        self.history = []
        self.trial += 1
        self.weight_step.fill(0)
        self.input_step.fill(0)

    def act(self, public_input, motor_active=None, *, record=False):
        p = self.parameters
        drive = np.einsum("bi,bij->bj", public_input, p["drive"]) + p["bias"]
        candidate_input, gate_input = np.split(drive, 2, axis=-1)
        prior = self.state
        incoming = np.matmul(self.matrix, prior[..., None])[..., 0]
        candidate = np.tanh(candidate_input + incoming)
        gate = expit(gate_input)
        self.state = (1 - gate) * candidate + gate * prior
        logit = (p["readout"] * self.state).sum(axis=1) + p["offset"]
        if motor_active is not None:
            if self.learn_edges:
                derivative = (1 - gate) * (1 - candidate ** 2) * p["readout"] * motor_active[:, None]
                self.arc_trace += derivative[..., None] * prior[:, None, :]
                self.common_trace += derivative * incoming
            self.logit_sum += motor_active * logit
            self.trace_ticks += motor_active
        if record and self.learning_rate:
            self.history.append((public_input, prior, candidate, gate, self.state, motor_active))
        return logit >= 0

    def gradients(self, credit, *, dense_effective=False):
        """Exact gradient of the within-request logit surrogate, with carried state fixed."""
        delay = np.zeros_like(self.state)
        readout = np.zeros_like(delay)
        candidate_derivatives, gate_derivatives = [], []
        for inputs, prior, candidate, gate, current, active in reversed(self.history):
            if active is not None:
                amplitude = (active * credit)[:, None]
                delay += self.parameters["readout"] * amplitude
                readout += current * amplitude
            local = delay * (1 - gate) * (1 - candidate ** 2)
            candidate_derivatives.append(local)
            gate_derivatives.append(delay * (prior - candidate) * gate * (1 - gate))
            delay = gate * delay + np.matmul(local[:, None, :], self.matrix)[:, 0, :]
        candidate = np.stack(candidate_derivatives[::-1], axis=1)
        gate = np.stack(gate_derivatives[::-1], axis=1)
        prior = np.stack([record[1] for record in self.history], axis=1)
        inputs = np.stack([record[0] for record in self.history], axis=-1)
        local = np.concatenate((candidate, gate), axis=-1)
        recurrent = np.matmul(candidate.transpose(0, 2, 1), prior)
        if not dense_effective:
            recurrent = recurrent / np.sqrt(self.degree)[..., None] * self.mask
        return recurrent, np.matmul(inputs, local), local.sum(axis=1), readout, credit * self.trace_ticks

    def _update_weights(self, credit, *, gradients=None):
        if gradients is None:
            gradients = self.gradients(credit)
        squares = sum((gradient.reshape(len(credit), -1) ** 2).sum(axis=1) for gradient in gradients)
        scale = self.learning_rate * np.minimum(1, GRADIENT_CLIP / np.sqrt(np.maximum(squares, 1e-12)))
        for i, (name, gradient) in enumerate(zip(WEIGHT_NAMES, gradients, strict=True)):
            values = self.raw if name == "raw" else self.parameters[name]
            prior = values.copy()
            if name == "raw" and self.pin_newborn:
                gradient = np.where(self.newborn_trial < 0, gradient, 0.)
            values += gradient * scale.reshape((-1,) + (1,) * (values.ndim - 1))
            change = np.abs(values - prior)
            self.weight_step[:, i] = change.reshape(len(credit), -1).max(axis=1)
            if name == "drive":
                self.input_step[:] = change.max(axis=-1)
            if name == "raw":
                changed = (self.newborn_trial >= 0) & (change > 0)
                self.newborn_first_change[changed & (self.newborn_first_change < 0)] = self.trial
                self.newborn_changes += changed
                self.newborn_path += np.where(changed, change, 0)
        self.refresh_matrices()

    def finish_request(self, contacted, shown, *, donor=None):
        observed = (contacted >= 0) & (shown != 0)
        if np.any(observed & (self.trace_ticks == 0)):
            raise ValueError("A contact receipt requires a preceding movement phase")
        ticks = self.trace_ticks.clip(1)
        side = 2 * contacted - 1
        direction = side * shown
        credit = np.where(observed, direction * expit(-direction * (self.logit_sum / ticks)) / ticks, 0.)
        self.receive_count += observed
        if self.learn_edges:
            d = self.degree
            grow_value = self.parameters["gain"]
            higher = np.sqrt(d + 1)
            lower = np.sqrt(np.maximum(1, d - 1))
            add = (grow_value / higher)[..., None] * self.arc_trace
            add += ((np.sqrt(d) / higher - 1) * self.common_trace)[..., None]
            remove = (-self.raw / lower[..., None]) * self.arc_trace
            remove += ((np.sqrt(d) / lower - 1) * self.common_trace)[..., None]
            for score, change in ((self.grow_scores, add), (self.retire_scores, remove)):
                decay = self.schedule.decay
                score[observed] = decay * score[observed] + (1 - decay) * (credit[:, None, None] * change)[observed]
        if self.learning_rate and observed.any():
            self._update_weights(credit)
        if self.learn_edges:
            growth = observed & (self.receive_count % self.schedule.growth_every == 0)
            retirement = observed & (self.receive_count % self.schedule.retirement_every == 0)
            return self._edit(growth, retirement, random=False)
        if donor is not None:
            return self.random_edit(donor)
        return np.full((len(self.state), 2), -1, dtype=np.int16), self._empty_events()

    def _empty_events(self):
        return np.full((len(self.state), 2, len(BIRTH_FIELDS)), np.nan, dtype=np.float32)

    def newborn_records(self, lanes, target, source, end_degree):
        arc = lanes, target, source
        birth_receipt = self.newborn_receipt[arc]
        return np.stack((self.newborn_trial[arc], birth_receipt, self.newborn_raw[arc], self.newborn_indegree[arc],
                         self.raw[arc], end_degree, self.receive_count[lanes] - birth_receipt,
                         self.newborn_changes[arc], self.newborn_first_change[arc], self.newborn_path[arc]), axis=-1)

    def _edit(self, growth, retirement, *, random):
        old_degree = self.degree
        events = super()._edit(growth, retirement, random=random)
        records = self._empty_events()
        lanes = np.flatnonzero(events[:, 0] >= 0)
        target, source = np.divmod(events[lanes, 0], WIDTH)
        arc = lanes, target, source
        self.raw[arc] = self.newborn_raw[arc] = self.parameters["gain"][lanes, target]
        self.newborn_trial[arc] = self.trial
        self.newborn_receipt[arc] = self.receive_count[lanes]
        self.newborn_indegree[arc] = self.degree[lanes, target]
        records[lanes, 0] = self.newborn_records(*arc, self.degree[lanes, target])

        lanes = np.flatnonzero(events[:, 1] >= 0)
        target, source = np.divmod(events[lanes, 1], WIDTH)
        arc = lanes, target, source
        before_removal = old_degree[lanes, target] + ((events[lanes, 0] // WIDTH) == target)
        records[lanes, 1] = self.newborn_records(*arc, before_removal)
        self.raw[arc] = self.newborn_raw[arc] = self.newborn_path[arc] = 0
        self.newborn_trial[arc] = self.newborn_first_change[arc] = -1
        self.newborn_receipt[arc] = self.newborn_indegree[arc] = self.newborn_changes[arc] = 0
        if (events >= 0).any():
            self.refresh_matrices()
        return events, records
