"""A continuous graph with receipt-trained fast contextual and slow graph weights."""

import numpy as np

from . import coupled_birth
from .free_graph_model import strongly_connected
from .joint_live_model import WIDTH, expit


MEMORY = 48
FAST_FIELDS = ("head_step", "context_write", "head_before", "head_feature", "context_prediction", "match_mass")


class Population(coupled_birth.Population):
    """The only teacher is the signed taste delivered at the body's own contact."""

    def __init__(self, *args, config, contextual=True, **kwargs):
        self.config = config
        self.context_gain = np.float32(config["context_gain"] if contextual else 0)
        self.temperature = np.float32(config["temperature"])
        super().__init__(*args, birth_scale=config["birth_scale"], **kwargs)
        n = len(self.state)
        self.head = np.ones(n, dtype=np.float32)
        self.keys = np.zeros((n, MEMORY, 2, WIDTH), dtype=np.float32)
        self.key_mass = np.zeros((n, MEMORY), dtype=np.float32)
        self.coefficients = np.zeros_like(self.key_mass)
        self.fast_step = np.zeros((n, len(FAST_FIELDS)), dtype=np.float32)

    def begin_request(self, *, learn_edges=False):
        super().begin_request()
        self.head_sum = np.zeros_like(self.head)
        self.key_slot = np.zeros(len(self.state), dtype=np.int8)
        self.key = np.zeros((len(self.state), 2, WIDTH), dtype=np.float32)
        self.key_derivative = np.zeros_like(self.key)
        self.key_norm = np.ones((len(self.state), 2, 1), dtype=np.float32)
        self.context = np.zeros_like(self.head)
        self.match_mass = np.zeros_like(self.head)
        self.fast_step.fill(0)

    def _read_context(self, lanes):
        # Similarity between the circuit changes caused by the two marked public
        # pulses. Addresses are never discrete odor identities or table cells.
        key = self.key[lanes]
        keys = self.keys[lanes]
        spread = np.float32(self.config["kernel_strength"])
        similarity = np.einsum("bkij,bij->bk", keys, key, optimize=False) * .5
        weight = self.key_mass[lanes] * np.exp(spread * (similarity - 1))
        mass = weight.sum(axis=1)
        denominator = mass + np.float32(self.config["memory_prior"])
        coefficients = self.coefficients[lanes]
        context = (weight * coefficients).sum(axis=1) / denominator
        sensitivity = (weight * (coefficients - context[:, None])) * (spread * .5 / denominator[:, None])
        derivative = np.einsum("bk,bkij->bij", sensitivity, keys, optimize=False)
        tangent = derivative - (derivative * key).sum(axis=-1, keepdims=True) * key
        self.key_derivative[lanes] = tangent / self.key_norm[lanes]
        self.context[lanes] = context
        self.match_mass[lanes] = mass

    def act(self, public_input, motor_active=None, *, record=False):
        p = self.parameters
        drive = np.einsum("bi,bij->bj", public_input, p["drive"]) + p["bias"]
        candidate_input, gate_input = np.split(drive, 2, axis=-1)
        prior = self.state
        incoming = np.matmul(self.matrix, prior[..., None])[..., 0]
        candidate = np.tanh(candidate_input + incoming)
        gate = expit(gate_input)
        self.state = (1 - gate) * candidate + gate * prior
        raw_logit = (p["readout"] * self.state).sum(axis=1) + p["offset"]
        response = np.tanh(raw_logit / self.temperature)
        first, second = np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
        if self.context_gain and record and motor_active is None:
            # The marked carrier is a continuous public input; this branch has
            # no access to trial phases, table identities, or rewarded sides.
            marked = public_input[:, 3] > .5
            first = np.flatnonzero(marked & (self.key_slot == 0))
            second = np.flatnonzero(marked & (self.key_slot == 1))
            for slot, lanes in enumerate((first, second)):
                surprise = self.state[lanes] - prior[lanes]
                norm = np.sqrt((surprise ** 2).sum(axis=1, keepdims=True)).clip(1e-8)
                self.key[lanes, slot] = surprise / norm
                self.key_norm[lanes, slot] = norm
                self.key_slot[lanes] += 1
            if len(second):
                self._read_context(second)
        logits = self.head * response + self.context_gain * self.context
        if motor_active is not None:
            self.head_sum += motor_active * response
            self.logit_sum += motor_active * logits
            self.trace_ticks += motor_active
        if record:
            self.history.append((public_input, prior, candidate, gate, self.state, motor_active, response, first, second))
        return logits >= 0

    def gradients(self, credit, *, dense_effective=True):
        """Differentiate the actual blended motor logit through this request."""
        delay = np.zeros_like(self.state)
        readout = np.zeros_like(delay)
        offset = np.zeros_like(credit)
        key_credit = self.context_gain * credit * self.trace_ticks
        candidate_derivatives, gate_derivatives = [], []
        for inputs, prior, candidate, gate, current, active, head, first, second in reversed(self.history):
            if active is not None:
                amplitude = active * credit * self.head * (1 - head ** 2) / self.temperature
                delay += self.parameters["readout"] * amplitude[:, None]
                readout += current * amplitude[:, None]
                offset += amplitude
            for slot, lanes in enumerate((first, second)):
                delay[lanes] += self.key_derivative[lanes, slot] * key_credit[lanes, None]
            local = delay * (1 - gate) * (1 - candidate ** 2)
            candidate_derivatives.append(local)
            gate_derivatives.append(delay * (prior - candidate) * gate * (1 - gate))
            delay = gate * delay + np.matmul(local[:, None, :], self.matrix)[:, 0, :]
            for slot, lanes in enumerate((first, second)):
                delay[lanes] -= self.key_derivative[lanes, slot] * key_credit[lanes, None]
        candidate = np.stack(candidate_derivatives[::-1], axis=1)
        gate = np.stack(gate_derivatives[::-1], axis=1)
        prior = np.stack([record[1] for record in self.history], axis=1)
        inputs = np.stack([record[0] for record in self.history], axis=-1)
        local = np.concatenate((candidate, gate), axis=-1)
        recurrent = np.matmul(candidate.transpose(0, 2, 1), prior)
        if not dense_effective:
            recurrent *= self.mask
        return recurrent, np.matmul(inputs, local), local.sum(axis=1), readout, offset

    def finish_request(self, contacted, shown, *, donor=None):
        if donor is not None:
            raise ValueError("An open life updates from its own body's receipt")
        observed = (contacted >= 0) & (shown != 0)
        if np.any(observed & (self.trace_ticks == 0)):
            raise ValueError("A contact receipt requires a preceding movement phase")
        events = np.full((len(self.state), 2), -1, dtype=np.int16)
        records = self._empty_events()
        if not observed.any():
            return events, records

        ticks = self.trace_ticks.clip(1)
        direction = (2 * contacted - 1) * shown
        overall = self.logit_sum / ticks
        credit = np.where(observed, direction * expit(-direction * overall) / ticks, 0.)
        ascent, *other = self.gradients(credit)
        self.receive_count += observed
        decay = self.schedule.decay
        # The squared effective coefficient sets a trust-region cost. Any
        # existing edge can still be removed when observed improvement pays it.
        removal = -ascent * self.raw - np.float32(self.config["retirement_cost"]) * self.raw ** 2
        self.retire_scores[observed] = decay * self.retire_scores[observed] + (1 - decay) * removal[observed]
        growth = observed & (self.receive_count % self.schedule.growth_every == 0)
        retire = observed & (self.receive_count % self.schedule.retirement_every == 0)
        grow_lanes, target, source = self._growth_candidates(growth, ascent)
        arc = grow_lanes, target, source
        recurrent = ascent * self.mask
        recurrent[arc] = ascent[arc] * np.float32(self.config["birth_scale"])
        self.mask[arc] = True
        self._update_weights(credit, gradients=(recurrent, *other))
        events[grow_lanes, 0] = target * WIDTH + source
        self.newborn_raw[arc] = self.raw[arc]
        self.newborn_trial[arc] = self.trial
        self.newborn_receipt[arc] = self.receive_count[grow_lanes]
        self.newborn_indegree[arc] = self.degree[grow_lanes, target]
        records[grow_lanes, 0] = self.newborn_records(*arc, self.degree[grow_lanes, target])
        self._retire(retire, events, records)
        self._fast_update(direction, observed, ticks)
        return events, records

    def _retire(self, retire, events, records):
        for lane in np.flatnonzero(retire):
            mask = self.mask[lane]
            scores = self.retire_scores[lane]
            allowed = mask & (scores > 0) & (mask.sum(axis=0) > 1)[None, :] & (mask.sum(axis=1) > 1)[:, None]
            if events[lane, 0] >= 0:
                allowed.flat[events[lane, 0]] = False
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

    def _fast_update(self, direction, observed, ticks):
        lanes = np.flatnonzero(observed)
        mean_head = self.head_sum[lanes] / ticks[lanes]
        prior_head = self.head[lanes].copy()
        predicted = prior_head * mean_head
        error = direction[lanes] - predicted
        rate = np.float32(self.config["head_rate"])
        # Predict a signed physical label. A positive initial value preserves
        # the checkpoint's output. Subsequent signs are learned by regression.
        self.head[lanes] += rate * mean_head * error / (.25 + mean_head ** 2)
        self.fast_step[lanes, 0] = self.head[lanes] - prior_head
        self.fast_step[lanes, 2] = prior_head
        self.fast_step[lanes, 3] = mean_head
        self.fast_step[lanes, 4] = self.context[lanes]
        self.fast_step[lanes, 5] = self.match_mass[lanes]
        if self.context_gain:
            self.key_mass[lanes] *= np.float32(self.config["memory_decay"])
            ready = self.key_slot[lanes] == 2
            lane = lanes[ready]
            position = (self.receive_count[lane] - 1) % MEMORY
            self.keys[lane, position] = self.key[lane]
            self.key_mass[lane, position] = 1.
            self.coefficients[lane, position] = error[ready]
            self.fast_step[lane, 1] = error[ready]
