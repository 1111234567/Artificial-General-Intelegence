"""An end-free sensory-motor graph whose activity emits its own graph edits."""
from dataclasses import dataclass

import numpy as np


N = 16
WRITER = slice(12, 16)
MOTOR_TICKS = 5
CONTACT_TICKS = 4
TASTE_TICKS = 3
LEAK = 0.70
EDGE_STEP = 0.18
BIAS_STEP = 0.035
BIRTH_STEP = 0.12
LIMIT = 2.50
EPSILON = 0.0015
ARMS = ("active", "primitive", "none", "shuffle")
DIAGONAL = np.eye(N, dtype=bool)


@dataclass
class Graph:
    w: np.ndarray
    b: np.ndarray
    mask: np.ndarray

    def clone(self):
        return Graph(self.w.copy(), self.b.copy(), self.mask.copy())

    def lanes(self, indices):
        return Graph(self.w[indices].copy(), self.b[indices].copy(), self.mask[indices].copy())


@dataclass
class Proposal:
    edge_delta: np.ndarray
    bias_delta: np.ndarray
    birth: np.ndarray
    birth_value: np.ndarray
    retire: np.ndarray
    birth_probability: np.ndarray
    retire_probability: np.ndarray


@dataclass
class Receipt:
    edge_delta: np.ndarray
    bias_delta: np.ndarray
    birth_status: np.ndarray
    retire_status: np.ndarray
    edge_count: np.ndarray
    retired_value: np.ndarray


STATUS = {0: "no_op", 1: "applied", 2: "write_protected", 3: "absent_or_occupied",
          4: "connectivity_guard", 5: "writes_disabled"}


def spawn_graph(seed, batch=1):
    """A shared 108-edge birth scaffold and independently randomized edge values."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((N, N), dtype=bool)
    nodes = np.arange(N)
    mask[(nodes + 1) % N, nodes] = True
    # Make raw odor and physical feedback paths short enough for the contact window.
    mask[10:12, :10] = True
    mask[12:16, 6:12] = True
    mask[12:16, :6] = True
    mask[:6, 6:10] = True
    candidates = np.flatnonzero(~(mask | DIAGONAL))
    mask.flat[rng.choice(candidates, 108 - int(mask.sum()), replace=False)] = True
    w = np.where(mask, rng.normal(0.0, 0.115, (N, N)), 0).astype(np.float32)
    b = rng.normal(0.0, 0.025, N).astype(np.float32)
    b[14:16] += 0.33
    return Graph(np.repeat(w[None], batch, 0), np.repeat(b[None], batch, 0),
                 np.repeat(mask[None], batch, 0))


def repeated_graph(graph, batch):
    if len(graph.w) != 1:
        raise ValueError("Expected a single graph to repeat")
    return Graph(np.repeat(graph.w, batch, 0), np.repeat(graph.b, batch, 0),
                 np.repeat(graph.mask, batch, 0))


def rest(graph):
    return np.zeros_like(graph.b)


def sense(left, right, contact=None, taste=None):
    stimulus = np.zeros((len(left), N), dtype=np.float32)
    stimulus[:, :3] = np.asarray(left, dtype=np.float32)
    stimulus[:, 3:6] = np.asarray(right, dtype=np.float32)
    if contact is not None:
        stimulus[:, 6] = np.asarray(contact) < 0
        stimulus[:, 7] = np.asarray(contact) > 0
    if taste is not None:
        stimulus[:, 8] = np.maximum(taste, 0)
        stimulus[:, 9] = np.maximum(-np.asarray(taste), 0)
    return stimulus


def advance(graph, h, stimulus, ticks=1):
    w = graph.w * graph.mask
    drive = graph.b + stimulus
    for _ in range(ticks):
        h = (1.0 - LEAK) * h + LEAK * np.tanh((w @ h[..., None])[..., 0] + drive)
    return h


def motor(h):
    return h[:, 11] - h[:, 10]


def contact_states(graph, h, left, right, side, taste):
    pre = advance(graph, h, sense(left, right, side), CONTACT_TICKS)
    post = advance(graph, pre, sense(left, right, side, taste), TASTE_TICKS)
    return pre, post


def view_arrays(graph, view, view_graph):
    if view == "blind":
        return (np.zeros_like(graph.w), np.zeros_like(graph.b), np.zeros_like(graph.mask))
    observed = graph if view_graph is None else view_graph
    if view == "corrupt":
        return (np.roll(observed.w, 5, axis=1), np.roll(observed.b, 5, axis=1),
                np.roll(observed.mask, 5, axis=1))
    if view != "own":
        raise ValueError(f"Unknown view {view!r}")
    return observed.w, observed.b, observed.mask


def write_fields(pre, post, visible_w, visible_b, visible_mask, tanh=np.tanh):
    """Fixed signal decoding shared between the trainable forward and NumPy deployment.

    A pre-contact source trace and post-contact target activity address the edge
    field. All control signals come from the closed graph. Optional readback
    changes the candidate values using the same gain from a writable node.
    """
    source = tanh(2.0 * pre)
    receiver = tanh(2.0 * (post - 0.35 * pre))
    local = receiver[:, :, None] * source[:, None, :]
    edge_gate = tanh(2.5 * post[:, 12])
    bias_gate = tanh(2.5 * post[:, 13])
    read_gain = post[:, 15]
    edge_delta = EDGE_STEP * (edge_gate[:, None, None] * local
                              + 0.30 * read_gain[:, None, None] * tanh(visible_w) * visible_mask)
    bias_delta = BIAS_STEP * (bias_gate[:, None] * receiver
                              + 0.30 * read_gain[:, None] * tanh(visible_b))
    return local, edge_gate, edge_delta, bias_delta


def structural_addresses(graph, post, local, edge_gate, visible_w, visible_b, visible_mask, key):
    """One optional birth and one optional retirement, with a true no-op each."""
    rng = key if isinstance(key, np.random.Generator) else np.random.default_rng(key)
    shape = graph.w.shape
    gate_birth = np.clip(post[:, 14], 0.0, 1.0)
    gate_retire = np.clip(post[:, 15], 0.0, 1.0)
    birth_probability = 0.16 * gate_birth
    retire_probability = 0.20 * gate_retire
    trigger = rng.random((2, len(post)))
    noise = rng.gumbel(size=(2, *shape))
    readback = post[:, 13, None, None] * (np.tanh(visible_w) + 0.25 * visible_mask
                                          + 0.15 * np.tanh(visible_b[:, :, None]))
    birth_score = 3.0 * local + readback + noise[0]
    retire_score = -3.0 * local - readback + noise[1]
    birth_eligible = np.broadcast_to(~DIAGONAL, shape)
    retire_eligible = birth_eligible

    def address(score, eligible, threshold, draw):
        index = np.where(eligible, score, -np.inf).reshape(len(post), -1).argmax(axis=1)
        selected = (draw < threshold) & eligible.any(axis=(1, 2))
        return np.where(selected, index, -1)

    birth = address(birth_score, birth_eligible, birth_probability, trigger[0])
    retire = address(retire_score, retire_eligible, retire_probability, trigger[1])
    rows, cols = np.divmod(birth.clip(min=0), N)
    birth_value = BIRTH_STEP * np.tanh(2.0 * (local[np.arange(len(post)), rows, cols] + edge_gate))
    return birth, birth_value, retire, birth_probability, retire_probability


def sparse_slots(edge_delta, bias_delta):
    """Rank only emitted magnitudes over the same candidate domain in each view."""
    edge_score = np.where(DIAGONAL, -np.inf, np.abs(edge_delta)).reshape(len(edge_delta), -1)
    best_edges = np.argsort(-edge_score, axis=1, kind="stable")[:, :4]
    chosen = np.zeros_like(edge_score, dtype=bool)
    np.put_along_axis(chosen, best_edges, True, axis=1)
    best_bias = np.abs(bias_delta).argmax(axis=1)
    chosen_bias = np.arange(N) == best_bias[:, None]
    return chosen.reshape(edge_delta.shape), chosen_bias


def proposal(graph, pre, post, key, view="blind", view_graph=None, sparse=False):
    visible_w, visible_b, visible_mask = view_arrays(graph, view, view_graph)
    local, edge_gate, edge_delta, bias_delta = write_fields(pre, post, visible_w, visible_b, visible_mask)
    structural = structural_addresses(graph, post, local, edge_gate, visible_w, visible_b,
                                      visible_mask, key)
    if sparse:
        edge_slot, bias_slot = sparse_slots(edge_delta, bias_delta)
        edge_delta = np.where(edge_slot, edge_delta, 0.0)
        bias_delta = np.where(bias_slot, bias_delta, 0.0)
    return Proposal(edge_delta, bias_delta, *structural)


def connected(mask):
    """Strong connectivity for receiver-by-sender Boolean adjacency."""
    for arcs in (mask, mask.T):
        reached = 1
        frontier = 1
        rows = [sum(1 << int(v) for v in np.flatnonzero(row)) for row in arcs]
        while frontier:
            bit = frontier & -frontier
            frontier ^= bit
            new = rows[bit.bit_length() - 1] & ~reached
            reached |= new
            frontier |= new
        if reached != (1 << N) - 1:
            return False
    return True


def apply(graph, edit, arm="active"):
    """Apply emitted edits under one intervention. Invalid retirements are logged."""
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}")
    result = graph.clone()
    edge_delta = np.zeros_like(graph.w)
    bias_delta = np.zeros_like(graph.b)
    active = arm != "none"
    protect = arm == "primitive"
    if active:
        edge_delta = np.where(graph.mask & (np.abs(edit.edge_delta) >= EPSILON),
                              edit.edge_delta, 0)
        bias_delta = np.where(np.abs(edit.bias_delta) >= EPSILON, edit.bias_delta, 0)
        if protect:
            edge_delta[:, WRITER] = 0
            bias_delta[:, WRITER] = 0
        result.w = np.clip(graph.w + edge_delta, -LIMIT, LIMIT)
        result.b = np.clip(graph.b + bias_delta, -LIMIT, LIMIT)
        edge_delta = result.w - graph.w
        bias_delta = result.b - graph.b
    birth_status = np.zeros(len(graph.w), dtype=np.int8)
    retire_status = np.zeros_like(birth_status)
    retired_value = np.zeros(len(graph.w), dtype=graph.w.dtype)
    for indices, statuses, newborn in ((edit.birth, birth_status, True),
                                        (edit.retire, retire_status, False)):
        for lane in np.flatnonzero(indices >= 0):
            target, source = divmod(int(indices[lane]), N)
            if not active:
                statuses[lane] = 5
            elif protect and target >= WRITER.start:
                statuses[lane] = 2
            elif target == source or bool(result.mask[lane, target, source]) == newborn:
                statuses[lane] = 3
            else:
                result.mask[lane, target, source] = newborn
                if not newborn and not connected(result.mask[lane]):
                    result.mask[lane, target, source] = True
                    statuses[lane] = 4
                    continue
                if not newborn:
                    retired_value[lane] = result.w[lane, target, source]
                result.w[lane, target, source] = edit.birth_value[lane] if newborn else 0
                statuses[lane] = 1
    return result, Receipt(edge_delta, bias_delta, birth_status, retire_status,
                           result.mask.sum(axis=(1, 2)), retired_value)


def transplant_writer(base, donor, origin=None):
    """Copy authoring inputs only; with an origin, transfer the donor change."""
    result = base.clone()
    if origin is None:
        result.w[:, WRITER] = donor.w[:, WRITER]
        result.b[:, WRITER] = donor.b[:, WRITER]
        result.mask[:, WRITER] = donor.mask[:, WRITER]
    else:
        support_changed = origin.mask[:, WRITER] != donor.mask[:, WRITER]
        result.mask[:, WRITER] = np.where(support_changed, donor.mask[:, WRITER],
                                         base.mask[:, WRITER])
        result.w[:, WRITER] += donor.w[:, WRITER] - origin.w[:, WRITER]
        result.b[:, WRITER] += donor.b[:, WRITER] - origin.b[:, WRITER]
        result.w *= result.mask
    return result


def step(graph, h, left, right, taste_by_side, key, arm="active", view="blind",
         view_graph=None, shuffle_indices=None, edit_scale=1.0, sparse=False, proposal_fn=proposal):
    """Run a physical query, one contact, actual taste and one emitted edit round."""
    choice_state = advance(graph, h, sense(left, right), MOTOR_TICKS)
    side = np.where(motor(choice_state) > 0, 1, -1)
    actual_taste = np.asarray(taste_by_side)[np.arange(len(h)), (side > 0).astype(int)]
    pre, post = contact_states(graph, choice_state, left, right, side, actual_taste)
    if arm == "shuffle":
        if shuffle_indices is None:
            raise ValueError("The pairing of shuffled taste must be supplied explicitly")
        _, edit_post = contact_states(graph, choice_state, left, right, side,
                                      actual_taste[shuffle_indices])
    else:
        edit_post = post
    edit = proposal_fn(graph, pre, edit_post, key, view=view, view_graph=view_graph, sparse=sparse)
    if edit_scale != 1.0:
        edit.edge_delta = edit.edge_delta * edit_scale
        edit.bias_delta = edit.bias_delta * edit_scale
    changed, receipt = apply(graph, edit, arm)
    return changed, post, side, actual_taste, edit, receipt
