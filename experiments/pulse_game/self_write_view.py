"""A local structural sensor and transactional writer on randomized recurrent bodies."""

from dataclasses import dataclass

import numpy as np


WIDTH = 4
PAIRS = [(receiver, sender) for receiver in range(WIDTH)
         for sender in range(WIDTH) if receiver != sender]
RECEIVERS, SENDERS = map(np.asarray, zip(*PAIRS, strict=True))
EDGES = len(PAIRS)
SLOTS = EDGES + WIDTH
ADDRESS_CHANNELS = 1 + 2 * WIDTH
STRUCTURE_CHANNELS = 4
TICKS = 3
FEATURES = ADDRESS_CHANNELS + STRUCTURE_CHANNELS + WIDTH + 2 * TICKS * WIDTH
DELTAS = np.array([-0.5, -0.25, 0.25, 0.5], dtype=np.float32)
REMOVE = 8
STRIDE = 9
NOOP = SLOTS * STRIDE
ACTION_COUNT = NOOP + 1
MODES = ("veridical", "current_only", "blind", "yoked", "stale")


@dataclass(frozen=True)
class Body:
    exists: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class Trial:
    reference: Body
    damaged: Body
    drive: np.ndarray
    disturbance: np.ndarray
    inverse: np.ndarray
    yoke: np.ndarray


@dataclass(frozen=True)
class Receipt:
    body: Body
    valid: np.ndarray
    changed: np.ndarray
    slot: np.ndarray
    operation: np.ndarray
    delta: np.ndarray


def decode(action):
    encoded = np.minimum(action, NOOP - 1)
    return encoded // STRIDE, encoded % STRIDE


def apply(body, action):
    """Mutate one addressed port per lane, using actual support for validation."""
    action = np.asarray(action, dtype=np.int64)
    if action.shape != body.values.shape[:1] or np.any((action < 0) | (action > NOOP)):
        raise ValueError("One in-range command is required per body")
    rows = np.arange(len(action))
    slot, operation = decode(action)
    edge = slot < EDGES
    exists = body.exists[rows, slot]
    value = body.values[rows, slot]
    shift = operation < 4
    add = (operation >= 4) & (operation < REMOVE)
    remove = operation == REMOVE
    dose = DELTAS[operation % 4]
    next_value = np.where(shift, value + dose, np.where(add, dose, 0))
    shift_valid = shift & exists & (np.abs(next_value) <= 1.25) & (~edge | (next_value != 0))
    removable = remove & exists & np.isin(value, DELTAS)
    valid = shift_valid | (edge & ((add & ~exists) | removable))
    changed = valid & (action != NOOP)
    values = body.values.copy()
    support = body.exists.copy()
    kept_rows, kept_slots = rows[changed], slot[changed]
    values[kept_rows, kept_slots] = next_value[changed]
    support[kept_rows, kept_slots] = ~remove[changed]
    return Receipt(Body(support, values), valid | (action == NOOP), changed,
                   np.where(action == NOOP, -1, slot), np.where(action == NOOP, -1, operation),
                   np.where(changed, next_value - value, 0).astype(np.float32))


def exact_rows(first, second):
    return ((first.exists == second.exists) & (first.values == second.values)).all(axis=1)


def inverse_action(before, action, receipt):
    slot, operation = decode(action)
    old_value = before.values[np.arange(len(action)), slot]
    dose_index = np.argmax(old_value[:, None] == DELTAS, axis=1)
    inverse_op = np.where(operation < 4, 3 - operation,
                          np.where(operation == REMOVE, 4 + dose_index, REMOVE))
    if np.any(receipt.changed & (operation == REMOVE) & ~np.isin(old_value, DELTAS)):
        raise ValueError("Removed value is outside the available add magnitudes")
    return np.where(receipt.changed, slot * STRIDE + inverse_op, NOOP)


def random_trial(seed, batch):
    if batch < 2:
        raise ValueError("Yoked readback requires at least two bodies")
    rng = np.random.default_rng(seed)
    exists = np.column_stack((rng.random((batch, EDGES)) < 0.52,
                              np.ones((batch, WIDTH), dtype=bool)))
    # Eligible removes always have an attainable inverse in the intervention alphabet.
    exists[:, 0] |= ~exists[:, :EDGES].any(axis=1)
    exists[:, EDGES - 1] &= ~exists[:, :EDGES].all(axis=1)
    values = rng.choice(DELTAS, size=exists.shape) * exists
    reference = Body(exists, values)
    rows = np.arange(batch)
    present_edge = np.argmax(np.where(exists[:, :EDGES], rng.random((batch, EDGES)), -1), axis=1)
    missing_edge = np.argmax(np.where(~exists[:, :EDGES], rng.random((batch, EDGES)), -1), axis=1)
    bias_slot = EDGES + rng.integers(WIDTH, size=batch)
    kind = rng.integers(5, size=batch)
    edge_value = values[rows, present_edge]
    shift_valid = edge_value[:, None] + DELTAS != 0
    edge_shift = np.argmax(np.where(shift_valid, rng.random((batch, 4)), -1), axis=1)
    action = np.select([kind == 0, kind == 1, kind == 2, kind == 3],
                       [present_edge * STRIDE + edge_shift,
                        missing_edge * STRIDE + rng.integers(4, 8, size=batch),
                        present_edge * STRIDE + REMOVE,
                        bias_slot * STRIDE + rng.integers(4, size=batch)], default=NOOP)
    receipt = apply(reference, action)
    if not receipt.valid.all():
        raise AssertionError("Unexpected invalid disturbance")
    inverse = inverse_action(reference, action, receipt)
    drive = rng.choice(DELTAS, size=(batch, WIDTH))
    yoke = np.roll(np.arange(batch), int(rng.integers(1, batch)))
    return Trial(reference, receipt.body, drive, action, inverse, yoke)


def run_body(body, drive, ticks=TICKS):
    """Every measured response is recomputed from the actual sparse recurrence."""
    matrix = np.zeros((len(body.values), WIDTH, WIDTH), dtype=np.float32)
    matrix[:, RECEIVERS, SENDERS] = body.values[:, :EDGES] * body.exists[:, :EDGES]
    bias = body.values[:, EDGES:]
    state = np.broadcast_to(drive, (len(body.values), *drive.shape[1:]))
    history = []
    for _ in range(ticks):
        state = np.tanh(np.einsum("brs,bps->bpr", matrix, state) + bias[:, None, :])
        history.append(state)
    return np.stack(history, axis=1)


def bank_drive(trial):
    eye = 0.5 * np.eye(WIDTH, dtype=np.float32)
    return np.broadcast_to(np.concatenate((eye, -eye))[None, :, :],
                           (len(trial.drive), 2 * WIDTH, WIDTH))


def motor_error(body, trial, reference):
    return np.square(run_body(body, bank_drive(trial)) - reference).mean(axis=(1, 2, 3))


def public_signals(trial):
    drive = trial.drive[:, None, :]
    reference_trace = run_body(trial.reference, drive).reshape(len(drive), -1)
    damaged_trace = run_body(trial.damaged, drive).reshape(len(drive), -1)
    return np.concatenate((trial.drive, reference_trace, damaged_trace), axis=1)


def local_features(trial, mode, public=None):
    """Expose local raw structure and the same ordinary activity in each arm."""
    if mode not in MODES:
        raise ValueError(f"Unknown observation mode {mode!r}")
    batch = len(trial.drive)
    old = trial.reference
    current = trial.damaged
    if mode == "yoked":
        donor = trial.yoke
        old = Body(old.exists[donor], old.values[donor])
        current = Body(current.exists[donor], current.values[donor])
    if mode == "stale":
        current = old
    blank = np.zeros_like(trial.reference.values)
    old_values = blank if mode in ("blind", "current_only") else old.values
    old_exists = blank if mode in ("blind", "current_only") else old.exists
    new_values = blank if mode == "blind" else current.values
    new_exists = blank if mode == "blind" else current.exists
    edge = np.arange(SLOTS) < EDGES
    eye = np.eye(WIDTH, dtype=np.float32)
    receiver = eye[np.concatenate((RECEIVERS, np.arange(WIDTH)))]
    sender = np.concatenate((eye[SENDERS], np.zeros_like(eye)))
    ports = np.column_stack((edge, receiver, sender))
    addressing = np.broadcast_to(ports, (batch, SLOTS, ADDRESS_CHANNELS))
    structure = np.stack((old_exists, old_values, new_exists, new_values), axis=-1)
    if public is None:
        public = public_signals(trial)
    return np.concatenate((addressing, structure,
                           np.broadcast_to(public[:, None, :], (batch, SLOTS, public.shape[1]))),
                          axis=-1).astype(np.float32)
