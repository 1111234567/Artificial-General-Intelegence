"""Recover per-generation edge ages and direct coefficient changes."""

import numpy as np


COHORTS = ('initial', 'novel_first', 'novel_regrown', 'initial_regrown')
ABSENT = -2
INITIAL = -1
CHANGE_FLOORS = (0., 1e-6, 1e-4, 1e-3, 1e-2)
QUANTILES = (0., .1, .25, .5, .75, .9, .99, 1.)


def edge_identity(initial_mask, edits, final_mask):
    """Trace each receiver/sender slot through additions and retirements."""
    if initial_mask.shape != final_mask.shape or edits.shape[:-2] != initial_mask.shape[:-2]:
        raise ValueError('Expected matching lives in initial masks, edits, and final masks')
    width = initial_mask.shape[-1]
    size = width * width
    start = initial_mask.reshape(-1, size)
    births = np.where(start, INITIAL, ABSENT).astype(np.int16)
    generation = start.astype(np.int16)
    events = edits.reshape(len(start), edits.shape[-2], 2)
    for trial in range(events.shape[1]):
        additions, retirements = events[:, trial].T
        adding = np.flatnonzero(additions >= 0)
        retiring = np.flatnonzero(retirements >= 0)
        create = (adding, additions[adding])
        retire = (retiring, retirements[retiring])
        if (np.any(additions[adding] // width == additions[adding] % width)
                or np.any(births[create] != ABSENT)
                or np.any(births[retire] == ABSENT)
                or np.any(additions[retiring] == retirements[retiring])):
            raise RuntimeError(f'Invalid graph edit at trial {trial}')
        births[create] = trial
        generation[create] += 1
        births[retire] = ABSENT
    birth_trial = births.reshape(initial_mask.shape)
    generation = generation.reshape(initial_mask.shape)
    if not np.array_equal(birth_trial != ABSENT, final_mask):
        raise RuntimeError('Final support differs from edit history')
    cohort = np.full(final_mask.shape, -1, dtype=np.int8)
    cohort[birth_trial == INITIAL] = COHORTS.index('initial')
    cohort[final_mask & ~initial_mask & (generation == 1)] = COHORTS.index('novel_first')
    cohort[final_mask & ~initial_mask & (generation > 1)] = COHORTS.index('novel_regrown')
    cohort[final_mask & initial_mask & (generation > 1)] = COHORTS.index('initial_regrown')
    if not np.array_equal(cohort >= 0, final_mask):
        raise RuntimeError('A surviving edge lacks a generation class')
    return birth_trial, generation, cohort


def combine_coefficients(initial_raw, new_raw, final_raw, birth_trial):
    """Use each surviving generation's actual coefficient at birth as reference."""
    reference = np.where(birth_trial == INITIAL, initial_raw, new_raw)
    alive = birth_trial != ABSENT
    if not np.isfinite(reference[alive]).all() or not np.isfinite(final_raw[alive]).all():
        raise RuntimeError('An active edge has a nonfinite coefficient')
    return np.where(alive, reference, 0)


def age_in_receipts(birth_trial, shown):
    """Count genuine signed contacts strictly after each birth request."""
    signed = (shown != 0).astype(np.int16)
    cumulative = np.concatenate((np.zeros_like(signed[..., :1]), signed.cumsum(axis=-1)), axis=-1)
    at_birth = np.take_along_axis(cumulative[..., None, None, :],
                                 (birth_trial.clip(INITIAL) + 1)[..., None], axis=-1)[..., 0]
    return np.where(birth_trial != ABSENT, cumulative[..., -1, None, None] - at_birth, -1)


def summarize(reference, final, age, received):
    count = len(reference)
    if not count:
        return {'edges': 0}
    delta = final.astype(np.float64) - reference
    mag = np.abs(delta)
    opposite = (np.sign(reference) * np.sign(final)) == -1
    return dict(edges=count,
                absolute_delta_quantiles=np.quantile(mag, QUANTILES).tolist(),
                signed_delta_quantiles=np.quantile(delta, QUANTILES).tolist(),
                age_requests_quantiles=np.quantile(age, QUANTILES).tolist(),
                age_signed_receipts_quantiles=np.quantile(received, QUANTILES).tolist(),
                fraction_exceeding_thresholds=[float((mag > limit).mean()) for limit in CHANGE_FLOORS],
                opposite_nonzero_signs=int(opposite.sum()),
                from_zero_to_nonzero=int(((reference == 0) & (final != 0)).sum()),
                from_nonzero_to_zero=int(((reference != 0) & (final == 0)).sum()),
                no_post_birth_receipt_edges=int((received == 0).sum()),
                changed_without_post_birth_receipt=int(((received == 0) & (mag > 0)).sum()))
