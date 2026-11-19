"""Score a changed first choice only when both physical bodies make contact."""

import numpy as np


def first_contact_pairs(protocol, result):
    """Pair the first probe of each life by the protocol's two-family row order."""
    query = protocol.query
    n, _ = query.shape
    if n % 2 or n == 0 or not np.all(query.any(axis=1)):
        raise ValueError("Expected two equal families with a first probe in every life")
    row = np.arange(n)
    first = query.argmax(axis=1)
    sides = result.contact_side[row, first]
    relative = sides == protocol.related_side[row, first]
    a, b = np.split(sides, 2)
    x, y = np.split(relative, 2)
    coverage = (a >= 0) & (b >= 0)
    switched = coverage & (x != y)
    paired = int(coverage.sum())
    conditional = float(switched.sum() / paired) if paired else None
    return {
        "first_side_even": a,
        "first_side_odd": b,
        "paired_first_contact": coverage.astype(float),
        "switched": switched.astype(float),
        "switch_given_paired_first_contact": np.asarray(conditional),
    }
