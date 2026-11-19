"""Public-information upper reference with perfect odor assignment."""

import argparse
from functools import lru_cache
from itertools import product
import json
from pathlib import Path

import numpy as np

from . import active_symmetry_world as world
from .spike_analysis import estimates


def likelihood(noise):
    bits = np.array(list(product((-1, 1), repeat=4)), dtype=np.int8)
    latent = bits[bits.sum(1) == 0]
    observed_a, observed_b = np.repeat(bits, len(bits), axis=0), np.tile(bits, (len(bits), 1))
    a = np.where(observed_a[:, None] == latent[None], 1-noise, noise).prod(axis=2)
    b_pos = np.where(observed_b[:, None] == latent[None], 1-noise, noise).prod(axis=2)
    b_neg = np.where(observed_b[:, None] == -latent[None], 1-noise, noise).prod(axis=2)
    return np.column_stack(((a * b_pos).mean(1), (a * b_neg).mean(1)))


@lru_cache(maxsize=None)
def ideal_grid(c0, c1, limit, step_count, noise):
    """Interpolate a separately convex value function to obtain an upper bound."""
    if step_count % 2:
        raise ValueError("The midpoint of the belief grid must be present")
    # Entire observed sign vectors, integrating over the six balanced A patterns.
    table = likelihood(noise)
    prior = np.linspace(0., 1., step_count + 1)
    evidence = prior[:, None] * table[:, 0] + (1-prior[:, None]) * table[:, 1]
    posterior = prior[:, None] * table[:, 0] / evidence
    z = posterior * step_count
    low = np.floor(z).astype(int)
    high = np.minimum(low + 1, step_count)
    proportion = z - low
    # Harvest chooses the better sign after one noisy anchor, and sees no outcome.
    harvest = (1-2*noise) * np.abs(2*prior-1)
    total = np.zeros((len(prior), len(prior)))
    center = step_count // 2
    values = [0.]
    for _ in range(limit):
        left = (total[low] * (1-proportion)[:, :, None] + total[high] * proportion[:, :, None])
        right = (total.T[low] * (1-proportion)[:, :, None] + total.T[high] * proportion[:, :, None])
        add0 = np.einsum("nkr,nk->nr", left, evidence, optimize=False) - c0
        add1 = np.einsum("nkr,nk->nr", right, evidence, optimize=False).T - c1
        total = .5 * (np.maximum(add0, total + harvest[:, None]) + np.maximum(add1, total + harvest[None, :]))
        values.append(float(total[center, center]))
    return tuple(values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("confirmation", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    records = {}
    for file in args.confirmation:
        context = json.loads(file.read_text(encoding="utf-8"))
        panel = world.build(context["held_worlds"])
        cases = {}
        for k in (40, 80):
            values = [ideal_grid(*map(float, cost), 14, k, .04)[int(h)] for cost, h in zip(panel.costs, panel.horizon)]
            cases[str(k)] = estimates(values)
        records[file.name] = {"perfect_matching_grid_bound": cases,
                              "even_knowing_family_expected_ceiling": estimates(.92 * panel.horizon),
                              "actual_public_policy": context["results"]["public"]["net"],
                              "prior_agreement_only_ideal": context["public_ideal_expected_net"]}
    args.out.write_text(json.dumps({"method": "Use ideal known odor-matching and full observed sign vectors; no rule, future visits or private future payouts. "
                                    "Separate convexity in each independent arena belief bounds the value above with chord interpolation on the belief grid.",
                                    "records": records}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
