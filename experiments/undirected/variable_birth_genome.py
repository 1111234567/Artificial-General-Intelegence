"""Inherit a variable set of reciprocal birth edges and a local plasticity rule."""
from __future__ import annotations

import math
import random

import torch

from .evolved_plasticity import ACTIONS, INPUTS, JOBS, NODES, RULE_FEATURES, WEIGHT_LIMIT


FOUNDER_MEAN_DEGREE = 16


def make_genome(seed, nodes=NODES):
    """Draw each unordered pair independently for the founder's birth graph."""
    if nodes < INPUTS + JOBS * ACTIONS:
        raise ValueError("graph must contain all fixed input and action ports")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    pair_count = nodes * (nodes - 1) // 2
    weights = torch.randn(pair_count, generator=rng)
    node_bias = torch.randn(nodes, generator=rng) * .025
    node_bias[INPUTS:INPUTS + JOBS * ACTIONS:ACTIONS] += .04
    rule = torch.randn(2, len(RULE_FEATURES), generator=rng) * .18
    rule[0, 0] *= .2
    rule[0, 1] -= .2

    mask = torch.rand(pair_count, generator=rng).lt(FOUNDER_MEAN_DEGREE / (nodes - 1))
    weights.masked_fill_(~mask, 0.)
    norm = float(weights.norm())
    if norm:
        weights.mul_(.65 * math.sqrt(nodes / 2) / norm)
    return {"birth_w": weights.clamp_(-WEIGHT_LIMIT, WEIGHT_LIMIT), "birth_mask": mask,
            "node_bias": node_bias, "rule": rule}


def mutate_genome(parent, seed, wide=False):
    """Copy birth state, propose pair insertions or deletions, and perturb values."""
    rng = torch.Generator(device="cpu").manual_seed(seed)
    chooser = random.Random(seed)
    child = {key: value.clone() for key, value in parent.items()}
    weights, mask = child["birth_w"], child["birth_mask"]
    scale = 2.5 if wide else 1.
    new_weight_sd = .65 / math.sqrt(FOUNDER_MEAN_DEGREE)
    for _ in range(4 if wide else 1):
        # Sample the direction first so an excess of absent pairs does not favor growth.
        adding = bool(chooser.getrandbits(1))
        eligible = mask.ne(adding).nonzero(as_tuple=True)[0]
        if eligible.numel():
            edge = eligible[chooser.randrange(eligible.numel())]
            mask[edge] = adding
            if adding:
                weights[edge] = chooser.gauss(0., new_weight_sd)

    touched = torch.rand(weights.shape, generator=rng).lt(.16 if wide else .09) & mask
    weights.add_(torch.randn(weights.shape, generator=rng) * touched * (.037 * scale))
    weights.clamp_(-WEIGHT_LIMIT, WEIGHT_LIMIT).masked_fill_(~mask, 0.)
    for key, step, limit in (("node_bias", .009, .35), ("rule", .065, 2.)):
        value = child[key]
        value.add_(torch.randn(value.shape, generator=rng) * (step * scale)).clamp_(-limit, limit)
    return child


def candidate(entry, previous, nodes):
    """Build the founder, elite, or child designated by the generation plan."""
    if entry["parent"] is None:
        return make_genome(entry["seed"], nodes)
    parent = previous["genomes"][entry["parent"]]
    if entry["kind"] == "elite":
        return parent
    return mutate_genome(parent, entry["seed"], wide=entry["kind"] == "wide")
