"""Independently sample raw/context every request and pair natural revisits per life."""

from dataclasses import replace
from functools import lru_cache

import numpy as np
import torch

from . import uncapped_private_symbol_iid_world as iid
from .evolved_plasticity_world import FIRST_FINISH, N_SYMBOLS, NEW, RAW


def schedule(*, seed, actors, visits, device="cpu"):
    tape = iid.schedule(seed=seed, actors=actors, visits=visits, device=device)
    number = int(seed) % (1 << 64)
    raw = np.column_stack([
        np.random.default_rng(np.random.SeedSequence([number & 0xffffffff, number >> 32, actor, 4]))
        .integers(N_SYMBOLS, size=len(tape.raw)) for actor in range(actors)
    ])
    raw = torch.from_numpy(raw).to(tape.target.device)
    index = torch.arange(actors, device=tape.target.device)
    target = FIRST_FINISH + tape.maps[index, tape.context, raw]
    port = tape.public[..., 0, :]
    port[..., RAW] = (torch.nn.functional.one_hot(raw, N_SYMBOLS)[:, None] * port[..., NEW, None])
    return replace(tape, raw=raw, target=target)


def training_schedule(*, seed, actors, visits, device="cpu"):
    if actors < 4 or actors % 4:
        raise ValueError("training uses four equal, nonempty world blocks")
    tapes = [schedule(seed=seed * 4 + block, actors=actors // 4, visits=visits, device=device)
             for block in range(4)]
    dims = dict(public=2, stage=2, target=1, raw=1, context=1, maps=0)
    return replace(tapes[0], **{key: torch.cat([getattr(t, key) for t in tapes], dim=dim)
                               for key, dim in dims.items()})


@lru_cache(maxsize=2)
def tape_on_cpu(seed, actors, visits, training):
    factory = training_schedule if training else schedule
    return factory(seed=seed, actors=actors, visits=visits, device="cpu")


def partners(raw, context):
    """For each eligible A revisit, give its B partner's request index or -1.

    Both contexts must have occurred before either member. Waiting/omitting
    an answer cannot change membership; no pair crosses an actor boundary.
    """
    total, actors = raw.shape
    indices = torch.arange(total, device=raw.device)[:, None, None]
    result = torch.full_like(raw, -1)
    for symbol in range(N_SYMBOLS):
        a = raw.eq(symbol) & context.eq(0)
        b = raw.eq(symbol) & context.eq(1)
        # Exclude both initial presentations and any occurrence before both have appeared.
        experienced = (a.cumsum(0) - a.long()).gt(0) & (b.cumsum(0) - b.long()).gt(0)
        a = a & experienced
        b = b & experienced
        match = (b[:, None, :] & a[None, :, :]
                 & b.cumsum(0)[:, None, :].eq(a.cumsum(0)[None, :, :]))
        result = torch.maximum(result, torch.where(match, indices, -1).amax(0))
    return result
