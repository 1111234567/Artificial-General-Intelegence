"""Supervision targets derived from a sequence of public observations."""
from __future__ import annotations

import torch

from experiments.associative_state.bridge import FEATURES_PER_JOB, PublicReceiptBinder
from experiments.experience_transfer.train_transfer import SEEDS
from experiments.lifelong.world import NEW, REPLACE


def build(source_seed: int, obs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return CPU int64 query/key/value targets with -1 outside their masks.

    Query targets identify the current command as context_index * 4 + raw.
    Write targets identify a previously issued command and its accepted FINISH
    value. A same-slot receipt and command can contribute both kinds of target.
    These targets are for the loss. The model's input remains the public stream.
    """
    if source_seed not in SEEDS:
        raise ValueError("source_seed must select executor seed 0, 1, or 2")
    if (obs.ndim != 3 or obs.shape[-1] != 2 * FEATURES_PER_JOB
            or obs.shape[0] == 0 or obs.shape[1] == 0 or not obs.is_floating_point()):
        raise ValueError("obs must be a nonempty floating-point [time,batch,56] tensor")

    frames = obs.detach().cpu()
    ticks, batch, _ = frames.shape
    source = SEEDS[source_seed]
    contexts = (source["history_cue"], source["B_cue"])
    binder = PublicReceiptBinder(batch, n_jobs=2, contexts=contexts)

    public = frames.reshape(ticks, batch, 2, FEATURES_PER_JOB)
    command = (public[..., NEW - 1] > .5) | (public[..., REPLACE - 1] > .5)
    family = public[..., 24:28].argmax(-1)
    raw = public[..., 5:9].argmax(-1)
    query = (4 * (family == contexts[1]).long() + raw).masked_fill(~command, -1)
    write_key = torch.full_like(query, -1)
    write_value = torch.full_like(query, -1)

    for tick, frame in enumerate(frames):
        for job, (key, value, matched) in enumerate(binder.consume(frame)):
            write_key[tick, :, job] = key.masked_fill(~matched, -1)
            write_value[tick, :, job] = value.masked_fill(~matched, -1)

    return dict(query=query, write_key=write_key, write_value=write_value)
