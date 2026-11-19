"""Actual midpoint-teacher trajectories with fresh, private association rules.

Only ``obs`` is a policy input.  ``actions`` contains supervised action labels;
the rule tables and world snapshots are audit/environment data.  The teacher's
chosen action is executed before its receipt can enter the next observation.
No binder, synthetic receipt, or current/future label is added to public56.
"""
from __future__ import annotations

from numbers import Integral
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.associative_state.bridge import RandomMappingWorld
from experiments.experience_transfer.train_transfer import SEEDS

FORMAT = "learned-binding-actual-teacher-stream-v1"
DT = .25
SPLITS = {"train": 1, "eval": 2}


def _integer(name: str, value: int, *, positive: bool = False) -> int:
    if (isinstance(value, bool) or not isinstance(value, Integral)
            or value < int(positive)):
        raise ValueError(f"{name} must be a {'positive' if positive else 'nonnegative'} integer")
    return int(value)


def _identity(seed: int, stream_id: int, random_seed: int, split: str,
              batch: int) -> dict[str, Any]:
    """Separate data randomness from the source executor's seed/cue pair.

    Train and eval world seeds and actor IDs have disjoint upper-bit domains.
    Independent SeedSequence children supply world, actor and mapping draws.
    Batch size does not change the identities/rules of an existing actor prefix.
    """
    seed = _integer("seed", seed)
    if seed not in SEEDS:
        raise ValueError("seed must select source executor 0, 1, or 2")
    stream_id = _integer("stream_id", stream_id)
    random_seed = _integer("random_seed", random_seed)
    batch = _integer("batch", batch, positive=True)
    if batch >= 2**32:
        raise ValueError("batch must be smaller than 2**32")
    if split not in SPLITS:
        raise ValueError("split must be 'train' or 'eval'")
    domain = SPLITS[split]
    sequence = np.random.SeedSequence([0x4C42494E, domain, seed, random_seed, stream_id])
    world_sequence, actor_sequence, mapping_sequence = sequence.spawn(3)
    world_word = int(world_sequence.generate_state(1, dtype=np.uint64)[0])
    actor_word = int(actor_sequence.generate_state(1, dtype=np.uint64)[0])
    world_seed = (domain << 62) | (world_word & (2**62 - 1))
    actor_base = (domain << 62) | ((actor_word & (2**30 - 1)) << 32)
    actors = tuple(actor_base + lane for lane in range(batch))
    rng = np.random.default_rng(mapping_sequence)
    a = np.argsort(rng.random((batch, 4)), axis=-1)
    b = a[:, [1, 0, 2, 3]].copy()
    mappings = torch.from_numpy(np.stack((a, b), axis=1)).long()
    return dict(source_seed=seed, stream_id=stream_id, random_seed=random_seed,
                split=split, world_seed=world_seed, actor_ids=actors,
                contexts=(SEEDS[seed]["history_cue"], SEEDS[seed]["B_cue"]),
                mappings=mappings)


def make_world(seed: int, batch: int = 32,
               duration_range: tuple[float, float] = (1., 4.),
               gap_range: tuple[float, float] = (1., 4.),
               stream_id: int = 0, *, random_seed: int = 0,
               split: str = "train") -> RandomMappingWorld:
    """Create a fresh continuous AB world, usable by teacher or own actions.

    ``seed`` chooses the old executor's context pair. ``random_seed``,
    ``stream_id`` and ``split`` identify an independent new environment, not
    the old executor's training distribution.  Worlds are always CPU public56.
    """
    identity = _identity(seed, stream_id, random_seed, split, batch)
    return RandomMappingWorld(
        seed=identity["world_seed"], batch_size=batch, n_jobs=2, active_jobs=2,
        actor_ids=identity["actor_ids"], dt=DT, duration_range=duration_range,
        gap_range=gap_range, contexts=identity["contexts"],
        family_allowed=identity["contexts"], mappings=identity["mappings"],
        device="cpu")


def reward_from_observation(obs: torch.Tensor) -> torch.Tensor:
    """Public outcome reward for the *preceding* action, never a write mask.

    Accepts ``[batch,56]`` or leading trajectory dimensions ending in public56.
    A successful FINISH receives +1; any rejected attempted action receives
    -.1; successful BEGIN, WAIT and absent receipts receive zero.  Rewards are
    detached environment outcomes and expose no oracle payload or eligibility.
    For own-action PG, consume the next observation before rewarding an action;
    ``reward(obs[t])`` must not be assigned to the action chosen at ``t``.
    """
    if not isinstance(obs, torch.Tensor) or obs.ndim < 2 or obs.shape[-1] != 56:
        raise ValueError("obs must be a tensor with shape [...,batch,56]")
    if not torch.is_floating_point(obs):
        raise ValueError("public observations must be floating point")
    public = obs.detach().reshape(*obs.shape[:-1], 2, 28)
    previous_action = public[..., 11:22]
    attempted = previous_action[..., 1:].sum(-1) > .5
    finish = previous_action[..., 3:].sum(-1) > .5
    rejected = (public[..., 23] > .5) & attempted
    success = (public[..., 22] > .5) & ~rejected
    return (success & finish).to(obs.dtype) - .1 * rejected.to(obs.dtype)


def generate(seed: int, batch: int = 32, ticks: int = 1024,
             duration_range: tuple[float, float] = (1., 4.),
             gap_range: tuple[float, float] = (1., 4.),
             stream_id: int = 0, *, random_seed: int = 0,
             split: str = "train") -> dict[str, Any]:
    """Collect one uninterrupted, actual-action midpoint-teacher trajectory.

    Both halves sample A/B contexts; there is no midstream reset or revision.
    The stream starts before its first observe and ends after its final act.
    Consequently ``final_world`` retains the last action's pending receipt;
    it has not been silently consumed by an extra teacher observation.
    """
    ticks = _integer("ticks", ticks, positive=True)
    identity = _identity(seed, stream_id, random_seed, split, batch)
    world = make_world(seed, batch, duration_range, gap_range, stream_id,
                       random_seed=random_seed, split=split)
    initial_world = world.state_dict()
    observations, actions = [], []
    for _ in range(ticks):
        obs = world.observe()
        label = world.oracle("midpoint")
        observations.append(obs.clone())
        actions.append(label.clone())
        world.act(label)
    return dict(
        obs=torch.stack(observations), actions=torch.stack(actions),
        initial_world=initial_world, final_world=world.state_dict(),
        metadata=dict(
            format=FORMAT, **identity, batch=int(batch), ticks=ticks, dt=DT,
            duration_range=tuple(map(float, duration_range)),
            gap_range=tuple(map(float, gap_range)),
            context_policy="mixed_AB_for_entire_stream", revision=False,
            teacher="midpoint", teacher_actions_executed=True,
            policy_inputs=("obs",),
            actions_are_supervised_labels=True,
            world_snapshots_and_mappings_are_private_audit_data=True,
            receipt_alignment="obs[t] describes action[t-1]; obs[0] has no preceding action",
            final_receipt_pending=True))
