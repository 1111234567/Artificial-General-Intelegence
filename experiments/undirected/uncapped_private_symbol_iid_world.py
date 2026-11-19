"""Sample private-symbol contexts independently for every actor and request."""

from dataclasses import replace

import numpy as np
import torch

from . import uncapped_private_symbol_span as span
from . import uncapped_private_symbol_world as private_world
from .evolved_plasticity_world import CONTEXT, FIRST_FINISH, N_ACTIONS, NEW, READY
from .plasticity_continuous_world import _ratio
from .uncapped_directed_reward_run import TIMING, WORLD_BLOCKS


def schedule(*, seed, actors, device, visits, timing=TIMING, pad_to=None):
    """Use the sealed maps, raw permutations and timing with a fresh Bernoulli context tape."""
    tape = private_world.schedule(seed=seed, actors=actors, device=device, visits=visits,
                                  timing=timing, pad_to=pad_to)
    requests, actors = tape.raw.shape
    seed = int(seed) % (1 << 64)
    low, high = seed & 0xffffffff, seed >> 32
    contexts = np.empty((requests, actors), dtype=np.int64)
    for actor in range(actors):
        # Stream 1 draws the private maps and raw order, and stream 2 draws READY waits.
        rng = np.random.default_rng(np.random.SeedSequence([low, high, actor, 3]))
        contexts[:, actor] = rng.integers(2, size=requests)
    context = torch.from_numpy(contexts).to(tape.raw.device)
    target = FIRST_FINISH + tape.maps[torch.arange(actors, device=context.device), context, tape.raw]
    port = tape.public[..., 0, :]
    new = port[..., NEW].bool()
    cue = port[..., CONTEXT]
    cue.zero_()
    cue[..., 0] = new & context[:, None, :].eq(0)
    cue[..., 1] = new & context[:, None, :].eq(1)
    return replace(tape, context=context, target=target)


def training_schedule(*, seed, actors, device, visits, timing=TIMING, pad_to=None):
    """Keep four independent and equally sized training world blocks."""
    per_block, rest = divmod(actors, WORLD_BLOCKS)
    if rest or per_block < 1:
        raise ValueError("training needs four equally sized nonempty actor blocks")
    tapes = [schedule(seed=seed * WORLD_BLOCKS + block, actors=per_block, device=device,
                      visits=visits, timing=timing, pad_to=pad_to) for block in range(WORLD_BLOCKS)]
    return replace(
        tapes[0],
        **{field: torch.cat([getattr(tape, field) for tape in tapes], dim=2)
           for field in ("public", "stage")},
        **{field: torch.cat([getattr(tape, field) for tape in tapes], dim=1)
           for field in ("target", "context", "raw")},
        maps=torch.cat([tape.maps for tape in tapes], dim=0),
    )


def late_symbol_metrics(actions, tape):
    """Score first legal FINISH over the second half of the life for each context."""
    late = slice(tape.target.size(0) // 2, None)
    actions = private_world.normalize_actions(actions[late])
    ready = tape.public[late, :, :, 0, READY].bool() & tape.stage[late].eq(2)
    legal = actions.ge(FIRST_FINISH) & actions.lt(N_ACTIONS) & ready.unsqueeze(2)
    attempted, first_tick = legal.max(dim=1)
    first_action = actions.gather(1, first_tick.unsqueeze(1)).squeeze(1)
    correct = attempted & first_action.eq(tape.target[late].unsqueeze(1))

    metrics = {}
    present = []
    for index, name in enumerate(("a", "b")):
        mask = tape.context[late].eq(index).unsqueeze(1)
        count = mask.sum()
        rate = _ratio((correct & mask).sum((0, 2)), count)
        metrics[f"symbol_late_{name}"] = rate
        metrics[f"symbol_late_{name}_requests"] = count.expand_as(rate)
        present.append(count.gt(0).int())
    metrics["symbol_late_mean"] = _ratio(metrics["symbol_late_a"] + metrics["symbol_late_b"],
                                         present[0] + present[1])
    return metrics


@torch.no_grad()
def observed_rollout(policy, tape, table, *, training):
    """Keep the exact action trace and report late scores from the second half."""
    rollout = span.observed_rollout if training else private_world.observed_rollout
    metrics, actions = rollout(policy, tape, table)
    return {**metrics, **late_symbol_metrics(actions, tape)}, actions
