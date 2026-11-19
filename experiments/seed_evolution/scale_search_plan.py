"""Deterministic, equal-budget plans for the seed evolution scale experiment."""
from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping


SOURCE_SEEDS = (0, 1, 2)
ARMS = ("deep", "medium", "wide", "random")
SCHEDULES = {"deep": (60, 32), "medium": (30, 64), "wide": (15, 128), "random": (15, 128)}
SHARED_COUNT = 128
BUDGET = 2048
UNIQUE_CANDIDATES = len(SOURCE_SEEDS) * (SHARED_COUNT + len(ARMS) * (BUDGET - SHARED_COUNT))
PARENTS = 8
SIGMA = .15
INNER_STEPS = 64
FINAL_STEPS = 512
DEV_STREAMS = (1200, 1201)
TEST_STREAMS = (2200, 2201)
ACTORS = 16
SUPPORT_TICKS = 2048
PROBE_TICKS = 2048
RANDOM_SEED = 78111
B_TICKS = 4096

_SEED_BASES = {"initialization": 3_100_000, "mutation": 3_200_000, "parent_selection": 3_300_000}

CONFIG = dict(
    repeats=len(SOURCE_SEEDS),
    source_seeds=list(SOURCE_SEEDS),
    shared_count=SHARED_COUNT,
    budget_per_arm_per_repeat=BUDGET,
    unique_candidates=UNIQUE_CANDIDATES,
    schedules={arm: dict(generations=generations, population=population)
               for arm, (generations, population) in SCHEDULES.items()},
    parents=PARENTS,
    sigma=SIGMA,
    seed_bases=dict(_SEED_BASES),
    inner_steps=INNER_STEPS,
    final_steps=FINAL_STEPS,
    dev_streams=list(DEV_STREAMS),
    test_streams=list(TEST_STREAMS),
    actors=ACTORS,
    support_ticks=SUPPORT_TICKS,
    probe_ticks=PROBE_TICKS,
    random_seed=RANDOM_SEED,
    b_ticks=B_TICKS,
)


def scoring_options(split: str) -> dict:
    if split == "dev":
        streams = DEV_STREAMS
    elif split == "test":
        streams = TEST_STREAMS
    else:
        raise ValueError(f"unknown scoring split: {split}")
    return dict(stream_ids=streams, support_ticks=SUPPORT_TICKS, probe_ticks=PROBE_TICKS,
                actors=ACTORS, random_seed=RANDOM_SEED)


def _generation_size(arm: str, generation: int) -> int:
    if type(generation) is not int:
        raise ValueError(f"generation must be an integer: {generation}")
    if arm == "shared" and generation == 0:
        return SHARED_COUNT
    if arm in SCHEDULES and 1 <= generation <= SCHEDULES[arm][0]:
        return SCHEDULES[arm][1]
    raise ValueError(f"unknown generation {generation} for {arm}")


def candidate_ids(arm: str, generation: int) -> tuple[str, ...]:
    return tuple(f"{arm}_g{generation:02d}_i{index:03d}"
                 for index in range(_generation_size(arm, generation)))


def pool_ids(arm: str, through_generation: int | None = None) -> tuple[str, ...]:
    """List one arm's candidates through the given generation, including shared initialization."""
    shared = candidate_ids("shared", 0)
    if arm == "shared":
        if through_generation is not None:
            _generation_size(arm, through_generation)
        return shared
    if arm not in SCHEDULES:
        raise ValueError(f"unknown search arm: {arm}")
    generations, _ = SCHEDULES[arm]
    limit = generations if through_generation is None else through_generation
    if type(limit) is not int or not 0 <= limit <= generations:
        raise ValueError(f"unknown generation {limit} for {arm}")
    return shared + tuple(candidate for generation in range(1, limit + 1)
                          for candidate in candidate_ids(arm, generation))


def ranked(ids: Iterable[str], records: Mapping[str, dict]) -> list[str]:
    def rank_key(candidate_id):
        row = records[candidate_id]
        fitness = row["fitness"]
        if not math.isfinite(fitness):
            raise ValueError(f"non-finite development fitness: {candidate_id}")
        return -fitness, row["ordinal"], candidate_id

    return sorted(ids, key=rank_key)


def generation_plan(rep: int, arm: str, generation: int, records: Mapping[str, dict]) -> dict:
    """Materialize a generation from the completed development scores for one source.

    Mutation and parent-selection seeds are paired by source and candidate ordinal
    across the three evolution schedules. Seed families and source ranges are disjoint.
    The shared and random arms can be materialized without any scored records.
    """
    if type(rep) is not int or rep not in SOURCE_SEEDS:
        raise ValueError(f"unknown source seed: {rep}")

    ids = candidate_ids(arm, generation)
    evolves = arm in SCHEDULES and arm != "random"
    top = ranked(pool_ids(arm, generation - 1), records)[:PARENTS] if evolves else []
    first_ordinal = 0 if generation == 0 else SHARED_COUNT + (generation - 1) * len(ids)
    candidates = []
    for index, candidate_id in enumerate(ids):
        ordinal = first_ordinal + index
        offset = rep * BUDGET + ordinal
        if evolves:
            init_seed = None
            mutation_seed = _SEED_BASES["mutation"] + offset
            parent_selection_seed = _SEED_BASES["parent_selection"] + offset
            parent_id = random.Random(parent_selection_seed).choice(top)
        else:
            init_seed = _SEED_BASES["initialization"] + offset
            mutation_seed = None
            parent_selection_seed = None
            parent_id = None
        candidates.append(dict(rep=rep, source_seed=rep, arm=arm, generation=generation,
                               index=index, ordinal=ordinal, id=candidate_id, parent_id=parent_id,
                               init_seed=init_seed, mutation_seed=mutation_seed, sigma=SIGMA,
                               parent_selection_seed=parent_selection_seed))
    return dict(rep=rep, arm=arm, generation=generation, top8_before=top, candidates=candidates)
