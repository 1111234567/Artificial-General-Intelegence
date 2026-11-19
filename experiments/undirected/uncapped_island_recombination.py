"""Plan and build variable-birth candidates for an isolated island population."""

from __future__ import annotations

import random

import torch

from . import variable_birth_genome


SLOT_COUNTS = {"elite": 2, "local": 13, "wide": 2, "intra": 12, "inter": 1, "founder": 2}
DONOR_FRACTION = 0.125
RULE_DONOR_FRACTION = 0.25
RULE_ROW_NAMES = ("weight_update", "growth_pruning")


def _schedule(width):
    """Apportion the 32-slot mix at any island size using largest remainders."""
    total = sum(SLOT_COUNTS.values())
    counts, remainders = {}, {}
    for kind, slots in SLOT_COUNTS.items():
        counts[kind], remainders[kind] = divmod(width * slots, total)
    unassigned = width - sum(counts.values())
    for kind in sorted(remainders, key=remainders.get, reverse=True)[:unassigned]:
        counts[kind] += 1
    return [kind for kind, count in counts.items() for _ in range(count)]


def plan(config, generation, previous, warm_founders=()):
    """Produce a deterministic, lane-interleaved plan with parent provenance."""
    islands, population = config["islands"], config["population"]
    if islands < 1 or population < 1 or population % islands or generation < 1:
        raise ValueError("Use a positive generation and evenly sized nonempty islands")
    width = population // islands
    rng = random.Random(config["seed"] * 1_000_003 + generation - 1)
    seeds = iter(rng.sample(range(1, 2**31), population))
    crosses = [{"intra": [], "inter": []} for _ in range(islands)]
    if previous:
        pools = [list({row["id"]: row for row in pool}.values()) for pool in previous["islands"]]
        if len(pools) != islands or not all(pools):
            raise ValueError("Previous checkpoint must have a nonempty elite pool for every island")
        exact = [0] * islands
        mates = [[other for other in range(islands) if other != island] for island in range(islands)]
        schedules = [_schedule(width) for _ in range(islands)]
        for schedule in schedules:
            rng.shuffle(schedule)
    else:
        founders = list({row["id"]: row for row in warm_founders}.values())

    entries = []
    for lane in range(width):
        for island in range(islands):
            identifier = f"g{generation-1:03d}-i{island:02d}-c{lane:04d}"
            kind, parent, donor = "founder", None, None
            parent_island = donor_island = None
            if previous:
                kind = schedules[island][lane]
                pool = pools[island]
                if kind != "founder":
                    parent_island = island
                    if kind == "elite":
                        parent = pool[exact[island] % len(pool)]
                        exact[island] += 1
                    elif kind == "intra" and len(pool) > 1:
                        parent, donor = rng.sample(pool, 2)
                        donor_island = island
                    else:
                        parent = rng.choice(pool)
                        if kind == "inter" and mates[island]:
                            donor_island = rng.choice(mates[island])
                            donor = rng.choice(pools[donor_island])
                        elif kind in ("intra", "inter"):
                            kind = "local"
            elif len(entries) < len(founders):
                kind, parent = "warm", founders[len(entries)]
                parent_island = parent.get("island")

            lineage = parent["lineage"] if parent else identifier
            if kind == "warm":
                lineage = f"archive:{lineage}"
            entries.append({
                "id": identifier, "island": island, "kind": kind, "seed": next(seeds),
                "parent": parent["id"] if parent else None, "parent_island": parent_island,
                "lineage": lineage,
                "donor": donor["id"] if donor else None, "donor_island": donor_island,
                "donor_lineage": donor["lineage"] if donor else None,
            })
            if kind == "warm":
                entries[-1]["source_lineage"] = parent["lineage"]
            elif kind in ("intra", "inter"):
                entries[-1]["donor_rule_row"] = None
                crosses[island][kind].append(entries[-1])

    # Select within each island and crossover kind without changing the structural plan's RNG.
    rule_rng = random.Random(f"rule-crossover:{config['seed']}:{generation}")
    for island_crosses in crosses:
        for group in island_crosses.values():
            for entry in rule_rng.sample(group, int(len(group) * RULE_DONOR_FRACTION)):
                entry["donor_rule_row"] = rule_rng.randrange(len(RULE_ROW_NAMES))
    return entries


def crossover_genome(primary, donor, seed, *, donor_rule_row=None):
    """Copy sampled birth edges and optionally one complete donor rule row."""
    rng = torch.Generator(device="cpu").manual_seed(seed)
    pair_count = primary["birth_mask"].numel()
    loci = torch.randperm(pair_count, generator=rng)[:round(pair_count * DONOR_FRACTION)]
    child = {key: primary[key].clone() for key in ("birth_mask", "birth_w", "node_bias", "rule")}
    for key in ("birth_mask", "birth_w"):
        child[key][loci] = donor[key][loci]
    if donor_rule_row is not None:
        if donor_rule_row not in range(len(RULE_ROW_NAMES)):
            raise ValueError(f"Invalid donor rule row: {donor_rule_row}")
        child["rule"][donor_rule_row].copy_(donor["rule"][donor_rule_row])
    return child


def candidate(entry, previous, nodes, warm_genomes=None):
    """Build or retrieve only the heritable birth state for a planned row."""
    if entry["kind"] == "warm":
        if warm_genomes is None:
            raise ValueError("Warm founders require the archived birth genomes")
        return warm_genomes[entry["parent"]]
    if entry["kind"] in ("intra", "inter"):
        parents = previous["genomes"]
        return crossover_genome(parents[entry["parent"]], parents[entry["donor"]], entry["seed"],
                                donor_rule_row=entry.get("donor_rule_row"))
    return variable_birth_genome.candidate(entry, previous, nodes)
