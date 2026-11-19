"""Breed matched reciprocal and directed birth graphs in four islands."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import io
import json
import math
from pathlib import Path
import random

import torch

from . import uncapped_island_recombination as recombination, variable_birth_genome
from .evolved_plasticity import NODES as ARCHIVED_NODES, RULE_FEATURES, WEIGHT_LIMIT


NODES = 96
ISLANDS = 4
DIRECTED_ISLANDS = (0, 2)
PAIRED_ISLANDS = ((0, 1), (2, 3))
REWARD_RATE = .02
REWARD_RATE_LIMITS = (.005, .08)
REWARD_RATE_LOG_STEP = .08
AUDITS = {
    896: "0896.json",
    972: "0972-stop-fresh.json",
    1012: "1024-tail-paired-audit.json",
    1024: "1024.json",
}
SLOT_COUNTS = recombination.SLOT_COUNTS
DONOR_FRACTION = recombination.DONOR_FRACTION
RULE_DONOR_FRACTION = recombination.RULE_DONOR_FRACTION
RULE_ROW_NAMES = recombination.RULE_ROW_NAMES


def _with_topology(genome, directed):
    return {**genome, "directed": torch.tensor(directed, dtype=torch.bool, device="cpu")}


def _symmetric(genome):
    return {
        **genome,
        "birth_w": genome["birth_w"].repeat(2),
        "birth_mask": genome["birth_mask"].repeat(2),
        "reward_rate": torch.tensor(REWARD_RATE, dtype=torch.float32, device="cpu"),
    }


@lru_cache(maxsize=None)
def _archive_loci(nodes):
    left, right = torch.triu_indices(ARCHIVED_NODES, ARCHIVED_NODES, offset=1, device="cpu")
    return left * (2 * nodes - left - 1) // 2 + right - left - 1


def expand_archived(genome, nodes=NODES):
    """Preserve 80-node endpoints and append isolated nodes with zero biases."""
    if nodes < ARCHIVED_NODES:
        raise ValueError("Warm-start graph cannot omit archived nodes")
    pair_count = nodes * (nodes - 1) // 2
    loci = _archive_loci(nodes)
    expanded = {key: genome[key].new_zeros(pair_count) for key in ("birth_w", "birth_mask")}
    for key, tensor in expanded.items():
        tensor[loci] = genome[key]
    expanded["node_bias"] = genome["node_bias"].new_zeros(nodes)
    expanded["node_bias"][:ARCHIVED_NODES] = genome["node_bias"]
    expanded["rule"] = genome["rule"].clone()
    return _symmetric(expanded)


def load_warm(archive_dir, warm_total, seed):
    """Load one distinct archived birth genome per pair of warm-start slots."""
    from .uncapped_archived_founders import _birth_genome
    from .uncapped_ready_signal_run import rank

    if warm_total < 0 or warm_total % ISLANDS:
        raise ValueError("Warm-start slots must be nonnegative and evenly divided among four islands")
    archive_dir = Path(archive_dir).resolve()
    provenance = {
        "source_run": str(archive_dir), "count": warm_total, "per_island": warm_total // ISLANDS,
        "distinct_birth_genomes": warm_total // 2, "seed": seed, "files": {},
        "source_generations": [], "selected": [],
    }
    if not warm_total:
        return [], {}, provenance

    groups, champions = {}, []
    rng = random.Random(seed)
    for generation, audit_file in AUDITS.items():
        checkpoint_path = f"generations/{generation:04d}/elites.pt"
        audit_path = f"audits/{audit_file}"
        checkpoint_bytes = (archive_dir / checkpoint_path).read_bytes()
        audit_bytes = (archive_dir / audit_path).read_bytes()
        for path, content in ((checkpoint_path, checkpoint_bytes), (audit_path, audit_bytes)):
            provenance["files"][path] = hashlib.sha256(content).hexdigest()
        checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
        audit = json.loads(audit_bytes)
        audited_id = audit["candidate"] if "candidate" in audit else audit["checkpoints"][str(generation)]
        rows = [row for pool in checkpoint["islands"] for row in pool]
        rng.shuffle(rows)
        rows.sort(key=rank, reverse=True)
        group = []
        for row in rows:
            identifier = row["id"]
            genome, digest = _birth_genome(checkpoint["genomes"][identifier], identifier)
            record = {
                "id": identifier, "lineage": row["lineage"], "island": row["island"],
                "generation": generation, "checkpoint_audit": audit_path, "audit_champion": identifier == audited_id,
                "genome_sha256": digest,
            }
            group.append((record, genome))
            if record["audit_champion"]:
                champions.append((record, genome))
        if not any(record["audit_champion"] for record, _ in group):
            raise ValueError(f"Audited candidate {audited_id} is missing from {checkpoint_path}")
        groups[generation] = group

    selected, seen, lineages = [], set(), Counter()

    def select(item):
        record, genome = item
        selected.append((record, genome))
        seen.add(record["genome_sha256"])
        lineages[record["lineage"]] += 1

    goal = warm_total // 2
    for item in champions:
        if len(selected) < goal and item[0]["genome_sha256"] not in seen:
            select(item)
    while len(selected) < goal:
        start = len(selected)
        for group in groups.values():
            item = min(
                (item for item in group if item[0]["genome_sha256"] not in seen),
                key=lambda item: lineages[item[0]["lineage"]], default=None,
            )
            if item is not None:
                select(item)
                if len(selected) == goal:
                    break
        if len(selected) == start:
            raise ValueError(f"Requested {goal} distinct archived genomes, found only {start}")

    rows, genomes = [], {}
    lanes = [0, 0]
    by_source = [Counter(), Counter()]
    for record, genome in selected:
        generation = record["generation"]
        pair = min((i for i, lane in enumerate(lanes) if lane < warm_total // ISLANDS),
                   key=lambda i: (by_source[i][generation], lanes[i], i))
        rows.append({**record, "pair": pair, "lane": lanes[pair]})
        genomes[record["id"]] = expand_archived(genome)
        lanes[pair] += 1
        by_source[pair][generation] += 1
    provenance["source_generations"] = sorted({row["generation"] for row in rows})
    provenance["selected"] = rows
    return rows, genomes, provenance


def plan(config, generation, previous, warm_founders=()):
    """Lane-interleave paired islands and draw all ancestry within topology."""
    islands, population = config["islands"], config["population"]
    if islands != ISLANDS or population < 16 or population % islands or generation < 1:
        raise ValueError("Use four equally sized islands, population >= 16, and a positive generation")
    if bool(previous) != (generation > 1):
        raise ValueError("A previous checkpoint is required after generation one")
    width = population // islands
    rng = random.Random(config["seed"] * 1_000_003 + generation - 1)
    seeds = iter(rng.sample(range(1, 2**31), population // 2))
    if previous:
        pools = [list({row["id"]: row for row in pool}.values()) for pool in previous["islands"]]
        if len(pools) != islands or not all(pools):
            raise ValueError("Previous checkpoint must have a nonempty elite pool for every island")
        schedules = [recombination._schedule(width) for _ in PAIRED_ISLANDS]
        for schedule in schedules:
            rng.shuffle(schedule)
        exact = [0] * islands
    else:
        founders = {(row["pair"], row["lane"]): row for row in warm_founders}
        if len(founders) != len(warm_founders) or any(
            pair not in range(len(PAIRED_ISLANDS)) or lane not in range(width) for pair, lane in founders
        ):
            raise ValueError("Warm ancestors must occupy distinct in-range paired lanes")

    entries = []
    crosses = [{"intra": [], "inter": []} for _ in range(islands)]
    for lane in range(width):
        for pair, matched_islands in enumerate(PAIRED_ISLANDS):
            seed = next(seeds)
            for island in matched_islands:
                identifier = f"g{generation-1:03d}-i{island:02d}-c{lane:04d}"
                kind, parent, donor = "founder", None, None
                parent_island = donor_island = None
                if previous:
                    kind = schedules[pair][lane]
                    if kind != "founder":
                        chooser = random.Random(seed)
                        pool = pools[island]
                        parent_island = island
                        if kind == "elite":
                            parent = pool[exact[island] % len(pool)]
                            exact[island] += 1
                        elif kind == "intra" and len(pool) > 1:
                            parent, donor = chooser.sample(pool, 2)
                            donor_island = island
                        else:
                            parent = chooser.choice(pool)
                            if kind == "inter":
                                donor_island = island ^ 2
                                donor = chooser.choice(pools[donor_island])
                            elif kind == "intra":
                                kind = "local"
                elif (pair, lane) in founders:
                    kind, parent = "warm", founders[pair, lane]
                    parent_island = parent["island"]

                founder_family = f"p{pair}-c{lane:04d}"
                lineage = parent["lineage"] if parent else f"g{generation-1:03d}-{founder_family}"
                entry = {
                    "id": identifier, "island": island, "pair": pair, "kind": kind, "seed": seed,
                    "directed": island in DIRECTED_ISLANDS, "lineage": lineage,
                    "parent": parent["id"] if parent else None, "parent_island": parent_island,
                    "donor": donor["id"] if donor else None, "donor_island": donor_island,
                    "donor_lineage": donor["lineage"] if donor else None,
                }
                if not previous:
                    entry["paired_founder"] = founder_family
                if kind == "warm":
                    entry.update(lineage=f"archive:{lineage}", source_lineage=lineage,
                                 source_generation=parent["generation"], source_genome_sha256=parent["genome_sha256"])
                elif kind in ("intra", "inter"):
                    entry["donor_rule_row"] = None
                    crosses[island][kind].append(entry)
                entries.append(entry)

    for island, island_crosses in enumerate(crosses):
        for kind, group in island_crosses.items():
            rule_rng = random.Random(f"rule-crossover:{config['seed']}:{generation}:{island // 2}:{kind}")
            for entry in rule_rng.sample(group, int(len(group) * RULE_DONOR_FRACTION)):
                entry["donor_rule_row"] = rule_rng.randrange(len(RULE_ROW_NAMES))
    return entries


def _mutated_reward_rate(rate, seed, scale):
    step = random.Random(f"reward-rate:{seed}").gauss(0., REWARD_RATE_LOG_STEP * scale)
    return (rate * math.exp(step)).clamp_(*REWARD_RATE_LIMITS)


def _mutate(parent, seed, *, directed, wide):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    chooser = random.Random(seed)
    child = {key: value.clone() for key, value in parent.items()}
    all_weights, all_masks = child["birth_w"], child["birth_mask"]
    pairs = all_masks.numel() // 2
    weights, mask = (all_weights, all_masks) if directed else (all_weights[:pairs], all_masks[:pairs])
    scale = 2.5 if wide else 1.
    proposals = (4 if wide else 1) * (2 if directed else 1)
    new_weight_sd = .65 / math.sqrt(variable_birth_genome.FOUNDER_MEAN_DEGREE)
    for _ in range(proposals):
        adding = bool(chooser.getrandbits(1))
        eligible = mask.ne(adding).nonzero(as_tuple=True)[0]
        if eligible.numel():
            locus = eligible[chooser.randrange(eligible.numel())]
            mask[locus] = adding
            if adding:
                weights[locus] = chooser.gauss(0., new_weight_sd)

    touched = torch.rand(weights.shape, generator=rng).lt(.16 if wide else .09) & mask
    weights.add_(torch.randn(weights.shape, generator=rng) * touched * (.037 * scale))
    weights.clamp_(-WEIGHT_LIMIT, WEIGHT_LIMIT).masked_fill_(~mask, 0.)
    # Keep bias and rule perturbations matched when the paired parents agree.
    parameter_rng = torch.Generator(device="cpu").manual_seed(seed + 2**31)
    for key, step, limit in (("node_bias", .009, .35), ("rule", .065, 2.)):
        value = child[key]
        value.add_(torch.randn(value.shape, generator=parameter_rng) * (step * scale)).clamp_(-limit, limit)
    if not directed:
        all_weights[pairs:].copy_(weights)
        all_masks[pairs:].copy_(mask)
    child["reward_rate"] = _mutated_reward_rate(parent["reward_rate"], seed, scale)
    return child


def _crossover(primary, donor, seed, *, directed, donor_rule_row):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    pairs = primary["birth_mask"].numel() // 2
    count = 2 * pairs if directed else pairs
    loci = torch.randperm(count, generator=rng)[:round(count * DONOR_FRACTION)]
    if not directed:
        loci = torch.cat((loci, loci + pairs))
    child = {key: value.clone() for key, value in primary.items()}
    for key in ("birth_mask", "birth_w"):
        child[key][loci] = donor[key][loci]
    if donor_rule_row is not None:
        child["rule"][donor_rule_row].copy_(donor["rule"][donor_rule_row])
    child["reward_rate"] = _mutated_reward_rate(primary["reward_rate"], seed, 1.)
    return child


def candidate(entry, previous, nodes, warm_genomes=None):
    """Build a planned candidate's CPU birth state without advancing any RNG."""
    kind, seed, directed = entry["kind"], entry["seed"], entry["directed"]
    if kind == "warm":
        return _with_topology(warm_genomes[entry["parent"]], directed)
    if kind == "founder":
        return _with_topology(_symmetric(variable_birth_genome.make_genome(seed, nodes)), directed)

    genomes = previous["genomes"]
    primary = genomes[entry["parent"]]
    if bool(primary["directed"]) != directed:
        raise ValueError("A parent cannot cross topology arms")
    if kind == "elite":
        return primary
    if kind in ("intra", "inter"):
        donor = genomes[entry["donor"]]
        if bool(donor["directed"]) != directed:
            raise ValueError("A donor cannot cross topology arms")
        return _crossover(primary, donor, seed, directed=directed, donor_rule_row=entry.get("donor_rule_row"))
    return _mutate(primary, seed, directed=directed, wide=kind == "wide")


def validate_genome(genome, nodes=NODES):
    """Validate the shared doubled arc layout and reciprocal storage invariant."""
    arcs = nodes * (nodes - 1)
    layout = {
        "birth_w": ((arcs,), torch.float32), "birth_mask": ((arcs,), torch.bool),
        "node_bias": ((nodes,), torch.float32), "rule": ((2, len(RULE_FEATURES)), torch.float32),
        "directed": ((), torch.bool), "reward_rate": ((), torch.float32),
    }
    for key, (shape, dtype) in layout.items():
        tensor = genome.get(key)
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != shape
                or tensor.dtype != dtype or tensor.device.type != "cpu"):
            raise ValueError(f"Invalid CPU genome tensor: {key}")
        if dtype == torch.float32 and not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite genome tensor: {key}")
    if ((genome["reward_rate"] < REWARD_RATE_LIMITS[0])
            or (genome["reward_rate"] > REWARD_RATE_LIMITS[1])):
        raise ValueError("Reward rate is out of range")
    if genome["birth_w"].masked_select(~genome["birth_mask"]).count_nonzero():
        raise ValueError("Inactive birth arcs must have zero weights")
    if not bool(genome["directed"]):
        for key in ("birth_w", "birth_mask"):
            upper, reverse = genome[key].chunk(2)
            if not torch.equal(upper, reverse):
                raise ValueError(f"Reciprocal halves differ in {key}")


def verify_structure(*, warm_founders=(), warm_genomes=None, nodes=NODES, seed=1):
    """Exercise matched initialization and all inherited breeding paths on CPU."""
    warm_slots = 2 * len(warm_founders)
    config = {"population": max(128, warm_slots + 8), "islands": ISLANDS, "seed": seed}
    first = plan(config, 1, {}, warm_founders=warm_founders)
    previous = {"islands": [[] for _ in range(ISLANDS)], "genomes": {}}
    _, right = torch.triu_indices(nodes, nodes, offset=1)
    incident = (right >= ARCHIVED_NODES).repeat(2)
    for directed, reciprocal in zip(first[::2], first[1::2], strict=True):
        a, b = (candidate(row, {}, nodes, warm_genomes) for row in (directed, reciprocal))
        for genome in (a, b):
            validate_genome(genome, nodes)
        if not bool(a["directed"]) or bool(b["directed"]):
            raise ValueError("Paired founders have incorrect topology flags")
        if any(not torch.equal(a[key], b[key]) for key in a if key != "directed"):
            raise ValueError("Paired founders differ at birth")
        if directed["kind"] == "warm":
            if a["birth_mask"][incident].any() or a["node_bias"][ARCHIVED_NODES:].count_nonzero():
                raise ValueError("Archived founder has nonzero appended-node birth state")
        else:
            for row, genome in ((directed, a), (reciprocal, b)):
                pool = previous["islands"][row["island"]]
                if len(pool) < 2:
                    pool.append(row)
                    previous["genomes"][row["id"]] = genome

    config["population"] = 128
    second = plan(config, 2, previous)
    if second != plan(config, 2, previous):
        raise ValueError("Breeding plan changed when regenerated for resume")
    asymmetric = 0
    for row in second:
        child = candidate(row, previous, nodes)
        validate_genome(child, nodes)
        if row["kind"] == "inter" and row["donor_island"] != row["island"] ^ 2:
            raise ValueError("Cross-island donor has the wrong topology")
        upper, reverse = child["birth_mask"].chunk(2)
        asymmetric += int(bool(child["directed"]) and not torch.equal(upper, reverse))
    if not asymmetric:
        raise ValueError("Directed offspring did not acquire independent topology")
    return {"founder_pairs_checked": len(first) // 2, "warm_slots_checked": warm_slots,
            "offspring_checked": len(second), "directed_asymmetric_masks": asymmetric}
