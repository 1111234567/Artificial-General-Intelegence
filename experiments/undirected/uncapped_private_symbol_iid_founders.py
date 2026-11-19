"""Seed the iid private-symbol search from final paired-run birth genomes."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import random
import re

import torch

from . import uncapped_directed_breeding as directed
from . import uncapped_private_symbol_selection as selection
from .uncapped_directed_reward_run import rank
from .uncapped_private_symbol_breeding import _digest, plan as _private_plan


SOURCE_GENERATION = 1024
SOURCE_PAIRS = tuple(range(1, 5))
SOURCE_ARMS = ("downhill", "nondecreasing")
STRATA = len(SOURCE_PAIRS) * len(SOURCE_ARMS) * directed.ISLANDS
MAX_WARM = 2048
_BIRTH_FIELDS = ("birth_w", "birth_mask", "node_bias", "rule", "directed", "reward_rate")


def _source_rows(archive_dir, files, requests, width, seed):
    for pair in SOURCE_PAIRS:
        for arm in SOURCE_ARMS:
            branch = f"pair-{pair:02d}/{arm}"
            path = f"{branch}/generations/{SOURCE_GENERATION:04d}/elites.pt"
            content = (archive_dir / path).read_bytes()
            files[path] = hashlib.sha256(content).hexdigest()
            checkpoint = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
            pools, genomes = checkpoint["islands"], checkpoint["genomes"]
            identifiers = [row["id"] for pool in pools for row in pool]
            if (len(pools) != directed.ISLANDS or not all(pools)
                    or len(identifiers) != len(genomes) or set(identifiers) != genomes.keys()):
                raise ValueError(f"Invalid elite pools or candidate IDs in {path}")
            for island, pool in enumerate(pools):
                prefix = f"g{SOURCE_GENERATION - 1:03d}-i{island:02d}-c"
                selection.validate_rule(pool, selection.POST_ANSWER_RULE, path)
                for row in pool:
                    ident = row["id"]
                    if (row["island"] != island or not re.fullmatch(rf"{prefix}[0-9]{{4}}", ident)
                            or int(ident[len(prefix):]) >= width or not row["directed"]
                            or not isinstance(row["lineage"], str) or not row["lineage"]):
                        raise ValueError(f"Invalid source candidate {ident} in {path}")
                    score, _ = selection.selected_score(row, requests, selection.POST_ANSWER_RULE)
                    if row["selection_score"] != score:
                        raise ValueError(f"Invalid v3 score for {ident} in {path}")
                ranked = sorted(pool, key=lambda row: row["id"])
                random.Random(f"iid-warm-ties:{seed}:{branch}:{island}").shuffle(ranked)
                ranked.sort(key=rank, reverse=True)
                records = []
                for position, row in enumerate(ranked, start=1):
                    ident = row["id"]
                    genome = {name: genomes[ident][name] for name in _BIRTH_FIELDS}
                    directed.validate_genome(genome)
                    if not bool(genome["directed"]):
                        raise ValueError(f"Expected a directed birth genome for {ident} in {path}")
                    record = {
                        "id": f"{branch}/{ident}",
                        "source_candidate_id": ident,
                        "source_pair": pair,
                        "source_arm": arm,
                        "source_island": island,
                        "source_checkpoint": path,
                        "source_rank": position,
                        "source_selection_score": row["selection_score"],
                        "lineage": row["lineage"],
                        "generation": SOURCE_GENERATION,
                        "genome_sha256": _digest(genome),
                    }
                    records.append((record, genome))
                yield records


def _select(groups, quota, seed):
    """Allocate ranked rows across groups and maximize distinct birth digests."""
    order = list(range(len(groups)))
    random.Random(f"iid-warm-group-order:{seed}").shuffle(order)
    queues = [iter(group) for group in groups]
    owners = {}
    counts = [0] * len(groups)
    for _ in range(quota):
        for group in order:
            for row in queues[group]:
                digest = row[0]["genome_sha256"]
                if digest not in owners:
                    owners[digest] = group, row
                    counts[group] += 1
                    break

    def claim(group, visited):
        # A full group can give up a digest if it can claim a replacement.
        visited.add(group)
        for row in groups[group]:
            digest = row[0]["genome_sha256"]
            owner = owners.get(digest)
            if owner is None or (owner[0] not in visited and claim(owner[0], visited)):
                owners[digest] = group, row
                return True
        return False

    for group in order:
        while counts[group] < quota and claim(group, set()):
            counts[group] += 1

    chosen = [[] for _ in groups]
    for group, row in owners.values():
        chosen[group].append(row)
    for group, selected in zip(groups, chosen, strict=True):
        if len(selected) < quota:
            used = {row[0]["id"] for row in selected}
            extras = [row for row in group if row[0]["id"] not in used]
            selected.extend(extras[:quota - len(selected)])
        selected.sort(key=lambda row: row[0]["source_rank"])
    return chosen, len(owners)


def load_warm(archive_dir, warm_total=MAX_WARM, seed=1):
    """Return destination descriptors, CPU birth tensors, and hashed source provenance."""
    if type(warm_total) is not int or not 0 <= warm_total <= MAX_WARM or warm_total % STRATA:
        raise ValueError(f"Warm count must be a multiple of {STRATA} between 0 and {MAX_WARM}")
    archive_dir = Path(archive_dir).resolve()
    per_stratum = warm_total // STRATA
    provenance = {
        "source_run": str(archive_dir), "count": warm_total, "per_island": warm_total // directed.ISLANDS,
        "per_source_branch_island": per_stratum, "seed": seed, "source_generations": [],
        "distinct_birth_genomes": 0, "selection_rule": selection.POST_ANSWER_RULE,
        "selection": "Rank generation-1024 elites within each source branch and island, break rank ties "
                     "with a seeded shuffle, and prefer globally distinct birth genomes at equal quotas.",
        "files": {}, "selected": [],
    }
    if not warm_total:
        return [], {}, provenance

    files = provenance["files"]
    records = {}
    for path in ("protocol.json", "progress.json"):
        content = (archive_dir / path).read_bytes()
        files[path] = hashlib.sha256(content).hexdigest()
        records[path] = json.loads(content)
    protocol, progress = records["protocol.json"], records["progress.json"]
    if (protocol["task"] != "private_symbol" or protocol["generations"] != SOURCE_GENERATION
            or protocol["pairs"] != len(SOURCE_PAIRS) or protocol["islands"] != directed.ISLANDS
            or protocol["nodes"] != directed.NODES or progress["state"] != "complete"
            or progress["generation"] != SOURCE_GENERATION
            or any(progress["arms_progress"].get(f"{pair:02d}/{arm}", {}).get("sealed_generations")
                   != SOURCE_GENERATION for pair in SOURCE_PAIRS for arm in SOURCE_ARMS)):
        raise ValueError(f"Expected the completed generation-{SOURCE_GENERATION} private-symbol run in {archive_dir}")

    groups = list(_source_rows(archive_dir, files, selection.requests_per_candidate(protocol),
                               protocol["population"] // directed.ISLANDS, seed))
    if any(len(group) < per_stratum for group in groups):
        raise ValueError("The source checkpoint has too few elites for the requested branch/island quota")
    chosen, distinct = _select(groups, per_stratum, seed)
    descriptors, bank = [], {}
    lanes = [0] * directed.ISLANDS
    for tier in zip(*chosen, strict=True):
        for record, genome in tier:
            island = record["source_island"]
            descriptors.append({**record, "island": island, "lane": lanes[island]})
            lanes[island] += 1
            bank[record["id"]] = genome
    provenance.update(source_generations=[SOURCE_GENERATION], distinct_birth_genomes=distinct, selected=descriptors)
    return descriptors, bank, provenance


def plan(config, generation, previous, warm_founders=()):
    """Overlay warm births on the first plan and reuse private-symbol breeding afterward."""
    entries = _private_plan(config, generation, previous)
    if generation != 1 or not warm_founders:
        return entries

    width = config["population"] // config["islands"]
    assignments = {(row["island"], row["lane"]): row for row in warm_founders}
    if (len(assignments) != len(warm_founders) or len(assignments) > min(MAX_WARM, config["population"] // 2)
            or len({row["id"] for row in warm_founders}) != len(assignments)
            or any(island != row["source_island"] or island not in range(config["islands"])
                   or lane not in range(width) for (island, lane), row in assignments.items())):
        raise ValueError("Warm founders require unique IDs and in-range lanes within their source islands")
    if config.get("nodes", directed.NODES) != directed.NODES:
        raise ValueError(f"The warm birth genomes require {directed.NODES} policy nodes")
    for entry in entries:
        lane = int(entry["id"].rsplit("c", 1)[1])
        source = assignments.get((entry["island"], lane))
        if source is not None:
            # Both arms of a source pair used the same random founder schedule;
            # the older archive lineages were shared across all source pairs.
            lineage = source["lineage"]
            if not lineage.startswith("archive:"):
                lineage = f"pair-{source['source_pair']:02d}:{lineage}"
            entry.update(kind="warm", parent=source["id"], parent_island=source["source_island"],
                         lineage=f"archive:{lineage}", source_lineage=source["lineage"],
                         source_generation=source["generation"], source_genome_sha256=source["genome_sha256"],
                         source_candidate_id=source["source_candidate_id"], source_pair=source["source_pair"],
                         source_arm=source["source_arm"], source_rank=source["source_rank"])
    return entries
