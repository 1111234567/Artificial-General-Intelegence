"""Directed private-symbol searches with matched admission and inheritance trials.

The two search arms consume identical admission draws. Disabling downhill admission
changes only the final decision for negative score differences. A small subset of
ordinary local offspring are matched siblings: both receive the same mutation,
then one receives a quarter of its parent's measured within-life weight change.
Only birth-active child arcs inherit that change; no learned topology is inherited.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import random
from statistics import fmean, median, pstdev
import threading

import torch

from . import uncapped_directed_breeding as breeding
from . import uncapped_directed_exploration as exploration
from .evolved_plasticity import WEIGHT_LIMIT


SOURCE_GENERATIONS = (1472, 1984, 2048)
MAX_WARM = 256
INHERITANCE_PAIRS_PER_ISLAND = 8
INHERITANCE_FRACTION = .25
_SELECTION_LOCK = threading.RLock()


def protocol(population, islands, *, allow_downhill=True,
             inheritance_pairs_per_island=INHERITANCE_PAIRS_PER_ISLAND):
    settings = exploration.protocol(population, islands)
    available = settings["ordinary_offspring_per_island"]["local"]
    if not 0 <= inheritance_pairs_per_island <= available // 2:
        raise ValueError("Inheritance pairs must fit inside ordinary local offspring")
    return {
        **settings,
        "selection_temperature": exploration.TEMPERATURE if allow_downhill else 0.,
        "allow_downhill": bool(allow_downhill),
        "admission_rng": "always consume the original proposal and uniform draw, including T=0",
        "inheritance_pairs_per_island": inheritance_pairs_per_island,
        "inheritance_candidates_per_island": 2 * inheritance_pairs_per_island,
        "inheritance_fraction": INHERITANCE_FRACTION,
        "inheritance_order": "same local mutation first; add parent delta only to active child arcs; clamp weights",
        "inheritance_topology": "child mutation topology retained; no learned additions or deletions inherited",
        "inheritance_delta": "supplied measured parent lifetime delta; production tape independent of child scoring tape",
        "inheritance_budget": "replace ordinary local rows, preserve population and protected exploration rows",
    }


def _digest(genome):
    digest = hashlib.sha256()
    for name in ("birth_w", "birth_mask", "node_bias", "rule", "directed", "reward_rate"):
        tensor = genome[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_warm(archive_dir, warm_total=MAX_WARM, seed=1):
    """Return descriptors, CPU genomes and file-hashed provenance from the old run.

    Unlike the old reciprocal comparison, each destination slot gets a distinct
    genome. All paired search arms can use this same returned bank unchanged.
    Descriptors use ``island`` and ``lane`` for their new destination, and retain
    ``source_island`` for provenance. Genome-map keys are archived candidate IDs.
    """
    if not 0 <= warm_total <= MAX_WARM or warm_total % breeding.ISLANDS:
        raise ValueError("Use at most 256 warm slots, evenly divided among four islands")
    archive_dir = Path(archive_dir).resolve()
    provenance = {
        "source_run": str(archive_dir), "count": warm_total,
        "per_island": warm_total // breeding.ISLANDS, "seed": seed,
        "distinct_birth_genomes": 0, "source_generations": [], "files": {}, "selected": [],
        "selection_rule": "audited island champions first, then source-round-robin rank with founder-family diversity",
    }
    if not warm_total:
        return [], {}, provenance

    groups, champions = {}, []
    rng = random.Random(seed)
    for generation in SOURCE_GENERATIONS:
        checkpoint_path = f"generations/{generation:04d}/elites.pt"
        audit_path = f"audits/{generation:04d}.json"
        checkpoint_bytes = (archive_dir / checkpoint_path).read_bytes()
        audit_bytes = (archive_dir / audit_path).read_bytes()
        for path, content in ((checkpoint_path, checkpoint_bytes), (audit_path, audit_bytes)):
            provenance["files"][path] = hashlib.sha256(content).hexdigest()
        checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
        audit = json.loads(audit_bytes)
        champion_ids = {item["id"] for item in audit.get("champions", {}).values()}
        if audit.get("candidate"):
            champion_ids.add(audit["candidate"])
        rows = [row for pool in checkpoint["islands"] for row in pool]
        if not champion_ids.issubset({row["id"] for row in rows}):
            raise ValueError(f"An audited champion is missing from {checkpoint_path}")
        rng.shuffle(rows)
        rows.sort(key=exploration.rank, reverse=True)
        group = []
        for row in rows:
            genome = checkpoint["genomes"][row["id"]]
            breeding.validate_genome(genome, breeding.NODES)
            if not bool(genome["directed"]):
                raise ValueError(f"Warm source {row['id']} is not directed")
            record = {
                "id": row["id"], "lineage": row["lineage"], "source_island": row["island"],
                "generation": generation, "checkpoint_audit": audit_path,
                "audit_champion": row["id"] in champion_ids,
                "genome_sha256": _digest(genome),
            }
            group.append((record, genome))
            if record["audit_champion"]:
                champions.append((record, genome))
        groups[generation] = group

    selected, seen, lineages = [], set(), Counter()

    def select(item):
        record, _ = item
        selected.append(item)
        seen.add(record["genome_sha256"])
        lineages[record["lineage"]] += 1

    for item in champions:
        if len(selected) < warm_total and item[0]["genome_sha256"] not in seen:
            select(item)
    while len(selected) < warm_total:
        start = len(selected)
        for group in groups.values():
            item = min((item for item in group if item[0]["genome_sha256"] not in seen),
                       key=lambda item: lineages[item[0]["lineage"]], default=None)
            if item is not None:
                select(item)
                if len(selected) == warm_total:
                    break
        if len(selected) == start:
            raise ValueError(f"Requested {warm_total} distinct genomes; found {start}")

    rows, genomes = [], {}
    lanes = [0] * breeding.ISLANDS
    by_source = [Counter() for _ in lanes]
    for record, genome in selected:
        generation = record["generation"]
        island = min((i for i, lane in enumerate(lanes) if lane < warm_total // breeding.ISLANDS),
                     key=lambda i: (by_source[i][generation], lanes[i], i))
        rows.append({**record, "island": island, "lane": lanes[island]})
        genomes[record["id"]] = genome
        lanes[island] += 1
        by_source[island][generation] += 1
    provenance.update(distinct_birth_genomes=len(seen),
                      source_generations=sorted({row["generation"] for row in rows}), selected=rows)
    return rows, genomes, provenance


def _pairs(config, generation, entries):
    count = config.get("inheritance_pairs_per_island", INHERITANCE_PAIRS_PER_ISLAND)
    protocol(config["population"], config["islands"],
             allow_downhill=config.get("allow_downhill", True),
             inheritance_pairs_per_island=count)
    for island in range(config["islands"]):
        local = [row for row in entries if row["island"] == island
                 and row["kind"] == "local" and "exploration" not in row]
        rng = random.Random(f"private-inheritance:{config['seed']}:{generation}:{island}")
        chosen = rng.sample(local, 2 * count)
        for number in range(count):
            ordinary, inherited = chosen[2 * number:2 * number + 2]
            # The second entry retains its own candidate ID and therefore an
            # independently visible score, while its full birth recipe is paired.
            inherited.update({key: ordinary[key] for key in (
                "kind", "seed", "directed", "lineage", "parent", "parent_island",
                "donor", "donor_island", "donor_lineage",
            )})
            pair = f"g{generation-1:03d}-i{island:02d}-inherit-{number:02d}"
            for row, fraction in ((ordinary, 0.), (inherited, INHERITANCE_FRACTION)):
                row["inheritance_trial"] = {
                    "pair": pair, "role": "parent_delta" if fraction else "ordinary_birth",
                    "fraction": fraction, "learned_parent": ordinary["parent"],
                }
    return entries


def plan(config, generation, previous, warm_founders=()):
    """Make replayable plans; root may collect marked parent deltas before scoring."""
    entries = exploration.plan(config, generation, previous, warm_founders=())
    if previous:
        return _pairs(config, generation, entries)

    width = config["population"] // config["islands"]
    founders = {(row["island"], row["lane"]): row for row in warm_founders}
    if (len(founders) != len(warm_founders) or len(founders) > MAX_WARM
            or any(island not in range(config["islands"]) or lane not in range(width)
                   for island, lane in founders)):
        raise ValueError("Warm ancestors need distinct in-range island/lane destinations, at most 256")
    for row in entries:
        lane = int(row["id"].rsplit("c", 1)[1])
        parent = founders.get((row["island"], lane))
        if parent is not None:
            row.update(kind="warm", parent=parent["id"], parent_island=parent["source_island"],
                       lineage=f"archive:{parent['lineage']}", source_lineage=parent["lineage"],
                       source_generation=parent["generation"], source_genome_sha256=parent["genome_sha256"])
    return entries


def inheritance_parents(config, generation, previous):
    """Return unique marked parents, in deterministic first-pair order (<=32)."""
    if not previous:
        return []
    entries = plan(config, generation, previous)
    return list(dict.fromkeys(row["inheritance_trial"]["learned_parent"] for row in entries
                              if row.get("inheritance_trial", {}).get("fraction", 0.) > 0.))


def candidate(entry, previous, nodes, warm_genomes=None):
    child = breeding.candidate(entry, previous, nodes, warm_genomes)
    trial = entry.get("inheritance_trial")
    if trial is None or not trial["fraction"]:
        return child
    delta = previous.get("learned_deltas", {}).get(trial["learned_parent"])
    if (not isinstance(delta, torch.Tensor) or delta.device.type != "cpu"
            or delta.dtype != torch.float32 or delta.shape != child["birth_w"].shape
            or not torch.isfinite(delta).all()):
        raise ValueError(f"Missing or invalid measured parent delta for {trial['learned_parent']}")
    # Local mutation already cloned the parent. Never modify checkpoint tensors.
    child["birth_w"].add_(delta, alpha=trial["fraction"])
    child["birth_w"].clamp_(-WEIGHT_LIMIT, WEIGHT_LIMIT).masked_fill_(~child["birth_mask"], 0.)
    return child


def inheritance_summary(rows, selected_ids=()):
    groups = {}
    for row in rows:
        trial = row.get("inheritance_trial")
        if trial is not None:
            group = groups.setdefault(trial["pair"], {})
            if trial["role"] in group:
                raise ValueError("Duplicate inheritance sibling role")
            group[trial["role"]] = row
    differences, inherited_selected, ordinary_selected = {}, 0, 0
    selected_ids = set(selected_ids)
    for group in groups.values():
        if set(group) != {"ordinary_birth", "parent_delta"}:
            raise ValueError("Incomplete inheritance sibling pair")
        ordinary, inherited = group["ordinary_birth"], group["parent_delta"]
        if (ordinary["seed"], ordinary["parent"], ordinary["island"]) != (
                inherited["seed"], inherited["parent"], inherited["island"]):
            raise ValueError("Inheritance siblings do not share a birth recipe")
        for key in ("selection_score", "first_return_mean", "first_return_worst_context", "strict", "idle"):
            if key in ordinary and key in inherited:
                differences.setdefault(key, []).append(inherited[key] - ordinary[key])
        inherited_selected += inherited["id"] in selected_ids
        ordinary_selected += ordinary["id"] in selected_ids
    return {
        "pairs": len(groups), "fraction": INHERITANCE_FRACTION,
        "inherited_selected": inherited_selected, "ordinary_selected": ordinary_selected,
        "paired_difference": {key: {
            "mean": fmean(values), "median": median(values), "std": pstdev(values),
            "wins": sum(value > 0. for value in values), "ties": sum(value == 0. for value in values),
            "losses": sum(value < 0. for value in values),
        } for key, values in differences.items()},
    }


def elites(config, rows, previous, *, warm_genomes=None):
    """Reuse exploration selection; temporarily bind this file's genome builder.

    Root calls selection serially. The lock and finally block also prevent our
    own selection calls from leaving modified globals after an exception.
    """
    with _SELECTION_LOCK:
        original_candidate, original_admit = exploration.candidate, exploration._admit

        def admit(*args, **kwargs):
            decision = original_admit(*args, **kwargs)
            if not config.get("allow_downhill", True) and decision["delta"] < 0.:
                decision.update(id=decision["incumbent"], probability=0., accepted=False, accepted_downhill=False)
            return decision

        try:
            exploration.candidate, exploration._admit = candidate, admit
            result = exploration.elites(config, rows, previous, warm_genomes=warm_genomes)
        finally:
            exploration.candidate, exploration._admit = original_candidate, original_admit
    result["inheritance_summary"] = inheritance_summary(rows, result["genomes"])
    result["allow_downhill"] = bool(config.get("allow_downhill", True))
    return result
