"""Replay the previous run's births and breed matched local-rule populations."""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path
import random
from statistics import fmean

import torch

from . import uncapped_directed_breeding as directed
from . import uncapped_directed_exploration as exploration
from .local_rule_policy import default_plastic, PLASTIC_FIELDS
from .uncapped_directed_reward_run import rank
from .uncapped_private_symbol_breeding import _digest


ARCHIVE_GENERATION = 1024
ARCHIVE_BRANCH = Path("pair-01/downhill")
PLASTIC_MUTATION_STEP = .065
PLASTIC_PERTURBATION = .08


def _archive_sources(archive, files):
    def content(relative):
        relative = Path(relative)
        data = (archive / relative).read_bytes()
        files[relative.as_posix()] = hashlib.sha256(data).hexdigest()
        return data

    root_progress = json.loads(content("progress.json"))
    settings = json.loads(content(ARCHIVE_BRANCH / "protocol.json"))
    generation = ARCHIVE_GENERATION
    if (root_progress.get("state") != "complete"
            or root_progress.get("arms_progress", {}).get("01/downhill", {}).get("sealed_generations") != generation
            or settings.get("task") != "private_symbol" or settings.get("context_sampling") != "iid_per_request"
            or settings.get("generations") != generation or settings.get("islands") != directed.ISLANDS
            or settings.get("population") != 4096 or settings.get("nodes") != directed.NODES
            or settings.get("inheritance_pairs_per_island") != 0):
        raise ValueError(f"Expected the completed IID generation-{generation} source run in {archive}")

    final = ARCHIVE_BRANCH / "generations" / f"{generation:04d}"
    previous = ARCHIVE_BRANCH / "generations" / f"{generation - 1:04d}" / "elites.pt"
    # This is the immutable on-disk plan used for the original evaluation.
    entries = json.loads(content(final / "plan.json"))
    if len(entries) != settings["population"] or len({row["id"] for row in entries}) != len(entries):
        raise ValueError("Incomplete or duplicate entries in the archived plan")
    rows = []
    batch_size = settings["batch_size"]
    for start in range(0, len(entries), batch_size):
        path = final / "batches" / f"{start:04d}.json.gz"
        group = json.loads(gzip.decompress(content(path)))
        recipes = entries[start:start + batch_size]
        if len(group) != len(recipes) or any(
            any(row.get(key) != value for key, value in entry.items())
            for row, entry in zip(group, recipes, strict=True)
        ):
            raise ValueError(f"Archived scores do not match the replay plan in {path}")
        rows.extend(group)

    parents = torch.load(io.BytesIO(content(previous)), map_location="cpu", weights_only=True)
    sealed = torch.load(io.BytesIO(content(final / "elites.pt")), map_location="cpu", weights_only=True)
    return rows, parents, sealed


def reconstruct_warm(archive, count):
    """Replay the scored final population and take globally distinct ranked births."""
    archive = Path(archive).resolve()
    if not 0 <= count <= 2048 or count % directed.ISLANDS:
        raise ValueError("Use at most 2048 warm births, with equal island quotas")
    quota = count // directed.ISLANDS
    provenance = {
        "source_run": str(archive), "count": count, "per_island": quota,
        "source_generation": ARCHIVE_GENERATION if count else None,
        "selection": "Historical rank by island on committed final-generation rows; "
                     "reconstruct birth state from the saved plan and previous-generation parents; "
                     "skip SHA-256 duplicates across all destination islands.",
        "files": {}, "selected": [], "distinct_birth_genomes": 0,
    }
    if not count:
        return [], {}, provenance

    rows, parents, sealed = _archive_sources(archive, provenance["files"])
    representative_counts = Counter()
    by_id = {row["id"]: row for row in rows}
    sealed_genomes = sealed["genomes"]
    sealed_digests = set()
    for identifier, stored in sealed_genomes.items():
        archived = by_id[identifier]
        replayed = directed.candidate(archived, parents, directed.NODES)
        digest = _digest(replayed)
        if digest != _digest(stored):
            raise ValueError(f"Final archive elite {identifier} does not match its reconstructed birth")
        sealed_digests.add(digest)
        representative_counts[archived["island"]] += 1

    pools = [iter(enumerate(sorted((row for row in rows if row["island"] == island),
                                  key=rank, reverse=True), start=1))
             for island in range(directed.ISLANDS)]
    selected, bank, seen = [], {}, set()
    for lane in range(quota):
        for island, pool in enumerate(pools):
            for position, row in pool:
                genome = directed.candidate(row, parents, directed.NODES)
                digest = _digest(genome)
                if digest not in seen:
                    break
            else:
                raise ValueError(f"Source island {island} exhausted after {lane} distinct births")
            directed.validate_genome(genome, directed.NODES)
            if not bool(genome["directed"]):
                raise ValueError(f"Warm source {row['id']} is not directed")
            seen.add(digest)
            identifier = row["id"]
            selected.append({"id": identifier, "lineage": row["lineage"], "island": island, "lane": lane,
                             "source_generation": ARCHIVE_GENERATION, "source_rank": position,
                             "source_selection_score": row["selection_score"], "source_genome_sha256": digest,
                             "saved_elite": identifier in sealed_genomes})
            bank[identifier] = {**genome, "plastic": default_plastic()}
    provenance.update(selected=selected, distinct_birth_genomes=len(seen),
                      source_scored_candidates=len(rows), source_saved_elites=len(sealed_genomes),
                      source_saved_distinct_births=len(sealed_digests),
                      source_saved_elites_per_island={str(island): size for island, size in sorted(representative_counts.items())},
                      selected_saved_elites=sum(row["saved_elite"] for row in selected))
    return selected, bank, provenance


def plan(config, generation, previous, warm=()):
    """Share seed, operator, and budgets across arms while retaining their own parents."""
    entries = exploration.plan(config, generation, previous)
    if generation == 1:
        assignments = {(row["island"], row["lane"]): row for row in warm}
        for offset, entry in enumerate(entries):
            archived = assignments.get((entry["island"], offset // config["islands"]))
            if archived is not None:
                entry.update(kind="warm", parent=archived["id"], parent_island=archived["island"],
                             lineage=f"archive:{archived['lineage']}", source_lineage=archived["lineage"],
                             source_generation=archived["source_generation"],
                             source_genome_sha256=archived["source_genome_sha256"])
        if sum(entry["kind"] == "warm" for entry in entries) != len(warm):
            raise ValueError("Warm founders do not occupy distinct in-range slots")
        return entries

    pairs = config["perturbation_pairs_per_island"]
    for island in range(config["islands"]):
        local = [row for row in entries if row["island"] == island and row["kind"] == "local"
                 and "exploration" not in row]
        rng = random.Random(f"plastic-pairs:{config['seed']}:{generation}:{island}")
        chosen = rng.sample(local, 2 * pairs)
        for pair, (plus, minus) in enumerate(zip(chosen[::2], chosen[1::2], strict=True)):
            # Retain the lane's paired candidate ID and scheduled seed for auditing.
            identifier = minus["id"]
            scheduled_seed = minus["seed"]
            minus.update(plus, id=identifier, scheduled_seed=scheduled_seed)
            seed = rng.randrange(1, 2**31)
            for row, sign in ((plus, 1), (minus, -1)):
                row["perturbation"] = {"pair": pair, "seed": seed, "sign": sign}
    return entries


def candidate(entry, previous, nodes, warm_genomes=None):
    genome = directed.candidate(entry, previous, nodes, warm_genomes)
    kind = entry["kind"]
    if kind == "founder":
        return {**genome, "plastic": default_plastic()}
    if kind in ("warm", "elite"):
        return genome

    rng = torch.Generator(device="cpu").manual_seed(entry["seed"] + 2**34)
    plastic = genome["plastic"]
    if kind in ("intra", "inter"):
        donor = previous["genomes"][entry["donor"]]["plastic"]
        mask = torch.rand(plastic.shape, generator=rng) < .25
        # Draw at least one locus, including on very small allele vectors.
        mask[torch.randint(plastic.numel(), (), generator=rng)] = True
        plastic = torch.where(mask, donor, plastic)
    scale = 2.5 if kind == "wide" else 1.
    plastic = (plastic + torch.randn(plastic.shape, generator=rng) * (PLASTIC_MUTATION_STEP * scale)).clamp_(-1., 1.)
    trial = entry.get("perturbation")
    if trial is not None:
        direction_rng = torch.Generator(device="cpu").manual_seed(trial["seed"])
        direction = torch.rand(plastic.shape, generator=direction_rng).ge_(.5).mul_(2.).sub_(1.)
        step = (1. - plastic.abs()).clamp_(max=PLASTIC_PERTURBATION)
        plastic.add_(step * direction, alpha=trial["sign"])
    return {**genome, "plastic": plastic}


def _perturbation_summary(rows, selected):
    groups = {}
    for row in rows:
        trial = row.get("perturbation")
        if trial is not None:
            groups.setdefault((row["island"], trial["pair"]), {})[trial["sign"]] = row
    comparisons = []
    for (island, pair), siblings in sorted(groups.items()):
        plus, minus = siblings[1], siblings[-1]
        if (plus["parent"], plus["seed"], plus["perturbation"]["seed"]) != (
            minus["parent"], minus["seed"], minus["perturbation"]["seed"]
        ):
            raise ValueError("Mismatched plastic perturbation siblings")
        comparisons.append({"island": island, "pair": pair, "direction_seed": plus["perturbation"]["seed"],
                            "plus": plus["id"], "minus": minus["id"],
                            "selection_delta": plus["selection_score"] - minus["selection_score"],
                            "strict_delta": plus["strict"] - minus["strict"],
                            "plus_retained": plus["id"] in selected, "minus_retained": minus["id"] in selected})
    return {"pairs": len(comparisons), "paired": comparisons,
            "mean_selection_delta": fmean(row["selection_delta"] for row in comparisons) if comparisons else None}


def select(config, generation, rows, previous, warm_genomes):
    """Use the established same-tape contests and lineage-capped parent selection."""
    settings = config["breeding"]
    slots = settings["slots_per_island"]
    pools, genomes, branches, champions = [], {}, [], []
    for island in range(config["islands"]):
        ranked = sorted((row for row in rows if row["island"] == island), key=rank, reverse=True)
        by_id = {row["id"]: row for row in ranked}
        cache = {}

        def genome_for(row):
            identifier = row["id"]
            if identifier not in cache:
                cache[identifier] = candidate(row, previous, config["nodes"], warm_genomes)
            return cache[identifier]

        if previous:
            groups = [[] for _ in range(slots)]
            for row in ranked:
                if "exploration" in row:
                    groups[row["exploration"]["slot"]].append(row)
            states = [exploration._admit(config, generation, island, slot, group) for slot, group in enumerate(groups)]
            exploration._refresh(states, ranked, island, generation, settings, genome_for)
        else:
            states = [{"slot": slot, "id": row["id"], "opened_generation": generation + 1}
                      for slot, row in enumerate(exploration._diverse_start(ranked, slots))]
        protected = [by_id[state["id"]] for state in states]
        ordinary = exploration._ordinary_elites(ranked, protected, settings["ordinary_parents_per_island"],
                                                settings["ordinary_lineage_cap"])
        pools.append(ordinary + protected)
        genomes.update({row["id"]: genome_for(row) for row in pools[-1]})
        branches.append(states)
        champions.append({"row": ranked[0], "genome": genome_for(ranked[0])})

    return {"islands": pools, "genomes": genomes, "exploration": branches, "champions": champions,
            "perturbation_summary": _perturbation_summary(rows, genomes)}
