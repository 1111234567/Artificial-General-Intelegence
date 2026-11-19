"""Select distinct birth genomes from the completed uncapped experiment."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import random

import torch

from experiments.undirected.evolved_plasticity import NODES, RULE_FEATURES
from experiments.undirected.uncapped_plasticity_run import rank


SOURCE_GENERATIONS = (768, 1024, 896, 512, 256)
MAX_ARCHIVED_FOUNDERS = len(SOURCE_GENERATIONS) * 128
_PAIRS = NODES * (NODES - 1) // 2
_BIRTH_TENSORS = {
    "birth_w": ((_PAIRS,), torch.float32),
    "birth_mask": ((_PAIRS,), torch.bool),
    "node_bias": ((NODES,), torch.float32),
    "rule": ((2, len(RULE_FEATURES)), torch.float32),
}


@dataclass(frozen=True)
class _ArchivedGenome:
    candidate_id: str
    lineage: str
    generation: int
    genome: dict[str, torch.Tensor]
    digest: str
    audit_champion: bool


def _birth_genome(tensors, candidate_id):
    genome = {}
    digest = hashlib.sha256()
    for name, (shape, dtype) in _BIRTH_TENSORS.items():
        tensor = tensors.get(name)
        if not isinstance(tensor, torch.Tensor) or tensor.shape != shape or tensor.dtype != dtype:
            raise ValueError(f"Archived candidate {candidate_id} has an incompatible {name} tensor")
        genome[name] = tensor
        digest.update(tensor.numpy().tobytes())
    return genome, digest.hexdigest()


def load_founders(run_dir: Path, count: int, seed: int) -> tuple[
    list[dict[str, str]], dict[str, dict[str, torch.Tensor]], dict
]:
    """Return archived founder descriptors, CPU birth tensors, and selection provenance."""
    if count < 0:
        raise ValueError(f"Archived founder count must be nonnegative, got {count}")

    run_dir = run_dir.resolve()
    provenance = {
        "source_run": str(run_dir),
        "count": count,
        "seed": seed,
        "source_generations": [],
        "files": {},
        "selected": [],
    }
    if count == 0:
        return [], {}, provenance

    rng = random.Random(seed)
    groups = {}
    champions = []
    for generation in SOURCE_GENERATIONS:
        checkpoint_path = f"generations/{generation:04d}/elites.pt"
        audit_path = f"audits/{generation:04d}.json"
        checkpoint_bytes = (run_dir / checkpoint_path).read_bytes()
        audit_bytes = (run_dir / audit_path).read_bytes()
        checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
        champion_id = json.loads(audit_bytes)["candidate"]
        provenance["files"][checkpoint_path] = hashlib.sha256(checkpoint_bytes).hexdigest()
        provenance["files"][audit_path] = hashlib.sha256(audit_bytes).hexdigest()

        rows = [row for island in checkpoint["islands"] for row in island]
        rng.shuffle(rows)
        rows.sort(key=rank, reverse=True)
        group = []
        for row in rows:
            candidate_id = row["id"]
            genome, digest = _birth_genome(checkpoint["genomes"][candidate_id], candidate_id)
            record = _ArchivedGenome(
                candidate_id, row["lineage"], generation, genome, digest, candidate_id == champion_id,
            )
            group.append(record)
        champion = next((record for record in group if record.audit_champion), None)
        if champion is None:
            raise ValueError(f"Audit champion {champion_id} is missing from {run_dir / checkpoint_path}")
        champions.append(champion)
        groups[generation] = group

    selected = []
    ids = set()
    digests = set()
    lineage_counts = Counter()

    def available(record):
        return record.candidate_id not in ids and record.digest not in digests

    def select(record):
        selected.append(record)
        ids.add(record.candidate_id)
        digests.add(record.digest)
        lineage_counts[record.lineage] += 1

    for champion in champions:
        if len(selected) < count and available(champion):
            select(champion)

    # Give each generation a seat per pass. Within a generation, favor the
    # founder lineage with the fewest seats so far and then its best-ranked elite.
    while len(selected) < count:
        previous_count = len(selected)
        for group in groups.values():
            record = min(
                (record for record in group if available(record)),
                key=lambda record: lineage_counts[record.lineage],
                default=None,
            )
            if record is not None:
                select(record)
                if len(selected) == count:
                    break
        if len(selected) == previous_count:
            raise ValueError(f"Requested {count} archived founders, but only {len(selected)} distinct genomes exist")

    descriptors = [{"id": record.candidate_id, "lineage": record.lineage} for record in selected]
    genomes = {record.candidate_id: record.genome for record in selected}
    provenance["source_generations"] = sorted({record.generation for record in selected})
    provenance["selected"] = [
        {
            **descriptor,
            "generation": record.generation,
            "genome_sha256": record.digest,
            "audit_champion": record.audit_champion,
        }
        for descriptor, record in zip(descriptors, selected, strict=True)
    ]
    return descriptors, genomes, provenance
