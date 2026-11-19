"""Test what the finished curriculum stored in birth genomes and lifetime changes."""
from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import json
from pathlib import Path
import statistics

import torch

from experiments.seed_evolution.large_runner import _write
from experiments.undirected.evolved_plasticity import GraphBatch
from experiments.undirected.evolved_plasticity_world import FIRST_FINISH, N_SYMBOLS
from experiments.undirected.plasticity_curriculum import make_phase_schedule, rollout_phase
from experiments.undirected.plasticity_curriculum_run import ARMS, OUT, candidate


SAVE_DIR = OUT.with_name("2026-11-16-plasticity-inheritance")
METRICS = ("fitness", "completion", "strict", "timing_clean", "ready_coverage",
           "version_accuracy", "symbol_accuracy", "strict_a_first", "strict_return", "idle")


def for_same_maps(tape, maps):
    """Keep the exact public episode tape and switch only private labels."""
    request_count, actors = tape.raw.shape
    contexts = (torch.arange(request_count, device=tape.raw.device) // tape.requests_per_phase == 1).long()
    actor = torch.arange(actors, device=tape.raw.device)
    versions = (tape.target - FIRST_FINISH) // N_SYMBOLS
    target = FIRST_FINISH + N_SYMBOLS * versions + maps[actor, contexts[:, None], tape.raw]
    return replace(tape, target=target, maps=maps)


def graft(graph, parent):
    """Transfer the completed life's reciprocal edge weights and mask to a fresh life."""
    graph.mask.copy_(parent.mask)
    graph.w.copy_(parent.w)
    graph.g.copy_(graph.mask).mul_(.12).sub_(.06)
    graph._update_matrix()
    if graph.mode != "frozen":
        graph._index_live()
    return graph


def scores(graph, tape, stage):
    metrics = rollout_phase(graph, tape, graph.population, stage=stage)
    return {key: value.cpu().tolist() for key, value in metrics.items() if key in METRICS}


def aggregate(rows):
    return {key: statistics.fmean(row[key] for row in rows) for key in METRICS}


def one(graph, tape, stage="conflict"):
    return {key: values[0] for key, values in scores(graph, tape, stage).items()}


def contrasts(trials, pairs):
    output = {}
    for first, second in pairs:
        outputs = {}
        for key in METRICS:
            differences = [trial[first][key] - trial[second][key] for trial in trials]
            sem = statistics.stdev(differences) / len(differences) ** .5
            outputs[key] = {"mean": statistics.fmean(differences), "sem": sem, "paired": differences}
        output[f"{first}_minus_{second}"] = outputs
    return output


def trace_transfer(path, seeds, actors, device):
    checkpoints = torch.load(path / "champions.pt", map_location="cpu", weights_only=True)
    output = {}
    for arm, record in checkpoints.items():
        genome = record["genome"]
        no_direct_receipt = {**genome, "rule": genome["rule"].clone()}
        no_direct_receipt["rule"][:, 5:] = 0
        trials = {"same_maps": [], "new_maps": []}
        changes = []
        for source_seed in seeds:
            target_seed = source_seed + 100_000
            source = make_phase_schedule(stage="conflict", seed=source_seed, actors=actors, device=device)
            destination = make_phase_schedule(stage="conflict", seed=target_seed, actors=actors, device=device)
            parent = GraphBatch([genome], actors, device, mode=arm, seed=source_seed)
            parent_scores = one(parent, source)
            same_maps = for_same_maps(destination, source.maps)
            if not torch.equal(same_maps.public, destination.public):
                raise AssertionError("Private-map intervention changed the public tape")
            birth_w = genome["birth_w"].to(parent.device).view(1, 1, -1)
            birth_mask = genome["birth_mask"].to(parent.device).view(1, 1, -1)
            changes.append({
                "seed": source_seed, "completion": parent_scores["completion"],
                "net_l1_change_per_actor": (parent.w - birth_w).abs().sum(-1).mean().item(),
                "support_changes_per_actor": (parent.mask != birth_mask).sum(-1).float().mean().item(),
            })
            for label, tape in (("same_maps", same_maps), ("new_maps", destination)):
                direct = GraphBatch([genome, no_direct_receipt], actors, device, mode=arm, seed=target_seed)
                evaluated = scores(direct, tape, "conflict")
                trial = {"seed": source_seed, "destination_seed": target_seed}
                trial["birth_native"] = {key: values[0] for key, values in evaluated.items()}
                trial["birth_no_direct_receipt"] = {key: values[1] for key, values in evaluated.items()}
                for mode in ("frozen", arm):
                    offspring = GraphBatch([genome], actors, device, mode=mode, seed=target_seed)
                    if mode == "frozen":
                        trial["birth_frozen"] = one(offspring, tape)
                        offspring = GraphBatch([genome], actors, device, mode=mode, seed=target_seed)
                    label_for_mode = "acquired_frozen" if mode == "frozen" else "acquired_native"
                    trial[label_for_mode] = one(graft(offspring, parent), tape)
                trials[label].append(trial)
        comparisons = (
            ("birth_native", "birth_frozen"), ("birth_native", "birth_no_direct_receipt"),
            ("acquired_native", "birth_native"), ("acquired_frozen", "birth_frozen"),
        )
        output[arm] = {
            "candidate": record["id"], "source_lives": changes,
            **{label: {"trials": rows, "modes": {name: aggregate([r[name] for r in rows])
                          for name in rows[0] if isinstance(rows[0][name], dict)},
                       "contrasts": contrasts(rows, comparisons)}
               for label, rows in trials.items()},
        }
    return output


def read_elites(path, generation):
    return torch.load(path / "generations" / f"{generation:04d}" / "elites.pt",
                      map_location="cpu", weights_only=True)


def reselection(path, protocol, seeds, actors, device):
    """Rerank an archived pool by real strict completion, then test on fresh worlds."""
    output = {}
    for stage in ("timing", "version"):
        generation = protocol["stage_bounds"][stage][1]
        folder = path / "generations" / f"{generation:04d}"
        last = read_elites(path, generation)
        previous = read_elites(path, generation - 1)
        audit = json.loads((path / "audits" / f"{stage}.json").read_bytes())
        output[stage] = {}
        for arm in ARMS:
            rows = [
                row for batch in sorted((folder / "batches" / arm).glob("*.json.gz"))
                for row in json.loads(gzip.decompress(batch.read_bytes()))
            ]
            actual = audit["candidates"][arm]
            winner = max(rows, key=lambda r: (r["strict"], r["completion"], r["timing_clean"], -r["idle"]))
            genomes = [last[arm]["genomes"][actual], candidate(winner, previous[arm], protocol["nodes"])]
            trials = []
            for seed in seeds:
                tape = make_phase_schedule(stage=stage, seed=seed, actors=actors, device=device)
                graph = GraphBatch(genomes, actors, device, mode=arm, seed=seed)
                evaluated = scores(graph, tape, stage)
                trials.append({
                    "seed": seed,
                    "fitness_selected": {key: values[0] for key, values in evaluated.items()},
                    "strict_selected": {key: values[1] for key, values in evaluated.items()},
                })
            output[stage][arm] = {
                "fitness_selected": actual, "strict_selected": winner["id"], "trials": trials,
                "modes": {name: aggregate([row[name] for row in trials])
                          for name in ("fitness_selected", "strict_selected")},
                "contrasts": contrasts(trials, (("strict_selected", "fitness_selected"),)),
            }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=OUT)
    parser.add_argument("--out", type=Path, default=SAVE_DIR / "counterfactual.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--actors", type=int, default=16)
    args = parser.parse_args()
    if args.seeds < 2 or args.actors < 1:
        parser.error("At least two paired seeds and one actor are required")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    protocol = json.loads((args.run_dir / "protocol.json").read_bytes())
    if json.loads((args.run_dir / "progress.json").read_bytes())["state"] != "complete":
        parser.error("The run must have completed")
    seeds = list(range(72_180_301, 72_180_301 + args.seeds))
    with torch.no_grad():
        selection = reselection(args.run_dir, protocol, seeds, args.actors, args.device)
        print("Finished strict-completion reselection", flush=True)
        transfer = trace_transfer(args.run_dir, seeds, args.actors, args.device)
    _write(args.out, {
        "actors": args.actors, "source_seeds": seeds, "device": args.device,
        "strict_reselection": selection, "transfer": transfer,
    })
    print(args.out, flush=True)


if __name__ == "__main__":
    main()
