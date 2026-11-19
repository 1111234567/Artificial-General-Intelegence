"""Small paired study of reward-triggered direct weight editing."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import random
import statistics

import torch

from experiments.seed_evolution.large_runner import _save, _utc, _write
from experiments.undirected import uncapped_reward_editor as editor
from experiments.undirected import variable_birth_genome as birth
from experiments.undirected.evolved_plasticity import INPUTS, NODES
from experiments.undirected.evolved_plasticity_world import N_ACTIONS
from experiments.undirected.uncapped_ready_signal_run import rank as archived_rank


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "data/undirected/2026-11-16-plasticity-island-crossover-v1/generations/0768/elites.pt"
OUT = ROOT / "data/undirected/2026-11-17-reward-edit-pilot-v1"
METRICS = ("strict", "completion", "ready_coverage", "symbol_accuracy", "symbol_coverage",
           "wrong_answer_rate", "return_first_correct", "return_A_first_correct", "return_B_first_correct",
           "edit_count", "edit_abs_delta", "edit_signed_delta")
VARIANTS = {"editor": (True, True), "skip_write": (False, True), "reward_input_zero": (True, False)}
_U, _V = torch.triu_indices(NODES, NODES, 1)
_ACTION_PAIR = ((_U >= INPUTS) & (_U < INPUTS + N_ACTIONS)) | ((_V >= INPUTS) & (_V < INPUTS + N_ACTIONS))


def random_edge(mask, exclude, chooser, prefer_action=False):
    eligible = mask.clone()
    for edge in exclude:
        eligible[edge] = False
    if prefer_action and bool((eligible & _ACTION_PAIR).any()):
        eligible &= _ACTION_PAIR
    indices = eligible.nonzero(as_tuple=True)[0].tolist()
    if not indices:
        indices = [edge for edge in range(mask.numel()) if edge not in exclude]
    return chooser.choice(indices)


def add_editor(genome, chooser):
    first = random_edge(genome["birth_mask"], (), chooser, prefer_action=True)
    second = random_edge(genome["birth_mask"], (first,), chooser)
    return {**genome, "edit_edges": torch.tensor([first, second], dtype=torch.long)}


def mutate(parent, chooser):
    child = birth.mutate_genome(parent, chooser.randrange(1, 2**31))
    addresses = child["edit_edges"]
    if chooser.random() < .35:
        slot = chooser.randrange(2)
        addresses[slot] = random_edge(child["birth_mask"], addresses.tolist(), chooser)
    return child


def initial_population(chooser, size):
    checkpoint = torch.load(ARCHIVE, map_location="cpu", weights_only=True)
    pools = [sorted(pool, key=archived_rank, reverse=True) for pool in checkpoint["islands"]]
    warm = [pool[rank]["id"] for rank in range(3) for pool in pools]
    genomes = [checkpoint["genomes"][name] for name in warm]
    genomes.extend(birth.make_genome(chooser.randrange(1, 2**31)) for _ in range(size - len(warm)))
    return [add_editor(genome, chooser) for genome in genomes], warm


def as_rows(metrics, size):
    columns = {name: metrics[name].detach().cpu().tolist() for name in METRICS}
    return [{name: values[i] for name, values in columns.items()} for i in range(size)]


def score(row):
    return row["strict"] - .05 * row["wrong_answer_rate"] + .5 * row["return_first_correct"]


def summarize(rows):
    return {key: statistics.fmean(row[key] for row in rows) for key in METRICS}


def comparison(rows, first, second):
    contrasts = {}
    for name in METRICS:
        differences = [statistics.fmean(a[name] - b[name] for a, b in
                                       zip(row[first], row[second], strict=True)) for row in rows]
        center = statistics.fmean(differences)
        half_width = 2.364624 * statistics.stdev(differences) / len(differences) ** .5
        contrasts[name] = {"mean": center, "interval95": [center - half_width, center + half_width],
                           "seeds_higher": sum(x > 0 for x in differences),
                           "seeds_lower": sum(x < 0 for x in differences)}
    return contrasts


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    chooser = random.Random(12_911_628)
    population, warm = initial_population(chooser, args.population)
    protocol = {
        "source_archive": str(ARCHIVE), "archive_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        "warm_founders": warm, "population": args.population, "generations": args.generations,
        "actors": 16, "visits": 8, "device": args.device,
        "timing": {"min_wait": 8, "max_wait": 12, "finish_window": 4},
        "selection": "strict - 0.05 wrong_answer_rate + 0.5 return_first_correct",
        "link_initialization": "one address initially touches a main output; the other is sampled from birth edges",
        "paired_variants": list(VARIANTS),
        "sources": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (Path(__file__), Path(editor.__file__))},
    }
    _write(args.out / "protocol.json", protocol)
    history = []
    for generation in range(1, args.generations + 1):
        world_seed = 78_150_100 + generation
        started = _utc()
        metrics = editor.evaluate(population, seed=world_seed, device=args.device)
        rows = as_rows(metrics, len(population))
        order = sorted(range(len(rows)), key=lambda i: (score(rows[i]), rows[i]["strict"]), reverse=True)
        best = rows[order[0]]
        history.append({"generation": generation, "seed": world_seed, "started": started,
                        "winner": best, "population_mean": summarize(rows)})
        _write(args.out / "history.json", history)
        print(f"g={generation} selection={score(best):.4f} strict={best['strict']:.4f} "
              f"return_first={best['return_first_correct']:.4f} edit_count={best['edit_count']:.1f}", flush=True)
        elites = [population[i] for i in order[:8]]
        if generation < args.generations:
            population = elites + [mutate(chooser.choice(elites), chooser)
                                   for _ in range(args.population - len(elites) - 8)]
            population += [add_editor(birth.make_genome(chooser.randrange(1, 2**31)), chooser) for _ in range(8)]

    tested = elites[:4]
    _save(args.out / "elites.pt", {"genomes": tested, "selection": [rows[i] for i in order[:4]]})
    trials = []
    for seed in range(79_156_000, 79_156_008):
        trial = {"seed": seed}
        for name, (edit_enabled, reward_enabled) in VARIANTS.items():
            metrics = editor.evaluate(tested, seed=seed, device=args.device, edit_enabled=edit_enabled,
                                      reward_enabled=reward_enabled)
            trial[name] = as_rows(metrics, len(tested))
        trials.append(trial)
        print(f"holdout seed {seed} done", flush=True)
    results = {
        "completed_utc": _utc(), "tested_genomes": len(tested),
        "means": {name: summarize([row for trial in trials for row in trial[name]]) for name in VARIANTS},
        "contrasts": {
            "editor_minus_skip_write": comparison(trials, "editor", "skip_write"),
            "editor_minus_reward_input_zero": comparison(trials, "editor", "reward_input_zero"),
        },
        "paired": trials,
    }
    _write(args.out / "holdout.json", results)
    for name, metrics in results["means"].items():
        print(name, {k: round(metrics[k], 5) for k in
                     ("strict", "symbol_accuracy", "return_first_correct", "return_A_first_correct", "edit_count")})
    print(args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.population < 16 or args.generations < 1:
        parser.error("Use at least 16 candidates and one generation.")
    if (args.out / "protocol.json").exists():
        parser.error("Choose a new pilot output directory.")
    run(args)


if __name__ == "__main__":
    main()
