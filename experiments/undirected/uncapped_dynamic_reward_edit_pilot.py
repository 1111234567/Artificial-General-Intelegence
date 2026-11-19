"""Compare directly written, dynamically selected edges with skipped writes."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import random
import statistics

import torch

from experiments.seed_evolution.large_runner import _save, _utc, _write
from experiments.undirected import uncapped_dynamic_reward_editor as editor
from experiments.undirected import uncapped_reward_edit_pilot as fixed_pilot
from experiments.undirected import variable_birth_genome as birth
from experiments.undirected.evolved_plasticity import NODES


OUT = fixed_pilot.OUT.with_name("2026-11-17-dynamic-reward-edit-pilot-v1")
METRICS = (*fixed_pilot.METRICS, "edit_unique_edges")


def new_selector(genome, chooser):
    rng = torch.Generator().manual_seed(chooser.randrange(1, 2**31))
    core = {key: value for key, value in genome.items() if key != "edit_edges"}
    return {**core, "edit_selector": torch.randn((2, NODES), generator=rng) * .2}


def mutate(parent, chooser):
    child = birth.mutate_genome(parent, chooser.randrange(1, 2**31))
    rng = torch.Generator().manual_seed(chooser.randrange(1, 2**31))
    child["edit_selector"].add_(torch.randn((2, NODES), generator=rng) * .05)
    return child


def rows_for(genomes, **settings):
    measured = editor.evaluate(genomes, **settings)
    columns = {key: measured[key].cpu().tolist() for key in METRICS}
    return [{key: values[i] for key, values in columns.items()} for i in range(len(genomes))]


def means(rows):
    return {key: statistics.fmean(row[key] for row in rows) for key in METRICS}


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    chooser = random.Random(12_911_628)
    founders, warm = fixed_pilot.initial_population(chooser, args.population)
    population = [new_selector(genome, chooser) for genome in founders]
    history = []
    _write(args.out / "protocol.json", {
        "population": args.population, "generations": args.generations, "actors": 16, "visits": 8,
        "source_archive": str(fixed_pilot.ARCHIVE),
        "archive_sha256": hashlib.sha256(fixed_pilot.ARCHIVE.read_bytes()).hexdigest(),
        "warm_founders": warm, "device": args.device,
        "timing": {"min_wait": 8, "max_wait": 12, "finish_window": 4},
        "selection": "strict - 0.05 wrong_answer_rate + 0.5 return_first_correct",
        "editor": {"controllers": 2, "nodes_per_controller": editor.SELECTOR_NODES,
                   "pairs_per_controller": editor.PAIRS_PER_CONTROLLER, "weight_step": editor.EDIT_STEP},
        "write": "direct signed command after ordinary plasticity",
        "paired_variants": fixed_pilot.VARIANTS, "holdout_seeds": list(range(79_156_000, 79_156_008)),
        "precision": "BF16 recurrent forward, FP32 master weights and address scores",
        "protocol_reference": str(fixed_pilot.OUT / "protocol.json"),
        "sources": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (Path(__file__), Path(editor.__file__))},
    })
    for generation in range(1, args.generations + 1):
        world_seed = 78_150_100 + generation
        rows = rows_for(population, seed=world_seed, device=args.device)
        order = sorted(range(len(rows)), key=lambda i: (fixed_pilot.score(rows[i]), rows[i]["strict"]), reverse=True)
        best = rows[order[0]]
        history.append({"generation": generation, "seed": world_seed, "winner": best, "population_mean": means(rows)})
        _write(args.out / "history.json", history)
        print(f"generation={generation} score={fixed_pilot.score(best):.4f} "
              f"return_first={best['return_first_correct']:.4f} unique_edits={best['edit_unique_edges']:.1f}", flush=True)
        elites = [population[i] for i in order[:8]]
        if generation < args.generations:
            population = elites + [mutate(chooser.choice(elites), chooser) for _ in range(args.population - 16)]
            population += [new_selector(birth.make_genome(chooser.randrange(1, 2**31)), chooser) for _ in range(8)]

    tested = elites[:4]
    _save(args.out / "elites.pt", {"genomes": tested, "selection": [rows[i] for i in order[:4]]})
    trials = []
    for seed in range(79_156_000, 79_156_008):
        trial = {"seed": seed}
        for name, (edit_enabled, reward_enabled) in fixed_pilot.VARIANTS.items():
            trial[name] = rows_for(tested, seed=seed, device=args.device, edit_enabled=edit_enabled,
                                   reward_enabled=reward_enabled)
        trials.append(trial)
        print(f"holdout seed {seed} done", flush=True)
    result = {
        "completed_utc": _utc(), "tested_genomes": len(tested),
        "means": {name: means([row for trial in trials for row in trial[name]]) for name in fixed_pilot.VARIANTS},
        "contrasts": {
            "editor_minus_skip_write": fixed_pilot.comparison(trials, "editor", "skip_write"),
            "editor_minus_reward_input_zero": fixed_pilot.comparison(trials, "editor", "reward_input_zero"),
        },
        "paired": trials,
    }
    _write(args.out / "holdout.json", result)
    for name, result_row in result["means"].items():
        print(name, {key: round(result_row[key], 5) for key in
                     ("strict", "symbol_accuracy", "return_first_correct", "return_A_first_correct",
                      "edit_count", "edit_unique_edges")}, flush=True)


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
