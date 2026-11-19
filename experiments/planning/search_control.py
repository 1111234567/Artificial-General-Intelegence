"""POST-HOC search-only controls for an already frozen planning experiment.

This diagnostic was added after the first main-seed test result was seen.
It does not alter the main protocol, datasets, trained models, or results.
Both controls reuse the exact same bounded best-first search implementation
and paired cases as neural guidance, while using no trainable network.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Sequence

import torch

from . import run, world


POSTHOC_NOTE = (
    "POST-HOC diagnostic added after main seed0 test results were inspected. "
    "This was not a preregistered comparison; main protocol, checkpoints and results are unchanged. "
    "The primary control is uniform_unique: equal prior mass for each unique observable recipe. "
    "uniform_action gives equal mass to action IDs, so recipe mass depends on duplicate multiplicity. "
    "Expanded/scored states match search decision counts, not neural FLOPs; actual neural core calls are zero."
)


class UniformPolicy(torch.nn.Module):
    """Uniform observable priors compatible with ``run.neural_best_first``.

    ``unique`` uses logits[a] = -log(number of equivalent recipe action IDs),
    giving each (consume, produce) group the same total legal probability.
    ``action`` uses zero logits, leaving action IDs equally probable. Neither
    variant reads goal, inventory, metadata, solver information, or learned
    parameters. Legality still comes from the public simulator.
    """

    def __init__(self, mode: str = "unique") -> None:
        super().__init__()
        if mode not in {"unique", "action"}:
            raise ValueError("mode must be unique or action")
        self.mode = mode
        # The search queries one problem at a time. Cache only observable
        # recipes, and invalidate on changed recipes, shape, or device.
        self._cached_recipes: torch.Tensor | None = None
        self._cached_logits: torch.Tensor | None = None

    def forward(self, features: torch.Tensor, legal: torch.Tensor,
                rounds: int | None = None) -> torch.Tensor:
        if rounds not in {None, 1}:
            raise ValueError("a uniform prior is one search scoring decision")
        if features.ndim != 3 or (features.shape[-1] - 1) % 4:
            raise ValueError("expected public rule features [batch, actions, 4*n_items+1]")
        if legal.shape != features.shape[:2] or legal.dtype != torch.bool:
            raise ValueError("legal must be a boolean [batch, actions] mask")
        logits = torch.zeros(legal.shape, dtype=torch.float64, device=features.device)
        if self.mode == "unique":
            n_items = (features.shape[-1] - 1) // 4
            for index in range(len(features)):
                recipes = features[index, :, :2 * n_items]
                cached = self._cached_recipes
                if (cached is None or cached.shape != recipes.shape
                        or cached.device != recipes.device or not torch.equal(cached, recipes)):
                    _, inverse, counts = torch.unique(recipes, dim=0, return_inverse=True,
                                                      return_counts=True)
                    self._cached_recipes = recipes.clone()
                    self._cached_logits = -counts[inverse].to(torch.float64).log()
                logits[index] = self._cached_logits
        return logits.masked_fill(~legal, -torch.inf)


def search_only(problem: world.Problem, *, mode: str, budget: int, max_steps: int,
                branch: int, beam_width: int) -> dict:
    """Use the frozen search engine; reinterpret its counter without a network."""
    result = run.neural_best_first(UniformPolicy(mode), problem, device="cpu",
                                   core_budget=budget, max_steps=max_steps,
                                   branch=branch, beam_width=beam_width)
    expanded = result.pop("core_calls")
    result.update(core_calls=0, neural_core_calls=0, expanded_scored_states=expanded,
                  expansion_budget=budget, prior=mode, posthoc=True)
    if result["stop_reason"] == "core_budget":
        result["stop_reason"] = "expansion_budget"
    return result


def aggregate(rows: Sequence[dict]) -> list[dict]:
    """Preserve the main summary schema and add the non-neural expansion cost."""
    summaries = run.aggregate_results(rows)
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for row in rows:
        method, split, family, depth = (row[key] for key in
                                        ("method", "distribution", "family", "depth"))
        for key in ((method, "all", "all", "all"),
                    (method, split, "all", "all"), (method, split, family, depth)):
            buckets[key].append(row["expanded_scored_states"])
    for summary in summaries:
        values = buckets[tuple(summary[key] for key in ("method", "distribution", "family", "depth"))]
        summary.update(mean_expanded_scored_states=sum(values) / len(values),
                       mean_neural_core_calls=0.0, posthoc=True)
    return summaries


def evaluate(out: Path, *, threads: int = 1,
             modes: Sequence[str] = ("unique", "action")) -> dict:
    """Write separate one-shot post-hoc files, leaving every main file intact."""
    if threads < 1 or not modes or len(set(modes)) != len(modes):
        raise ValueError("threads must be positive and variants must be distinct")
    if not set(modes) <= {"unique", "action"}:
        raise ValueError("unsupported uniform prior")
    config, manifest = run.verify_experiment(out)
    records = run.load_records(out / manifest["datasets"]["test"]["file"])
    subset = run.paired_subset(records, config["beam_limit"])
    subset_ids = {record["case_id"] for record in subset}
    main_summary_path = out / "summary_test.json"
    if main_summary_path.exists():
        main_summary = json.loads(main_summary_path.read_text(encoding="utf-8"))
        if set(main_summary["paired_search_subset_ids"]) != subset_ids:
            raise ValueError("the post-hoc subset does not match the main paired search subset")
    paths = {"started": out / "uniform_search.started.json",
             "rows": out / "uniform_search_test.jsonl",
             "summary": out / "summary_uniform_search_test.json"}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"post-hoc uniform search output already exists: {out}")
    # Exclusive creation prevents an accidental duplicate diagnostic run.
    with paths["started"].open("x", encoding="utf-8") as stream:
        stream.write(run.json_text({
            "posthoc": True, "posthoc_note": POSTHOC_NOTE,
            "config_sha256": manifest["config_sha256"], "seed": config["seed"],
            "source_sha256": run.file_sha256(Path(__file__)),
            "budgets": config["beam_budgets"], "beam_branch": config["beam_branch"],
            "beam_width": config["beam_width"], "max_steps": config["max_steps"],
            "paired_subset_ids": sorted(subset_ids), "modes": list(modes),
            "device": "cpu", "threads": threads, "torch": str(torch.__version__),
            "command": sys.argv,
        }) + "\n")
    torch.set_num_threads(threads)
    start = time.perf_counter()
    rows = []
    with paths["rows"].open("x", encoding="utf-8", buffering=1, newline="\n") as stream:
        for mode in modes:
            for budget in config["beam_budgets"]:
                successes = 0
                for record in subset:
                    result = search_only(run.problem_from_record(record), mode=mode,
                                         budget=budget, max_steps=config["max_steps"],
                                         branch=config["beam_branch"], beam_width=config["beam_width"])
                    row = run.make_row(record, f"uniform_{mode}_c{budget}", result)
                    stream.write(run.json_text(row) + "\n")
                    rows.append(row)
                    successes += bool(result["success"])
                print(f"{out.name}: uniform_{mode}_c{budget} {successes}/{len(subset)}", flush=True)
    summary = {
        "posthoc": True, "posthoc_note": POSTHOC_NOTE,
        "split": "test_paired_search_subset", "seed": config["seed"],
        "config_sha256": manifest["config_sha256"],
        "seconds": time.perf_counter() - start,
        "all_cases": aggregate(rows), "paired_search_subset_ids": sorted(subset_ids),
        "primary_method_prefix": "uniform_unique", "secondary_method_prefix": "uniform_action",
        "budgets": config["beam_budgets"], "beam_branch": config["beam_branch"],
        "beam_width": config["beam_width"], "max_steps": config["max_steps"],
        "device": "cpu", "threads": threads,
        "source_sha256": run.file_sha256(Path(__file__)),
        "notes": [
            "No trainable parameters or checkpoint is loaded.",
            "Each final plan is independently executed and validated by the main checker.",
            "The simulator supplies current legality, exact transitions, and goal stopping.",
            "No oracle plan, distance table, generator metadata, or optimal-length metric enters a prior.",
            "core_calls and neural_core_calls are zero; expanded_scored_states counts search decisions.",
            "Equal expansion limits do not imply equal FLOPs or equal latency with neural guidance.",
            "Uniform action IDs retain duplicate-recipe multiplicity; uniform_unique removes that prior bias.",
        ],
    }
    with paths["summary"].open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(f"saved {paths['summary']}", flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, nargs="+", required=True,
                        help="one or more already frozen main experiment directories")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--variant", choices=("unique", "action"), nargs="+",
                        default=["unique", "action"], help="unique is the primary post-hoc control")
    args = parser.parse_args(argv)
    for out in args.out:
        evaluate(out.resolve(), threads=args.threads, modes=args.variant)


if __name__ == "__main__":
    main()
