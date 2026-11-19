"""Audit the expanded seed-evolution run and its committed generations."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import random
from tempfile import NamedTemporaryFile

import torch

from experiments.learned_binding import audit as binding_audit, common
from experiments.seed_evolution import audit


OUT = audit.ROOT / "data/seed-evolution/2026-11-11-large-v1"
ARMS = audit.ARMS
require = audit.require
same_tree = audit.same_tree
read = binding_audit.read_json


def validate_protocol(protocol):
    expected = dict(repeats=3, generations=32, population=64, parents=8, sigma=.15, inner_steps=64,
                    final_steps=512, source_seeds=[0, 1, 2], dev_streams=[1100, 1101],
                    test_streams=[2100, 2101], actors=16, support_ticks=2048, probe_ticks=2048,
                    random_seed=78111, b_ticks=4096, search_device="cpu", final_device="cpu",
                    dtype="float32", threads_per_worker=1)
    same_tree(protocol["config"], expected, "expanded experiment configuration")
    require(protocol["unique_candidates"] == 12096 and protocol["budget_per_arm_per_repeat"] == 2048
            and protocol["shared_first_generation"] is True, "expanded search budget")
    sample = [dict(rep=rep, id=audit.candidate_id(arm, generation, 0))
              for rep in expected["source_seeds"]
              for arm, generation in (("shared", 0), ("evolution", 31), ("random", 31))]
    same_tree(protocol["audit_candidates"], sample, "expanded candidate replay sample")
    return protocol["config"]


def committed_generations(root, generations):
    """Capture the committed prefix without touching in-flight artifacts."""
    found = set()
    for path in root.glob("generation_*.json"):
        suffix = path.stem.removeprefix("generation_")
        require(suffix.isdigit() and str(int(suffix)) == suffix, "invalid generation commit filename")
        found.add(int(suffix))
    count = len(found)
    require(count <= generations and found == set(range(count)), "noncontiguous generation commits")
    return count


def verify_attempt(folder, record):
    """Validate the stage manifests used for interrupted candidate recovery."""
    attempt = read(folder / "attempt.json")
    train = read(folder / "training.json")
    evaluation = read(folder / "evaluation-time.json")
    require(type(record["attempt"]) is int and record["attempt"] >= 1
            and attempt["attempt"] == record["attempt"], "candidate attempt marker differs")
    require(math.isfinite(record["elapsed_seconds"]) and record["elapsed_seconds"] >= 0,
            "invalid candidate elapsed time")
    reused = record["reused_artifacts"]
    require(len(reused) == len(set(reused)) and set(reused) <= {"initial.pt", "inner64.pt", "score.json"},
            "invalid recovered candidate artifacts")
    for prefix, manifest, artifact, hash_key in (
            ("train", train, "inner64.pt", "checkpoint_sha256"),
            ("eval", evaluation, "score.json", "score_sha256")):
        attempt_key, seconds_key = f"{prefix}_attempt", f"{prefix}_seconds"
        number = record[attempt_key]
        require(type(number) is int and 1 <= number <= record["attempt"]
                and manifest[attempt_key] == number, "candidate stage attempt differs: " + prefix)
        require((artifact in reused) == (number < record["attempt"]), "candidate stage recovery differs: " + prefix)
        seconds = record[seconds_key]
        require(math.isfinite(seconds) and seconds >= 0 and manifest[seconds_key] == seconds,
                "candidate stage elapsed time differs: " + prefix)
        require(manifest[hash_key] == record["artifacts"][artifact], "candidate stage hash differs: " + prefix)


class SearchLedger:
    """Reconstruct the two cumulative histories for each fixed executor."""

    def __init__(self, root, config, executors):
        self.root = root
        self.config = config
        self.executors = executors
        repeats = config["source_seeds"]
        self.records = {str(rep): {} for rep in repeats}
        self.genomes = {rep: {} for rep in repeats}
        self.schedules = {}
        self.genome_hashes = {rep: set() for rep in repeats}
        self.cumulative = {(rep, arm): [] for rep in repeats for arm in ARMS}
        self.history = []

    def expected_plan(self, generation):
        entries, candidates = [], []
        for rep in self.config["source_seeds"]:
            for arm in (("shared",) if generation == 0 else ARMS):
                prior = self.cumulative.get((rep, arm), [])
                top = audit.rank_ids(prior, self.records[str(rep)])[:self.config["parents"]]
                parent_seed = 650000 + rep * 100 + generation
                chooser = random.Random(parent_seed)
                new_ids = []
                for index in range(self.config["population"]):
                    cid = audit.candidate_id(arm, generation, index)
                    new_ids.append(cid)
                    parent = chooser.choice(top) if arm == "evolution" else None
                    candidates.append(dict(
                        rep=rep, source_seed=rep, arm=arm, generation=generation, index=index,
                        id=cid, parent_id=parent,
                        init_seed=None if parent else 710000 + rep * 10000 + generation * 100 + index,
                        mutation_seed=810000 + rep * 10000 + generation * 100 + index if parent else None,
                        sigma=self.config["sigma"], parent_selection_seed=parent_seed if parent else None,
                    ))
                entries.append(dict(rep=rep, arm=arm, generation=generation, top8_before=top, new_ids=new_ids))
        return dict(generation=generation, entries=entries, candidates=candidates)

    def verify_plan(self, generation, path):
        plan = read(path)
        same_tree(plan, self.expected_plan(generation), "independent generation plan")
        return plan

    def candidate(self, descriptor, required):
        rep, cid = descriptor["rep"], descriptor["id"]
        folder = audit.candidate_directory(self.root, rep, cid)
        marker = folder / "candidate.json"
        if not required and not marker.is_file():
            return False
        record = read(marker)
        same_tree({key: record[key] for key in descriptor}, descriptor, "candidate does not match sealed plan")
        require(cid not in self.records[str(rep)], "repeated candidate")
        require(record["genome_hash"] not in self.genome_hashes[rep], "duplicate initial genome")
        verify_attempt(folder, record)
        schedule = audit.audit_candidate(self.root, record, self.config, self.executors[rep], self.genomes[rep])
        if rep not in self.schedules:
            self.schedules[rep] = schedule
        else:
            same_tree(schedule, self.schedules[rep], "unpaired development schedule")
        self.records[str(rep)][cid] = record
        self.genome_hashes[rep].add(record["genome_hash"])
        return True

    def complete_generation(self, plan, snapshot):
        generation = plan["generation"]
        require(snapshot["generation"] == generation, "generation commit number differs")
        expected_ids = {(desc["rep"], desc["id"]) for desc in plan["candidates"]}
        saved_ids = {(row["rep"], row["id"]) for row in snapshot["candidate_ids"]}
        require(saved_ids == expected_ids and len(snapshot["candidate_ids"]) == len(expected_ids),
                "generation commit candidate coverage differs")
        rows = []
        for entry in plan["entries"]:
            rep = entry["rep"]
            for arm in ARMS if generation == 0 else (entry["arm"],):
                ids = self.cumulative[rep, arm]
                ids.extend(entry["new_ids"])
                winner = audit.rank_ids(ids, self.records[str(rep)])[0]
                rows.append(dict(**{**entry, "arm": arm}, evaluated_ids=list(ids), best_id=winner,
                                 best_fitness=self.records[str(rep)][winner]["fitness"]))
        same_tree(snapshot["history"], rows, "generation selection and paired candidate budgets")
        self.history.extend(rows)
        # The cumulative top-eight threshold can only improve. An excluded
        # genome can never be selected as a parent in a later generation.
        for rep, genomes in self.genomes.items():
            top = audit.rank_ids(self.cumulative[rep, "evolution"],
                                 self.records[str(rep)])[:self.config["parents"]]
            self.genomes[rep] = {cid: genomes[cid] for cid in top}

    def winners(self, budget):
        winners = []
        for rep in self.config["source_seeds"]:
            for arm in ARMS:
                ids = self.cumulative[rep, arm]
                require(len(ids) == budget and len(set(ids)) == budget, "unequal or repeated candidate budget")
                winners.append(dict(rep=rep, arm=arm, id=audit.rank_ids(ids, self.records[str(rep)])[0]))
        return winners


def audit_search(root, protocol, config, executors, metadata_only):
    root = Path(root)
    search_path = root / "search.json"
    ledger = SearchLedger(root, config, executors)
    sealed = committed_generations(root, config["generations"])
    if not metadata_only:
        require(sealed == config["generations"] and search_path.is_file(), "expanded search is incomplete")

    for generation in range(sealed):
        path = root / f"generation_{generation}.json"
        snapshot = read(path)
        plan_path = audit.checked_file(root, f"plans/generation_{generation:02d}.json", snapshot["plan_sha256"])
        plan = ledger.verify_plan(generation, plan_path)
        for descriptor in plan["candidates"]:
            ledger.candidate(descriptor, required=True)
        ledger.complete_generation(plan, snapshot)

    active_count = 0
    active_path = root / "plans" / f"generation_{sealed:02d}.json"
    if sealed < config["generations"] and active_path.is_file():
        plan = ledger.verify_plan(sealed, active_path)
        for descriptor in plan["candidates"]:
            active_count += ledger.candidate(descriptor, required=False)

    count = sum(map(len, ledger.records.values()))
    search_complete = sealed == config["generations"] and search_path.is_file()
    winners = None
    if search_complete:
        require(count == protocol["unique_candidates"], "expanded search candidate count differs")
        winners = ledger.winners(protocol["budget_per_arm_per_repeat"])
        expected = dict(history=ledger.history, records=ledger.records, winners=winners,
                        unique_candidates=count)
        same_tree(read(search_path), expected, "expanded search freeze")
        for rep, records in ledger.records.items():
            folders = {path.name for path in (root / "search" / f"rep_{rep}").iterdir() if path.is_dir()}
            require(folders == set(records), "candidate folders differ from frozen search")

    return dict(sealed_generations=sealed, active_generation_candidates=active_count,
                search_complete=search_complete, candidates_provenance_checked=count,
                committed_budget_by_repeat={
                    str(rep): {arm: len(ledger.cumulative[rep, arm]) for arm in ARMS}
                    for rep in config["source_seeds"]},
                winners=winners)


def audit_all(root=OUT, workers=9, *, metadata_only=False):
    """Return a progress audit or the final report after all independent replays."""
    root = Path(root).resolve()
    torch.set_num_threads(1)
    protocol_path = root / "protocol.json"
    protocol = read(protocol_path)
    config = validate_protocol(protocol)
    source_names = {path.replace("\\", "/") for path in protocol["source_hashes"]}
    require({"experiments/seed_evolution/" + name for name in
             ("core.py", "runner.py", "large_runner.py", "large-plan.md")}
            <= source_names, "expanded protocol does not bind its execution code and plan")
    sources = audit.verify_sources(protocol)
    if not metadata_only:
        freeze = read(root / "final-freeze.json")
        require(freeze["all_final_training_complete_before_test"] is True, "final training barrier absent")
        completion = read(root / "test-completion.json")
    executors = {rep: common.executor(rep) for rep in config["source_seeds"]}
    progress = audit_search(root, protocol, config, executors, metadata_only)
    report = dict(**progress, mode="metadata_only" if metadata_only else "full",
                  full_audit_passed=False, protocol_sha256=common.sha(protocol_path),
                  audit_sha256=common.sha(__file__), audit_helper_sha256=common.sha(audit.__file__),
                  fixed_ce_streams=len(sources), target_generations=config["generations"],
                  target_unique_candidates=protocol["unique_candidates"],
                  target_budget_per_arm_per_repeat=protocol["budget_per_arm_per_repeat"])
    if metadata_only:
        return report

    winners = progress["winners"]
    same_tree([row["winner"] for row in freeze["models"]], winners, "final trained winners")
    same_tree([row["winner"] for row in completion["models"]], winners, "held-out evaluated winners")
    identities = [identity for rep in config["source_seeds"] for identity in audit.world_identities(
        rep, [*config["dev_streams"], *config["test_streams"]], config["random_seed"], config["actors"])]
    require(len({identity["world_seed"] for identity in identities}) == len(identities),
            "development and held-out worlds overlap")
    require(len({actor for identity in identities for actor in identity["actor_ids"]})
            == len(identities) * config["actors"], "development and held-out actors overlap")
    expected_files = {f"{step:03d}_{condition}{suffix}"
                      for step, condition in audit.CONDITIONS for suffix in (".json", ".pt")}
    expected_files.add("completion.json")
    for winner in winners:
        folder = root / "final" / f"rep_{winner['rep']}" / winner["arm"] / "evaluations"
        require(folder.is_dir() and {path.name for path in folder.iterdir()
                                    if not path.name.startswith(".")} == expected_files,
                "held-out report inventory differs")

    sample = {(row["rep"], row["id"]) for row in protocol["audit_candidates"]}
    selected = {(row["rep"], row["id"]) for row in winners}
    candidate_jobs = [(root, rep, cid, sources, config, (rep, cid) in sample)
                      for rep, cid in sorted(sample | selected)]
    final_jobs = [(root, row, result, config, sources)
                  for row, result in zip(freeze["models"], completion["models"])]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        candidates = list(pool.map(audit.replay_candidate, candidate_jobs))
        finals = list(pool.map(audit.audit_final, final_jobs))

    shared = []
    for left, right in zip(finals[::2], finals[1::2]):
        if left["winner"]["id"] == right["winner"]["id"]:
            same_tree(left["model_hashes"], right["model_hashes"], "shared winner trained models")
            same_tree(left["fitness"], right["fitness"], "shared winner held-out scores")
            shared.append(left["winner"]["rep"])
    ledger_count = sum(row["heldout_ledgers_recounted"] for row in finals)
    replay_count = sum(row["heldout_policy_replays"] for row in finals)
    require(ledger_count == 72 and replay_count == 9, "incomplete held-out ledger or policy replay count")
    require(sum(row["independent_ce_updates"] for row in candidates) == 9 * config["inner_steps"],
            "incomplete independent development training replay count")
    require(sum(len(row["replayed_ce_updates"]) for row in finals) == 12,
            "incomplete independent final training replay count")
    report.update(full_audit_passed=True, development_replays=candidates, final_checks=finals,
                  shared_winner_repeats=shared, heldout_ledgers_recounted=ledger_count,
                  heldout_policy_replays=replay_count, heldout_reports=36,
                  search_sha256=common.sha(root / "search.json"),
                  final_freeze_sha256=common.sha(root / "final-freeze.json"),
                  test_completion_sha256=common.sha(root / "test-completion.json"))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    report = audit_all(args.root, args.workers, metadata_only=args.metadata_only)
    if args.metadata_only:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    handle = NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.root,
                                prefix=".audit-", suffix=".tmp", delete=False)
    pending = Path(handle.name)
    try:
        with handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        pending.replace(args.root / "audit.json")
    finally:
        pending.unlink(missing_ok=True)
    print(f"Audited {report['candidates_provenance_checked']} candidates and "
          f"{report['heldout_ledgers_recounted']} held-out ledgers.")


if __name__ == "__main__":
    main()
