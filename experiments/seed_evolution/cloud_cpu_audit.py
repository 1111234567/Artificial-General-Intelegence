"""Audit a Linux CPU continuation of the sealed Windows seed search."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import math
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.learned_binding import audit as binding_audit, common
from experiments.seed_evolution import audit, execution_audit, large_audit


require = audit.require
same_tree = audit.same_tree
read = binding_audit.read_json
load = binding_audit.load_pt
MIGRATION_FORMAT = "seed-evolution-debian-cpu-migration-v1"
ORIGINAL_NAMES = ("protocol.json", "execution-amendment.json", "data-freeze.json")
_ARCHIVE_NOTE = (
    "Windows candidates retain their original initialization, checkpoints and fitness. "
    "Their frozen files, lineage, budget, training provenance and recorded score ledgers "
    "were checked. Their initialization, training and closed-loop development scores "
    "were not regenerated on Linux."
)


def _verify_migration(root, protocol):
    migration = read(root / "migration.json")
    require(migration["format"] == MIGRATION_FORMAT, "unknown CPU migration format")
    original_names = set(migration["original"])
    require(set(ORIGINAL_NAMES) <= original_names <= {*ORIGINAL_NAMES, "progress.json"},
            "original migration file inventory differs")
    originals = {name: audit.checked_file(root, f"migration/original/{name}", digest)
                 for name, digest in migration["original"].items()}
    require(common.sha(root / "protocol.json") == migration["portable_protocol_sha256"],
            "portable protocol differs from migration")
    original = read(originals["protocol.json"])
    path_fields = {"source_hashes", "origins", "data_freeze_ref"}
    same_tree({key: value for key, value in protocol.items() if key not in path_fields},
              {key: value for key, value in original.items() if key not in path_fields},
              "portable protocol scientific fields")
    for name, digest in original["source_hashes"].items():
        require(protocol["source_hashes"].get(name.replace("\\", "/")) == digest,
                "original protocol source differs: " + name)
    same_tree(protocol["origins"],
              {seed: dict(row, path=row["path"].replace("\\", "/"))
               for seed, row in original["origins"].items()}, "portable executor origins")
    original_freeze_ref, portable_freeze_ref = original["data_freeze_ref"], protocol["data_freeze_ref"]
    require(original_freeze_ref["sha256"] == migration["original"]["data-freeze.json"]
            and portable_freeze_ref["path"] == original_freeze_ref["path"].replace("\\", "/"),
            "frozen training manifest reference differs")
    portable_freeze = read(audit.checked_file(audit.ROOT, portable_freeze_ref["path"],
                                              portable_freeze_ref["sha256"]))
    original_freeze = read(originals["data-freeze.json"])
    normalized_streams = [dict(row, path=row["path"].replace("\\", "/"))
                          for row in original_freeze["streams"]]
    same_tree(portable_freeze, dict(original_freeze, streams=normalized_streams),
              "portable frozen training streams")

    amendment = read(originals["execution-amendment.json"])
    require(amendment["format"] == "seed-evolution-execution-v1"
            and amendment["original_protocol_sha256"] == migration["original"]["protocol.json"],
            "original execution amendment does not bind the original protocol")
    require(set(amendment["backend_metadata"]) == {"cpu", "hybrid"},
            "original execution backend inventory differs")
    same_tree(amendment["hybrid_indices"], list(range(8)), "original paired hybrid indices")
    for name, digest in amendment["sources"].items():
        audit.checked_file(audit.ROOT, name.replace("\\", "/"), digest)
    start, cutover = amendment["start_generation"], migration["cutover_generation"]
    require(type(start) is int and type(cutover) is int and 1 <= start <= cutover <= 32,
            "invalid migration cutover")
    execution = execution_audit.Execution(start, tuple(amendment["hybrid_indices"]),
                                          amendment["backend_metadata"],
                                          migration["original"]["execution-amendment.json"])
    execution_audit._verify_prefix(root, amendment, execution)
    generations, plans = migration["historical_generations"], migration["historical_plans"]
    expected_plans = {f"plans/generation_{generation:02d}.json" for generation in range(cutover)}
    require(set(generations) == {f"generation_{generation}.json" for generation in range(cutover)},
            "archived generation commit inventory differs")
    allowed_plans = expected_plans | {f"plans/generation_{cutover:02d}.json"}
    require(expected_plans <= set(plans) <= allowed_plans, "archived plan inventory differs")
    for name, digest in (*generations.items(), *plans.items()):
        audit.checked_file(root, name, digest)
    require(migration["historical_count"] == len(migration["historical_candidates"]),
            "archived candidate inventory count differs")
    return migration, execution


def _audit_archived_candidate(root, record, config, source_executor, genomes, *, execution):
    """Verify the sealed inputs and original score without cross-platform regeneration."""
    rep, cid = record["rep"], record["id"]
    folder = audit.candidate_directory(root, rep, cid)
    require(record["arm"] != "evolution" or record["parent_id"] in genomes,
            "archived mutation parent is unavailable")
    require(set(record["artifacts"]) == audit.ARTIFACTS, "archived candidate artifact bindings differ")
    files = {name: audit.checked_file(folder, name, digest)
             for name, digest in record["artifacts"].items()}
    genome = load(files["initial.pt"])
    require(audit.genome_hash(genome) == record["genome_hash"], "archived genome hash differs")
    checkpoint = load(files["inner64.pt"])
    executor = source_executor.state_dict()
    audit.audit_frozen_checkpoint(checkpoint, rep, config["inner_steps"], genome, executor)
    trained_hash = binding_audit.parameter_hash(checkpoint["model"])
    require(trained_hash == record["trained_hash"], "archived trained model hash differs")
    expected_execution = execution.expected_metadata(execution.backend(cid))
    execution_audit.audit_training_report(read(files["training.json"]), rep, config["inner_steps"],
                                          genome, executor, expected_execution=expected_execution)
    score = read(files["score.json"])
    audit.verify_scoring(score, rep, config["dev_streams"], trained_hash, config)
    require(score["fitness"] == record["fitness"], "archived candidate fitness differs")
    genomes[cid] = genome
    return [{key: stream[key] for key in audit.EXOGENOUS_FIELDS} for stream in score["streams"]]


def _audit_final_training(root, row, config, stream_manifest):
    winner = row["winner"]
    rep, arm, cid = (winner[key] for key in ("rep", "arm", "id"))
    folder = root / "final" / f"rep_{rep}" / arm
    checkpoint_path = audit.checked_file(audit.ROOT, row["path"], row["sha256"])
    require(checkpoint_path == (folder / "training.pt").resolve(), "unexpected final training path")
    trained = load(checkpoint_path)
    same_tree(read(folder / "training.json"), dict(winner=winner, sha256=row["sha256"],
                                                  metadata=trained["metadata"], log=trained["log"]),
              "final training record")
    final_freeze_time = (root / "final-freeze.json").stat().st_mtime_ns
    search_freeze_time = (root / "search.json").stat().st_mtime_ns
    require(search_freeze_time <= checkpoint_path.stat().st_mtime_ns <= final_freeze_time,
            "final training order differs from search and test barriers")
    candidate = audit.candidate_directory(root, rep, cid)
    artifacts = read(candidate / "candidate.json")["artifacts"]
    genome = load(audit.checked_file(candidate, "initial.pt", artifacts["initial.pt"]))
    origin = audit.checked_file(candidate, "inner64.pt", artifacts["inner64.pt"])
    training = audit.checked_file(candidate, "training.json", artifacts["training.json"])
    origin_metadata = read(training)["metadata"]
    executor = common.executor(rep).state_dict()
    encoder = {name.removeprefix("encoder."): value for name, value in genome.items()
               if name.startswith("encoder.")}
    proxy, final = config["inner_steps"], config["final_steps"]
    expected_metadata = dict(
        source_seed=rep, steps=final, device="cpu", genome_hash=audit.genome_hash(genome),
        executor_hash=binding_audit.parameter_hash(executor),
        encoder_hash=binding_audit.parameter_hash(encoder),
        stream_indices=[audit.historical_stream_at(step) for step in range(proxy, final)],
        initial_state=dict(h=0, z=0, memory=0), reset_every_stream=True,
        optimizer="Adam", fresh_optimizer=False, learning_rate=audit.HISTORICAL_CE["ce_lr"],
        wait_weight=audit.HISTORICAL_CE["wait_weight"], grad_clip_norm=5.,
        trainable_parameters=2258, encoder_unchanged=True, executor_unchanged=True,
        tf32=False, origin64_sha256=artifacts["inner64.pt"],
        origin64_training_sha256=artifacts["training.json"],
        origin64_device=origin_metadata["device"], origin64_execution=origin_metadata.get("execution"),
        continuation_start_update=proxy + 1, continuation_updates=final - proxy,
        torch_version=str(torch.__version__), dtype="float32", amp=False,
    )
    same_tree(trained["metadata"], expected_metadata, "CPU continuation metadata and origin")
    checkpoints = trained["checkpoints"]
    require(set(checkpoints) == {0, proxy, proxy + 1, final - 1, final},
            "CPU continuation replay checkpoints differ")
    for step, checkpoint in checkpoints.items():
        audit.audit_frozen_checkpoint(checkpoint, rep, step, genome, executor)
    same_tree(checkpoints[proxy], load(origin), "CPU continuation resumes the exact archived model and Adam")
    log = trained["log"]
    logged_steps = sorted({proxy + 1, final - 1, final} | set(range(proxy + 64, final + 1, 64)))
    same_tree([entry["update"] for entry in log], logged_steps, "CPU continuation log coverage")
    for entry in log:
        require(entry["stream"] == audit.historical_stream_at(entry["update"] - 1)
                and math.isfinite(entry["loss"]) and math.isfinite(entry["grad_norm"]),
                "CPU continuation log stream or metrics differ")
    replay_steps = (proxy, final - 1)
    streams = audit.load_source_streams(stream_manifest, rep,
                                        [audit.historical_stream_at(step) for step in replay_steps])
    replayed = []
    for step in replay_steps:
        metrics = audit.replay_checkpoint_update(checkpoints[step], checkpoints[step + 1],
                                                 streams[audit.historical_stream_at(step)])
        logged = next(entry for entry in log if entry["update"] == step + 1)
        for name in ("loss", "grad_norm"):
            require(metrics[name] == logged[name], "CPU continuation replay metric differs: " + name)
        replayed.append(metrics)
    return checkpoints, replayed, final_freeze_time


def _audit_final(job):
    root, row, completed, config, stream_manifest = job
    torch.set_num_threads(1)
    winner = row["winner"]
    rep, arm = winner["rep"], winner["arm"]
    checkpoints, replayed, final_freeze_time = _audit_final_training(root, row, config, stream_manifest)
    folder = root / "final" / f"rep_{rep}" / arm / "evaluations"
    same_tree(read(folder / "completion.json"), completed, "held-out completion record")
    names = [f"{step:03d}_{condition}" for step, condition in audit.CONDITIONS]
    same_tree([item["name"] for item in completed["conditions"]], names, "held-out condition inventory")
    expected_files = {name + suffix for name in names for suffix in (".json", ".pt")} | {"completion.json"}
    require({path.name for path in folder.iterdir() if not path.name.startswith(".")} == expected_files,
            "held-out file inventory differs")
    recounts, policy_replays = 0, 0
    model_hashes = {}
    for (step, condition), item in zip(audit.CONDITIONS, completed["conditions"], strict=True):
        score_path = audit.checked_file(folder, item["name"] + ".json", item["sha256"])
        score = read(score_path)
        archive_path = audit.checked_file(folder, item["name"] + ".pt", score["archives_sha256"])
        require(min(score_path.stat().st_mtime_ns, archive_path.stat().st_mtime_ns) >= final_freeze_time,
                "held-out artifacts predate final training freeze")
        require(score["winner"] == winner and score["step"] == step and score["condition"] == condition,
                "held-out artifact identity differs")
        model_hash = binding_audit.parameter_hash(checkpoints[step]["model"])
        model_hashes[step] = model_hash
        audit.verify_scoring(score, rep, config["test_streams"], model_hash, config, condition)
        require(score["fitness"] == item["fitness"], "held-out summary fitness differs")
        archives = load(archive_path)
        selected = step == config["final_steps"] and (condition == "own" or (rep == 0 and arm == "evolution"))
        policy_replays += audit.audit_evaluation_archive(score, archives, config,
                                                        config["test_streams"][:1] if selected else [])
        recounts += len(archives)
    return dict(winner=winner, replayed_ce_updates=replayed, model_hashes=model_hashes,
                heldout_ledgers_recounted=recounts, heldout_policy_replays=policy_replays,
                fitness={item["name"]: item["fitness"] for item in completed["conditions"]})


def audit_all(root=large_audit.OUT, workers=6, *, metadata_only=False):
    """Reconstruct the search ledger and replay the Linux portion on its actual backend."""
    root = Path(root).resolve()
    torch.set_num_threads(1)
    protocol = read(root / "protocol.json")
    config = large_audit.validate_protocol(protocol)
    migration, execution = _verify_migration(root, protocol)
    streams = audit.verify_sources(protocol)
    executors = {rep: common.executor(rep) for rep in config["source_seeds"]}
    frozen = migration["historical_candidates"]
    checked, historical = set(), set()
    counts = Counter()
    original_candidate = audit.audit_candidate

    def verify_candidate(run, record, settings, source, genomes):
        rep, cid = record["rep"], record["id"]
        name = f"search/rep_{rep}/{cid}/candidate.json"
        if name not in frozen:
            require(record["generation"] >= migration["cutover_generation"],
                    "unarchived candidate in a historical generation")
            return original_candidate(run, record, settings, source, genomes)
        require(record["generation"] <= migration["cutover_generation"], "archived candidate past cutover")
        plan = f"plans/generation_{record['generation']:02d}.json"
        require(plan in migration["historical_plans"], "archived candidate lacks an archived generation plan")
        audit.checked_file(run, name, frozen[name])
        checked.add(name)
        historical.add((rep, cid))
        counts[execution.backend(cid)] += 1
        return _audit_archived_candidate(run, record, settings, source, genomes, execution=execution)

    with patch.object(audit, "audit_candidate", verify_candidate):
        progress = large_audit.audit_search(root, protocol, config, executors, metadata_only)
    require(checked == set(frozen), "migration candidate archive coverage differs")
    report = dict(**progress, mode="metadata_only" if metadata_only else "full", full_audit_passed=False,
                  protocol_sha256=common.sha(root / "protocol.json"), audit_sha256=common.sha(__file__),
                  audit_helper_sha256=common.sha(audit.__file__), migration_sha256=common.sha(root / "migration.json"),
                  original_execution_amendment_sha256=execution.sha256, fixed_ce_streams=len(streams),
                  target_generations=config["generations"], target_unique_candidates=protocol["unique_candidates"],
                  target_budget_per_arm_per_repeat=protocol["budget_per_arm_per_repeat"],
                  historical_candidates_provenance_checked=len(checked),
                  linux_cpu_candidates_provenance_checked=progress["candidates_provenance_checked"] - len(checked),
                  historical_backend_counts={backend: counts[backend] for backend in ("legacy_cpu", "cpu", "hybrid")},
                  historical_initializations_replayed=0, historical_training_replays=0,
                  historical_development_replays=0, migration_disclosure=_ARCHIVE_NOTE)
    if metadata_only:
        return report

    freeze, completion = read(root / "final-freeze.json"), read(root / "test-completion.json")
    require(freeze["all_final_training_complete_before_test"] is True, "final training barrier absent")
    winners = progress["winners"]
    same_tree([row["winner"] for row in freeze["models"]], winners, "final trained winners")
    same_tree([row["winner"] for row in completion["models"]], winners, "held-out evaluated winners")
    identities = [identity for rep in config["source_seeds"] for identity in audit.world_identities(
        rep, [*config["dev_streams"], *config["test_streams"]], config["random_seed"], config["actors"])]
    require(len({row["world_seed"] for row in identities}) == len(identities),
            "development and held-out worlds overlap")
    require(len({actor for row in identities for actor in row["actor_ids"]}) == len(identities) * config["actors"],
            "development and held-out actors overlap")
    final_generation = config["generations"] - 1
    sample = {(rep, audit.candidate_id(arm, final_generation, 0))
              for rep in config["source_seeds"] for arm in audit.ARMS}
    require(not sample & historical, "last-generation CPU replay sample includes archived candidates")
    selected = {(row["rep"], row["id"]) for row in winners}
    candidate_jobs = [(root, rep, cid, streams, config, True)
                      for rep, cid in sorted((sample | selected) - historical)]
    final_jobs = [(root, row, result, config, streams)
                  for row, result in zip(freeze["models"], completion["models"], strict=True)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        candidates = list(pool.map(audit.replay_candidate, candidate_jobs))
        finals = list(pool.map(_audit_final, final_jobs))

    shared = []
    for left, right in zip(finals[::2], finals[1::2], strict=True):
        if left["winner"]["id"] == right["winner"]["id"]:
            same_tree(left["model_hashes"], right["model_hashes"], "shared winner trained models")
            same_tree(left["fitness"], right["fitness"], "shared winner held-out scores")
            shared.append(left["winner"]["rep"])
    ledger_count = sum(row["heldout_ledgers_recounted"] for row in finals)
    replay_count = sum(row["heldout_policy_replays"] for row in finals)
    require(ledger_count == 72 and replay_count == 9, "incomplete held-out ledger or policy replay count")
    require(len(candidates) == len(candidate_jobs)
            and all(row["independent_ce_updates"] == config["inner_steps"] for row in candidates),
            "incomplete Linux development training replay coverage")
    require(sum(len(row["replayed_ce_updates"]) for row in finals) == 12,
            "incomplete Linux final continuation replay coverage")
    report.update(full_audit_passed=True, development_replays=candidates, final_checks=finals,
                  historical_winners=[row for row in winners if (row["rep"], row["id"]) in historical],
                  continuation_disclosure="Each finalist resumes its stored model and Adam state after "
                  "update 64. Updates 65 and 512 and the selected held-out trajectories were replayed on Linux.",
                  shared_winner_repeats=shared, heldout_ledgers_recounted=ledger_count,
                  heldout_policy_replays=replay_count, heldout_reports=36,
                  search_sha256=common.sha(root / "search.json"), final_freeze_sha256=common.sha(root / "final-freeze.json"),
                  test_completion_sha256=common.sha(root / "test-completion.json"))
    return report
