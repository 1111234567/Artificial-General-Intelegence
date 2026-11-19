"""Freeze scale-search winners, retrain them, and score independent worlds."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
import math
import multiprocessing as mp
from pathlib import Path
from statistics import mean

import torch

from experiments.learned_binding import common
from experiments.learned_binding.audit import parameter_hash
from experiments.seed_evolution import audit, core, gpu_batch_train, scale_search_plan as plan
from experiments.seed_evolution.large_runner import (
    _commit_pair, _load, _lock, _recover_pair, _verify_hash, _write,
)
from experiments.seed_evolution.runner import read


STEPS = (0, plan.INNER_STEPS, plan.FINAL_STEPS)
EXECUTOR_CHUNK = len(plan.ARMS)
TRIAL_ARTIFACTS = ("initial.pt", "inner64.pt", "training.json", "score.json")


def _seal(path, payload):
    if path.is_file():
        if read(path) != payload:
            raise ValueError(f"sealed record differs: {path}")
    else:
        _write(path, payload)
    return common.sha(path)


def _source_dir(root, source):
    return root / "final" / f"source_{source}"


def _report_path(root, winner):
    return _source_dir(root, winner["source_seed"]) / winner["arm"] / "report.json"


def _check_score(score, source, model_hash, split):
    options = plan.scoring_options(split)
    worlds = score["streams"]
    if (score["source_seed"] != source or score["parameter_hash"] != model_hash
            or tuple(world["stream_id"] for world in worlds) != options["stream_ids"]
            or any(score[key] != options[key] for key in ("support_ticks", "probe_ticks", "actors", "random_seed"))
            or score["read_enabled"] is not True or score["write_enabled"] is not True
            or score["teacher_support"] is not False or score["b_ticks"] != 0
            or not math.isclose(score["fitness"], audit.conflict_fitness(worlds), rel_tol=0, abs_tol=1e-12)):
        raise ValueError(f"{split} evaluation does not match its model or scoring protocol")


def _worlds(score):
    fields = ("stream_id", "actor_ids", "world_start_hash", "private_mapping_hash")
    return [{key: world[key] for key in fields} for world in score["streams"]]


def _select(root, winners):
    expected = [(seed, arm) for seed in plan.SOURCE_SEEDS for arm in plan.ARMS]
    selected = {(winner["source_seed"], winner["arm"]): winner for winner in winners}
    if len(winners) != len(expected) or set(selected) != set(expected):
        raise ValueError("one completed winner is required for each source and search arm")

    frozen = []
    for seed, arm in expected:
        winner = selected[seed, arm]
        trial = (root / winner["trial_dir"]).resolve()
        relative = trial.relative_to(root).as_posix()
        record_path = trial / "candidate.json"
        record = read(record_path)
        artifacts = {name: record["artifacts"][name] for name in TRIAL_ARTIFACTS}
        for name, digest in artifacts.items():
            _verify_hash(trial / name, digest)
        if (record["source_seed"] != seed or record["arm"] not in ("shared", arm)
                or record["id"] != winner["id"] or record["fitness"] != winner["fitness"]
                or read(trial / "training.json")["checkpoint_sha256"] != artifacts["inner64.pt"]):
            raise ValueError(f"winner differs from its completed trial: {trial}")

        genome = _load(trial / "initial.pt")
        genome_hash = core.genome_hash(genome)
        if genome_hash != record["genome_hash"]:
            raise ValueError(f"winner genome differs from the trial record: {trial}")
        proxy = _load(trial / "inner64.pt")
        if proxy["source_seed"] != seed or proxy["updates"] != plan.INNER_STEPS:
            raise ValueError(f"winner proxy uses a different source or step count: {trial}")
        proxy_hash = parameter_hash(proxy["model"])
        score = read(trial / "score.json")
        _check_score(score, seed, proxy_hash, "dev")
        if score["fitness"] != record["fitness"] or proxy_hash != record["model_hash"]:
            raise ValueError(f"winner fitness or model hash differs from its development report: {trial}")

        frozen.append(dict(source_seed=seed, arm=arm, trial_dir=relative, id=record["id"],
                           fitness=score["fitness"], candidate_sha256=common.sha(record_path),
                           artifacts=artifacts, genome_hash=genome_hash,
                           proxy_model_sha256=proxy_hash, development_worlds=_worlds(score)))
    return frozen


def _train_source(root, winners, search_freeze_sha256, reuse_only):
    source = winners[0]["source_seed"]
    folder = _source_dir(root, source)
    checkpoint_path, metadata_path = folder / "training.pt", folder / "training.json"
    metadata = _recover_pair(checkpoint_path, metadata_path, "checkpoint_sha256")
    if metadata is None:
        if reuse_only:
            raise ValueError(f"frozen final package is missing: {checkpoint_path}")
        genomes = [_load(root / winner["trial_dir"] / "initial.pt") for winner in winners]
        trained = gpu_batch_train.train_batch(
            source, genomes, plan.FINAL_STEPS, device="cuda:0", dtype=torch.bfloat16,
            executor_chunk=EXECUTOR_CHUNK, snapshot_steps=STEPS,
        )
        training = trained["metadata"]
        expected = dict(source_seed=source, device="cuda:0", dtype="float32", executor_dtype="bfloat16",
                        executor_chunk=EXECUTOR_CHUNK, candidates=len(plan.ARMS), steps=plan.FINAL_STEPS,
                        stream_indices=[core._stream_index(i) for i in range(plan.FINAL_STEPS)], tf32=False)
        if any(training[key] != value for key, value in expected.items()):
            raise ValueError(f"final training metadata differs from the protocol for source {source}")
        if (set(trained["snapshots"]) != set(STEPS)
                or any(len(checkpoints) != len(winners) for checkpoints in trained["snapshots"].values())):
            raise ValueError(f"missing final checkpoints for source {source}")

        executor = common.executor(source).state_dict()
        models, hashes = [], []
        for slot, (winner, genome) in enumerate(zip(winners, genomes, strict=True)):
            if core.genome_hash(genome) != winner["genome_hash"]:
                raise ValueError(f"final training initialization differs from its freeze: {winner['id']}")
            snapshots = {step: trained["snapshots"][step][slot] for step in STEPS}
            for step, snapshot in snapshots.items():
                audit.audit_frozen_checkpoint(snapshot, source, step, genome, executor)
            losses = trained["losses"][slot]
            grad_norms = trained["grad_norms"][slot]
            if (len(losses) != plan.FINAL_STEPS or len(grad_norms) != plan.FINAL_STEPS
                    or not all(math.isfinite(x) for x in (*losses, *grad_norms))):
                raise ValueError(f"incomplete final training log for {winner['id']}")
            models.append(dict(winner=winner, checkpoints=snapshots, losses=losses, grad_norms=grad_norms))
            hashes.append({str(step): parameter_hash(snapshot["model"]) for step, snapshot in snapshots.items()})

        metadata = _commit_pair(
            checkpoint_path, metadata_path, dict(models=models),
            dict(winners=winners, search_freeze_sha256=search_freeze_sha256, metadata=training,
                 timing=trained["timing"], model_hashes=hashes), "checkpoint_sha256",
        )
    if (metadata["winners"] != winners or metadata["search_freeze_sha256"] != search_freeze_sha256
            or len(metadata["model_hashes"]) != len(winners)):
        raise ValueError(f"final training package differs from the selected winners: {metadata_path}")
    metadata_sha256 = common.sha(metadata_path)
    return [dict(winner=winner, batch_lane=slot, checkpoint_path=checkpoint_path.relative_to(root).as_posix(),
                 checkpoint_sha256=metadata["checkpoint_sha256"], training_metadata_sha256=metadata_sha256,
                 model_sha256=hashes, proxy64_model_matches_retrain64=(
                     winner["proxy_model_sha256"] == hashes[str(plan.INNER_STEPS)]))
            for slot, (winner, hashes) in enumerate(zip(winners, metadata["model_hashes"], strict=True))]


def _score_final(root, row, final_freeze_sha256):
    winner = row["winner"]
    checkpoint_path = root / row["checkpoint_path"]
    _verify_hash(checkpoint_path, row["checkpoint_sha256"])
    model = _load(checkpoint_path)["models"][row["batch_lane"]]
    if model["winner"] != winner:
        raise ValueError(f"checkpoint lane differs from the frozen winner: {checkpoint_path}")

    results = []
    for step in STEPS:
        state = model["checkpoints"][step]["model"]
        model_hash = parameter_hash(state)
        if model_hash != row["model_sha256"][str(step)]:
            raise ValueError(f"frozen model differs for {winner['id']} at step {step}")
        score = core.score_model(winner["source_seed"], state, **plan.scoring_options("test"))
        _check_score(score, winner["source_seed"], model_hash, "test")
        results.append(dict(step=step, score=score))
    payload = dict(model=row, final_freeze_sha256=final_freeze_sha256, results=results)
    _write(_report_path(root, winner), payload)


def _read_report(path, row, freeze_hash):
    report = read(path)
    if (report["model"] != row or report["final_freeze_sha256"] != freeze_hash
            or tuple(entry["step"] for entry in report["results"]) != STEPS):
        raise ValueError(f"held-out report differs from the frozen model: {path}")
    winner = row["winner"]
    for result in report["results"]:
        _check_score(result["score"], winner["source_seed"], row["model_sha256"][str(result["step"])], "test")
    return report


def _check_worlds(reports):
    worlds_by_source = {}
    for report in reports:
        winner = report["model"]["winner"]
        source = winner["source_seed"]
        worlds = _worlds(report["results"][0]["score"])
        if worlds_by_source.setdefault(source, worlds) != worlds:
            raise ValueError(f"held-out world identity differs across arms for source {source}")
        if any(_worlds(result["score"]) != worlds for result in report["results"]):
            raise ValueError(f"held-out world identity differs across steps for source {source}")
        dev = winner["development_worlds"]
        if ({actor for world in dev for actor in world["actor_ids"]}
                & {actor for world in worlds for actor in world["actor_ids"]}
                or {world["world_start_hash"] for world in dev} & {world["world_start_hash"] for world in worlds}):
            raise ValueError(f"development and held-out worlds overlap for source {source}")


def _aggregate(reports):
    curves = []
    fitness = {}
    for report in reports:
        winner = report["model"]["winner"]
        source, arm = winner["source_seed"], winner["arm"]
        entries = []
        for result in report["results"]:
            step, score = result["step"], result["score"]
            fitness[source, arm, step] = score["fitness"]
            family_rates = {group: mean(world["groups"][group]["strict"] / world["groups"][group]["n"]
                                        for world in score["streams"])
                            for group in ("A_conflict", "B_conflict")}
            entries.append(dict(step=step, fitness=score["fitness"], strict_conflict_rates=family_rates))
        start, proxy, final = (point["fitness"] for point in entries)
        curves.append(dict(source_seed=source, arm=arm, id=winner["id"], dev_fitness=winner["fitness"],
                           points=entries, gain_64_minus_0=proxy-start, gain_512_minus_64=final-proxy,
                           gain_512_minus_0=final-start))

    comparisons = []
    for left, right in combinations(plan.ARMS, 2):
        for step in STEPS:
            pairs = [dict(source_seed=source, difference=fitness[source, left, step] - fitness[source, right, step])
                     for source in plan.SOURCE_SEEDS]
            comparisons.append(dict(left=left, right=right, step=step, paired_differences=pairs,
                                    mean_paired_difference=mean(pair["difference"] for pair in pairs)))
    return dict(adaptation_curves=curves, comparisons=comparisons, paired_repeats=len(plan.SOURCE_SEEDS))


def finish(root, winners, workers, progress):
    """Finalize 12 committed trials. Report phase progress with ``progress(phase, **fields)``."""
    root = Path(root).resolve()
    workers = core._integer("workers", workers, 1)
    winners = list(winners)
    total_models = len(plan.SOURCE_SEEDS) * len(plan.ARMS)
    search_completion_path = root / "search-completion.json"
    search_freeze_path, final_freeze_path = root / "final-search-freeze.json", root / "final-freeze.json"
    completion_path, results_path = root / "test-completion.json", root / "results.json"

    with _lock(root / ".scale-final.lock"):
        search_completion = read(search_completion_path)
        if search_completion["candidates"] != plan.UNIQUE_CANDIDATES or search_completion["winners"] != winners:
            raise ValueError("finalization requires all winners from a completed search")
        selected = _select(root, winners)
        score_protocol = {**plan.scoring_options("test"), "stream_ids": list(plan.TEST_STREAMS)}
        search_freeze_sha256 = _seal(search_freeze_path, dict(
            winners=selected, protocol_sha256=common.sha(root / "protocol.json"), checkpoint_steps=list(STEPS),
            search_completion_sha256=common.sha(search_completion_path), held_out_score_protocol=score_protocol,
            final_device="cuda:0", executor_dtype="bfloat16", policy_dtype="float32",
            final_batch_width=len(plan.ARMS), final_executor_chunk=EXECUTOR_CHUNK,
            final_implementation_sha256=common.sha(__file__),
        ))
        reuse_only = final_freeze_path.is_file()
        if not reuse_only and (completion_path.exists() or results_path.exists()
                               or any(_report_path(root, winner).exists() for winner in selected)):
            raise ValueError("held-out artifacts exist without a final model freeze")
        models = []
        for source in plan.SOURCE_SEEDS:
            progress("finals", state="running", source_seed=source,
                     completed_models=len(models), total_models=total_models)
            models.extend(_train_source(root, [winner for winner in selected if winner["source_seed"] == source],
                                        search_freeze_sha256, reuse_only))
        final_freeze_sha256 = _seal(final_freeze_path, dict(
            search_freeze_sha256=search_freeze_sha256, models=models,
            all_final_training_complete_before_test=True,
        ))
        progress("finals", state="complete", completed_models=len(models), total_models=total_models)

        paths = [_report_path(root, row["winner"]) for row in models]
        if completion_path.is_file():
            committed = read(completion_path)
            if committed["final_freeze_sha256"] != final_freeze_sha256 or len(committed["reports"]) != total_models:
                raise ValueError("held-out completion differs from the final model freeze")
            for path, entry in zip(paths, committed["reports"], strict=True):
                if entry["path"] != path.relative_to(root).as_posix():
                    raise ValueError(f"unexpected held-out report path: {entry['path']}")
                _verify_hash(path, entry["sha256"])
        pending = [row for row, path in zip(models, paths, strict=True) if not path.is_file()]
        completed = total_models - len(pending)
        progress("test", state="running", completed_models=completed, total_models=total_models)
        if pending:
            with ProcessPoolExecutor(max_workers=min(workers, len(pending)), mp_context=mp.get_context("spawn")) as pool:
                futures = [pool.submit(_score_final, root, row, final_freeze_sha256) for row in pending]
                for future in as_completed(futures):
                    future.result()
                    completed += 1
                    progress("test", state="running", completed_models=completed, total_models=total_models)
        reports = [_read_report(path, row, final_freeze_sha256) for path, row in zip(paths, models, strict=True)]
        _check_worlds(reports)
        report_hashes = [dict(path=path.relative_to(root).as_posix(), sha256=common.sha(path)) for path in paths]
        completion_sha256 = _seal(completion_path, dict(final_freeze_sha256=final_freeze_sha256,
                                                        reports=report_hashes))
        result = dict(final_freeze_sha256=final_freeze_sha256, test_completion_sha256=completion_sha256,
                      test_streams=list(plan.TEST_STREAMS), reports=report_hashes, **_aggregate(reports))
        _seal(results_path, result)
        progress("test", state="complete", completed_models=completed, total_models=total_models)
        return result
