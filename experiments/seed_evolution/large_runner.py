"""Run the sealed 32-generation seed search with recoverable artifact commits."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
import traceback
from uuid import uuid4

from experiments.seed_evolution import runner as base
from experiments.seed_evolution import core
from experiments.learned_binding import common
import torch


ROOT = base.ROOT
OUT = ROOT / "data/seed-evolution/2026-11-11-large-v1"
CONFIG = dict(base.CONFIG, generations=32, population=64, parents=8)
ARMS = ("evolution", "random")
CONDITIONS = ((0, "own"), (64, "own"), (512, "own"), (512, "teacher"),
              (512, "no_read"), (512, "Bdelay"))
CANDIDATE_ARTIFACTS = ("initial.pt", "inner64.pt", "score.json", "training.json")


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _replace(source, target):
    # Windows readers can briefly deny replacement while polling progress.
    for delay in (.01, .02, .04, .08, .16):
        try:
            source.replace(target)
            return
        except PermissionError as error:
            if os.name != "nt" or error.winerror not in (5, 32, 33):
                raise
            time.sleep(delay)
    source.replace(target)


def _atomic(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        write(temporary)
        _replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write(path, value):
    payload = _json_bytes(value)
    _atomic(path, lambda temporary: temporary.write_bytes(payload))


def _save(path, value):
    _atomic(path, lambda temporary: torch.save(value, temporary))


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _verify_hash(path, expected):
    if base.sha(path) != expected:
        raise ValueError(f"artifact hash differs: {path}")


@contextmanager
def _lock(path):
    """An OS lock is released even if the process exits during a write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"another worker holds {path}") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _pending(path):
    return path.with_name(f".{path.name}.pending")


def _recover_pair(data_path, metadata_path, digest_key):
    if metadata_path.is_file():
        metadata = base.read(metadata_path)
        _verify_hash(data_path, metadata[digest_key])
        return metadata
    pending_metadata = _pending(metadata_path)
    if not pending_metadata.is_file():
        if data_path.exists():
            raise ValueError(f"missing artifact manifest: {metadata_path}")
        return None
    metadata = base.read(pending_metadata)
    if not data_path.is_file():
        pending_data = _pending(data_path)
        _verify_hash(pending_data, metadata[digest_key])
        _replace(pending_data, data_path)
    else:
        _verify_hash(data_path, metadata[digest_key])
    _replace(pending_metadata, metadata_path)
    return metadata


def _commit_pair(data_path, metadata_path, value, metadata, digest_key, *, save=_save):
    pending_data, pending_metadata = _pending(data_path), _pending(metadata_path)
    save(pending_data, value)
    metadata = dict(metadata, **{digest_key: base.sha(pending_data)})
    _write(pending_metadata, metadata)
    _replace(pending_data, data_path)
    _replace(pending_metadata, metadata_path)
    return metadata


def _candidate_dir(rep, cid):
    return OUT / "search" / f"rep_{rep}" / cid


def _plan_path(generation):
    return OUT / "plans" / f"generation_{generation:02d}.json"


def _unique_budget():
    return CONFIG["repeats"] * CONFIG["population"] * (2 * CONFIG["generations"] - 1)


def _activate():
    base.OUT, base.CONFIG = OUT, CONFIG


def _check():
    _activate()
    protocol = base.check()
    if (protocol["unique_candidates"] != _unique_budget()
            or protocol["budget_per_arm_per_repeat"] != CONFIG["generations"] * CONFIG["population"]
            or protocol["shared_first_generation"] is not True):
        raise ValueError("the expanded search budget differs from protocol.json")
    return protocol


def declare():
    with _lock(OUT / ".run.lock"):
        protocol_path = OUT / "protocol.json"
        if protocol_path.exists():
            raise FileExistsError(protocol_path)
        sources = [Path(__file__), Path(base.__file__), Path(core.__file__),
                   Path(__file__).with_name("large-plan.md")]
        sources.extend(ROOT / "experiments/learned_binding" / name for name in
                       ("model.py", "read_first.py", "frozen_trace.py", "pretrain.py", "data.py",
                        "interaction.py", "common.py", "audit.py"))
        sources.extend(ROOT / "experiments" / name for name in
                       ("associative_state/bridge.py", "lifelong/world.py", "continuity/world.py",
                        "experience_transfer/train_transfer.py", "learn_remember/gru_train.py"))
        origins = {str(rep): common.source_checkpoint(rep) for rep in CONFIG["source_seeds"]}
        protocol = dict(client_date="2026-11-11", declared_utc=_utc(), config=CONFIG,
                        source_hashes={str(path.relative_to(ROOT)): base.sha(path) for path in sources},
                        origins={rep: dict(path=str(path.relative_to(ROOT)), sha256=base.sha(path))
                                 for rep, path in origins.items()},
                        data_freeze_ref=dict(path=str(core.DATA_FREEZE.relative_to(ROOT)),
                                             sha256=base.sha(core.DATA_FREEZE)),
                        unique_candidates=_unique_budget(),
                        budget_per_arm_per_repeat=CONFIG["population"] * CONFIG["generations"],
                        shared_first_generation=True,
                        audit_candidates=[dict(rep=rep, id=f"{arm}_g{generation:02d}_i00")
                                          for rep in CONFIG["source_seeds"]
                                          for arm, generation in (("shared", 0),
                                              ("evolution", CONFIG["generations"] - 1),
                                              ("random", CONFIG["generations"] - 1))])
        _write(protocol_path, protocol)
        return protocol


def _plan(generation, records, pools):
    entries, candidates = [], []
    for rep in CONFIG["source_seeds"]:
        for arm in ("shared",) if generation == 0 else ARMS:
            prior = pools.get((rep, arm), [])
            top = base.ranked(prior, records[rep])[:CONFIG["parents"]]
            parent_seed = 650000 + rep * 100 + generation
            chooser = random.Random(parent_seed)
            ids = []
            for index in range(CONFIG["population"]):
                cid = f"{arm}_g{generation:02d}_i{index:02d}"
                ids.append(cid)
                parent = chooser.choice(top) if arm == "evolution" else None
                offset = rep * 10000 + generation * 100 + index
                candidates.append(dict(rep=rep, source_seed=rep, arm=arm, generation=generation,
                                       index=index, id=cid, parent_id=parent,
                                       init_seed=None if parent else 710000 + offset,
                                       mutation_seed=810000 + offset if parent else None,
                                       sigma=CONFIG["sigma"], parent_selection_seed=parent_seed if parent else None))
            entries.append(dict(rep=rep, arm=arm, generation=generation, top8_before=top, new_ids=ids))
    return dict(generation=generation, entries=entries, candidates=candidates)


def _committed_candidate(descriptor):
    path = _candidate_dir(descriptor["rep"], descriptor["id"])
    marker = path / "candidate.json"
    if not marker.is_file():
        return None
    record = base.read(marker)
    if (any(record[key] != value for key, value in descriptor.items())
            or set(record["artifacts"]) != set(CANDIDATE_ARTIFACTS)):
        raise ValueError(f"candidate commit differs from its plan: {marker}")
    for name, digest in record["artifacts"].items():
        _verify_hash(path / name, digest)
    return record


def _genome(descriptor):
    rep, parent_id = descriptor["rep"], descriptor["parent_id"]
    if parent_id is None:
        return core.new_genome(rep, descriptor["init_seed"])
    parent_path = _candidate_dir(rep, parent_id)
    parent_record = base.read(parent_path / "candidate.json")
    initial = parent_path / "initial.pt"
    _verify_hash(initial, parent_record["artifacts"]["initial.pt"])
    return core.mutate_genome(_load(initial), descriptor["mutation_seed"], sigma=descriptor["sigma"])


def _score_options(split):
    return dict(stream_ids=tuple(CONFIG[f"{split}_streams"]), support_ticks=CONFIG["support_ticks"],
                probe_ticks=CONFIG["probe_ticks"], actors=CONFIG["actors"], random_seed=CONFIG["random_seed"])


def _candidate_job(descriptor):
    rep, cid = descriptor["rep"], descriptor["id"]
    path = _candidate_dir(rep, cid)
    with _lock(path / ".candidate.lock"):
        completed = _committed_candidate(descriptor)
        if completed is not None:
            return completed
        start = time.perf_counter()
        attempt_path = path / "attempt.json"
        attempt = 1 + (base.read(attempt_path)["attempt"] if attempt_path.is_file() else 0)
        _write(attempt_path, dict(attempt=attempt, started_utc=_utc()))
        genome = _genome(descriptor)
        initial = path / "initial.pt"
        reused = []
        if initial.is_file():
            common.assert_tree_equal(_load(initial), genome, "resumed initialization")
            reused.append("initial.pt")
        else:
            _save(initial, genome)
        genome_digest = core.genome_hash(genome)

        model_path, training_path = path / "inner64.pt", path / "training.json"
        training = _recover_pair(model_path, training_path, "checkpoint_sha256")
        if training is None:
            begin = time.perf_counter()
            steps = CONFIG["inner_steps"]
            trained = core.inner_train(rep, genome, steps=steps, checkpoints=(0, steps),
                                       device=CONFIG["search_device"])
            training = dict(metadata=trained["metadata"], log=trained["log"],
                            train_seconds=time.perf_counter() - begin, train_attempt=attempt)
            checkpoint = trained["checkpoints"][steps]
            training = _commit_pair(model_path, training_path, checkpoint, training, "checkpoint_sha256")
        else:
            checkpoint = _load(model_path)
            reused.append("inner64.pt")
        if (training["metadata"]["genome_hash"] != genome_digest
                or checkpoint["source_seed"] != rep or checkpoint["updates"] != CONFIG["inner_steps"]
                or core.genome_hash(genome) != genome_digest):
            raise ValueError(f"inner checkpoint differs from its initialization or budget: {path}")

        model_digest = base.model_hash(checkpoint["model"])
        score_path, eval_time_path = path / "score.json", path / "evaluation-time.json"
        evaluation = _recover_pair(score_path, eval_time_path, "score_sha256")
        if evaluation is None:
            begin = time.perf_counter()
            score = core.score_model(rep, checkpoint["model"], **_score_options("dev"))
            evaluation = dict(eval_seconds=time.perf_counter() - begin, eval_attempt=attempt)
            evaluation = _commit_pair(score_path, eval_time_path, score, evaluation,
                                      "score_sha256", save=_write)
        else:
            score = base.read(score_path)
            reused.append("score.json")
        if (score["source_seed"] != rep or score["parameter_hash"] != model_digest
                or not math.isfinite(score["fitness"])):
            raise ValueError(f"development score differs from its checkpoint: {path}")
        record = dict(descriptor, genome_hash=genome_digest, trained_hash=model_digest,
                      fitness=score["fitness"], attempt=attempt, elapsed_seconds=time.perf_counter() - start,
                      train_seconds=training["train_seconds"], eval_seconds=evaluation["eval_seconds"],
                      train_attempt=training["train_attempt"], eval_attempt=evaluation["eval_attempt"],
                      reused_artifacts=reused,
                      artifacts={name: base.sha(path / name) for name in CANDIDATE_ARTIFACTS})
        _write(path / "candidate.json", record)
        return record


def _init_worker():
    torch.set_num_threads(CONFIG["threads_per_worker"])
    _check()


def _parallel(jobs, worker, workers):
    """Bound in-flight work so a failed item never releases later jobs."""
    if not jobs:
        return
    remaining = iter(jobs)
    pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker)
    in_flight = {}
    try:
        for item in remaining:
            in_flight[pool.submit(worker, item)] = item
            if len(in_flight) == workers:
                break
        while in_flight:
            finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            batch = []
            for future in finished:
                item = in_flight.pop(future)
                try:
                    batch.append(future.result())
                except Exception as error:
                    raise RuntimeError(f"{worker.__name__} failed for {item}") from error
            yield from batch
            for _ in batch:
                item = next(remaining, None)
                if item is not None:
                    in_flight[pool.submit(worker, item)] = item
    except BaseException:
        if hasattr(pool, "terminate_workers"):
            pool.terminate_workers()
        else:
            pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True, cancel_futures=True)


def _distributions(plan, records):
    summaries = []
    for entry in plan["entries"]:
        rep = entry["rep"]
        values = [records[rep][cid]["fitness"] for cid in entry["new_ids"] if cid in records[rep]]
        row = dict(rep=rep, arm=entry["arm"], generation=plan["generation"], n=len(values),
                   planned=len(entry["new_ids"]))
        if values:
            row.update(min=min(values), max=max(values), mean=statistics.fmean(values),
                       median=statistics.median(values))
        summaries.append(row)
    return summaries


def _progress(phase, state, **fields):
    path = OUT / "progress.json"
    if "generation_distributions" not in fields and path.is_file():
        prior = base.read(path)
        for name in ("generation_distributions", "completed_generations"):
            if name in prior:
                fields.setdefault(name, prior[name])
    _write(path, dict(phase=phase, state=state, updated_utc=_utc(), **fields))


@contextmanager
def _phase(name):
    with _lock(OUT / ".run.lock"):
        try:
            yield _check()
        except BaseException as error:
            failure = dict(phase=name, timestamp_utc=_utc(), message=str(error), traceback=traceback.format_exc())
            _write(OUT / "failure.json", failure)
            prior = OUT / "progress.json"
            progress = base.read(prior) if prior.is_file() else {}
            for key in ("phase", "state", "updated_utc", "failure"):
                progress.pop(key, None)
            if name == "search":
                progress["completed_candidates"] = sum(1 for _ in (OUT / "search").glob("rep_*/*/candidate.json"))
                progress["total_candidates"] = _unique_budget()
            _progress(name, "failed", failure=failure["message"], **progress)
            raise


def _search_progress(generation, records, plan, completed_distributions, *, committed=False):
    current = _distributions(plan, records)
    _progress("search", "running", generation=generation,
              completed_generations=generation + int(committed),
              completed_candidates=sum(map(len, records.values())), total_candidates=_unique_budget(),
              generation_distributions=[*completed_distributions, *current])
    return current


def _seal_generation(plan, records, pools):
    generation = plan["generation"]
    history = []
    for entry in plan["entries"]:
        rep = entry["rep"]
        for arm in ARMS if generation == 0 else (entry["arm"],):
            ids = pools[rep, arm]
            ids.extend(entry["new_ids"])
            best = base.ranked(ids, records[rep])[0]
            history.append(dict(entry, arm=arm, evaluated_ids=list(ids), best_id=best,
                                best_fitness=records[rep][best]["fitness"]))
    return dict(generation=generation, history=history,
                candidate_ids=[dict(rep=row["rep"], id=row["id"]) for row in plan["candidates"]],
                plan_sha256=base.sha(_plan_path(generation)))


def _check_search_completion(summary):
    pairs = {(rep, arm) for rep in CONFIG["source_seeds"] for arm in ARMS}
    if (summary["unique_candidates"] != _unique_budget()
            or sum(map(len, summary["records"].values())) != _unique_budget()
            or len(summary["winners"]) != len(pairs)
            or {(row["rep"], row["arm"]) for row in summary["winners"]} != pairs
            or len(summary["history"]) != len(pairs) * CONFIG["generations"]
            or not all((OUT / f"generation_{g}.json").is_file() for g in range(CONFIG["generations"]))):
        raise ValueError("search.json does not contain the full expanded search")
    return summary


def search(workers=14):
    with _phase("search"):
        result_path = OUT / "search.json"
        if result_path.is_file():
            summary = _check_search_completion(base.read(result_path))
            previous = status()
            if previous.get("phase") == "search" and previous["state"] != "complete":
                _progress("search", "complete", completed_generations=CONFIG["generations"],
                          completed_candidates=_unique_budget(), total_candidates=_unique_budget())
            return summary
        records = {rep: {} for rep in CONFIG["source_seeds"]}
        pools = {(rep, arm): [] for rep in records for arm in ARMS}
        history, distributions = [], []
        for generation in range(CONFIG["generations"]):
            plan = _plan(generation, records, pools)
            plan_path, commit_path = _plan_path(generation), OUT / f"generation_{generation}.json"
            if plan_path.is_file():
                common.assert_tree_equal(base.read(plan_path), plan, "sealed generation plan")
            elif commit_path.exists():
                raise ValueError(f"sealed generation lacks its plan: {commit_path}")
            else:
                _write(plan_path, plan)
            missing = []
            for descriptor in plan["candidates"]:
                record = _committed_candidate(descriptor)
                if record is None:
                    missing.append(descriptor)
                else:
                    records[record["rep"]][record["id"]] = record
            if missing and commit_path.exists():
                raise ValueError(f"sealed generation lacks committed candidates: {commit_path}")
            _search_progress(generation, records, plan, distributions)
            for record in _parallel(missing, _candidate_job, workers):
                records[record["rep"]][record["id"]] = record
                _search_progress(generation, records, plan, distributions)
            committed = _seal_generation(plan, records, pools)
            if commit_path.is_file():
                common.assert_tree_equal(base.read(commit_path), committed, "sealed generation history")
            else:
                _write(commit_path, committed)
                print(json.dumps(dict(event="generation", generation=generation,
                                      completed_candidates=sum(map(len, records.values())))), flush=True)
            history.extend(committed["history"])
            distributions.extend(_search_progress(generation, records, plan, distributions, committed=True))
        winners = [dict(rep=rep, arm=arm, id=base.ranked(pools[rep, arm], records[rep])[0])
                   for rep in records for arm in ARMS]
        summary = dict(history=history, records=records, winners=winners,
                       unique_candidates=sum(map(len, records.values())))
        _check_search_completion(summary)
        _write(result_path, summary)
        _progress("search", "complete", completed_generations=CONFIG["generations"],
                  completed_candidates=_unique_budget(), total_candidates=_unique_budget(),
                  generation_distributions=distributions)
        return summary


def _final_dir(winner):
    return OUT / "final" / f"rep_{winner['rep']}" / winner["arm"]


def _train_final(winner):
    path = _final_dir(winner)
    with _lock(path / ".final.lock"):
        checkpoint_path, metadata_path = path / "training.pt", path / "training.json"
        metadata = _recover_pair(checkpoint_path, metadata_path, "sha256")
        if metadata is None:
            candidate = _candidate_dir(winner["rep"], winner["id"])
            record = base.read(candidate / "candidate.json")
            for name in ("initial.pt", "inner64.pt"):
                _verify_hash(candidate / name, record["artifacts"][name])
            genome = _load(candidate / "initial.pt")
            steps, proxy = CONFIG["final_steps"], CONFIG["inner_steps"]
            trained = core.inner_train(winner["rep"], genome, steps=steps,
                                       checkpoints=(0, 1, proxy, steps - 1, steps), device=CONFIG["final_device"])
            common.assert_tree_equal(_load(candidate / "inner64.pt"), trained["checkpoints"][proxy], "search/final64")
            metadata = _commit_pair(checkpoint_path, metadata_path, trained,
                                    dict(winner=winner, metadata=trained["metadata"], log=trained["log"]), "sha256")
        common.assert_tree_equal(metadata["winner"], winner, "trained winner")
        return dict(winner=winner, path=str(checkpoint_path.relative_to(ROOT)), sha256=metadata["sha256"])


def _winner_stage(name, rows, worker, workers, marker_name, output_key, expected_winners, **extra):
    results = {}
    for result in _parallel(rows, worker, workers):
        winner = result["winner"]
        results[winner["rep"], winner["arm"]] = result
        _progress(name, "running", completed_models=len(results), total_models=len(expected_winners),
                  completed_candidates=_unique_budget(), total_candidates=_unique_budget())
    ordered = [results[winner["rep"], winner["arm"]] for winner in expected_winners]
    summary = dict(**extra, **{output_key: ordered})
    _write(OUT / marker_name, summary)
    _progress(name, "complete", completed_models=len(ordered), total_models=len(expected_winners),
              completed_candidates=_unique_budget(), total_candidates=_unique_budget())
    return summary


def finals(workers=6):
    with _phase("finals"):
        winners = _check_search_completion(base.read(OUT / "search.json"))["winners"]
        frozen = OUT / "final-freeze.json"
        if frozen.is_file():
            result = base.read(frozen)
            common.assert_tree_equal([row["winner"] for row in result["models"]], winners, "final winners")
            for row in result["models"]:
                _verify_hash(ROOT / row["path"], row["sha256"])
            return result
        _progress("finals", "running", completed_models=0, total_models=len(winners),
                  completed_candidates=_unique_budget(), total_candidates=_unique_budget())
        return _winner_stage("finals", winners, _train_final, workers, "final-freeze.json", "models",
                             winners, all_final_training_complete_before_test=True)


def _test_final(row):
    winner = row["winner"]
    checkpoint_path = ROOT / row["path"]
    _verify_hash(checkpoint_path, row["sha256"])
    trained = _load(checkpoint_path)
    dest = checkpoint_path.parent / "evaluations"
    with _lock(dest / ".test.lock"):
        items = []
        for step, condition in CONDITIONS:
            stem = dest / f"{step:03d}_{condition}"
            archive_path, score_path = stem.with_suffix(".pt"), stem.with_suffix(".json")
            score = _recover_pair(archive_path, score_path, "archives_sha256")
            if score is None:
                score = core.score_model(winner["rep"], trained["checkpoints"][step]["model"],
                                         **_score_options("test"), archive=True,
                                         teacher_support=condition == "teacher", read_enabled=condition != "no_read",
                                         b_ticks=CONFIG["b_ticks"] if condition == "Bdelay" else 0)
                archives = score.pop("archives")
                score.update(winner=winner, step=step, condition=condition)
                score = _commit_pair(archive_path, score_path, archives, score, "archives_sha256")
            if (score["winner"] != winner or score["step"] != step or score["condition"] != condition
                    or score["parameter_hash"] != base.model_hash(trained["checkpoints"][step]["model"])):
                raise ValueError(f"held-out score does not match the frozen checkpoint: {score_path}")
            items.append(dict(name=stem.name, fitness=score["fitness"], sha256=base.sha(score_path)))
        completion = dict(winner=winner, conditions=items)
        marker = dest / "completion.json"
        if marker.is_file():
            common.assert_tree_equal(base.read(marker), completion, "held-out completion")
        else:
            _write(marker, completion)
        return completion


def tests(workers=6):
    with _phase("test"):
        frozen = base.read(OUT / "final-freeze.json")
        if frozen["all_final_training_complete_before_test"] is not True:
            raise ValueError("all finalists must be frozen before held-out testing")
        winners = _check_search_completion(base.read(OUT / "search.json"))["winners"]
        common.assert_tree_equal([row["winner"] for row in frozen["models"]], winners, "held-out winners")
        marker = OUT / "test-completion.json"
        if marker.is_file():
            result = base.read(marker)
            common.assert_tree_equal([row["winner"] for row in result["models"]], winners, "tested winners")
            for row in result["models"]:
                dest = _final_dir(row["winner"]) / "evaluations"
                common.assert_tree_equal(base.read(dest / "completion.json"), row, "held-out completion")
                for item in row["conditions"]:
                    _verify_hash(dest / f"{item['name']}.json", item["sha256"])
            return result
        _progress("test", "running", completed_models=0, total_models=len(winners),
                  completed_candidates=_unique_budget(), total_candidates=_unique_budget())
        return _winner_stage("test", frozen["models"], _test_final, workers, "test-completion.json", "models", winners)


def status():
    path = OUT / "progress.json"
    return base.read(path) if path.is_file() else dict(state="declared" if (OUT / "protocol.json").is_file() else "new")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "search", "finals", "test", "run", "status"))
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")
    if args.command == "declare":
        protocol = declare()
        print(json.dumps(dict(declared=str(OUT), unique_candidates=protocol["unique_candidates"])), flush=True)
    elif args.command == "status":
        print(json.dumps(status(), ensure_ascii=False, indent=2), flush=True)
    elif args.command == "search":
        search(args.workers or 14)
    elif args.command == "finals":
        finals(args.workers or 6)
    elif args.command == "test":
        tests(args.workers or 6)
    else:
        search(args.workers or 14)
        finals(6)
        tests(6)


if __name__ == "__main__":
    main()
