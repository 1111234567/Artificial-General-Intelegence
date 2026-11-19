"""Run the equal-budget population-width study with pipelined GPU and CPU stages."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import math
import os
from pathlib import Path
import statistics

import torch

from experiments.learned_binding import common
from experiments.seed_evolution import core, device_benchmark, gpu_batch_train, large_runner as files
from experiments.seed_evolution import scale_search_plan as plan


DEFAULT_OUT = core.ROOT / "data/seed-evolution/2026-11-15-scale-bf16-v1"
BATCH = 40
CHUNK = 8
STUDY_SIZE = len(plan.SOURCE_SEEDS) * (plan.SHARED_COUNT + len(plan.ARMS) * (plan.BUDGET - plan.SHARED_COUNT))
ARTIFACTS = ("initial.pt", "inner64.pt", "training.json", "score.json")


def _read(path):
    return files.base.read(path)


def _trial_dir(root, descriptor):
    return root / "search" / f"rep_{descriptor['rep']}" / descriptor["id"]


def _committed(root, descriptor, *, verify=True):
    folder = _trial_dir(root, descriptor)
    marker = folder / "candidate.json"
    if not marker.is_file():
        return None
    record = _read(marker)
    if (any(record[key] != value for key, value in descriptor.items())
            or set(record["artifacts"]) != set(ARTIFACTS)
            or not math.isfinite(record["fitness"])):
        raise ValueError(f"invalid candidate commit: {marker}")
    if verify:
        for name, digest in record["artifacts"].items():
            files._verify_hash(folder / name, digest)
    return record


def _genome(root, descriptor):
    if descriptor["parent_id"] is None:
        return core.new_genome(descriptor["rep"], descriptor["init_seed"])
    parent = _trial_dir(root, dict(rep=descriptor["rep"], id=descriptor["parent_id"]))
    record = _read(parent / "candidate.json")
    initial = parent / "initial.pt"
    files._verify_hash(initial, record["artifacts"]["initial.pt"])
    return core.mutate_genome(files._load(initial), descriptor["mutation_seed"], descriptor["sigma"])


def _training(folder):
    return files._recover_pair(folder / "inner64.pt", folder / "training.json", "checkpoint_sha256")


def _train(root, descriptors):
    stages = {}
    for descriptor in descriptors:
        folder = _trial_dir(root, descriptor)
        if _committed(root, descriptor) is None and _training(folder) is None:
            stages[descriptor["id"]] = folder
    if not stages:
        return

    genomes = [_genome(root, descriptor) for descriptor in descriptors]
    # Keep lane count and executor chunk fixed, including on short batches.
    padded = genomes + [genomes[0]] * (BATCH - len(genomes))
    source = descriptors[0]["rep"]
    output = gpu_batch_train.train_batch(
        source, padded, plan.INNER_STEPS, dtype=torch.bfloat16, executor_chunk=CHUNK,
    )
    checkpoint_rows = zip(descriptors, genomes, output["checkpoints"][:len(genomes)],
                          output["losses"][:len(genomes)], output["grad_norms"][:len(genomes)], strict=True)
    for descriptor, genome, checkpoint, losses, norms in checkpoint_rows:
        if descriptor["id"] not in stages:
            continue
        folder = stages[descriptor["id"]]
        initial = folder / "initial.pt"
        if initial.is_file():
            common.assert_tree_equal(files._load(initial), genome, "resumed seed genome")
        else:
            files._save(initial, genome)
        training = dict(descriptor=descriptor, genome_hash=core.genome_hash(genome),
                        training=output["metadata"], batch_timing=output["timing"],
                        losses=losses, grad_norms=norms)
        files._commit_pair(folder / "inner64.pt", folder / "training.json", checkpoint, training,
                           "checkpoint_sha256")


def _score(root, descriptor):
    folder = _trial_dir(root, descriptor)
    with files._lock(folder / ".candidate.lock"):
        record = _committed(root, descriptor)
        if record is not None:
            return record
        training = _training(folder)
        if training is None or training["descriptor"] != descriptor:
            raise ValueError(f"prepared training does not match candidate plan: {folder}")
        checkpoint = files._load(folder / "inner64.pt")
        if checkpoint["updates"] != plan.INNER_STEPS or checkpoint["source_seed"] != descriptor["rep"]:
            raise ValueError(f"candidate training checkpoint has the wrong budget or source: {folder}")
        score = core.score_model(descriptor["rep"], checkpoint["model"], **plan.scoring_options("dev"))
        if not math.isfinite(score["fitness"]):
            raise ValueError(f"candidate score is not finite: {folder}")
        files._write(folder / "score.json", score)
        record = dict(descriptor, fitness=score["fitness"], genome_hash=training["genome_hash"],
                      model_hash=score["parameter_hash"],
                      artifacts={name: common.sha(folder / name) for name in ARTIFACTS})
        files._write(folder / "candidate.json", record)
        return record


def _issue(root, generation, records):
    arms = ("shared",) if generation == 0 else tuple(
        arm for arm in plan.ARMS if generation <= plan.SCHEDULES[arm][0])
    plans = [plan.generation_plan(rep, arm, generation, records[rep])
             for rep in plan.SOURCE_SEEDS for arm in arms]
    value = {"round": generation, "plans": plans}
    path = root / "plans" / f"round_{generation:03d}.json"
    if path.is_file():
        common.assert_tree_equal(_read(path), value, "issued scale-search plan")
    else:
        files._write(path, value)
    return plans


def _batches(plans):
    for rep in plan.SOURCE_SEEDS:
        items = [row for part in plans if part["rep"] == rep for row in part["candidates"]]
        for start in range(0, len(items), BATCH):
            yield items[start:start + BATCH]


def _collect(futures, records):
    for future in futures:
        record = future.result()
        records[record["rep"]][record["id"]] = record


def _round(root, generation, records, pool, progress):
    plans = _issue(root, generation, records)
    pending = []
    for index, descriptors in enumerate(_batches(plans)):
        missing = []
        for descriptor in descriptors:
            record = _committed(root, descriptor)
            if record is None:
                missing.append(descriptor)
            else:
                records[record["rep"]][record["id"]] = record
        if not missing:
            continue
        progress("search", round=generation, active_rep=descriptors[0]["rep"],
                 active_batch=index, gpu_candidates=len(missing), pending_scores=len(pending))
        _train(root, descriptors)
        _collect(pending, records)
        pending = [pool.submit(_score, root, descriptor) for descriptor in missing]
    _collect(pending, records)
    summary = []
    for item in plans:
        rep, arm = item["rep"], item["arm"]
        candidates = [records[rep][row["id"]]["fitness"] for row in item["candidates"]]
        ids = [row["id"] for row in item["candidates"]] if arm == "shared" else plan.pool_ids(arm, generation)
        winner = records[rep][plan.ranked(ids, records[rep])[0]]
        summary.append(dict(rep=rep, arm=arm, generation=generation, scored=len(candidates),
                            mean=statistics.fmean(candidates), median=statistics.median(candidates),
                            best_new=max(candidates), best_id=winner["id"], best_fitness=winner["fitness"]))
    path = root / "rounds" / f"round_{generation:03d}.json"
    if path.is_file():
        common.assert_tree_equal(_read(path), summary, "sealed scale-search round")
    else:
        files._write(path, summary)
        print(f"Round {generation} complete. Committed {sum(map(len, records.values()))}/{STUDY_SIZE}.", flush=True)


def _protocol(root, device):
    original = _read(files.OUT / "protocol.json")
    locked = dict(original["source_hashes"])
    for path in (__file__, plan.__file__, gpu_batch_train.__file__):
        relative = Path(path).resolve().relative_to(core.ROOT).as_posix()
        locked[relative] = common.sha(path)
    for relative, digest in locked.items():
        files._verify_hash(core.ROOT / relative.replace("\\", "/"), digest)
    for reference in (*original["origins"].values(), original["data_freeze_ref"]):
        files._verify_hash(core.ROOT / reference["path"].replace("\\", "/"), reference["sha256"])
    cuda = torch.cuda.get_device_properties(device)
    study = dict(format="seed-evolution-population-width-v1", plan=plan.CONFIG,
                 gpu_batch=BATCH, executor_chunk=CHUNK, executor_dtype="bfloat16",
                 recurrent_and_optimizer_dtype="float32", unique_candidates=STUDY_SIZE,
                 device=cuda.name, torch=str(torch.__version__), cuda=torch.version.cuda,
                 source_hashes=locked, origins=original["origins"], data_freeze_ref=original["data_freeze_ref"])
    path = root / "protocol.json"
    if path.is_file():
        common.assert_tree_equal(_read(path), study, "scale-search protocol")
    else:
        files._write(path, study)


def run(root, workers, founders_only=False):
    root = root.resolve()
    if root == files.OUT.resolve():
        raise ValueError("choose a new output directory for this experiment")
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device_benchmark.strict_fp32()
    torch.backends.cudnn.benchmark = False
    records = {rep: {} for rep in plan.SOURCE_SEEDS}

    def progress(phase, **fields):
        files._write(root / "progress.json", dict(phase=phase, updated_utc=files._utc(),
                     completed_candidates=sum(map(len, records.values())), total_candidates=STUDY_SIZE,
                     score_workers=workers, gpu_batch=BATCH, **fields))

    with files._lock(root / ".pipeline.lock"):
        try:
            files._write(root / "process.json", dict(pid=os.getpid(), started_utc=files._utc(),
                                                    workers=workers))
            _protocol(root, torch.device("cuda:0"))
            with ProcessPoolExecutor(max_workers=workers) as pool:
                max_round = 0 if founders_only else max(count for count, _ in plan.SCHEDULES.values())
                for generation in range(max_round + 1):
                    _round(root, generation, records, pool, progress)
                    progress("search", round=generation, round_complete=True)
            if founders_only:
                return
            if sum(map(len, records.values())) != STUDY_SIZE:
                raise ValueError("completed search does not match the evaluation budget")
            winners = []
            for rep in plan.SOURCE_SEEDS:
                for arm in plan.ARMS:
                    record = records[rep][plan.ranked(plan.pool_ids(arm), records[rep])[0]]
                    winners.append(dict(source_seed=rep, arm=arm, id=record["id"], fitness=record["fitness"],
                                        trial_dir=_trial_dir(root, record).relative_to(root).as_posix()))
            files._write(root / "search-completion.json", dict(winners=winners, candidates=STUDY_SIZE))
            from experiments.seed_evolution import scale_search_final

            scale_search_final.finish(root, winners, workers, progress)
            progress("complete")
        except Exception as error:
            progress("failed", error=repr(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--score-workers", type=int, default=min(20, os.process_cpu_count() or 1))
    parser.add_argument("--founders-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.score_workers <= 61:
        parser.error("score worker count must be in 1..61")
    run(args.output, args.score_workers, args.founders_only)


if __name__ == "__main__":
    main()
