"""Replay archived candidates on a second host without joining the search."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch

from experiments.learned_binding import audit as binding_audit, common
from experiments.seed_evolution import audit, core, hybrid_backend, large_runner, lazy_training_streams


FORMAT = "seed-evolution-lan-probe-v1"
ARTIFACTS = {"initial.pt", "inner64.pt", "training.json", "score.json"}
DEV = {key: large_runner.CONFIG[key] for key in ("dev_streams", "support_ticks", "probe_ticks", "actors", "random_seed")}
STEPS = large_runner.CONFIG["inner_steps"]
MIB = 1 << 20


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _same(left, right):
    try:
        binding_audit.same_tree(left, right)
    except binding_audit.AuditError:
        return False
    return True


def _gpu_status():
    keys = ("name", "driver_version", "memory.total", "memory.used", "memory.free")
    command = ["nvidia-smi", f"--query-gpu={','.join(keys)}", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=8)
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": str(error)}
    devices = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != len(keys):
            return {"error": "unexpected nvidia-smi device record", "stdout": result.stdout}
        name, driver, total, used, free = row
        memory = [int(value) if value.isdigit() else None for value in (total, used, free)]
        devices.append(dict(name=name, driver_version=driver, total_mib=memory[0],
                            used_mib=memory[1], free_mib=memory[2]))
    return {"devices": devices}


def _cpu_capability():
    return torch.backends.cpu.get_cpu_capability()


def _host_info():
    return dict(host=platform.node(), platform=platform.platform(), machine=platform.machine(),
                processor=platform.processor(), logical_cpus=os.cpu_count(), python=sys.version,
                torch=str(torch.__version__), numpy=np.__version__, torch_cuda=torch.version.cuda,
                torch_cpu_capability=_cpu_capability(), torch_build=torch.__config__.show(),
                nvidia_smi_before=_gpu_status())


def _verify_frozen_inputs(root, manifest):
    hashes = manifest.get("frozen_inputs_sha256")
    if hashes is None:
        return 0
    for name, digest in hashes.items():
        audit.checked_file(root, name, digest)
    return len(hashes)


def _load_reference(root, row, backend):
    if type(row["rep"]) is not int or row["rep"] not in (0, 1, 2):
        raise ValueError("reference requires a historical source seed")
    if not re.fullmatch(r"(?:shared|evolution|random)_g\d{2}_i\d{2}", row["id"]):
        raise ValueError("invalid reference candidate ID")
    if set(row["artifacts"]) != ARTIFACTS:
        raise ValueError("reference requires the four bound candidate artifacts")
    directory = (root / row["directory"]).resolve()
    if not directory.is_relative_to(root):
        raise ValueError("reference directory must stay within the bundle")
    files = {name: audit.checked_file(directory, name, digest) for name, digest in row["artifacts"].items()}
    genome, checkpoint = (torch.load(files[name], map_location="cpu", weights_only=False)
                          for name in ("initial.pt", "inner64.pt"))
    training, score = (_read(files[name]) for name in ("training.json", "score.json"))
    rep = row["rep"]
    metadata = training["metadata"]
    expected_device = "cpu+cuda_executor" if backend == "hybrid" else "cpu"
    if (metadata["source_seed"], metadata["steps"], metadata["device"], metadata["genome_hash"]) != (
            rep, STEPS, expected_device, core.genome_hash(genome)):
        raise ValueError("archived training does not match the reference source, budget, backend, or genome")
    audit.audit_frozen_checkpoint(checkpoint, rep, STEPS, genome, common.executor(rep).state_dict())
    audit.verify_scoring(score, rep, DEV["dev_streams"], binding_audit.parameter_hash(checkpoint["model"]), DEV)
    return genome, checkpoint, training, score


def _tensor_difference(expected, observed):
    if expected.shape != observed.shape or expected.dtype != observed.dtype:
        raise ValueError("checkpoint tensor shapes or dtypes differ")
    diff = (expected.double() - observed.double()).abs()
    if not bool(torch.isfinite(diff).all()):
        raise ValueError("nonfinite checkpoint tensor difference")
    before = expected.contiguous().numpy().tobytes()
    after = observed.contiguous().numpy().tobytes()
    scalar_bytes = expected.element_size()
    differing = 0 if before == after else int(np.count_nonzero(np.any(
        np.frombuffer(before, dtype=np.uint8).reshape(-1, scalar_bytes)
        != np.frombuffer(after, dtype=np.uint8).reshape(-1, scalar_bytes), axis=1)))
    return dict(scalars=diff.numel(), bitwise_different=differing,
                max_abs=float(diff.max()) if diff.numel() else 0.)


def _tensor_differences(expected, observed):
    if expected.keys() != observed.keys():
        raise ValueError("checkpoint tensor keys differ")
    rows = {str(key): _tensor_difference(value, observed[key]) for key, value in expected.items()}
    return dict(scalars=sum(row["scalars"] for row in rows.values()),
                bitwise_different=sum(row["bitwise_different"] for row in rows.values()),
                max_abs=max((row["max_abs"] for row in rows.values()), default=0.),
                by_tensor=rows)


def _checkpoint_comparison(expected, observed):
    before, after = (checkpoint["optimizer"]["state"] for checkpoint in (expected, observed))
    return dict(
        bitwise_equal=_same(expected, observed),
        model=_tensor_differences(expected["model"], observed["model"]),
        optimizer={field: _tensor_differences({key: row[field] for key, row in before.items()},
                                             {key: row[field] for key, row in after.items()})
                   for field in ("step", "exp_avg", "exp_avg_sq")},
    )


def _score_comparison(expected, observed):
    before = {row["stream_id"]: row for row in expected["streams"]}
    after = {row["stream_id"]: row for row in observed["streams"]}
    if before.keys() != after.keys():
        raise ValueError("development world IDs differ from archive")
    streams = []
    for stream_id, old in before.items():
        new = after[stream_id]
        stages = {stage: dict(archived=old[stage]["actions_hash"], observed=new[stage]["actions_hash"],
                              equal=old[stage]["actions_hash"] == new[stage]["actions_hash"])
                  for stage in ("support", "probe")}
        streams.append(dict(stream_id=stream_id, actions=stages, groups_equal=_same(old["groups"], new["groups"]),
                            archived_fitness=old["fitness"], observed_fitness=new["fitness"]))
    return dict(full_score_equal=_same(expected, observed),
                score_except_parameter_hashes_equal=_same(_without_parameter_hashes(expected),
                                                         _without_parameter_hashes(observed)),
                archived_parameter_hash=expected["parameter_hash"], observed_parameter_hash=observed["parameter_hash"],
                archived_fitness=expected["fitness"], observed_fitness=observed["fitness"],
                all_actions_equal=all(stage["equal"] for row in streams for stage in row["actions"].values()),
                streams=streams)


def _without_parameter_hashes(value):
    if isinstance(value, dict):
        return {key: _without_parameter_hashes(item) for key, item in value.items() if key != "parameter_hash"}
    if isinstance(value, list):
        return [_without_parameter_hashes(item) for item in value]
    return value


def _train(rep, genome, backend):
    cuda = backend == "hybrid"
    gpu = None
    if cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("hybrid probe requires a CUDA device visible to PyTorch")
        # Set the required cuBLAS workspace before the first CUDA context is
        # created. The production backend validates it again inside training.
        previous = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if previous not in {":4096:8", ":16:8"}:
            raise ValueError("invalid CUBLAS_WORKSPACE_CONFIG for hybrid replay")
        free, total = torch.cuda.mem_get_info(0)
        torch.cuda.reset_peak_memory_stats(0)
        gpu = dict(free_before_mib=free / MIB, total_mib=total / MIB)
    streams = lazy_training_streams.LazyTrainingStreams()
    start = perf_counter()
    with streams.install():
        trained = hybrid_backend.train(rep, genome, STEPS, (0, STEPS), backend=backend, cache=True)
    if cuda:
        torch.cuda.synchronize(0)
        free, _ = torch.cuda.mem_get_info(0)
        gpu.update(free_after_mib=free / MIB, torch_peak_allocated_mib=torch.cuda.max_memory_allocated(0) / MIB,
                   torch_peak_reserved_mib=torch.cuda.max_memory_reserved(0) / MIB)
    return trained, perf_counter() - start, gpu


def _probe(root, row, backend):
    genome, archived, training, score = _load_reference(root, row, backend)
    rep = row["rep"]
    trained, train_seconds, gpu = _train(rep, genome, backend)
    checkpoint = trained["checkpoints"][STEPS]
    audit.audit_frozen_checkpoint(checkpoint, rep, STEPS, genome, common.executor(rep).state_dict())
    comparison = _checkpoint_comparison(archived, checkpoint)
    start = perf_counter()
    replay = core.score_model(rep, checkpoint["model"], **audit.scoring_options(DEV, "dev_streams"))
    eval_seconds = perf_counter() - start
    score_comparison = _score_comparison(score, replay)
    log_equal = _same(training["log"], trained["log"])
    metadata_equal = _same(_arithmetic_metadata(training), _arithmetic_metadata(trained))
    exact = comparison["bitwise_equal"] and log_equal and metadata_equal and score_comparison["full_score_equal"]
    return dict(reference=dict(rep=rep, id=row["id"], artifacts=row["artifacts"]), exact_parity=exact,
                checkpoint=comparison, training_log_equal=log_equal, arithmetic_metadata_equal=metadata_equal,
                archived_log=training["log"], observed_log=trained["log"],
                archived_execution=training["metadata"].get("execution"),
                observed_execution=trained["metadata"]["execution"],
                development_score=score_comparison, train_seconds=train_seconds, eval_seconds=eval_seconds,
                gpu_memory=gpu, nvidia_smi_after=_gpu_status())


def _arithmetic_metadata(training):
    return {key: value for key, value in training["metadata"].items() if key != "execution"}


def probe(manifest_path, backend):
    manifest_path = Path(manifest_path).resolve()
    manifest = _read(manifest_path)
    if manifest["format"] != FORMAT:
        raise ValueError("unknown LAN probe manifest")
    modes = ("cpu", "hybrid") if backend == "both" else (backend,)
    references = manifest["references"]
    if len({references[mode]["rep"] for mode in modes}) != 1:
        raise ValueError("selected references must share a source seed")
    torch.set_num_threads(1)
    report = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
                  manifest_sha256=common.sha(manifest_path), host=_host_info(),
                  frozen_inputs_checked=_verify_frozen_inputs(manifest_path.parent, manifest),
                  results={mode: _probe(manifest_path.parent, references[mode], mode) for mode in modes})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "hybrid", "both"), default="both")
    parser.add_argument("--output", type=Path, help="Write a copy of the JSON report to a new file")
    args = parser.parse_args()
    report = probe(args.manifest, args.backend)
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    print(payload, end="")
    cpu = report["results"].get("cpu")
    return 2 if cpu is not None and not cpu["exact_parity"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
