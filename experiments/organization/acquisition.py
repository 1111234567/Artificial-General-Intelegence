"""Source-bound zero-shrink acquisition controls around the frozen native trainer.

Formal prepare changes only the three stated variant choices and shrink_step=0.
No numerical model, live state, optimizer, teacher, or replay rule is replaced.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch

from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import atomic_json, state_digest
from experiments.credit.run import file_sha256
from experiments.lifelong import run as native
from experiments.lifelong.evaluate import evaluate as evaluate_fork
from experiments.loops.audit_run import verify_files

ROOT = Path(__file__).resolve().parents[2]
FORMAT = "organization-acquisition-protocol-v1"
DEFAULT_EVIDENCE = ROOT / "docs/evidence/organization-2026-10-30"
DEFAULT_RAW = ROOT / "data/organization/acquisition-v1"
OLD_EVIDENCE = ROOT / "docs/evidence/lifelong-2026-10-29"
KEYS = ("architecture", "schedule", "learning", "coverage")
VARIANTS = (
    dict(architecture="fixed_shrink", schedule="mixed", learning="history_replay", coverage="wide"),
    dict(architecture="open_shrink", schedule="mixed", learning="history_replay", coverage="wide"),
    dict(architecture="fixed_shrink", schedule="mixed", learning="history_replay", coverage="narrow"),
)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalized(value):
    return json.loads(json.dumps(value))


def execution_files():
    """Deliberately explicit: unrelated organization files may still evolve."""
    paths = [Path(__file__), Path(__file__).with_name("test_acquisition.py")]
    paths += [ROOT / "experiments/lifelong" / name for name in native.RUNTIME_SOURCES]
    paths += [ROOT / "experiments" / name for name in native.DEPENDENCIES]
    paths += [ROOT / name for name in (
        "experiments/organization/__init__.py", "experiments/lifelong/__init__.py",
        "experiments/lifelong/evaluate.py", "experiments/lifelong/queue.py",
        "experiments/continuity/evaluate.py", "experiments/integration/diagnose.py",
        "experiments/credit/run.py", "experiments/credit/local.py", "experiments/credit/batch.py",
        "experiments/credit/windows_runner.py", "experiments/loops/audit_run.py",
        "experiments/loops/structure.py", "experiments/loops/interventions.py",
    )]
    return {str(p.resolve()): file_sha256(p) for p in paths}


def evaluation_specs(checkpoint_ticks, steps=2048):
    return [dict(tick=tick, family=family, condition=condition, steps=steps)
            for tick in checkpoint_ticks if tick
            for family in range(4) for condition in ("short", "long_both")]


def baseline_bindings(checkpoint_ticks):
    """Use the already strictly audited CSV and bind its raw checkpoint bytes."""
    completion_path = OLD_EVIDENCE / "completion.json"
    completion = _json(completion_path)
    if not completion.get("complete") or completion.get("audit", {}).get("status") != "passed" or completion["audit"].get("errors"):
        raise ValueError("old native study has no passed strict audit")
    artifact_path = OLD_EVIDENCE / "aggregate-v1/artifacts.json"
    artifacts = _json(artifact_path)
    if artifacts != completion["aggregate_artifacts"]:
        raise ValueError("old aggregate artifact binding mismatch")
    names = ("checkpoints.csv", "matched_memory_per_seed.csv", "per_seed.csv", "supplement_per_seed.csv")
    files = {str(completion_path): file_sha256(completion_path), str(artifact_path): file_sha256(artifact_path)}
    for name in names:
        path = artifact_path.parent / name
        if file_sha256(path) != artifacts[name]:
            raise ValueError("old audited CSV changed")
        files[str(path)] = artifacts[name]
    wanted = {f"seed_{seed:03d}__" + "__".join(v[k] for k in KEYS)
              for seed in (0, 1, 2) for v in VARIANTS}
    entries = {identifier: {} for identifier in wanted}
    with (artifact_path.parent / "checkpoints.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["id"] not in wanted or int(row["tick"]) not in checkpoint_ticks:
                continue
            tick = int(row["tick"])
            if str(tick) in entries[row["id"]]:
                raise ValueError("duplicate audited baseline checkpoint")
            label = "__".join(row[k] for k in KEYS)
            expected = ROOT / "data/lifelong/main-v1/training" / f"seed_{int(row['seed']):03d}" / label / "checkpoints" / f"tick_{tick:06d}.pt"
            cp = Path(row["checkpoint"])
            if cp.resolve() != expected.resolve() or file_sha256(cp) != row["checkpoint_sha256"]:
                raise ValueError("foreign or changed old raw checkpoint")
            entries[row["id"]][str(tick)] = {k: row[k] for k in (
                "checkpoint", "checkpoint_sha256", "semantic_hash", "cpu_semantic_hash",
                "initial_parameter_hash", "birth_hash")}
    if any(set(rows) != {str(t) for t in checkpoint_ticks} for rows in entries.values()):
        raise ValueError("missing audited baseline snapshots")
    return dict(files=files, branches=entries, shrink_step=2e-4,
                note="Read-only existing main-v1 controls; no baseline retraining.")


def prepare(folder=DEFAULT_EVIDENCE, raw=DEFAULT_RAW, *, template=None, baseline=None):
    """template/baseline injection is for finite mechanical tests, not CLI runs."""
    folder, raw = Path(folder).resolve(), Path(raw).resolve()
    path = folder / "acquisition-protocol-v1.json"
    targets = [path] + [folder / f"acquisition-{kind}-commands-v1.json"
                        for kind in ("training", "evaluation", "baseline")]
    if any(p.exists() for p in targets) or raw.exists():
        raise FileExistsError("acquisition prepare requires new protocol, manifests and raw root")
    config = native.NativeConfig(device="cuda") if template is None else template
    if config.shrink_step != 2e-4:
        raise ValueError("template must describe the original shrink baseline")
    ticks = [0, config.main_ticks, config.main_ticks + config.withdrawal_ticks, config.total_ticks]
    old = baseline_bindings(ticks) if baseline is None else baseline
    branches = []
    for seed in (0, 1, 2):
        for variant in VARIANTS:
            label = "__".join(variant[k] for k in KEYS)
            old_id = f"seed_{seed:03d}__{label}"
            identifier = old_id + "__shrink0"
            cfg = replace(config, seed=seed, **variant, shrink_step=0.)
            branches.append(dict(id=identifier, seed=seed, **variant, shrink_step=0.,
                native_config=_normalized(asdict(cfg)), baseline_id=old_id,
                output=str(raw / "training" / f"seed_{seed:03d}" / (label + "__shrink0"))))
    plan = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
        sources=execution_files(), raw_root=str(raw), branches=branches, variants=VARIANTS,
        seeds=[0, 1, 2], native_template=_normalized(asdict(config)),
        target_ticks=config.total_ticks, checkpoint_ticks=ticks,
        latest_interval_ticks=32768 if template is None else config.macrocycle_ticks,
        training_device=config.device, evaluation_device="cpu", threads=1,
        dt=config.dt, evaluation_specs=evaluation_specs(ticks, 2048 if template is None else 8),
        baseline=old, interpretation="Paired zero versus 2e-4 shrink acquisition controls; complete native lifetime and state, no causal topology claim.")
    atomic_json(path, plan)
    frozen = {**plan["sources"], **old["files"], str(path): file_sha256(path)}
    jobs = {kind: [] for kind in ("training", "evaluation", "baseline")}
    for item in branches:
        out = Path(item["output"])
        tail = ["--protocol", str(path), "--branch-id", item["id"]]
        prefix = [sys.executable, "-m", "experiments.organization.acquisition"]
        jobs["training"].append(dict(id=item["id"], command=prefix + ["train"] + tail,
            outputs=[str(out / "latest.pt"), str(out / "launch-complete.json")],
            log=str(raw / "logs" / f"train_{item['id']}.log")))
        output = raw / "evaluation" / f"{item['id']}.json"
        jobs["evaluation"].append(dict(id=item["id"], command=prefix + ["evaluate"] + tail +
            ["--checkpoint", str(out / "latest.pt"), "--output", str(output)],
            outputs=[str(output)], log=str(raw / "logs" / f"evaluate_{item['id']}.log")))
    for kind, rows in jobs.items():
        atomic_json(folder / f"acquisition-{kind}-commands-v1.json",
                    dict(cwd=str(ROOT), file_hashes=frozen, jobs=rows))
    return path


def read_plan(path, identifier):
    path = Path(path).resolve()
    sha, plan = file_sha256(path), _json(path)
    if plan.get("format") != FORMAT or plan.get("threads") != 1 or plan.get("evaluation_device") != "cpu":
        raise ValueError("invalid acquisition protocol")
    verify_files(plan["sources"])
    if plan["sources"] != execution_files():
        raise ValueError("acquisition source inventory changed")
    verify_files(plan["baseline"]["files"])
    if len(plan["branches"]) != 9 or len({b["id"] for b in plan["branches"]}) != 9:
        raise ValueError("invalid acquisition branch inventory")
    template = native.NativeConfig(**plan["native_template"])
    ticks = [0, template.main_ticks, template.main_ticks + template.withdrawal_ticks, template.total_ticks]
    if plan["checkpoint_ticks"] != ticks or plan["target_ticks"] != template.total_ticks:
        raise ValueError("acquisition checkpoint budget changed")
    if plan["evaluation_specs"] != evaluation_specs(ticks, plan["evaluation_specs"][0]["steps"]):
        raise ValueError("invalid acquisition evaluation inventory")
    found = []
    for seed in (0, 1, 2):
        for variant in VARIANTS:
            label = "__".join(variant[k] for k in KEYS)
            old_id = f"seed_{seed:03d}__{label}"
            expected = dict(id=old_id + "__shrink0", seed=seed, **variant, shrink_step=0.,
                native_config=_normalized(asdict(replace(template, seed=seed, **variant, shrink_step=0.))),
                baseline_id=old_id, output=str(Path(plan["raw_root"]) / "training" /
                    f"seed_{seed:03d}" / (label + "__shrink0")))
            if expected not in plan["branches"]:
                raise ValueError("foreign acquisition branch/config/output")
            if expected["id"] == identifier:
                found.append(expected)
    if len(found) != 1:
        raise ValueError("unknown acquisition branch")
    return plan, found[0], sha


def normalized_hash(trainer, device):
    value = trainer.semantic_state()
    value["config"]["device"] = device
    return state_digest(value)


def validate(trainer, plan, item, tick=None):
    expected = dict(item["native_config"], device=str(trainer.device))
    if _normalized(asdict(trainer.config)) != expected or trainer.config.shrink_step != 0.:
        raise ValueError("foreign acquisition native configuration")
    if (trainer.hidden_resets or trainer.ticks % trainer.config.chunk_ticks or
            not 0 <= trainer.ticks <= plan["target_ticks"] or
            (tick is not None and trainer.ticks != tick)):
        raise ValueError("invalid acquisition progress")
    old = plan["baseline"]["branches"][item["baseline_id"]]["0"]
    if (trainer.initial_parameter_hash != old["initial_parameter_hash"] or
            state_digest(trainer.birth) != old["birth_hash"]):
        raise ValueError("acquisition birth differs from paired original seed")
    if trainer.operator.config.effective_shrink_step != 0.:
        raise ValueError("acquisition operator still shrinks")


def verify_end(path, identifier, sha):
    if file_sha256(path) != sha:
        raise ValueError("acquisition protocol changed")
    read_plan(path, identifier)


def _snapshot(out, trainer, plan, item):
    cp = out / "checkpoints" / f"tick_{trainer.ticks:06d}.pt"
    if cp.exists():
        expected = normalized_hash(trainer, plan["training_device"])
        with _preserve_process_state():
            saved = native.NativeTrainer.from_checkpoint(cp, device="cpu")
            validate(saved, plan, item, trainer.ticks)
            if normalized_hash(saved, plan["training_device"]) != expected:
                raise ValueError("refusing to overwrite different acquisition snapshot")
    else:
        trainer.save_checkpoint(cp)


def completion(plan, item, sha):
    out = Path(item["output"])
    done = _json(out / "launch-complete.json")
    if (done.get("format") != "lifelong-launch-complete-v1" or done.get("branch") != item or
            done.get("protocol_sha256") != sha or done.get("stage_ticks") != plan["target_ticks"] or
            done.get("target") != plan["target_ticks"] or
            done.get("checkpoint_sha256") != file_sha256(out / "latest.pt") or
            set(done.get("checkpoints", {})) != {str(t) for t in plan["checkpoint_ticks"]}):
        raise ValueError("invalid acquisition completion")
    for tick, entry in done["checkpoints"].items():
        cp = out / "checkpoints" / f"tick_{int(tick):06d}.pt"
        if entry.get("path") != str(cp) or file_sha256(cp) != entry["sha256"]:
            raise ValueError("acquisition snapshot bytes changed")
    with _preserve_process_state():
        final = native.NativeTrainer.from_checkpoint(out / "latest.pt", device="cpu")
        validate(final, plan, item, plan["target_ticks"])
        if (done.get("birth_hash") != state_digest(final.birth) or
                done.get("initial_parameter_hash") != final.initial_parameter_hash):
            raise ValueError("acquisition completion birth binding changed")
        if normalized_hash(final, plan["training_device"]) != done["checkpoints"][str(plan["target_ticks"])]["semantic_hash"]:
            raise ValueError("acquisition latest differs from final snapshot")
    return done


def train(path, identifier, *, stop_at=None):
    torch.set_num_threads(1)
    plan, item, sha = read_plan(path, identifier)
    out = Path(item["output"])
    if (out / "launch-complete.json").exists():
        completion(plan, item, sha)
        verify_end(path, identifier, sha)
        return
    start = out.with_name(out.name + ".launch-start.json")
    binding = dict(format="organization-acquisition-start-v1", branch=item, protocol_sha256=sha)
    if start.exists():
        if _json(start) != binding:
            raise ValueError("foreign acquisition start binding")
    else:
        if out.exists():
            raise ValueError("refusing unbound existing acquisition output")
        atomic_json(start, binding)
    latest = out / "latest.pt"
    birth_cp = out / "checkpoints/tick_000000.pt"
    resume = latest if latest.exists() else birth_cp if birth_cp.exists() else None
    if resume is None:
        if out.exists() and any(out.iterdir()):
            raise ValueError("existing acquisition files without a resumable checkpoint")
        trainer = native.NativeTrainer(native.NativeConfig(**item["native_config"]))
    else:
        trainer = native.NativeTrainer.from_checkpoint(resume, device=plan["training_device"])
    validate(trainer, plan, item)
    target = plan["target_ticks"] if stop_at is None else stop_at
    if type(target) is not int or not trainer.ticks <= target <= plan["target_ticks"] or target % trainer.config.chunk_ticks:
        raise ValueError("invalid acquisition stopping tick")
    if trainer.ticks in plan["checkpoint_ticks"]:
        _snapshot(out, trainer, plan, item)
    trainer.save_checkpoint(latest)
    native.write_run(out, trainer)
    try:
        while trainer.ticks < target:
            trainer.train_update()
            if trainer.ticks in plan["checkpoint_ticks"]:
                _snapshot(out, trainer, plan, item)
            if trainer.ticks % plan["latest_interval_ticks"] == 0:
                verify_end(path, identifier, sha)
                trainer.save_checkpoint(latest)
                native.write_run(out, trainer)
                print(json.dumps(dict(branch=identifier, tick=trainer.ticks,
                                      target=plan["target_ticks"], seconds=trainer.seconds)), flush=True)
    finally:
        if trainer._at_boundary:
            verify_end(path, identifier, sha)
            trainer.save_checkpoint(latest)
            native.write_run(out, trainer)
    if trainer.ticks != plan["target_ticks"]:
        return
    entries = {}
    with _preserve_process_state():
        for tick in plan["checkpoint_ticks"]:
            cp = out / "checkpoints" / f"tick_{tick:06d}.pt"
            saved = native.NativeTrainer.from_checkpoint(cp, device="cpu")
            validate(saved, plan, item, tick)
            entries[str(tick)] = dict(path=str(cp), sha256=file_sha256(cp),
                semantic_hash=normalized_hash(saved, plan["training_device"]),
                initial_parameter_hash=saved.initial_parameter_hash, birth_hash=state_digest(saved.birth))
    verify_end(path, identifier, sha)
    atomic_json(out / "launch-complete.json", dict(format="lifelong-launch-complete-v1",
        branch=item, protocol_sha256=sha, target=plan["target_ticks"], stage_ticks=trainer.ticks,
        checkpoint_sha256=file_sha256(latest), checkpoints=entries, hidden_resets=0,
        initial_parameter_hash=trainer.initial_parameter_hash, birth_hash=state_digest(trainer.birth),
        baseline_id=item["baseline_id"], baseline_files=plan["baseline"]["files"]))


def evaluate(path, identifier, checkpoint, output):
    torch.set_num_threads(1)
    plan, item, sha = read_plan(path, identifier)
    out, output = Path(item["output"]), Path(output).resolve()
    expected_output = Path(plan["raw_root"]) / "evaluation" / f"{identifier}.json"
    if (Path(checkpoint).resolve() != (out / "latest.pt").resolve() or
            output != expected_output.resolve() or output.exists()):
        raise ValueError("wrong acquisition evaluation path or existing output")
    done = completion(plan, item, sha)
    records, states = [], {}
    with _preserve_process_state():
        for tick in plan["checkpoint_ticks"][1:]:
            entry = done["checkpoints"][str(tick)]
            cp = Path(entry["path"])
            trainer = native.NativeTrainer.from_checkpoint(cp, device="cpu")
            validate(trainer, plan, item, tick)
            before = state_digest(trainer.semantic_state())
            normalized = normalized_hash(trainer, plan["training_device"])
            if normalized != entry["semantic_hash"]:
                raise ValueError("acquisition snapshot semantics changed")
            states[str(tick)] = dict(checkpoint=str(cp), checkpoint_sha256=entry["sha256"],
                state_hash=before, training_device_normalized_state_hash=normalized,
                family_recency=trainer.report()["family_recency"])
            for spec in (s for s in plan["evaluation_specs"] if s["tick"] == tick):
                record = evaluate_fork(trainer.model, trainer.world, trainer.h, family=spec["family"],
                    condition=spec["condition"], steps=spec["steps"], dt=plan["dt"], device="cpu")
                if any(record[key] for key in ("world_resets", "hidden_resets", "parameter_updates")):
                    raise AssertionError("acquisition evaluation reset live state or learned")
                record["probe"] = spec
                records.append(record)
            if state_digest(trainer.semantic_state()) != before or file_sha256(cp) != entry["sha256"]:
                raise AssertionError("acquisition frozen evaluation changed its source")
    verify_end(path, identifier, sha)
    atomic_json(output, dict(format="organization-acquisition-evaluation-v1", branch=item,
        protocol_sha256=sha, source_unchanged=True, states=states, records=records,
        baseline_id=item["baseline_id"], birth_hash=done["birth_hash"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_EVIDENCE / "acquisition-protocol-v1.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--branch-id")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.mode == "prepare":
        print(prepare(args.output, args.raw))
    elif args.mode == "train":
        train(args.protocol, args.branch_id)
    else:
        evaluate(args.protocol, args.branch_id, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
