"""Source-bound nonlinear-field and functional-structure organization study.

Formal preparation declares 27 uninterrupted lifetimes: three seeds and nine
predeclared variants. Mechanical tests may pass a smaller NativeConfig, while
keeping the same variant inventory and all state/provenance checks.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch

from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import atomic_json, state_digest
from experiments.credit.run import file_sha256
from experiments.lifelong.evaluate import evaluate as evaluate_fork
from experiments.loops.audit_run import verify_files
from experiments.organization import run as native


ROOT = Path(__file__).resolve().parents[2]
FORMAT = "organization-protocol-v1"
DEFAULT_EVIDENCE = ROOT / "docs/evidence/organization-2026-10-30"
DEFAULT_RAW = ROOT / "data/organization/main-v1"
KEYS = ("field_mode", "structural_rule")
VARIANTS = (
    dict(field_mode="plain", structural_rule="passive"),
    dict(field_mode="parallel", structural_rule="passive"),
    dict(field_mode="coupled", structural_rule="passive"),
    dict(field_mode="plain", structural_rule="functional_fixed"),
    dict(field_mode="plain", structural_rule="functional_task"),
    dict(field_mode="plain", structural_rule="functional_magnitude"),
    dict(field_mode="plain", structural_rule="functional_random"),
    dict(field_mode="coupled", structural_rule="functional_fixed"),
    dict(field_mode="coupled", structural_rule="functional_task"),
)
COMMON = dict(architecture="open_shrink", shrink_step=0., schedule="mixed",
              learning="history_replay", coverage="wide")
SEEDS = (0, 1, 2)
_BIRTH_KEYS = {"model_hash", "hidden_hash", "world_hash", "optimizer_hash",
               "encoder_hash", "head_hash", "initial_mask_hash", "initial_nonzero"}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalized(value):
    return json.loads(json.dumps(value))


def execution_files():
    """Explicit execution inventory; acquisition and report development stay separate."""
    names = (
        "organization/__init__.py", "organization/model.py", "organization/test_model.py",
        "organization/run.py", "organization/test_run.py", "organization/plasticity.py",
        "organization/test_plasticity.py", "organization/test_functional_run.py",
        "organization/study.py", "organization/test_study.py",
        "lifelong/__init__.py", "lifelong/world.py", "lifelong/schedule.py", "lifelong/replay.py",
        "lifelong/evaluate.py", "lifelong/queue.py",
        "continuity/run.py", "continuity/model.py", "continuity/world.py", "continuity/evaluate.py",
        "factors/structure.py", "rewiring/plasticity.py", "integration/diagnose.py", "integration/run.py",
        "credit/run.py", "credit/local.py", "credit/batch.py", "credit/windows_runner.py",
        "loops/audit_run.py", "loops/structure.py", "loops/interventions.py",
    )
    return {str((ROOT / "experiments" / name).resolve()): file_sha256(ROOT / "experiments" / name)
            for name in names}


def evaluation_specs(checkpoint_ticks, steps=2048):
    if type(steps) is not int or steps < 1:
        raise ValueError("positive evaluation steps required")
    return [dict(tick=tick, family=family, condition=condition, steps=steps)
            for tick in checkpoint_ticks if tick
            for family in range(4) for condition in ("short", "long_both")]


def _template_checks(config):
    fields = native.NativeConfig.__dataclass_fields__
    if not {"structural_rule", "structural_candidates"}.issubset(fields):
        raise RuntimeError("organization trainer lacks functional structural-policy support")
    if config.field_mode != "plain" or config.structural_rule != "passive":
        raise ValueError("study template must be the matched plain/passive birth")
    if any(getattr(config, key) != value for key, value in COMMON.items()):
        raise ValueError("organization study requires open eligibility, zero shrink and mixed/history/wide")
    if config.structural_candidates != 4 or config.structural_window_updates != 8:
        raise ValueError("organization study fixes four candidates per kind and an eight-update window")


def _branches(template, raw):
    rows = []
    for seed in SEEDS:
        for variant in VARIANTS:
            label = "__".join(variant[key] for key in KEYS)
            config = replace(template, seed=seed, **variant, **COMMON)
            rows.append(dict(id=f"seed_{seed:03d}__{label}", seed=seed, **variant, **COMMON,
                native_config=_normalized(asdict(config)),
                output=str(Path(raw) / "training" / f"seed_{seed:03d}" / label)))
    return rows


def _birth_signature(trainer):
    return dict(model_hash=state_digest(trainer.model.state_dict()),
        hidden_hash=state_digest(trainer.h), world_hash=state_digest(trainer.world.state_dict()),
        optimizer_hash=state_digest(trainer.optimizer.state_dict()),
        encoder_hash=state_digest(trainer.model.encoder.state_dict()),
        head_hash=state_digest(trainer.model.head.state_dict()),
        initial_mask_hash=trainer.birth["initial_mask_hash"],
        initial_nonzero=int(trainer.model.recurrent.detach().count_nonzero()))


def birth_bindings(template):
    """Construct every actual newborn on CPU; compare values rather than config labels."""
    shared, births = {}, {}
    with _preserve_process_state():
        for item in _branches(replace(template, device="cpu"), Path("unused")):
            trainer = native.NativeTrainer(native.NativeConfig(**item["native_config"]))
            trainer._validate_continuation()
            signature = _birth_signature(trainer)
            seed = str(item["seed"])
            if seed in shared and shared[seed] != signature:
                raise ValueError("organization variants differ in newborn parameters, h, world or Adam")
            shared[seed] = signature
            births[item["id"]] = state_digest(trainer.birth)
            half = trainer.config.width // 2
            weight = trainer.model.recurrent.detach()
            if (bool(weight[:half, half:].ne(0).any()) or bool(weight[half:, :half].ne(0).any())
                    or signature["initial_nonzero"] != trainer.config.width * trainer.config.effective_fan_in
                    or trainer.ticks or trainer.hidden_resets or len(trainer.optimizer.state)
                    or bool(trainer.h.ne(0).any())):
                raise ValueError("organization birth is not the declared cold block-diagonal state")
            del trainer
    return dict(shared_by_seed=shared, branch_birth_hashes=births,
        note="Actual CPU newborn values match across fields and structural policies. Operator state and policy config may differ; the block-diagonal birth differs from the old lifelong study.")


def prepare(folder=DEFAULT_EVIDENCE, raw=DEFAULT_RAW, *, template=None, training_device="cuda"):
    """A formal CPU or CUDA run keeps the full budget; template is test-only."""
    folder, raw = Path(folder).resolve(), Path(raw).resolve()
    path = folder / "organization-protocol-v1.json"
    targets = [path] + [folder / f"organization-{kind}-commands-v1.json"
                        for kind in ("training", "evaluation", "baseline")]
    if any(target.exists() for target in targets) or raw.exists():
        raise FileExistsError("organization prepare requires new protocol, manifests and raw root")
    config = native.NativeConfig(device=training_device) if template is None else template
    _template_checks(config)
    sources = execution_files()
    bindings = birth_bindings(config)
    ticks = list(config.checkpoint_ticks)
    steps = 2048 if template is None else 8
    branches = _branches(config, raw)
    plan = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
        sources=sources, raw_root=str(raw), branches=branches, variants=VARIANTS, seeds=SEEDS,
        native_template=_normalized(asdict(config)), mechanical_test=template is not None,
        target_ticks=config.total_ticks, checkpoint_ticks=ticks,
        latest_interval_ticks=32768 if template is None else config.macrocycle_ticks,
        training_device=config.device, evaluation_device="cpu", threads=1, dt=config.dt,
        evaluation_steps=steps, evaluation_specs=evaluation_specs(ticks, steps),
        birth_bindings=bindings, baseline=dict(jobs=0, note="No new or imported baseline jobs; matched plain controls are included among the 27 runs."),
        expected_counts=dict(training_runs=27, full_checkpoints=108, forks_per_branch=24, frozen_forks=648),
        comparison_groups=dict(
            nonlinear_fields=[v for v in VARIANTS if v["structural_rule"] == "passive"],
            structural_selection=[v for v in VARIANTS if v["field_mode"] == "plain" and v["structural_rule"] != "passive"],
            joint_field_structure=[v for v in VARIANTS if v["structural_rule"] in ("functional_fixed", "functional_task")]),
        learning="One averaged live128 and auxiliary suffix128 gradient, then one Adam update. Detached history warmup, persistent live h/world/optimizer and phase changes follow the frozen native protocol.",
        structure="Functional trials use four birth and four exit candidates after each eight-update window, pending until the next real live block and its selected auxiliary history. Dormant Adam moments keep learning. Birth and exit batches have separate acceptance decisions against the equal-weight live/history loss; the exit baseline includes accepted births. Exit trials halve weights, or set them to zero once their magnitude is at most the learning rate. Full-ablation loss is recorded separately. The fixed policy runs the same probes but accepts neither batch. Extra forward work is recorded separately. No global shrink.",
        field_control="All field modes match allocated parameters, persistent state and recurrent matrix MACs. Parallel and coupled also match tanh counts. Two symmetric banks have no assigned memory, timer or task roles.",
        primary="Matched passive field comparison; plain functional selection-policy comparison; plain versus coupled under fixed/task structural rules. First assess main-end acquisition before interpreting withdrawal retention.",
        limits=[
            "The plain newborn is matched within this study and differs from the old lifelong random cross-bank sparse function.",
            "Functional trials use the next actual training block; they are not independent held-out evaluation.",
            "Structural controls match candidate and probe rules, while accepted capacity and wall-time trajectories may diverge.",
            "Match live ticks and stated auxiliary work, then report actual events, structural forward work, edges and wall time separately.",
            "The available state graph already admits feedback cycles; the field comparison tests nonlinear transition organization rather than a graph cycle count.",
            "Frozen forks preserve each source h/world and score the full eleven-action argmax without learning or a cold reset.",
            "Three trained seeds remain the independent repetitions; snapshots, actors and forks do not add seeds.",
        ])
    verify_files(sources)
    if execution_files() != sources:
        raise ValueError("organization sources changed during preparation")
    atomic_json(path, plan)
    frozen = {**sources, str(path): file_sha256(path)}
    jobs = {kind: [] for kind in ("training", "evaluation", "baseline")}
    for item in branches:
        out = Path(item["output"])
        tail = ["--protocol", str(path), "--branch-id", item["id"]]
        prefix = [sys.executable, "-m", "experiments.organization.study"]
        jobs["training"].append(dict(id=item["id"], command=prefix + ["train"] + tail,
            outputs=[str(out / "latest.pt"), str(out / "launch-complete.json")],
            log=str(raw / "logs" / f"train_{item['id']}.log")))
        output = raw / "evaluation" / f"{item['id']}.json"
        jobs["evaluation"].append(dict(id=item["id"], command=prefix + ["evaluate"] + tail +
            ["--checkpoint", str(out / "latest.pt"), "--output", str(output)],
            outputs=[str(output)], log=str(raw / "logs" / f"evaluate_{item['id']}.log")))
    for kind, rows in jobs.items():
        atomic_json(folder / f"organization-{kind}-commands-v1.json",
                    dict(cwd=str(ROOT), file_hashes=frozen, jobs=rows))
    return path


def read_plan(path, identifier):
    path = Path(path).resolve()
    sha, plan = file_sha256(path), _json(path)
    if plan.get("format") != FORMAT or plan.get("threads") != 1 or plan.get("evaluation_device") != "cpu":
        raise ValueError("invalid organization protocol")
    verify_files(plan["sources"])
    if plan["sources"] != execution_files():
        raise ValueError("organization source inventory changed")
    template = native.NativeConfig(**plan["native_template"])
    _template_checks(template)
    if (plan.get("seeds") != list(SEEDS) or plan.get("variants") != list(VARIANTS)
            or plan.get("branches") != _branches(template, Path(plan["raw_root"]).resolve())):
        raise ValueError("foreign organization branch/config/output inventory")
    if (plan["checkpoint_ticks"] != list(template.checkpoint_ticks)
            or plan["target_ticks"] != template.total_ticks or plan["dt"] != template.dt
            or plan["training_device"] != template.device
            or plan["evaluation_specs"] != evaluation_specs(template.checkpoint_ticks, plan["evaluation_steps"])):
        raise ValueError("organization checkpoint/evaluation budget changed")
    mechanical = plan.get("mechanical_test")
    if type(mechanical) is not bool:
        raise ValueError("organization protocol must declare its execution budget")
    if (not mechanical and (plan["native_template"] != _normalized(asdict(native.NativeConfig(device=template.device)))
                            or plan["evaluation_steps"] != 2048)):
        raise ValueError("formal organization budget differs from the declared native defaults")
    expected_interval = template.macrocycle_ticks if mechanical else 32768
    if plan["latest_interval_ticks"] != expected_interval:
        raise ValueError("organization periodic checkpoint budget changed")
    bindings = plan["birth_bindings"]
    if (set(bindings["shared_by_seed"]) != {str(seed) for seed in SEEDS}
            or set(bindings["branch_birth_hashes"]) != {item["id"] for item in plan["branches"]}
            or any(set(row) != _BIRTH_KEYS for row in bindings["shared_by_seed"].values())):
        raise ValueError("incomplete organization newborn binding inventory")
    matches = [item for item in plan["branches"] if item["id"] == identifier]
    if len(matches) != 1:
        raise ValueError("unknown organization branch")
    return plan, matches[0], sha


def normalized_hash(trainer, device):
    value = trainer.semantic_state()
    value["config"]["device"] = device
    return state_digest(value)


def validate(trainer, plan, item, tick=None):
    expected = dict(item["native_config"], device=str(trainer.device))
    if _normalized(asdict(trainer.config)) != expected or trainer.config.shrink_step != 0.:
        raise ValueError("foreign organization native configuration")
    if (trainer.model.field_mode != item["field_mode"] or trainer.model.kind != "learned"
            or trainer.hidden_resets or trainer.ticks % trainer.config.chunk_ticks
            or not 0 <= trainer.ticks <= plan["target_ticks"] or (tick is not None and trainer.ticks != tick)):
        raise ValueError("invalid organization field or progress")
    shared = plan["birth_bindings"]["shared_by_seed"][str(item["seed"])]
    if (trainer.initial_parameter_hash != shared["model_hash"]
            or trainer.birth["initial_mask_hash"] != shared["initial_mask_hash"]
            or state_digest(trainer.birth) != plan["birth_bindings"]["branch_birth_hashes"][item["id"]]):
        raise ValueError("organization birth differs from its matched seed")
    if trainer.ticks == 0 and _birth_signature(trainer) != shared:
        raise ValueError("organization newborn h/world/Adam values differ from the declared birth")
    if item["structural_rule"] == "passive":
        if trainer.operator.config.effective_shrink_step != 0.:
            raise ValueError("passive organization operator still shrinks")
    elif trainer.operator.config.mode != item["structural_rule"]:
        raise ValueError("organization structural operator differs from its declared rule")
    trainer._validate_continuation()


def verify_end(path, identifier, sha):
    if file_sha256(path) != sha:
        raise ValueError("organization protocol changed")
    read_plan(path, identifier)


def _snapshot(out, trainer, plan, item):
    cp = Path(out) / "checkpoints" / f"tick_{trainer.ticks:06d}.pt"
    if cp.exists():
        expected = normalized_hash(trainer, plan["training_device"])
        with _preserve_process_state():
            saved = native.NativeTrainer.from_checkpoint(cp, device="cpu")
            validate(saved, plan, item, trainer.ticks)
            if normalized_hash(saved, plan["training_device"]) != expected:
                raise ValueError("refusing to overwrite different organization snapshot")
    else:
        trainer.save_checkpoint(cp)


def completion(plan, item, sha):
    out = Path(item["output"])
    done = _json(out / "launch-complete.json")
    if (done.get("format") != "lifelong-launch-complete-v1"
            or done.get("study_format") != "organization-launch-complete-v1" or done.get("branch") != item
            or done.get("protocol_sha256") != sha or done.get("stage_ticks") != plan["target_ticks"]
            or done.get("target") != plan["target_ticks"]
            or done.get("checkpoint_sha256") != file_sha256(out / "latest.pt")
            or set(done.get("checkpoints", {})) != {str(tick) for tick in plan["checkpoint_ticks"]}):
        raise ValueError("invalid organization completion")
    for tick, entry in done["checkpoints"].items():
        cp = out / "checkpoints" / f"tick_{int(tick):06d}.pt"
        if entry.get("path") != str(cp) or file_sha256(cp) != entry["sha256"]:
            raise ValueError("organization snapshot bytes changed")
    with _preserve_process_state():
        final = native.NativeTrainer.from_checkpoint(out / "latest.pt", device="cpu")
        validate(final, plan, item, plan["target_ticks"])
        if (done.get("birth_hash") != state_digest(final.birth)
                or done.get("initial_parameter_hash") != final.initial_parameter_hash
                or normalized_hash(final, plan["training_device"]) != done["checkpoints"][str(plan["target_ticks"])]["semantic_hash"]):
            raise ValueError("organization latest differs from its final snapshot or birth binding")
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
    binding = dict(format="organization-launch-start-v1", branch=item, protocol_sha256=sha)
    if start.exists():
        if _json(start) != binding:
            raise ValueError("foreign organization start binding")
    else:
        if out.exists():
            raise ValueError("refusing unbound existing organization output")
        atomic_json(start, binding)
    latest, birth_cp = out / "latest.pt", out / "checkpoints/tick_000000.pt"
    resume = latest if latest.exists() else birth_cp if birth_cp.exists() else None
    if resume is None:
        if out.exists() and any(out.iterdir()):
            raise ValueError("existing organization files without a resumable checkpoint")
        trainer = native.NativeTrainer(native.NativeConfig(**item["native_config"]))
    else:
        trainer = native.NativeTrainer.from_checkpoint(resume, device=plan["training_device"])
    validate(trainer, plan, item)
    target = plan["target_ticks"] if stop_at is None else stop_at
    if type(target) is not int or not trainer.ticks <= target <= plan["target_ticks"] or target % trainer.config.chunk_ticks:
        raise ValueError("invalid organization stopping tick")
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
        study_format="organization-launch-complete-v1", branch=item, protocol_sha256=sha,
        target=plan["target_ticks"], stage_ticks=trainer.ticks, checkpoint_sha256=file_sha256(latest),
        checkpoints=entries, hidden_resets=0, initial_parameter_hash=trainer.initial_parameter_hash,
        birth_hash=state_digest(trainer.birth), work=trainer.report()["work"]))


def evaluate(path, identifier, checkpoint, output):
    torch.set_num_threads(1)
    plan, item, sha = read_plan(path, identifier)
    out, output = Path(item["output"]), Path(output).resolve()
    expected_output = Path(plan["raw_root"]) / "evaluation" / f"{identifier}.json"
    if (Path(checkpoint).resolve() != (out / "latest.pt").resolve()
            or output != expected_output.resolve() or output.exists()):
        raise ValueError("wrong organization evaluation path or existing output")
    done = completion(plan, item, sha)
    records, states = [], {}
    with _preserve_process_state():
        for tick in plan["checkpoint_ticks"][1:]:
            entry = done["checkpoints"][str(tick)]
            cp = Path(entry["path"])
            trainer = native.NativeTrainer.from_checkpoint(cp, device="cpu")
            validate(trainer, plan, item, tick)
            before, normalized = state_digest(trainer.semantic_state()), normalized_hash(trainer, plan["training_device"])
            if normalized != entry["semantic_hash"]:
                raise ValueError("organization snapshot semantics changed")
            states[str(tick)] = dict(checkpoint=str(cp), checkpoint_sha256=entry["sha256"],
                state_hash=before, training_device_normalized_state_hash=normalized,
                family_recency=trainer.report()["family_recency"],
                field_mode=trainer.model.field_mode, structural_rule=trainer.config.structural_rule)
            for spec in (spec for spec in plan["evaluation_specs"] if spec["tick"] == tick):
                record = evaluate_fork(trainer.model, trainer.world, trainer.h, family=spec["family"],
                    condition=spec["condition"], steps=spec["steps"], dt=plan["dt"], device="cpu")
                if (any(record[key] for key in ("world_resets", "hidden_resets", "parameter_updates"))
                        or record["model"] != trainer.model.config()):
                    raise AssertionError("organization evaluation changed state, learned or lost its field configuration")
                record["probe"] = spec
                record["organization"] = dict(field_mode=trainer.model.field_mode,
                    structural_rule=trainer.config.structural_rule, structural_updates_during_fork=0)
                records.append(record)
            if state_digest(trainer.semantic_state()) != before or file_sha256(cp) != entry["sha256"]:
                raise AssertionError("organization frozen evaluation changed its source")
    verify_end(path, identifier, sha)
    atomic_json(output, dict(format="organization-evaluation-v1", branch=item,
        protocol_sha256=sha, source_unchanged=True, states=states, records=records,
        birth_hash=done["birth_hash"], initial_parameter_hash=done["initial_parameter_hash"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_EVIDENCE / "organization-protocol-v1.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--branch-id")
    parser.add_argument("--stop-at", type=int)
    parser.add_argument("--training-device", default="cuda", help="Formal preparation device; does not reduce any budget.")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.mode == "prepare":
        print(prepare(args.output, args.raw, training_device=args.training_device))
    elif args.mode == "train":
        train(args.protocol, args.branch_id, stop_at=args.stop_at)
    else:
        evaluate(args.protocol, args.branch_id, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
