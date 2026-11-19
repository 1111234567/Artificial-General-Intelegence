"""Source-bound continuous graph study, including numerical-failure terminals.

Formal preparation declares fourteen variants and three trained seeds.  A
mechanical budget must be requested explicitly; passing a template never changes
evaluation length.  Failed numerical trajectories have no comparable endpoint
score, and are collected as terminals rather than silently dropped or replaced.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import torch

from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import state_digest
from experiments.lifelong.evaluate import evaluate as evaluate_fork
from experiments.undirected import run as native
from experiments.undirected.model import contextual_input_scale
from experiments.undirected.ledger import LiveLedger, read_ledger
from experiments.undirected.queue import atomic_json, file_sha256, verify_files


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv/Scripts/python.exe"
FORMAT = "undirected-protocol-v1"
TERMINAL_FORMAT = "undirected-branch-terminal-v1"
DEFAULT_EVIDENCE = ROOT / "docs/evidence/undirected-2026-10-30"
DEFAULT_RAW = ROOT / "data/undirected/main-v1"
SEEDS = (0, 1, 2)
KEYS = ("interface", "mode", "credit", "structure", "course")
VARIANTS = tuple(
    dict(interface=interface, mode=mode, credit="recorded512", structure="fixed", course="warmup")
    for interface in ("core", "ports") for mode in ("undirected", "untied")
) + tuple(
    dict(interface="ports", mode="undirected", credit=credit, structure="fixed", course="warmup")
    for credit in ("tbptt128", "rtrl", "rtrl_reset")
) + (
    dict(interface="ports", mode="untied", credit="rtrl", structure="fixed", course="warmup"),
) + tuple(
    dict(interface="ports", mode="undirected", credit="rtrl", structure=structure, course="warmup")
    for structure in ("fixed_probe", "grow", "retire", "both")
) + tuple(
    dict(interface="ports", mode="undirected", credit=credit, structure="fixed", course="wide")
    for credit in ("tbptt128", "rtrl")
)
REPORTING_POLICY = {
    "independent_repetitions": 3,
    "endpoint_mean": "Report only when all three declared seeds succeed at the requested endpoint.",
    "numerical_failure": "Preserve and report the failed seed; no zero score, replacement seed, sensitivity clipping or reset.",
    "missing_or_execution_failure": "Not a scientific numerical failure; repair execution and resume the same source-bound branch.",
    "forks": "Actors, conditions and snapshots do not increase the number of independent trained seeds.",
}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalized(value):
    return json.loads(json.dumps(value, allow_nan=False))


def formal_template():
    return native.NativeConfig(interface="ports", mode="undirected", credit="rtrl", structure="fixed", course="warmup",
        width=80, batch_size=8, n_jobs=2, initial_degree=16, birth_gain=.6, dtype="float64", device="cpu",
        world_device="cpu", cpu_threads=1, chunk_ticks=128, main_ticks=196608, warmup_ticks=65536,
        withdrawal_ticks=32768, return_ticks=8192)


def execution_files():
    """Pin new execution modules/tests, the live ledger and old dependencies.

    Report, plotting and future analysis files are intentionally outside this
    inventory.  This is an explicit list, never a glob over a developing folder.
    """
    names = (
        "undirected/__init__.py", "undirected/model.py", "undirected/test_model.py",
        "undirected/credit.py", "undirected/test_credit.py", "undirected/plasticity.py", "undirected/test_plasticity.py",
        "undirected/run.py", "undirected/test_run.py", "undirected/study.py", "undirected/test_study.py",
        "undirected/queue.py", "undirected/test_queue.py",
        "undirected/ledger.py", "undirected/test_ledger.py",
        "lifelong/__init__.py", "lifelong/world.py", "lifelong/schedule.py", "lifelong/replay.py", "lifelong/evaluate.py",
        "continuity/__init__.py", "continuity/run.py", "continuity/model.py", "continuity/world.py", "continuity/evaluate.py",
        "integration/diagnose.py",  # the frozen observer reports this provenance-file hash
    )
    return {str((ROOT / "experiments" / name).resolve()): file_sha256(ROOT / "experiments" / name) for name in names}


def evaluation_specs(checkpoint_ticks, steps=2048):
    if type(steps) is not int or steps < 1:
        raise ValueError("positive evaluation steps required")
    return [dict(tick=tick, family=family, condition=condition, steps=steps)
            for tick in checkpoint_ticks if tick for family in range(4) for condition in ("short", "long_both")]


def _template_checks(config, mechanical):
    if not isinstance(config, native.NativeConfig) or type(mechanical) is not bool:
        raise ValueError("explicit NativeConfig and mechanical flag required")
    if any(getattr(config, key) != getattr(formal_template(), key) for key in KEYS + ("seed",)):
        raise ValueError("template must be the canonical ports/undirected/rtrl/fixed/warmup seed-zero branch")
    if (config.device, config.world_device, config.dtype, config.cpu_threads) != ("cpu", "cpu", "float64", 1):
        raise ValueError("this study uses CPU float64 with one thread in every arm")
    if config.pulse_gain != .25:
        raise ValueError("matched interfaces require the declared .25 observation pulse")
    if not mechanical and config != formal_template():
        raise ValueError("formal configuration differs from the declared budget; smaller templates require mechanical=True")
    if len(config.snapshot_ticks()) != 6:
        raise ValueError("study requires six distinct aligned snapshots including birth")


def _branches(template, raw):
    return [dict(id=f"seed_{seed:03d}__" + "__".join(variant[key] for key in KEYS), seed=seed, **variant,
                 native_config=_normalized(asdict(replace(template, seed=seed, **variant))),
                 output=str(Path(raw) / "training" / f"seed_{seed:03d}" / "__".join(variant[key] for key in KEYS)))
            for seed in SEEDS for variant in VARIANTS]


def _function_signature(model):
    """Nonzero-state, multi-step functional probe plus explicit port invariants."""
    dtype, device = next(model.parameters()).dtype, next(model.parameters()).device
    initial = torch.linspace(-.17, .19, 2 * model.width, dtype=dtype, device=device).reshape(2, model.width)
    observations = torch.linspace(-.23, .31, 5 * 2 * model.input_size, dtype=dtype, device=device).reshape(5, 2, model.input_size)
    elapsed = (0., .125, .25, .375, .5)
    scale = torch.as_tensor(contextual_input_scale(model.n_jobs), device=device, dtype=dtype)
    input_ports = torch.arange(model.input_size, device=device)
    output_ports = torch.arange(model.input_size, model.input_size + model.n_jobs * model.n_actions, device=device)
    with torch.no_grad():
        if hasattr(model, "encoder"):
            encoder = initial.new_zeros(model.width, model.input_size)
            encoder[input_ports, torch.arange(model.input_size, device=device)] = .25
            head = initial.new_zeros(model.n_jobs * model.n_actions, model.width)
            head[torch.arange(len(output_ports), device=device), output_ports] = 1.
            if (model.readout != "linear" or not torch.equal(model.encoder.weight, encoder)
                    or not torch.equal(model.head.weight, head) or bool(model.head.bias.ne(0).any())):
                raise ValueError("learned interfaces are not born as the exact matched linear selectors")
        else:
            config = model.config()
            if (config["pulse_gain"] != .25 or config["input_ports"] != input_ports.tolist()
                    or config["output_ports"] != output_ports.tolist()):
                raise ValueError("fixed graph port identities differ")
        normalized = model.normalized_observation(observations[0])
        if not torch.equal(normalized, observations[0] * scale):
            raise ValueError("birth observation units differ from duration/16 and window/2.5")
        expected_readout = initial[..., output_ports]
        if not torch.equal(model.read_state(initial), expected_readout):
            raise ValueError("birth readout is not the same raw output-port selector")
        _, pulse_h = model(observations[0], initial, 0.)
        pulse = initial.new_zeros(initial.shape)
        pulse[..., input_ports] = .25 * normalized
        if not torch.equal(pulse_h, initial + pulse):
            raise ValueError("birth input is not the same additive .25 port pulse")
        h, outputs, states = initial, [], []
        for observation, dt in zip(observations, elapsed):
            output, h = model(observation, h, dt)
            outputs.append(output)
            states.append(h)
    return {
        "function_hash": state_digest(dict(initial_h=initial, observations=observations, elapsed=elapsed,
                                            logits=torch.stack(outputs), hidden=torch.stack(states))),
        "scale_hash": state_digest(scale), "readout_selector_hash": state_digest(output_ports),
        "input_selector_hash": state_digest(input_ports), "pulse_gain": .25,
        "probe": "Five deterministic observations with nonzero initial h, dt=0/.125/.25/.375/.5; complete logits and hidden trajectory.",
    }


def _birth_signature(trainer):
    trainer._validate_continuation()
    matrix = trainer.model.recurrent_matrix().detach()
    mask = trainer.operator.initial_mask.detach().cpu()
    indices = trainer.model.pair_index.detach().cpu()
    degrees = torch.bincount(indices[:, mask].reshape(-1), minlength=trainer.config.width)
    if (not torch.equal(matrix, matrix.T) or bool(matrix.diag().ne(0).any())
            or not bool(degrees.eq(trainer.config.initial_degree).all())
            or trainer.ticks or trainer.hidden_resets or trainer.optimizer.state or bool(trainer.h.ne(0).any())
            or trainer.credit.updates or trainer.credit.sensitivity is not None or trainer.credit.tape
            or (trainer.model.mode == "untied" and bool(trainer.model.edge_skew.ne(0).any()))):
        raise ValueError("actual newborn is not the matched sparse zero-state graph")
    return dict(model_hash=state_digest(trainer.model.state_dict()), matrix_hash=state_digest(matrix),
        hidden_hash=state_digest(trainer.h), world_hash=state_digest(trainer.world.state_dict()),
        optimizer_hash=state_digest(trainer.optimizer.state_dict()), birth_hash=state_digest(trainer.birth),
        initial_mask_hash=state_digest(mask), model_config=_normalized(trainer.model.config()),
        parameter_layout=_normalized(trainer.semantic_state()["parameter_layout"]),
        total_parameters=sum(parameter.numel() for parameter in trainer.model.parameters()),
        dynamic_parameters=trainer.credit.dynamic_parameter_count, dynamic_names=list(trainer.credit.dynamic_names),
        local_names=list(trainer.credit.local_names), **_function_signature(trainer.model))


def birth_bindings(template):
    shared, worlds, signatures = {}, {}, {}
    threads = torch.get_num_threads()
    try:
        with _preserve_process_state():
            for item in _branches(template, Path("unused")):
                trainer = native.NativeTrainer(native.NativeConfig(**item["native_config"]))
                signature = _birth_signature(trainer)
                seed = str(item["seed"])
                physical = {key: signature[key] for key in ("matrix_hash", "hidden_hash", "initial_mask_hash", "function_hash",
                                                           "scale_hash", "readout_selector_hash", "input_selector_hash", "pulse_gain")}
                world_key = f"{seed}/{item['course']}"
                if seed in shared and shared[seed] != physical:
                    raise ValueError("same-seed variants differ in physical W, h, ports or complete birth function")
                if world_key in worlds and worlds[world_key] != signature["world_hash"]:
                    raise ValueError("same-seed/course variants differ in complete world birth")
                shared[seed], worlds[world_key], signatures[item["id"]] = physical, signature["world_hash"], signature
                del trainer
    finally:
        torch.set_num_threads(threads)
    return dict(shared_function_by_seed=shared, world_by_seed_course=worlds, by_branch=signatures,
        note="Physical W/h and the complete nonzero-state birth function match across interfaces and modes. "
             "Model/optimizer/layout hashes remain branch-specific. Complete worlds match within seed/course; "
             "wide and warmup intentionally start with different future-request regimes.")


def prepare(folder=DEFAULT_EVIDENCE, raw=DEFAULT_RAW, *, template=None, mechanical=False, evaluation_steps=2048):
    folder, raw = Path(folder).resolve(), Path(raw).resolve()
    path = folder / "protocol-v1.json"
    targets = [path] + [folder / f"{kind}-commands-v1.json" for kind in ("training", "evaluation", "baseline")]
    if any(target.exists() for target in targets) or raw.exists():
        raise FileExistsError("prepare requires new protocol, manifests and raw root")
    config = formal_template() if template is None else template
    _template_checks(config, mechanical)
    if type(evaluation_steps) is not int or evaluation_steps < 1 or (not mechanical and evaluation_steps != 2048):
        raise ValueError("formal evaluation is 2048 steps; mechanical evaluation length must be explicit")
    if not PYTHON.is_file():
        raise FileNotFoundError("the declared project .venv Python is unavailable")
    sources = execution_files()
    bindings = birth_bindings(config)
    branches = _branches(config, raw)
    ticks = list(config.snapshot_ticks())
    specs = evaluation_specs(ticks, evaluation_steps)
    plan = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(), sources=sources,
        raw_root=str(raw), branches=branches, variants=list(VARIANTS), seeds=list(SEEDS),
        native_template=_normalized(asdict(config)), mechanical_test=mechanical,
        target_ticks=config.total_ticks, checkpoint_ticks=ticks, latest_interval_ticks=32768,
        training_device="cpu", evaluation_device="cpu", dtype="float64", threads=1, dt=config.dt,
        python_executable=str(PYTHON.resolve()), environment=dict(python=platform.python_version(), torch=torch.__version__),
        evaluation_steps=evaluation_steps, evaluation_specs=specs, birth_bindings=bindings,
        expected_counts=dict(training_runs=len(branches), full_checkpoints=len(branches) * len(ticks),
                             forks_per_branch=len(specs), frozen_forks=len(branches) * len(specs)),
        baseline=dict(jobs=0, note="All matched controls are among the fourteen native variants; no imported baseline jobs."),
        reporting_policy=REPORTING_POLICY,
        learning="Every 128 real live ticks give one pre-update live gradient plus one equally weighted auxiliary gradient, "
                 "followed by one Adam update. Auxiliary warmup128 is detached and only its suffix128 is differentiated. "
                 "Only live RTRL has no fixed time-credit truncation; recorded512 remains a finite window.",
        numerical_failure="A typed runtime numerical failure becomes an immutable numerical_failure terminal and an evaluation "
                          "with empty records and endpoint_comparable=false. Execution/resource/code errors remain nonzero and resumable.",
        limits=["The shared-offset derivative stops through realized optimizer/structure changes, actions and the world.",
                "All live h, world, receipts, optimizer, S and history continue across course/withdrawal boundaries.",
                "Actual compute, parameter layouts, real request exposure and graph capacity are reported separately.",
                "Frozen evaluation changes future request draws only, preserving complete inherited state."])
    verify_files(sources)
    if execution_files() != sources:
        raise ValueError("study sources changed during preparation")
    atomic_json(path, plan)
    frozen = {**sources, str(path): file_sha256(path)}
    jobs = {kind: [] for kind in ("training", "evaluation", "baseline")}
    for item in branches:
        out = Path(item["output"])
        common = dict(id=item["id"], branch_id=item["id"], branch=item, terminal=str(out / "terminal.json"),
                      protocol_sha256=file_sha256(path), target_ticks=plan["target_ticks"])
        prefix = [str(PYTHON.resolve()), "-m", "experiments.undirected.study"]
        tail = ["--protocol", str(path), "--branch-id", item["id"]]
        jobs["training"].append(dict(**common, command=prefix + ["train"] + tail,
            outputs=[str(out / "checkpoint_latest.pt"), str(out / "terminal.json")],
            log=str(raw / "logs" / f"train_{item['id']}.log")))
        output = raw / "evaluation" / f"{item['id']}.json"
        jobs["evaluation"].append(dict(**common, command=prefix + ["evaluate"] + tail +
            ["--checkpoint", str(out / "checkpoint_latest.pt"), "--output", str(output)],
            outputs=[str(output)], log=str(raw / "logs" / f"evaluate_{item['id']}.log")))
    for kind, rows in jobs.items():
        atomic_json(folder / f"{kind}-commands-v1.json", dict(format="undirected-command-plan-v1", kind=kind,
            cwd=str(ROOT), protocol=str(path), protocol_sha256=file_sha256(path), file_hashes=frozen, jobs=rows))
    return path


def read_plan(path, identifier):
    path = Path(path).resolve()
    plan, sha = _json(path), file_sha256(path)
    if (plan.get("format") != FORMAT or plan.get("training_device") != "cpu" or plan.get("evaluation_device") != "cpu"
            or plan.get("threads") != 1 or plan.get("dtype") != "float64"):
        raise ValueError("invalid study device/format")
    verify_files(plan["sources"])
    if plan["sources"] != execution_files():
        raise ValueError("study source inventory changed")
    mechanical = plan.get("mechanical_test")
    template = native.NativeConfig(**plan["native_template"])
    _template_checks(template, mechanical)
    if (plan.get("seeds") != list(SEEDS) or plan.get("variants") != list(VARIANTS)
            or plan.get("branches") != _branches(template, Path(plan["raw_root"]).resolve())):
        raise ValueError("foreign branch/config/output inventory")
    if (plan.get("checkpoint_ticks") != list(template.snapshot_ticks()) or plan.get("target_ticks") != template.total_ticks
            or plan.get("dt") != template.dt or plan.get("latest_interval_ticks") != 32768
            or plan.get("evaluation_specs") != evaluation_specs(template.snapshot_ticks(), plan["evaluation_steps"])
            or (not mechanical and plan["evaluation_steps"] != 2048)
            or plan.get("python_executable") != str(PYTHON.resolve())
            or plan.get("environment") != dict(python=platform.python_version(), torch=torch.__version__)
            or plan.get("reporting_policy") != REPORTING_POLICY):
        raise ValueError("study runtime/evaluation budget or reporting policy changed")
    counts = dict(training_runs=42, full_checkpoints=252, forks_per_branch=40, frozen_forks=1680)
    if plan.get("expected_counts") != counts:
        raise ValueError("incomplete declared experimental inventory")
    bindings = plan["birth_bindings"]
    if (set(bindings["shared_function_by_seed"]) != {str(seed) for seed in SEEDS}
            or set(bindings["world_by_seed_course"]) != {f"{seed}/{course}" for seed in SEEDS for course in ("wide", "warmup")}
            or set(bindings["by_branch"]) != {item["id"] for item in plan["branches"]}):
        raise ValueError("incomplete actual newborn bindings")
    for item in plan["branches"]:
        signature = bindings["by_branch"][item["id"]]
        shared = bindings["shared_function_by_seed"][str(item["seed"])]
        if (any(signature[key] != value for key, value in shared.items())
                or signature["world_hash"] != bindings["world_by_seed_course"][f"{item['seed']}/{item['course']}"]):
            raise ValueError("birth matches were amended")
    matches = [item for item in plan["branches"] if item["id"] == identifier]
    if len(matches) != 1:
        raise ValueError("unknown study branch")
    return plan, matches[0], sha


def validate(trainer, plan, item, tick=None):
    if (_normalized(asdict(trainer.config)) != item["native_config"] or trainer.hidden_resets
            or trainer.ticks % trainer.config.chunk_ticks or not 0 <= trainer.ticks <= plan["target_ticks"]
            or (tick is not None and trainer.ticks != tick)):
        raise ValueError("foreign native configuration or progress")
    expected = plan["birth_bindings"]["by_branch"][item["id"]]
    if (state_digest(trainer.birth) != expected["birth_hash"]
            or trainer.birth["initial_parameter_hash"] != expected["model_hash"]
            or trainer.birth["initial_hidden_hash"] != expected["hidden_hash"]
            or trainer.birth["initial_world_hash"] != expected["world_hash"]
            or trainer.birth["initial_optimizer_hash"] != expected["optimizer_hash"]
            or _normalized(trainer.semantic_state()["parameter_layout"]) != expected["parameter_layout"]
            or list(trainer.credit.dynamic_names) != expected["dynamic_names"]
            or list(trainer.credit.local_names) != expected["local_names"]):
        raise ValueError("branch newborn/layout provenance differs")
    if trainer.ticks == 0 and _birth_signature(trainer) != expected:
        raise ValueError("newborn values or complete function differ from their declaration")
    trainer._validate_continuation()


def verify_end(path, identifier, sha):
    if file_sha256(path) != sha:
        raise ValueError("protocol changed during execution")
    read_plan(path, identifier)


def _snapshot(out, trainer, plan, item, sha):
    """Canonical snapshots are immutable, separate from runner recovery files."""
    cp = Path(out) / "checkpoints" / f"tick_{trainer.ticks:06d}.pt"
    index_path = Path(out) / "snapshots.json"
    index = _json(index_path) if index_path.exists() else dict(protocol_sha256=sha, branch_id=item["id"], entries={})
    if index.get("protocol_sha256") != sha or index.get("branch_id") != item["id"]:
        raise ValueError("foreign snapshot index")
    expected = state_digest(trainer.semantic_state())
    if cp.exists():
        with _preserve_process_state():
            saved = native.NativeTrainer.from_checkpoint(cp, device="cpu")
            validate(saved, plan, item, trainer.ticks)
            if state_digest(saved.semantic_state()) != expected:
                raise ValueError("refusing to overwrite a different immutable snapshot")
    else:
        trainer.save_checkpoint(cp)
    entry = dict(path=str(cp.resolve()), sha256=file_sha256(cp), semantic_hash=expected)
    key = str(trainer.ticks)
    if key in index["entries"] and index["entries"][key] != entry:
        raise ValueError("immutable snapshot bytes changed")
    index["entries"][key] = entry
    atomic_json(index_path, index)
    return entry


def _entries(out, plan, item, sha):
    index = _json(Path(out) / "snapshots.json")
    if index.get("protocol_sha256") != sha or index.get("branch_id") != item["id"]:
        raise ValueError("foreign snapshot index")
    for key, entry in index["entries"].items():
        cp = Path(out) / "checkpoints" / f"tick_{int(key):06d}.pt"
        if int(key) not in plan["checkpoint_ticks"] or entry["path"] != str(cp.resolve()) or file_sha256(cp) != entry["sha256"]:
            raise ValueError("snapshot inventory or bytes changed")
    return index["entries"]


def ledger_binding(item, sha):
    return dict(protocol_sha256=sha, branch=item,
                recording_convention="committed-live-block-pre-parameter-trajectory-v1")


def _ledger_summary(out, item, sha, updates):
    state = read_ledger(Path(out) / "live_records", expected_binding=ledger_binding(item, sha))
    entries = state["entries"]
    if len(entries) != updates:
        raise ValueError("terminal ledger omits successful history or contains unmatched ahead records")
    return {key: state[key] for key in ("binding_hash", "binding_path", "binding_sha256", "manifest_path", "manifest_sha256")} | dict(
        records=len(entries), last_tick=entries[-1]["end_tick"] if entries else 0,
        compressed_bytes=sum(entry["compressed_bytes"] for entry in entries),
        uncompressed_bytes=sum(entry["uncompressed_bytes"] for entry in entries))


def completion(plan, item, sha):
    out = Path(item["output"])
    terminal = _json(out / "terminal.json")
    status, last = terminal.get("status"), terminal.get("last_tick")
    success = status == "success"
    if (terminal.get("format") != TERMINAL_FORMAT or terminal.get("branch") != item or terminal.get("protocol_sha256") != sha
            or terminal.get("target_ticks") != plan["target_ticks"] or status not in ("success", "numerical_failure")
            or type(last) is not int or not 0 <= last <= plan["target_ticks"]
            or (success and last != plan["target_ticks"]) or (not success and last >= plan["target_ticks"])
            or terminal.get("completed") is not success or terminal.get("endpoint_comparable") is not success):
        raise ValueError("untrusted study terminal")
    cp = out / ("checkpoint_latest.pt" if success else "checkpoint_last_complete.pt")
    if terminal.get("checkpoint_path") != str(cp.resolve()) or terminal.get("checkpoint_sha256") != file_sha256(cp):
        raise ValueError("terminal checkpoint bytes changed")
    entries = _entries(out, plan, item, sha)
    if terminal.get("checkpoints") != entries or (success and set(entries) != {str(tick) for tick in plan["checkpoint_ticks"]}):
        raise ValueError("terminal snapshot inventory differs")
    if not success:
        failure = terminal.get("failure")
        failure_path = out / "failure.json"
        if (not isinstance(failure, dict) or failure.get("path") != str(failure_path.resolve())
                or failure.get("sha256") != file_sha256(failure_path)):
            raise ValueError("untrusted numerical failure binding")
        detail = _json(failure_path)
        if (detail.get("format") != "undirected-training-failure-v1" or detail.get("status") != "algorithm_numerical_failure"
                or detail.get("exception_type") != "FloatingPointError" or detail.get("config") != item["native_config"]
                or detail.get("last_complete_tick") != last or detail.get("last_complete_checkpoint") != str(cp.resolve())
                or detail.get("declared_lifetime_ticks") != plan["target_ticks"]):
            raise ValueError("untrusted numerical failure classification/provenance")
    elif terminal.get("failure") is not None:
        raise ValueError("successful endpoint cannot claim an algorithm failure")
    with _preserve_process_state():
        trainer = native.NativeTrainer.from_checkpoint(cp, device="cpu")
        validate(trainer, plan, item, last)
        if terminal.get("checkpoint_semantic_hash") != state_digest(trainer.semantic_state()):
            raise ValueError("terminal checkpoint semantic mismatch")
    if terminal.get("ledger") != _ledger_summary(out, item, sha, last // item["native_config"]["chunk_ticks"]):
        raise ValueError("terminal live ledger binding changed")
    return terminal


def _terminal(out, trainer, plan, item, sha, *, numerical=False):
    cp = Path(out) / ("checkpoint_last_complete.pt" if numerical else "checkpoint_latest.pt")
    failure = Path(out) / "failure.json"
    value = dict(format=TERMINAL_FORMAT, branch=item, protocol_sha256=sha,
        status="numerical_failure" if numerical else "success", target_ticks=plan["target_ticks"], last_tick=trainer.ticks,
        completed=not numerical, endpoint_comparable=not numerical,
        checkpoint_path=str(cp.resolve()), checkpoint_sha256=file_sha256(cp),
        checkpoint_semantic_hash=state_digest(trainer.semantic_state()), checkpoints=_entries(out, plan, item, sha),
        failure=dict(path=str(failure.resolve()), sha256=file_sha256(failure)) if numerical else None,
        ledger=_ledger_summary(out, item, sha, trainer.updates),
        birth_hash=state_digest(trainer.birth), work=deepcopy(trainer.work),
        note="Numerical failure is an observed branch outcome, not a zero behavioral score." if numerical else "Declared lifetime completed.")
    atomic_json(Path(out) / "terminal.json", value)
    return completion(plan, item, sha)


def train(path, identifier, *, stop_at=None):
    torch.set_num_threads(1)
    plan, item, sha = read_plan(path, identifier)
    out = Path(item["output"])
    if (out / "terminal.json").exists():
        result = completion(plan, item, sha)
        verify_end(path, identifier, sha)
        return result
    start = out.with_name(out.name + ".launch-start.json")
    binding = dict(format="undirected-launch-start-v1", branch=item, protocol_sha256=sha)
    if start.exists():
        if _json(start) != binding:
            raise ValueError("foreign study start binding")
    else:
        if out.exists():
            raise ValueError("refusing an unbound existing output")
        atomic_json(start, binding)
    config = native.NativeConfig(**item["native_config"])
    candidates = [cp for cp in (out / "checkpoint_latest.pt", out / "checkpoint_last_complete.pt") if cp.exists()]
    if candidates:
        loaded = []
        for cp in candidates:
            trainer = native.NativeTrainer.from_checkpoint(cp, device="cpu")
            validate(trainer, plan, item)
            loaded.append((trainer.ticks, cp))
        _, resume = max(loaded, key=lambda row: row[0])
        # Checkpoint loading restores process RNGs; the last iterated candidate
        # need not be the selected newest one.
        trainer = native.NativeTrainer.from_checkpoint(resume, device="cpu")
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError("existing files have no complete resumable checkpoint")
        trainer = native.NativeTrainer(config)
        validate(trainer, plan, item, 0)
        out.mkdir(parents=True, exist_ok=True)
        resume = out / "checkpoint_latest.pt"
        trainer.save_checkpoint(resume)
    target = plan["target_ticks"] if stop_at is None else stop_at
    if type(target) is not int or not trainer.ticks <= target <= plan["target_ticks"] or target % config.chunk_ticks:
        raise ValueError("invalid study stopping tick")
    writer = LiveLedger(out / "live_records", ledger_binding(item, sha))
    # A final-update callback may have failed after the complete checkpoint was
    # saved.  Reconcile its last record even if no further native.train call is
    # needed; older missing records are never invented from a short checkpoint.
    writer(trainer)
    # A callback failure can leave last_complete newer than latest, including
    # at the final target where no further native.train call would promote it.
    trainer.save_checkpoint(out / "checkpoint_latest.pt")
    resume = out / "checkpoint_latest.pt"
    if (out / "snapshots.json").exists():
        _entries(out, plan, item, sha)
    if trainer.ticks in plan["checkpoint_ticks"]:
        _snapshot(out, trainer, plan, item, sha)
    stops = sorted({tick for tick in plan["checkpoint_ticks"] if trainer.ticks < tick <= target} | {target})
    for stop in stops:
        if stop == trainer.ticks:
            continue
        try:
            trainer = native.train(config, out, resume=resume, stop_at=stop, on_update=writer)
        except Exception as error:
            # The runner writes failure.json and the last *completed* boundary.
            # Only typed numerical exceptions become scientific outcomes.  Do
            # not classify arbitrary code/resource errors by a message substring.
            failure_path = out / "failure.json"
            if not isinstance(error, FloatingPointError) or not failure_path.exists():
                raise
            failure = _json(failure_path)
            if failure.get("status") != "algorithm_numerical_failure" or failure.get("stage", "learning_update") != "learning_update":
                raise
            last = out / "checkpoint_last_complete.pt"
            if (failure.get("format") != "undirected-training-failure-v1"
                    or failure.get("status") != "algorithm_numerical_failure"
                    or failure.get("config") != item["native_config"]
                    or failure.get("exception_type") != type(error).__name__ or failure.get("exception") != str(error)
                    or failure.get("last_complete_checkpoint") != str(last.resolve())
                    or failure.get("declared_target_tick") != stop or failure.get("declared_lifetime_ticks") != plan["target_ticks"]):
                raise ValueError("runner numerical failure provenance differs") from error
            trainer = native.NativeTrainer.from_checkpoint(last, device="cpu")
            validate(trainer, plan, item, failure["last_complete_tick"])
            if trainer.ticks >= plan["target_ticks"]:
                raise ValueError("numerical failure cannot claim a complete endpoint") from error
            verify_end(path, identifier, sha)
            return _terminal(out, trainer, plan, item, sha, numerical=True)
        validate(trainer, plan, item, stop)
        resume = out / "checkpoint_latest.pt"
        if trainer.ticks in plan["checkpoint_ticks"]:
            _snapshot(out, trainer, plan, item, sha)
        verify_end(path, identifier, sha)
    if trainer.ticks == plan["target_ticks"]:
        return _terminal(out, trainer, plan, item, sha)
    return dict(status="pending", last_tick=trainer.ticks, target_ticks=plan["target_ticks"])


def _checked_evaluation(value, plan, item, sha, terminal):
    """Validate a completed result before an interrupted queue reuses it."""
    if not isinstance(value, dict) or value.get("result_hash") != state_digest({key: item for key, item in value.items() if key != "result_hash"}):
        raise ValueError("evaluation result content hash mismatch")
    terminal_path = Path(item["output"]) / "terminal.json"
    success = terminal["status"] == "success"
    if (value.get("format") != "undirected-evaluation-v1" or value.get("branch") != item or value.get("protocol_sha256") != sha
            or value.get("terminal_status") != terminal["status"] or value.get("endpoint_comparable") is not success
            or value.get("source_unchanged") is not True or value.get("terminal_path") != str(terminal_path.resolve())
            or value.get("terminal_sha256") != file_sha256(terminal_path) or value.get("last_tick") != terminal["last_tick"]
            or value.get("target_ticks") != plan["target_ticks"] or value.get("planned_forks") != len(plan["evaluation_specs"])):
        raise ValueError("foreign evaluation result binding")
    records, states = value.get("records"), value.get("states")
    if not isinstance(records, list) or not isinstance(states, dict) or value.get("actual_forks") != len(records):
        raise ValueError("incomplete evaluation result inventory")
    if not success:
        if records or states or value.get("omitted_specs") != plan["evaluation_specs"]:
            raise ValueError("numerical failure must have empty unscored evaluation records")
        return value
    if ([record.get("probe") for record in records] != plan["evaluation_specs"] or value.get("omitted_specs") != []
            or set(states) != {str(tick) for tick in plan["checkpoint_ticks"] if tick}):
        raise ValueError("incomplete frozen evaluation forks")
    for key, state in states.items():
        entry = terminal["checkpoints"][key]
        if (state.get("checkpoint") != entry["path"] or state.get("checkpoint_sha256") != entry["sha256"]
                or state.get("state_hash") != entry["semantic_hash"]
                or state.get("parameter_layout") != plan["birth_bindings"]["by_branch"][item["id"]]["parameter_layout"]):
            raise ValueError("evaluation checkpoint/layout binding mismatch")
    for record in records:
        state = states[str(record["probe"]["tick"])]
        if (record.get("undirected") != {key: item[key] for key in KEYS}
                or any(record.get(key) != 0 for key in ("world_resets", "hidden_resets", "parameter_updates"))
                or record.get("start_hidden_hash") != state["hidden_hash"]
                or record.get("start_world_state_hash") != state["world_hash"]
                or record.get("parameter_hash") != state["model_hash"]
                or record.get("source_continuation_hash") != state["source_continuation_hash"]):
            raise ValueError("evaluation fork lost its complete source binding")
    return value


def evaluate(path, identifier, checkpoint, output):
    torch.set_num_threads(1)
    plan, item, sha = read_plan(path, identifier)
    out, output = Path(item["output"]), Path(output).resolve()
    expected = Path(plan["raw_root"]) / "evaluation" / f"{identifier}.json"
    if Path(checkpoint).resolve() != (out / "checkpoint_latest.pt").resolve() or output != expected.resolve():
        raise ValueError("foreign evaluation path")
    terminal = completion(plan, item, sha)
    if output.exists():
        value = _checked_evaluation(_json(output), plan, item, sha, terminal)
        verify_end(path, identifier, sha)
        return value
    records, states = [], {}
    if terminal["status"] == "success":
        with _preserve_process_state(), torch.no_grad():
            for tick in plan["checkpoint_ticks"][1:]:
                entry = terminal["checkpoints"][str(tick)]
                cp = Path(entry["path"])
                trainer = native.NativeTrainer.from_checkpoint(cp, device="cpu")
                validate(trainer, plan, item, tick)
                before = state_digest(trainer.semantic_state())
                if before != entry["semantic_hash"]:
                    raise ValueError("frozen snapshot semantic mismatch")
                states[str(tick)] = dict(checkpoint=str(cp), checkpoint_sha256=entry["sha256"], state_hash=before,
                    credit_age=trainer.credit.updates, parameter_layout=trainer.semantic_state()["parameter_layout"],
                    hidden_hash=state_digest({"hidden": trainer.h}), world_hash=state_digest(trainer.world.state_dict()),
                    model_hash=state_digest(trainer.model.state_dict()), source_continuation_hash=state_digest(
                        dict(model=trainer.model.state_dict(), world=trainer.world.state_dict(), hidden=trainer.h)))
                for spec in (row for row in plan["evaluation_specs"] if row["tick"] == tick):
                    record = evaluate_fork(trainer.model, trainer.world, trainer.h, family=spec["family"],
                        condition=spec["condition"], steps=spec["steps"], dt=plan["dt"], device="cpu")
                    if (any(record[key] for key in ("world_resets", "hidden_resets", "parameter_updates"))
                            or _normalized(record["model"]) != _normalized(trainer.model.config())):
                        raise AssertionError("frozen fork reset, learned or lost its actual interface")
                    record["probe"], record["undirected"] = spec, {key: item[key] for key in KEYS}
                    records.append(record)
                if state_digest(trainer.semantic_state()) != before or file_sha256(cp) != entry["sha256"]:
                    raise AssertionError("frozen evaluation changed its complete source")
    verify_end(path, identifier, sha)
    value = dict(format="undirected-evaluation-v1", branch=item, protocol_sha256=sha,
        terminal_status=terminal["status"], endpoint_comparable=terminal["endpoint_comparable"], source_unchanged=True,
        terminal_path=str((out / "terminal.json").resolve()), terminal_sha256=file_sha256(out / "terminal.json"),
        last_tick=terminal["last_tick"], target_ticks=plan["target_ticks"], states=states, records=records,
        planned_forks=len(plan["evaluation_specs"]), actual_forks=len(records),
        omitted_specs=[] if terminal["status"] == "success" else plan["evaluation_specs"],
        note="No endpoint score exists for a numerical_failure branch; empty records are not zero scores." if not records else
             "Forty autonomous continuations from the five declared nonzero snapshots; no learning during forks.")
    value = _normalized(value)
    value["result_hash"] = state_digest(value)
    _checked_evaluation(value, plan, item, sha, terminal)
    atomic_json(output, value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--folder", default=str(DEFAULT_EVIDENCE))
    p.add_argument("--raw", default=str(DEFAULT_RAW))
    for command in ("train", "evaluate"):
        p = sub.add_parser(command)
        p.add_argument("--protocol", required=True)
        p.add_argument("--branch-id", required=True)
        if command == "train":
            p.add_argument("--stop-at", type=int)
        else:
            p.add_argument("--checkpoint", required=True)
            p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(prepare(args.folder, args.raw), flush=True)
    elif args.command == "train":
        train(args.protocol, args.branch_id, stop_at=args.stop_at)
    else:
        evaluate(args.protocol, args.branch_id, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
