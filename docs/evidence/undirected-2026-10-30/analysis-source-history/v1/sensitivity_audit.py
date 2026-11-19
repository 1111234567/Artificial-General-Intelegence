"""Independent projected forward-sensitivity calibration of a complete ledger.

This analysis reconstructs each block at its recorded start state and parameter
values.  torch.func.jvp advances two fixed random parameter directions without
forming a state Jacobian or calling the training credit engine.  Only canonical
checkpoints supply reference S tensors.  Two projections calibrate the recorded
history numerically; they do not prove every entry of S and measure no behavior.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
from pathlib import Path
import platform

import torch
from torch.func import functional_call, jvp

from experiments.undirected import ledger
from experiments.undirected.model import UndirectedModel, PortGraphModel


FORMAT_VERSION = "undirected-sensitivity-audit-v1"
AUDIT_SEED = 731
DIRECTIONS = 2
ATOL = 1e-9
RTOL = 1e-8
ROOT = Path(__file__).resolve().parents[2]
_MODEL_CLASSES = {"UndirectedModel": UndirectedModel, "PortGraphModel": PortGraphModel}


def _normalized(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _finite(tensor, label, *, dtype=None):
    if not isinstance(tensor, torch.Tensor) or (dtype is not None and tensor.dtype != dtype) or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"nonfinite or wrong-dtype {label}")


def _norm_summary(value):
    """Scaled L2 avoids squaring very large or very small history tangents."""
    _finite(value, "norm input")
    scale = float(value.detach().abs().max()) if value.numel() else 0.
    if scale == 0:
        return dict(l2=0., scale=0., scaled_l2=0., log10_l2=None)
    scaled = float(torch.linalg.vector_norm(value.detach() / scale))
    magnitude = scale * scaled
    return dict(l2=magnitude if math.isfinite(magnitude) else None, scale=scale, scaled_l2=scaled,
                log10_l2=math.log10(scale) + math.log10(scaled))


def _compare(actual, reference, label):
    """Elementwise F64 tolerances plus L2 error under a shared safe scale."""
    _finite(actual, label, dtype=torch.float64)
    _finite(reference, f"reference {label}", dtype=torch.float64)
    if actual.shape != reference.shape:
        raise ValueError(f"{label} shape differs")
    scale = max(float(actual.abs().max()), float(reference.abs().max())) if actual.numel() else 0.
    if scale == 0:
        return dict(max_abs_error=0., relative_l2_error=0., common_scale=0., max_tolerance_ratio=0.)
    left, right = actual / scale, reference / scale
    difference = left - right
    absolute = difference.abs()
    tolerance = ATOL / scale + RTOL * right.abs()
    if not bool((absolute <= tolerance).all()):
        ratio = float((absolute / tolerance).max())
        description = f"{ratio:.6g}" if math.isfinite(ratio) else "overflowing"
        raise ValueError(f"{label} differs beyond atol={ATOL}/rtol={RTOL}; tolerance ratio {description}")
    reference_l2 = float(torch.linalg.vector_norm(right))
    error_l2 = float(torch.linalg.vector_norm(difference))
    relative = error_l2 / reference_l2 if reference_l2 else (0. if error_l2 == 0 else None)
    absolute_error = float(absolute.max()) * scale
    ratio = float((absolute / tolerance).max())
    return dict(max_abs_error=absolute_error if math.isfinite(absolute_error) else None,
                relative_l2_error=relative if relative is None or math.isfinite(relative) else None,
                common_scale=scale, max_tolerance_ratio=ratio if math.isfinite(ratio) else None)


def _check_sources(payload):
    observed = {}
    for field, root in (("source_hashes", ROOT / "experiments/undirected"),
                        ("dependency_hashes", ROOT / "experiments")):
        values = payload.get(field)
        if not isinstance(values, Mapping):
            raise ValueError("ledger lacks runtime source fingerprints")
        for name, expected in values.items():
            if not isinstance(name, str) or not _sha(expected):
                raise ValueError("invalid ledger source fingerprint")
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()) or ledger.file_sha256(path) != expected:
                raise ValueError("recorded runtime source fingerprint changed")
            observed[str(path)] = expected
    return observed


def _model(payload):
    name, config = payload.get("model_class"), payload.get("model_config")
    if name not in _MODEL_CLASSES or not isinstance(config, Mapping):
        raise ValueError("unsupported ledger model class/configuration")
    model = _MODEL_CLASSES[name](**config).to(device="cpu", dtype=torch.float64)
    if _normalized(model.config()) != _normalized(config):
        raise ValueError("model configuration changed during audit construction")
    layout = [[name, list(value.shape), str(value.dtype)] for name, value in model.named_parameters()]
    if payload.get("parameter_layout") != layout:
        raise ValueError("ledger parameter layout differs from the actual model")
    dynamic = payload.get("dynamic_names")
    local = payload.get("local_names")
    if (dynamic != list(model.dynamic_parameter_names())
            or local != [name for name, _ in model.named_parameters() if name not in dynamic]):
        raise ValueError("ledger dynamic/local parameter declaration differs")
    return model


def _validate_record(payload, model, dynamic_names):
    record = payload.get("record")
    required = {"parameters", "buffers", "observations", "targets", "elapsed", "start_h", "end_h", "start_tick", "end_tick"}
    if not isinstance(record, Mapping) or set(record) != required:
        raise ValueError("incomplete recorded transition")
    parameters, buffers = record["parameters"], record["buffers"]
    expected_parameters = dict(model.named_parameters())
    expected_buffers = dict(model.named_buffers())
    if set(parameters) != set(expected_parameters) or set(buffers) != set(expected_buffers):
        raise ValueError("recorded transition parameter/buffer inventory differs")
    for name, value in parameters.items():
        _finite(value, f"parameter {name}", dtype=torch.float64)
        if value.shape != expected_parameters[name].shape:
            raise ValueError("recorded parameter shape changed")
    for name, value in buffers.items():
        expected = expected_buffers[name]
        if (not isinstance(value, torch.Tensor) or value.dtype != expected.dtype or value.shape != expected.shape
                or not torch.equal(value.cpu(), expected.detach().cpu())):
            raise ValueError("fixed model buffer differs from the recorded configuration")
    _finite(record["start_h"], "actual start_h", dtype=torch.float64)
    _finite(record["end_h"], "actual end_h", dtype=torch.float64)
    h0, h1, observations, elapsed = record["start_h"], record["end_h"], record["observations"], record["elapsed"]
    if (h0.ndim != 2 or h1.shape != h0.shape or h0.shape[-1] != model.width
            or observations.ndim != 3 or observations.shape[1:] != (len(h0), model.input_size)
            or not isinstance(elapsed, (tuple, list)) or len(elapsed) != len(observations)
            or record["end_tick"] - record["start_tick"] != len(observations)):
        raise ValueError("recorded transition dimensions/time differ")
    _finite(observations, "observations")
    if not observations.is_floating_point() or any(type(dt) not in (int, float) or not math.isfinite(dt) or dt < 0 for dt in elapsed):
        raise ValueError("invalid observation dtype or elapsed time")
    sizes = [parameters[name].numel() for name in dynamic_names]
    flat = torch.cat([parameters[name].detach().cpu().reshape(-1) for name in dynamic_names])
    fixed = {name: value.detach().cpu() for name, value in parameters.items()}
    fixed_buffers = {name: value.detach().cpu() for name, value in buffers.items()}

    def transition(hidden, dynamic):
        values = dict(fixed)
        for name, value in zip(dynamic_names, dynamic.split(sizes)):
            values[name] = value.reshape(expected_parameters[name].shape)
        for observation, dt in zip(observations, elapsed):
            _, hidden = functional_call(model, (values, fixed_buffers),
                                        (observation.detach().cpu(), hidden, dt), strict=True)
        return hidden

    return transition, flat


def _checkpoint_inventory(checkpoints):
    if not isinstance(checkpoints, Mapping):
        raise ValueError("checkpoint inventory must be a tick mapping")
    result = {}
    for key, specification in checkpoints.items():
        try:
            tick = int(key)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid canonical checkpoint tick") from error
        if isinstance(key, bool) or tick < 0 or str(tick) != str(key) or tick in result:
            raise ValueError("invalid or duplicate canonical checkpoint tick")
        if isinstance(specification, (str, Path)):
            path, expected = Path(specification).resolve(), None
        elif isinstance(specification, Mapping) and isinstance(specification.get("path"), str):
            path, expected = Path(specification["path"]).resolve(), specification.get("sha256")
            if not _sha(expected):
                raise ValueError("canonical checkpoint needs a valid supplied sha256")
        else:
            raise ValueError("invalid canonical checkpoint specification")
        digest = ledger.file_sha256(path)
        if expected is not None and digest != expected:
            raise ValueError("canonical checkpoint bytes differ from their binding")
        result[tick] = dict(path=str(path), sha256=digest, supplied_sha256_verified=expected is not None)
    return result


def _check_checkpoint(entry, tick, age, hidden, tangents, directions, payload, method):
    state = torch.load(entry["path"], map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping) or state.get("ticks") != tick or state.get("updates") != age:
        raise ValueError("canonical checkpoint clock differs from ledger")
    config, credit = state.get("config"), state.get("credit")
    if (not isinstance(config, Mapping) or config.get("credit") != method or not isinstance(credit, Mapping)
            or credit.get("update_age") != age or list(credit.get("dynamic_names", ())) != payload["dynamic_names"]):
        raise ValueError("checkpoint method/age/dynamic layout differs")
    if _normalized(credit.get("parameter_layout")) != payload["parameter_layout"]:
        raise ValueError("checkpoint full parameter layout differs")
    if "model_config" in state and _normalized(state["model_config"]) != payload["model_config"]:
        raise ValueError("checkpoint model configuration differs")
    _finite(state.get("hidden"), "checkpoint hidden", dtype=torch.float64)
    if not torch.equal(state["hidden"], hidden):
        raise ValueError("canonical checkpoint hidden state differs from actual ledger boundary")
    if age and (not credit.get("tape") or ledger.content_hash(credit["tape"][-1]) != ledger.content_hash(payload["record"])):
        raise ValueError("checkpoint latest recorded transition differs from ledger")
    sensitivity = credit.get("sensitivity")
    result = dict(tick=tick, update_age=age, **entry)
    if age == 0:
        if sensitivity is not None:
            raise ValueError("newborn checkpoint cannot contain a historical sensitivity")
        result.update(sensitivity_check="newborn_zero", projection_comparisons=[])
    elif method in ("rtrl", "rtrl_reset"):
        _finite(sensitivity, "saved sensitivity", dtype=torch.float64)
        if sensitivity.shape != (*hidden.shape, directions.shape[-1]):
            raise ValueError("saved sensitivity tensor layout differs")
        projection = torch.einsum("bhp,kp->kbh", sensitivity, directions)
        result.update(sensitivity_check="projected_full_S", projection_comparisons=[
            _compare(tangents[index], projection[index], f"saved S projection at tick {tick}, direction {index}")
            for index in range(DIRECTIONS)])
    else:
        if sensitivity is not None:
            raise ValueError("non-RTRL checkpoint unexpectedly contains full S")
        result.update(sensitivity_check="skipped_method_has_no_full_S", projection_comparisons=[])
    return result


def audit_ledger(folder, checkpoints=None, *, expected_binding=None, seed=AUDIT_SEED):
    """Calibrate all ledger blocks, comparing S only at supplied checkpoints.

    ``checkpoints`` maps ticks to paths, or to canonical ``{path, sha256}``
    entries.  Supplying canonical hashes anchors the reference bytes.  The two
    directions and F64 tolerances are fixed; the seed is reported, never drawn
    from or written into training RNG state.
    """
    _integer(seed, "audit seed")
    folder = Path(folder).resolve()
    inventory = ledger.read_ledger(folder, expected_binding=expected_binding, verify_files=True)
    entries = inventory["entries"]
    if not entries:
        raise ValueError("sensitivity audit requires at least one completed live block")
    checkpoint_files = _checkpoint_inventory({} if checkpoints is None else checkpoints)
    payload = ledger.load_record(folder, entries[0], expected_binding_hash=inventory["binding_hash"])
    model = _model(payload)
    method = payload.get("method")
    if method not in ("tbptt128", "recorded512", "recorded1024", "rtrl", "rtrl_reset"):
        raise ValueError("ledger must explicitly declare its temporal credit method")
    dynamic_names = payload["dynamic_names"]
    dynamic_count = sum(dict(model.named_parameters())[name].numel() for name in dynamic_names)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    directions = (torch.randint(2, (DIRECTIONS, dynamic_count), generator=generator, device="cpu") * 2 - 1).to(torch.float64)
    directions /= math.sqrt(dynamic_count)
    tangents = torch.zeros(DIRECTIONS, *payload["record"]["start_h"].shape, dtype=torch.float64)
    first_start = payload["record"]["start_tick"]
    if first_start != 0:
        raise ValueError("complete sensitivity history must begin at birth")
    source_files = _check_sources(payload)
    immutable_metadata = {key: payload.get(key) for key in ("method", "model_class", "model_config", "parameter_layout", "dynamic_names", "local_names",
                                                             "source_hashes", "dependency_hashes")}
    inputs = {inventory["binding_path"]: inventory["binding_sha256"], inventory["manifest_path"]: inventory["manifest_sha256"],
              **{str((folder / entry["file"]).resolve()): entry["sha256"] for entry in entries},
              **{entry["path"]: entry["sha256"] for entry in checkpoint_files.values()}, **source_files}
    analysis_sources = {str(Path(__file__).resolve()): ledger.file_sha256(__file__),
                        str(Path(ledger.__file__).resolve()): ledger.file_sha256(ledger.__file__),
                        str((ROOT / "experiments/undirected/model.py").resolve()): ledger.file_sha256(ROOT / "experiments/undirected/model.py")}
    inputs.update(analysis_sources)
    rows, checkpoint_results, seen_ticks, jvp_calls, model_steps = [], [], set(), 0, 0
    previous_h = None
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        if 0 in checkpoint_files:
            checkpoint_results.append(_check_checkpoint(checkpoint_files[0], 0, 0, payload["record"]["start_h"],
                tangents, directions, payload, method))
            seen_ticks.add(0)
        for entry in entries:
            payload = ledger.load_record(folder, entry, expected_binding_hash=inventory["binding_hash"])
            if {key: payload.get(key) for key in immutable_metadata} != immutable_metadata:
                raise ValueError("runtime model/method/layout/source changed within the live ledger")
            if _check_sources(payload) != source_files:
                raise ValueError("runtime source inventory changed within the live ledger")
            record = payload["record"]
            if previous_h is not None and not torch.equal(record["start_h"], previous_h):
                raise ValueError("recorded live hidden state is discontinuous")
            transition, flat = _validate_record(payload, model, dynamic_names)
            if method == "rtrl_reset":
                tangents = torch.zeros_like(tangents)
            next_tangents, primal_checks = [], []
            for index in range(DIRECTIONS):
                primal, tangent = jvp(transition, (record["start_h"].detach().cpu(), flat), (tangents[index], directions[index]))
                primal_checks.append(_compare(primal, record["end_h"], f"recorded primal at tick {record['end_tick']}"))
                _finite(tangent, "JVP tangent", dtype=torch.float64)
                next_tangents.append(tangent.detach())
                jvp_calls += 1
                model_steps += len(record["elapsed"])
            tangents = torch.stack(next_tangents)
            previous_h = record["end_h"]
            tick = record["end_tick"]
            rows.append(dict(update_age=entry["update_age"], start_tick=record["start_tick"], end_tick=tick,
                             content_hash=entry["content_hash"], sha256=entry["sha256"],
                             tangent_reset=method == "rtrl_reset", primal_comparisons=primal_checks,
                             tangent_norms=[_norm_summary(value) for value in tangents]))
            if tick in checkpoint_files:
                checkpoint_results.append(_check_checkpoint(checkpoint_files[tick], tick, entry["update_age"], record["end_h"],
                    tangents, directions, payload, method))
                seen_ticks.add(tick)
        if seen_ticks != set(checkpoint_files):
            raise ValueError("canonical checkpoint is not a completed ledger boundary")
        for path, expected in inputs.items():
            if ledger.file_sha256(path) != expected:
                raise ValueError("audit input/source bytes changed during analysis")
    finally:
        torch.set_num_threads(threads)
    checked_ticks = [row["tick"] for row in checkpoint_results if row["sensitivity_check"] == "projected_full_S"]
    return dict(format=FORMAT_VERSION, status="passed", method=method, dtype="float64", device="cpu", cpu_threads=1,
        directions=DIRECTIONS, seed=seed, direction_distribution="Independent Rademacher signs divided by sqrt(P_dynamic)",
        direction_hash=ledger.content_hash(directions), dynamic_parameters=dynamic_count, dynamic_names=dynamic_names,
        atol=ATOL, rtol=RTOL, binding_hash=inventory["binding_hash"], manifest_sha256=inventory["manifest_sha256"],
        record_count=len(rows), checkpoint_count=len(checkpoint_results), records=rows, checkpoints=checkpoint_results,
        sensitivity_checked_ticks=checked_ticks, sensitivity_calibrated=bool(checked_ticks),
        validation_scope="primal_history_and_projected_saved_S" if checked_ticks else "primal_history_only_no_saved_S_comparison",
        expected_binding_verified=expected_binding is not None,
        sensitivity_skipped_ticks=[row["tick"] for row in checkpoint_results if row["sensitivity_check"] == "skipped_method_has_no_full_S"],
        work=dict(jvp_calls=jvp_calls, model_step_calls=model_steps, credit_engine_jacobian_calls=0, optimizer_steps=0, world_steps=0),
        input_fingerprints=inputs, analysis_sources=analysis_sources,
        environment=dict(python=platform.python_version(), torch=torch.__version__),
        limitations=["Two random projections provide numerical calibration, not a proof of every full sensitivity entry.",
                     "Realized parameter/structure changes, observations, targets, actions and world are constants.",
                     "No optimizer/world is rerun and no training-time JVP or sensitivity repair is performed.",
                     "Methods without full saved S receive primal/history reconstruction checks but no full-S comparison."])


def audit_terminal(terminal_path, *, seed=AUDIT_SEED):
    """Audit a study terminal with its exact ledger and checkpoint byte bindings."""
    terminal_path = Path(terminal_path).resolve()
    terminal_sha = ledger.file_sha256(terminal_path)
    value = json.loads(terminal_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("format") != "undirected-branch-terminal-v1":
        raise ValueError("unsupported scientific terminal")
    branch, summary = value.get("branch"), value.get("ledger")
    if not isinstance(branch, Mapping) or not isinstance(summary, Mapping) or not _sha(value.get("protocol_sha256")):
        raise ValueError("terminal lacks branch/protocol/ledger binding")
    method = branch.get("credit")
    if not isinstance(branch.get("native_config"), Mapping) or branch["native_config"].get("credit") != method:
        raise ValueError("terminal branch credit method differs from native configuration")
    last, target = _integer(value.get("last_tick"), "terminal last tick"), _integer(value.get("target_ticks"), "terminal target", 1)
    success = value.get("status") == "success"
    if (value.get("status") not in ("success", "numerical_failure") or value.get("completed") is not success
            or value.get("endpoint_comparable") is not success or (success and last != target)
            or (not success and last >= target)):
        raise ValueError("terminal status/progress differs")
    if not isinstance(summary.get("manifest_path"), str) or not isinstance(summary.get("binding_path"), str):
        raise ValueError("terminal lacks exact ledger paths")
    folder = Path(summary["manifest_path"]).resolve().parent
    if (Path(summary["manifest_path"]).resolve() != folder / "manifest.jsonl"
            or Path(summary["binding_path"]).resolve() != folder / "binding.json"):
        raise ValueError("foreign terminal ledger paths")
    expected_binding = dict(protocol_sha256=value["protocol_sha256"], branch=dict(branch),
                            recording_convention="committed-live-block-pre-parameter-trajectory-v1")
    inventory = ledger.read_ledger(folder, expected_binding=expected_binding, verify_files=True)
    for name in ("binding_hash", "binding_path", "binding_sha256", "manifest_path", "manifest_sha256"):
        if summary.get(name) != inventory[name]:
            raise ValueError("terminal ledger summary fingerprint differs")
    entries = inventory["entries"]
    if (summary.get("records") != len(entries) or summary.get("last_tick") != last
            or (entries[-1]["end_tick"] if entries else 0) != last
            or summary.get("compressed_bytes") != sum(entry["compressed_bytes"] for entry in entries)
            or summary.get("uncompressed_bytes") != sum(entry["uncompressed_bytes"] for entry in entries)):
        raise ValueError("terminal ledger inventory differs")
    checkpoints = dict(value.get("checkpoints", {}))
    final = dict(path=value.get("checkpoint_path"), sha256=value.get("checkpoint_sha256"))
    if not isinstance(final["path"], str) or not _sha(final["sha256"]) or ledger.file_sha256(final["path"]) != final["sha256"]:
        raise ValueError("terminal last-complete checkpoint bytes differ")
    # A failed run can stop between canonical snapshots.  Its preserved last
    # complete state is another useful independent projection reference.
    if str(last) not in checkpoints and last not in checkpoints:
        checkpoints[str(last)] = final
    if not entries:
        result = dict(format=FORMAT_VERSION, status="not_applicable_no_successful_live_blocks", method=method,
            record_count=0, checkpoint_count=0, sensitivity_calibrated=False,
            validation_scope="no_successful_live_history", expected_binding_verified=True,
            binding_hash=inventory["binding_hash"], manifest_sha256=inventory["manifest_sha256"],
            directions=DIRECTIONS, seed=seed, atol=ATOL, rtol=RTOL,
            records=[], checkpoints=[], sensitivity_checked_ticks=[], sensitivity_skipped_ticks=[],
            input_fingerprints={inventory["binding_path"]: inventory["binding_sha256"],
                                inventory["manifest_path"]: inventory["manifest_sha256"], final["path"]: final["sha256"]},
            limitations=["No successful live block exists, so no sensitivity projection calibration is claimed."])
    else:
        result = audit_ledger(folder, checkpoints, expected_binding=expected_binding, seed=seed)
        if result["method"] != method:
            raise ValueError("terminal method differs from the live ledger")
    if ledger.file_sha256(terminal_path) != terminal_sha or ledger.file_sha256(final["path"]) != final["sha256"]:
        raise ValueError("terminal/reference bytes changed during analysis")
    result["input_fingerprints"][str(terminal_path)] = terminal_sha
    result["input_fingerprints"][final["path"]] = final["sha256"]
    result["terminal_path"], result["terminal_sha256"] = str(terminal_path), terminal_sha
    result["terminal_status"], result["endpoint_comparable"] = value["status"], value["endpoint_comparable"]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--ledger")
    inputs.add_argument("--terminal", help="Study terminal with ledger and canonical checkpoint fingerprints")
    parser.add_argument("--checkpoints", help="Required with --ledger: JSON tick mapping or a study terminal JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=AUDIT_SEED)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite an existing sensitivity audit")
    if args.terminal:
        if args.checkpoints:
            raise ValueError("--terminal already binds its checkpoints")
        result = audit_terminal(args.terminal, seed=args.seed)
    else:
        if not args.checkpoints:
            parser.error("--ledger requires --checkpoints")
        value = json.loads(Path(args.checkpoints).read_text(encoding="utf-8"))
        if isinstance(value, dict) and value.get("format") == "undirected-branch-terminal-v1":
            if Path(args.ledger).resolve() != Path(value["ledger"]["manifest_path"]).resolve().parent:
                raise ValueError("--ledger differs from the scientific terminal binding")
            result = audit_terminal(args.checkpoints, seed=args.seed)
        else:
            result = audit_ledger(args.ledger, value, seed=args.seed)
    from experiments.undirected.queue import atomic_json
    atomic_json(output, result)


if __name__ == "__main__":
    main()
