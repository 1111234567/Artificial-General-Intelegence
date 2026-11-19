"""Frozen evaluations that continue a saved world's complete causal history.

Every condition forks the saved world, receipts, clock and numerical hidden
state. Only future NEW draws change regime. Primary denominators contain
post-fork requests sampled under the evaluation regime whose entire deadline
window has matured. Inherited, prior-contract and right-censored requests are
reported separately. The optional teacher readout
probe is a diagnostic of the existing FINISH head, not autonomous success or
proof that inaccessible content has disappeared from the hidden state.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.continuity.run import (
    Trainer, atomic_json, restore_rng_state, rng_state, runtime_record,
    state_digest, synchronize,
)
from experiments.continuity.world import NEW, REPLACE, OnlineWorld

FORMAT_VERSION = "continuity-evaluation-v1"
DEFAULT_CONDITIONS = ("iid", "long_wait", "longer", "sparse", "conflict")
CONDITIONS = {
    "iid": dict(duration_range=(1.0, 4.0), gap_range=(0.5, 2.0),
                window_range=(1.5, 2.5), cancel=0.20, replace=0.30),
    "long_wait": dict(duration_range=(4.0, 8.0), gap_range=(0.5, 2.0),
                      window_range=(1.5, 2.5), cancel=0.20, replace=0.30),
    "longer": dict(duration_range=(8.0, 16.0), gap_range=(0.5, 2.0),
                   window_range=(1.5, 2.5), cancel=0.20, replace=0.30),
    "sparse": dict(duration_range=(1.0, 4.0), gap_range=(4.0, 12.0),
                   window_range=(1.5, 2.5), cancel=0.20, replace=0.30),
    "conflict": dict(duration_range=(1.0, 4.0), gap_range=(0.25, 0.75),
                     window_range=(1.5, 2.5), cancel=0.30, replace=0.55),
}


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _preserve_process_state():
    """Loading a trainer restores its RNGs; an evaluation must leave ours alone."""
    saved_rng = rng_state()
    settings = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
    )
    try:
        yield
    finally:
        restore_rng_state(saved_rng)
        torch.use_deterministic_algorithms(settings[0], warn_only=settings[1])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = settings[2:4]
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = settings[4:6]


class _Cohort:
    def __init__(self, name: str, world: OnlineWorld):
        self.name = name
        self.counts = torch.zeros((world.batch_size, 3), dtype=torch.long, device=world.device)
        self.digest = hashlib.sha256(f"{FORMAT_VERSION}/{name}".encode())
        self.actor_ids = np.asarray(world.actor_ids, dtype=np.uint64)

    def add(self, world: OnlineWorld, mask: torch.Tensor) -> None:
        self.counts[:, 0] += mask.sum(1)
        self.counts[:, 1] += (mask & world.matured_completed).sum(1)
        self.counts[:, 2] += (mask & world.matured_clean_completed).sum(1)
        positions = mask.nonzero().detach().cpu().numpy()
        if not len(positions):
            return
        # Only identity and exogenous timing enter this hash, never policy outcomes.
        self.digest.update(self.actor_ids[positions[:, 0]].tobytes())
        self.digest.update(positions[:, 1].astype(np.int64).tobytes())
        for value in (world.matured_request_id, world.matured_issued_at, world.matured_duration):
            self.digest.update(value[mask].detach().cpu().contiguous().numpy().tobytes())

    def result(self) -> dict[str, Any]:
        counts = self.counts.cpu()
        matured, completed, clean = map(int, counts.sum(0).tolist())
        return {"matured": matured, "completed": completed, "clean_completed": clean,
                "completion_rate": _rate(completed, matured),
                "clean_completion_rate": _rate(clean, matured),
                "request_fingerprint": self.digest.hexdigest(),
                "per_actor": {name: counts[:, index].tolist() for index, name in enumerate(
                    ("matured", "completed", "clean_completed"))}}


def _new_since_fork(request_ids: torch.Tensor, world: OnlineWorld, fork_tick: int) -> torch.Tensor:
    # IDs encode issuance tick. This also handles an explicitly newborn fork at
    # tick -1, whose first observation shares physical time zero with creation.
    return (request_ids >= 0) & (request_ids.div(world.n_jobs, rounding_mode="floor") > fork_tick)


def _pending_summary(world: OnlineWorld, fork_tick: int, evaluation_regime_id: int) -> dict[str, Any]:
    pending = world._score_pending
    issued_after = _new_since_fork(world._score_request_id, world, fork_tick)
    primary = pending & issued_after & (world.current_regime_id == evaluation_regime_id)
    carryover = pending & issued_after & ~primary
    inherited = pending & ~issued_after
    return {name: {"count": int(mask.sum()),
                   "completed": int((mask & world._score_completed).sum()),
                   "clean_completed": int((mask & world._score_clean).sum())}
            for name, mask in (("primary", primary), ("inherited", inherited),
                               ("carryover_contract", carryover))}


@torch.inference_mode()
def evaluate_fork(model, world: OnlineWorld, hidden: torch.Tensor, *,
                  condition: str = "iid", steps: int = 2048, dt: float = 0.25,
                  reset_hidden: bool = False, boundary_interval: int = 512,
                  teacher_forced: bool = False,
                  device: str | torch.device | None = None) -> dict[str, Any]:
    """Continue detached clones; leave the supplied model/world/hidden untouched.

    Reset interventions occur before processing the observation at positive
    absolute tick multiples of ``boundary_interval``. Observations, including
    receipts and a possible command at the cut, remain unchanged. Boundary
    cohorts use issued_at < cut_time < issued_at + declared_duration and final
    eligibility, independently of each policy's BEGIN time or active status.
    """
    steps = _positive_integer("steps", steps)
    boundary_interval = _positive_integer("boundary_interval", boundary_interval)
    if condition not in CONDITIONS:
        raise ValueError(f"unknown evaluation condition {condition!r}")
    if not math.isfinite(float(dt)) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    saved_world = world.state_dict()
    if not saved_world["acted"]:
        raise ValueError("evaluation forks must be after a completed action")
    if tuple(hidden.shape) != (world.batch_size, model.width):
        raise ValueError("hidden state does not match the saved world/model")
    if model.n_jobs != world.n_jobs or model.input_size != world.input_size:
        raise ValueError("model and world observation/action shapes differ")
    target_device = torch.device(hidden.device if device is None else device)
    live_model = deepcopy(model).to(target_device).eval()
    state = hidden.detach().to(target_device).clone()
    live_world = OnlineWorld.from_state_dict(saved_world, device=world.device)
    initial_parameter_hash = state_digest(live_model.state_dict())
    start_hidden_hash = state_digest({"hidden": state})
    fork_tick, fork_time = live_world.tick, live_world.now
    initial_counts = {name: int(value.sum()) for name, value in live_world.counts.items()}
    initial_pending = int(live_world._score_pending.sum())
    initial_active = int(live_world.active.sum())
    initial_started = int((live_world.active & live_world.started).sum())
    live_world.set_regime(**CONDITIONS[condition], name=f"evaluation/{condition}", dt=dt)
    evaluation_regime_id = live_world.regime_id
    cohorts = {name: _Cohort(name, live_world) for name in
               ("primary", "inherited", "carryover_contract", "boundary_crossing",
                "inherited_boundary_crossing", "carryover_boundary_crossing")}
    tracked_ids = live_world.request_id.clone()
    tracked_regimes = live_world.current_regime_id.clone()
    crossed = torch.zeros_like(live_world.active)
    action_digest = hashlib.sha256(f"{FORMAT_VERSION}/actions".encode())
    observation_digest = hashlib.sha256(f"{FORMAT_VERSION}/observations".encode())
    boundaries = []
    primary_issued = carryover_issued = 0
    probe_count = probe_correct = 0
    probe_digest = hashlib.sha256(f"{FORMAT_VERSION}/teacher-finish-probe".encode())
    rms_start = float(state.double().square().mean().sqrt())
    rms_sum, rms_max = 0.0, rms_start
    first_observation_hash = None
    synchronize(target_device)
    started = time.perf_counter()
    for _ in range(steps):
        observations = live_world.observe()
        issued_after = _new_since_fork(live_world.matured_request_id, live_world, fork_tick)
        mature = live_world.matured_mask
        if (mature & (tracked_ids != live_world.matured_request_id)).any():
            raise AssertionError("matured request lost its regime/boundary tracking metadata")
        primary = issued_after & (tracked_regimes == evaluation_regime_id)
        carryover = issued_after & ~primary
        crossed_mature = crossed & (tracked_ids == live_world.matured_request_id)
        cohorts["primary"].add(live_world, mature & primary)
        cohorts["inherited"].add(live_world, mature & ~issued_after)
        cohorts["carryover_contract"].add(live_world, mature & carryover)
        cohorts["boundary_crossing"].add(live_world, mature & primary & crossed_mature)
        cohorts["inherited_boundary_crossing"].add(live_world, mature & ~issued_after & crossed_mature)
        cohorts["carryover_boundary_crossing"].add(live_world, mature & carryover & crossed_mature)
        command = (live_world.event_kind == NEW) | (live_world.event_kind == REPLACE)
        eligible_command = command & live_world.event_eligible
        under_evaluation = live_world.event_regime_id == evaluation_regime_id
        primary_issued += int((eligible_command & under_evaluation).sum())
        carryover_issued += int((eligible_command & ~under_evaluation).sum())
        tracked_ids = torch.where(command, live_world.event_request_id, tracked_ids)
        tracked_regimes = torch.where(command, live_world.event_regime_id, tracked_regimes)
        crossed = torch.where(command, False, crossed)
        if live_world.tick > 0 and live_world.tick % boundary_interval == 0:
            at_cut = (live_world.current_eligible & (live_world.request_id >= 0)
                      & (live_world.issued_at < live_world.now)
                      & (live_world.now < live_world.earliest_ready))
            crossed |= at_cut
            boundaries.append({"tick": live_world.tick, "time": live_world.now,
                               "target_count": int(at_cut.sum())})
            if reset_hidden:
                state = torch.zeros_like(state)
        raw_observations = observations.detach().cpu().contiguous().numpy().tobytes()
        observation_digest.update(raw_observations)
        if first_observation_hash is None:
            first_observation_hash = state_digest({"observation": observations})
        logits, state = live_model.step(observations.to(target_device), state, live_world.elapsed)
        if not torch.isfinite(state).all() or not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite state/logits in frozen continuation")
        rms = float(state.double().square().mean().sqrt())
        rms_sum += rms
        rms_max = max(rms_max, rms)
        if teacher_forced:
            actions = live_world.oracle("midpoint")
            mask = (actions >= 3) & live_world.current_eligible & _new_since_fork(
                live_world.request_id, live_world, fork_tick) & (
                    live_world.current_regime_id == evaluation_regime_id)
            if mask.any():
                wanted = (actions[mask] - 3).to(target_device)
                readout = logits[..., 3:][mask.to(target_device)].argmax(-1)
                probe_count += int(mask.sum())
                probe_correct += int((readout == wanted).sum())
                for value in (live_world.request_id[mask], readout, wanted):
                    probe_digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            actions = logits.argmax(-1).to(live_world.device)
        action_digest.update(actions.detach().cpu().contiguous().numpy().tobytes())
        live_world.act(actions)
    synchronize(target_device)
    seconds = time.perf_counter() - started
    counters = {name: int(value.sum()) - initial_counts[name]
                for name, value in live_world.counts.items()}
    cohort_results = {name: cohort.result() for name, cohort in cohorts.items()}
    censored = _pending_summary(live_world, fork_tick, evaluation_regime_id)
    if primary_issued != cohort_results["primary"]["matured"] + censored["primary"]["count"]:
        raise AssertionError("post-fork eligible requests were lost or counted twice")
    if carryover_issued != (cohort_results["carryover_contract"]["matured"]
                           + censored["carryover_contract"]["count"]):
        raise AssertionError("prior-contract post-fork replacements were lost or counted twice")
    if initial_pending != cohort_results["inherited"]["matured"] + censored["inherited"]["count"]:
        raise AssertionError("inherited eligible windows were lost or counted twice")
    final_parameter_hash = state_digest(live_model.state_dict())
    if final_parameter_hash != initial_parameter_hash:
        raise AssertionError("frozen evaluation changed model parameters")
    result = {
        "format": FORMAT_VERSION, "condition": condition,
        "control": "hidden_reset" if reset_hidden else "continuous",
        "action_policy": "teacher" if teacher_forced else "model",
        "world_resets": 0, "parameter_updates": 0,
        "steps": steps, "dt": float(dt), "batch_size": live_world.batch_size,
        "n_jobs": live_world.n_jobs, "active_jobs": live_world.active_jobs,
        "actor_ids": list(live_world.actor_ids), "case_ids": list(live_world.case_ids),
        "fork_tick": fork_tick, "fork_time": fork_time,
        "end_tick": live_world.tick, "end_time": live_world.now,
        "physical_duration": live_world.now - fork_time,
        "regime": asdict(live_world.regime), "evaluation_regime_id": evaluation_regime_id,
        "boundary_interval": boundary_interval, "boundaries": boundaries,
        "hidden_resets": len(boundaries) if reset_hidden else 0,
        "primary_eligible_issued": primary_issued,
        "carryover_contract_eligible_issued": carryover_issued,
        "post_fork_eligible_issued": primary_issued + carryover_issued,
        "primary": cohort_results["primary"],
        "inherited": {"initial_eligible_pending": initial_pending,
                      "initial_active": initial_active, "initial_started": initial_started,
                      **cohort_results["inherited"]},
        "carryover_contract": cohort_results["carryover_contract"],
        "boundary_crossing": cohort_results["boundary_crossing"],
        "inherited_boundary_crossing": cohort_results["inherited_boundary_crossing"],
        "carryover_boundary_crossing": cohort_results["carryover_boundary_crossing"],
        "censored": censored,
        "post_fork_counters": counters,
        "post_fork_finish_precision": _rate(counters["completed_all"], counters["finish_attempted"]),
        "post_fork_false_actions_per_1000_opportunities": (
            counters["false_action"] * 1000 / (steps * live_world.batch_size * live_world.active_jobs)),
        "start_event_hash": saved_world["event_hash"], "end_event_hash": live_world.event_hash,
        "start_world_state_hash": state_digest(saved_world),
        "end_world_state_hash": state_digest(live_world.state_dict()),
        "start_hidden_hash": start_hidden_hash, "end_hidden_hash": state_digest({"hidden": state}),
        "first_observation_hash": first_observation_hash,
        "observation_hash": observation_digest.hexdigest(), "action_hash": action_digest.hexdigest(),
        "parameter_hash": initial_parameter_hash,
        "state_rms": {"start": rms_start, "final": rms, "mean": rms_sum / steps, "max": rms_max},
        "model": live_model.config(), "seconds": seconds,
        "denominator": ("eligible requests issued after the fork from a NEW sampled under the evaluation "
                        "regime (including its replacements), with deadline windows matured by the end"),
        "boundary_definition": "issued_at < cut_time < issued_at + duration; final eligible; policy-independent",
    }
    if teacher_forced:
        result["content_probe"] = {
            "count": probe_count, "correct": probe_correct,
            "accuracy": _rate(probe_correct, probe_count), "fingerprint": probe_digest.hexdigest(),
            "denominator": ("post-fork eligible evaluation-regime teacher FINISH opportunities, "
                            "including still-censored windows"),
            "interpretation": (
                "Conditional argmax over the existing eight FINISH version/payload logits at causal "
                "teacher-selected times. Teacher actions determine receipts. This excludes WAIT/BEGIN "
                "competition but still combines encoding, retention and existing-head readout; it does "
                "not measure autonomous timing or prove that other hidden information is absent."),
        }
    return result


def evaluate_checkpoint(checkpoint: str | Path, *,
                        conditions: Iterable[str] = DEFAULT_CONDITIONS, steps: int = 2048,
                        dt: float = 0.25, device: str = "cpu", content_probe: bool = False,
                        reset_interval: int = 512, strict_sources: bool = True) -> dict[str, Any]:
    """Read one complete training checkpoint and return compact fork records."""
    path = Path(checkpoint).resolve()
    names = tuple(conditions)
    if not names or len(set(names)) != len(names) or any(name not in CONDITIONS for name in names):
        raise ValueError("conditions must be a nonempty sequence of distinct known conditions")
    steps = _positive_integer("steps", steps)
    reset_interval = _positive_integer("reset_interval", reset_interval)
    if not math.isfinite(float(dt)) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    checkpoint_sha256 = _file_sha256(path)
    with _preserve_process_state():
        trainer = Trainer.from_checkpoint(path, device=device, strict_sources=strict_sources)
        start_semantic_hash = state_digest(trainer.semantic_state())
        records = []
        for condition in names:
            common = dict(condition=condition, steps=steps, dt=dt,
                          boundary_interval=reset_interval, device=device)
            baseline = evaluate_fork(trainer.model, trainer.world, trainer.h, **common)
            records.append(baseline)
            if condition == "iid":
                reset = evaluate_fork(trainer.model, trainer.world, trainer.h,
                                      reset_hidden=True, **common)
                for field in ("start_world_state_hash", "start_hidden_hash", "end_event_hash"):
                    if baseline[field] != reset[field]:
                        raise AssertionError(f"paired hidden reset changed the external world: {field}")
                for cohort in ("primary", "inherited", "carryover_contract", "boundary_crossing",
                               "inherited_boundary_crossing", "carryover_boundary_crossing"):
                    if baseline[cohort]["request_fingerprint"] != reset[cohort]["request_fingerprint"]:
                        raise AssertionError(f"paired hidden reset changed the request cohort: {cohort}")
                reset["paired_baseline_action_hash"] = baseline["action_hash"]
                records.append(reset)
            if content_probe:
                records.append(evaluate_fork(trainer.model, trainer.world, trainer.h,
                                             teacher_forced=True, **common))
        if start_semantic_hash != state_digest(trainer.semantic_state()):
            raise AssertionError("read-only forks changed the source trainer continuation")
        sources = {name: _file_sha256(Path(__file__).parent / name) for name in
                   ("world.py", "model.py", "run.py", "evaluate.py", "test_evaluate.py")}
        result = {
            "format": FORMAT_VERSION, "checkpoint": str(path),
            "checkpoint_sha256": checkpoint_sha256, "checkpoint_state_hash": start_semantic_hash,
            "training_variant": asdict(trainer.variant), "training_config": asdict(trainer.config),
            "training_stage": trainer.stage, "adaptation_mode": trainer.adaptation_mode,
            "training_updates": trainer.updates, "training_ticks": trainer.ticks,
            "model": trainer.model.parameter_report(), "records": records,
            "evaluation_args": dict(conditions=list(names), steps=steps, dt=float(dt), device=device,
                                    content_probe=content_probe, reset_interval=reset_interval,
                                    strict_sources=strict_sources),
            "source_hashes": sources, "checkpoint_source_hashes": trainer.source_hashes,
            "runtime": runtime_record(device),
            "limitations": [
                "Different duration regimes share the saved history and future random variates, "
                "but changed renewal times produce different future event schedules.",
                "Primary task scores exclude inherited pending work, post-fork replacements retaining "
                "a prior-regime contract, and right-censored deadline windows.",
                "Frozen fork evaluation measures retained behavior and state, not test-time learning.",
            ],
        }
    if checkpoint_sha256 != _file_sha256(path):
        raise AssertionError("checkpoint file changed during evaluation")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS), default=DEFAULT_CONDITIONS)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reset-interval", type=int, default=512)
    parser.add_argument("--content-probe", action="store_true")
    args = parser.parse_args(argv)
    evaluations = [evaluate_checkpoint(
        path, conditions=args.conditions, steps=args.steps, dt=args.dt, device=args.device,
        content_probe=args.content_probe, reset_interval=args.reset_interval) for path in args.checkpoint]
    atomic_json(args.output, {"format": FORMAT_VERSION, "evaluations": evaluations})
    print(f"Saved {sum(len(item['records']) for item in evaluations)} frozen continuation records to {args.output}")


if __name__ == "__main__":
    main()
