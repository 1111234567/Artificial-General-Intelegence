"""Replay two archived BF16 champions across four paired timing conditions."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import math
from pathlib import Path
from statistics import mean, stdev

import torch

from experiments.seed_evolution.large_runner import _utc, _write
from . import plasticity_roomy_world as roomy_world
from . import uncapped_bf16_backend as backend
from .evolved_plasticity_world import FIRST_FINISH, N_ACTIONS, _receipt_table
from .plasticity_continuous_world import _FixedGuess, rollout_continuous
from .uncapped_bf16_run import configure
from .uncapped_plasticity_run import OUT, base


OUTPUT = OUT.with_name("2026-11-16-uncapped-timing-probe") / "paired.json"
SEEDS = tuple(range(138_614_830, 138_614_838))
ACTORS = 16
VISITS = 8
TIMINGS = {
    "old": (1, 3, 2), "wider_finish": (1, 3, 8),
    "longer_wait": (8, 16, 2), "both": (8, 16, 8),
}
METRICS = (
    "strict", "completion", "strict_return", "ready_coverage", "legal_attempt",
    "version_accuracy", "symbol_accuracy", "off_ready_rate", "mean_edges", "final_edges",
)
CONTRASTS = {
    "wider_finish_at_short_wait": {"wider_finish": 1, "old": -1},
    "longer_wait_at_narrow_finish": {"longer_wait": 1, "old": -1},
    "wider_finish_at_long_wait": {"both": 1, "longer_wait": -1},
    "longer_wait_at_wide_finish": {"both": 1, "wider_finish": -1},
    "both_vs_old": {"both": 1, "old": -1},
    "interaction": {"both": 1, "longer_wait": -1, "wider_finish": -1, "old": 1},
}


class RecordedFinishes(backend._BF16Policy):
    def __init__(self, birth, actors, device):
        super().__init__(birth, actors, device)
        self.primary_actions = []

    def act(self, obs):
        action = super().act(obs)
        self.primary_actions.append(action[..., 0])
        return action


def load_champions(run):
    genomes, records = {}, {}
    for generation in (768, 1024):
        label = f"g{generation}"
        checkpoint = run / "generations" / f"{generation:04d}" / "elites.pt"
        audit = run / "audits" / f"{generation:04d}.json"
        identifier = base.read(audit)["candidate"]
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        genomes[label] = saved["genomes"][identifier]
        records[label] = {"id": identifier, "checkpoint": str(checkpoint), "audit": str(audit),
                          "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    return genomes, records


def finish_metrics(policy, tape, completion):
    requests, ticks, actors = tape.stage.shape
    actions = torch.stack(policy.primary_actions).reshape(requests, ticks, policy.population, actors)
    legal = (tape.stage.unsqueeze(2) == 2) & (actions >= FIRST_FINISH) & (actions < N_ACTIONS)
    correct = legal & actions.eq(tape.target[:, None, None, :])
    prior_correct = correct.cumsum(1) - correct.int()
    legal &= prior_correct.eq(0)
    first = legal & (legal.cumsum(1) == 1)
    recovered = (correct & legal & ~first).any(1).float().mean((0, 2))
    first_correct = (correct & first).any(1).float().mean((0, 2))
    torch.testing.assert_close(first_correct + recovered, completion, rtol=0, atol=0)
    covered = legal.any(1).sum((0, 2))
    return {
        "legal_finishes_per_request": legal.sum((0, 1, 3)) / (requests * actors),
        "first_finish_joint_accuracy": (correct & first).sum((0, 1, 3)) / covered.clamp_min(1),
        "completion_after_wrong_legal_finish": recovered,
    }


@torch.no_grad()
def measure(birth, tape, device):
    tape = backend._device_tape(replace(tape, public=tape.public.bfloat16()), device)
    policy = RecordedFinishes(birth, ACTORS, device)
    table = _receipt_table(device).bfloat16()
    fields = backend._rollout(policy, tape, table)
    metrics = {key: fields[key] for key in METRICS}
    metrics.update(finish_metrics(policy, tape, fields["completion"]))
    metrics["weight_change_per_actor_tick"] = fields["weight_change"] / (policy.step * ACTORS)
    metrics["end_weight_l1_from_birth"] = (policy.w - birth["birth_w"][:, None]).abs().sum(-1).mean(-1)
    return {key: value.tolist() for key, value in metrics.items()}


def summarize(rows):
    names = list(rows[0]["results"]["old"])
    summary, paired = {}, {}
    for name in names:
        keys = rows[0]["results"]["old"][name]
        summary[name] = {
            timing: {key: mean(row["results"][timing][name][key] for row in rows) for key in keys}
            for timing in TIMINGS
        }
        paired[name] = {}
        for contrast, coeffs in CONTRASTS.items():
            effects = {}
            for key in keys:
                differences = [sum(coefficient * row["results"][timing][name][key]
                                   for timing, coefficient in coeffs.items()) for row in rows]
                average = mean(differences)
                half = 2.364624251 * stdev(differences) / math.sqrt(len(SEEDS))
                effects[key] = {"mean": average, "paired_seed_ci95": [average - half, average + half]}
            paired[name][contrast] = effects
    return summary, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=OUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    configure()
    device = torch.device("cuda", torch.cuda.current_device())
    genomes, records = load_champions(args.run_dir)
    _, birth = backend._stack_genomes(list(genomes.values()))
    birth = {key: value.to(device) for key, value in birth.items()}
    rows = []
    for seed in SEEDS:
        results, baseline = {}, {}
        reference = None
        for label, (minimum, maximum, window) in TIMINGS.items():
            tape = roomy_world.make_roomy_schedule(
                seed=seed, actors=ACTORS, device="cpu", visits=VISITS,
                min_wait=minimum, max_wait=maximum, finish_window=window,
            )
            contract = (tape.target, tape.context, tape.raw, tape.maps)
            if reference is None:
                reference = contract
            if any(not left.equal(right) for left, right in zip(reference, contract, strict=True)):
                raise AssertionError("Timing conditions must use the same contracts")
            fixed = rollout_continuous(_FixedGuess(0), tape, 1)
            baseline[label] = {key: fixed[key].item() for key in METRICS if key in fixed}
            if baseline[label]["strict"] != .25 or baseline[label]["completion"] != .25:
                raise AssertionError("Fixed guess must solve exactly one in four requests")
            columns = measure(birth, tape, device)
            results[label] = {name: {key: column[i] for key, column in columns.items()}
                              for i, name in enumerate(genomes)}
        rows.append({"seed": seed, "results": results, "fixed_guess": baseline})
        print(f"Evaluated seed {seed} ({len(rows)}/{len(SEEDS)})", flush=True)

    summary, paired = summarize(rows)
    sources = (Path(__file__), Path(roomy_world.__file__), Path(backend.__file__),
               Path(__file__).with_name("plasticity_continuous_world.py"))
    _write(args.output, {
        "schema": 1, "created_utc": _utc(), "champions": records,
        "precision": "native CUDA BF16 forward, FP32 master weights and gates",
        "device": torch.cuda.get_device_name(device), "torch_version": torch.__version__,
        "seeds": list(SEEDS), "actors_per_seed": ACTORS, "visits": VISITS,
        "timings": {label: {"min_wait": low, "max_wait": high, "finish_window": window}
                    for label, (low, high, window) in TIMINGS.items()},
        "units": "Proportions except edge counts, weight movement and legal finishes per request",
        "intervals": "Two-sided t interval for differences across the eight paired world seeds",
        "sources": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "means": summary, "paired_contrasts": paired, "per_seed": rows,
    })
    print(args.output)


if __name__ == "__main__":
    main()
