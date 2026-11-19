"""Local exposure-dependent deadband for negative receptor evidence."""

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np

from .ring_alt_confidence import ConfidenceForager, paired_margins
from .ring_alt_policy import ForagingSpec, ReceptorForager
from .ring_alt_probe import run as probe_run
from .ring_alt_run import summarize_result, world_panel
from .ring_world import SIZE, rollout
from .spike_analysis import estimates


class MaturingReceptor(ConfidenceForager):
    """Each receptor family's gate relaxes as its own negative evidence grows."""

    def __init__(self, spec, *, birth_seed, action_seeds, margin, rate):
        super().__init__(spec, birth_seed=birth_seed, action_seeds=action_seeds, margin=margin)
        self.maturation = rate

    def reset(self, n):
        super().reset(n)
        self.witnesses = np.zeros((n, SIZE), dtype=np.float32)

    def _learn(self, use, contact, sign, eligibility, prediction):
        super()._learn(use, contact, sign, eligibility, prediction)
        if self.spec.learning_rate:
            self.witnesses += (use & (sign < 0))[:, None] * contact

    def _valence(self, current):
        value = ReceptorForager._valence(self, current)
        threshold = self.margin * np.exp(-self.maturation * np.maximum(0., self.witnesses - 1.))
        return np.sign(value) * np.maximum(np.abs(value) - threshold, 0.)


def constructor(tape, spec, seed, arm, protocol):
    action = [np.random.SeedSequence((s, 0x414C5452)) for s in tape.seeds]
    kwargs = {"birth_seed": seed, "action_seeds": action}
    if arm == "raw":
        return ReceptorForager(spec, **kwargs)
    if arm == "fixed":
        return ConfidenceForager(spec, margin=protocol["deadband"], **kwargs)
    return MaturingReceptor(spec, margin=protocol["deadband"], rate=protocol["maturation"], **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frozen", type=Path)
    parser.add_argument("adaptive", type=Path)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    cfg = ForagingSpec(**json.loads(args.frozen.read_text(encoding="utf-8"))["spec"])
    protocol = json.loads(args.adaptive.read_text(encoding="utf-8"))
    births = tuple(protocol["births"])
    test = world_panel(protocol["probe_start"], protocol["probe_worlds"], 34, panel="evaluation")
    training = world_panel(protocol["training_start"], protocol["training_worlds"], 224, panel="evaluation")
    rows = {}
    probes = {}
    for name in ("raw", "fixed", "maturing"):
        def make(tape, spec, seed):
            return constructor(tape, spec, seed, name, protocol)

        with contextlib.redirect_stdout(io.StringIO()):
            p_rows, causal = probe_run(test, cfg, births, factory=make)
        probes[name] = {"summary": causal, "rows": p_rows}
        rows[name] = []
        for seed in births:
            policy = make(training, cfg, seed)
            outcome = rollout(policy, training)
            score = summarize_result(training, outcome)
            rows[name].append({"seed": seed, "scores": score})
            print(f"{seed} {name:8s} late {np.mean(score['late_first']):.3f}", flush=True)
    # With only one wrong receipt, local evidence remains at the deadband's
    # initial setting. The two conditional probes must therefore agree.
    for fixed, dynamic in zip(probes["fixed"]["rows"], probes["maturing"]["rows"], strict=True):
        np.testing.assert_array_equal(fixed["revisit"], dynamic["revisit"])
    late = {name: np.mean([row["scores"]["late_first"] for row in group], axis=0) for name, group in rows.items()}
    summary = {"causal_rescue": paired_margins(probes["raw"], probes["maturing"]),
               "late_first": {name: estimates(values) for name, values in late.items()},
               "late_maturing_minus_raw": estimates(late["maturing"] - late["raw"]),
               "late_maturing_minus_fixed": estimates(late["maturing"] - late["fixed"])}
    args.artifact.write_text(json.dumps({"protocol": protocol, "life_seeds": training.seeds, "probe_seeds": test.seeds,
                                         "probes": probes, "training": rows, "summary": summary}, separators=(",", ":"))
                             + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
