"""Calibrate delivered-taste fork labels on a graph with a known edit effect."""

from argparse import ArgumentParser
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from . import edit_receipt_graph as native
from . import self_param_writer as fw
from .edit_receipt_train import ROOT, Tape, future_fork, paired_targets


SAMPLES = 4096
NOISE = .15
FIT_SEED = 811039
TEST_SEED = 811040
SHUFFLE_SEED = 811041
SHUFFLES = 256
EDGE = (11, 0)
OUTPUT = ROOT / "docs/evidence/edit-receipt-2026-11-19/positive-control.json"


def fixture():
    """Zero-weight spanning cycle plus a writable cue-to-motor edge."""
    width = fw.BODY + 8
    weights = torch.zeros(1, width, width)
    present = torch.zeros_like(weights, dtype=torch.bool)
    nodes = torch.arange(width)
    present[0, (nodes + 1) % width, nodes] = True
    present[0, EDGE[0], EDGE[1]] = True
    return fw.Graph(weights, torch.zeros(1, width), present)


def world(seed):
    rng = np.random.default_rng(seed)
    preference = rng.choice((-1, 1), size=SAMPLES).astype(np.float32)
    orientation = rng.choice((-1, 1), size=(SAMPLES, 2)).astype(np.float32)
    odor = rng.uniform(.8, 1.35, size=(SAMPLES, 3)).astype(np.float32)
    odor[:, 1:] = rng.uniform(-.8, .8, size=(SAMPLES, 2))
    x = odor[:, None, :] * orientation[..., None]
    truth = preference[:, None] * orientation
    slips = np.where(rng.random((SAMPLES, 2)) < NOISE, -1, 1).astype(np.float32)
    draws = rng.random((SAMPLES, 2, 4), dtype=np.float32)
    direction = rng.permutation(np.tile([-1., 1.], SAMPLES // 2).astype(np.float32))
    tape = Tape(*(torch.from_numpy(a) for a in (x, truth, truth * slips, draws)))
    return tape, torch.from_numpy(preference), torch.from_numpy(direction)


@torch.no_grad()
def score(seed, *, magnitude=.8):
    tape, preference, direction = world(seed)
    state = native.start(fixture(), SAMPLES)
    proposal = native.propose(state, *tape.contact(0))
    address = torch.full((SAMPLES,), EDGE[0] * state.graph.width + EDGE[1])
    # Hold the prefix fixed and supply the constructed proposal at the executor.
    command = fw.Command(address, torch.full_like(address, fw.Op.ADJUST), magnitude * direction)
    proposal = replace(proposal, command=command)
    fork = future_fork(state, proposal, tape, 0, 1, sensing="own", keep_connected=True)

    # Inputs available before commit are the cue, own side, delivered taste,
    # and proposed command. Hidden preference is only used for the audit below.
    noisy_preference = tape.x[:, 0, 0].sign() * proposal.side * proposal.taste
    cells = 2 * (noisy_preference > 0).long() + (direction > 0).long()
    oracle_direction = preference * direction
    nonzero = fork.physical != 0
    if magnitude > 0:
        assert fork.eligible.all() and (fork.expected * oracle_direction > 0).all()
        assert (fork.physical[nonzero] == oracle_direction[nonzero]).all()
        future_slip = tape.outcome[:, 1] * tape.truth[:, 1]
        assert (fork.taste[nonzero] == fork.physical[nonzero] * future_slip[nonzero]).all()
    else:
        assert not fork.eligible.any() and not nonzero.any() and not fork.taste.any()
    return cells, fork, oracle_direction


def fit(cells, target, used):
    return torch.stack([target[used & (cells == i)].mean() for i in range(4)])


def metrics(prediction, fork, oracle):
    changed = fork.physical != 0
    predicted_sign = prediction.sign()
    return {
        "delivered_taste_mse": float((prediction - fork.taste).square().mean()),
        "latent_effect_direction_accuracy": float((predicted_sign == oracle).float().mean()),
        "physical_direction_accuracy_when_changed": float((predicted_sign[changed] == fork.physical[changed]).float().mean()),
        "delivered_direction_accuracy_when_changed": float((predicted_sign[changed] == fork.taste[changed]).float().mean()),
    }


def run():
    fit_cells, fit_fork, _ = score(FIT_SEED)
    test_cells, test_fork, oracle = score(TEST_SEED)
    _, null, _ = score(TEST_SEED, magnitude=0.)
    rng = np.random.default_rng(SHUFFLE_SEED)
    order = torch.arange(SAMPLES)
    causal_target, causal_used = paired_targets(fit_fork.taste, fit_fork.eligible, order, shuffled=False)
    causal = fit(fit_cells, causal_target, causal_used)
    causal_scores = metrics(causal[test_cells], test_fork, oracle)
    shuffled_cells, shuffled_scores = [], []
    for _ in range(SHUFFLES):
        order = torch.from_numpy(rng.permutation(SAMPLES))
        target, used = paired_targets(fit_fork.taste, fit_fork.eligible, order, shuffled=True)
        shuffled = fit(fit_cells, target, used)
        shuffled_cells.append(shuffled)
        shuffled_scores.append(metrics(shuffled[test_cells], test_fork, oracle))
    shuffled_summary = {}
    for key in causal_scores:
        values = np.array([score[key] for score in shuffled_scores])
        shuffled_summary[key] = {"mean": float(values.mean()),
                                 "central_95_percent": np.quantile(values, [.025, .975]).tolist()}
    zero_mse = float(test_fork.taste.square().mean())
    assert causal_scores["delivered_taste_mse"] < zero_mse
    assert causal_scores["delivered_taste_mse"] < shuffled_summary["delivered_taste_mse"]["central_95_percent"][0]

    return {
        "purpose": "Verify that paired delivered tastes identify an edit with a measurable one-contact causal consequence. An external four-cell probe measures the information in the fitted labels.",
        "reproduce": "python -m experiments.pulse_game.edit_receipt_positive_control",
        "design": {
            "fit_and_test_worlds_each": SAMPLES, "fit_seed": FIT_SEED, "test_seed": TEST_SEED,
            "shuffle_seed": SHUFFLE_SEED, "shuffled_fits": SHUFFLES, "slip_probability_per_contact": NOISE,
            "edit": "Externally supplied ±0.8 adjustment to an initially zero cue-to-motor edge on the actual graph at contact 0. Private preference, orientation and edit sign vary. Full and suppressed commits share upcoming odor, slip and stochastic motor draw.",
            "label": "Accepted minus suppressed realized taste at contact 1, divided by 2, computed by the main experiment's future_fork estimator.",
            "inputs": "Signs of the command payload and the focal preference inferred from the observed odor, own side, and actual delivered taste. Latent preference is used only for verification and scoring.",
            "cells": ["clue−/edit−", "clue−/edit+", "clue+/edit−", "clue+/edit+"],
        },
        "heldout_fork": {
            "applied": int(test_fork.eligible.sum()),
            "changed_side": int(test_fork.changed_side.sum()),
            "improved_physical_count": int((test_fork.physical > 0).sum()),
            "worsened_physical_count": int((test_fork.physical < 0).sum()),
            "mean_abs_expected_physical_effect": float(test_fork.expected.abs().mean()),
            "physical_sign_matches_constructed_direction_when_changed": True,
            "expected_physical_sign_matches_constructed_direction_all_worlds": True,
            "zero_payload_control": {"applied": int(null.eligible.sum()), "changed_side": int(null.changed_side.sum())},
        },
        "fit_cell_prediction": {"causal": causal.tolist(),
                                "shuffled_mean": torch.stack(shuffled_cells).mean(0).tolist()},
        "heldout_probe": {
            "causal": causal_scores,
            "shuffled": shuffled_summary,
            "zero": {"delivered_taste_mse": zero_mse},
        },
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
