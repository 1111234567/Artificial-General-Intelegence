"""Equal-edge, equal-weight, receiver-matched attenuation controls for motifs."""

import argparse
import json
from pathlib import Path

import numpy as np

from .attractor_motif import GAP, WINDOWS, choices, responses, vector
from .attractor_motif_find import FIT_RANGE, experiment, unpack
from .spike_analysis import estimates


ALPHA = .35
NEW_WORLD = range(8_904_000_001, 8_904_000_065)
ACTIVE = WINDOWS["early"]


def attainable(enc, chosen, target, syn, rng):
    assigned = np.zeros_like(enc.weight)
    for row in range(len(assigned)):
        weights = enc.weight[row]
        current = target[row]
        slots = sorted(np.flatnonzero(chosen[row]), key=lambda i: -abs(current[i]))
        available = np.ones_like(weights, dtype=bool)
        for slot in slots:
            signal = current[slot]
            candidates = available & (np.sign(weights) == np.sign(signal)) & (np.abs(weights) >= abs(signal))
            ids = np.flatnonzero(candidates)
            if not ids.size:
                raise ValueError("No feasible equal-signed attenuation")
            distance = np.linalg.norm(syn[:, enc.src[row, ids]] - syn[:, enc.src[row, slot], None], axis=0)
            # Prefer a distinct arc; permit overlap only when required by the sign budget.
            costs = 3.0 * chosen[row, ids] + .8 * distance + rng.gumbel(0., .45, len(ids))
            i = ids[costs.argmin()]
            assigned[row, i] = signal / weights[i]
            available[i] = False
    return assigned


def features(enc, brain, attenuate):
    partial = responses(enc, brain, mask=attenuate, window=ACTIVE)
    spikes = partial["pulses"][:, ACTIVE[0]:GAP + 1].mean() / enc.weight.shape[0]
    mean = partial["mean_input"][slice(*ACTIVE)]
    return partial, float(spikes), mean


def calibrate(enc, brain, baseline, chosen, seed):
    target = ALPHA * chosen.astype(np.float32)
    delta = target * enc.weight
    activity = baseline["state"]["synaptic"][slice(*ACTIVE)].mean(axis=1)
    _, reference, mean = features(enc, brain, target)
    rng = np.random.default_rng(seed)
    scored = []
    for _ in range(48):
        alternative = attainable(enc, chosen, delta, activity, rng)
        shift = alternative * enc.weight
        assert np.count_nonzero(shift) == np.count_nonzero(delta)
        np.testing.assert_allclose(shift.sum(axis=1), delta.sum(axis=1), rtol=1e-6, atol=5e-7)
        np.testing.assert_allclose(np.square(shift).sum(axis=1), np.square(delta).sum(axis=1), rtol=1e-6, atol=5e-7)
        _, firing, currents = features(enc, brain, alternative)
        mismatch = float(np.sqrt(np.square(currents - mean).mean()))
        overlap = int(np.count_nonzero(chosen & (alternative != 0)))
        cost = abs(reference - firing) / .001 + .5 * mismatch / .10 + overlap / max(1, np.count_nonzero(chosen))
        scored.append((cost, alternative, firing, mismatch, overlap))
    return target, sorted(scored, key=lambda row: row[0])[:8], reference


def measure(enc, q, w, base, target, controls):
    r, rate, current = features(enc, q, target)
    selected = (choices(w, r, q) == q.goals).mean(1)
    scores = []
    prior = (choices(w, base, q) == q.goals).mean(1)
    rates, currents = [], []
    for _, ctrl, _, _, _ in controls:
        replay, firing, mean = features(enc, q, ctrl)
        scores.append((choices(w, replay, q) == q.goals).mean(1))
        rates.append(firing)
        currents.append(float(np.sqrt(np.square(mean - current).mean())))
    average_control = np.mean(scores, axis=0)
    return {"candidate": estimates(selected), "matched_control": estimates(average_control),
            "candidate_minus_matched": estimates(selected - average_control), "site_effect": estimates(selected - prior),
            "candidate_rate": rate, "condition_rate_mean": float(np.mean(rates)),
            "per_control_rate": rates, "root_mean_square_current_difference": currents}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frozen", type=Path)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    released = json.loads(args.frozen.read_text(encoding="utf-8"))
    v = vector()
    fit_worlds = tuple(range(FIT_RANGE, FIT_RANGE + 24))
    output = {}
    for birth in released["births"]:
        enc0, q0, _, base0, _ = experiment(v, birth, fit_worlds)
        enc, q, w, base, root = experiment(v, birth, NEW_WORLD)
        row = {"reference": root}
        geom = released["anatomy"][str(birth)]["geometry"]
        for i, name in enumerate(("sender_cut", "cycle_cut")):
            selected = unpack(geom[name]["selected"])
            scale, peers, firing = calibrate(enc0, q0, base0, selected, birth + 230912 + i)
            delta = scale * enc0.weight
            ratio = [float(np.linalg.norm(alt * enc0.weight) / np.linalg.norm(delta)) for _, alt, _, _, _ in peers]
            row[name] = {**measure(enc, q, w, base, scale, peers), "changed_edges": int(selected.sum()),
                         "candidate_shift_l2": float(np.linalg.norm(delta)), "control_shift_l2_ratios": ratio,
                         "chosen_controls": [{"overlap": overlap, "calibration_rate": rate, "calibration_current_gap": current,
                                              "edge_deltas": [(*map(int, idx), float((alt * enc0.weight)[tuple(idx)]))
                                                              for idx in np.argwhere(alt)]}
                                             for _, alt, rate, current, overlap in peers], "calibration_target_rate": firing}
            print(birth, name, row[name]["candidate_minus_matched"], 'L2 ratio', ratio,
                  'mean spike gap', row[name]["candidate_rate"] - row[name]["condition_rate_mean"], flush=True)
        output[str(birth)] = row
    result = {"method": "attenuate 35% of nominated edge; on same receiver and sign match every counterpart's delta magnitude exactly",
              "held_worlds": list(NEW_WORLD), "data": output}
    args.artifact.write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
