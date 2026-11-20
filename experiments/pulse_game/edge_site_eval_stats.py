"""Crossed inference for the frozen edge-site evaluation."""

import numpy as np


METRICS = ("a", "b", "both")
N_BOOT = 8192
BOOT_SEED = 5814903
B_MARGIN = -0.05


class CrossedBootstrap:
    def __init__(self, births, roots):
        rng = np.random.default_rng(BOOT_SEED)
        self.birth = self.weights(rng, births)
        self.root = self.weights(rng, roots)

    @staticmethod
    def weights(rng, count):
        return rng.multinomial(count, np.full(count, 1 / count), size=N_BOOT).astype(np.float32) / count

    def intervals(self, means):
        n_birth, n_root = self.birth.shape[1], self.root.shape[1]
        if means.shape[:2] != (n_birth, n_root):
            raise ValueError("Each supported statistic requires all crossed birth and geometry cells")
        counts = means.shape[2:]
        scores = means.reshape(n_birth, n_root, -1).transpose(0, 2, 1).copy()
        sampled = scores.reshape(-1, n_root) @ self.root.T
        sampled = sampled.reshape(n_birth, -1, N_BOOT)
        draws = np.einsum("bsi,ib->si", sampled, self.birth, dtype=np.float64)
        return np.quantile(draws, (0.025, 0.975), axis=1).reshape(2, *counts)


def score(choices):
    return np.concatenate((choices, choices.all(axis=-1, keepdims=True)), axis=-1)


def summarize(scores, mode_names, bootstrap):
    """Input shape is birth × geometry × repeat × arm × actual metric."""
    if scores.ndim != 5 or scores.dtype != np.bool_ or scores.shape[-1] != len(METRICS):
        raise ValueError(f"Unexpected actual-choice array {scores.shape} {scores.dtype}")
    means = scores.mean(2, dtype=np.float32)
    actual_full = means[:, :, mode_names.index("full"), None, :]
    control_names = tuple(mode for mode in mode_names if mode != "full")
    control_index = tuple(mode_names.index(mode) for mode in control_names)
    diffs = actual_full - means[:, :, control_index, :]
    intervals = bootstrap.intervals(np.concatenate((means, diffs), axis=2))
    denom = int(np.prod(scores.shape[:3]))
    counts = scores.sum((0, 1, 2))
    birth_counts = scores.sum((1, 2))
    per_birth = means.mean(axis=1, dtype=np.float64)
    birth_diff = diffs.mean(axis=1, dtype=np.float64)

    def interval(i, j, by_birth):
        return dict(mean=float(by_birth.mean()), ci95=intervals[:, i, j].tolist(),
                    per_birth=by_birth.tolist(), birth_range=[float(by_birth.min()), float(by_birth.max())])

    rates = {}
    for i, mode in enumerate(mode_names):
        rates[mode] = {}
        for j, metric in enumerate(METRICS):
            rates[mode][metric] = dict(interval(i, j, per_birth[:, i, j]),
                                       successes=int(counts[i, j]), denominator=denom,
                                       successes_per_birth=birth_counts[:, i, j].tolist(),
                                       denominator_per_birth=denom // len(scores))
    contrasts = {}
    for i, control in enumerate(control_names):
        contrasts[f"full_minus_{control}"] = {
            metric: interval(len(mode_names) + i, j, birth_diff[:, i, j])
            for j, metric in enumerate(METRICS)
        }
    return dict(rates=rates, contrasts=contrasts)


def primary_gates(summary):
    native = summary["contrasts"]["full_minus_native"]
    random = summary["contrasts"]["full_minus_random"]
    gates = dict(
        a_superior_to_native=native["a"]["ci95"][0] > 0,
        both_superior_to_native=native["both"]["ci95"][0] > 0,
        b_noninferior_to_native=native["b"]["ci95"][0] > B_MARGIN,
        both_superior_to_random=random["both"]["ci95"][0] > 0,
    )
    return dict(gates, all_pass=all(gates.values()), b_risk_margin=B_MARGIN)
