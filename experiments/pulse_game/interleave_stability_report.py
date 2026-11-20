"""Recompute paired effects of slow references from stored choices of individual lives."""

import json
from hashlib import sha256

import numpy as np

from .interleave_capacity_report import Crossed
from .interleave_interference import DATA, EVIDENCE


STAGES = (16, 32, 64)  # Blocks of four physical contacts.
N_BOOT = 4000


def mixed_rows(summary, label, birth):
    filename = f"{birth}.npz"
    path = DATA / "integrated-stability" / label / filename
    if sha256(path.read_bytes()).hexdigest() != summary["sha256"][filename]:
        raise ValueError(f"Unexpected trace at {path}")
    blocks = summary["blocks"]
    names = summary["metric_names"]
    rows = {}
    with np.load(path) as trace:
        for branch in ("native", "birth", "slow"):
            probe = trace[branch + "/probe"]
            scores = {}
            for metric in ("old_after_fresh", "fresh", "ctx_both_mark", "natural_quiz"):
                stages = []
                for stage in STAGES:
                    cut = blocks.index(stage)
                    if metric == "natural_quiz":
                        first, end = 4 * blocks[cut - 1], 4 * stage
                        quizzes = np.arange(first, end) % 4 >= 2
                        values = trace[branch + "/earned"][:, first:end][:, quizzes].mean(1)
                    else:
                        values = probe[cut, names.index(metric)].mean(1)
                    if metric == "ctx_both_mark":
                        seen = (trace["context_firsts"] < 4 * stage).all(1)
                        values = np.where(seen & trace["ctx_conflict"], values, np.nan)
                    stages.append(values.reshape(-1, 2))
                scores[metric] = np.stack(stages)
                observed = np.mean([np.nanmean(s) for s in stages])
                expected = np.mean([summary["roots"][str(birth)][f"{branch}/{k}"][metric] for k in STAGES])
                if not np.isclose(observed, expected, rtol=0, atol=1e-12):
                    raise ValueError(f"Published and raw rates disagree for {label}/{birth}/{branch}/{metric}")
            rows[branch] = scores
    return rows


def summarize_mixed(label):
    summary = json.loads((EVIDENCE / f"integrated-stability-{label}.json").read_text())
    roots = [mixed_rows(summary, label, b) for b in summary["births"]]
    count = len(roots)
    n = summary["independent_worlds_per_birth"]
    rng = np.random.default_rng(583831)
    draws = rng.integers(count, size=(N_BOOT, count))
    # Training births have independently generated geometries. Keep both sign
    # histories, the same repeated life and all matched branches together.
    world_weights = rng.multinomial(n, np.full(n, 1 / n), size=(count, N_BOOT))

    def rate(a, b, metric):
        per_birth = []
        point = []
        for i, root in enumerate(roots):
            x = root[a][metric]
            if b is not None:
                x = x - root[b][metric]
            valid = np.isfinite(x)
            totals = world_weights[i] @ np.nan_to_num(x).sum(2).T
            counts = world_weights[i] @ valid.sum(2).T
            if np.any(counts == 0):
                raise ValueError("A bootstrap stratum contains no eligible context")
            per_birth.append((totals / counts).mean(1))
            point.append(float(np.nanmean(x, axis=(1, 2)).mean()))
        boot = np.stack(per_birth)[draws, np.arange(N_BOOT)[:, None]].mean(1)
        return dict(mean=float(np.mean(point)), ci95=np.quantile(boot, [.025, .975]).tolist(), by_birth=point)

    comparisons = (("native", None), ("birth", None), ("slow", None),
                   ("birth", "native"), ("slow", "native"), ("slow", "birth"))
    return {a if b is None else f"{a} minus {b}":
            {m: rate(a, b, m) for m in roots[0][a]} for a, b in comparisons}


def summarize_local():
    result = json.loads((EVIDENCE / "local-consolidation.json").read_text())
    protocol = result["plan"]
    stages = [protocol["ages"].index(k) for k in (16, 64, 128)]
    roots = protocol["test_births"]
    intervals = Crossed(len(roots), protocol["test_worlds"], 81381, N_BOOT)
    raw = {}
    for branch in ("birth", "global", "local", "permuted", "mean", "opposite_A"):
        tables = []
        for root in roots:
            key = f"test-{root}-absent_a-4-{branch}.npz"
            f = DATA / "local-consolidation" / key
            if sha256(f.read_bytes()).hexdigest() != result["confirmation_raw_sha256"][key]:
                raise ValueError(f"Unexpected trace at {f}")
            with np.load(f) as trace:
                tables.append(trace["correct"][stages].mean((0, 2, 3)))
        raw[branch] = np.stack(tables)
    scores = dict(raw)
    for key in ("global", "permuted", "mean", "opposite_A"):
        scores["local minus " + key] = raw["local"] - raw[key]
    return {label: {name: intervals(table[:, :, i]) for i, name in enumerate(protocol["metrics"])}
            for label, table in scores.items()}


if __name__ == "__main__":
    labels = [f"{family}-{cohort}" for family in ("joint", "uniform", "quest")
              for cohort in ("original", "confirmation")]
    result = dict(
        draws=N_BOOT,
        methods="Mixed lives: equal weights at 64, 128 and 256 contacts, each metric paired by world "
                "and training birth. Both signed histories of the same world stay together; worlds are "
                "nested within birth. Natural quiz uses the interval ending at each age. Conflict probes "
                "are included only after all three types were actually observed, just as in the source. "
                "Local slow rates: 16, 64 and 128 contacts, initially four A/B reviews and no further A. "
                "The local experiment has shared geometries crossed with training birth; all laws stay paired. "
                "All intervals are pointwise.",
        mixed={label: summarize_mixed(label) for label in labels},
        local=summarize_local(),
    )
    (EVIDENCE / "stability-paired.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf8")
