"""Paired clustered summaries and portable outcomes for the sealed gate study."""

import platform
import zipfile

import numpy as np

from . import relation_study, relation_world
from .trust_gate import FEATURES
from .trust_study import ARMS, BIRTHS, DATA, EVIDENCE, REG, root_ids


class Intervals:
    """Resample independent births and complete physical worlds together."""

    def __init__(self, n_births, n_worlds):
        generator = np.random.default_rng(REG["inference"]["bootstrap_seed"])
        draws = REG["inference"]["bootstrap_draws"]
        self.births = generator.multinomial(n_births, np.ones(n_births) / n_births, draws) / n_births
        self.worlds = generator.multinomial(n_worlds, np.ones(n_worlds) / n_worlds, draws) / n_worlds
        self.shape = n_births, n_worlds, 4

    def estimate(self, life):
        x = np.asarray(life).reshape(self.shape)
        finite = np.isfinite(x)
        count = finite.sum(axis=2)
        by_world = np.divide(np.nansum(x, axis=2), count, out=np.zeros_like(count, dtype=float), where=count > 0)
        valid = count > 0
        observed = ((self.births @ by_world) * self.worlds).sum(axis=1)
        if not valid.all():
            available = ((self.births @ valid) * self.worlds).sum(axis=1)
            observed = observed[available > 0] / available[available > 0]
        return dict(mean=float(by_world[valid].mean()),
                    ci95=np.quantile(observed, [.025, .975]).tolist(),
                    by_birth=[float(row[keep].mean()) for row, keep in zip(by_world, valid)])


def _read():
    ids = np.asarray(root_ids(REG["roots"]["holdout"]))
    ordered_parity = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]], dtype=np.int8)
    rows = []
    for front in BIRTHS:
        with np.load(DATA / "holdout" / f"{front}.npz", allow_pickle=False) as case:
            np.testing.assert_array_equal(case["root"], np.repeat(ids, 4))
            np.testing.assert_array_equal(case["parity"], np.tile(ordered_parity, (len(ids), 1)))
            rows.append({k: case[k] for k in case.files if k not in ("root", "parity")})
    if any(set(row) != set(rows[0]) for row in rows):
        raise ValueError("Holdout arrays differ by front")
    return ids, {key: np.stack([row[key] for row in rows]) for key in rows[0]}


def report():
    ids, bank = _read()
    estimate = Intervals(len(BIRTHS), len(ids)).estimate
    histories = relation_world.HISTORIES
    doses = REG["doses"]

    def metric(arm, history, dose, name, field="correct"):
        return bank[f"{arm}/{history}/{dose}/{name}/{field}"]

    curves, gains = {}, {}
    for arm in ARMS:
        curves[arm], gains[arm] = {}, {}
        for h in histories:
            curves[arm][h] = {}
            for d in doses:
                new = metric(arm, h, d, "new")
                old = metric(arm, h, d, "old")
                curves[arm][h][str(d)] = dict(
                    new=estimate(new.mean(2)), old=estimate(old.mean(2)), joint=estimate((new & old).mean(2)),
                    new_contact=estimate((metric(arm, h, d, "new", "choice") >= 0).mean(2)),
                    old_contact=estimate((metric(arm, h, d, "old", "choice") >= 0).mean(2)),
                    new_anchor=estimate(metric(arm, h, d, "new", "anchor").mean(2)),
                    old_anchor=estimate(metric(arm, h, d, "old", "anchor").mean(2)))
            metrics = [metric(arm, h, d, "new").mean(2) for d in (0, 2, 4)]
            gains[arm][h] = (metrics[1] + metrics[2]) / 2 - metrics[0]

    effects, old, differences = {}, {}, {}
    for h in histories:
        effects[h] = {arm: estimate(gains["learned"][h] - gains[arm][h])
                      for arm in ARMS if arm != "learned"}
        differences[h] = dict(always_minus_local=estimate(gains["always"][h] - gains["local"][h]))
        learned_late = metric("learned", h, 16, "old").mean(2)
        old[h] = dict(
            late_versus_initial=estimate(learned_late - metric("learned", h, 0, "old").mean(2)),
            late_versus_always=estimate(learned_late - metric("always", h, 16, "old").mean(2)),
            late_versus_local=estimate(learned_late - metric("local", h, 16, "old").mean(2)))
    pooled = {arm: estimate(np.mean([gains["learned"][h] - gains[arm][h] for h in histories], axis=0))
              for arm in ARMS if arm != "learned"}

    def lower(stat):
        return stat["ci95"][0]

    early = dict(related_retention=lower(effects["related"]["always"]) > -.02,
                 related_positive=lower(effects["related"]["local"]) > 0.,
                 unrelated_relief=lower(effects["unrelated"]["always"]) > 0.,
                 unrelated_noninferiority=lower(effects["unrelated"]["local"]) > -.02)
    specificity = {arm: lower(pooled[arm]) > 0. for arm in ("constant", "swapped")}
    retention = {h: lower(old[h]["late_versus_initial"]) > -.05
                 and lower(curves["learned"][h]["16"]["old"]) > .5 for h in histories}
    decision = dict(early=early, early_joint=all(early.values()), specificity=specificity,
                    specificity_joint=all(specificity.values()), old=retention, old_joint=all(retention.values()))

    gates, coverage, sensitivity = {}, {}, {}
    complete = np.ones((len(BIRTHS), 4 * len(ids)), dtype=bool)
    for h in histories:
        old_taste = bank[f"{h}/history_taste"]
        new_taste = bank[f"{h}/teaching_taste"]
        mask = np.arange(old_taste.shape[2]) % 3 != 0
        complete &= (old_taste[:, :, mask] != 0).all(axis=2)
        complete &= (new_taste[:, :, [1, 2, 4, 5]] != 0).all(axis=2)
        coverage[h] = dict(
            history_signed_receipts=estimate((old_taste != 0).sum(axis=2)),
            b_signed_receipts=estimate((new_taste != 0).sum(axis=2)),
            b_complete_pairs=estimate((new_taste.reshape(*new_taste.shape[:2], -1, 3)[:, :, :, 1:] != 0)
                                      .all(axis=3).sum(axis=2)))
        gates[h] = {}
        for i, step in enumerate(range(2, new_taste.shape[2], 3), 1):
            live = bank[f"learned/{h}/gate"][:, :, step]
            swapped = bank[f"swapped/{h}/gate"][:, :, step]
            evidence = bank[f"learned/{h}/evidence"][:, :, step]
            gates[h][str(i)] = dict(eligible=estimate(np.isfinite(live)), gate=estimate(live),
                                    swapped_gate=estimate(swapped),
                                    live_features={name: estimate(evidence[:, :, j])
                                                   for j, name in enumerate(FEATURES)})
    for h in histories:
        sensitivity[h] = {arm: estimate(np.where(complete, gains["learned"][h] - gains[arm][h], np.nan))
                          for arm in ARMS if arm != "learned"}
    gates["related_minus_unrelated"] = {
        str(i): estimate(bank["learned/related/gate"][:, :, step] - bank["learned/unrelated/gate"][:, :, step])
        for i, step in enumerate(range(2, 3 * doses[-1] // 2, 3), 1)
    }
    same = (bank["related/history_side"] == bank["unrelated/history_side"]).all(axis=2)
    coverage["same_history_contact_sides"] = estimate(same)
    coverage["complete_matched_early_exposure"] = estimate(complete)
    coverage["swap_available_on_recipient_writes"] = {}
    for h, other in (("related", "unrelated"), ("unrelated", "related")):
        written = np.isfinite(bank[f"learned/{h}/gate"][:, :, 2::3])
        donor = np.isfinite(bank[f"learned/{other}/gate"][:, :, 2::3])
        denominator = written.sum(axis=2)
        fraction = np.divide((written & donor).sum(axis=2), denominator,
                             out=np.full_like(denominator, np.nan, dtype=float), where=denominator > 0)
        coverage["swap_available_on_recipient_writes"][h] = estimate(fraction)
    closure = {front: relation_study.read(DATA / "holdout" / f"{front}-closure.json") for front in BIRTHS}
    summary = dict(decision=decision, effects=effects, fixed_prior_effect=differences, pooled=pooled, old=old,
                   curves=curves, early_gains={arm: {h: estimate(v) for h, v in xs.items()}
                                             for arm, xs in gains.items()},
                   gate=gates, coverage=coverage, complete_pair_sensitivity=sensitivity, closure=closure)
    relation_study.write(EVIDENCE / "summary.json", summary)
    archive = EVIDENCE / "paired-outcomes.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
        for front in BIRTHS:
            path = DATA / "holdout" / f"{front}.npz"
            bundle.write(path, path.name)
    relation_study.write(EVIDENCE / "artifacts.json", dict(
        runtime=dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform()),
        files={n: relation_study.digest(EVIDENCE / n) for n in ("protocol.json", "checkpoint.json", "dev-fit.json",
               "seal.json", "sealed-at.json", "summary.json", archive.name)},
        roots=ids.tolist(),
        holdout_rows=int(len(BIRTHS) * len(ids) * 4)))
    print("Joint decision:", decision, flush=True)

