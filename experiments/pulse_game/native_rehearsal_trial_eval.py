"""Freeze both graphs and measure their real actions and the live use of the trial path."""

from argparse import ArgumentParser
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch

from . import edge_site_eval_stats as stats
from . import interleave_revisit_train as storage
from . import native_rehearsal_fit as original
from . import native_rehearsal_trial as trial
from . import native_rehearsal_trial_fit as fit


LIVE = trial.live
N = trial.REGION
ROOTS = 192
BLOCK = 16
REPEATS = 16
SILENT_TICKS = 12
SEEDS = (85419937, 85419959)
MODES = ("natural", "cut_candidate", "birth_only", "twin_candidate", "adoption_off", "transplant_fast", "matched_shuffle")
PHYSICS = ("candidate_l1_sum", "applied_main_l1_sum", "max_final_main_fast_change", "max_final_activity_change")


def writable(state):
    return state.fast[:, :N, :N]


def run_quiet(model, start, *, cut=False, birth_only=False, adoption_off=False, donors=None):
    zero = start.h.new_zeros((len(start.h), 10))
    state = start
    candidates = []
    total = zero.new_zeros((len(zero), 2))
    for tick in range(SILENT_TICKS):
        previous = state
        candidates.append(previous.candidate)
        state, _ = model.tick(previous, zero, read_override=False if cut else None, birth_only=birth_only,
                              adoption_off=adoption_off, candidate_override=None if donors is None else donors[tick])
        delta = writable(state) - writable(previous)
        if (delta.abs() > previous.candidate.abs() + 2e-7).any() or (delta * previous.candidate < 0).any():
            raise AssertionError("A real main-graph write exceeded or reversed the local candidate")
        total += torch.stack((previous.candidate.abs().sum((1, 2)), delta.abs().sum((1, 2))), dim=1)
    if adoption_off and not torch.equal(writable(start), writable(state)):
        raise AssertionError("Closing adoption allowed a primary recurrent write")
    return state, previous, candidates, total


def final_shuffle(before, natural):
    """Exchange neighboring same-sign realized writes whenever each edge can supply the new magnitude."""
    start = writable(before).flatten(1)
    end = writable(natural).flatten(1)
    candidate = before.candidate.flatten(1)
    dose = end - start
    capacity = (start + candidate).clamp(-trial.LIMIT, trial.LIMIT) - start
    transformed = end.clone()
    for sign in (-1, 1):
        ranked = (sign * candidate).masked_fill(sign * dose <= 0, -torch.inf).argsort(dim=1, descending=True)
        pairs = ranked.reshape(len(start), -1, 2)
        existing = dose.gather(1, ranked).reshape_as(pairs)
        swapped = existing.flip(2)
        available = capacity.gather(1, ranked).reshape_as(pairs)
        origins = start.gather(1, ranked).reshape_as(pairs)
        revised = origins + swapped
        actual = revised - origins
        feasible = ((sign * existing > 0) & (sign * swapped <= sign * available)
                    & (sign * actual > 0) & ((actual - swapped).abs() <= 1e-7)).all(2)
        # With variable batch magnitudes the two signs sort independently. Preserve the other sign's moves.
        update = torch.where(feasible[..., None], revised, transformed.gather(1, ranked).reshape_as(pairs))
        transformed = transformed.scatter(1, ranked, update.flatten(1))
    after = transformed - start
    imbalance = torch.stack(((after.clamp_min(0) - dose.clamp_min(0)).double().sum(1).abs(),
                             (after.clamp_max(0) - dose.clamp_max(0)).double().sum(1).abs()), dim=1)
    if imbalance.max() > 2e-6 or not torch.equal(after.ne(0).sum(1), dose.ne(0).sum(1)):
        raise AssertionError("The final same-dose shuffle changed signed write totals or counts")
    moved = (transformed != end).sum(1)
    fast = natural.fast.clone()
    fast[:, :N, :N] = transformed.reshape(-1, N, N)
    return replace(natural, fast=fast), moved, imbalance


def contact_sides(e, visits):
    sides = torch.empty_like(e.laws)
    for obj, _, side, _ in visits:
        sides.scatter_(1, obj[:, None], side[:, None])
    return sides


@torch.no_grad()
def one_block(model, e):
    _, start, _, visits = LIVE.support(model, e)
    present_side = contact_sides(e, visits)
    twin = torch.arange(len(start.h)) ^ 12
    if not (torch.equal(e.vectors[:, 1:], e.vectors[twin, 1:]) and torch.equal(e.laws[:, 1:], e.laws[twin, 1:])
            and torch.equal(e.laws[:, 0], -e.laws[twin, 0])):
        raise AssertionError("Candidate donors must share B/C and reverse the older A condition")
    natural, previous, drafts, dose = run_quiet(model, start)
    alt = [run_quiet(model, start, **options) for options in (
        dict(cut=True), dict(cut=True, birth_only=True), dict(donors=[value[twin] for value in drafts]),
        dict(adoption_off=True),
    )]
    cut = alt[0][0]
    transplant = replace(natural, fast=cut.fast)
    mixed, moved, imbalance = final_shuffle(previous, natural)
    states = [natural, *(a[0] for a in alt), transplant, mixed]
    physical = [dose, *(a[3] for a in alt), alt[0][3], dose]
    order = torch.stack((e.first_query, 1 - e.first_query), dim=1)
    actual_targets = e.orientation[:, 3:5].gather(1, order) * e.laws[:, (0, 2)]
    if not model.read_candidate:
        for control in (cut, alt[2][0]):
            if any(not torch.equal(getattr(natural, f.name), getattr(control, f.name)) for f in fields(trial.State)):
                raise AssertionError("An unread draft changed the no-trial computation")
    choices = []
    physical_metrics = []
    for state, step_dose in zip(states, physical, strict=True):
        choice, _, payout, _ = LIVE.queries(model, state, e)
        if not torch.equal(choice * actual_targets, payout):
            raise AssertionError("Actual query payout does not match the choice and environment law")
        choices.append(torch.stack((choice, payout), dim=1))
        physical_metrics.append(torch.cat((step_dose,
                                           (writable(state) - writable(natural)).abs().flatten(1).amax(1)[:, None],
                                           (state.h - natural.h).abs().amax(1)[:, None]), dim=1))
    return dict(readings=torch.stack(choices).to(torch.int8).numpy(),
                physics=torch.stack(physical_metrics).numpy(),
                support_sides=present_side.to(torch.int8).numpy(),
                same_support=present_side.eq(present_side[twin]).numpy(),
                shuffled_addresses=moved.numpy(), shuffle_error=imbalance.numpy())


def evaluate(model, world):
    records = []
    count = len(world.laws)
    for start in range(0, count, BLOCK * REPEATS):
        subset = slice(start, start + BLOCK * REPEATS)
        e = LIVE.Episode(*(getattr(world, field.name)[subset] for field in fields(LIVE.Episode)))
        records.append(one_block(model, e))
    rows = {name: np.concatenate([record[name] for record in records], axis=1 if name in {"readings", "physics"} else 0)
            for name in records[0]}
    if not np.isin(rows["readings"], [-1, 1]).all():
        raise AssertionError("A formal query did not record a real side and payout")
    return rows


def rates_and_deltas(raw, e, births):
    close = (e.angle.reshape(-1, REPEATS)[:, 0] <= 30).numpy()
    laws = e.laws.reshape(-1, REPEATS, 3).numpy()
    conflict = laws[:, :, 0] != laws[:, :, 2]
    reads = tuple((kind, mode) for kind in trial.KINDS for mode in MODES)

    def scores(kind, birth, mode):
        payouts = raw[kind, birth]["readings"][MODES.index(mode), :, 1, :]
        return stats.score(payouts.reshape(-1, REPEATS, 2) > 0)

    report = {}
    for relation, matching in (("conflict", conflict), ("agree", ~conflict)):
        matrix = np.stack([[scores(kind, birth, mode)[matching].reshape(len(close), -1, 3)[close].mean(1)
                           for kind, mode in reads] for birth in births]).transpose(0, 2, 1, 3)
        group_count = matrix.shape[1]
        bootstrap = stats.CrossedBootstrap(len(births), group_count)
        comparisons = [("trial_minus_no_trial", ("trial", "natural"), ("no_trial", "natural"))]
        comparisons.extend((f"{kind}_natural_minus_{mode}", (kind, "natural"), (kind, mode))
                           for kind in trial.KINDS for mode in MODES if mode != "natural")
        joined = np.concatenate((matrix, np.stack([matrix[:, :, reads.index(a)] - matrix[:, :, reads.index(b)]
                                                    for _, a, b in comparisons], axis=2)), axis=2)
        intervals = bootstrap.intervals(joined.astype(np.float32))
        mean = joined.mean(axis=(0, 1))
        by_birth = joined.mean(axis=1)

        def row(i):
            return {key: dict(mean=float(mean[i, j]), ci95=intervals[:, i, j].tolist(),
                              per_birth=by_birth[:, i, j].tolist()) for j, key in enumerate(stats.METRICS)}

        report[relation] = dict(geometry_roots=group_count, rates={"/".join(read): row(i) for i, read in enumerate(reads)},
                                contrasts={name: row(len(reads) + i) for i, (name, _, _) in enumerate(comparisons)})

    physical = {}
    for kind in trial.KINDS:
        values = [raw[kind, birth] for birth in births]
        both_match = [item["same_support"][:, 1:].all(1).reshape(-1, REPEATS) & conflict & close[:, None] for item in values]
        b_match = [item["same_support"][:, 2].reshape(-1, REPEATS) & conflict & close[:, None] for item in values]

        def matched_rates(masks):
            arrays = [[scores(kind, birth, mode)[mask] for birth, mask in zip(births, masks, strict=True)]
                      for mode in ("natural", "twin_candidate")]
            return dict(lanes_per_birth=[int(mask.sum()) for mask in masks],
                        rates={mode: dict(zip(stats.METRICS, np.concatenate(a).mean(0).tolist(), strict=True))
                               for mode, a in zip(("natural", "twin_candidate"), arrays, strict=True)})

        rollout = np.concatenate([item["physics"] for item in values], axis=1)
        physical[kind] = dict(same_b_c_fraction=np.concatenate([item["same_support"] for item in values])[:, 1:].all(1).mean(),
                              same_b_only=matched_rates(b_match), same_b_and_c=matched_rates(both_match),
                              average_physics={mode: dict(zip(PHYSICS, values.tolist(), strict=True))
                                               for mode, values in zip(MODES, rollout.mean(axis=1), strict=True)},
                              maximum_shuffle_error=max(float(item["shuffle_error"].max()) for item in values),
                              shuffled_addresses_mean=float(np.concatenate([item["shuffled_addresses"] for item in values]).mean()))
    report["physics"] = physical
    return report


def run(smoke=False):
    torch.set_num_threads(1)
    births, seeds, count = (original.BIRTHS[:1], SEEDS[:1], 16) if smoke else (original.BIRTHS, SEEDS, ROOTS)
    evidence = dict(births=births, seeds=seeds, modes=MODES, roots=count, scored_routes_per_birth=original.STEPS * original.BATCH,
                    plan_sha256=original.digest(fit.PLAN), source_sha256=original.digest(Path(__file__)), panels={}, fits={})
    folder = original.DATA / "trial-eval"
    if not smoke:
        folder.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        world = LIVE.scenes(seed, count, evaluation=True)
        raw, sealed = {}, {}
        for birth in births:
            for kind in trial.KINDS:
                model = trial.TrialGraph(birth, kind).eval()
                if not smoke:
                    model, ledger, sha = fit.load_final(kind, birth)
                    evidence["fits"][f"{kind}/{birth}"] = dict(sha256=sha, final_128=ledger[-128:].mean(0).tolist())
                row = evaluate(model, world)
                raw[kind, birth] = row
                if not smoke:
                    path = folder / f"{seed}-{kind}-{birth}.npz"
                    storage._write_raw(path, **row, kind=kind, birth=birth, seed=seed, mode_names=MODES,
                                       dose_names=PHYSICS, vectors=world.vectors.numpy(), laws=world.laws.numpy(),
                                       draws=world.draws.numpy(), orientation=world.orientation.numpy(),
                                       first_old=world.first_old.numpy(), first_query=world.first_query.numpy(),
                                       angle=world.angle.numpy())
                    sealed[path.name] = original.digest(path)
                print(seed, kind, birth, "scored", flush=True)
        evidence["panels"][str(seed)] = dict(files=sealed, **rates_and_deltas(raw, world, births))
    if not smoke:
        storage._write_json(original.EVIDENCE / "trial-results.json", evidence)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args().smoke)
