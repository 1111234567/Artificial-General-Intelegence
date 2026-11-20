"""Cross genuine outgoing menu responses with a separately tasted object history."""

from argparse import ArgumentParser
from dataclasses import fields, replace
from itertools import product
from pathlib import Path
import json

import numpy as np
import torch

from . import fork_lesson_fit as fit
from . import fork_lesson_world as world


KINDS = ("whole", "bridge", "reward")
WORLDS = (42887091, 42887129)
ROOTS = 96
ROOTS_PER_BATCH = 8
FACTORS = tuple(product((0, 1), repeat=4))
MODES = ("matched_query", "other_door_query", "same_amplitude")
METRICS = ("target_forecast", "other_forecast", "target_minus_other", "target_motor_logit",
           "actually_chose_target", "settled_return_agreement")
DATA = fit.DATA.parent / "dialogue"
PLAN = fit.EVIDENCE / "dialogue-plan.md"
RESAMPLES = 2000


def rows_for(roots):
    count = len(roots.first)
    e = replace(roots, **{f.name: getattr(roots, f.name).repeat_interleave(len(FACTORS), 0) for f in fields(roots)})
    target, other_public, other_private, target_private = torch.tensor(FACTORS).repeat(count, 1).T
    target_object = column(e.door_order, target)
    values = torch.zeros_like(e.values)
    values = at(values, target_object, 2 * target_private - 1)
    values = at(values, 1 - target_object, 2 * other_private - 1)
    flips = at(e.flips, 1 - target, 2 * other_public - 1)
    return replace(e, values=values, flips=flips), target, target_object


def column(values, where):
    return values.gather(1, where[:, None]).squeeze(1)


def at(values, where, new):
    source = torch.as_tensor(new, dtype=values.dtype, device=values.device)
    source = source.expand(len(values))[:, None]
    return values.scatter(1, where[:, None], source)


def window(model, state, **controls):
    return world.choice_window(model, state, return_outgoing=True, **controls)


def assert_own(model, ready, cue, activity, back, outward):
    own, own_activity, own_back, own_out = window(model, cue, override=back, outbound_override=outward)
    if (not torch.equal(activity, own_activity) or not torch.equal(back, own_back)
            or not torch.equal(outward, own_out)
            or any(not torch.equal(getattr(own, f.name), getattr(ready, f.name)) for f in fields(ready))):
        raise AssertionError("A receiver's own neural currents did not reproduce its decision bit for bit")


def queries(model, donor, e, door):
    result = []
    for sign in (-1., 1.):
        menu = replace(e, flips=at(e.flips, door, sign))
        result.append(window(model, world.cue(model, donor, menu))[3])
    return result


def replies(model, histories, outgoing):
    return torch.stack([torch.stack([window(model, h, outbound_override=q)[2] for h in histories]) for q in outgoing])


def contrast(cells):
    return (cells[1, 1] - cells[1, 0] - cells[0, 1] + cells[0, 0]) / 4


def amplitude(delta):
    return delta.square().mean(dim=(0, -1)).sqrt()


def score_replies(model, feedback, cue, e, target, paid):
    cells = []
    for q in range(2):
        row = []
        for h in range(2):
            state, activity, _, _ = window(model, cue, override=feedback[q, h])
            forecast = fit.forecast_scores(activity)
            choice = world.choose(model, state, e)
            _, outcome = world.commit(model, state, e, choice)
            if not torch.equal(outcome.payoff, column(paid, choice.side)):
                raise AssertionError("A reply intervention changed the immutable payoff of its selected route")
            target_forecast = column(forecast, target)
            other_forecast = column(forecast, 1 - target)
            motor = choice.logit * (2 * target - 1)
            walked_to_target = (choice.side == target).float()
            row.append(torch.stack((target_forecast, other_forecast, target_forecast - other_forecast,
                                    motor, walked_to_target, outcome.payoff), dim=1))
        cells.append(torch.stack(row))
    return torch.stack(cells)


def run_batch(model, roots):
    e, target, target_object = rows_for(roots)
    donor = world.support(model, e)
    cue = world.cue(model, donor, e)
    ready, activity, natural_back, natural_out = window(model, cue)
    assert_own(model, ready, cue, activity, natural_back, natural_out)
    # Commit both routes in the unmodified environment to establish the receiver's payoff table.
    paid = torch.stack([world.commit(model, ready, e, world.choose(model, ready, e, side=side))[1].payoff
                        for side in (0, 1)], dim=1)
    natural_choice = world.choose(model, ready, e)
    _, natural_outcome = world.commit(model, ready, e, natural_choice)
    natural_payoff = natural_outcome.payoff
    if not torch.equal(natural_payoff, column(paid, natural_choice.side)):
        raise AssertionError("Forced and natural rollouts settled the same receiver route differently")
    histories = []
    for sign in (-1., 1.):
        history = replace(e, values=at(e.values, target_object, sign))
        histories.append(world.cue(model, world.support(model, history), e))
    outgoing = queries(model, donor, e, target)
    wrong_out = queries(model, donor, e, 1 - target)
    matching = replies(model, histories, outgoing)
    wrong_door = replies(model, histories, wrong_out)
    delta = matching - natural_back
    permuted = delta.roll(shifts=7, dims=-1) + natural_back
    torch.testing.assert_close((permuted - natural_back).norm(dim=-1), delta.norm(dim=-1), rtol=1e-5, atol=1e-6)

    raw_cells = torch.stack([score_replies(model, reply, cue, e, target, paid)
                             for reply in (matching, wrong_door, permuted)])
    per_row = torch.stack([contrast(cell) for cell in raw_cells])
    strict = paid[:, 0] != paid[:, 1]
    actual_target = column(paid, target)
    # For strict states the two agreeing cells versus the two disagreeing cells equal 2× the signed interaction.
    per_row[:, :, -1] *= 2 * actual_target

    def means(values):
        return values.reshape(*values.shape[:-1], len(roots.first), len(FACTORS)).mean(-1)

    per_root = means(per_row.transpose(1, 2))
    per_root[:, -1] = means(per_row[:, :, -1] * strict) / means(strict.float())
    return {
        "estimate": per_root.permute(0, 2, 1).numpy(),
        "cells": raw_cells.permute(0, 3, 1, 2, 4).numpy(),
        "target": target.numpy(),
        "paid": paid.numpy(),
        "strict": strict.numpy(),
        "natural_target_pick": (natural_choice.side == target).numpy(),
        "natural_payoff": natural_payoff.numpy(),
        "query_change_rms": means(amplitude(outgoing[1] - outgoing[0])).numpy(),
        "wrong_query_change_rms": means(amplitude(wrong_out[1] - wrong_out[0])).numpy(),
        "reply_interaction_rms": means(amplitude(contrast(matching))).numpy(),
        "wrong_reply_interaction_rms": means(amplitude(contrast(wrong_door))).numpy(),
        "reply_total_change_rms": torch.stack([means(amplitude(d)) for q in delta for d in q]).mean(0).numpy(),
    }


@torch.no_grad()
def evaluate(model, seed, roots_total):
    all_roots = world.scenes(seed, roots_total)
    batches = []
    for start in range(0, roots_total, ROOTS_PER_BATCH):
        stop = start + ROOTS_PER_BATCH
        roots = replace(all_roots, **{f.name: getattr(all_roots, f.name)[start:stop] for f in fields(all_roots)})
        batches.append(run_batch(model, roots))
    return {name: np.concatenate([batch[name] for batch in batches], axis=int(name in ("cells", "estimate")))
            for name in batches[0]}


def signature(checkpoint):
    sources = (__file__, PLAN, world.__file__, fit.__file__, fit.PROTOCOL)
    return {"checkpoint_sha256": checkpoint, "sources": {Path(path).name: fit.digest(path) for path in sources}}


def destination(kind, birth, seed):
    return DATA / f"{kind}-{birth}-{seed}.npz"


def run(kind, birth, seed, *, smoke=False):
    model, _, digest = fit.load_final(kind, birth)
    provenance = json.dumps(signature(digest), sort_keys=True)
    path = destination(kind, birth, seed)
    if path.is_file() and not smoke:
        with np.load(path, allow_pickle=False) as cached:
            if cached["signature"].item() != provenance:
                raise ValueError(f"The dialogue artifact has different inputs at {path}")
        return
    result = evaluate(model, seed, ROOTS_PER_BATCH if smoke else ROOTS)
    print(kind, birth, seed, result["estimate"].mean(1)[:, 2].round(5).tolist(), flush=True)
    if not smoke:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, signature=provenance, **result)
        temporary.replace(path)


def summarize_pair(values, generator):
    births, roots = values.shape
    birth_draw = generator.integers(births, size=(RESAMPLES, births))
    root_draw = generator.integers(roots, size=(RESAMPLES, roots))
    samples = values[birth_draw[:, :, None], root_draw[:, None, :]].mean(axis=(1, 2))
    return {"mean": float(values.mean()), "ci95": np.quantile(samples, (.025, .975)).tolist(),
            "birth": values.mean(1).tolist()}


def publish():
    report = {"roots": ROOTS, "rows_per_root": len(FACTORS), "bootstrap_resamples": RESAMPLES,
              "same_history_own_bridge_replay": "bitwise exact in every included row", "results": {}}
    for seed in WORLDS:
        for kind in KINDS:
            samples, diagnostic = [], []
            for birth in fit.BIRTHS:
                path = destination(kind, birth, seed)
                with np.load(path, allow_pickle=False) as source:
                    saved_sig = json.loads(source["signature"].item())
                    active_sig = signature(fit.digest(fit.destination(kind, birth)))
                    if saved_sig != active_sig:
                        raise ValueError(f"The dialogue report has stale provenance at {path}")
                    samples.append(source["estimate"])
                    diagnostic.append({name: float(source[name].mean()) for name in (
                        "query_change_rms", "wrong_query_change_rms", "reply_interaction_rms",
                        "wrong_reply_interaction_rms", "reply_total_change_rms", "natural_payoff", "strict",
                    )})
            samples = np.stack(samples)
            paired = np.concatenate((samples, samples[:, :1] - samples[:, 1:]), axis=1)
            labels = (*MODES, "versus_other_door_query", "versus_same_amplitude")
            generator = np.random.default_rng(seed)
            metrics = {label: {metric: summarize_pair(paired[:, index, :, j], generator)
                               for j, metric in enumerate(METRICS)} for index, label in enumerate(labels)}
            report["results"].setdefault(str(seed), {})[kind] = {"estimates": metrics, "birth_diagnostics": diagnostic}
    target = fit.EVIDENCE / "dialogue-results.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=KINDS)
    parser.add_argument("--birth", type=int, choices=fit.BIRTHS)
    parser.add_argument("--seed", type=int, choices=WORLDS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    kinds = KINDS if args.kind is None else (args.kind,)
    births = fit.BIRTHS if args.birth is None else (args.birth,)
    seeds = WORLDS if args.seed is None else (args.seed,)
    for kind, birth, seed in product(kinds, births, seeds):
        run(kind, birth, seed, smoke=args.smoke)
    if not args.smoke and all(destination(*condition).is_file() for condition in product(KINDS, fit.BIRTHS, WORLDS)):
        publish()


if __name__ == "__main__":
    main()
