"""Replace a frozen replica's response with a context-independent side template."""

from argparse import ArgumentParser
from dataclasses import fields

import numpy as np
import torch
from torch.nn import functional as F

from . import fork_lesson_fit as base
from . import fork_lesson_whatif as trial
from . import fork_lesson_whatif_fit as fit
from . import fork_lesson_whatif_fit_audit as audit
from . import fork_lesson_world as world
from . import noon_market_stats as stats


CALIBRATION = (2130712610, 96)
TESTS = ((2130712749, 144), (2130712893, 96))
REPEATS = 64
BATCH_ROOTS = 4
MODES = ("own", "fixed_bit", "blank", "bridge")
DATA = base.DATA.parent / "bit-relay"
PLAN = base.EVIDENCE / "bit-relay-plan.md"


def batches(seed, roots):
    first = world.scenes(seed, roots)
    factors = torch.from_numpy(np.indices((2,) * 6).reshape(6, -1).T.copy())
    for begin in range(0, roots, BATCH_ROOTS):
        end = min(begin + BATCH_ROOTS, roots)
        repeated = {field.name: getattr(first, field.name)[begin:end].repeat_interleave(REPEATS, 0)
                    for field in fields(first)}
        v0, v1, order, left, right, history = factors.repeat(end - begin, 1).T
        repeated.update(values=torch.stack((v0, v1), 1).float() * 2 - 1, first=history,
                        door_order=torch.stack((order, 1 - order), 1),
                        flips=torch.stack((left, right), 1).float() * 2 - 1)
        yield world.Episode(**repeated)


def side_tensor(e, side):
    return torch.full_like(e.first, side)


def blank_path(model, prepared):
    """Save the complete no-query process in an independently evolving body."""
    n = world.REGION
    body = model.body
    inward = body.weight[:, n:] * body.mask[:n, n:]
    outward = body.return_weight * body.mask[n:, :n]
    state = prepared
    inputs = state.h.new_zeros((len(state.h), world.PORTS))
    states, feedback = [], []
    for tick in range(trial.TRIAL_TICKS):
        primary, replica = state.h.split(n, 1)
        outbound = None
        if tick == 0:
            outbound = trial.bridge_current(outward, state.fast[:, n:, :n], primary)
            outbound = outbound + torch.zeros_like(outbound)
        back = trial.bridge_current(inward, state.fast[:, :n, n:], replica)
        if tick:
            back = back + F.linear(replica[:, trial.MOTOR_PORTS], model.relay)
        state, returned = body.tick(state, inputs, outbound_override=outbound, feedback_override=back)
        states.append(state)
        feedback.append(returned)
    return states, torch.stack(feedback)


def primary_tick(body, state, replica_blank, feedback):
    """Advance real primary units using the blank replica at every within-tick read."""
    n = world.REGION
    weights = body.weight[:, :n] * body.mask[:n, :n]
    own = F.linear(state.h[:, :n], weights)
    own = own + torch.bmm(state.fast[:, :n, :n], state.h[:, :n, None]).squeeze(2)
    own = own + feedback
    current = torch.cat((own, torch.zeros_like(own)), dim=1)
    target = torch.tanh(current + body.bias + torch.zeros_like(state.h))
    changed = state.h + body.leak_logit.sigmoid() * (target - state.h)
    h = torch.cat((changed[:, :n], replica_blank.h[:, n:]), dim=1)
    trace = state.trace + body.keep_logit.neg().sigmoid() * (state.h - state.trace)
    trace = torch.cat((trace[:, :n], replica_blank.trace[:, n:]), dim=1)

    replica_now = h[:, n:]
    receiver = body.adopt_bias + body.adopt_receiver * replica_now
    sender = body.adopt_sender * replica_now
    accepted = state.candidate * (receiver[:, :, None] + sender[:, None, :]).sigmoid()
    modulation = F.linear(h, body.gate * body.mask).tanh()
    post = (h + body.post_bias).tanh() * modulation * body.edit_rate.sigmoid()
    revised = post[:, :, None] * trace[:, None, :] * body.mask
    proposed = torch.cat((accepted, revised[:, :n, n:]), dim=2)
    fast = (state.fast[:, :n] + proposed).clamp(-world.LIMIT, world.LIMIT)
    fast = torch.cat((fast, replica_blank.fast[:, n:]), dim=1)
    return world.State(h, fast, trace, revised[:, :n, :n])


def inject(model, prepared, blank_states, feedback, template, *, assert_all_blank=False):
    state = prepared
    n = world.REGION
    injected = feedback + template[:, None, :]
    for tick, blank in enumerate(blank_states):
        state = primary_tick(model.body, state, blank, injected[tick])
        if (not torch.equal(state.h[:, n:], blank.h[:, n:])
                or not torch.equal(state.trace[:, n:], blank.trace[:, n:])
                or not torch.equal(state.fast[:, n:], blank.fast[:, n:])):
            raise ValueError("A replica state changed from its independent blank trajectory")
        if assert_all_blank and not audit.equal_state(state, blank):
            largest = {field.name: float((getattr(state, field.name) - getattr(blank, field.name)).abs().max())
                       for field in fields(state)}
            raise ValueError(f"Primary-only calculation failed to reproduce the blank at tick {tick}: {largest}")
        if tick == trial.FORECAST_TICK - 1:
            prediction = model.forecast(state)
    return prediction, state


@torch.no_grad()
def calibrate(model):
    seed, roots = CALIBRATION
    total = torch.zeros((2, trial.TRIAL_TICKS, world.REGION), dtype=torch.float64)
    for e in batches(seed, roots):
        state = trial.start_trial(model, e)
        blank = trial.perform_trial(model, state, side_tensor(e, 0), blank=True).feedback
        for side in (0, 1):
            real = trial.perform_trial(model, state, side_tensor(e, side)).feedback
            total[side] += (real.double() - blank.double()).sum(1)
    return (total / (roots * REPEATS)).float()


def settled(body, state, e, logit, side):
    zero = torch.zeros_like(logit)
    _, outcome = world.commit(body, state, e, world.Choice(logit, side, zero, zero))
    return outcome.payoff


def score_mode(body, final_states, forecast, motor, q, e, truth):
    rows = torch.arange(len(q))
    error = (forecast - truth).square().mean(1)
    strict = truth[:, 0] != truth[:, 1]
    signed_gap = (forecast[:, 1] - forecast[:, 0]) * truth[:, 1]
    rank = (signed_gap > 0).float() + 0.5 * (signed_gap == 0)
    stochastic, greedy = [], []
    for side, state in enumerate(final_states):
        logit = motor(state)
        actual = (e.draws < logit.sigmoid()).long()
        best_guess = (logit > 0).long()
        stochastic.append(settled(body, state, e, logit, actual))
        greedy.append(settled(body, state, e, logit, best_guess))
        if not torch.equal(stochastic[-1], truth[rows, actual]) or not torch.equal(greedy[-1], truth[rows, best_guess]):
            raise ValueError(f"Physical branch at test side {side} failed to reach the correct payout")

    def reduce(values, mask=None):
        values = values if mask is None else values[mask]
        return values.float().reshape(len(q) // REPEATS, -1).mean(1).numpy()

    stochastic = torch.stack(stochastic, dim=1)
    greedy = torch.stack(greedy, dim=1)
    return {
        "mse": reduce(error), "strict_rank": reduce(rank, strict), "strict_signed_gap": reduce(signed_gap, strict),
        "strict_preferred": reduce(stochastic[rows, q] > 0, strict),
        "strict_preferred_greedy": reduce(greedy[rows, q] > 0, strict),
        "strict_forecast_pick": reduce(truth[rows, (forecast[:, 1] > forecast[:, 0]).long()] > 0, strict),
    }


@torch.no_grad()
def test_batch(model, bridge, e, templates, query_draw):
    prepared = trial.start_trial(model, e)
    states, feedback = blank_path(model, prepared)
    blank = trial.perform_trial(model, prepared, side_tensor(e, 0), blank=True)
    if not audit.equal_state(states[-1], blank.final_state) or not torch.equal(feedback, blank.feedback):
        raise ValueError("Independent blank path did not reproduce the trainer's native blank")
    inject(model, prepared, states, feedback, torch.zeros_like(templates[0]), assert_all_blank=True)
    # Both substitutions are complete before we ever run the two positive-control queries.
    replacement = [inject(model, prepared, states, feedback, template) for template in templates]
    own = [trial.perform_trial(model, prepared, side_tensor(e, side)) for side in (0, 1)]
    options = {
        "fixed_bit": replacement,
        "own": [(entry.forecast, entry.final_state) for entry in own],
        "blank": [(blank.forecast, blank.final_state)] * 2,
    }
    motor = model.motor_logit
    q = (query_draw < model.query_logit(prepared).sigmoid()).long()
    z = motor(blank.final_state)
    truth = torch.stack([settled(model.body, blank.final_state, e, z, side_tensor(e, side))
                         for side in (0, 1)], dim=1)
    output = {name: score_mode(model.body, [item[1] for item in entries],
                               torch.stack([item[0] for item in entries], dim=1), motor, q, e, truth)
              for name, entries in options.items()}
    bridge_cue = world.cue(bridge, world.support(bridge, e), e)
    bridge_final, activities, _ = world.choice_window(bridge, bridge_cue)
    output["bridge"] = score_mode(bridge, [bridge_final] * 2, base.forecast_scores(activities), bridge.logit, q, e, truth)
    return {name + "/" + metric: value for name, results in output.items() for metric, value in results.items()}


def run(birth):
    torch.set_num_threads(1)
    model, _, anchor = fit.load_final("binding", birth)
    bridge, _, bridge_anchor = base.load_final("bridge", birth)
    model.requires_grad_(False)
    bridge.requires_grad_(False)
    provenance = {"binding_sha256": anchor, "bridge_sha256": bridge_anchor, "plan_sha256": base.digest(PLAN),
                  "source_sha256": base.digest(__file__)}
    templates = calibrate(model)
    cal_artifact = DATA / f"calibration-{birth}.npz"
    base.storage._write_raw(cal_artifact, templates=templates.numpy(), seed=CALIBRATION[0], roots=CALIBRATION[1],
                            **provenance)
    for seed, roots in TESTS:
        rng = np.random.default_rng([seed, 582323])
        queries = torch.from_numpy(np.repeat(rng.random(roots, dtype=np.float32), REPEATS))
        results = []
        index = 0
        for e in batches(seed, roots):
            n = len(e.first)
            results.append(test_batch(model, bridge, e, templates, queries[index:index + n]))
            index += n
        raw = {key: np.concatenate([result[key] for result in results]) for key in results[0]}
        snapshot = DATA / f"eval-{seed}-{birth}.npz"
        base.storage._write_raw(snapshot, **raw, calibration_sha256=base.digest(cal_artifact), seed=seed, roots=roots,
                                **provenance)
        print(birth, seed, {mode: round(float(raw[mode + "/strict_rank"].mean()), 5) for mode in MODES}, flush=True)


def summarize():
    output = {}
    for seed, roots in TESTS:
        bank, sha = {}, {}
        for birth in base.BIRTHS:
            artifact = DATA / f"eval-{seed}-{birth}.npz"
            cal = DATA / f"calibration-{birth}.npz"
            sha[str(birth)] = {"calibration": base.digest(cal), "test": base.digest(artifact)}
            with np.load(artifact, allow_pickle=False) as data:
                if (str(data["plan_sha256"]) != base.digest(PLAN) or str(data["source_sha256"]) != base.digest(__file__)
                        or str(data["calibration_sha256"]) != sha[str(birth)]["calibration"]):
                    raise ValueError(f"Inconsistent frozen provenance in {artifact}")
                for key in data.files:
                    if "/" not in key:
                        continue
                    mode, metric = key.split("/")
                    bank.setdefault(mode, {}).setdefault(metric, []).append(data[key])
        groups = {mode: {metric: np.stack(values) for metric, values in group.items()} for mode, group in bank.items()}
        contrasts = (("own", "fixed_bit"), ("fixed_bit", "blank"), ("own", "bridge"), ("fixed_bit", "bridge"))
        output[str(seed)] = {"roots": roots, "sha256": sha, "scores": stats.summarize(groups, roots, base.BIRTHS,
                                                                                  seed, contrasts)}
    path = base.EVIDENCE / "bit-relay-results.json"
    base.storage._write_json(path, {"calibration": CALIBRATION, "plan_sha256": base.digest(PLAN),
                                   "blank_replica_and_zero_template_verified_each_test_tick": True,
                                   "model_births": base.BIRTHS, "reports": output})
    print(path, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--birth", type=int, choices=base.BIRTHS)
    args = parser.parse_args()
    if args.birth:
        run(args.birth)
    else:
        summarize()
