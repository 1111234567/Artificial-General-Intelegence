"""Run the fixed side-only return control on the native 40-node trial bodies."""

from argparse import ArgumentParser

import numpy as np
import torch

from . import fork_lesson_bit_relay as relay
from . import fork_lesson_native_request as request
from . import fork_lesson_native_request_fit as fit


BASE = fit.prior
WORLD = request.WORLD
CALIBRATION = (2141086109, 96)
TESTS = ((2141086197, 96), (2141086241, 96))
MODES = ("native", "fixed_side", "blank")
DATA = BASE.DATA.parent / "native-relay"
PLAN = BASE.EVIDENCE / "native-relay-plan.md"


@torch.no_grad()
def calibrate(model):
    seed, roots = CALIBRATION
    total = torch.zeros((2, request.protocol.TRIAL_TICKS, WORLD.REGION), dtype=torch.float64)
    for scene in relay.batches(seed, roots):
        prepared = request.prepare(model, scene)
        blank = request.try_side(model, prepared, relay.side_tensor(scene, 0), mode="blank").returns
        for side in (0, 1):
            own = request.try_side(model, prepared, relay.side_tensor(scene, side)).returns
            total[side] += (own.double() - blank.double()).sum(1)
    return (total / (roots * relay.REPEATS)).float()


def blank_path(model, prepared):
    state = prepared
    n = WORLD.REGION
    inputs = state.h.new_zeros((len(state.h), WORLD.PORTS))
    states, received = [], []
    for tick in range(request.protocol.TRIAL_TICKS):
        outward = None
        if tick == 0:
            outward = request.protocol.bridge_current(model.return_weight * model.mask[n:, :n],
                                                       state.fast[:, n:, :n], state.h[:, :n])
            outward = outward + torch.zeros_like(outward)
        state, back = model.tick(state, inputs, outbound_override=outward)
        states.append(state)
        received.append(back)
    return states, torch.stack(received)


def inject(model, prepared, blank_states, current, template, *, zero_check=False):
    state = prepared
    n = WORLD.REGION
    received = current + template[:, None, :]
    for tick, blank in enumerate(blank_states):
        state = relay.primary_tick(model, state, blank, received[tick])
        if (not torch.equal(state.h[:, n:], blank.h[:, n:])
                or not torch.equal(state.trace[:, n:], blank.trace[:, n:])
                or not torch.equal(state.fast[:, n:], blank.fast[:, n:])):
            raise AssertionError("A replacement trial altered the replica's blank activity, trace, or fast weights")
        if zero_check and not relay.audit.equal_state(state, blank):
            raise AssertionError(f"Zero template failed bitwise blank reproduction at tick {tick}")
        if tick == request.protocol.FORECAST_TICK - 1:
            prediction = state.h[:, request.READOUT]
    return prediction, state


@torch.no_grad()
def batch(model, scene, query_draw, templates):
    prepared = request.prepare(model, scene)
    left, right = (relay.side_tensor(scene, side) for side in (0, 1))
    blank_states, back = blank_path(model, prepared)
    blank = request.try_side(model, prepared, left, mode="blank")
    if not relay.audit.equal_state(blank.final, blank_states[-1]) or not torch.equal(blank.returns, back):
        raise AssertionError("Instrumented blank differs from the native method")
    prediction, final = inject(model, prepared, blank_states, back, torch.zeros_like(templates[0]), zero_check=True)
    if not torch.equal(prediction, blank.forecast) or not relay.audit.equal_state(final, blank.final):
        raise AssertionError("The instrumented primary forecast does not reproduce native blank")

    fixed = [inject(model, prepared, blank_states, back, template) for template in templates]
    # Only now run the recipient's original counterpart trajectories for the positive control.
    own = [request.try_side(model, prepared, side) for side in (left, right)]
    options = {"native": [(entry.forecast, entry.final) for entry in own], "fixed_side": fixed,
               "blank": [(blank.forecast, blank.final)] * 2}
    query = request.select(model, prepared, query_draw, exploration=0.).side
    truth = torch.stack([request.settle(model, blank.final, scene,
                                       request.select(model, blank.final, scene.draws, exploration=0., forced=side))
                         for side in (left, right)], dim=1)
    if not torch.equal(truth, WORLD.oracle(scene)):
        raise AssertionError("Real settlements differ from the independent environment reference")
    scores = {}
    for name, branches in options.items():
        prediction = torch.stack([entry[0] for entry in branches], dim=1)
        states = [entry[1] for entry in branches]
        measures = relay.score_mode(model, states, prediction, model.logit, query, scene, truth)
        scores.update({name + "/" + measure: values for measure, values in measures.items()})
    return scores


def run(birth):
    torch.set_num_threads(1)
    model, _, checkpoint = fit.load("binding", birth)
    if model.width != 40 or len(list(model.modules())) != 1 or request.READOUT != 14:
        raise AssertionError("Expected the frozen 20+20 body with a single original prediction unit")
    sources = (__file__, request.__file__, relay.__file__, WORLD.__file__, PLAN)
    signature = {"checkpoint": checkpoint, "source": BASE.digest(__file__), "plan": BASE.digest(PLAN)}
    templates = calibrate(model)
    cal = DATA / f"calibration-{birth}.npz"
    BASE.storage._write_raw(cal, templates=templates.numpy(), seed=CALIBRATION[0], roots=CALIBRATION[1],
                            source_sha256=[BASE.digest(path) for path in sources], **signature)
    for seed, roots in TESTS:
        query_rng = np.random.default_rng([seed, 582323])
        queries = torch.from_numpy(np.repeat(query_rng.random(roots, dtype=np.float32), relay.REPEATS))
        ledger = []
        start = 0
        for scene in relay.batches(seed, roots):
            end = start + len(scene.first)
            ledger.append(batch(model, scene, queries[start:end], templates))
            start = end
        values = {key: np.concatenate([record[key] for record in ledger]) for key in ledger[0]}
        BASE.storage._write_raw(DATA / f"eval-{seed}-{birth}.npz", **values, calibration=BASE.digest(cal),
                                roots=roots, seed=seed, **signature)
        print(birth, seed, {mode: round(float(values[mode + "/strict_rank"].mean()), 5) for mode in MODES}, flush=True)


def summarize():
    reports = {}
    for seed, roots in TESTS:
        bank, fingerprints = {}, {}
        for birth in fit.BIRTHS:
            path = DATA / f"eval-{seed}-{birth}.npz"
            cal = DATA / f"calibration-{birth}.npz"
            fingerprints[str(birth)] = {"calibration_sha256": BASE.digest(cal), "data_sha256": BASE.digest(path)}
            with np.load(path, allow_pickle=False) as raw:
                if (raw["calibration"].item() != BASE.digest(cal) or raw["source"].item() != BASE.digest(__file__)
                        or raw["plan"].item() != BASE.digest(PLAN) or int(raw["roots"]) != roots):
                    raise AssertionError(f"Evaluation provenance mismatch at {path}")
                if raw["checkpoint"].item() != BASE.digest(fit.artifact("binding", birth)):
                    raise AssertionError(f"Evaluation used a different frozen body at {path}")
                for key in raw.files:
                    if "/" in key:
                        mode, metric = key.split("/")
                        bank.setdefault(mode, {}).setdefault(metric, []).append(raw[key])
        groups = {name: {metric: np.stack(values) for metric, values in group.items()} for name, group in bank.items()}
        contrasts = (("native", "fixed_side"), ("fixed_side", "blank"), ("native", "blank"))
        reports[str(seed)] = {"roots": roots, "raw_sha256": fingerprints,
                              "scores": relay.stats.summarize(groups, roots, fit.BIRTHS, seed, contrasts)}
    path = BASE.EVIDENCE / "native-relay-results.json"
    BASE.storage._write_json(path, {"reports": reports, "calibration": CALIBRATION, "births": fit.BIRTHS,
                                   "nodes": [20, 20], "forecast_unit": request.READOUT,
                                   "replica_frozen_to_blank_each_test_tick": True,
                                   "zero_template_bitwise_reproduced_blank": True})
    print(path, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--birth", type=int, choices=fit.BIRTHS)
    args = parser.parse_args()
    if args.birth:
        run(args.birth)
    else:
        summarize()
