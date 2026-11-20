"""Run the frozen ticket bodies through continuous food and price decisions."""

from argparse import ArgumentParser
from io import BytesIO
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import torch

from . import interleave_revisit_train as io
from . import noon_coverage as coverage, noon_ticket as ticket
from . import noon_transfer as card, noon_transfer_archive as archive, noon_transfer_custody as custody
from . import noon_transfer_ticket as models


PLAN = card.OUT / "transfer-lifetime-plan.json"
OUTPUT = card.DATA / "transfer" / "lifetime"
NET = ticket.net
MODES = ("natural", "always_buy", "always_pass", "matched_shuffle")


def make_tape(plan, seed, worlds):
    """Share prices and physical randomness across four private histories."""
    streams = np.random.SeedSequence([plan["domain"], seed]).spawn(5)
    geometry, public, private, draws, shuffles = (np.random.default_rng(x) for x in streams)
    steps = plan["opportunities"]
    recurring = plan["recurring_objects"]
    block = plan["block_size"]
    new = plan["new_targets_per_block"]
    n_initial = plan["initial_contacts"]
    n_distractors = steps // plan["distractor_contact_every"]
    novel = steps // block * new
    total = recurring + novel + n_distractors
    histories = plan["private_histories_per_geometry"]
    scale = 1.35
    axis = geometry.normal(size=(worlds, total, 3))
    axis = (scale * axis / np.linalg.norm(axis, axis=2, keepdims=True)).astype(np.float32)

    intro = public.permuted(np.broadcast_to(np.arange(recurring), (worlds, recurring)), axis=1)[:, :n_initial]
    targets = []
    for offset in range(0, novel, new):
        ids = np.r_[np.arange(recurring), np.arange(recurring + offset, recurring + offset + new)]
        targets.append(public.permuted(np.broadcast_to(ids, (worlds, block)), axis=1))
    targets = np.concatenate(targets, axis=1)
    distractors = np.broadcast_to(np.arange(recurring + novel, total), (worlds, n_distractors))
    law = np.repeat((2 * private.integers(0, 2, (worlds, total)) - 1).astype(np.float32), histories, axis=0)
    crosses = np.tile(np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], dtype=np.float32), (worlds, 1))
    law[:, :recurring] *= crosses[:, 0, None]
    law[:, recurring:] *= crosses[:, 1, None]
    world = np.arange(worlds)[:, None]

    def share(value):
        return np.repeat(value, histories, axis=0)

    def sign(shape):
        return (2 * public.integers(0, 2, shape) - 1).astype(np.float32)

    def sense(ids, orient):
        smell = share(axis[world, ids] * orient[..., None])
        value = np.take_along_axis(law, share(ids), axis=1) * share(orient)
        return smell, value

    price = public.choice(plan["prices_cents"], size=(worlds, steps)).astype(np.int16)
    price[:, 1] = price[:, 0]
    price = share(price)
    ticket_side = share(sign((worlds, steps)))
    center = np.zeros((*price.shape, 3), dtype=np.float32)
    center[..., 2] = 2 * (price.astype(np.float32) / 100)
    mark = np.zeros_like(center)
    mark[..., 0] = 1.25 * ticket_side
    initial_x, initial_value = sense(intro, sign((worlds, n_initial)))
    distractor_x, distractor_value = sense(distractors, sign((worlds, n_distractors)))
    preview, _ = sense(targets, sign((worlds, steps)))
    review, review_value = sense(targets, sign((worlds, steps)))
    final, final_value = sense(targets, sign((worlds, steps)))
    panel = dict(
        price=price, ticket_side=ticket_side, offer_left=center-mark, offer_right=center+mark,
        intro_ids=share(intro), target_ids=share(targets), private_laws=law,
        initial_x=initial_x, initial_value=initial_value,
        distractor_x=distractor_x, distractor_value=distractor_value,
        preview=preview, review=review, review_value=review_value, final=final, final_value=final_value,
        gaps=share(public.integers(0, 7, (worlds, steps, 2))),
        initial_draw=share(draws.random((worlds, n_initial, 5), dtype=np.float32)),
        step_draw=share(draws.random((worlds, steps, 4, 5), dtype=np.float32)),
        delayed_draw=share(draws.random((worlds, steps, 5), dtype=np.float32)),
        shuffle_key=share(shuffles.random((worlds, plan["yoke_replicates"], steps), dtype=np.float32)),
    )
    return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in panel.items()}


def equal_quota(natural, shuffled, price, plan):
    start = 0
    for end in plan["yoke_stage_ends"]:
        for amount in plan["prices_cents"]:
            chosen = price[:, start:end] == amount
            np.testing.assert_array_equal((natural[:, start:end] & chosen).sum(1),
                                          (shuffled[:, start:end] & chosen).sum(1))
        start = end


def shuffled_schedules(natural, tape, plan):
    """Redistribute exactly the natural purchases in each lane, price and stage."""
    price = tape["price"].numpy()
    draw = tape["shuffle_key"].numpy()
    matched = np.zeros((draw.shape[1], *natural.shape), dtype=bool)
    for lane, row in enumerate(natural):
        start = 0
        for end in plan["yoke_stage_ends"]:
            for amount in plan["prices_cents"]:
                slots = np.flatnonzero(price[lane, start:end] == amount) + start
                bought = row[slots].sum()
                positions = np.argsort(draw[lane][:, slots], axis=1, kind="stable")[:, :bought]
                matched[np.arange(len(matched))[:, None], lane, slots[positions]] = True
            start = end
    for schedule in matched:
        equal_quota(natural, schedule, price, plan)
    return matched


def paid_sense(view, choice, left, right, taste, draws):
    seen = NET.observe(view, choice, left, right, taste)
    return NET.commit(view, seen, draws).state


def accumulate(trace, x, side, taste):
    selected = torch.nn.functional.normalize(x * side[:, None], dim=1)
    return trace + coverage.moments(selected) * taste.abs()[:, None]


@torch.no_grad()
def life(graph, rule, tape, plan, *, instrument, gain, environment, forced=None):
    """Advance a life without resetting either its graph or six-scalar trace."""
    lanes, steps = tape["price"].shape
    view = rule.expand(lanes)
    state = NET.start(graph, lanes)
    trace = graph.bias.new_zeros((lanes, 6))
    food_contacts = torch.zeros(lanes, dtype=torch.int16)
    zero = graph.bias.new_zeros(lanes)
    initial = []
    for i in range(plan["initial_contacts"]):
        x = tape["initial_x"][:, i]
        state, _, side, taste = ticket.encounter(view, state, x, tape["initial_value"][:, i], tape["initial_draw"][:, i])
        trace = accumulate(trace, x, side, taste)
        food_contacts += (taste != 0).to(food_contacts.dtype)
        initial.append(torch.stack((side, taste), 1))
    outputs = []
    period = plan["distractor_contact_every"]
    for i in range(steps):
        u = tape["step_draw"][:, i]
        disturb_side, disturb_taste = zero, zero
        if (i+1) % period == 0:
            j = (i+1) // period - 1
            x = tape["distractor_x"][:, j]
            state, _, disturb_side, disturb_taste = ticket.encounter(
                view, state, x, tape["distractor_value"][:, j], u[:, 0])
            trace = accumulate(trace, x, disturb_side, disturb_taste)
            food_contacts += (disturb_taste != 0).to(food_contacts.dtype)

        state = NET.wait(view, state, tape["gaps"][:, i, 0], max_ticks=6)
        x = tape["preview"][:, i]
        state = NET._advance(view, state, NET.native.sensory(state.graph, -x, x), NET.REACTION_TICKS)
        left, right = tape["offer_left"][:, i], tape["offer_right"][:, i]
        signal = coverage.cue(trace, x, instrument) if instrument else zero
        if instrument:
            menu = coverage.instrument({"offer_left": left, "offer_right": right}, signal, gain)
            left, right = menu["offer_left"], menu["offer_right"]
        route = tape["ticket_side"][:, i]
        use = None if forced is None else torch.where(forced[:, i], route, -route)
        choice = NET.act(view, state, left, right, u[:, 1, 0], side_override=use)
        buy = choice.side == route
        fee = tape["price"][:, i] * buy
        state = paid_sense(view, choice, left, right, -fee.to(graph.bias.dtype)/100, u[:, 1, 1:])
        before_count = food_contacts.clone()

        x = tape["review"][:, i]
        state, _, review_side, review_taste = ticket.encounter(view, state, x, tape["review_value"][:, i] * buy, u[:, 2])
        trace = accumulate(trace, x, review_side, review_taste)
        food_contacts += (review_taste != 0).to(food_contacts.dtype)
        state = NET.wait(view, state, tape["gaps"][:, i, 1], max_ticks=6)

        x = tape["final"][:, i]
        harvest = NET.act(view, state, -x, x, u[:, 3, 0])
        earned = harvest.side * tape["final_value"][:, i]
        delivered = earned if environment == "immediate" else zero
        state = paid_sense(view, harvest, -x, x, delivered, u[:, 3, 1:])
        trace = accumulate(trace, x, harvest.side, delivered)
        food_contacts += (delivered != 0).to(food_contacts.dtype)
        outputs.append(dict(buy=buy, fee=fee, ticket_side=choice.side, ticket_logit=choice.motor_logit,
                            review_side=review_side, review_taste=review_taste, harvest_side=harvest.side,
                            signed_harvest=earned, delivered_harvest=delivered, distractor_side=disturb_side,
                            distractor_taste=disturb_taste, food_count_before=before_count, coverage_signal=signal))

    window = torch.stack([r["signed_harvest"] for r in outputs], 1)
    if environment == "delayed":
        # The environment pays from prior choices; this tail only delivers their withheld sensations.
        for i in range(steps):
            x, draw = tape["final"][:, i], tape["delayed_draw"][:, i]
            side = outputs[i]["harvest_side"]
            now = NET.act(view, state, -x, x, draw[:, 0], side_override=side)
            state = paid_sense(view, now, -x, x, window[:, i], draw[:, 1:])
            trace = accumulate(trace, x, side, window[:, i])
            food_contacts += (window[:, i] != 0).to(food_contacts.dtype)
    torch.testing.assert_close(trace[:, :3].sum(1), food_contacts.to(trace.dtype), atol=2e-5, rtol=1e-6)
    if not all(torch.isfinite(x).all() for x in (state.hidden, state.graph.weights, state.graph.bias)):
        raise FloatingPointError("A frozen long life became nonfinite")
    results = {k: torch.stack([output[k] for output in outputs], 1).numpy() for k in outputs[0]}
    results["initial"] = torch.stack(initial, 1).numpy()
    results["settled_food_count"] = food_contacts.numpy()
    results["final_coverage_trace"] = trace.numpy()
    return results


def routes(graph, rule, tape, plan, environment, instrument, gain):
    result = {"natural": life(graph, rule, tape, plan, instrument=instrument, gain=gain, environment=environment)}
    natural = result["natural"]["buy"]
    schedules = shuffled_schedules(natural, tape, plan)
    instructions = [
        ("always_buy", np.ones_like(natural)), ("always_pass", np.zeros_like(natural)),
        *((f"shuffle_{i + 1}", schedule) for i, schedule in enumerate(schedules)),
    ]
    lanes = len(natural)
    for start in range(0, len(instructions), 2):
        batch = instructions[start:start+2]
        combined = {k: torch.cat([v] * len(batch)) for k, v in tape.items()}
        forced = torch.from_numpy(np.concatenate([decision for _, decision in batch]))
        replay = life(graph, rule, combined, plan, environment=environment, instrument=instrument, gain=gain, forced=forced)
        for i, (name, _) in enumerate(batch):
            result[name] = {k: v[i*lanes:(i+1)*lanes] for k, v in replay.items()}
    return {f"{mode}/{k}": v for mode, values in result.items() for k, v in values.items()}


def verify_and_score(raw, tape, plan, environment, instrument):
    price = tape["price"].numpy()
    signed_ticket = tape["ticket_side"].numpy()
    n_histories = plan["private_histories_per_geometry"]
    raw_modes = ("natural", "always_buy", "always_pass",
                 *(f"shuffle_{i+1}" for i in range(plan["yoke_replicates"])))
    scores, differences = {}, {}
    for mode in raw_modes:
        d = {k.removeprefix(mode + "/"): raw[k] for k in raw.files if k.startswith(mode + "/")}
        bought, gathered = d["buy"], d["signed_harvest"]
        np.testing.assert_array_equal(bought, d["ticket_side"] == signed_ticket)
        np.testing.assert_array_equal(d["fee"], price * bought)
        np.testing.assert_array_equal(d["review_taste"], d["review_side"] * tape["review_value"].numpy() * bought)
        np.testing.assert_array_equal(gathered, d["harvest_side"] * tape["final_value"].numpy())
        np.testing.assert_array_equal(d["delivered_harvest"], gathered if environment == "immediate" else np.zeros_like(gathered))
        np.testing.assert_array_equal(d["initial"][:, :, 1], d["initial"][:, :, 0] * tape["initial_value"].numpy())
        period = plan["distractor_contact_every"]
        np.testing.assert_array_equal(d["distractor_taste"][:, period-1::period],
                                      d["distractor_side"][:, period-1::period] * tape["distractor_value"].numpy())
        for key in ("distractor_side", "distractor_taste"):
            free = (np.arange(price.shape[1]) + 1) % period == 0
            np.testing.assert_array_equal(d[key][:, ~free], 0)
        signed_food = (d["review_taste"] != 0).astype(np.int64) + (d["delivered_harvest"] != 0)
        free_food = d["distractor_taste"] != 0
        experienced = np.cumsum(signed_food + free_food, 1)
        initial = (d["initial"][:, :, 1] != 0).sum(1)
        prior = initial[:, None] + np.column_stack((np.zeros(len(initial), dtype=int), experienced[:, :-1])) + free_food
        np.testing.assert_array_equal(prior, d["food_count_before"])
        tail = (gathered != 0).sum(1) if environment == "delayed" else 0
        np.testing.assert_array_equal(d["settled_food_count"], initial + experienced[:, -1] + tail)
        if instrument:
            validate_coverage(d, tape, plan, environment, instrument)
        else:
            np.testing.assert_array_equal(d["coverage_signal"], 0)
        if mode == "always_buy":
            np.testing.assert_array_equal(bought, True)
        if mode == "always_pass":
            np.testing.assert_array_equal(bought, False)
        if mode.startswith("shuffle"):
            equal_quota(raw["natural/buy"], bought, price, plan)
        obtained = (gathered > 0).astype(np.int64)
        measures = {"net": obtained*100 - d["fee"], "gross": obtained, "fees": d["fee"], "purchases": bought,
                    "changed": bought != raw["natural/buy"]}
        scores[mode] = {}
        for k, values in measures.items():
            cumulative = np.cumsum(values, axis=1)
            for end in plan["report_through"]:
                quantity = cumulative[:, end-1].reshape(-1, n_histories).mean(1)
                if k in ("net", "fees"):
                    quantity = quantity / 100
                if k == "changed":
                    quantity = quantity / end
                scores[mode][f"{end}/{k}"] = quantity
                if mode.startswith("shuffle") and k in ("fees", "purchases"):
                    np.testing.assert_array_equal(quantity, scores["natural"][f"{end}/{k}"])
        if mode.startswith("shuffle"):
            differences[mode] = scores[mode]
    scores["matched_shuffle"] = {key: np.mean([group[key] for group in differences.values()], axis=0)
                                 for key in scores["natural"]}
    return {mode: scores[mode] for mode in MODES}


def validate_coverage(raw, tape, plan, environment, instrument):
    """Rebuild each scalar shown at decision time from arrived sensory events."""
    n, steps = raw["buy"].shape
    trace = torch.zeros((n, 6), dtype=tape["preview"].dtype)
    initial = torch.from_numpy(raw["initial"])
    for i in range(plan["initial_contacts"]):
        trace = accumulate(trace, tape["initial_x"][:, i], initial[:, i, 0], initial[:, i, 1])
    period = plan["distractor_contact_every"]
    events = {k: torch.from_numpy(raw[k]) for k in ("distractor_side", "distractor_taste", "review_side",
              "review_taste", "harvest_side", "signed_harvest", "delivered_harvest", "coverage_signal")}
    for i in range(steps):
        if (i+1) % period == 0:
            j = (i+1) // period - 1
            trace = accumulate(trace, tape["distractor_x"][:, j], events["distractor_side"][:, i],
                               events["distractor_taste"][:, i])
        value = coverage.cue(trace, tape["preview"][:, i], instrument)
        torch.testing.assert_close(value, events["coverage_signal"][:, i], atol=2e-6, rtol=2e-6)
        trace = accumulate(trace, tape["review"][:, i], events["review_side"][:, i], events["review_taste"][:, i])
        trace = accumulate(trace, tape["final"][:, i], events["harvest_side"][:, i], events["delivered_harvest"][:, i])
    if environment == "delayed":
        for i in range(steps):
            trace = accumulate(trace, tape["final"][:, i], events["harvest_side"][:, i], events["signed_harvest"][:, i])
    torch.testing.assert_close(trace, torch.from_numpy(raw["final_coverage_trace"]), atol=2e-6, rtol=2e-6)


def checkpoint(cohort, model, birth, supplement):
    family, arm, _ = models.MODELS[model]
    source = supplement["model_custody"][model]
    identity = source["models"][str(birth)]
    archive.verify(card.ROOT / identity["path"], identity["sha256"])
    graph, rule = card.checkpoint.load_checkpoint(card.DATA / family / cohort, "odor", arm, birth, models.STAGE)
    instrument = None
    gain = None
    files = {"checkpoint": identity}
    if family == "coverage":
        scale = source["paid_menu_gain_only"]
        archive.verify(card.ROOT / scale["path"], scale["sha256"])
        with np.load(card.ROOT / scale["path"]) as z:
            index = z["births"].tolist().index(birth)
            gain = torch.as_tensor(z["gain"][index, 0], dtype=graph.bias.dtype)
        instrument = arm
        files["gain"] = scale
    return graph, rule, instrument, gain, files


def intervals(bank, plan, n, seed):
    statistic = card.stats.Crossed(n, plan["geometries_per_birth"], seed+992, 4000)
    rates, paired = {}, {}
    for environment, entries in bank.items():
        rates[environment], paired[environment] = {}, {}
        for model, trials in entries.items():
            rates[environment][model] = {mode: {k: statistic(v) for k, v in cells.items()} for mode, cells in trials.items()}
            paired[environment][model] = {}
            for alternative in MODES[1:]:
                paired[environment][model][f"natural minus {alternative}"] = {
                    k: statistic(v - trials[alternative][k]) for k, v in trials["natural"].items()}
    cross_model = {}
    for environment, entries in bank.items():
        cross_model[environment] = {}
        for a, b in (("ticket_native", "ticket_coin"), ("coverage_evidence", "coverage_dose"),
                     ("coverage_evidence", "ticket_native"), ("coverage_dose", "ticket_native")):
            if a not in entries or b not in entries:
                continue
            x, y = entries[a], entries[b]
            cross_model[environment][f"{a} minus {b}"] = {
                "natural": {k: statistic(v - y["natural"][k]) for k, v in x["natural"].items()},
                "purchase_allocation_advantage": {k: statistic((v - x["matched_shuffle"][k]) -
                                                               (y["natural"][k] - y["matched_shuffle"][k]))
                                                  for k, v in x["natural"].items()},
            }
    return rates, paired, cross_model


def assess_birth(cohort, model, birth, tape, plan, environment, supplement, signature, directory):
    graph, rule, instrument, gain, files = checkpoint(cohort, model, birth, supplement)
    source_key = json.dumps(files, sort_keys=True)
    file = directory / f"{environment}-{model}-{birth}.npz"
    if not file.exists():
        result = routes(graph, rule, tape, plan, environment, instrument, gain)
        io._write_raw(file, signature=signature, source_key=source_key, **result)
    with np.load(file) as raw:
        if (str(raw["signature"]), str(raw["source_key"])) != (signature, source_key):
            raise ValueError(f"Stale long life trace {file}")
        measured = verify_and_score(raw, tape, plan, environment, instrument)
    return measured, {"trace": custody.record(file), **files}


def run(cohort, smoke=False):
    torch.set_num_threads(1)
    plan = archive.read(PLAN)
    seed = plan["cohorts"][cohort]
    source = card.OUT / f"transfer-ticket-coverage-{cohort}-8000.json"
    supplement = archive.read(source)
    births = supplement["births"]
    worlds = 2 if smoke else plan["geometries_per_birth"]
    tape = make_tape(plan, seed + 491 if smoke else seed, worlds)
    if smoke:
        graph, rule, instrument, gain, _ = checkpoint(cohort, "coverage_evidence", births[0], supplement)
        for environment in plan["environments"]:
            start = perf_counter()
            result = routes(graph, rule, tape, plan, environment, instrument, gain)
            raw_file = BytesIO()
            np.savez(raw_file, **result)
            raw_file.seek(0)
            with np.load(raw_file) as raw:
                verify_and_score(raw, tape, plan, environment, instrument)
            print(f"{cohort} {environment}, {worlds} worlds: {perf_counter()-start:.2f} seconds", flush=True)
        return

    provenance = {Path(m.__file__).name: archive.digest(Path(m.__file__))
                  for m in (ticket, coverage, NET, NET.native, card.checkpoint)}
    provenance[Path(__file__).name] = archive.digest(Path(__file__))
    identities = dict(plan=archive.digest(PLAN), code=provenance, torch=str(torch.__version__), numpy=np.__version__)
    signature = json.dumps(identities, sort_keys=True)
    directory = OUTPUT / cohort
    directory.mkdir(parents=True, exist_ok=True)
    world_file = directory / "worlds.npz"
    if not world_file.exists():
        io._write_raw(world_file, signature=signature, **{k: v.numpy() for k, v in tape.items()})
    with np.load(world_file) as raw:
        if str(raw["signature"]) != signature:
            raise ValueError(f"World record belongs to another protocol: {world_file}")
        for key, tensor in tape.items():
            np.testing.assert_array_equal(tensor.numpy(), raw[key])
    bank = {}
    hashes = {}
    for environment in plan["environments"]:
        bank[environment], hashes[environment] = {}, {}
        for model in plan["models_in_order"]:
            cells = []
            hashes[environment][model] = {}
            for birth in births:
                start = perf_counter()
                score, index = assess_birth(cohort, model, birth, tape, plan, environment, supplement, signature, directory)
                cells.append(score)
                hashes[environment][model][str(birth)] = index
                print(cohort, environment, model, birth, f"{perf_counter()-start:.2f}s", flush=True)
            bank[environment][model] = {mode: {k: np.stack([row[mode][k] for row in cells]) for k in cells[0][mode]}
                                        for mode in MODES}
            observed, differences, models_compared = intervals(bank, plan, len(births), seed)
            output = dict(cohort=cohort, seed=seed, births=births, protocol=identities, worlds=custody.record(world_file),
                          geometry_count=worlds, history_cross=plan["private_histories_per_geometry"],
                          shuffle_replicates=plan["yoke_replicates"], hashes=hashes, models_index=custody.record(source),
                          rates=observed, paired=differences, cross_model=models_compared)
            io._write_json(directory / "progress.json", output)
            headline = differences[environment][model]["natural minus matched_shuffle"]
            print(cohort, environment, model, "net allocation advantage",
                  [round(headline[f"{end}/net"]["mean"], 4) for end in plan["report_through"]], flush=True)
    io._write_json(card.OUT / f"transfer-lifetime-{cohort}.json", output)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, choices=tuple(card.swap.COHORTS))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.cohort, smoke=args.smoke)
