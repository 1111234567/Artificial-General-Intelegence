"""Deploy the frozen resource readout once at each original ticket menu."""

from argparse import ArgumentParser
from io import BytesIO
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import torch

from . import interleave_revisit_train as io, noon_resource_readout as readout
from . import noon_transfer as card, noon_transfer_archive as archive, noon_transfer_custody as custody
from . import noon_transfer_lifetime as life, noon_transfer_lifetime_report as audit


PLAN = card.OUT / "transfer-resource-readout-plan.json"
OUT = life.OUTPUT / "resource-readout"
NATIVE = life.NET
COVERAGE = life.coverage


class OfferReadout:
    """Intercept only the priced menu action after its trace and preview are available."""

    def __init__(self, arm, weight):
        self.arm, self.weight = arm, weight
        self.waiting = None
        self.events = []

    def __getattr__(self, key):
        return getattr(NATIVE, key)

    moments = staticmethod(COVERAGE.moments)

    def cue(self, trace, preview, instrument):
        if self.waiting is not None:
            raise ValueError("Previous paid menu did not consume its readout")
        s = COVERAGE.projection(trace, preview)
        self.waiting = (trace, preview, s)
        return s

    @staticmethod
    def instrument(menu, signal, gain):
        return menu

    def act(self, rule, state, left, right, draw=None, *, stochastic=True, side_override=None):
        if self.waiting is not None:
            trace, preview, signal = self.waiting
            self.waiting = None
            fields = {"offer_left": left, "offer_right": right, "preview": preview}
            feature = readout.features(fields, trace, self.arm).double()
            logits = feature @ self.weight
            self.events.append(torch.stack((trace[:, :3].sum(1).double()/3, signal.double(), logits), 1))
            if side_override is None:
                buy = draw.double() < logits.sigmoid()
                marked = torch.where(right[:, 0] > left[:, 0], 1., -1.)
                side_override = torch.where(buy, marked, -marked)
        return NATIVE.act(rule, state, left, right, draw, stochastic=stochastic, side_override=side_override)


def life_with_readout(graph, rule, tape, plan, environment, arm, weight, forced=None):
    policy = OfferReadout(arm, weight)
    if life.NET is not NATIVE or life.coverage is not COVERAGE:
        raise RuntimeError("Concurrent paid-menu hook")
    life.NET, life.coverage = policy, policy
    try:
        raw = life.life(graph, rule, tape, plan, instrument="evidence", gain=None, environment=environment, forced=forced)
    finally:
        life.NET, life.coverage = NATIVE, COVERAGE
    if policy.waiting is not None or len(policy.events) != plan["opportunities"]:
        raise AssertionError("A readout must be consumed exactly once per paid offer")
    table = torch.stack(policy.events, 1).numpy()
    raw.update(readout_dose=table[:, :, 0], readout_signal=table[:, :, 1], readout_logit=table[:, :, 2])
    return raw


def routes(graph, rule, tape, plan, environment, arm, weight):
    result = {"natural": life_with_readout(graph, rule, tape, plan, environment, arm, weight)}
    natural = result["natural"]["buy"]
    shuffled = life.shuffled_schedules(natural, tape, plan)
    alternatives = [
        ("always_buy", np.ones_like(natural)), ("always_pass", np.zeros_like(natural)),
        *((f"shuffle_{i + 1}", schedule) for i, schedule in enumerate(shuffled)),
    ]
    lanes = len(natural)
    for offset in range(0, len(alternatives), 2):
        batch = alternatives[offset:offset+2]
        complete = {k: torch.cat([v]*len(batch)) for k, v in tape.items()}
        assigned = torch.from_numpy(np.concatenate([choice for _, choice in batch]))
        replays = life_with_readout(graph, rule, complete, plan, environment, arm, weight, assigned)
        for index, (mode, _) in enumerate(batch):
            result[mode] = {k: v[index*lanes:(index+1)*lanes] for k, v in replays.items()}
    return {f"{mode}/{k}": v for mode, table in result.items() for k, v in table.items()}


def coefficients(seed, arm, step):
    directory = readout.OUT / str(seed) / arm
    run = directory / "run.json"
    source = directory / f"step-{step}.pt"
    manifest = archive.read(run)
    archive.verify(readout.PLAN, manifest["plan"])
    frozen = torch.load(source, map_location="cpu", weights_only=True)
    if (frozen["meta"], frozen["step"], manifest["arm"], manifest["seed"]) != (archive.digest(run), step, arm, seed):
        raise ValueError(f"Foreign coefficient checkpoint: {source}")
    if int(frozen["optimizer"]["state"][0]["step"]) != step:
        raise ValueError(f"Unexpected coefficient update count: {source}")
    values = frozen["weight"]
    if values.shape != (4,) or values.dtype != torch.float64 or not values.isfinite().all():
        raise ValueError(f"Expected four finite shared float64 values: {source}")
    index = {"params": custody.record(source), "manifest": custody.record(run), "weights": values.tolist(),
             "training_parents": manifest["parents"], "step": step}
    return values, index


def verify_readout(raw, tape, plan, environment, arm, weight):
    audit.verify_replays(raw, tape, plan)
    scores = life.verify_and_score(raw, tape, plan, environment, "evidence")
    modes = ("natural", "always_pass", "always_buy",
             *(f"shuffle_{i+1}" for i in range(plan["yoke_replicates"])))
    menus = {k: tensor.numpy() for k, tensor in tape.items() if k in ("offer_left", "offer_right")}
    x = (menus["offer_left"][:, :, 2] + menus["offer_right"][:, :, 2])*.25
    scaled_price = ((x - .25)*4).astype(np.float64)
    for mode in modes:
        s = raw[f"{mode}/readout_signal"]
        dose = raw[f"{mode}/readout_dose"]
        y = raw[f"{mode}/readout_logit"]
        np.testing.assert_array_equal(s, raw[f"{mode}/coverage_signal"])
        np.testing.assert_allclose(dose, raw[f"{mode}/food_count_before"]/3, atol=2e-5, rtol=0)
        np.testing.assert_allclose(y, weight[0] + weight[1]*scaled_price +
                                   (dose.astype(np.float32).astype(float)*weight[2] if arm != "price" else 0.) +
                                   (s*weight[3] if arm == "directional" else 0.), atol=2e-5, rtol=1e-7)
        if mode == "natural":
            buy = tape["step_draw"][:, :, 1, 0].double() < torch.from_numpy(y).sigmoid()
            np.testing.assert_array_equal(buy.numpy(), raw["natural/buy"])
    return scores


def distributions(rows):
    result = {}
    for point in (2, 8, 32):
        key = str(point)
        result[key] = {}
        for field in ("signal", "dose", "logit"):
            values = np.concatenate([x[field][:, point-1] for x in rows])
            result[key][field] = dict(zip(("p05", "median", "p95"),
                                          np.quantile(values, [.05, .5, .95]).tolist(), strict=True))
        arrived = np.concatenate([x["dose"][:, point-1] for x in rows])
        result[key]["fraction_above_8_food_tastes"] = float((arrived > 8/3 + 2e-5).mean())
        prob = torch.from_numpy(np.concatenate([x["logit"][:, point-1] for x in rows])).sigmoid().numpy()
        result[key]["prob_below_0_01_or_above_0_99"] = float(((prob < .01) | (prob > .99)).mean())
    values = np.concatenate([x["signal"][:, 16:] for x in rows])
    dose = np.concatenate([x["dose"][:, 16:] for x in rows])
    result["17-32"] = {"signal_quantiles_05_50_95": np.quantile(values, [.05, .5, .95]).tolist(),
                       "dose_quantiles_05_50_95": np.quantile(dose, [.05, .5, .95]).tolist()}
    return result


def summarize(bank, draws):
    rates = {model: {mode: {k: draws(v) for k, v in cells.items()} for mode, cells in arms.items()}
             for model, arms in bank.items()}
    contrasts = {}
    for model, output in bank.items():
        contrasts[model] = {
            f"natural minus {mode}": {k: draws(v-output[mode][k]) for k, v in output["natural"].items()}
            for mode in life.MODES[1:]
        }
    between = {}
    for a, b in (("directional", "price"), ("directional", "dose"), ("dose", "price"),
                 ("directional", "bare_native"), ("price", "bare_native")):
        x, y = bank[a], bank[b]
        between[f"{a} minus {b}"] = {
            "natural": {k: draws(v-y["natural"][k]) for k, v in x["natural"].items()},
            "allocation": {k: draws((v-x["matched_shuffle"][k])-(y["natural"][k]-y["matched_shuffle"][k]))
                           for k, v in x["natural"].items()},
        }
    return {"rates": rates, "within": contrasts, "between": between}


def matrices(rows):
    return {mode: {k: np.stack([root[mode][k] for root in rows]) for k in rows[0][mode]}
            for mode in life.MODES}


@torch.no_grad()
def run(cohort, seed, smoke=False):
    torch.set_num_threads(1)
    protocol = archive.read(PLAN)
    setup = archive.read(life.PLAN)
    primary = seed == protocol["primary_training_seed"]
    environments = protocol["primary_environments"] if primary else protocol["additional_environment"]
    old = card.OUT / f"transfer-lifetime-{cohort}.json"
    baseline = archive.read(old)
    parent = archive.read(card.ROOT / baseline["models_index"]["path"])
    stage, births = protocol["training_step"], baseline["births"]
    n = 2 if smoke else baseline["geometry_count"]
    world_seed = baseline["seed"]
    tape = life.make_tape(setup, world_seed + 697 if smoke else world_seed, n)
    if smoke:
        g, r, *_ = life.checkpoint(cohort, "ticket_native", births[0], parent)
        weight, _ = coefficients(seed, "directional", stage)
        for environment in environments:
            raw = BytesIO()
            np.savez(raw, **routes(g, r, tape, setup, environment, "directional", weight))
            raw.seek(0)
            with np.load(raw) as result:
                verify_readout(result, tape, setup, environment, "directional", weight.numpy())
            print(cohort, seed, environment, "physical smoke matched", flush=True)
        return
    archive.verify(life.PLAN, baseline["protocol"]["plan"])
    world = card.ROOT / baseline["worlds"]["path"]
    archive.verify(world, baseline["worlds"]["sha256"])
    with np.load(world) as original:
        for key, tensor in tape.items():
            np.testing.assert_array_equal(tensor.numpy(), original[key])
    directory = OUT / str(seed) / cohort
    directory.mkdir(parents=True, exist_ok=True)
    policy = {a: coefficients(seed, a, stage) for a in protocol["arms"]}
    anchor = archive.read(card.OUT / "transfer-ticket-coverage-original-8000.json")["model_custody"]["ticket_native"]["models"]
    expected = {b: row["sha256"] for b, row in anchor.items()}
    if not all(meta["training_parents"] == expected for _, meta in policy.values()):
        raise ValueError("A readout was trained on different ticket bodies")
    sources = {Path(m.__file__).name: archive.digest(Path(m.__file__)) for m in (life, readout, NATIVE, COVERAGE)}
    sources[Path(__file__).name] = archive.digest(Path(__file__))
    key = json.dumps({"plan": archive.digest(PLAN), "worlds": baseline["worlds"]["sha256"], "source": sources}, sort_keys=True)
    draws = card.stats.Crossed(len(births), n, world_seed+992, 4000)
    rates, signals, raw_index, all_arrays = {}, {}, {}, {}
    for environment in environments:
        replay = baseline["hashes"][environment]["ticket_native"]
        rows = []
        raw_index[environment] = {"bare_native": {}}
        for b in births:
            src = replay[str(b)]["trace"]
            path = card.ROOT / src["path"]
            archive.verify(path, src["sha256"])
            with np.load(path) as data:
                rows.append(life.verify_and_score(data, tape, setup, environment, None))
            raw_index[environment]["bare_native"][str(b)] = replay[str(b)]
        models = {"bare_native": matrices(rows)}
        signals[environment] = {}
        for arm, (weight, identity) in policy.items():
            rows, logs = [], []
            raw_index[environment][arm] = {}
            for birth in births:
                start = perf_counter()
                g, r, _, _, body = life.checkpoint(cohort, "ticket_native", birth, parent)
                path = directory / f"{environment}-{arm}-{birth}.npz"
                seal = json.dumps(dict(body=body, coefficients=identity["params"]), sort_keys=True)
                if not path.is_file():
                    raw = routes(g, r, tape, setup, environment, arm, weight)
                    io._write_raw(path, signature=key, bodies=seal, **raw)
                with np.load(path) as raw:
                    if (str(raw["signature"]), str(raw["bodies"])) != (key, seal):
                        raise ValueError(f"Stale readout replay {path}")
                    row = verify_readout(raw, tape, setup, environment, arm, weight.numpy())
                    rows.append(row)
                    logs.append({k: raw[f"natural/readout_{k}"] for k in ("signal", "dose", "logit")})
                raw_index[environment][arm][str(birth)] = {"trace": custody.record(path), **body}
                print(cohort, seed, environment, arm, birth, f"{perf_counter()-start:.2f}s", flush=True)
            models[arm] = matrices(rows)
            signals[environment][arm] = distributions(logs)
            headline = {mode: [round(float(models[arm][mode][f"{i}/net"].mean()), 5) for i in setup["report_through"]]
                        for mode in life.MODES}
            print(cohort, seed, environment, arm, "net through decisions 2/8/32", headline, flush=True)
        all_arrays[environment] = models
        rates[environment] = summarize(models, draws)
        progress = dict(cohort=cohort, seed=seed, births=births, worlds=n, root_seed=world_seed, protocol=custody.record(PLAN),
                        source=sources, coefficients={a: meta for a, (_, meta) in policy.items()},
                        results=rates, signals=signals, raw=raw_index)
        io._write_json(directory / "progress.json", progress)
    matrices_file = directory / "per-world.npz"
    io._write_raw(matrices_file, **{f"{e}/{a}/{m}/{k}": v for e, arms in all_arrays.items()
                                    for a, modes in arms.items() for m, cells in modes.items() for k, v in cells.items()})
    progress.update(per_world=custody.record(matrices_file), previous_baseline=custody.record(old))
    io._write_json(card.OUT / f"transfer-resource-readout-{seed}-{cohort}.json", progress)


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--cohort", choices=tuple(card.swap.COHORTS), required=True)
    p.add_argument("--seed", type=int, choices=readout.SEEDS, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    run(args.cohort, args.seed, args.smoke)
